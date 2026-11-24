"""
main_ddp_movie_ae.py — Multi-GPU DDP training for LatentAlignModel (movie/ML-20M version).
Backbones: EASE, EDLAE (FR/LR), Linear-DAE and Multi-DAE (checkpoint name ending in _deep.pt).

Usage:
    torchrun --nproc_per_node=1 experiments/ML-20M/main_ddp_movie_ae.py --classifier-path results/<run>/best.pt [args]
"""

import os
import json
import argparse
from datetime import datetime
import uuid
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model_ddp_ae import LatentAlignModel
from utils import (
    set_seed, read_data, ML20MDataset, ML20MDatasetVal,
    collate_fn, collate_fn_val,
    recall_ndcg_multi_k_full, hellinger_loss, infonce_loss,
)

# ============================================================================
# DDP helpers
# ============================================================================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return dist.get_rank() == 0

def gather_embeddings(z):
    world_size = dist.get_world_size()
    if world_size == 1:
        return z
    gathered = [torch.zeros_like(z) for _ in range(world_size)]
    dist.all_gather(gathered, z.contiguous())
    gathered[dist.get_rank()] = z  # keep grad for local shard
    return torch.cat(gathered, dim=0)


# ============================================================================
# Scheduler
# ============================================================================

def get_warmup_linear_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Logging
# ============================================================================

log_file = None

def log(msg):
    if is_main():
        print(msg)
        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()


# ============================================================================
# Losses
# ============================================================================

def multinomial_loss(logits, inputs):
    log_probs = F.log_softmax(logits, dim=1)
    loss = -torch.sum(log_probs * inputs, dim=1) / logits.size(1)
    return loss.mean()

def mse_loss(logits, inputs):
    return F.mse_loss(logits, inputs)

def vicreg_cov_loss(z1, z2):
    z1 = z1 - z1.mean(0)
    z2 = z2 - z2.mean(0)
    N = z1.size(0)
    cov = (z1.T @ z2) / (N - 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / (z1.size(1) ** 2)

def infonce_loss_global(z_text, z_colab):
    z_text_g = gather_embeddings(z_text)
    z_colab_g = gather_embeddings(z_colab)
    return infonce_loss(z_text_g, z_colab_g)


def nmse_loss(logits, inputs):
    return torch.nn.functional.mse_loss(
        torch.nn.functional.normalize(logits, dim=-1),
        torch.nn.functional.normalize(inputs, dim=-1),
        reduction='mean'
    )
# ============================================================================
# Trackers
# ============================================================================

class LossTracker:
    def __init__(self, names):
        self.names = names
        self.history = []
        self.reset()
    def reset(self):
        self.totals = {n: 0.0 for n in self.names}
        self.count = 0
    def update(self, batch_size, **kwargs):
        for n, v in kwargs.items():
            self.totals[n] += v * batch_size
        self.count += batch_size
    def end_epoch(self):
        avgs = {n: self.totals[n] / self.count for n in self.names}
        self.history.append(avgs)
        self.reset()
        return avgs
    def print_history(self):
        for i, h in enumerate(self.history):
            parts = " | ".join(f"{n}={h[n]:.4f}" for n in self.names)
            log(f"Epoch {i+1}: {parts}")


class MetricAccumulator:
    def __init__(self):
        self.history = []
        self.reset()
    def reset(self):
        self.all_ndcg_100 = []
        self.all_recall_50 = []
        self.all_recall_20 = []
    def update(self, scores, train_mask, target):
        res = recall_ndcg_multi_k_full(scores, train_mask, target, [100, 50, 20])
        self.all_ndcg_100.append(res["ndcg@100"])
        self.all_recall_50.append(res["recall@50"])
        self.all_recall_20.append(res["recall@20"])
    def end_epoch(self):
        metrics = {
            "ndcg@100": torch.cat(self.all_ndcg_100).mean().item(),
            "recall@50": torch.cat(self.all_recall_50).mean().item(),
            "recall@20": torch.cat(self.all_recall_20).mean().item(),
        }
        self.history.append(metrics)
        self.reset()
        return metrics
    def print_best(self):
        best = max(self.history, key=lambda m: m["ndcg@100"])
        ep = self.history.index(best) + 1
        log(f"Best Epoch {ep}: NDCG@100={best['ndcg@100']:.4f} | "
            f"Recall@50={best['recall@50']:.4f} | Recall@20={best['recall@20']:.4f}")


# ============================================================================
# Evaluation (rank 0 only)
# ============================================================================

def run_eval(model, loader, device, tracker_joint, tracker_text, tracker_colab, desc="Val"):
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, disable=not is_main()):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            summaries = batch["summary"]
            with torch.autocast(device_type="cuda"):
                (z_colab, z_text, z_joint,
                 y_colab, y_text, y_joint,
                 _, _) = raw_model(x, summaries)
            tracker_joint.update(y_joint, x, y)
            tracker_text.update(y_text, x, y)
            tracker_colab.update(y_colab, x, y)

    m_joint = tracker_joint.end_epoch()
    m_text = tracker_text.end_epoch()
    m_colab = tracker_colab.end_epoch()
    log(f"  [JOINT]  NDCG@100={m_joint['ndcg@100']:.4f} | R@50={m_joint['recall@50']:.4f} | R@20={m_joint['recall@20']:.4f}")
    log(f"  [TEXT]   NDCG@100={m_text['ndcg@100']:.4f} | R@50={m_text['recall@50']:.4f} | R@20={m_text['recall@20']:.4f}")
    log(f"  [COLAB]  NDCG@100={m_colab['ndcg@100']:.4f} | R@50={m_colab['recall@50']:.4f} | R@20={m_colab['recall@20']:.4f}")
    return m_joint, m_text, m_colab


# ============================================================================
# Args
# ============================================================================

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", type=str, default="data/ml-20m")
parser.add_argument("--classifier-path", type=str, required=True,
                    help="Stage-1 checkpoint (best.pt from main_first_stage.py)")
parser.add_argument("--encoder-path", type=str, default="checkpoints/ml20m/enc_edlae_lowrank.pt")
parser.add_argument("--decoder-path", type=str, default="checkpoints/ml20m/dec_edlae_lowrank.pt")
parser.add_argument("--logdir", type=str, default="results_movie")

parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
parser.add_argument("--num-workers", type=int, default=6)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument("--lr-text", type=float, default=1e-6)
parser.add_argument("--lr-decoder", type=float, default=1e-5)
parser.add_argument("--lr-mapper", type=float, default=1e-4)
parser.add_argument("--lr-bias", type=float, default=0)

parser.add_argument("--loss-type", type=str, default="nmse",
                    choices=["multinomial", "mse", "nmse", "hellinger", "distill", "no"])
parser.add_argument("--text-coef", type=float, default=1.0)
parser.add_argument("--joint-coef", type=float, default=0.0)
parser.add_argument("--tag-coef", type=float, default=0.1)

parser.add_argument("--infonce", default=False, action="store_true")
parser.add_argument("--repulsion-coef-max", type=float, default=1.0,
                    help="Initial weight of the alignment loss")
parser.add_argument("--repulsion-coef-min", type=float, default=0.1,
                    help="Final weight of the alignment loss")
parser.add_argument("--repulsion-anneal-epochs", type=int, default=10,
                    help="Epochs over which the alignment weight decays linearly from max to min")
parser.add_argument("--ortho-coef", type=float, default=0.0)

parser.add_argument("--tau", type=float, default=0.003)
parser.add_argument("--use-linear", default=False, action="store_true")
parser.add_argument("--nonlinear", default=False, action="store_true")
parser.add_argument("--weighted", default=False, action="store_true")


# ============================================================================
# Main
# ============================================================================

def main():
    global log_file

    config = parser.parse_args()

    # --- DDP setup ---
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    if is_main():
        print(config)
        print(f"World size: {world_size} | Effective batch size: {config.batch_size * world_size}")

    set_seed(config.seed + dist.get_rank())

    # --- Data ---
    data = read_data(config.data_path)

    dataset_tr = ML20MDataset(data["X_tr"], data["S_tr"], data["n_items"])
    train_sampler = DistributedSampler(dataset_tr, num_replicas=world_size,
                                        rank=dist.get_rank(), shuffle=True)
    loader_tr = DataLoader(dataset_tr, batch_size=config.batch_size, sampler=train_sampler,
                           num_workers=config.num_workers, collate_fn=collate_fn,
                           pin_memory=True, drop_last=True)

    dataset_val = ML20MDatasetVal(data["X_val"], data["S_val"], data["n_items"], data["Y_val"])
    loader_val = DataLoader(dataset_val, batch_size=100, shuffle=False,
                            num_workers=config.num_workers, collate_fn=collate_fn_val)

    dataset_test = ML20MDatasetVal(data["X_test"], data["S_test"], data["n_items"], data["Y_test"])
    loader_test = DataLoader(dataset_test, batch_size=100, shuffle=False,
                             num_workers=config.num_workers, collate_fn=collate_fn_val)

    # --- Model ---
    model = LatentAlignModel(
        classifier_path=config.classifier_path,
        encoder_path=config.encoder_path,
        decoder_path=config.decoder_path,
        n_items=data["n_items"],
        use_linear=config.use_linear,
        tau=config.tau,
        nonlinear=config.nonlinear,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)
    raw_model = model.module

    params = [
        {"params": raw_model.auto_model.parameters(), "lr": config.lr_text},
        {"params": list(raw_model.linear_tags.parameters()), "lr": config.lr_text},
        {"params": raw_model.decoder_text.parameters(), "lr": config.lr_decoder},
        {"params": raw_model.mapper.parameters(), "lr": config.lr_mapper},
        {"params": [raw_model.bias], "lr": config.lr_bias},
    ]
    optimizer = torch.optim.AdamW(params)

    n_params = sum(p.numel() for g in params for p in g["params"] if p.requires_grad)
    log(f"Optimized params: {n_params:,} ({n_params / 1e6:.2f}M)")
    log(f"Per-GPU batch size: {config.batch_size} | Effective: {config.batch_size * world_size}")

    warmup_steps = len(loader_tr)
    total_steps = len(loader_tr) * config.epochs
    scheduler = get_warmup_linear_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1)

    scaler = GradScaler("cuda")

    ts = datetime.now()
    uid = uuid.uuid4().hex[:6]  # 6-char random suffix, collision probability negligible
    exp_name = f"{ts.year}_{ts.month:02d}_{ts.day:02d}-{ts.hour:02d}_{ts.minute:02d}_{ts.second:02d}-{uid}"
    exp_dir = os.path.join(config.logdir, exp_name)
    if is_main():
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(vars(config), f, indent=2)
        log_file = open(os.path.join(exp_dir, "train_log.txt"), "w")

    dist.barrier()

    log(str(config))
    log(exp_dir)

    # --- Trackers ---
    loss_tracker = LossTracker(["recon", "align", "tag", "total"])
    val_joint = MetricAccumulator()
    val_text = MetricAccumulator()
    val_colab = MetricAccumulator()

    best_ndcg_text = 0.0
    best_ndcg_joint = 0.0

    anneal_steps = len(loader_tr) * config.repulsion_anneal_epochs
    delta = (config.repulsion_coef_max - config.repulsion_coef_min) / max(anneal_steps, 1)
    align_coef = config.repulsion_coef_max
    tag_weights = raw_model.tag_weights if config.weighted else None

    # --- Training loop ---
    for epoch in range(config.epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        pbar = tqdm(loader_tr, desc=f"Epoch {epoch+1}/{config.epochs}", disable=not is_main())
        log(f"align_coef = {align_coef:.6f}")

        for batch in pbar:
            x = batch["x"].to(device)
            summaries = batch["summary"]
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda"):
                (z_colab, z_text, z_joint,
                 y_colab, y_text, y_joint,
                 y_tags_true, y_tags_pred) = model(x, summaries, p=config.dropout)

                # Reconstruction
                if config.loss_type == "multinomial":
                    l_recon = (config.text_coef * multinomial_loss(y_text, x)
                               + config.joint_coef * multinomial_loss(y_joint, x))
                elif config.loss_type == "mse":
                    l_recon = (config.text_coef * mse_loss(y_text, x)
                               + config.joint_coef * mse_loss(y_joint, x))
                elif config.loss_type == "nmse":
                    l_recon = (config.text_coef * nmse_loss(y_text, x)
                               + config.joint_coef * nmse_loss(y_joint, x))
                elif config.loss_type == "hellinger":
                    l_recon = (config.text_coef * hellinger_loss(y_text, x)
                               + config.joint_coef * hellinger_loss(y_joint, x))
                elif config.loss_type == "distill":
                    l_recon = mse_loss(y_text, y_colab)
                else:
                    l_recon = torch.tensor(0.0, device=device)

                # Tag aux
                l_tag = config.tag_coef * hellinger_loss(y_tags_pred, y_tags_true, tag_weights)

                # Alignment: global InfoNCE across GPUs with --infonce, otherwise cosine
                if config.infonce:
                    l_align = infonce_loss_global(z_text, z_colab.detach()) * align_coef
                else:
                    l_align = (1 - F.cosine_similarity(z_text, z_colab.detach()).mean()) * align_coef

                if config.ortho_coef > 0:
                    l_align = l_align + vicreg_cov_loss(z_text, z_colab.detach()) * config.ortho_coef

                loss = l_recon + l_tag + l_align

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            align_coef = max(align_coef - delta, config.repulsion_coef_min)

            bs = x.size(0)
            loss_tracker.update(
                bs,
                recon=l_recon.item() if torch.is_tensor(l_recon) else l_recon,
                align=l_align.item(),
                tag=l_tag.item(),
                total=loss.item(),
            )
            if is_main():
                pbar.set_postfix(
                    recon=f"{l_recon.item() if torch.is_tensor(l_recon) else 0:.4f}",
                    align=f"{l_align.item():.4f}",
                    tag=f"{l_tag.item():.4f}")

        epoch_losses = loss_tracker.end_epoch()
        log(f"Epoch {epoch+1} losses: " + " | ".join(f"{k}={v:.4f}" for k, v in epoch_losses.items()))

        # --- Validation (rank 0) ---
        if is_main():
            log(f"\n--- Validation Epoch {epoch+1} ---")
            m_joint, m_text, m_colab = run_eval(
                model, loader_val, device, val_joint, val_text, val_colab,
                desc=f"[VAL] Epoch {epoch+1}")

            if m_text["ndcg@100"] > best_ndcg_text:
                best_ndcg_text = m_text["ndcg@100"]
                torch.save(raw_model.state_dict(), os.path.join(exp_dir, "best_text.pt"))
                log(f"  -> New best text NDCG@100: {best_ndcg_text:.4f}")

            if m_joint["ndcg@100"] > best_ndcg_joint:
                best_ndcg_joint = m_joint["ndcg@100"]
                torch.save(raw_model.state_dict(), os.path.join(exp_dir, "best_joint.pt"))
                log(f"  -> New best joint NDCG@100: {best_ndcg_joint:.4f}")

            torch.save(raw_model.state_dict(), os.path.join(exp_dir, "last.pt"))

        dist.barrier()

    # --- Summary ---
    if is_main():
        log("\n" + "=" * 60)
        log("Training Loss History:")
        loss_tracker.print_history()
        log("\nValidation Best (Joint):")
        val_joint.print_best()
        log("Validation Best (Text):")
        val_text.print_best()
        log("Validation Best (Colab):")
        val_colab.print_best()

        # --- Test ---
        log("\n" + "=" * 60)
        log("TEST — Best Text Model")
        log("=" * 60)
        test_model = LatentAlignModel(
            config.classifier_path, config.encoder_path, config.decoder_path,
            data["n_items"], config.use_linear, config.tau, config.nonlinear
        ).to(device)
        test_model.load_state_dict(torch.load(os.path.join(exp_dir, "best_text.pt"), map_location=device))
        t_j, t_t, t_c = MetricAccumulator(), MetricAccumulator(), MetricAccumulator()
        run_eval(test_model, loader_test, device, t_j, t_t, t_c, desc="[TEST-TEXT]")

        log("\n" + "=" * 60)
        log("TEST — Best Joint Model")
        log("=" * 60)
        test_model.load_state_dict(torch.load(os.path.join(exp_dir, "best_joint.pt"), map_location=device))
        t_j2, t_t2, t_c2 = MetricAccumulator(), MetricAccumulator(), MetricAccumulator()
        run_eval(test_model, loader_test, device, t_j2, t_t2, t_c2, desc="[TEST-JOINT]")

        log_file.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()