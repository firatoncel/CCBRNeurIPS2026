"""
main_tears.py — DDP training for TearsAlignModel.

TEARS-style alignment using our pretrained Multi-VAE / MacridVAE checkpoints
and our text encoder. Loss is 3-branch multinomial reconstruction + KL +
Bures-Wasserstein OT alignment.

Usage:
    torchrun --nproc_per_node=1 experiments/MSD/main_tears.py \
        --data-path data/msd \
        --encoder-path checkpoints/msd/enc_mult_vae.pt \
        --decoder-path checkpoints/msd/dec_mult_vae.pt \
        --epsilon 1.0 --max-anneal-steps 10000 --min-anneal 0.5
"""

import os
import json
import argparse
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model_tears import TearsAlignModel, TearsOTLoss


from utils import (
    set_seed, read_data, MSDDataset, MSDDatasetVal,
    collate_fn, collate_fn_val,
    recall_ndcg_multi_k_full,
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
            if n in self.totals:
                self.totals[n] += v * batch_size
        self.count += batch_size
    def end_epoch(self):
        avgs = {n: (self.totals[n] / self.count if self.count > 0 else 0.0)
                for n in self.names}
        self.history.append(avgs)
        self.reset()
        return avgs
    def print_history(self):
        for i, h in enumerate(self.history):
            parts = " | ".join(f"{n}={h[n]:.4f}" for n in self.names)
            log(f"Epoch {i+1}: {parts}")


class MetricAccumulator:
    """Stores per-batch (NDCG@100, R@50, R@20) and gives averages over an epoch."""
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
        if not self.history:
            log("(no history)")
            return
        best = max(self.history, key=lambda m: m["ndcg@100"])
        ep = self.history.index(best) + 1
        log(f"Best Epoch {ep}: NDCG@100={best['ndcg@100']:.4f} | "
            f"Recall@50={best['recall@50']:.4f} | Recall@20={best['recall@20']:.4f}")


# ============================================================================
# Evaluation (rank 0 only)
# ============================================================================
def run_eval(model, loader, device,
             tracker_merged, tracker_text, tracker_rec,
             alpha, desc="Val"):
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, disable=not is_main()):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            summaries = batch["summary"]
            with torch.autocast(device_type="cuda"):
                (logits_merged, logits_rec, logits_text,
                 _, _, _, _) = raw_model(x, summaries, alpha=alpha)
            tracker_merged.update(logits_merged, x, y)
            tracker_text.update(logits_text, x, y)
            tracker_rec.update(logits_rec, x, y)

    m_merged = tracker_merged.end_epoch()
    m_text = tracker_text.end_epoch()
    m_rec = tracker_rec.end_epoch()
    log(f"  [MERGED] NDCG@100={m_merged['ndcg@100']:.4f} | R@50={m_merged['recall@50']:.4f} | R@20={m_merged['recall@20']:.4f}")
    log(f"  [TEXT]   NDCG@100={m_text['ndcg@100']:.4f} | R@50={m_text['recall@50']:.4f} | R@20={m_text['recall@20']:.4f}")
    log(f"  [REC]    NDCG@100={m_rec['ndcg@100']:.4f} | R@50={m_rec['recall@50']:.4f} | R@20={m_rec['recall@20']:.4f}")
    return m_merged, m_text, m_rec


# ============================================================================
# Args
# ============================================================================
parser = argparse.ArgumentParser()

parser.add_argument("--data-path", type=str, default="data/msd")
parser.add_argument("--encoder-path", type=str, required=True,
                    help="Path to *_mult_vae.pt or *_macrid_vae.pt")
parser.add_argument("--decoder-path", type=str, required=True)
parser.add_argument("--logdir", type=str, default="results_tears")
parser.add_argument("--prefix", type=str, default="mult_vae")

parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate for AdamW (TEARS uses 1e-3)")

# --- Reconstruction weights ---
parser.add_argument("--recon-tau", type=float, default=1.0,
                    help="Weight on merged-branch BCE")
parser.add_argument("--text-tau", type=float, default=1.0,
                    help="Weight on text-branch BCE")
parser.add_argument("--rec-tau", type=float, default=1.0,
                    help="Weight on collab-branch BCE")

# --- KL annealing (TEARS-style linear ramp) ---
parser.add_argument("--max-anneal-steps", type=int, default=10000,
                    help="Linear KL anneal step count")
parser.add_argument("--min-anneal", type=float, default=0.5,
                    help="Saturation value of the KL anneal coefficient")
parser.add_argument("--anneal-cap", type=float, default=0.5,
                    help="Anneal value once max_anneal_steps reached")

# --- OT alignment ---
parser.add_argument("--epsilon", type=float, default=1.0,
                    help="Coefficient on the Bures-Wasserstein OT loss")

# --- Mixing ---
parser.add_argument("--alpha", type=float, default=0.5,
                    help="Mixing coefficient of the merged branch (training and test)")

# --- LoRA ---
parser.add_argument("--no-lora", action="store_true",
                    help="Disable LoRA on the text encoder")
parser.add_argument("--lora-r", type=int, default=64)
parser.add_argument("--lora-alpha", type=int, default=16)
parser.add_argument("--lora-dropout", type=float, default=0.1)


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

    dataset_tr = MSDDataset(data["X_tr"], data["S_tr"], data["n_items"])
    train_sampler = DistributedSampler(dataset_tr, num_replicas=world_size,
                                        rank=dist.get_rank(), shuffle=True)
    loader_tr = DataLoader(dataset_tr, batch_size=config.batch_size, sampler=train_sampler,
                           num_workers=config.num_workers, collate_fn=collate_fn,
                           pin_memory=True, drop_last=True)

    dataset_val = MSDDatasetVal(data["X_val"], data["S_val"], data["n_items"], data["Y_val"])
    loader_val = DataLoader(dataset_val, batch_size=100, shuffle=False,
                            num_workers=config.num_workers, collate_fn=collate_fn_val)

    dataset_test = MSDDatasetVal(data["X_test"], data["S_test"], data["n_items"], data["Y_test"])
    loader_test = DataLoader(dataset_test, batch_size=100, shuffle=False,
                             num_workers=config.num_workers, collate_fn=collate_fn_val)

    # --- Model ---
    model = TearsAlignModel(
        encoder_path=config.encoder_path,
        decoder_path=config.decoder_path,
        alpha=config.alpha,
        lora=not config.no_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    ).to(device)

    backbone_kind = model.backbone_kind
    log(f"[main_tears] backbone_kind={backbone_kind}")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)
    raw_model = model.module

    # --- Optimizer (TEARS uses AdamW for the LLM-based variants) ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    n_params = sum(p.numel() for p in trainable)
    log(f"Trainable params: {n_params:,} ({n_params / 1e6:.2f}M)")

    warmup_steps = len(loader_tr)
    total_steps = len(loader_tr) * config.epochs
    scheduler = get_warmup_linear_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1)

    scaler = GradScaler("cuda")

    # --- Loss ---
    if backbone_kind == "macrid":
        kfac = raw_model.vae.n_concepts
        split = raw_model.vae.latent_dim
    else:
        kfac = split = None
    loss_fn = TearsOTLoss(
        recon_tau=config.recon_tau,
        text_tau=config.text_tau,
        rec_tau=config.rec_tau,
        backbone_kind=backbone_kind,
        kfac=kfac, split=split,
    )

    # --- Experiment dir (rank 0) ---
    ts = datetime.now()
    exp_name = f"{ts.year}_{ts.month}_{ts.day}-{ts.hour}_{ts.minute}_{ts.second}_{config.prefix}"
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
    loss_tracker = LossTracker(["bce", "bce_merged", "bce_text", "bce_rec",
                                 "wasserstein", "total"])
    val_merged = MetricAccumulator()
    val_text = MetricAccumulator()
    val_rec = MetricAccumulator()

    best_ndcg_avg = 0.0
    best_ndcg_merged = 0.0

    update_count = 0

    # --- Training loop ---
    for epoch in range(config.epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        pbar = tqdm(loader_tr, desc=f"Epoch {epoch+1}/{config.epochs}",
                    disable=not is_main())

        for batch in pbar:
            x = batch["x"].to(device)
            summaries = batch["summary"]
            optimizer.zero_grad()

            # KL annealing schedule (TEARS-style linear ramp)
            if config.max_anneal_steps > update_count:
                anneal = min(config.min_anneal,
                             1.0 * update_count / config.max_anneal_steps)
            else:
                anneal = config.anneal_cap

            with torch.autocast(device_type="cuda"):
                (logits_merged, logits_rec, logits_text,
                 text_mu, text_logvar,
                 rec_mu, rec_logvar) = model(x, summaries)

                total, BCE, wasserstein, BCE_rec, BCE_text, BCE_merged = loss_fn(
                    logits_merged=logits_merged,
                    logits_rec=logits_rec,
                    logits_text=logits_text,
                    x=x,
                    text_mu=text_mu, text_logvar=text_logvar,
                    rec_mu=rec_mu, rec_logvar=rec_logvar,
                    anneal=anneal,
                    epsilon=config.epsilon,
                )

            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_count += 1

            bs = x.size(0)
            loss_tracker.update(
                bs,
                bce=BCE.item(),
                bce_merged=BCE_merged.item(),
                bce_text=BCE_text.item(),
                bce_rec=BCE_rec.item(),
                wasserstein=wasserstein.item(),
                total=total.item(),
            )
            if is_main():
                pbar.set_postfix(
                    bce=f"{BCE.item():.4f}",
                    wl=f"{wasserstein.item():.4f}",
                    anneal=f"{anneal:.3f}",
                )

        epoch_losses = loss_tracker.end_epoch()
        log(f"Epoch {epoch+1}: " +
            " | ".join(f"{k}={v:.4f}" for k, v in epoch_losses.items()) +
            f" | anneal={anneal:.3f}")

        # --- Validation (rank 0) ---
        if is_main():
            log(f"\n--- Validation Epoch {epoch+1} ---")
            m_merged, m_text, m_rec = run_eval(
                model, loader_val, device,
                val_merged, val_text, val_rec,
                alpha=config.alpha,
                desc=f"[VAL] Epoch {epoch+1}")

            ndcg_avg = (m_merged["ndcg@100"] + m_text["ndcg@100"] + m_rec["ndcg@100"]) / 3.0
            log(f"  [AVG]    NDCG@100={ndcg_avg:.4f}")

            if ndcg_avg > best_ndcg_avg:
                best_ndcg_avg = ndcg_avg
                torch.save(raw_model.state_dict(), os.path.join(exp_dir, "best_avg.pt"))
                log(f"  -> New best avg NDCG@100: {best_ndcg_avg:.4f}")

            if m_merged["ndcg@100"] > best_ndcg_merged:
                best_ndcg_merged = m_merged["ndcg@100"]
                torch.save(raw_model.state_dict(), os.path.join(exp_dir, "best_merged.pt"))
                log(f"  -> New best merged NDCG@100: {best_ndcg_merged:.4f}")

            torch.save(raw_model.state_dict(), os.path.join(exp_dir, "last.pt"))

        dist.barrier()

    # --- Summary + test (rank 0) ---
    if is_main():
        log("\n" + "=" * 60)
        log("Training Loss History:")
        loss_tracker.print_history()
        log("\nValidation Best (Merged):")
        val_merged.print_best()
        log("Validation Best (Text):")
        val_text.print_best()
        log("Validation Best (Rec):")
        val_rec.print_best()

        # --- Test: best_merged checkpoint, merged branch at alpha=config.alpha ---
        log("\n" + "=" * 60)
        log(f"TEST — best_merged checkpoint, alpha={config.alpha}")
        log("=" * 60)
        test_model = TearsAlignModel(
            encoder_path=config.encoder_path,
            decoder_path=config.decoder_path,
            alpha=config.alpha,
            lora=not config.no_lora,
            lora_r=config.lora_r, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        ).to(device)
        test_model.load_state_dict(
            torch.load(os.path.join(exp_dir, "best_merged.pt"), map_location=device))
        t_m, t_t, t_r = MetricAccumulator(), MetricAccumulator(), MetricAccumulator()
        run_eval(test_model, loader_test, device,
                 t_m, t_t, t_r, alpha=config.alpha,
                 desc="[TEST]")

        log_file.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
