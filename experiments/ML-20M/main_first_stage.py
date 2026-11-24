import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from utils import (
    hellinger_loss, set_seed, read_data,
    ML20MDataset, ML20MDatasetVal, collate_fn, collate_fn_val
)
from model import Classifier


# ============================================================
# Distribution metrics
# ============================================================
def eval_distribution_metrics(pred_logits, target_counts):
    if isinstance(pred_logits, np.ndarray):
        pred_logits = torch.from_numpy(pred_logits)
        target_counts = torch.from_numpy(target_counts)

    p = target_counts / (target_counts.sum(dim=-1, keepdim=True) + 1e-6)
    q = F.softmax(pred_logits, dim=-1)

    hellinger = ((p.sqrt() - q.sqrt()).pow(2).sum(dim=-1) / 2).sqrt().mean().item()
    cosine = F.cosine_similarity(p, q, dim=-1).mean().item()

    alpha = 0.01
    q_smooth = (1 - alpha) * q + alpha * p
    kl = (p * torch.log(p / (q_smooth + 1e-10) + 1e-10))
    kl = (kl * (p > 0).float()).sum(dim=-1).mean().item()

    p_np, q_np = p.numpy(), q.numpy()
    spearman_vals = []
    for i in range(len(p_np)):
        if p_np[i].std() < 1e-8 or q_np[i].std() < 1e-8:
            spearman_vals.append(0.0)
        else:
            spearman_vals.append(spearmanr(p_np[i], q_np[i]).correlation)
    spearman = np.mean(spearman_vals)

    return hellinger, cosine, spearman, kl


# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-workers', type=int, default=6)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--latent-dim', type=int, default=1024)
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--lora-r', type=int, default=8)
parser.add_argument('--lora-alpha', type=int, default=16)
parser.add_argument('--lr-text', type=float, default=1e-6)
parser.add_argument('--lr-W', type=float, default=3e-4)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--data-path', type=str, default="data/ml-20m")
parser.add_argument('--model-name', type=str, default="BAAI/bge-m3")
parser.add_argument('--tag-matrix-path', type=str, default="data/ml-20m/tag_matrix.npy")
parser.add_argument('--tag2idx-path', type=str, default="data/ml-20m/tag2idx.json")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--embeddings-name', type=str, default="constrained")
parser.add_argument('--logdir', type=str, default="results")
parser.add_argument('--replace-cls', default=False, action='store_true')
parser.add_argument('--lora', default=False, action='store_true')
parser.add_argument('--weighted', default=False, action='store_true')

config = parser.parse_args()
print(config)
device = config.device

# ============================================================
# Data
# ============================================================
data = read_data(config.data_path)

dataset_tr = ML20MDataset(data['X_tr'], data['S_tr'], data['n_items'])
loader_tr = DataLoader(dataset_tr, batch_size=config.batch_size, shuffle=True,
                       num_workers=config.num_workers, collate_fn=collate_fn)

dataset_val = ML20MDatasetVal(data['X_val'], data['S_val'], data['n_items'], data['Y_val'])
loader_val = DataLoader(dataset_val, batch_size=250, shuffle=False,
                        num_workers=config.num_workers, collate_fn=collate_fn_val)

dataset_test = ML20MDatasetVal(data['X_test'], data['S_test'], data['n_items'], data['Y_test'])
loader_test = DataLoader(dataset_test, batch_size=250, shuffle=False,
                         num_workers=config.num_workers, collate_fn=collate_fn_val)

# ============================================================
# Model
# ============================================================
model = Classifier(
    item_emb_path=f'{config.data_path}/item_embeddings_{config.embeddings_name}.npy',
    d_latent=config.latent_dim,
    model_name=config.model_name,
    replace_cls=config.replace_cls,
    normalize=False,
    lora=config.lora,
    lora_r=config.lora_r,
    lora_alpha=config.lora_alpha,
    tag_matrix_path=config.tag_matrix_path,
    tag2idx_path=config.tag2idx_path,
    tau=config.tau,
).to(device)
print(model)
params = [
    {"params": model.linear_tags.parameters(), "lr": config.lr_W},
    {"params": model.text_encoder.parameters(), "lr": config.lr_text},
]
optimizer = torch.optim.AdamW(params)
n_params = sum(p.numel() for g in params for p in g["params"] if p.requires_grad)

# ============================================================
# Logging setup
# ============================================================
a = datetime.now()
exp_date = f"{a.year}_{a.month}_{a.day}-{a.hour}_{a.minute}_{a.second}"
main_path = f"{config.logdir}/{exp_date}"
os.makedirs(main_path)

with open(f'{main_path}/config.txt', 'w') as f:
    print(vars(config), file=f)

log_file = open(f'{main_path}/train_log.txt', 'w')

def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log(str(config))
log(f"Optimized params: {n_params:,} ({n_params/1e6:.2f}M)")
log(main_path)

set_seed(config.seed)
scaler = GradScaler('cuda')
best_val_hellinger = 10000
tag_weights = model.tag_weights if config.weighted else None
print(tag_weights)

# Tag names for plotting
with open(config.tag2idx_path) as f:
    tag2idx = json.load(f)
tag_names = sorted(tag2idx, key=tag2idx.get)

# ============================================================
# Train loop
# ============================================================
for epoch in range(config.epochs):
    model.train()
    pbar = tqdm(loader_tr, desc=f"Epoch {epoch+1}")
    total_loss, n_batches = 0, 0
    all_pred, all_true = [], []

    for batch in pbar:
        x = batch['x'].to(device)
        summaries = batch['summary']
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            y_tags_true, y_tags_pred, z_text = model(x, summaries, config.dropout)
            loss = hellinger_loss(y_tags_pred, y_tags_true, tag_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        all_pred.append(y_tags_pred.detach().float().cpu())
        all_true.append(y_tags_true.detach().float().cpu())
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'Loss': f"{total_loss/n_batches:.4f}"})

    # --- Tag distribution plot (epoch 0) ---
    if epoch == 0:
        all_true_np = torch.cat(all_true).numpy()
        tag_freq = all_true_np.sum(axis=0)
        tag_order = np.argsort(tag_freq)[::-1]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        axes[0].hist(all_true_np.sum(axis=1), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title("Active tags per sample")
        axes[0].set_xlabel("# active tags")
        axes[0].set_ylabel("# samples")

        axes[1].bar(range(len(tag_order)), tag_freq[tag_order], alpha=0.7)
        axes[1].set_xticks(range(len(tag_order)))
        axes[1].set_xticklabels([tag_names[i] for i in tag_order], rotation=90, fontsize=7)
        axes[1].set_title("Per-tag frequency")
        axes[1].set_ylabel("Total weight")
        plt.tight_layout()
        plt.savefig(f'{main_path}/tag_distribution.png', dpi=150)
        plt.close()
        log("  -> Tag distribution plot saved")

    # --- Train metrics ---
    hell, cos, spear, kl = eval_distribution_metrics(
        torch.cat(all_pred).numpy(), torch.cat(all_true).numpy()
    )
    log(f"Epoch {epoch} [TRAIN] Hell={hell:.4f} Cos={cos:.4f} Spear={spear:.4f} KL={kl:.4f} | Loss={total_loss/n_batches:.4f}")

    # --- Validation ---
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(loader_val, desc=f" [VAL] Epoch {epoch+1}"):
            x = batch['x'].to(device)
            summaries = batch['summary']
            with torch.autocast(device_type="cuda"):
                y_tags_true, y_tags_pred, z_text = model(x, summaries)
            all_pred.append(y_tags_pred.detach().float().cpu())
            all_true.append(y_tags_true.detach().float().cpu())

    hell, cos, spear, kl = eval_distribution_metrics(
        torch.cat(all_pred).numpy(), torch.cat(all_true).numpy()
    )
    log(f"[VAL] Hell={hell:.4f} Cos={cos:.4f} Spear={spear:.4f} KL={kl:.4f}")

    if hell < best_val_hellinger:
        best_val_hellinger = hell
        torch.save(model, f'{main_path}/best.pt')
        log(f"  -> New best Hellinger: {best_val_hellinger:.4f} | Cos={cos:.4f} Spear={spear:.4f} KL={kl:.4f}")


# ============================================================
# Test
# ============================================================
def evaluate_test(model_path, label):
    model = torch.load(model_path, weights_only=False)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(loader_test, desc=f" [TEST] {label}"):
            x = batch['x'].to(device)
            summaries = batch['summary']
            with torch.autocast(device_type="cuda"):
                y_tags_true, y_tags_pred, z_text = model(x, summaries)
            all_pred.append(y_tags_pred.detach().float().cpu())
            all_true.append(y_tags_true.detach().float().cpu())

    hell, cos, spear, kl = eval_distribution_metrics(
        torch.cat(all_pred).numpy(), torch.cat(all_true).numpy()
    )
    log(f"[TEST — {label}] Hell={hell:.4f} Cos={cos:.4f} Spear={spear:.4f} KL={kl:.4f}")


evaluate_test(f'{main_path}/best.pt', "BEST")
log_file.close()