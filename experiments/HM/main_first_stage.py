import os
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
    hellinger_loss, compute_pos_weights, set_seed, read_data,
    HMSummaryDataset, HMSummaryDatasetVal, collate_fn, collate_fn_val
)
from model import Classifier


# ============================================================
# Distribution metrics
# ============================================================
def eval_distribution_metrics(pred_logits, target_counts):
    """Hellinger, cosine, Spearman and KL between the predicted and target distributions."""
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
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--latent-dim', type=int, default=1024)
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--lr-text', type=float, default=5e-6)
parser.add_argument('--lr-W', type=float, default=3e-4)
parser.add_argument('--cat-coef', type=float, default=1, help="Weight of the product-type loss")
parser.add_argument('--col-coef', type=float, default=1, help="Weight of the colour loss")
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--data-path', type=str, default="data/hm")
parser.add_argument('--model-name', type=str, default="BAAI/bge-m3")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--embeddings-name', type=str, default="constrained")
parser.add_argument('--logdir', type=str, default="results")
parser.add_argument('--replace-cls', default=False, action='store_true')
parser.add_argument('--weight', default=False, action='store_true',
                    help="Weight the Hellinger loss per class (see utils.compute_pos_weights)")
parser.add_argument('--augment', default=False, action='store_true',
                    help="Train on the augmented users and summaries")

config = parser.parse_args()
print(config)
device = config.device

# ============================================================
# Data
# ============================================================
data = read_data(config.data_path, augment=config.augment)

dataset_tr = HMSummaryDataset(data['X_tr'], data['S_tr'], data['n_items'])
loader_tr = DataLoader(dataset_tr, batch_size=config.batch_size, shuffle=True,
                       num_workers=config.num_workers, collate_fn=collate_fn)

dataset_val = HMSummaryDatasetVal(data['X_val'], data['S_val'], data['n_items'], data['Y_val'])
loader_val = DataLoader(dataset_val, batch_size=10, shuffle=False,
                        num_workers=config.num_workers, collate_fn=collate_fn_val)

dataset_test = HMSummaryDatasetVal(data['X_test'], data['S_test'], data['n_items'], data['Y_test'])
loader_test = DataLoader(dataset_test, batch_size=10, shuffle=False,
                         num_workers=config.num_workers, collate_fn=collate_fn_val)

# ============================================================
# Model
# ============================================================
model = Classifier(
    item_emb_path=f'{config.data_path}/item_embeddings_{config.embeddings_name}.npy',
    path=config.data_path,
    d_latent=config.latent_dim,
    model_name=config.model_name,
    replace_cls=config.replace_cls,
    normalize=False,
    tau=config.tau,
).to(device)
params = [
    {"params": model.linear_color.parameters(), "lr": config.lr_W},
    {"params": model.linear.parameters(), "lr": config.lr_W},
    {"params": model.text_encoder.parameters(), "lr": config.lr_text},
]
optimizer = torch.optim.AdamW(params)
n_params = sum(p.numel() for g in params for p in g["params"] if p.requires_grad)
print(f"Optimized params: {n_params:,} ({n_params/1e6:.2f}M)")

# ============================================================
# Logging setup
# ============================================================
a = datetime.now()
exp_date = f"{a.year}_{a.month}_{a.day}-{a.hour}_{a.minute}_{a.second}"
main_path = f"{config.logdir}/{exp_date}"
os.makedirs(main_path)
print(main_path)

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
if config.weight:
    cat_pos_weight, col_pos_weight = compute_pos_weights(
        f'{config.data_path}/item_to_product_type.json', f'{config.data_path}/item_color.json')
    cat_pos_weight = cat_pos_weight.to(device)
    col_pos_weight = col_pos_weight.to(device)
else:
    cat_pos_weight = None
    col_pos_weight = None

# ============================================================
# Train loop
# ============================================================
for epoch in range(config.epochs):
    model.train()
    pbar = tqdm(loader_tr, desc=f"Epoch {epoch+1}")
    total_loss, n_batches = 0, 0
    all_pred, all_true = [], []
    all_pred_c, all_true_c = [], []

    for batch in pbar:
        x = batch['x'].to(device)
        summaries = batch['summary']
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            y_aux_true, y_aux_true_color, y_aux_pred, y_aux_pred_color, z_text = model(x, summaries, config.dropout)
            loss = (
                config.cat_coef * hellinger_loss(y_aux_pred, y_aux_true, cat_pos_weight) +
                config.col_coef * hellinger_loss(y_aux_pred_color, y_aux_true_color, col_pos_weight)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        all_pred.append(y_aux_pred.detach().float().cpu())
        all_true.append(y_aux_true.detach().float().cpu())
        all_pred_c.append(y_aux_pred_color.detach().float().cpu())
        all_true_c.append(y_aux_true_color.detach().float().cpu())
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'Loss': f"{total_loss/n_batches:.4f}"})

    # --- Target distribution plots (epoch 0) ---
    if epoch == 0:
        all_cat_np = torch.cat(all_true).numpy()
        all_col_np = torch.cat(all_true_c).numpy()

        cat_keys = model.product_dict
        col_keys = model.color_dict
        cat_names = sorted(cat_keys, key=cat_keys.get)
        col_names = sorted(col_keys, key=col_keys.get)

        # Per-sample active label count
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(all_cat_np.sum(axis=1), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title("Product Type — active labels per sample")
        axes[0].set_xlabel("# active product types")
        axes[0].set_ylabel("# samples")
        axes[1].hist(all_col_np.sum(axis=1), bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_title("Color — active labels per sample")
        axes[1].set_xlabel("# active colors")
        axes[1].set_ylabel("# samples")
        plt.tight_layout()
        plt.savefig(f'{main_path}/target_histograms_per_sample.png', dpi=150)
        plt.close()

        # Per-class frequency with label names (sorted by frequency)
        cat_freq = all_cat_np.sum(axis=0)
        cat_order = np.argsort(cat_freq)[::-1]
        fig2, ax2 = plt.subplots(figsize=(16, 6))
        ax2.bar(range(len(cat_order)), cat_freq[cat_order], alpha=0.7)
        ax2.set_xticks(range(len(cat_order)))
        ax2.set_xticklabels([cat_names[i] for i in cat_order], rotation=90, fontsize=8)
        ax2.set_title("Product Type — per-class frequency")
        ax2.set_ylabel("# samples")
        plt.tight_layout()
        plt.savefig(f'{main_path}/target_cat_freq.png', dpi=150)
        plt.close()

        col_freq = all_col_np.sum(axis=0)
        col_order = np.argsort(col_freq)[::-1]
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.bar(range(len(col_order)), col_freq[col_order], alpha=0.7, color='orange')
        ax3.set_xticks(range(len(col_order)))
        ax3.set_xticklabels([col_names[i] for i in col_order], rotation=90, fontsize=8)
        ax3.set_title("Color — per-class frequency")
        ax3.set_ylabel("# samples")
        plt.tight_layout()
        plt.savefig(f'{main_path}/target_col_freq.png', dpi=150)
        plt.close()

        log(f"  -> Histograms saved to {main_path}/")

    # --- Train metrics ---
    cat_hell, cat_cos, cat_spear, cat_kl = eval_distribution_metrics(
        torch.cat(all_pred).numpy(), torch.cat(all_true).numpy())
    col_hell, col_cos, col_spear, col_kl = eval_distribution_metrics(
        torch.cat(all_pred_c).numpy(), torch.cat(all_true_c).numpy())
    log(
        f"Epoch {epoch} [TRAIN] "
        f"Cat: Hell={cat_hell:.4f} Cos={cat_cos:.4f} Spear={cat_spear:.4f} KL={cat_kl:.4f} | "
        f"Col: Hell={col_hell:.4f} Cos={col_cos:.4f} Spear={col_spear:.4f} KL={col_kl:.4f} | "
        f"Loss={total_loss/n_batches:.4f}"
    )

    # --- Validation ---
    model.eval()
    all_pred, all_true = [], []
    all_pred_c, all_true_c = [], []
    with torch.no_grad():
        for batch in tqdm(loader_val, desc=f" [VAL] Epoch {epoch+1}"):
            x = batch['x'].to(device)
            summaries = batch['summary']
            with torch.autocast(device_type="cuda"):
                y_aux_true, y_aux_true_color, y_aux_pred, y_aux_pred_color, z_text = model(x, summaries)
            all_pred.append(y_aux_pred.detach().float().cpu())
            all_true.append(y_aux_true.detach().float().cpu())
            all_pred_c.append(y_aux_pred_color.detach().float().cpu())
            all_true_c.append(y_aux_true_color.detach().float().cpu())

    cat_hell, cat_cos, cat_spear, cat_kl = eval_distribution_metrics(
        torch.cat(all_pred).numpy(), torch.cat(all_true).numpy())
    col_hell, col_cos, col_spear, col_kl = eval_distribution_metrics(
        torch.cat(all_pred_c).numpy(), torch.cat(all_true_c).numpy())
    log(
        f"[VAL] "
        f"Cat: Hell={cat_hell:.4f} Cos={cat_cos:.4f} Spear={cat_spear:.4f} KL={cat_kl:.4f} | "
        f"Col: Hell={col_hell:.4f} Cos={col_cos:.4f} Spear={col_spear:.4f} KL={col_kl:.4f}"
    )

    # Model selection: mean validation Hellinger of the heads with a non-zero weight
    if config.cat_coef == 0:
        val_hellinger = col_hell
    elif config.col_coef == 0:
        val_hellinger = cat_hell
    else:
        val_hellinger = (cat_hell + col_hell) / 2

    if val_hellinger < best_val_hellinger:
        best_val_hellinger = val_hellinger
        torch.save(model, f'{main_path}/best_joint.pt')
        log(f"  -> New best avg Hellinger: {best_val_hellinger:.4f} (cat={cat_hell:.4f}, col={col_hell:.4f}) | KL: cat={cat_kl:.4f} col={col_kl:.4f}")


# ============================================================
# Test
# ============================================================
def evaluate_test(model_path, label):
    model = torch.load(model_path, weights_only=False)
    model.eval()
    all_pred, all_true = [], []
    all_pred_c, all_true_c = [], []
    with torch.no_grad():
        for batch in tqdm(loader_test, desc=f" [TEST] {label}"):
            x = batch['x'].to(device)
            summaries = batch['summary']
            with torch.autocast(device_type="cuda"):
                y_aux_true, y_aux_true_color, y_aux_pred, y_aux_pred_color, z_text = model(x, summaries)
            all_pred.append(y_aux_pred.detach().float().cpu())
            all_true.append(y_aux_true.detach().float().cpu())
            all_pred_c.append(y_aux_pred_color.detach().float().cpu())
            all_true_c.append(y_aux_true_color.detach().float().cpu())

    cat_hell, cat_cos, cat_spear, cat_kl = eval_distribution_metrics(
        torch.cat(all_pred).numpy(), torch.cat(all_true).numpy())
    col_hell, col_cos, col_spear, col_kl = eval_distribution_metrics(
        torch.cat(all_pred_c).numpy(), torch.cat(all_true_c).numpy())
    log(
        f"[TEST — {label}] "
        f"Cat: Hell={cat_hell:.4f} Cos={cat_cos:.4f} Spear={cat_spear:.4f} KL={cat_kl:.4f} | "
        f"Col: Hell={col_hell:.4f} Cos={col_cos:.4f} Spear={col_spear:.4f} KL={col_kl:.4f}"
    )


evaluate_test(f'{main_path}/best_joint.pt', "BEST JOINT")
log_file.close()
