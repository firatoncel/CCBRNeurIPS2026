"""BPR matrix factorisation that produces the item embeddings used by CCBR (H&M).

Writes item_embeddings_constrained.npy (and mf_training_history.json) to
--output-dir, which defaults to --data-path. The preprocessed data already
ships these embeddings, so running this overwrites them unless you pass
another --output-dir.

Usage (from the repository root):
    python experiments/HM/mf_upgraded.py --data-path data/hm
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import numpy as np
import os
import json


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ------------------------------------------------------
# 1) Dataset with Multi-Negative Sampling
# ------------------------------------------------------

class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, interactions, n_items, n_negatives=10):
        self.users = list(interactions.keys())
        self.interactions = interactions
        self.n_items = n_items
        self.n_negatives = n_negatives
        # Pre-convert to sets for O(1) lookup instead of list membership check
        self.interaction_sets = {
            uid: set(items) for uid, items in interactions.items()
        }

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        pos_set = self.interaction_sets[uid]

        # 1 positive item
        pos = random.choice(self.interactions[uid])

        # N negative items via batch rejection (much faster than 1-by-1 loop)
        negatives = []
        while len(negatives) < self.n_negatives:
            candidates = torch.randint(0, self.n_items, (self.n_negatives * 2,))
            for c in candidates.tolist():
                if c not in pos_set:
                    negatives.append(c)
                    if len(negatives) == self.n_negatives:
                        break

        return int(uid), pos, negatives


def collate_fn(batch):
    uids, positives, negatives = zip(*batch)
    uids = torch.tensor(uids, dtype=torch.long)
    positives = torch.tensor(positives, dtype=torch.long)
    negatives = torch.tensor(negatives, dtype=torch.long)  # (B, n_neg)
    return uids, positives, negatives


# ------------------------------------------------------
# 2) Matrix Factorization Model
# ------------------------------------------------------

class MF(nn.Module):
    def __init__(self, n_users, n_items, dim=128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

        # Xavier init — better than fixed std for varying dim sizes
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def forward_pos_neg(self, u, pos, neg):
        """
        u:   (B,)
        pos: (B,)
        neg: (B, n_neg)
        """
        u_e = self.user_emb(u)          # (B, d)
        pos_e = self.item_emb(pos)      # (B, d)
        neg_e = self.item_emb(neg)      # (B, n_neg, d)

        pos_scores = (u_e * pos_e).sum(dim=1)                        # (B,)
        neg_scores = (u_e.unsqueeze(1) * neg_e).sum(dim=2)           # (B, n_neg)

        return pos_scores, neg_scores


# ------------------------------------------------------
# 3) BPR Loss with Multi-Negative Support
# ------------------------------------------------------

def bpr_loss_multi_neg(pos_scores, neg_scores):
    """
    pos_scores: (B,)
    neg_scores: (B, n_neg)

    For each negative, compute BPR and average across negatives and batch.
    """
    # (B, n_neg) — broadcast pos_scores across negatives
    diff = pos_scores.unsqueeze(1) - neg_scores
    loss = -torch.log(torch.sigmoid(diff) + 1e-8)
    return loss.mean()


# ------------------------------------------------------
# 4) Training Function
# ------------------------------------------------------

def train_mf(
    interactions,
    n_users,
    n_items,
    dim=768,
    lr=1e-3,
    weight_decay=1e-4,
    item_weight_decay=1e-2,
    norm_lambda=1e-3,
    epochs=250,
    batch_size=512,
    n_negatives=10,
    device="cuda",
    log_every=10,
):
    set_seed(0)

    dataset = InteractionDataset(interactions, n_items, n_negatives=n_negatives)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = MF(n_users, n_items, dim).to(device)

    # Separate param groups: heavier weight decay on item embeddings
    optimizer = optim.AdamW([
        {"params": model.user_emb.parameters(), "weight_decay": weight_decay},
        {"params": model.item_emb.parameters(), "weight_decay": item_weight_decay},
    ], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Tracking
    history = {
        "epoch_loss": [],
        "epoch_norm_penalty": [],
        "lr": [],
        "item_emb_norm_mean": [],
        "item_emb_norm_std": [],
        "user_emb_norm_mean": [],
        "user_emb_norm_std": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_norm_pen = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for uid, pos, neg in pbar:
            uid = uid.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)

            pos_scores, neg_scores = model.forward_pos_neg(uid, pos, neg)
            loss = bpr_loss_multi_neg(pos_scores, neg_scores)

            # L2 penalty on item embeddings
            norm_penalty = (model.item_emb.weight ** 2).sum(dim=1).mean()
            loss = loss + norm_lambda * norm_penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_norm_pen += norm_penalty.item()
            n_batches += 1
            pbar.set_postfix(
                loss=f"{epoch_loss / n_batches:.4f}",
                norm_pen=f"{epoch_norm_pen / n_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_norm_pen = epoch_norm_pen / n_batches
        current_lr = scheduler.get_last_lr()[0]

        # Embedding norm monitoring (cheap, no grad needed)
        with torch.no_grad():
            item_norms = model.item_emb.weight.norm(dim=1)
            user_norms = model.user_emb.weight.norm(dim=1)

        history["epoch_loss"].append(avg_loss)
        history["epoch_norm_penalty"].append(avg_norm_pen)
        history["lr"].append(current_lr)
        history["item_emb_norm_mean"].append(item_norms.mean().item())
        history["item_emb_norm_std"].append(item_norms.std().item())
        history["user_emb_norm_mean"].append(user_norms.mean().item())
        history["user_emb_norm_std"].append(user_norms.std().item())

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"  [Epoch {epoch+1:>3d}] "
                f"loss={avg_loss:.4f} | "
                f"lr={current_lr:.2e} | "
                f"item_norm={item_norms.mean():.3f}±{item_norms.std():.3f} | "
                f"user_norm={user_norms.mean():.3f}±{user_norms.std():.3f}"
            )

    return model, history


# ------------------------------------------------------
# 5) Main
# ------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default="data/hm",
                        help="Folder with train_numeric.json and num_unique_items.txt")
    parser.add_argument("--output-dir", default=None,
                        help="Where the embeddings are written (default: --data-path)")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main_path = args.data_path
    out_dir = args.output_dir or args.data_path
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{main_path}/train_numeric.json") as f:
        interactions_int = json.load(f)
    interactions_int = {int(k): v for k, v in interactions_int.items()}

    n_items = int(np.loadtxt(f"{main_path}/num_unique_items.txt", dtype=int))
    n_users = len(interactions_int)
    print(f"Items: {n_items}, Users: {n_users}")

    model, history = train_mf(
        interactions_int,
        n_users,
        n_items,
        dim=1024,
        lr=1e-3,
        weight_decay=1e-4,
        item_weight_decay=1e-2,
        norm_lambda=1e-3,
        epochs=args.epochs,
        batch_size=512,
        n_negatives=10,
        log_every=10,
        device=args.device,
    )

    # Save item embeddings
    item_vectors = model.item_emb.weight.data.cpu().numpy()
    np.save(f"{out_dir}/item_embeddings_constrained.npy", item_vectors)

    # Save training history for analysis
    with open(f"{out_dir}/mf_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Print final embedding stats
    print("\n--- Final Embedding Statistics ---")
    item_norms = np.linalg.norm(item_vectors, axis=1)
    print(f"Item embedding norms: mean={item_norms.mean():.4f}, std={item_norms.std():.4f}, "
          f"min={item_norms.min():.4f}, max={item_norms.max():.4f}")
    print(f"Item embedding shape: {item_vectors.shape}")