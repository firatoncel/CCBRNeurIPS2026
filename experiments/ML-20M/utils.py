import re
import json
import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Seed
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================
# Loss functions
# ============================================================
def hellinger_loss(pred_logits, target_counts, weights=None):
    pe = target_counts / (target_counts.sum(dim=-1, keepdim=True) + 1e-6)
    qe = F.softmax(pred_logits, dim=-1)
    h_per_class = (pe.sqrt() - qe.sqrt()).pow(2)
    if weights is not None:
        h_per_class = h_per_class * weights.unsqueeze(0)
    h = h_per_class.sum(dim=-1) / 2.0
    return h.mean()


def infonce_loss(z_text, z_colab, temperature=0.07):
    z_t = F.normalize(z_text, dim=1)
    z_c = F.normalize(z_colab.detach(), dim=1)
    logits = z_t @ z_c.T / temperature
    labels = torch.arange(len(z_t), device=z_t.device)
    return F.cross_entropy(logits, labels)


# ============================================================
# Metrics
# ============================================================
def recall_ndcg_multi_k_full(logits, X_mask, Y_true, K_list):
    K_max = max(K_list)
    masked_logits = logits.masked_fill(X_mask == 1, float('-inf'))
    _, topk_idx = masked_logits.topk(K_max, dim=1)
    true_items = (Y_true > 0).float()
    num_true = torch.clamp(true_items.sum(dim=1), min=1)
    positions = torch.arange(1, K_max + 1, device=logits.device).float()
    discounts = 1.0 / torch.log2(positions + 1)
    results = {}
    for K in K_list:
        idx_k = topk_idx[:, :K]
        hits = true_items.gather(1, idx_k)
        trues = torch.clamp(num_true, max=K)
        recall_k = hits.sum(dim=1) / trues
        dcg = (hits * discounts[:K]).sum(dim=1)
        ideal_hits = torch.sort(true_items, descending=True, dim=1)[0][:, :K]
        ideal_dcg = (ideal_hits * discounts[:K]).sum(dim=1)
        ideal_dcg = torch.clamp(ideal_dcg, min=1e-6)
        ndcg_k = dcg / ideal_dcg
        results[f"recall@{K}"] = recall_k
        results[f"ndcg@{K}"] = ndcg_k
    return results


# ============================================================
# Data I/O
# ============================================================
def clean_summary(text):
    text = re.sub(r"\n\s*\n", " ", text)
    return text


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def listify_text(text_dict):
    return [clean_summary(text_dict[str(i)]['summary']) for i in range(len(text_dict))]


def listify(text_dict):
    return [text_dict[str(i)] for i in range(len(text_dict))]


def read_data(path):
    X_tr = listify(load_json(f"{path}/train_numeric.json"))
    X_val = listify(load_json(f"{path}/valid_numeric.json"))
    Y_val = listify(load_json(f"{path}/valid_te_numeric.json"))
    X_test = listify(load_json(f"{path}/test_numeric.json"))
    Y_test = listify(load_json(f"{path}/test_te_numeric.json"))

    try:
        S_tr = listify_text(load_json(f"{path}/train_numeric_summaries.json"))
    except Exception:
        S_tr = ["dummy"] * len(X_tr)

    try:
        S_val = listify_text(load_json(f"{path}/valid_numeric_summaries.json"))
    except Exception:
        S_val = ["dummy"] * len(X_val)

    try:
        S_test = listify_text(load_json(f"{path}/test_numeric_summaries.json"))
    except Exception:
        S_test = ["dummy"] * len(X_test)

    with open(f"{path}/num_unique_items.txt", "r", encoding="utf-8") as f:
        n_items = int(f.read().strip())

    print(f"# Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)} | Items: {n_items}")

    return {
        "S_tr": S_tr, "S_val": S_val, "S_test": S_test,
        "X_tr": X_tr, "X_val": X_val, "Y_val": Y_val,
        "X_test": X_test, "Y_test": Y_test,
        "n_items": n_items,
    }


# ============================================================
# Datasets
# ============================================================
class ML20MDataset(Dataset):
    """Train dataset: x (interaction vector) + summary text."""
    def __init__(self, X_json, S_json, n_items):
        self.X = X_json
        self.S = S_json
        self.n_items = n_items

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.zeros(self.n_items, dtype=torch.float32)
        x[self.X[idx]] = 1.0
        return {"x": x, "summary": self.S[idx]}


class ML20MDatasetVal(Dataset):
    """Val/Test dataset: x + y (held-out) + summary text."""
    def __init__(self, X_json, S_json, n_items, Y_json):
        self.X = X_json
        self.S = S_json
        self.Y = Y_json
        self.n_items = n_items

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.zeros(self.n_items, dtype=torch.float32)
        x[self.X[idx]] = 1.0
        y = torch.zeros(self.n_items, dtype=torch.float32)
        y[self.Y[idx]] = 1.0
        return {"x": x, "y": y, "summary": self.S[idx]}


# ============================================================
# Collate functions
# ============================================================
def collate_fn(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    summaries = [clean_summary(b["summary"]) for b in batch]
    return {"x": x, "summary": summaries}


def collate_fn_val(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    summaries = [clean_summary(b["summary"]) for b in batch]
    return {"x": x, "y": y, "summary": summaries}