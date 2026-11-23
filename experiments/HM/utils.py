import re
import json
import torch
import random
import numpy as np
import os
from collections import Counter
from torch.utils.data import Dataset
import torch.nn.functional as F


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
# Attributes (indices follow the sorted attribute names)
# ============================================================
CATEGORY_MAP = {'Bag': 0, 'Ballerinas': 1, 'Belt': 2, 'Bikini top': 3, 'Blazer': 4, 'Blouse': 5, 'Bodysuit': 6, 'Boots': 7, 'Bra': 8, 'Cardigan': 9, 'Coat': 10, 'Costumes': 11, 'Dress': 12, 'Hair clip': 13, 'Hair string': 14, 'Hair ties': 15, 'Hair/alice band': 16, 'Hairband': 17, 'Hat/beanie': 18, 'Hoodie': 19, 'Jacket': 20, 'Jumpsuit/Playsuit': 21, 'Leggings/Tights': 22, 'Necklace': 23, 'Night gown': 24, 'Nipple covers': 25, 'Polo shirt': 26, 'Pyjama bottom': 27, 'Pyjama set': 28, 'Robe': 29, 'Sandals': 30, 'Sarong': 31, 'Scarf': 32, 'Shirt': 33, 'Shorts': 34, 'Skirt': 35, 'Slippers': 36, 'Sneakers': 37, 'Socks': 38, 'Sunglasses': 39, 'Sweater': 40, 'Swimsuit': 41, 'Swimwear bottom': 42, 'Swimwear set': 43, 'T-shirt': 44, 'Top': 45, 'Trousers': 46, 'Underwear Tights': 47, 'Underwear body': 48, 'Underwear bottom': 49, 'Vest top': 50}

COLOR_MAP = {'Beige': 0, 'Black': 1, 'Blue': 2, 'Bronze/Copper': 3, 'Gold': 4, 'Green': 5, 'Grey': 6, 'Orange': 7, 'Pink': 8, 'Purple': 9, 'Red': 10, 'Silver': 11, 'Transparent': 12, 'Turquoise': 13, 'White': 14, 'Yellow': 15}


def compute_pos_weights(cat_json_path, col_json_path, device="cuda", max_weight=1):
    """
    Per-class weights for the product-type and colour heads: inverse square
    root of the number of catalogue items with that attribute, divided by the
    mean and clipped at max_weight.

    Returns:
        cat_pos_weight: torch.FloatTensor of shape (n_categories,)
        col_pos_weight: torch.FloatTensor of shape (n_colors,)
    """
    with open(cat_json_path, "r") as f:
        cat_labels = json.load(f)
    with open(col_json_path, "r") as f:
        col_labels = json.load(f)

    cat_counts = Counter(cat_labels.values())
    col_counts = Counter(col_labels.values())

    # Attributes with no item get a count of 1
    cat_freq = np.array([cat_counts.get(name, 1) for name, _ in
                         sorted(CATEGORY_MAP.items(), key=lambda x: x[1])])
    col_freq = np.array([col_counts.get(name, 1) for name, _ in
                         sorted(COLOR_MAP.items(), key=lambda x: x[1])])

    cat_w = 1.0 / np.sqrt(cat_freq.astype(np.float32))
    col_w = 1.0 / np.sqrt(col_freq.astype(np.float32))
    cat_w = np.clip((cat_w / cat_w.mean()), a_min=0, a_max=max_weight)
    col_w = np.clip((col_w / col_w.mean()), a_min=0, a_max=max_weight)

    cat_pos_weight = torch.FloatTensor(cat_w).to(device)
    col_pos_weight = torch.FloatTensor(col_w).to(device)

    print("=== Category weights ===")
    cat_sorted = sorted(CATEGORY_MAP.items(), key=lambda x: cat_w[x[1]], reverse=True)
    for name, idx in cat_sorted:
        print(f"  {name:25s} count={cat_freq[idx]:6d}  weight={cat_w[idx]:.3f}")

    print("\n=== Color weights  ===")
    col_sorted = sorted(COLOR_MAP.items(), key=lambda x: col_w[x[1]], reverse=True)
    for name, idx in col_sorted:
        print(f"  {name:25s} count={col_freq[idx]:6d}  weight={col_w[idx]:.3f}")

    return cat_pos_weight, col_pos_weight


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


def vicreg_cov_loss(z_text, z_colab):
    """Mean squared cross-covariance between the text and collaborative latents."""
    z_text = z_text - z_text.mean(0)
    z_colab = z_colab - z_colab.mean(0)
    N = z_text.size(0)
    cov = (z_text.T @ z_colab) / (N - 1)
    return cov.pow(2).mean()


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


def read_data(path, augment=False):
    """Load the H&M splits; with augment=True the augmented training users and
    summaries (train_numeric[_summaries]_augmented.json) are used."""
    suffix = "_augmented" if augment else ""
    S_tr = listify_text(load_json(f"{path}/train_numeric_summaries{suffix}.json"))
    S_val = listify_text(load_json(f"{path}/valid_numeric_summaries.json"))
    S_test = listify_text(load_json(f"{path}/test_numeric_summaries.json"))

    X_tr = listify(load_json(f"{path}/train_numeric{suffix}.json"))
    X_val = listify(load_json(f"{path}/valid_numeric.json"))
    Y_val = listify(load_json(f"{path}/valid_te_numeric.json"))
    X_test = listify(load_json(f"{path}/test_numeric.json"))
    Y_test = listify(load_json(f"{path}/test_te_numeric.json"))

    with open(f"{path}/num_unique_items.txt", "r", encoding="utf-8") as f:
        n_items = int(f.read().strip())

    print(f"# Train Users: {len(X_tr)} - # Val Users: {len(X_val)} - # Test Users: {len(X_test)} - # items: {n_items}")

    return {
        "S_tr": S_tr, "S_val": S_val, "S_test": S_test,
        "X_tr": X_tr, "X_val": X_val, "Y_val": Y_val,
        "X_test": X_test, "Y_test": Y_test,
        "n_items": n_items,
    }


# ============================================================
# Datasets
# ============================================================
class HMSummaryDataset(Dataset):
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


class HMSummaryDatasetVal(Dataset):
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
