"""Shared helpers: seeding, data loading, datasets and ranking metrics."""
import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# Data
# ============================================================================

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def listify(index_dict):
    """Turn a {"0": ..., "1": ..., ...} dict into a list ordered by index."""
    return [index_dict[str(i)] for i in range(len(index_dict))]


def read_data(path):
    """Read the preprocessed user-item splits stored in `path`.

    `*_numeric.json` maps each user index to the item indices the user
    interacted with; `*_te_numeric.json` holds the held-out items that are
    used as targets for validation and test users.
    """
    def split(name):
        return listify(load_json(os.path.join(path, f'{name}.json')))

    data = {
        'X_tr': split('train_numeric'),
        'X_val': split('valid_numeric'),
        'Y_val': split('valid_te_numeric'),
        'X_test': split('test_numeric'),
        'Y_test': split('test_te_numeric'),
    }
    with open(os.path.join(path, 'num_unique_items.txt'), encoding='utf-8') as f:
        data['n_items'] = int(f.read().strip())

    print(f"# Train: {len(data['X_tr'])} | Val: {len(data['X_val'])} | "
          f"Test: {len(data['X_test'])} | Items: {data['n_items']}")
    return data


def multi_hot(items, n_items):
    x = torch.zeros(n_items, dtype=torch.float32)
    x[items] = 1.0
    return x


class TrainDataset(Dataset):
    """Multi-hot interaction vector of each training user."""

    def __init__(self, X, n_items):
        self.X = X
        self.n_items = n_items

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return multi_hot(self.X[idx], self.n_items)


class ValidTestDataset(Dataset):
    """(fold-in interactions, held-out interactions) of each val/test user."""

    def __init__(self, X, Y, n_items):
        self.X = X
        self.Y = Y
        self.n_items = n_items

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return multi_hot(self.X[idx], self.n_items), multi_hot(self.Y[idx], self.n_items)


# ============================================================================
# Metrics
# ============================================================================

def recall_at_k(scores, train, target, k):
    """Recall@k normalised by min(k, #held-out items), as in Liang et al. (2018)."""
    scores = scores.clone()
    scores[train > 0] = float('-inf')
    topk = scores.topk(k, dim=1).indices
    target_bool = target > 0
    hits = torch.gather(target_bool, 1, topk).sum(dim=1).float()
    total = target_bool.sum(dim=1).float().clamp(min=1, max=k)
    return (hits / total).mean().item()


def ndcg_at_k(scores, train, target, k):
    scores = scores.clone()
    scores[train > 0] = float('-inf')
    topk = scores.topk(k, dim=1).indices
    topk_relevance = torch.gather(target, 1, topk).float()

    discounts = torch.log2(torch.arange(2, k + 2, device=scores.device).float())
    dcg = (topk_relevance / discounts).sum(dim=1)

    ideal_sorted = target.float().topk(k, dim=1).values
    idcg = (ideal_sorted / discounts).sum(dim=1)
    return (dcg / idcg.clamp(min=1e-8)).mean().item()


class MetricAccumulator:
    """Averages NDCG@100, Recall@50 and Recall@20 over users, one epoch at a time."""

    def __init__(self):
        self.history = []
        self.reset()

    def reset(self):
        self.total_ndcg_100 = 0.0
        self.total_recall_50 = 0.0
        self.total_recall_20 = 0.0
        self.count = 0

    def update(self, scores, train, target):
        batch_size = scores.size(0)
        self.total_ndcg_100 += ndcg_at_k(scores, train, target, 100) * batch_size
        self.total_recall_50 += recall_at_k(scores, train, target, 50) * batch_size
        self.total_recall_20 += recall_at_k(scores, train, target, 20) * batch_size
        self.count += batch_size

    def end_epoch(self):
        metrics = {
            'ndcg@100': self.total_ndcg_100 / self.count,
            'recall@50': self.total_recall_50 / self.count,
            'recall@20': self.total_recall_20 / self.count,
        }
        self.history.append(metrics)
        self.reset()
        return metrics

    def print_best(self):
        best = max(self.history, key=lambda m: m['ndcg@100'])
        best_epoch = self.history.index(best) + 1
        print(f"Best Epoch {best_epoch}: NDCG@100={best['ndcg@100']:.4f} | "
              f"Recall@50={best['recall@50']:.4f} | Recall@20={best['recall@20']:.4f}")


class LossTracker:
    def __init__(self):
        self.history = []
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.count = 0

    def update(self, loss, batch_size):
        self.total_loss += loss * batch_size
        self.count += batch_size

    def end_epoch(self):
        avg_loss = self.total_loss / self.count
        self.history.append(avg_loss)
        self.reset()
        return avg_loss

    def print_history(self):
        for i, loss in enumerate(self.history):
            print(f"Epoch {i + 1}: Loss={loss:.4f}")


def append_results_csv(csv_path, model_type, split, metrics):
    """Append one row of metrics to `csv_path`, writing a header if the file is new."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model', 'split', 'ndcg@100', 'recall@50', 'recall@20'])
        writer.writerow([
            model_type, split,
            f"{metrics['ndcg@100']:.6f}",
            f"{metrics['recall@50']:.6f}",
            f"{metrics['recall@20']:.6f}",
        ])
