"""Autoencoder baselines for collaborative filtering on H&M, ML-20M and MSD.

Model types
    linear, deep                    denoising autoencoders trained with gradient descent
    mult_vae, macrid_vae            VAE-based recommenders
    ease                            EASE (closed form)
    edlae_full, edlae_lowrank       EDLAE (closed form / alternating least squares)

Examples (from the repository root)
    python autoencoder_training/autoencoders.py --dataset ml20m --model-type ease
    python autoencoder_training/autoencoders.py --dataset msd --model-type mult_vae --tune --tune-trials 20
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (LossTracker, MetricAccumulator, TrainDataset, ValidTestDataset,
                   append_results_csv, read_data, set_seed)

MODEL_TYPES = ['linear', 'deep', 'mult_vae', 'macrid_vae', 'ease', 'edlae_full', 'edlae_lowrank']
ANALYTIC_MODELS = {'ease', 'edlae_full', 'edlae_lowrank'}
VAE_MODELS = {'mult_vae', 'macrid_vae'}


# ============================================================================
# Per-dataset settings
# ============================================================================

LRS = [3e-3, 1e-3, 5e-4, 3e-4, 1e-4]
WDS = [1e-2, 1e-4, 1e-6]
BATCH_SIZES = [128, 256, 512]
BETAS = [0.05, 0.1, 0.2, 0.3, 0.5]
EASE_REGS = [1, 5, 10, 25, 50, 100, 200, 250, 500, 750,
             1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

# Optuna search space shared by all datasets. The order of the keys is the
# order in which parameters are suggested to Optuna.
SEARCH_SPACE = {
    'edlae_lowrank': {'pdrop': [0.0, 0.25, 0.5, 0.67, 0.75, 0.8],
                      'L2': [1, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000, 30000],
                      'zero_diag': [True, False],
                      'hidden_dim': [1024]},
    'linear': {'lr': LRS, 'wd': WDS, 'batch_size': BATCH_SIZES,
               'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
               'hidden_dim': [1024]},
    'deep': {'lr': LRS, 'wd': WDS, 'batch_size': BATCH_SIZES,
             'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
             'hidden_dim': [1536, 2048, 3072], 'latent_dim': [1024]},
    'mult_vae': {'lr': LRS, 'wd': WDS, 'batch_size': BATCH_SIZES,
                 'dropout': [0.3, 0.5, 0.7],
                 'hidden_dim': [1536, 2048, 3072], 'latent_dim': [1024], 'beta': BETAS},
    'macrid_vae': {'lr': LRS, 'wd': WDS, 'batch_size': BATCH_SIZES,
                   'dropout': [0.3, 0.5, 0.7],
                   'hidden_dim': [1536, 2048, 3072], 'latent_dim': [1024], 'beta': BETAS,
                   'n_concepts': [3, 5, 7, 10], 'tau_concept': [0.05, 0.1, 0.5, 1.0]},
}

# `defaults` replace the command-line defaults for that dataset (they can still
# be overridden on the command line); `search_space` replaces entries of
# SEARCH_SPACE for that dataset.
DATASETS = {
    'hm': {
        'defaults': {'data_path': 'data/hm', 'logdir': 'checkpoints/hm',
                     'hidden_dim': 1536, 'lowrank_dtype': 'float64'},
        'search_space': {
            'edlae_lowrank': {'L2': [1, 5, 10, 20, 50, 100, 200, 500, 1000]},
            'deep': {'hidden_dim': [1536]},
            'mult_vae': {'hidden_dim': [512, 1024, 1536]},
            'macrid_vae': {'lr': [1e-3, 5e-4, 3e-4, 1e-4], 'hidden_dim': [1024, 1536]},
        },
    },
    'ml20m': {
        'defaults': {'data_path': 'data/ml-20m', 'logdir': 'checkpoints/ml20m',
                     'hidden_dim': 2048, 'lowrank_dtype': 'float32'},
        'search_space': {},
    },
    'msd': {
        'defaults': {'data_path': 'data/msd', 'logdir': 'checkpoints/msd',
                     'hidden_dim': 2048, 'lowrank_dtype': 'float32'},
        'search_space': {
            'edlae_lowrank': {'hidden_dim': [4096]},
            'linear': {'hidden_dim': [4096]},
            'deep': {'hidden_dim': [2048, 3072]},
            'mult_vae': {'hidden_dim': [2048]},
            'macrid_vae': {'lr': [1e-3, 5e-4, 3e-4, 1e-4], 'wd': [1e-4, 1e-6],
                           'hidden_dim': [2048, 3072]},
        },
    },
}


def search_space(dataset, model_type):
    return {**SEARCH_SPACE[model_type], **DATASETS[dataset]['search_space'].get(model_type, {})}


# ============================================================================
# Arguments
# ============================================================================

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    bool_flag = argparse.BooleanOptionalAction

    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    p.add_argument('--model-type', default='edlae_lowrank', choices=MODEL_TYPES)
    p.add_argument('--data-path', help='Folder with the preprocessed splits (default: per dataset)')
    p.add_argument('--logdir', help='Where enc/dec weights are saved (default: checkpoints/<dataset>)')
    p.add_argument('--results-csv', default=None,
                   help='CSV file to append valid/test metrics to')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=6)

    g = p.add_argument_group('gradient-trained models')
    g.add_argument('--epochs', type=int, default=50)
    g.add_argument('--batch-size', type=int, default=128)
    g.add_argument('--lr', type=float, default=3e-4)
    g.add_argument('--wd', type=float, default=1e-4)
    g.add_argument('--dropout-rate', type=float, default=0.5)
    g.add_argument('--hidden-dim', type=int, help='Hidden layer width (default: per dataset)')
    g.add_argument('--latent-dim', type=int, default=1024,
                   help='Latent size; also the rank of low-rank analytic models')
    g.add_argument('--bias', default=False, action='store_true')
    g.add_argument('--normalize', default=True, action=bool_flag,
                   help='L2-normalise the latent code of the linear/deep autoencoders')

    g = p.add_argument_group('VAE models')
    g.add_argument('--beta', type=float, default=0.2, help='KL weight')
    g.add_argument('--anneal-steps', type=int, default=0,
                   help='Linearly anneal beta from 0 over this many steps (0 = off)')
    g.add_argument('--n-concepts', type=int, default=7, help='MacridVAE macro concepts')
    g.add_argument('--tau-concept', type=float, default=1.0,
                   help='MacridVAE concept temperature')

    g = p.add_argument_group('analytic models')
    g.add_argument('--edlae-l2', type=float, default=50.0, help='L2 regularisation for EDLAE')
    g.add_argument('--edlae-pdrop', type=float, default=0.7, help='EDLAE dropout probability')
    g.add_argument('--edlae-iters', type=int, default=20,
                   help='Alternating iterations for low-rank EDLAE')
    g.add_argument('--edlae-tau', type=float, default=0.003,
                   help='Logit temperature of analytic models (does not change rankings)')
    g.add_argument('--edlae-zero-diag', default=True, action=bool_flag,
                   help='Zero-diagonal constraint in low-rank EDLAE')
    g.add_argument('--lowrank-dtype', choices=['float32', 'float64'],
                   help='Precision of low-rank EDLAE (default: per dataset)')
    g.add_argument('--ease-reg', type=float, default=2000.0, help='EASE L2 regularisation')
    g.add_argument('--ease-svd-rank', type=int, default=0,
                   help='If > 0, factorise the EASE matrix to this rank via SVD')

    g = p.add_argument_group('hyperparameter tuning')
    g.add_argument('--tune', default=False, action='store_true',
                   help='Tune on validation (Optuna; grid search for EASE), then retrain')
    g.add_argument('--tune-trials', type=int, default=20)
    return p


def parse_args(argv=None):
    parser = build_parser()
    known, _ = parser.parse_known_args(argv)
    parser.set_defaults(**DATASETS[known.dataset]['defaults'])
    return parser.parse_args(argv)


# ============================================================================
# Losses
# ============================================================================

def multinomial_loss(logits, target):
    log_probs = F.log_softmax(logits, dim=1)
    return -torch.sum(log_probs * target, dim=1).mean()


def kl_standard_normal(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def vae_loss(logits, x, mu, logvar, beta):
    return multinomial_loss(logits, x) + beta * kl_standard_normal(mu, logvar)


# ============================================================================
# Gradient-trained models
# ============================================================================

class LinearAutoEncoder(nn.Module):
    def __init__(self, num_items, hidden_dim, drop_rate=0.5, bias=False, normalize=True):
        super().__init__()
        self.drop_rate = drop_rate
        self.normalize = normalize
        self.encoder = nn.Linear(num_items, hidden_dim, bias=bias)
        self.decoder = nn.Linear(hidden_dim, num_items, bias=bias)

    def forward(self, x):
        if self.training:
            x = x * torch.bernoulli(torch.full_like(x, 1 - self.drop_rate))
        h = self.encoder(x)
        if self.normalize:
            h = F.normalize(h, dim=-1)
        return self.decoder(h)


class AutoEncoder(nn.Module):
    def __init__(self, num_items, hidden_dim, latent_dim, drop_rate=0.5, bias=False, normalize=True):
        super().__init__()
        self.drop_rate = drop_rate
        self.normalize = normalize
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim, bias=bias),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim, bias=bias),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=bias),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items, bias=bias),
        )

    def forward(self, x):
        if self.training:
            x = x * torch.bernoulli(torch.full_like(x, 1 - self.drop_rate))
        h = self.encoder(x)
        if self.normalize:
            h = F.normalize(h, dim=-1)
        return self.decoder(h)


def reparameterize(mu, logvar, training):
    if training:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    return mu


class MultVAE(nn.Module):
    """Mult-VAE (Liang et al., 2018)."""

    def __init__(self, num_items, hidden_dim, latent_dim, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate
        self.encoder = nn.Sequential(nn.Linear(num_items, hidden_dim), nn.Tanh())
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items),
        )

    def forward(self, x):
        if self.training:
            x = F.dropout(x, p=self.drop_rate, training=True)
        h = self.encoder(x)
        mu, logvar = self.mu_layer(h), self.logvar_layer(h)
        z = reparameterize(mu, logvar, self.training)
        return self.decoder(z), mu, logvar


class MacridVAE(nn.Module):
    """MacridVAE (Ma et al., 2019): items are softly clustered into K macro
    concepts; each concept has its own latent code and the decoder mixes the
    per-concept item scores."""

    def __init__(self, num_items, hidden_dim, latent_dim, drop_rate=0.5,
                 n_concepts=7, tau_concept=1.0):
        super().__init__()
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.n_concepts = n_concepts
        self.tau = tau_concept

        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.items = nn.Parameter(torch.empty(num_items, latent_dim))
        self.cores = nn.Parameter(torch.empty(n_concepts, latent_dim))
        nn.init.xavier_normal_(self.items)
        nn.init.xavier_normal_(self.cores)
        self.drop = nn.Dropout(drop_rate)

    def cluster(self):
        """Soft assignment of items to concepts: (M, D) items, (M, K) assignments."""
        items = F.normalize(self.items, dim=1)
        cores = F.normalize(self.cores, dim=1)
        cates = F.softmax(items @ cores.t() / self.tau, dim=1)
        return items, cates

    def encode(self, x, cates):
        n, K, M, D = x.shape[0], self.n_concepts, self.num_items, self.latent_dim
        x = self.drop(x)
        x_k = (x.unsqueeze(1) * cates.t().unsqueeze(0)).reshape(n * K, M)  # one input per concept
        h = self.encoder(x_k)
        return h[:, :D], h[:, D:]

    def decode(self, z, items, cates):
        K, M = self.n_concepts, self.num_items
        n = z.shape[0] // K
        probs = torch.exp(torch.mm(z, items.t()) / self.tau)            # (n*K, M)
        probs = (probs.view(n, K, M) * cates.t().unsqueeze(0)).sum(dim=1)  # (n, M)
        return torch.log(probs + 1e-10)

    def forward(self, x):
        items, cates = self.cluster()
        mu, logvar = self.encode(x, cates)
        z = reparameterize(mu, logvar, self.training)
        return self.decode(z, items, cates), mu, logvar


def predict(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def batch_loss(model, model_type, x, beta):
    if model_type in VAE_MODELS:
        logits, mu, logvar = model(x)
        return vae_loss(logits, x, mu, logvar, beta)
    return multinomial_loss(model(x), x)


@torch.no_grad()
def evaluate(model, loader, device, tracker=None, desc=None):
    tracker = tracker if tracker is not None else MetricAccumulator()
    model.eval()
    for x, y in (tqdm(loader, desc=desc) if desc else loader):
        x, y = x.to(device), y.to(device)
        tracker.update(predict(model, x), x, y)
    return tracker.end_epoch()


# ============================================================================
# Analytic models
# ============================================================================

def _set_diag(M, vals):
    idx = torch.arange(M.shape[0], device=M.device)
    M[idx, idx] = vals
    return M


def _get_diag(M):
    return torch.diagonal(M)


def _safe_inv(M, eps=1e-4):
    """Regularised inverse, falling back to the pseudo-inverse."""
    try:
        return torch.linalg.inv(M + eps * torch.eye(M.shape[0], device=M.device, dtype=M.dtype))
    except torch.linalg.LinAlgError:
        return torch.linalg.pinv(M)


def load_or_compute_gram(config, data, device):
    """X^T X of the training matrix. Uses <data_path>/gram.pt if it exists;
    otherwise accumulates it in batches without materialising X."""
    gram_path = os.path.join(config.data_path, 'gram.pt')
    if os.path.isfile(gram_path):
        print(f"Loading precomputed Gram matrix from {gram_path} ...")
        gram = torch.load(gram_path, map_location=device, weights_only=True)
        return gram['XtX'], gram['XtXdiag']

    n_items = data['n_items']
    loader = DataLoader(TrainDataset(data['X_tr'], n_items), batch_size=2048,
                        num_workers=config.num_workers, shuffle=False)
    print("Computing X^T X ...")
    XtX = torch.zeros(n_items, n_items, device=device)
    for x in loader:
        x = x.to(device)
        XtX += x.T @ x
    return XtX, _get_diag(XtX).clone()


class EASE:
    """EASE (Steck, 2019): B = -P / diag(P), diag(B) = 0, P = (X^T X + lambda I)^-1."""

    def __init__(self, reg=2000.0):
        self.reg = reg
        self.BB = None
        self.UU = None
        self.VV = None
        self.tau = 1.0

    @torch.no_grad()
    def fit(self, XtX, XtXdiag):
        XtX = XtX.clone()
        n = XtX.shape[0]
        _set_diag(XtX, XtXdiag + self.reg)
        P = torch.linalg.solve(XtX, torch.eye(n, device=XtX.device, dtype=XtX.dtype))
        BB = -P / torch.diagonal(P).unsqueeze(0)
        _set_diag(BB, 0.0)
        self.BB = BB
        print(f"  EASE fit complete. B stats: mean={BB.mean().item():.6f}, "
              f"std={BB.std().item():.6f}, max={BB.abs().max().item():.6f}")
        return BB

    def factorize(self, rank):
        """Low-rank approximation of B via truncated SVD."""
        U, S, Vt = torch.linalg.svd(self.BB, full_matrices=False)
        self.UU = U[:, :rank] @ torch.diag(S[:rank])
        self.VV = Vt[:rank, :].T
        err = (self.BB - self.UU @ self.VV.T).norm() / self.BB.norm()
        print(f"  EASE factorized to rank {rank}. Relative error: {err.item():.6f}")
        return self.UU, self.VV

    @torch.no_grad()
    def predict_batch(self, x):
        if self.UU is not None:
            return F.normalize(x @ self.UU, dim=-1) @ self.VV.T / self.tau
        return x @ self.BB / self.tau


class EDLAE_FullRank:
    def __init__(self, pdrop=0.5, L2=200.0):
        self.pdrop = pdrop
        self.L2 = L2
        self.BB = None
        self.tau = 1.0

    def fit(self, XtX, XtXdiag):
        XtX = XtX.clone()
        _set_diag(XtX, XtXdiag + self.L2 + XtXdiag * self.pdrop / (1.0 - self.pdrop))
        BB = torch.linalg.inv(XtX)
        BB /= -_get_diag(BB)
        _set_diag(BB, 0.0)
        self.BB = BB
        return BB

    @torch.no_grad()
    def predict_batch(self, x):
        return x @ self.BB / self.tau


class EDLAE_LowRank:
    def __init__(self, pdrop=0.0, L2=200.0, hidden_dim=1000, n_iters=10,
                 init_scale=1e-4, zero_diag=True, dtype='float32'):
        self.pdrop = pdrop
        self.L2 = L2
        self.hidden_dim = hidden_dim
        self.n_iters = n_iters
        self.init_scale = init_scale
        self.zero_diag = zero_diag
        self.dtype = torch.float64 if dtype == 'float64' else torch.float32
        self.UU = None
        self.VV = None
        self.tau = 1.0

    def fit(self, XtX, XtXdiag):
        XtX = XtX.clone().to(self.dtype)
        XtXdiag = XtXdiag.to(self.dtype)
        n = XtX.shape[0]

        dropout_reg = XtXdiag * self.pdrop / (1.0 - self.pdrop)
        reg = XtXdiag + self.L2 + dropout_reg
        _set_diag(XtX, reg)
        CC = torch.linalg.inv(XtX)

        VVt = torch.randn(self.hidden_dim, n, device=XtX.device).to(self.dtype) * self.init_scale

        for it in range(self.n_iters):
            # Update UU
            GG = _safe_inv(VVt @ VVt.T)
            if self.zero_diag:
                KK = VVt.T @ GG @ VVt
                gammaVec = torch.linalg.solve(CC * KK, _get_diag(KK))
            else:
                gammaVec = self.L2 + dropout_reg
            HH = CC * (-gammaVec)
            _set_diag(HH, _get_diag(HH) + 1.0)
            UU = HH @ (VVt.T @ GG)

            # Update VVt
            _set_diag(XtX, reg)
            HH2 = _safe_inv(UU.T @ XtX @ UU) @ UU.T
            _set_diag(XtX, XtXdiag)
            VVt = HH2 @ XtX
            if self.zero_diag:
                etaVec = _get_diag(UU @ VVt) / _get_diag(UU @ HH2)
                VVt -= HH2 * etaVec

            if (it + 1) % 5 == 0 or it == 0:
                print(f"  Iter {it + 1}/{self.n_iters}  "
                      f"|diag(B)|_max={_get_diag(UU @ VVt).abs().max().item():.6f}")

        self.UU = UU.float()
        self.VV = VVt.T.float()
        return self.UU, self.VV

    @torch.no_grad()
    def predict_batch(self, x):
        return F.normalize(x @ self.UU, dim=-1) @ self.VV.T / self.tau


def report_logit_stats(logits, label=""):
    """Print how peaked the softmax over items is (diagnostic only)."""
    sm = F.softmax(logits, dim=1)
    n_items = logits.shape[1]
    norm_entropy = (-(sm * sm.clamp(min=1e-12).log()).sum(dim=1) / np.log(n_items)).mean().item()
    top10 = sm.topk(min(10, n_items), dim=1).values.sum(dim=1).mean().item()
    top50 = sm.topk(min(50, n_items), dim=1).values.sum(dim=1).mean().item()
    std_val = logits.std().item()

    print(f"\n  === Logit Diagnostics [{label}] ===")
    print(f"  mean={logits.mean().item():.4f}  std={std_val:.4f}  "
          f"min={logits.min().item():.4f}  max={logits.max().item():.4f}")
    print(f"  norm_entropy={norm_entropy:.4f}  top10_mass={top10:.4f}  top50_mass={top50:.4f}")
    if std_val < 0.01 or norm_entropy > 0.99:
        verdict = "TOO FLAT"
    elif std_val > 50 or norm_entropy < 0.3:
        verdict = "TOO PEAKED"
    else:
        verdict = "OK"
    print(f"  VERDICT: {verdict}\n")


@torch.no_grad()
def evaluate_analytic(model, loader, device, tracker=None, report_logits=False, desc=None):
    tracker = tracker if tracker is not None else MetricAccumulator()
    first_batch = True
    for x, y in (tqdm(loader, desc=desc) if desc else loader):
        x, y = x.to(device), y.to(device)
        scores = model.predict_batch(x)
        if first_batch and report_logits:
            report_logit_stats(scores, label=f"tau={model.tau}")
        first_batch = False
        tracker.update(scores, x, y)
    return tracker.end_epoch()


def sweep_tau(model, valid_loader, device, n_candidates=20):
    """Set model.tau to the value that minimises validation cross-entropy."""
    model.tau = 1.0
    x, _ = next(iter(valid_loader))
    raw_std = model.predict_batch(x.to(device)).std().item()
    taus = np.geomspace(max(raw_std / 50.0, 1e-10), max(raw_std / 0.5, 1e-6), n_candidates)

    best_tau, best_loss = 1.0, float('inf')
    for tau in taus.tolist():
        model.tau = tau
        total_loss, total_count = 0.0, 0
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            total_loss += multinomial_loss(model.predict_batch(x), y).item() * x.shape[0]
            total_count += x.shape[0]
        if total_loss / total_count < best_loss:
            best_loss, best_tau = total_loss / total_count, tau
    model.tau = best_tau
    return best_tau


def build_analytic_model(config):
    t = config.model_type
    if t == 'edlae_full':
        print(f"\nFitting full-rank EDLAE (pdrop={config.edlae_pdrop}, L2={config.edlae_l2}) ...")
        return EDLAE_FullRank(pdrop=config.edlae_pdrop, L2=config.edlae_l2)
    if t == 'edlae_lowrank':
        print(f"\nFitting low-rank EDLAE (pdrop={config.edlae_pdrop}, L2={config.edlae_l2}, "
              f"dim={config.latent_dim}, iters={config.edlae_iters}, "
              f"zero_diag={config.edlae_zero_diag}) ...")
        return EDLAE_LowRank(pdrop=config.edlae_pdrop, L2=config.edlae_l2,
                             hidden_dim=config.latent_dim, n_iters=config.edlae_iters,
                             zero_diag=config.edlae_zero_diag, dtype=config.lowrank_dtype)
    print(f"\nFitting EASE (reg={config.ease_reg}) ...")
    return EASE(reg=config.ease_reg)


def save_enc_dec_pt(model, model_type, logdir):
    """Save encoder/decoder weights as enc_<model>.pt and dec_<model>.pt."""
    os.makedirs(logdir, exist_ok=True)
    enc_path = os.path.join(logdir, f'enc_{model_type}.pt')
    dec_path = os.path.join(logdir, f'dec_{model_type}.pt')

    if isinstance(model, (LinearAutoEncoder, AutoEncoder)):
        torch.save(model.encoder, enc_path)
        torch.save(model.decoder, dec_path)
    elif isinstance(model, MultVAE):
        torch.save({'encoder': model.encoder.state_dict(),
                    'mu_layer': model.mu_layer.state_dict(),
                    'logvar_layer': model.logvar_layer.state_dict()}, enc_path)
        torch.save(model.decoder.state_dict(), dec_path)
    elif isinstance(model, MacridVAE):
        # MacridVAE decodes through its item embeddings; there is no separate decoder module.
        torch.save({'encoder': model.encoder.state_dict(),
                    'items': model.items.data, 'cores': model.cores.data}, enc_path)
        torch.save({'items': model.items.data, 'cores': model.cores.data}, dec_path)
    elif getattr(model, 'UU', None) is not None:
        enc = nn.Linear(model.UU.shape[0], model.UU.shape[1], bias=False)
        enc.weight = nn.Parameter(model.UU.T)   # nn.Linear stores (out, in)
        dec = nn.Linear(model.VV.shape[1], model.VV.shape[0], bias=False)
        dec.weight = nn.Parameter(model.VV)     # (n_items, hidden)
        torch.save(enc, enc_path)
        torch.save(dec, dec_path)
    elif getattr(model, 'BB', None) is not None:
        # Full-rank item-item matrix: stored in both files for a uniform interface.
        torch.save(model.BB, enc_path)
        torch.save(model.BB, dec_path)
        print("  Note: full-rank model saved as the B matrix in both enc and dec files")
    else:
        print("  Warning: no weights to save.")
        return
    print(f"  Saved: {enc_path} and {dec_path}")


# ============================================================================
# Hyperparameter tuning
# ============================================================================

# Tuned parameter name -> config attribute used for the final retrain.
TUNED_PARAM_TO_CONFIG = {
    'pdrop': 'edlae_pdrop', 'L2': 'edlae_l2',
    'hidden_dim': 'hidden_dim', 'latent_dim': 'latent_dim',
    'n_iters': 'edlae_iters', 'lr': 'lr', 'wd': 'wd',
    'dropout': 'dropout_rate', 'epochs': 'epochs',
    'beta': 'beta', 'n_concepts': 'n_concepts',
    'ease_reg': 'ease_reg', 'tau_concept': 'tau_concept',
    'batch_size': 'batch_size', 'zero_diag': 'edlae_zero_diag',
}


def suggest_analytic_model(trial, config, space):
    t = config.model_type
    if t == 'edlae_full':
        pdrop = trial.suggest_float('pdrop', 0.0, 0.8, step=0.05)
        L2 = trial.suggest_float('L2', 1.0, 1e5, log=True)
        return EDLAE_FullRank(pdrop=pdrop, L2=L2)
    # edlae_lowrank
    p = {name: trial.suggest_categorical(name, choices) for name, choices in space.items()}
    n_iters = trial.suggest_int('n_iters', 20, 20)
    return EDLAE_LowRank(pdrop=p['pdrop'], L2=p['L2'], hidden_dim=p['hidden_dim'],
                         n_iters=n_iters, zero_diag=p['zero_diag'], dtype=config.lowrank_dtype)


def build_tuning_model(config, n_items, p):
    t = config.model_type
    if t == 'linear':
        return LinearAutoEncoder(n_items, p['hidden_dim'], p['dropout'], bias=False,
                                 normalize=config.normalize)
    if t == 'deep':
        return AutoEncoder(n_items, p['hidden_dim'], p['latent_dim'], p['dropout'], bias=False,
                           normalize=config.normalize)
    if t == 'mult_vae':
        return MultVAE(n_items, p['hidden_dim'], p['latent_dim'], p['dropout'])
    return MacridVAE(n_items, p['hidden_dim'], p['latent_dim'], p['dropout'],
                     n_concepts=p['n_concepts'], tau_concept=p['tau_concept'])


def run_optuna(config, data):
    """Tune config.model_type on the validation set; returns the Optuna study."""
    try:
        import optuna
    except ImportError:
        raise RuntimeError("Install optuna for tuning: pip install optuna")

    device = config.device
    n_items = data['n_items']
    space = search_space(config.dataset, config.model_type) if config.model_type in SEARCH_SPACE else {}
    valid_loader = DataLoader(ValidTestDataset(data['X_val'], data['Y_val'], n_items),
                              batch_size=config.batch_size, num_workers=config.num_workers,
                              shuffle=False)
    analytic = config.model_type in ANALYTIC_MODELS
    if analytic:
        XtX, XtXdiag = load_or_compute_gram(config, data, device)

    def objective(trial):
        if analytic:
            model = suggest_analytic_model(trial, config, space)
            model.fit(XtX.clone(), XtXdiag.clone())
            sweep_tau(model, valid_loader, device, n_candidates=20)
            return evaluate_analytic(model, valid_loader, device)['ndcg@100']

        p = {name: trial.suggest_categorical(name, choices) for name, choices in space.items()}
        epochs = trial.suggest_int('epochs', 20, 20)
        net = build_tuning_model(config, n_items, p).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=p['lr'], weight_decay=p['wd'])
        train_loader = DataLoader(TrainDataset(data['X_tr'], n_items), batch_size=p['batch_size'],
                                  num_workers=config.num_workers, shuffle=True)
        for _ in range(epochs):
            net.train()
            for x in train_loader:
                x = x.to(device)
                loss = batch_loss(net, config.model_type, x, p.get('beta'))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return evaluate(net, valid_loader, device)['ndcg@100']

    study = optuna.create_study(direction='maximize', study_name=f'tune_{config.model_type}',
                                sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=config.tune_trials, show_progress_bar=True)
    return study


def tune_ease(config, data):
    """Grid search over EASE_REGS on the validation set; returns the best reg."""
    XtX, XtXdiag = load_or_compute_gram(config, data, config.device)
    valid_loader = DataLoader(ValidTestDataset(data['X_val'], data['Y_val'], data['n_items']),
                              batch_size=config.batch_size, num_workers=config.num_workers,
                              shuffle=False)
    print(f"\n{'=' * 60}\n  EASE GRID SEARCH - {len(EASE_REGS)} reg values\n{'=' * 60}")
    best_reg, best_ndcg = None, -1.0
    for reg in EASE_REGS:
        model = EASE(reg=reg)
        model.fit(XtX.clone(), XtXdiag.clone())
        metrics = evaluate_analytic(model, valid_loader, config.device)
        tag = ""
        if metrics['ndcg@100'] > best_ndcg:
            best_ndcg, best_reg, tag = metrics['ndcg@100'], reg, " <-- BEST"
        print(f"  reg={reg:>5d}  NDCG@100={metrics['ndcg@100']:.6f}  "
              f"R@50={metrics['recall@50']:.6f}  R@20={metrics['recall@20']:.6f}{tag}")
    print(f"\n{'=' * 60}\n  BEST: reg={best_reg}  NDCG@100={best_ndcg:.6f}\n{'=' * 60}\n")
    return best_reg


def tune(config, data):
    """Tune on validation and write the best hyperparameters into `config`."""
    if config.model_type == 'ease':
        config.ease_reg = tune_ease(config, data)
        return

    study = run_optuna(config, data)
    print(f"\n{'=' * 60}")
    print("  TUNING COMPLETE - retraining with the best parameters")
    print(f"  Best NDCG@100 (val): {study.best_value:.5f}")
    print(f"  Best params: {study.best_params}")
    print(f"{'=' * 60}\n")
    for name, value in study.best_params.items():
        attr = TUNED_PARAM_TO_CONFIG.get(name, name.replace('-', '_'))
        if hasattr(config, attr):
            setattr(config, attr, value)
            print(f"  config.{attr} = {value}")


# ============================================================================
# Training / evaluation
# ============================================================================

def run_analytic(config, data):
    device = config.device
    XtX, XtXdiag = load_or_compute_gram(config, data, device)

    model = build_analytic_model(config)
    model.fit(XtX, XtXdiag)
    if config.model_type == 'ease' and config.ease_svd_rank > 0:
        print(f"\nFactorizing B to rank {config.ease_svd_rank} via SVD ...")
        model.factorize(rank=config.ease_svd_rank)
    model.tau = config.edlae_tau

    metrics = {}
    for split, X, Y in (('valid', data['X_val'], data['Y_val']),
                        ('test', data['X_test'], data['Y_test'])):
        loader = DataLoader(ValidTestDataset(X, Y, data['n_items']), batch_size=config.batch_size,
                            num_workers=config.num_workers, shuffle=False)
        print(f"\n--- {split.capitalize()} ---")
        metrics[split] = evaluate_analytic(model, loader, device, report_logits=True,
                                           desc="Evaluating")
        print(metrics[split])
    return model, metrics


def annealed_beta(config, step):
    if config.anneal_steps > 0:
        return min(config.beta, config.beta * step / config.anneal_steps)
    return config.beta


def build_model(config, n_items):
    t = config.model_type
    if t == 'linear':
        return LinearAutoEncoder(n_items, config.latent_dim, config.dropout_rate,
                                 config.bias, config.normalize)
    if t == 'mult_vae':
        return MultVAE(n_items, config.hidden_dim, config.latent_dim, config.dropout_rate)
    if t == 'macrid_vae':
        return MacridVAE(n_items, config.hidden_dim, config.latent_dim, config.dropout_rate,
                         n_concepts=config.n_concepts, tau_concept=config.tau_concept)
    return AutoEncoder(n_items, config.hidden_dim, config.latent_dim, config.dropout_rate,
                       config.bias, config.normalize)


def run_gradient(config, data):
    device = config.device
    n_items = data['n_items']
    model = build_model(config, n_items).to(device)

    def loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=config.batch_size,
                          num_workers=config.num_workers, shuffle=shuffle)

    train_loader = loader(TrainDataset(data['X_tr'], n_items), shuffle=True)
    valid_loader = loader(ValidTestDataset(data['X_val'], data['Y_val'], n_items), shuffle=False)
    test_loader = loader(ValidTestDataset(data['X_test'], data['Y_test'], n_items), shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), config.lr, weight_decay=config.wd)
    loss_tracker = LossTracker()
    valid_tracker = MetricAccumulator()
    best_ndcg, best_state, best_epoch = -1.0, None, -1
    step = 0

    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for x in pbar:
            x = x.to(device)
            beta = annealed_beta(config, step)
            loss = batch_loss(model, config.model_type, x, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tracker.update(loss.item(), x.shape[0])
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {epoch + 1}: loss={loss_tracker.end_epoch():.4f}")

        metrics = evaluate(model, valid_loader, device, valid_tracker, desc=f"Epoch {epoch + 1} [Val]")
        print(metrics)
        if metrics['ndcg@100'] > best_ndcg:
            best_ndcg, best_epoch = metrics['ndcg@100'], epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            print(f"  >> New best model at epoch {best_epoch} (NDCG@100={best_ndcg:.4f})")

    loss_tracker.print_history()
    valid_tracker.print_best()

    print(f"\nLoading best model from epoch {best_epoch} (NDCG@100={best_ndcg:.4f})")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, desc="Test")
    print(test_metrics)
    return model, {'valid': valid_tracker.history[best_epoch - 1], 'test': test_metrics}


def main(argv=None):
    config = parse_args(argv)
    print(config)
    set_seed(config.seed)
    data = read_data(config.data_path)

    if config.tune:
        tune(config, data)
        print("\n  Proceeding with retrain...\n")

    if config.model_type in ANALYTIC_MODELS:
        model, metrics = run_analytic(config, data)
    else:
        model, metrics = run_gradient(config, data)

    if config.results_csv:
        for split in ('valid', 'test'):
            append_results_csv(config.results_csv, config.model_type, split, metrics[split])
        print(f"  Results appended to {config.results_csv}")
    save_enc_dec_pt(model, config.model_type, config.logdir)


if __name__ == '__main__':
    main()
