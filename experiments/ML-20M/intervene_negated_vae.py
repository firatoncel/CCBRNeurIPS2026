"""
Evaluate LOO-Negated genre interventions on movie summary-based recommendations.

LOO-Negated: leave-one-out via explicit negation. Instead of silently dropping
a genre from the summary, the edited summary explicitly states that the user
"actively avoids" that genre. Expected effect on the target genre is NEGATIVE.

For each edited summary, counts how many of the top-k recommended items
(k = 5, 10, 20, 50) carry the target genre, compared with the original summary.

Input JSON structure (leave_one_out_negated.json):
  {user_id: {original_summary, original_genre_indices, removals: [
      {removed_genre, removed_genre_idx, edited_summary, ...}, ...]}}

Usage:
    python experiments/ML-20M/intervene_negated_vae.py --exp-dir results_movie/<run> \
        --interventions data/ml-20m/leave_one_out_negated.json
"""

import os
import csv
import json
import sys
import argparse
import numpy as np
from collections import defaultdict


class Tee:
    """Duplicate stdout to both console and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        sys.stdout = self.stdout
        self.file.close()


import torch
from tqdm import tqdm


from model_ddp_vae import LatentAlignModel
from utils import set_seed

# Genre definitions

GENRE2IDX = {
    "Action": 0, "Adventure": 1, "Animation": 2, "Children": 3,
    "Comedy": 4, "Crime": 5, "Documentary": 6, "Drama": 7,
    "Fantasy": 8, "Film-Noir": 9, "Horror": 10, "Musical": 11,
    "Mystery": 12, "Romance": 13, "Sci-Fi": 14, "Thriller": 15,
    "War": 16, "Western": 17,
}
GENRE_LIST = sorted(GENRE2IDX.keys(), key=lambda x: GENRE2IDX[x])
N_GENRES = len(GENRE_LIST)

CTRL_KS = [5, 10, 20, 50]

# LOO-Negated config (fixed, no need for a getter)
INTERVENTION_TYPE = "loo_negated"
LABEL = "Negated Removal"
LIST_KEY = "removals"
ATTR_IDX_KEY = "removed_genre_idx"
EXPECTED_SIGN = "NEGATIVE"
# Users need at least 2 genres in the original summary (same rule as removal)
EDIT_DENOM_OFFSET = -1


# ============================================================================
# Controllability eval, LOO-Negated, single-pass decode
# ============================================================================

def run_controllability_eval(interventions, model, genre_matrix):
    """
    Decode each summary ONCE with max(CTRL_KS), cache full ranking,
    then slice for all smaller k values.
    """
    max_k = max(CTRL_KS)

    user_ids = list(interventions.keys())

    # Collect all (user, intervention_item) pairs
    pairs = []
    for uid in user_ids:
        entry = interventions[uid]
        for item in entry.get(LIST_KEY, []):
            if item.get("edited_summary", ""):
                pairs.append((uid, item))

    n = len(pairs)
    if n == 0:
        print("No valid pairs, skipping controllability eval.")
        return {f"top_{k}": None for k in CTRL_KS}

    # Cache original top-max_k indices per user (one decode per unique user)
    orig_cache = {}
    edit_topk = []  # parallel to pairs

    print(f"\n  Decoding {n} pairs (max_k={max_k}, one pass)...")
    for i, (uid, item) in enumerate(tqdm(pairs, desc=f"  {LABEL} decode")):
        with torch.no_grad():
            if uid not in orig_cache:
                y_orig, _ = model.decode_text(interventions[uid]["original_summary"])
                orig_cache[uid] = torch.topk(y_orig.squeeze(), max_k).indices.cpu().numpy()

            y_edit, _ = model.decode_text(item["edited_summary"])
            edit_topk.append(torch.topk(y_edit.squeeze(), max_k).indices.cpu().numpy())

    # Compute metrics for each k by slicing
    all_results = {}

    for top_k in CTRL_KS:
        print(f"\n{'='*70}")
        print(f"  CONTROLLABILITY ({LABEL.upper()}) @ top-{top_k}")
        print(f"{'='*70}")

        genre_effects = defaultdict(list)
        genre_orig_vals = defaultdict(list)
        genre_edit_vals = defaultdict(list)

        for i, (uid, item) in enumerate(pairs):
            target_idx = item[ATTR_IDX_KEY]
            n_orig = len(interventions[uid].get("original_genre_indices", []))

            # Users need at least 2 genres in the original summary
            n_edit = n_orig + EDIT_DENOM_OFFSET
            if n_orig < 1 or n_edit < 1:
                continue

            orig_dist = genre_matrix[orig_cache[uid][:top_k]].sum(axis=0)
            edit_dist = genre_matrix[edit_topk[i][:top_k]].sum(axis=0)

            # Number of top-k items that carry the target genre
            orig_val = orig_dist[target_idx]
            edit_val = edit_dist[target_idx]

            delta = edit_val - orig_val
            genre_effects[target_idx].append(delta)
            genre_orig_vals[target_idx].append(orig_val)
            genre_edit_vals[target_idx].append(edit_val)

        # Aggregate per-genre
        per_genre = {}
        all_effects = []
        for g_idx in sorted(genre_effects.keys()):
            effs = np.array(genre_effects[g_idx])
            origs = np.array(genre_orig_vals[g_idx])
            edits = np.array(genre_edit_vals[g_idx])
            orig_mean = float(origs.mean())
            eff_mean = float(effs.mean())

            # Percent change; +/-inf if the original count is 0 but the edit changes it
            if orig_mean > 1e-8:
                pct = eff_mean / orig_mean * 100
            elif abs(eff_mean) > 1e-8:
                pct = float('inf') if eff_mean > 0 else float('-inf')
            else:
                pct = 0.0

            per_genre[GENRE_LIST[g_idx]] = {
                "effect": eff_mean,
                "original": orig_mean,
                "edited": float(edits.mean()),
                "pct_change": pct,
                "count": len(effs),
            }
            all_effects.extend(effs.tolist())

        n_users = len(set(uid for uid, _ in pairs))
        avg_effect = float(np.mean(all_effects)) if all_effects else 0.0

        print(f"    Users: {n_users} | Pairs: {n}")
        print(f"    Avg {LABEL.lower()} effect (should be {EXPECTED_SIGN}): {avg_effect:+.4f}")
        print()

        # Most-negative first (this is a removal-style intervention)
        sorted_genres = sorted(per_genre.items(), key=lambda x: x[1]["effect"])
        print(f"    {'Genre':<15s} {'orig':>8s} {'edit':>8s} {'delta':>10s} {'%chg':>8s} {'n':>6s}")
        print(f"    {'-'*58}")
        for genre, info in sorted_genres:
            pct_str = f"{info['pct_change']:>+7.1f}%" if np.isfinite(info['pct_change']) else f"{'inf':>8s}"
            print(f"    {genre:<15s} {info['original']:>8.4f} {info['edited']:>8.4f} {info['effect']:>+10.4f} {pct_str} {info['count']:>6d}")

        all_results[f"top_{top_k}"] = {
            "n_pairs": n,
            "n_users": n_users,
            "avg_effect": avg_effect,
            "per_genre": per_genre,
        }

    return all_results


# ============================================================================
# CSV helpers
# ============================================================================

def write_csv(rows, csv_path):
    if not rows:
        return
    all_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval='')
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path}")


def append_csv(rows, csv_path):
    if not rows:
        return
    all_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            existing_keys = list(reader.fieldnames) if reader.fieldnames else []
        merged_keys = list(existing_keys)
        for k in all_keys:
            if k not in set(existing_keys):
                merged_keys.append(k)
        all_keys = merged_keys
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval='')
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Appended to: {csv_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, required=True)
    parser.add_argument('--data-path', type=str, default="data/ml-20m")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--interventions", type=str, default="data/ml-20m/leave_one_out_negated.json",
                        help="Path to LOO-Negated interventions JSON")
    parser.add_argument("--genre-matrix", type=str, default="data/ml-20m/tag_matrix.npy",
                        help="Path to tag_matrix.npy (n_items x n_genres)")

    args = parser.parse_args()
    set_seed(args.seed)
    device = args.device

    # Log all output (filename uses loo_negated so it doesn't clobber loo/loi)
    log_path = os.path.join(args.exp_dir, f"eval_{INTERVENTION_TYPE}_log.txt")
    tee = Tee(log_path)

    # Load config
    config_path = os.path.join(args.exp_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Config: {config}")
    print(f"Intervention type: {INTERVENTION_TYPE}")

    # Number of items (needed to build the model)
    with open(os.path.join(args.data_path, "num_unique_items.txt")) as f:
        n_items = int(f.read().strip())

    # Load interventions + genre matrix
    with open(args.interventions, "r") as f:
        interventions = json.load(f)
    genre_matrix = np.load(args.genre_matrix)
    assert genre_matrix.shape[1] == N_GENRES, f"Expected {N_GENRES} genres, got {genre_matrix.shape[1]}"

    os.makedirs("results_all", exist_ok=True)
    csv_rows = []

    for ckpt_name in ["best_text"]:
        ckpt_path = os.path.join(args.exp_dir, f"{ckpt_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found, skipping.")
            continue

        print("\n\n" + "=" * 80)
        print(f"  EVALUATING: {ckpt_name} ({ckpt_path})")
        print("=" * 80)

        model = LatentAlignModel(
            classifier_path=config["classifier_path"],
            encoder_path=config["encoder_path"],
            decoder_path=config["decoder_path"],
            n_items=n_items,
            use_linear=config["use_linear"],
            tau=config["tau"],
            nonlinear=config["nonlinear"],
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        # Controllability
        ctrl_results = run_controllability_eval(interventions, model, genre_matrix)

        # Build CSV row
        row = {
            'exp_dir': args.exp_dir,
            'checkpoint': ckpt_name,
            'intervention_type': INTERVENTION_TYPE,
        }
        for cfg_key, cfg_val in config.items():
            row[f'cfg_{cfg_key}'] = cfg_val

        # Controllability metrics
        for k_key, k_res in ctrl_results.items():
            if k_res is None:
                continue
            row[f'ctrl_{k_key}_n_pairs'] = k_res['n_pairs']
            row[f'ctrl_{k_key}_n_users'] = k_res['n_users']
            row[f'ctrl_{k_key}_avg_effect'] = round(k_res['avg_effect'], 4)
            for genre, info in k_res['per_genre'].items():
                safe_genre = genre.lower().replace('-', '_').replace(' ', '_')
                row[f'ctrl_{k_key}_{safe_genre}_effect'] = round(info['effect'], 4)
                row[f'ctrl_{k_key}_{safe_genre}_orig'] = round(info['original'], 4)
                row[f'ctrl_{k_key}_{safe_genre}_edited'] = round(info['edited'], 4)
                # round() works on inf (returns inf), but guard for clarity
                pct = info['pct_change']
                row[f'ctrl_{k_key}_{safe_genre}_pct'] = round(pct, 1) if np.isfinite(pct) else pct
                row[f'ctrl_{k_key}_{safe_genre}_n'] = info['count']

        csv_rows.append(row)

    if csv_rows:
        csv_exp = os.path.join(args.exp_dir, f"eval_{INTERVENTION_TYPE}_results.csv")
        write_csv(csv_rows, csv_exp)

        csv_global = os.path.join("results_all", f"all_{INTERVENTION_TYPE}_results.csv")
        append_csv(csv_rows, csv_global)

    print("\n" + "=" * 80)
    print("ALL DONE.")
    print("=" * 80)
    tee.close()