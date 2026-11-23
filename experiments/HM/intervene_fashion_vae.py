"""
Evaluate LOO/LOI product-type and colour interventions on H&M summary-based
recommendations (Multi-VAE / MacridVAE backbones, model_ddp_vae.py).

Supports 4 intervention types:
  --intervention-type product_loo  (product-type removal, expect NEGATIVE effect)
  --intervention-type product_loi  (product-type addition, expect POSITIVE effect)
  --intervention-type color_loo    (colour removal, expect NEGATIVE effect)
  --intervention-type color_loi    (colour addition, expect POSITIVE effect)

For each edited summary, counts how many of the top-k recommended items
(k = 5, 10, 20, 50) have the target product type / colour, compared with the
original summary.

Input JSON structure (mode = category or color):
  LOO: {user_id: {original_summary, original_<mode>_indices, removals: [
          {removed_<mode>_idx, edited_summary, ...}, ...]}}
  LOI: {user_id: {original_summary, original_<mode>_indices, additions: [
          {added_<mode>_idx, edited_summary, ...}, ...]}}

Usage:
    python intervene_fashion_vae.py --exp-dir results_hm/exp1 \
        --intervention-type product_loo --interventions loo_category.json

    python intervene_fashion_vae.py --exp-dir results_hm/exp1 \
        --intervention-type color_loi --interventions loi_color.json
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

# ─── Attribute maps ────────────────────────────────────────────────────────

PRODUCT_TYPE_MAP = {
    'Bag': 0, 'Ballerinas': 1, 'Belt': 2, 'Bikini top': 3, 'Blazer': 4,
    'Blouse': 5, 'Bodysuit': 6, 'Boots': 7, 'Bra': 8, 'Cardigan': 9,
    'Coat': 10, 'Costumes': 11, 'Dress': 12, 'Hair clip': 13, 'Hair string': 14,
    'Hair ties': 15, 'Hair/alice band': 16, 'Hairband': 17, 'Hat/beanie': 18,
    'Hoodie': 19, 'Jacket': 20, 'Jumpsuit/Playsuit': 21, 'Leggings/Tights': 22,
    'Necklace': 23, 'Night gown': 24, 'Nipple covers': 25, 'Polo shirt': 26,
    'Pyjama bottom': 27, 'Pyjama set': 28, 'Robe': 29, 'Sandals': 30,
    'Sarong': 31, 'Scarf': 32, 'Shirt': 33, 'Shorts': 34, 'Skirt': 35,
    'Slippers': 36, 'Sneakers': 37, 'Socks': 38, 'Sunglasses': 39,
    'Sweater': 40, 'Swimsuit': 41, 'Swimwear bottom': 42, 'Swimwear set': 43,
    'T-shirt': 44, 'Top': 45, 'Trousers': 46, 'Underwear Tights': 47,
    'Underwear body': 48, 'Underwear bottom': 49, 'Vest top': 50
}

COLOR_MAP = {
    'Beige': 0, 'Black': 1, 'Blue': 2, 'Bronze/Copper': 3, 'Gold': 4,
    'Green': 5, 'Grey': 6, 'Orange': 7, 'Pink': 8, 'Purple': 9,
    'Red': 10, 'Silver': 11, 'Transparent': 12, 'Turquoise': 13,
    'White': 14, 'Yellow': 15,
}

IDX_TO_PRODUCT = {v: k for k, v in PRODUCT_TYPE_MAP.items()}
IDX_TO_COLOR = {v: k for k, v in COLOR_MAP.items()}

CTRL_KS = [5, 10, 20, 50]

# ─── Helpers ────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def build_item_to_attr_id(item_to_name_map, name_to_id_map):
    """Map item_id -> attribute_id (single attribute per item)."""
    result = {}
    for item_id, name in item_to_name_map.items():
        if name in name_to_id_map:
            result[int(item_id)] = name_to_id_map[name]
    return result

def topk_attr_count(scores_tensor, target_attr_id, item_to_attr_id, k):
    """How many of the top-k items carry the target attribute id."""
    _, topk_indices = torch.topk(scores_tensor, k=k)
    topk_ids = topk_indices.cpu().tolist()
    return sum(1 for iid in topk_ids if item_to_attr_id.get(iid) == target_attr_id)

# ─── Intervention type config ──────────────────────────────────────────────

def get_config(intervention_type):
    """Per-type knobs: {product, color} x {loo, loi}."""
    if intervention_type.startswith("product"):
        attr_map = PRODUCT_TYPE_MAP
        idx_to_name = IDX_TO_PRODUCT
        mode = "category"
    else:
        attr_map = COLOR_MAP
        idx_to_name = IDX_TO_COLOR
        mode = "color"

    is_removal = intervention_type.endswith("_loo")
    label = "Removal" if is_removal else "Addition"

    if is_removal:
        list_key = "removals"
        attr_idx_key = f"removed_{mode}_idx"
    else:
        list_key = "additions"
        attr_idx_key = f"added_{mode}_idx"

    # Key of the list of attributes mentioned in the original summary
    orig_indices_key = f"original_{mode}_indices"

    return {
        "attr_map": attr_map,
        "idx_to_name": idx_to_name,
        "mode": mode,
        "is_removal": is_removal,
        "label": label,
        "list_key": list_key,
        "attr_idx_key": attr_idx_key,
        "orig_indices_key": orig_indices_key,
    }


# ============================================================================
# Controllability eval — LOO/LOI, single-pass decode
# ============================================================================

def run_controllability_eval(interventions, model, item_to_attr_id, cfg):
    """Decode each summary once, cache the full score vector, then count
    attribute hits in the top-k for each k in CTRL_KS."""
    idx_to_name = cfg["idx_to_name"]
    list_key = cfg["list_key"]
    attr_idx_key = cfg["attr_idx_key"]
    orig_indices_key = cfg["orig_indices_key"]
    is_removal = cfg["is_removal"]
    label = cfg["label"]

    user_ids = list(interventions.keys())
    pairs = []
    for uid in user_ids:
        for item in interventions[uid].get(list_key, []):
            if item.get("edited_summary", ""):
                pairs.append((uid, item))

    n = len(pairs)
    if n == 0:
        print("No valid pairs, skipping.")
        return {f"top_{k}": None for k in CTRL_KS}

    orig_cache = {}
    edit_scores = []  # parallel to pairs

    print(f"\n  Decoding {n} pairs (one pass)...")
    for i, (uid, item) in enumerate(tqdm(pairs, desc=f"  {label} decode")):
        with torch.no_grad():
            if uid not in orig_cache:
                y_orig, _, _ = model.decode_text(interventions[uid]["original_summary"])
                orig_cache[uid] = y_orig.squeeze()
            y_edit, _, _ = model.decode_text(item["edited_summary"])
            edit_scores.append(y_edit.squeeze())

    all_results = {}
    for top_k in CTRL_KS:
        print(f"\n{'='*70}")
        print(f"  CONTROLLABILITY ({label.upper()}) @ top-{top_k}")
        print(f"{'='*70}")

        attr_effects = defaultdict(list)
        attr_orig_vals = defaultdict(list)
        attr_edit_vals = defaultdict(list)

        for i, (uid, item) in enumerate(pairs):
            target_idx = item[attr_idx_key]

            # Removal needs at least 2 attributes in the original summary, addition at least 1
            n_orig = len(interventions[uid][orig_indices_key])
            min_n_orig = 2 if is_removal else 1
            if n_orig < min_n_orig:
                continue

            # Number of top-k items that carry the target attribute
            orig_val = topk_attr_count(orig_cache[uid], target_idx, item_to_attr_id, top_k)
            edit_val = topk_attr_count(edit_scores[i], target_idx, item_to_attr_id, top_k)

            delta = edit_val - orig_val
            attr_effects[target_idx].append(delta)
            attr_orig_vals[target_idx].append(orig_val)
            attr_edit_vals[target_idx].append(edit_val)

        per_attr = {}
        all_deltas = []
        for a_idx in sorted(attr_effects.keys()):
            effs = np.array(attr_effects[a_idx])
            origs = np.array(attr_orig_vals[a_idx])
            edits = np.array(attr_edit_vals[a_idx])
            orig_mean = float(origs.mean())
            eff_mean = float(effs.mean())

            # Percent change; inf if the original count is 0
            if orig_mean > 0:
                pct = eff_mean / orig_mean * 100
            else:
                pct = np.inf

            name = idx_to_name.get(a_idx, str(a_idx))
            per_attr[name] = {
                "effect": eff_mean, "original": orig_mean,
                "edited": float(edits.mean()), "pct_change": pct, "count": len(effs),
            }
            all_deltas.extend(effs.tolist())

        n_users = len(set(uid for uid, _ in pairs))
        avg_effect = float(np.mean(all_deltas)) if all_deltas else 0.0
        expected = "NEGATIVE" if is_removal else "POSITIVE"

        print(f"    Users: {n_users} | Pairs: {n}")
        print(f"    Avg {label.lower()} effect (should be {expected}): {avg_effect:+.4f}")
        print()

        # LOO: most-negative first; LOI: most-positive first.
        sort_key = (lambda x: x[1]["effect"]) if is_removal else (lambda x: -x[1]["effect"])
        sorted_attrs = sorted(per_attr.items(), key=sort_key)
        print(f"    {'Attribute':<25s} {'orig':>8s} {'edit':>8s} {'delta':>10s} {'%chg':>8s} {'n':>6s}")
        print(f"    {'-'*68}")
        for name, info in sorted_attrs:
            print(f"    {name:<25s} {info['original']:>8.4f} {info['edited']:>8.4f} {info['effect']:>+10.4f} {info['pct_change']:>+7.1f}% {info['count']:>6d}")

        all_results[f"top_{top_k}"] = {
            "n_pairs": n, "n_users": n_users,
            "avg_effect": avg_effect, "per_attr": per_attr,
        }
    return all_results


# ─── CSV ────────────────────────────────────────────────────────────────────

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
    parser.add_argument('--data-path', type=str, default="easier_v2_vlm_llm")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--intervention-type", type=str, required=True,
                        choices=["product_loo", "product_loi", "color_loo", "color_loi"])
    parser.add_argument("--interventions", type=str, required=True,
                        help="Path to LOO/LOI interventions JSON")
    parser.add_argument("--item-color", type=str, default="easier_v2_vlm_llm/item_color.json")
    parser.add_argument("--item-product", type=str, default="easier_v2_vlm_llm/item_to_product_type.json")

    args = parser.parse_args()
    set_seed(args.seed)
    device = args.device

    cfg = get_config(args.intervention_type)

    log_name = f"eval_{args.intervention_type}_log.txt"
    tee = Tee(os.path.join(args.exp_dir, log_name))

    # Load config
    config_path = os.path.join(args.exp_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    print(f"Config: {config}")
    print(f"Intervention type: {args.intervention_type}")

    # Item-to-attribute lookup
    if cfg["mode"] == "color":
        item_to_attr_id = build_item_to_attr_id(load_json(args.item_color), COLOR_MAP)
    else:
        item_to_attr_id = build_item_to_attr_id(load_json(args.item_product), PRODUCT_TYPE_MAP)

    # Load interventions
    with open(args.interventions) as f:
        interventions = json.load(f)

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

        # Instantiate model from config, then load trained state_dict
        model = LatentAlignModel(
            classifier_path=config["base_path"],
            encoder_path=config["encoder_path"],
            decoder_path=config["decoder_path"],
            data_path=args.data_path,
            use_linear=config["use_linear"],
            tau=config["tau"],
            nonlinear=config["nonlinear"],
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        # Controllability
        ctrl_results = run_controllability_eval(interventions, model, item_to_attr_id, cfg)

        # CSV row
        row = {'exp_dir': args.exp_dir, 'checkpoint': ckpt_name, 'intervention_type': args.intervention_type}
        for cfg_key, cfg_val in config.items():
            row[f'cfg_{cfg_key}'] = cfg_val
        for k_key, k_res in ctrl_results.items():
            if k_res is None:
                continue
            row[f'ctrl_{k_key}_n_pairs'] = k_res['n_pairs']
            row[f'ctrl_{k_key}_n_users'] = k_res['n_users']
            row[f'ctrl_{k_key}_avg_effect'] = round(k_res['avg_effect'], 4)
            for attr_name, info in k_res['per_attr'].items():
                safe = attr_name.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
                row[f'ctrl_{k_key}_{safe}_effect'] = round(info['effect'], 4)
                row[f'ctrl_{k_key}_{safe}_pct'] = round(info['pct_change'], 1)
                row[f'ctrl_{k_key}_{safe}_orig'] = round(info['original'], 4)
                row[f'ctrl_{k_key}_{safe}_edit'] = round(info['edited'], 4)
                row[f'ctrl_{k_key}_{safe}_n'] = info['count']
        csv_rows.append(row)

    if csv_rows:
        csv_exp = os.path.join(args.exp_dir, f"eval_{args.intervention_type}_results.csv")
        write_csv(csv_rows, csv_exp)
        csv_global = os.path.join("results_all", f"all_{args.intervention_type}_results.csv")
        append_csv(csv_rows, csv_global)

    print("\n" + "=" * 80)
    print("ALL DONE.")
    print("=" * 80)
    tee.close()
