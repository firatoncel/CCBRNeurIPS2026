"""
intervene_fashion_tears.py — Attribute-intervention eval for TearsAlignModel
                             (fashion / H&M).

Text-only controllability eval. We run the text branch of TearsAlignModel
(text encoder -> mu/logvar -> z_text -> shared decoder) on edited summaries
with no collab input, and measure how the recommended item distribution
shifts under LOO (removal) or LOI (addition) attribute edits.

Supports 4 intervention types:
  --intervention-type product_loo
  --intervention-type product_loi
  --intervention-type color_loo
  --intervention-type color_loi

Each item is mapped to a single attribute id (its product type or its
color), so we use top-k attribute counts rather than a sum over an
(item x tag) matrix.

Top-k = 5, 10, 20, 50, using the same LOO/LOI JSON format as
intervene_fashion_ae.py / intervene_fashion_vae.py.

Usage:
    python intervene_fashion_tears.py --exp-dir results_tears/exp1 \
        --intervention-type product_loo --interventions loo_category.json

    python intervene_fashion_tears.py --exp-dir results_tears/exp1 \
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
import torch.nn.functional as F
from tqdm import tqdm

from model_tears import TearsAlignModel
from utils import set_seed


# ─── Attribute maps (same as intervene_fashion_ae.py) ──────────────────────

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
    """Per-type knobs that drive controllability eval. Fashion has 4 types:
    {product, color} x {loo, loi}.
    """
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
# Text-only decode (no collab input)
# ============================================================================

@torch.no_grad()
def decode_text_only(model, summary, device):
    """Run only the text branch of TearsAlignModel, no `x`.

    Returns logits over items, shape (1, num_items).
    """
    if not isinstance(summary, list):
        summary = [summary]

    text_h = model._encode_text(summary)                      # (1, text_dim)
    gauss = model.text_to_gaussian(text_h)                    # (1, 2*latent)
    text_mu = gauss[:, :model.latent_dim]
    text_logvar = gauss[:, model.latent_dim:]
    text_logvar = torch.clamp(text_logvar, min=-10.0, max=10.0)
    z_text = model.vae.reparameterize(text_mu, text_logvar)   # eval mode -> mu

    # Macrid needs per-concept normalization on z_text and a populated
    # decode cache. The cache is filled by _MacridBackbone.encode, so
    # poke a zero-input encode pass to populate it.
    if model.backbone_kind == "macrid":
        K, D = model.vae.n_concepts, model.vae.latent_dim
        # Populate cache with a dummy encode (cates / items_n depend only on
        # the learned parameters, so the dummy x doesn't pollute results).
        dummy_x = torch.zeros(1, model.vae.num_items, device=device)
        _ = model.vae.encode(dummy_x)
        z_text = F.normalize(z_text.view(-1, K, D), dim=-1).view(-1, K * D)

    logits = model.vae.decode(z_text)                          # (1, num_items)
    return logits


# ============================================================================
# Controllability eval — LOO/LOI on the text-only branch
# ============================================================================

def run_controllability_eval(interventions, model, item_to_attr_id, cfg, device):
    """Decode each summary ONCE via the text-only branch, cache the full
    score tensor per user, then count attribute hits in top-k for each k."""
    idx_to_name = cfg["idx_to_name"]
    list_key = cfg["list_key"]
    attr_idx_key = cfg["attr_idx_key"]
    orig_indices_key = cfg["orig_indices_key"]
    is_removal = cfg["is_removal"]
    label = cfg["label"]

    user_ids = list(interventions.keys())

    # Collect all (user, intervention_item) pairs
    pairs = []
    for uid in user_ids:
        for item in interventions[uid].get(list_key, []):
            if item.get("edited_summary", ""):
                pairs.append((uid, item))

    n = len(pairs)
    if n == 0:
        print("No valid pairs, skipping controllability eval.")
        return {f"top_{k}": None for k in CTRL_KS}

    # Cache the full score vector per summary. Each item has a single attribute
    # id, so attribute hits are counted with topk_attr_count.
    orig_cache = {}
    edit_scores = []  # parallel to pairs

    print(f"\n  Decoding {n} pairs (one pass)...")
    for i, (uid, item) in enumerate(tqdm(pairs, desc=f"  {label} decode")):
        if uid not in orig_cache:
            y_orig = decode_text_only(
                model, interventions[uid]["original_summary"], device)
            orig_cache[uid] = y_orig.squeeze().cpu()

        y_edit = decode_text_only(model, item["edited_summary"], device)
        edit_scores.append(y_edit.squeeze().cpu())

    # Compute metrics for each k
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
            orig_val = topk_attr_count(
                orig_cache[uid], target_idx, item_to_attr_id, top_k)
            edit_val = topk_attr_count(
                edit_scores[i], target_idx, item_to_attr_id, top_k)

            delta = edit_val - orig_val
            attr_effects[target_idx].append(delta)
            attr_orig_vals[target_idx].append(orig_val)
            attr_edit_vals[target_idx].append(edit_val)

        # Aggregate per-attribute
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
                "effect": eff_mean,
                "original": orig_mean,
                "edited": float(edits.mean()),
                "pct_change": pct,
                "count": len(effs),
            }
            all_deltas.extend(effs.tolist())

        n_users = len(set(uid for uid, _ in pairs))
        avg_effect = float(np.mean(all_deltas)) if all_deltas else 0.0
        expected = "NEGATIVE" if is_removal else "POSITIVE"

        print(f"    Users: {n_users} | Pairs: {n}")
        print(f"    Avg {label.lower()} effect (should be {expected}): "
              f"{avg_effect:+.4f}")
        print()

        # LOO: most-negative first; LOI: most-positive first.
        sort_key = (lambda x: x[1]["effect"]) if is_removal else (lambda x: -x[1]["effect"])
        sorted_attrs = sorted(per_attr.items(), key=sort_key)
        # Attribute names can be long ('Jumpsuit/Playsuit'), 25-wide column
        print(f"    {'Attribute':<25s} {'orig':>8s} {'edit':>8s} "
              f"{'delta':>10s} {'%chg':>8s} {'n':>6s}")
        print(f"    {'-'*68}")
        for name, info in sorted_attrs:
            print(f"    {name:<25s} {info['original']:>8.4f} "
                  f"{info['edited']:>8.4f} {info['effect']:>+10.4f} "
                  f"{info['pct_change']:>+7.1f}% {info['count']:>6d}")

        all_results[f"top_{top_k}"] = {
            "n_pairs": n,
            "n_users": n_users,
            "avg_effect": avg_effect,
            "per_attr": per_attr,
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
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--intervention-type", type=str, required=True,
                        choices=["product_loo", "product_loi",
                                 "color_loo", "color_loi"],
                        help="{product, color} x {loo, loi}")
    parser.add_argument("--interventions", type=str, required=True,
                        help="Path to LOO/LOI interventions JSON")
    parser.add_argument("--item-color", type=str,
                        default="easier_v2_vlm_llm/item_color.json",
                        help="JSON: {item_id: color_name}")
    parser.add_argument("--item-product", type=str,
                        default="easier_v2_vlm_llm/item_to_product_type.json",
                        help="JSON: {item_id: product_type_name}")

    # Which checkpoints saved by main_tears.py to evaluate. Defaults to
    # best_merged.pt; you can pass multiple names if you want.
    parser.add_argument("--checkpoints", type=str, nargs="+",
                        default=["best_merged"],
                        help="Checkpoint names (without .pt) to evaluate.")

    args = parser.parse_args()
    set_seed(args.seed)
    device = args.device

    cfg = get_config(args.intervention_type)

    # Log all output (filename includes intervention type so different
    # intervention types don't clobber each other)
    log_path = os.path.join(args.exp_dir,
                             f"eval_{args.intervention_type}_log.txt")
    tee = Tee(log_path)

    # Load training config
    config_path = os.path.join(args.exp_dir, 'config.json')
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    print(f"Train config: {train_config}")
    print(f"Intervention type: {args.intervention_type}")

    # Item -> attribute id lookup (single attribute per item)
    if cfg["mode"] == "color":
        item_to_attr_id = build_item_to_attr_id(
            load_json(args.item_color), COLOR_MAP)
    else:
        item_to_attr_id = build_item_to_attr_id(
            load_json(args.item_product), PRODUCT_TYPE_MAP)

    # Load interventions
    with open(args.interventions, "r") as f:
        interventions = json.load(f)

    os.makedirs("results_all", exist_ok=True)
    csv_rows = []

    for ckpt_name in args.checkpoints:
        ckpt_path = os.path.join(args.exp_dir, f"{ckpt_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found, skipping.")
            continue

        print("\n\n" + "=" * 80)
        print(f"  EVALUATING: {ckpt_name} ({ckpt_path})")
        print("=" * 80)

        # Build the same architecture as training. LoRA hyperparameters
        # come from the saved training config so the state_dict loads
        # cleanly.
        model = TearsAlignModel(
            encoder_path=train_config["encoder_path"],
            decoder_path=train_config["decoder_path"],
            alpha=train_config.get("alpha", 0.5),
            lora=not train_config.get("no_lora", False),
            lora_r=train_config.get("lora_r", 64),
            lora_alpha=train_config.get("lora_alpha", 16),
            lora_dropout=train_config.get("lora_dropout", 0.1),
        )
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] missing keys: {len(missing)} "
                  f"(first few: {missing[:5]})")
        if unexpected:
            print(f"[warn] unexpected keys: {len(unexpected)} "
                  f"(first few: {unexpected[:5]})")
        model.to(device)
        model.eval()

        # Controllability (text-only branch)
        ctrl_results = run_controllability_eval(
            interventions, model, item_to_attr_id, cfg, device)

        # Build CSV row
        row = {
            'exp_dir': args.exp_dir,
            'checkpoint': ckpt_name,
            'intervention_type': args.intervention_type,
        }
        for cfg_key, cfg_val in train_config.items():
            row[f'cfg_{cfg_key}'] = cfg_val

        # Controllability metrics
        for k_key, k_res in ctrl_results.items():
            if k_res is None:
                continue
            row[f'ctrl_{k_key}_n_pairs'] = k_res['n_pairs']
            row[f'ctrl_{k_key}_n_users'] = k_res['n_users']
            row[f'ctrl_{k_key}_avg_effect'] = round(k_res['avg_effect'], 4)
            for attr_name, info in k_res['per_attr'].items():
                safe = (attr_name.lower()
                                 .replace(' ', '_')
                                 .replace('/', '_')
                                 .replace('-', '_'))
                row[f'ctrl_{k_key}_{safe}_effect'] = round(info['effect'], 4)
                pct = info['pct_change']
                row[f'ctrl_{k_key}_{safe}_pct'] = (
                    round(pct, 1) if np.isfinite(pct) else pct)
                row[f'ctrl_{k_key}_{safe}_orig'] = round(info['original'], 4)
                row[f'ctrl_{k_key}_{safe}_edit'] = round(info['edited'], 4)
                row[f'ctrl_{k_key}_{safe}_n'] = info['count']

        csv_rows.append(row)

    if csv_rows:
        csv_exp = os.path.join(
            args.exp_dir, f"eval_{args.intervention_type}_tears_results.csv")
        write_csv(csv_rows, csv_exp)

        csv_global = os.path.join(
            "results_all", f"all_{args.intervention_type}_tears_results.csv")
        append_csv(csv_rows, csv_global)

    print("\n" + "=" * 80)
    print("ALL DONE.")
    print("=" * 80)
    tee.close()