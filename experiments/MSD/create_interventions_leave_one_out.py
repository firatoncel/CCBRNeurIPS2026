"""
Create the leave-one-out (LOO) edited summaries used in the controllability evaluation.
For each user summary, for EACH tag the user has:
  - Edit the summary to remove ONLY that single tag's mentions.

Produces: original + N edited summaries per user (where N = number of tags).
The edits are made by Qwen3.5-27B (MODEL_ID) with batched greedy decoding.
Requires: summaries.json and the extracted tags written by extract_tags.py.
"""

import json
import os
import argparse
import random
import numpy as np
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ─── Tag definitions (must match extract_tags.py) ──────────────────────────

TAG_MAP = {
    '60s': 0, '70s': 1, '80s': 2, '90s': 3, 'singer-songwriter': 4,
    'female vocalist': 5, 'female vocalists': 6, 'male vocalist': 7,
    'male vocalists': 8, 'piano': 9, 'acoustic': 10, 'alternative rock': 11,
    'amazing': 12, 'ambient': 13, 'american': 14, 'ballad': 15, 'blues': 16,
    'catchy': 17, 'chill': 18, 'chillout': 19, 'classic rock': 20,
    'classic': 21, 'cool': 22, 'country': 23, 'dance': 24, 'death metal': 25,
    'downtempo': 26, 'easy listening': 27, 'electro': 28, 'electronic': 29,
    'electronica': 30, 'emo': 31, 'epic': 32, 'experimental': 33, 'folk': 34,
    'fun': 35, 'funk': 36, 'funky': 37, 'good': 38, 'guitar': 39,
    'happy': 40, 'hard rock': 41, 'hardcore': 42, 'heavy metal': 43,
    'hip hop': 44, 'house': 45, 'indie pop': 46, 'indie rock': 47,
    'instrumental': 48, 'jazz': 49, 'lounge': 50, 'loved': 51,
    'melancholic': 52, 'melancholy': 53, 'mellow': 54, 'metal': 55,
    'new wave': 56, 'oldies': 57, 'party': 58, 'pop rock': 59,
    'post-punk': 60, 'progressive rock': 61, 'psychedelic': 62,
    'punk rock': 63, 'punk': 64, 'rap': 65, 'reggae': 66, 'relax': 67,
    'relaxing': 68, 'rnb': 69, 'romantic': 70, 'sad': 71, 'sexy': 72,
    'soft rock': 73, 'soul': 74, 'summer': 75, 'techno': 76, 'trance': 77,
    'upbeat': 78, 'world': 79
}

TAG_LIST = sorted(TAG_MAP.keys(), key=lambda x: TAG_MAP[x])

# ─── LLM-based summary editing ─────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-27B"


def load_model(device="cuda"):
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


REMOVE_PROMPT = """You are a precise text editor. Your task is to edit a music listening summary to REMOVE all mentions and implications of ONE specific music tag, while keeping the rest of the summary natural and coherent.

TAG TO REMOVE: {tag_to_remove}

ORIGINAL SUMMARY:
{summary}

RULES:
- Remove all explicit and implicit references to the specified tag.
- Do NOT add any new music attributes or genres.
- Keep the overall structure, tone, and remaining content intact.
- The edited summary should read naturally, as if that tag was never part of the user's taste.
- If removing the tag leaves an awkward gap or broken sentence, smooth it out minimally.
- Do NOT remove any other tags or attributes beyond the one specified.

Respond with ONLY the edited summary text, nothing else."""


def clean_output(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def batch_edit_summaries(
    prompts: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    batch_size: int = 8,
) -> list[str]:
    """Batch inference over a list of prompts."""
    all_results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        batch_texts = []
        for p in batch_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch_texts.append(text)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, prompt_ids in enumerate(inputs["input_ids"]):
            gen_ids = output_ids[j, prompt_ids.shape[0]:]
            result = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_results.append(clean_output(result))

    return all_results


# ─── Experiment creation ───────────────────────────────────────────────────

def create_experiments(
    summaries_path: str,
    tags_path: str,
    output_path: str,
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 8,
    save_every: int = 10,
):
    random.seed(seed)
    np.random.seed(seed)

    with open(summaries_path, "r") as f:
        summaries = json.load(f)
    with open(tags_path, "r") as f:
        extracted_tags = json.load(f)

    model, tokenizer = load_model(device=device)

    # Resume from existing output if available
    results = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} users already processed, skipping them.")

    total_edits = 0
    users_since_save = 0

    for user_id in tqdm(summaries.keys(), desc="Creating experiments"):
        if user_id in results:
            total_edits += len(results[user_id]["removals"])
            continue

        summary_text = summaries[user_id]["summary"]
        user_tag_indices = extracted_tags[user_id]["tag_indices"]
        user_tag_names = extracted_tags[user_id]["tag_names"]

        if not user_tag_indices:
            results[user_id] = {
                "original_summary": summary_text,
                "original_tags": user_tag_names,
                "original_tag_indices": user_tag_indices,
                "removals": [],
            }
            users_since_save += 1
            continue

        # Build all prompts for this user
        prompts = [
            REMOVE_PROMPT.format(tag_to_remove=tag_name, summary=summary_text)
            for tag_name in user_tag_names
        ]

        # Batched inference
        edited_summaries = batch_edit_summaries(
            prompts, model, tokenizer, batch_size=batch_size,
        )

        removals = []
        for tag_idx, tag_name, edited in zip(user_tag_indices, user_tag_names, edited_summaries):
            expected_remaining = [t for t in user_tag_indices if t != tag_idx]
            removals.append({
                "removed_tag": tag_name,
                "removed_tag_idx": tag_idx,
                "edited_summary": edited,
                "expected_tag_indices": expected_remaining,
                "expected_tag_names": [TAG_LIST[i] for i in expected_remaining],
            })
            total_edits += 1

        results[user_id] = {
            "original_summary": summary_text,
            "original_tags": user_tag_names,
            "original_tag_indices": user_tag_indices,
            "removals": removals,
        }

        users_since_save += 1
        tqdm.write(
            f"  [{user_id}] {len(user_tag_indices)} tags → {len(removals)} edited summaries"
        )

        if users_since_save >= save_every:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            users_since_save = 0
            tqdm.write(f"  Checkpoint saved ({len(results)} users, {total_edits} edits)")

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} users, {total_edits} total edits saved to {output_path}")
    return results


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create leave-one-out tag removal experiments")
    parser.add_argument("--summaries", type=str, required=True, help="Path to summaries JSON")
    parser.add_argument("--tags", type=str, required=True, help="Path to extracted tags JSON")
    parser.add_argument("--output", type=str, default="leave_one_out_interventions.json", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N users")
    args = parser.parse_args()

    create_experiments(
        summaries_path=args.summaries,
        tags_path=args.tags,
        output_path=args.output,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )