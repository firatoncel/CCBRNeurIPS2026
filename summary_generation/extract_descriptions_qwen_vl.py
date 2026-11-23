"""
H&M: item descriptions from product images with Qwen3-VL-32B-Instruct (vLLM).
Each item's detail_desc is replaced by a 1-2 sentence description generated from
its image; the original detail_desc is kept when the output is empty or too short.
"""

import json
import math
import os
import re
from PIL import Image
from vllm import LLM, SamplingParams
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

SYSTEM_PROMPT = """You are a fashion product description writer. Write a concise 1-2 sentence summary of the product.

RULES:
- Output EXACTLY 1-2 sentences. Nothing else.
- No bullet points, no numbered lists, no analysis steps, no drafts.
- Be specific about: product type, color, material, silhouette, and notable design details.
- Do NOT use marketing language or subjective opinions.
- Do NOT explain your reasoning or thought process.
- Start directly with the product description."""

GENERATE_TEMPLATE = """Write a 1-2 sentence product description based on the image. Output ONLY the summary sentences, nothing else."""


def clean_output(text):
    text = text.strip()
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "\nassistant\n" in text:
        text = text.split("\nassistant\n")[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # If model still dumped analysis, try to extract last 1-2 sentences
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Filter out lines that look like analysis/steps
    clean_lines = [l for l in lines if not l.startswith(('*', '-', '1.', '2.', '3.', '4.', '#', '**'))]
    if clean_lines:
        # Take last 1-2 non-empty lines (likely the actual summary)
        text = ' '.join(clean_lines[-2:])
    return text.strip()


def is_nan(val):
    """Check if value is NaN, None, empty, or string 'nan'/'NaN'."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ('nan', '', 'none', 'null'):
        return True
    return False


def build_conversation(has_image):
    content = []
    if has_image:
        content.append({"type": "image_url", "image_url": {"url": "placeholder"}})
    content.append({"type": "text", "text": GENERATE_TEMPLATE})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def run_enhancement(
    items_path: str,
    image_base_path: str,
    output_path: str,
    max_model_len: int = 4096,
    tp_size: int = 1,
):
    with open(items_path) as f:
        items = json.load(f)

    print(f"Total items: {len(items)}")

    # Init vLLM with multimodal support
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=max_model_len,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=image_base_path,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
    )

    # Build all conversations
    item_ids = list(items.keys())
    conversations = []

    print("Loading images and building prompts...")
    for iid in tqdm(item_ids, desc="Preparing"):
        entry = items[iid]
        image_path = entry.get("image_path", "")
        full_image_path = os.path.abspath(os.path.join(image_base_path, image_path))

        # Check that the image can be read
        try:
            Image.open(full_image_path).convert("RGB")
            has_image = True
        except Exception:
            has_image = False

        conv = build_conversation(has_image)

        if has_image:
            for msg in conv:
                if isinstance(msg["content"], list):
                    for block in msg["content"]:
                        if block.get("type") == "image_url":
                            block["image_url"] = {"url": f"file://{full_image_path}"}

        conversations.append(conv)

    # Feed in chunks to avoid OOM with multimodal, save after each chunk
    CHUNK_SIZE = 256
    checkpoint_path = output_path + ".checkpoint"
    failed = []

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            items = json.load(f)
        processed = {iid for iid in item_ids if items[iid].get("_enhanced", False)}
        print(f"Resuming from checkpoint: {len(processed)} already processed")
    else:
        processed = set()

    remaining_ids = [iid for iid in item_ids if iid not in processed]
    remaining_convs = [conversations[item_ids.index(iid)] for iid in remaining_ids]

    print(f"Running vLLM inference on {len(remaining_convs)} items in chunks of {CHUNK_SIZE}...")
    for start in tqdm(range(0, len(remaining_convs), CHUNK_SIZE), desc="Processing chunks"):
        chunk_ids = remaining_ids[start:start + CHUNK_SIZE]
        chunk_convs = remaining_convs[start:start + CHUNK_SIZE]

        outputs = llm.chat(
            messages=chunk_convs,
            sampling_params=sampling_params,
        )

        for iid, output in zip(chunk_ids, outputs):
            edited = clean_output(output.outputs[0].text)

            if not edited or len(edited) < 10:
                failed.append(iid)
                edited = items[iid].get("detail_desc", "")
                if is_nan(edited):
                    edited = ""

            items[iid]["detail_desc"] = edited
            items[iid]["_enhanced"] = True

        # Save checkpoint after each chunk
        with open(checkpoint_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  Checkpoint saved: {start + len(chunk_ids)}/{len(remaining_convs)}")

    # Final save — remove _enhanced flags
    for iid in item_ids:
        items[iid].pop("_enhanced", None)

    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nDone. {len(item_ids)} items processed.")
    print(f"  Failed: {len(failed)}")
    n_nan = sum(1 for iid in item_ids if is_nan(items[iid].get("detail_desc")))
    print(f"  Remaining NaN: {n_nan}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Items JSON")
    parser.add_argument("--image-base", type=str, required=True, help="Base path for images")
    parser.add_argument("--output", type=str, default="items_enhanced.json")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()

    run_enhancement(
        args.input, args.image_base, args.output,
        args.max_model_len, args.tp_size,
    )