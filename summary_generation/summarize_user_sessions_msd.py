"""
User Session Summary Generation for MSD using Qwen3-30B-A3B-Instruct-2507
- Text-only LLM via vLLM (tensor parallel, no vision)
- Processes train/valid/test splits with optional k-item limit per user
- Saves combined JSON with resume support
"""

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 1.0
    TOP_P = 1.0
    TOP_K = 1
    MSD_DIR = Path(__file__).parent / "MSD"
    OUTPUT_DIR = Path(__file__).parent / "msd_summaries"
    BATCH_SIZE = 16
    TENSOR_PARALLEL_SIZE = 1
    GPU_MEMORY_UTILIZATION = 0.9
    MAX_MODEL_LEN = 8192


PROMPT_TEMPLATE = (
    "Below are descriptions of songs a user has listened to:\n\n"
    "{items}\n"
    "Based on the songs above, summarize this user's music taste and preferences "
    "in 3-4 sentences. Focus on recurring genres, moods, tempos, instrumentation, "
    "and vocal styles across the songs.\n\n"
    "Summary of user's music preferences:"
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_safe(path: Path) -> Dict:
    """Load JSON, attempting recovery if the file is truncated/corrupted."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[WARN] Corrupted JSON detected: {path}. Attempting recovery...")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Find the last fully written entry (ends with '},\n')
        idx = content.rfind("\n  },\n")
        if idx > 0:
            candidate = content[: idx] + "\n  }\n}"
            try:
                data = json.loads(candidate)
                print(f"[WARN] Recovered {len(data)} entries. Re-saving clean file.")
                save_json(data, path)
                return data
            except json.JSONDecodeError:
                pass
        print("[WARN] Recovery failed. Starting fresh.")
        return {}


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_messages(
    item_ids: List[int],
    descriptions: Dict,
    k: Optional[int],
    max_input_tokens: int = 8000,
    user_id: Optional[str] = None,
) -> Tuple[List[Dict], int]:
    """Build a chat message for one user from song descriptions.

    - k is set   : use first k items (ordered history)
    - k is None  : use all items; if estimated tokens > max_input_tokens,
                   randomly sample down to fit (seed = user_id for reproducibility)

    Returns:
        messages       – chat message list (empty list if no valid descriptions)
        num_items_used – number of songs actually included
    """
    if k is not None:
        item_ids = item_ids[:k]

    # Collect descriptions for all candidate items
    candidates = []
    for item_id in item_ids:
        desc = descriptions.get(str(item_id))
        if desc:
            candidates.append((item_id, desc))

    if not candidates:
        return [], 0

    # Build full line strings upfront (includes "Song X: " prefix and newlines)
    # so char budget accounts for all overhead including the prompt template (~300 chars)
    lines_all = [f"Song {i + 1}: {desc}" for i, (_, desc) in enumerate(candidates)]

    # ~3.5 chars per token (conservative for music/English text); reserve 300 chars for template
    char_budget = int(max_input_tokens * 3.5) - 300

    total_chars = sum(len(l) for l in lines_all)
    if total_chars > char_budget:
        # Shuffle candidates randomly, then take items sequentially until budget is reached
        indexed = list(enumerate(lines_all))  # (orig_idx, line)
        rng = random.Random(int(user_id) if user_id is not None else 42)
        rng.shuffle(indexed)
        kept = []
        used = 0
        for orig_idx, line in indexed:
            if used + len(line) > char_budget:
                break
            kept.append((orig_idx, line))
            used += len(line)
        kept.sort(key=lambda x: x[0])
        lines = [line for _, line in kept]
    else:
        lines = lines_all
    prompt = PROMPT_TEMPLATE.format(items="\n".join(lines))
    return [{"role": "user", "content": prompt}], len(lines)


# ---------------------------------------------------------------------------
# LLM summarizer
# ---------------------------------------------------------------------------

class LLMSummarizer:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        enable_thinking: bool = False,
    ):
        print(f"Loading model : {model_name}")
        print(f"Tensor parallel: {tensor_parallel_size} GPU(s)")

        self.enable_thinking = enable_thinking

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        print("Model loaded successfully!")

    def generate_batch(
        self,
        batch_messages: List[List[Dict]],
        sampling_params: SamplingParams,
    ) -> List[str]:
        """Generate summaries for a batch of users."""
        valid_indices = []
        vllm_inputs = []

        for i, messages in enumerate(batch_messages):
            if not messages:
                continue
            valid_indices.append(i)
            vllm_inputs.append({"messages": messages})

        if not vllm_inputs:
            return [""] * len(batch_messages)

        outputs = self.llm.chat(
            messages=[inp["messages"] for inp in vllm_inputs],
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )

        results = [""] * len(batch_messages)
        for idx, output in zip(valid_indices, outputs):
            text = output.outputs[0].text.strip()
            # Strip <think>...</think> block if present
            if "</think>" in text:
                text = text.split("</think>", 1)[-1].strip()
            results[idx] = text if text else "No summary generated."

        return results


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_split(
    session_file: Path,
    descriptions: Dict,
    summarizer: LLMSummarizer,
    combined_json_path: Path,
    sampling_params: SamplingParams,
    batch_size: int,
    k: Optional[int],
    max_users: Optional[int],
    max_input_tokens: int = 8000,
) -> None:
    session_data = load_json(session_file)

    all_user_ids = list(session_data.keys())
    if max_users:
        all_user_ids = all_user_ids[:max_users]

    # Resume support: reload existing results but drop error entries so they get retried
    completed = load_json_safe(combined_json_path) if combined_json_path.exists() else {}
    error_uids = {uid for uid, v in completed.items() if v.get("summary", "").startswith("Error:")}
    for uid in error_uids:
        del completed[uid]
    if error_uids:
        print(f"Retrying       : {len(error_uids)} previously failed users")

    remaining = [uid for uid in all_user_ids if uid not in completed]

    print(f"Total users   : {len(all_user_ids)}")
    print(f"Already done  : {len(all_user_ids) - len(remaining)}")
    print(f"To process    : {len(remaining)}")

    if not remaining:
        print("All users already processed.")
        return

    n_batches = (len(remaining) + batch_size - 1) // batch_size

    for batch_start in tqdm(range(0, len(remaining), batch_size),
                             total=n_batches, desc="Batches"):
        batch_user_ids = remaining[batch_start: batch_start + batch_size]

        batch_messages = []
        batch_num_used = []
        for user_id in batch_user_ids:
            item_ids = session_data[user_id]
            messages, num_used = build_messages(
                item_ids, descriptions, k,
                max_input_tokens=max_input_tokens,
                user_id=user_id,
            )
            batch_messages.append(messages)
            batch_num_used.append(num_used)

        try:
            summaries = summarizer.generate_batch(batch_messages, sampling_params)

            for user_id, summary, num_used in zip(batch_user_ids, summaries, batch_num_used):
                if not summary:
                    summary = "No descriptions available for this user."
                completed[user_id] = {
                    "num_items": len(session_data[user_id]),
                    "num_items_used": num_used,
                    "summary": summary,
                }

        except Exception as e:
            print(f"\nBatch error (batch_start={batch_start}): {e}")
            print("Falling back to single-user processing...")
            for user_id, messages, num_used in zip(batch_user_ids, batch_messages, batch_num_used):
                if user_id in completed:
                    continue
                try:
                    sums = summarizer.generate_batch([messages], sampling_params)
                    completed[user_id] = {
                        "num_items": len(session_data[user_id]),
                        "num_items_used": num_used,
                        "summary": sums[0] or "No summary generated.",
                    }
                except Exception as se:
                    print(f"  Failed user {user_id}: {se}")
                    completed[user_id] = {
                        "num_items": len(session_data[user_id]),
                        "num_items_used": 0,
                        "summary": f"Error: {se}",
                    }

        save_json(completed, combined_json_path)

    print(f"\nCombined JSON saved → {combined_json_path}")
    print(f"  Users collected : {len(completed)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-30B-A3B MSD user music taste summary generator (vLLM)"
    )

    # --- Data ---
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "valid", "test", "valid_te", "test_te"],
                        help="Dataset split to process")
    parser.add_argument("--part", type=int, default=None, choices=[0, 1, 2, 3],
                        help="Train part index (0-3); only valid when --split train. "
                             "Uses train_part{N}_numeric.json as input and "
                             "train_part{N}{k_tag}_summaries.json as output.")
    parser.add_argument("--k", type=int, default=None,
                        help="Max items per user (None = use all)")
    parser.add_argument("--msd-dir", type=str, default=str(Config.MSD_DIR))
    parser.add_argument("--output-dir", type=str, default=str(Config.OUTPUT_DIR))

    # --- Model ---
    parser.add_argument("--model-name", type=str, default=Config.MODEL_NAME)
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=Config.TENSOR_PARALLEL_SIZE)
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=Config.GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=Config.MAX_MODEL_LEN)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen3 chain-of-thought (slower)")

    # --- Generation ---
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=Config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=Config.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=Config.TOP_P)
    parser.add_argument("--top-k", type=int, default=Config.TOP_K)

    # --- Misc ---
    parser.add_argument("--max-users", type=int, default=None,
                        help="Limit number of users (for debugging)")
    parser.add_argument("--max-input-tokens", type=int, default=8000,
                        help="Max input tokens per user; overflow is resolved by random sampling")

    args = parser.parse_args()

    msd_dir    = Path(args.msd_dir)
    output_dir = Path(args.output_dir)

    if args.part is not None:
        if args.split != "train":
            raise ValueError("--part is only valid when --split train")
        session_file = msd_dir / f"train_part{args.part}_numeric.json"
        k_tag = f"_k{args.k}" if args.k else ""
        combined_json = output_dir / f"train_part{args.part}{k_tag}_summaries.json"
    else:
        session_file  = msd_dir / f"{args.split}_numeric.json"
        k_tag = f"_k{args.k}" if args.k else ""
        combined_json = output_dir / f"{args.split}{k_tag}_summaries.json"

    descriptions_path = msd_dir / "descriptions.json"

    if not session_file.exists():
        raise FileNotFoundError(f"Session file not found: {session_file}")

    print("=" * 60)
    print(f"Session file  : {session_file}")
    print(f"Descriptions  : {descriptions_path}")
    print(f"Combined JSON : {combined_json}")
    print(f"k (items/user): {args.k if args.k else 'all'}")
    print(f"Max input tok : {args.max_input_tokens}")
    print("=" * 60)

    print("Loading descriptions...")
    descriptions = load_json(descriptions_path)
    print(f"Loaded {len(descriptions)} item descriptions")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )

    summarizer = LLMSummarizer(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )

    process_split(
        session_file=session_file,
        descriptions=descriptions,
        summarizer=summarizer,
        combined_json_path=combined_json,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
        k=args.k,
        max_users=args.max_users,
        max_input_tokens=args.max_input_tokens,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
