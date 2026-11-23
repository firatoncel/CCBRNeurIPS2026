"""
User Session Summarization for ML-20M using VLM-generated movie descriptions.
Uses interactions/descriptions.json (VLM descriptions, numeric IDs)
and session files from interactions/.
"""

import json
import argparse
import re
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Config:
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.8
    TOP_K = 20
    MAX_INPUT_LENGTH = 16384
    SESSION_DIR = Path(__file__).parent / "interactions"
    ITEM_DESCRIPTIONS_PATH = Path(__file__).parent / "interactions" / "descriptions.json"
    OUTPUT_DIR = Path(__file__).parent / "ml20m_session_summaries"
    MAX_ITEMS_PER_USER = 100
    BATCH_SIZE = 8
    CHECKPOINT_INTERVAL = 160


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def strip_think_tokens(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_user_movie_descriptions(
    user_items: List[int], item_descriptions: Dict, max_items: int = 50
) -> List[str]:
    descriptions = []
    for item_id in user_items[:max_items]:
        desc = item_descriptions.get(str(item_id), "").strip()
        if desc:
            descriptions.append(f"- {desc}")
    return descriptions


class QwenBatchSummarizer:
    def __init__(self, model_name: str = Config.MODEL_NAME):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model loaded successfully!")

    def create_prompt(self, movie_descriptions: List[str]) -> str:
        items_text = "\n".join(movie_descriptions)
        return (
            "Below is a list of movies that a user has watched, described by their "
            "genre, atmosphere, and narrative style.\n"
            "Based on these movies, summarize the user's movie preferences and taste "
            "in 3-4 sentences. Focus on recurring genres, tones, and themes.\n\n"
            f"Movies:\n{items_text}\n\n"
            "Summary of user's movie preferences:"
        )

    def prepare_batch_inputs(self, batch_descriptions: List[List[str]]) -> Dict:
        prompts = []
        for descriptions in batch_descriptions:
            if not descriptions:
                prompts.append(
                    "No movie descriptions available. Please provide a brief default response."
                )
            else:
                prompts.append(self.create_prompt(descriptions))

        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in messages_list
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_INPUT_LENGTH,
        ).to(self.model.device)

        return model_inputs

    def summarize_batch(
        self,
        batch_descriptions: List[List[str]],
        max_new_tokens: int = Config.MAX_NEW_TOKENS,
        temperature: float = Config.TEMPERATURE,
        top_p: float = Config.TOP_P,
        top_k: int = Config.TOP_K,
    ) -> List[str]:
        if not batch_descriptions:
            return []

        model_inputs = self.prepare_batch_inputs(batch_descriptions)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = model_inputs["input_ids"].shape[1]
        gen_only = generated_ids[:, prompt_len:]

        summaries = []
        for gen_ids in gen_only:
            output_ids = gen_ids.tolist()
            if self.tokenizer.eos_token_id in output_ids:
                output_ids = output_ids[: output_ids.index(self.tokenizer.eos_token_id)]
            summary = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            summary = strip_think_tokens(summary)
            summaries.append(summary or "No summary generated for this user.")

        return summaries

    def summarize_single(
        self,
        movie_descriptions: List[str],
        max_new_tokens: int = Config.MAX_NEW_TOKENS,
        temperature: float = Config.TEMPERATURE,
        top_p: float = Config.TOP_P,
        top_k: int = Config.TOP_K,
    ) -> str:
        results = self.summarize_batch(
            [movie_descriptions], max_new_tokens, temperature, top_p, top_k
        )
        return results[0] if results else "No summary generated."


def process_sessions(
    session_file: Path,
    item_descriptions: Dict,
    summarizer: QwenBatchSummarizer,
    output_path: Path,
    batch_size: int = Config.BATCH_SIZE,
    max_users: Optional[int] = None,
    max_items_per_user: int = Config.MAX_ITEMS_PER_USER,
    checkpoint_interval: int = Config.CHECKPOINT_INTERVAL,
    resume: bool = True,
    part: int = 0,
    num_parts: int = 1,
    gen_kwargs: Optional[Dict] = None,
) -> Dict:
    session_data = load_json(session_file)

    all_ids = list(session_data.keys())
    if max_users:
        all_ids = all_ids[:max_users]

    # Split into num_parts and pick the requested part
    chunk = len(all_ids) // num_parts
    start = part * chunk
    end = start + chunk if part < num_parts - 1 else len(all_ids)
    all_user_ids = all_ids[start:end]

    print(f"\nPart {part}/{num_parts-1}: users {start}~{end-1} ({len(all_user_ids)} users)")

    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    results = load_json(checkpoint_path) if (resume and checkpoint_path.exists()) else {}
    processed_users = set(results.keys())

    remaining = [uid for uid in all_user_ids if uid not in processed_users]

    print(f"Total users: {len(all_user_ids)}")
    print(f"Already processed: {len(processed_users)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All users already processed!")
        save_json(results, output_path)
        return results

    gen_kwargs = gen_kwargs or {}
    num_batches = (len(remaining) + batch_size - 1) // batch_size
    processed_since_ckpt = 0

    with tqdm(total=len(all_user_ids), initial=len(processed_users), desc="Summarizing") as pbar:
        for batch_idx in range(num_batches):
            batch_ids = remaining[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            batch_descriptions = []
            batch_meta = []
            for uid in batch_ids:
                items = session_data[uid]
                descs = get_user_movie_descriptions(
                    items, item_descriptions, max_items=max_items_per_user
                )
                batch_descriptions.append(descs)
                batch_meta.append({"user_id": uid, "num_items": len(items)})

            try:
                summaries = summarizer.summarize_batch(batch_descriptions, **gen_kwargs)
                for meta, summary in zip(batch_meta, summaries):
                    results[meta["user_id"]] = {
                        "num_items": meta["num_items"],
                        "summary": summary,
                    }
                processed_since_ckpt += len(batch_ids)
                pbar.update(len(batch_ids))

                if processed_since_ckpt >= checkpoint_interval:
                    save_json(results, checkpoint_path)
                    processed_since_ckpt = 0

            except Exception as e:
                print(f"\nBatch {batch_idx} failed: {e}. Falling back to single-item processing...")
                save_json(results, checkpoint_path)

                for meta, descs in zip(batch_meta, batch_descriptions):
                    if meta["user_id"] not in results:
                        try:
                            summary = summarizer.summarize_single(descs, **gen_kwargs)
                        except Exception as single_err:
                            summary = f"Error: {single_err}"
                        results[meta["user_id"]] = {
                            "num_items": meta["num_items"],
                            "summary": summary,
                        }
                        pbar.update(1)

    save_json(results, output_path)
    print(f"Results saved to {output_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint removed: {checkpoint_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ML-20M user sessions using VLM-generated movie descriptions"
    )
    parser.add_argument(
        "--session-file",
        type=str,
        default="train_numeric.json",
        choices=[
            "train_numeric.json",
            "valid_numeric.json",
            "test_numeric.json",
            "valid_te_numeric.json",
            "test_te_numeric.json",
        ],
        help="Session file to process (from interactions/)",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=str(Config.SESSION_DIR),
        help="Directory containing session files",
    )
    parser.add_argument(
        "--item-descriptions",
        type=str,
        default=str(Config.ITEM_DESCRIPTIONS_PATH),
        help="Path to VLM-generated descriptions.json",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=Config.MODEL_NAME,
        help="Hugging Face model name or local path",
    )
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=Config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=Config.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=Config.TOP_P)
    parser.add_argument("--top-k", type=int, default=Config.TOP_K)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-items-per-user", type=int, default=Config.MAX_ITEMS_PER_USER)
    parser.add_argument(
        "--output-dir", type=str, default=str(Config.OUTPUT_DIR), help="Output directory"
    )
    parser.add_argument("--checkpoint-interval", type=int, default=Config.CHECKPOINT_INTERVAL)
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoint)")
    parser.add_argument(
        "--part", type=int, default=0,
        help="Part index to process (0-based). Use with --num-parts.",
    )
    parser.add_argument(
        "--num-parts", type=int, default=1,
        help="Total number of parts to split the session file into.",
    )
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    output_dir = Path(args.output_dir)
    item_descriptions_path = Path(args.item_descriptions)

    print("Loading VLM movie descriptions...")
    item_descriptions = load_json(item_descriptions_path)
    print(f"Loaded {len(item_descriptions)} movie descriptions from {item_descriptions_path}")

    summarizer = QwenBatchSummarizer(model_name=args.model_name)

    session_path = session_dir / args.session_file
    stem = args.session_file.replace(".json", "")
    if args.num_parts > 1:
        output_path = output_dir / f"{stem}_part{args.part}_summaries.json"
    else:
        output_path = output_dir / f"{stem}_summaries.json"

    results = process_sessions(
        session_file=session_path,
        item_descriptions=item_descriptions,
        summarizer=summarizer,
        output_path=output_path,
        batch_size=args.batch_size,
        max_users=args.max_users,
        max_items_per_user=args.max_items_per_user,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        part=args.part,
        num_parts=args.num_parts,
        gen_kwargs={"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                    "top_p": args.top_p, "top_k": args.top_k},
    )

    print(f"\n{'='*50}")
    print(f"Done! Total users processed: {len(results)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
