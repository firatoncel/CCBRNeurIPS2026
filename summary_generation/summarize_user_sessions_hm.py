"""
H&M: user summaries from the descriptions of the items they bought
(item_metadata.json), using Qwen3-30B-A3B-Instruct-2507.
"""

import json
import argparse
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
    DATA_DIR = Path(__file__).parent / "easy_split_hm"
    OUTPUT_DIR = Path(__file__).parent / "easy_split_hm_summaries"
    MAX_ITEMS_PER_USER = 100


def load_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_user_item_descriptions(user_items: List[int], item_metadata: Dict, max_items: int = 100) -> List[str]:
    descriptions, seen = [], set()
    for item_id in user_items[:max_items]:
        item_key = str(item_id)
        if item_key in item_metadata:
            item = item_metadata[item_key]
            desc = item.get("detail_desc", "")
            prod_name = item.get("prod_name", "")
            product_type = item.get("product_type_name", "")
            colour = item.get("colour_group_name", "")
            if desc and desc not in seen:
                seen.add(desc)
                descriptions.append(f"- {prod_name} ({product_type}, {colour}): {desc}")
    return descriptions


class QwenSummarizer:
    def __init__(self, model_name: str = Config.MODEL_NAME):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto", 
            trust_remote_code=True
        )
        self.model.eval()
        print("Model loaded successfully!")

    def create_prompt(self, item_descriptions: List[str]) -> str:
        items_text = "\n".join(item_descriptions)
        return f"""Below is a list of clothing items that a user has interacted with (purchased or viewed).
        Based on these items, summarize the user's fashion preferences and style in 3-4 sentences.

        Items:
        {items_text}

        Summary of user's fashion preferences:"""

    def summarize(self, item_descriptions: List[str], max_new_tokens: int = Config.MAX_NEW_TOKENS, temperature: float = Config.TEMPERATURE, top_p: float = Config.TOP_P, top_k: int = Config.TOP_K) -> str:
        if not item_descriptions:
            return "No item descriptions available for this user."

        prompt = self.create_prompt(item_descriptions)
        messages = [
            {
                "role": "user", 
                "content": prompt
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k, 
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def process_sessions(session_file: Path, item_metadata: Dict, summarizer: QwenSummarizer, output_path: Path, max_users: Optional[int] = None, max_items_per_user: int = Config.MAX_ITEMS_PER_USER, gen_kwargs: Optional[Dict] = None) -> Dict:
    session_data = load_json(session_file)
    results = {}
    user_ids = list(session_data.keys())[:max_users] if max_users else list(session_data.keys())

    print(f"\nProcessing {len(user_ids)} users from {session_file}...")
    for user_id in tqdm(user_ids, desc="Summarizing"):
        user_items = session_data[user_id]
        descriptions = get_user_item_descriptions(user_items, item_metadata, max_items=max_items_per_user)
        summary = summarizer.summarize(descriptions, **(gen_kwargs or {}))
        results[user_id] = {"num_items": len(user_items), "num_unique_descriptions": len(descriptions), "summary": summary}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Summarize user sessions using Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--session-file", type=str, default="train_numeric.json", choices=["train_numeric.json", "valid_numeric.json", "test_numeric.json", "valid_te_numeric.json", "test_te_numeric.json"], help="Session file to process")
    parser.add_argument("--data-dir", type=str, default=str(Config.DATA_DIR), help="Directory containing data files")
    parser.add_argument("--model-name", type=str, default=Config.MODEL_NAME, help="Hugging Face model name")
    parser.add_argument("--max-new-tokens", type=int, default=Config.MAX_NEW_TOKENS, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=Config.TEMPERATURE, help="Sampling temperature (0.7 recommended)")
    parser.add_argument("--top-p", type=float, default=Config.TOP_P, help="Top-p sampling (0.8 recommended)")
    parser.add_argument("--top-k", type=int, default=Config.TOP_K, help="Top-k sampling (20 recommended)")
    parser.add_argument("--max-users", type=int, default=None, help="Maximum number of users to process")
    parser.add_argument("--max-items-per-user", type=int, default=Config.MAX_ITEMS_PER_USER, help="Maximum items per user")
    parser.add_argument("--output-dir", type=str, default=str(Config.OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    item_metadata = load_json(data_dir / "item_metadata.json")
    summarizer = QwenSummarizer(model_name=args.model_name)

    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                  "top_p": args.top_p, "top_k": args.top_k}

    session_path = data_dir / args.session_file
    output_path = output_dir / args.session_file.replace(".json", "_summaries.json")
    results = process_sessions(session_path, item_metadata, summarizer, output_path, args.max_users, args.max_items_per_user, gen_kwargs)

    print(f"\n{'='*50}")
    print(f"Processing complete! Total users: {len(results)}, Output: {output_path}")


if __name__ == "__main__":
    main()
