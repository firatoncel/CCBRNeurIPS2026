"""
MSD: shorten the Music Flamingo descriptions to 2-3 sentences with Qwen via vLLM.
Reads descriptions.json, outputs summarized_descriptions.json.
"""

import os
import json
import argparse
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a concise music annotator. "
    "Summarize the given music track description into a single paragraph of 2-3 sentences. "
    "Cover only: genre, mood, tempo feel, and main instruments or vocal style. "
    "No lyrics, no duration, no structure analysis, no chord names, no production details. "
    "No bullet points, no lists, no labels. Just a plain paragraph."
)

USER_TEMPLATE = "Summarize this track description:\n\n{description}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="descriptions.json")
    parser.add_argument("--output", type=str, default="summarized_descriptions.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        descriptions = json.load(f)

    # Load existing results for resume
    done = {}
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            done = json.load(f)
        print(f"Resuming: {len(done)} already done.")

    remaining = {k: v for k, v in descriptions.items() if k not in done}
    print(f"Total: {len(descriptions)}, Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sp = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        repetition_penalty=1.1,
    )

    chat_kwargs = {
        "chat_template_kwargs": {"enable_thinking": False},
    }

    CHUNK_SIZE = 100
    keys = list(remaining.keys())
    results = dict(done)

    for i in range(0, len(keys), CHUNK_SIZE):
        chunk_keys = keys[i : i + CHUNK_SIZE]

        conversations = []
        for k in chunk_keys:
            conversations.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(description=remaining[k])},
            ])

        outputs = llm.chat(conversations, sp, **chat_kwargs)

        for k, output in zip(chunk_keys, outputs):
            results[k] = output.outputs[0].text.strip()

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Checkpoint: {len(results)}/{len(descriptions)} done.")

    print(f"Done. {len(results)} summaries saved to {args.output}")


if __name__ == "__main__":
    main()