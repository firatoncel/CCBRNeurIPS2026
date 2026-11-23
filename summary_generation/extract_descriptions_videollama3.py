"""
Movie Trailer Item Description Generation for ML-20M using VideoLLaMA3-7B
- Processes mp4 trailer files in a given directory
- Generates per-movie text descriptions (genre, themes, mood, plot)
- Saves {movieId: description} JSON
- Supports resume (skips already-completed movies)
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import argparse
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA3-7B"
    TRAILER_DIR = Path(__file__).parent / "trailers"
    OUTPUT_PATH = Path(__file__).parent / "movie_descriptions.json"
    FPS = 1
    MAX_FRAMES = 180      # keep memory manageable; trailers ~90–170 s → 90–170 frames at 1fps
    MAX_NEW_TOKENS = 256
    SAVE_INTERVAL = 10    # checkpoint every N movies

PROMPT_TEMPLATE = (
      "You are analyzing a movie trailer for a recommendation system. "
      "Based only on the visual content of this trailer, write 2-3 sentences describing "
      "the genre, atmosphere, target audience, and narrative style. "
      "Do not use any external knowledge about this film."
)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json_safe(path: Path) -> Dict:
    """Load JSON with truncation recovery."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[WARN] Corrupted JSON: {path}. Attempting recovery...")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        idx = content.rfind('",\n  "')
        if idx > 0:
            # Cut after the last complete key-value pair
            val_end = content.rfind('"', 0, idx)
            candidate = content[:val_end + 1] + "\n}"
            try:
                data = json.loads(candidate)
                print(f"[WARN] Recovered {len(data)} entries.")
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
# Model
# ---------------------------------------------------------------------------

class VideoLLaMA3Summarizer:
    def __init__(self, model_path: str):
        print(f"Loading VideoLLaMA3 from: {model_path}")
        # Workaround: ALL_PARALLEL_STYLES is None when torch<2.5 or distributed
        # is unavailable, causing post_init to crash with "NoneType is not iterable".
        # Clearing base_model_tp_plan skips the broken validation.
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.base_model_tp_plan = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("Model loaded successfully!")

    @torch.inference_mode()
    def describe(
        self,
        video_path: str,
        fps: int,
        max_frames: int,
        max_new_tokens: int,
    ) -> str:
        conversation = [
            {"role": "system", "content": "You are a helpful movie description assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": fps,
                            "max_frames": max_frames,
                        },
                    },
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            },
        ]
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate movie descriptions from trailers using VideoLLaMA3-7B"
    )
    parser.add_argument("--trailer-dir", type=str, default=str(Config.TRAILER_DIR),
                        help="Directory containing {movieId}.mp4 files")
    parser.add_argument("--output-path", type=str, default=str(Config.OUTPUT_PATH),
                        help="Output JSON path: {movieId: description}")
    parser.add_argument("--model-path", type=str, default=Config.MODEL_PATH)
    parser.add_argument("--fps", type=int, default=Config.FPS)
    parser.add_argument("--max-frames", type=int, default=Config.MAX_FRAMES)
    parser.add_argument("--max-new-tokens", type=int, default=Config.MAX_NEW_TOKENS)
    parser.add_argument("--save-interval", type=int, default=Config.SAVE_INTERVAL)
    parser.add_argument("--max-movies", type=int, default=None,
                        help="Limit number of movies (for debugging)")
    args = parser.parse_args()

    trailer_dir = Path(args.trailer_dir)
    output_path = Path(args.output_path)

    # --- Discover mp4 files ---
    mp4_files = sorted(trailer_dir.glob("*.mp4"), key=lambda p: int(p.stem))
    if args.max_movies:
        mp4_files = mp4_files[: args.max_movies]
    print(f"  {len(mp4_files)} mp4 files found in {trailer_dir}")

    # --- Resume ---
    completed: Dict[str, str] = load_json_safe(output_path) if output_path.exists() else {}
    # Retry error entries
    error_ids = [mid for mid, desc in completed.items() if desc.startswith("Error:")]
    for mid in error_ids:
        del completed[mid]
    if error_ids:
        print(f"  Retrying {len(error_ids)} previously failed movies")

    remaining = [p for p in mp4_files if p.stem not in completed]
    print(f"  Already done : {len(mp4_files) - len(remaining)}")
    print(f"  To process   : {len(remaining)}")

    if not remaining:
        print("All movies already processed.")
        return

    # --- Load model ---
    summarizer = VideoLLaMA3Summarizer(model_path=args.model_path)

    # --- Process ---
    for i, mp4_path in enumerate(tqdm(remaining, desc="Movies")):
        movie_id = mp4_path.stem  # e.g. "1"

        try:
            description = summarizer.describe(
                video_path=str(mp4_path),
                fps=args.fps,
                max_frames=args.max_frames,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"\n[ERROR] movieId={movie_id}: {e}")
            description = f"Error: {e}"

        completed[movie_id] = description

        if (i + 1) % args.save_interval == 0:
            save_json(completed, output_path)
            print(f"  Checkpoint saved ({len(completed)} movies)")

    # Final save
    save_json(completed, output_path)
    print(f"\nDone! Descriptions saved to {output_path}")
    print(f"  Total movies described: {len(completed)}")


if __name__ == "__main__":
    main()
