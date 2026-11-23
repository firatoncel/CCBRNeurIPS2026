"""
MSD: music description generation with Music Flamingo + vLLM.
Processes numbered wav files (0.wav, 1.wav, ...) and writes {item_id: description}
to JSON; the descriptions are then shortened by create_descriptions_musicflamingo_summarizer.py.
"""

import os
import json
import glob
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams


PROMPT = (
    "In exactly 2-3 sentences, briefly describe this music track's "
    "genre, mood, tempo, and main instruments or vocal style. "
    "Do not analyze structure, lyrics, or production details."
)


def get_track_paths(tracks_dir: str) -> list[tuple[str, str]]:
    """Find all wav files and return sorted (filename_stem, path) pairs."""
    files = glob.glob(os.path.join(tracks_dir, "*.wav"))
    pairs = [(Path(f).stem, str(Path(f).resolve())) for f in files]
    pairs.sort(key=lambda x: x[0])
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks_dir", type=str, default="unused_wavs")
    parser.add_argument("--output", type=str, default="descriptions.json")
    parser.add_argument("--model", type=str, default="nvidia/music-flamingo-hf")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # Load existing results for resume
    done = {}
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            done = json.load(f)
        print(f"Resuming: {len(done)} tracks already done.")

    # Gather and filter tracks
    tracks = get_track_paths(args.tracks_dir)
    print(f"Found {len(tracks)} total tracks.")
    tracks = [(tid, tp) for tid, tp in tracks if str(tid) not in done]
    print(f"Remaining: {len(tracks)} to process.")

    if not tracks:
        print("Nothing to do.")
        return

    # Init vLLM
    llm = LLM(
        model=args.model,
        allowed_local_media_path=str(Path(args.tracks_dir).resolve()),
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        load_format="safetensors",
        ignore_patterns=["*.bin"],
    )

    sp = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        repetition_penalty=1.2,
    )

    # Process in chunks — checkpoint after each to survive crashes
    CHUNK_SIZE = 100
    results = dict(done)

    for i in range(0, len(tracks), CHUNK_SIZE):
        chunk = tracks[i : i + CHUNK_SIZE]

        conversations = []
        chunk_ids = []
        for track_id, track_path in chunk:
            conversations.append([{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "audio_url", "audio_url": {"url": Path(track_path).as_uri()}},
                ],
            }])
            chunk_ids.append(track_id)

        # vLLM continuous batching handles scheduling within the chunk
        outputs = llm.chat(conversations, sp)

        for track_id, output in zip(chunk_ids, outputs):
            results[str(track_id)] = output.outputs[0].text.strip()

        # Checkpoint
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Checkpoint: {len(results)}/{len(tracks) + len(done)} done.")

    print(f"Done. {len(results)} descriptions saved to {args.output}")


if __name__ == "__main__":
    main()