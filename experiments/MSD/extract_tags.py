"""
Extract music tag mentions from user listening summaries using Qwen.
Uses HuggingFace Transformers for local inference.
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ─── Tag Mapping ────────────────────────────────────────────────────────────

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

TAG_ALIASES = {
    # Genre aliases
    "r&b": "rnb",
    "r and b": "rnb",
    "rhythm and blues": "rnb",
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "prog rock": "progressive rock",
    "progressive": "progressive rock",
    "edm": "electronic",
    "lo-fi": "chill",
    "lofi": "chill",
    "chillhop": "chillout",
    "singer/songwriter": "singer-songwriter",
    "thrash metal": "heavy metal",
    "black metal": "metal",
    "post-rock": "experimental",
    "synthwave": "electronic",
    "synth": "electro",
    "grunge": "alternative rock",
    "shoegaze": "alternative rock",
    "dreampop": "indie pop",
    "dream pop": "indie pop",
    "garage rock": "alternative rock",
    "power pop": "pop rock",
    "nu metal": "metal",
    "nu-metal": "metal",
    "pop punk": "punk rock",
    "post-hardcore": "hardcore",
    "screamo": "emo",
    "gothic": "post-punk",
    "goth": "post-punk",
    "industrial": "electronic",
    "disco": "dance",
    "deep house": "house",
    "tech house": "house",
    "progressive house": "house",
    "drum and bass": "electronic",
    "dubstep": "electronic",
    "glitch": "experimental",
    "idm": "electronic",
    "breakbeat": "electronic",
    "new age": "ambient",
    "dark ambient": "ambient",
    "noise": "experimental",
    "minimal": "ambient",
    "bossa nova": "jazz",
    "latin jazz": "jazz",
    "swing": "jazz",
    "bebop": "jazz",
    "smooth jazz": "jazz",
    "acid jazz": "jazz",
    "trip-hop": "downtempo",
    "trip hop": "downtempo",
    "ska": "reggae",
    "baroque pop": "indie pop",
    "chamber pop": "indie pop",
    "bluegrass": "folk",
    "celtic": "folk",
    "gospel": "soul",
    "classical": "classic",

    # Mood aliases (only obvious ones)
    "atmospheric": "ambient",
    "dreamy": "ambient",
    "meditative": "ambient",
    "cinematic": "epic",
    "groovy": "funky",

    # Instrument aliases (only obvious ones)
    "orchestral": "instrumental",
    "synthesizer": "electronic",
    "synths": "electronic",
    "harmonica": "blues",
}

# ─── Model setup ────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-27B"


def load_model(device="cuda"):
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


# ─── Prompt ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise music attribute extractor. Given a user listening summary, identify which music tags are mentioned or clearly implied.

IMPORTANT RULES:
- Only extract tags that are explicitly mentioned or very strongly implied by the summary.
- Map related terms to the closest valid tag using these rules:
  - "R&B" or "rhythm and blues" → rnb
  - "hip-hop" → hip hop
  - "prog rock" or "progressive" (in rock context) → progressive rock
  - "EDM" or "dance music" → dance, electronic
  - "lo-fi" or "chillhop" → chill, chillout
  - "singer/songwriter" → singer-songwriter
  - "thrash metal" or "black metal" → metal, heavy metal
  - "post-rock" → experimental
  - "synthwave" or "synth" → electronic, electro
  - "bossa nova" or "latin jazz" → jazz, world
  - "grunge" → alternative rock
  - "ska" → punk, reggae
  - "trip-hop" → downtempo, electronic
  - "baroque pop" or "chamber pop" → indie pop
  - "shoegaze" → alternative rock, experimental
  - "dreampop" → indie pop, ambient
  - "garage rock" → punk, alternative rock
  - "power pop" → pop rock
  - "nu metal" → metal, hard rock
  - "female singer" or "woman vocalist" → female vocalist, female vocalists
  - "male singer" or "man vocalist" → male vocalist, male vocalists
  - "relaxed" or "soothing" or "calm" → relax, relaxing, mellow
  - "energetic" or "uplifting" → upbeat, fun
  - "dark" or "moody" → melancholic, melancholy
  - "retro" → oldies
  - "groove" or "groovy" → funky, funk
- Do NOT hallucinate tags not present or implied in the summary.

VALID TAGS:
{tags}

Respond ONLY with valid JSON, no other text:
{{"tags": ["tag1", "tag2", ...]}}""".format(
    tags=", ".join(TAG_LIST),
)


def build_messages(summary_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract music tags from this user listening summary:\n\n{summary_text}"},
    ]


# ─── Extraction ─────────────────────────────────────────────────────────────

def extract_json_from_response(text: str) -> dict | None:
    """Robustly parse JSON from model output, handling markdown fences and thinking tags."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[^{}]*\"tags\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    text = text.strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_from_summary(
    summary_text: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> dict:
    """Run extraction for a single summary. Returns {tag_indices: [...], tag_names: [...]}."""
    messages = build_messages(summary_text)
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    parsed = extract_json_from_response(raw_output)

    if parsed is None:
        print(f"  [WARN] Failed to parse JSON from model output:\n  {raw_output[:300]}")
        return {"tag_indices": [], "tag_names": [], "raw": raw_output}

    tag_indices = []
    for t in parsed.get("tags", []):
        t_lower = t.strip().lower()
        if t_lower in TAG_ALIASES:
            t_lower = TAG_ALIASES[t_lower]
        if t_lower in TAG_MAP:
            tag_indices.append(TAG_MAP[t_lower])
        else:
            print(f"  [WARN] Unknown tag '{t.strip()}', skipping.")

    tag_indices = sorted(set(tag_indices))
    tag_names = [TAG_LIST[idx] for idx in tag_indices]

    return {
        "tag_indices": tag_indices,
        "tag_names": tag_names,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def run_extraction(summaries_path: str, output_path: str, device: str = "cuda"):
    with open(summaries_path, "r") as f:
        summaries = json.load(f)

    model, tokenizer = load_model(device=device)

    results = {}
    for user_id, entry in tqdm(summaries.items(), desc="Extracting"):
        summary_text = entry["summary"]
        result = extract_from_summary(summary_text, model, tokenizer)
        results[user_id] = result
        tqdm.write(f"  [{user_id}] Tags: {result['tag_names']}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to summaries JSON")
    parser.add_argument("--output", type=str, default="tag_extraction_results.json", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    run_extraction(args.input, args.output, args.device)