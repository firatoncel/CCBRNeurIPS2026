# Summary generation

Scripts that produced the item descriptions and user summaries in `data/`.

| Dataset | Item descriptions | User summaries |
|---|---|---|
| H&M | `extract_descriptions_qwen_vl.py` (Qwen3-VL-32B, product images) | `summarize_user_sessions_hm.py` |
| ML-20M | `extract_descriptions_videollama3.py` (VideoLLaMA3-7B, trailers) | `summarize_user_sessions_ml20m.py` |
| MSD | `create_descriptions_musicflamingo.py` (Music Flamingo, audio), shortened by `create_descriptions_musicflamingo_summarizer.py` | `summarize_user_sessions_msd.py` |

User summaries are written by Qwen3-30B-A3B-Instruct-2507. The MSD scripts and
`extract_descriptions_qwen_vl.py` run on `vllm`, the others on `transformers`.
