# CCBRNeurIPS2026

Code for *Controllable and Content-Based Recommendations* (CCBR, NeurIPS 2026) on
H&M, MovieLens-20M and the Million Song Dataset (MSD).

| Folder | Contents |
|---|---|
| `autoencoder_training/` | collaborative filtering backbones (training, evaluation, tuning) |
| `experiments/HM/` | CCBR on H&M |
| `experiments/ML-20M/` | CCBR on ML-20M |
| `experiments/MSD/` | CCBR on MSD, plus LLM tag extraction and summary edits |
| `summary_generation/` | item descriptions and user summaries (VLM/LLM) |

Each `experiments/` folder contains the MF item embeddings (`mf_upgraded.py`), the
two CCBR training stages (`main_first_stage.py`, `main_ddp_*`), the TEARS baseline
(`main_tears.py`) and the controllability evaluation (`intervene_*`).

## Setup

```bash
pip install torch numpy tqdm optuna
pip install transformers sentence-transformers peft scipy matplotlib
```

## Data

The preprocessed splits and the LLM-edited summaries used in the controllability
experiments are zipped in `data/`:

```bash
cd data
unzip hm/hm.zip
cat ml-20m/ml-20m.zip.part-* > ml-20m.zip && unzip ml-20m.zip && rm ml-20m.zip
cat msd/msd.zip.part-* > msd.zip && unzip msd.zip && rm msd.zip
unzip hm/hm_interventions.zip -d hm
unzip ml-20m/ml20m_interventions.zip -d ml-20m
unzip msd/msd_interventions.zip -d msd
```

| Dataset | Items | Users (train / valid / test) |
|---|---|---|
| H&M | 1,943 | 68,201 / 1,000 / 1,000 |
| ML-20M | 11,355 | 125,269 / 5,000 / 5,000 |
| MSD | 31,046 | 156,188 / 5,000 / 5,000 |

The original datasets are subject to their own licenses.

## Backbones

Run from the repository root:

```bash
python autoencoder_training/autoencoders.py --dataset ml20m --model-type ease
python autoencoder_training/autoencoders.py --dataset msd --model-type edlae_lowrank --tune
```

| `--model-type` | Name in the paper |
|---|---|
| `ease` | EASE |
| `edlae_full` | EDLAE (FR) |
| `edlae_lowrank` | EDLAE (LR) |
| `linear` | Linear-DAE |
| `deep` | Multi-DAE |
| `mult_vae` | Multi-VAE |
| `macrid_vae` | MacridVAE |

Weights are saved to `checkpoints/<dataset>/`. `ease` and `edlae_full` save the full
item × item matrix, which `experiments/<dataset>/factorize.py` factorises by SVD
before CCBR training. `--help` lists all options.
