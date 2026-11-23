# CCBRNeurIPS2026

Collaborative filtering backbones used in *Controllable and Content-Based
Recommendations* (CCBR, NeurIPS 2026), on H&M, MovieLens-20M and the Million Song
Dataset (MSD).

| File | Contents |
|---|---|
| `autoencoder_training/autoencoders.py` | models, training, evaluation and hyperparameter tuning |
| `autoencoder_training/utils.py` | data loading, datasets and ranking metrics (NDCG@100, Recall@20/50) |
| `experiments/ML-20M/` | CCBR on ML-20M: MF, two-stage training, TEARS baseline, controllability evaluation |

## Setup

Python 3.9+ with PyTorch, NumPy and tqdm (`optuna` is only needed for `--tune`).
The experiments in `experiments/` also need `transformers`, `sentence-transformers`,
`peft`, `scipy` and `matplotlib`.

```bash
pip install torch numpy tqdm optuna
pip install transformers sentence-transformers peft scipy matplotlib
```

## Data

The preprocessed splits are stored as zip archives in `data/`. Extract them in place:

```bash
cd data
unzip hm/hm.zip
cat ml-20m/ml-20m.zip.part-* > ml-20m.zip && unzip ml-20m.zip && rm ml-20m.zip
cat msd/msd.zip.part-* > msd.zip && unzip msd.zip && rm msd.zip
unzip ml-20m/ml20m_interventions.zip -d ml-20m
```

`ml20m_interventions.zip` holds the LLM-edited summaries of the 5,000 ML-20M
validation users used in the controllability experiments: genre removal
(`leave_one_out_interventions.json`), addition (`leave_one_in_interventions.json`),
addition through unseen items (`leave_one_in_grounded.json`) and negation
(`leave_one_out_negated.json`).

| Dataset | Items | Users (train / valid / test) |
|---|---|---|
| H&M | 1,943 | 68,201 / 1,000 / 1,000 |
| ML-20M | 11,355 | 125,269 / 5,000 / 5,000 |
| MSD | 31,046 | 156,188 / 5,000 / 5,000 |

Each of `data/hm`, `data/ml-20m` and `data/msd` then contains `train_numeric.json`,
`valid_numeric.json`, `valid_te_numeric.json`, `test_numeric.json`,
`test_te_numeric.json` and `num_unique_items.txt`. Users are split disjointly
(strong generalization): validation and test users are evaluated on their held-out
items (`*_te_numeric.json`) given their remaining interactions. The original
datasets are subject to their own licenses.

## Usage

Run from the repository root (the default data and checkpoint paths are relative to it):

```bash
python autoencoder_training/autoencoders.py --dataset ml20m --model-type ease
python autoencoder_training/autoencoders.py --dataset hm --model-type mult_vae --epochs 50
python autoencoder_training/autoencoders.py --dataset msd --model-type edlae_lowrank --tune
```

| `--model-type` | Name in the paper | Reference |
|---|---|---|
| `ease` | EASE | Steck, 2019 |
| `edlae_full` | EDLAE (FR) | Steck, 2020 |
| `edlae_lowrank` | EDLAE (LR) | Steck, 2020 |
| `linear` | Linear-DAE | Steck, 2020 |
| `deep` | Multi-DAE | Liang et al., 2018 |
| `mult_vae` | Multi-VAE | Liang et al., 2018 |
| `macrid_vae` | MacridVAE | Ma et al., 2019 |

- `--tune` tunes on the validation set (Optuna, `--tune-trials`; a grid search for
  EASE), then retrains with the best parameters and reports validation and test
  metrics.
- The latent size is set with `--latent-dim` (default 1024; 4096 on MSD).
  The latent code of `linear` and `deep` is L2-normalised (`--no-normalize` turns
  this off); the VAE latents are not.
- Encoder/decoder weights are saved to `checkpoints/<dataset>/enc_<model>.pt` and
  `dec_<model>.pt`; `--results-csv results.csv` also appends the metrics to a CSV file.
- `ease` and `edlae_full` save the full items × items matrix in both files; it is
  factorised by SVD into an encoder/decoder pair (1024-d, 4096-d on MSD) in the
  CCBR alignment step.
- Analytic models (`ease`, `edlae_*`) work on the item-item Gram matrix XᵀX, which
  is computed at start-up or loaded from `<data-path>/gram.pt` if present. The
  full-rank models invert an items × items matrix, so MSD needs a GPU with enough
  memory.
- Per-dataset defaults and search spaces are defined in `DATASETS` at the top of
  `autoencoders.py`; `python autoencoder_training/autoencoders.py --help` lists
  every option.
