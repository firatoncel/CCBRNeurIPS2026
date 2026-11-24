# Data

Preprocessed splits, user summaries and summary edits for H&M, ML-20M and MSD.
See the main README for how to extract the archives. After extraction each
dataset folder (`hm/`, `ml-20m/`, `msd/`) contains:

## Common files

| File | Contents |
|---|---|
| `num_unique_items.txt` | number of items |
| `train_numeric.json` | `{user: [item, ...]}` interactions of the training users |
| `valid_numeric.json`, `test_numeric.json` | fold-in interactions of the validation / test users |
| `valid_te_numeric.json`, `test_te_numeric.json` | held-out items of the validation / test users (evaluation targets) |
| `*_numeric_summaries.json` | LLM-written user summaries for each split (see `summary_generation/`) |
| `item_embeddings_constrained.npy` | item embeddings from `experiments/<dataset>/mf_upgraded.py` |

Users and items are 0-based integer indices.

## H&M (`hm/`)

| File | Contents |
|---|---|
| `item_metadata.json` | article metadata (product type, colour, department, ...) |
| `item_to_product_type.json`, `item_color.json` | product type and colour of each item |
| `train_numeric_augmented.json`, `train_numeric_summaries_augmented.json` | augmented training users and summaries (`--augment`) |
| `loo_{category,color}.json` | leave-one-out summary edits (remove a product type / colour) |
| `loi_{category,color}.json` | leave-one-in summary edits (add a product type / colour) |
| `loi_grounded_{category,color}.json` | grounded leave-one-in edits |

## ML-20M (`ml-20m/`)

| File | Contents |
|---|---|
| `descriptions.json` | item descriptions generated from the trailers |
| `tag_matrix.npy` | items × genres matrix |
| `tag2idx.json`, `idx2tag.json`, `genre_mapping.json` | genre names and indices |
| `item_embeddings_unconstrained.npy` | alternative MF item embeddings (`main_first_stage.py --embeddings-name unconstrained`) |
| `mf_training_history.json` | training log written by `mf_upgraded.py` |
| `leave_one_out_interventions.json`, `leave_one_in_interventions.json` | leave-one-out / leave-one-in genre edits |
| `leave_one_in_grounded.json` | grounded leave-one-in edits |
| `leave_one_out_negated.json` | leave-one-out edits that state the user avoids the genre |

## MSD (`msd/`)

| File | Contents |
|---|---|
| `descriptions.json` | item descriptions generated from the audio |
| `tag_matrix.npy` | items × tags matrix |
| `tag2idx.json`, `idx2tag.json` | tag names and indices |
| `valid_indices.json` | tags of each item, as indices and names |
| `unique_sid.txt`, `unique_uid.txt` | original MSD song and user IDs, one per line in index order |
| `leave_one_out_interventions.json`, `leave_one_in_interventions.json` | leave-one-out / leave-one-in tag edits |
| `leave_one_in_grounded.json` | grounded leave-one-in edits |

These files are derived from the original datasets and remain subject to their
licenses; the MIT license of this repository covers the code only.
