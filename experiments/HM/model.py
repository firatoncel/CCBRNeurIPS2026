import torch
import os
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# Pooling
# ============================================================
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


def pool(last_hidden, model_name, attention_mask=None):
    if model_name.startswith('Qwen'):
        return last_token_pool(last_hidden, attention_mask)
    return last_hidden[:, 0, :]  # CLS token


def mean_clicked_item_emb(x, table, eps=1e-8):
    x = x.to(table.dtype)
    clicked_sum = x @ table
    denom = x.sum(dim=1, keepdim=True)
    return clicked_sum / (denom.clamp_min(1.0) + eps)


def aggregate_by_type(interactions, labels, n_types, normalize=False):
    """(B, n_items) interactions -> (B, n_types) attribute counts
    (presence indicators if normalize=True)."""
    one_hot = torch.zeros(len(labels), n_types).to(interactions.device)
    one_hot[torch.arange(len(labels)), labels] = 1  # (n_items, n_types)
    if normalize:
        return (interactions @ one_hot).bool().float()
    return (interactions @ one_hot).float()


# ============================================================
# Classifier (product-type and colour distributions from text)
# ============================================================
class Classifier(nn.Module):
    def __init__(self, item_emb_path, path, d_latent=1024, model_name="BAAI/bge-m3",
                 replace_cls=False, device="cuda:0", normalize=True, tau=1):
        super().__init__()
        self.device = device
        self.text_encoder = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        if len(self.text_encoder) > 2 and not isinstance(self.text_encoder[2], nn.Identity):
            self.text_encoder[2] = nn.Identity()
        print(self.text_encoder)

        self.replace_cls = replace_cls
        self.normalize = normalize
        self.tau = tau

        # Item -> product type / colour (path = data folder)
        with open(f"{path}/item_to_product_type.json") as f:
            product_types = json.load(f)
        with open(f"{path}/item_color.json") as f:
            colors = json.load(f)

        unique_types = sorted(set(product_types.values()))
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        self.n_types = len(unique_types)
        unique_colors = sorted(set(colors.values()))
        type_to_idx_color = {t: i for i, t in enumerate(unique_colors)}
        self.n_types_color = len(type_to_idx_color)
        print(f"Product types: {self.n_types} | Colors: {self.n_types_color}")
        self.labels = torch.from_numpy(np.array([type_to_idx[item] for item in product_types.values()])).to(device)
        self.labels_color = torch.from_numpy(np.array([type_to_idx_color[item] for item in colors.values()])).to(device)

        # One linear head per attribute
        self.linear = nn.Linear(d_latent, self.n_types)
        self.linear_color = nn.Linear(d_latent, self.n_types_color)
        self.item_table = torch.from_numpy(np.load(item_emb_path)).float().to(device)
        self.color_dict = type_to_idx_color
        self.product_dict = type_to_idx

    def forward(self, x, summary, p=0.0):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - p))
            x = x * mask

        features = self.text_encoder.tokenize(summary, max_length=1024)
        features = {k: v.to(self.device) for k, v in features.items()}

        input_ids = features["input_ids"]
        attn_mask = features["attention_mask"]
        inputs_embeds = self.text_encoder[0].auto_model.get_input_embeddings()(input_ids)
        if self.replace_cls:
            item_embeddings = mean_clicked_item_emb(x, self.item_table)
            if self.training:
                cls_mask = torch.rand(x.size(0), device=x.device) > 0.5
                inputs_embeds[:, 0] = torch.where(
                    cls_mask.unsqueeze(-1), item_embeddings, inputs_embeds[:, 0]
                )
            else:
                inputs_embeds[:, 0] = item_embeddings

        outputs = self.text_encoder[0].auto_model(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask
        )
        # Target attribute counts of the (dropout-masked) interactions
        y_aux_true = aggregate_by_type(x, self.labels, self.n_types, self.normalize)
        y_aux_true_color = aggregate_by_type(x, self.labels_color, self.n_types_color, self.normalize)

        pooled_output = pool(outputs.last_hidden_state, self.model_name, attn_mask)
        z_text = F.normalize(pooled_output, dim=-1)

        y_aux_pred = self.linear(z_text) / self.tau
        y_aux_pred_color = self.linear_color(z_text) / self.tau

        return y_aux_true, y_aux_true_color, y_aux_pred, y_aux_pred_color, z_text
