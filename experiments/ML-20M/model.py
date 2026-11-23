import torch
import os
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model
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


# ============================================================
# Classifier (tag distribution prediction from text)
# ============================================================
class Classifier(nn.Module):
    def __init__(self, item_emb_path, d_latent=1024,
                 model_name="BAAI/bge-m3", replace_cls=False, device="cuda:0",
                 normalize=True, lora=False, lora_r=8, lora_alpha=16,
                 tag_matrix_path="tag_matrix.npy", tag2idx_path="tag2idx.json", tau=1):
        super().__init__()
        self.device = device
        self.text_encoder = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        if len(self.text_encoder) > 2 and not isinstance(self.text_encoder[2], nn.Identity):
            self.text_encoder[2] = nn.Identity()
        print(self.text_encoder)

        self.lora = lora
        if self.lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "key", "value", "dense"],
                lora_dropout=0.1,
                bias="none",
            )
            self.text_encoder[0].auto_model = get_peft_model(
                self.text_encoder[0].auto_model, lora_config
            )
            self.text_encoder[0].auto_model.print_trainable_parameters()

        self.replace_cls = replace_cls
        self.normalize = normalize
        self.tau = tau

        # Load tag metadata
        with open(tag2idx_path) as f:
            self.tag2idx = json.load(f)
        self.n_tags = len(self.tag2idx)
        print(f"Tags: {self.n_tags}")

        # (n_items, n_tags) multi-hot matrix
        self.tag_matrix = torch.from_numpy(np.load(tag_matrix_path)).float().to(device)
        assert self.tag_matrix.shape[1] == self.n_tags

        # Inverse sqrt frequency weights for weighted Hellinger
        tag_freq = self.tag_matrix.sum(dim=0).cpu().numpy()
        w = 1.0 / np.sqrt(tag_freq + 1)
        w = w / w.mean()
        self.tag_weights = torch.from_numpy(w).float().to(device)

        # Single linear head for tag distribution prediction
        self.linear_tags = nn.Linear(d_latent, self.n_tags)

        self.item_table = torch.from_numpy(np.load(item_emb_path)).float().to(device)

    def _encode_text(self, summary, x=None):
        """Encode summaries; returns the L2-normalised pooled embedding (z_text)."""
        features = self.text_encoder.tokenize(summary, max_length=1024)
        features = {k: v.to(self.device) for k, v in features.items()}

        input_ids = features["input_ids"]
        attn_mask = features["attention_mask"]
        inputs_embeds = self.text_encoder[0].auto_model.get_input_embeddings()(input_ids)

        if self.replace_cls and x is not None:
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
        pooled_output = pool(outputs.last_hidden_state, self.model_name, attn_mask)
        z_text = F.normalize(pooled_output, dim=-1)
        return z_text

    def forward(self, x, summary, p=0.0):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - p))
            x = x * mask

        z_text = self._encode_text(summary, x)

        # Target tag counts of the (dropout-masked) interactions
        y_tags_true = x @ self.tag_matrix

        if self.normalize:
            y_tags_true = F.normalize(y_tags_true, p=1, dim=-1)

        y_tags_pred = self.linear_tags(z_text) / self.tau

        return y_tags_true, y_tags_pred, z_text

