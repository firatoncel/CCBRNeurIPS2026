import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import copy
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


def _infer_backbone_from_path(path):
    """
    Infer backbone type from checkpoint filename suffix.
      *_deep.pt   -> non-linear MLP autoencoder (Multi-DAE), nn.Sequential
      otherwise   -> linear (EASE / EDLAE / Linear-DAE)
    """
    if path is None:
        return "linear"
    name = os.path.basename(path).lower()
    if name.endswith("_deep.pt"):
        return "deep_ae"
    return "linear"


# ============================================================
# LatentAlignModel — DDP compatible (movie / ML-20M)
# ============================================================
class LatentAlignModel(nn.Module):
    def __init__(self, classifier_path, encoder_path, decoder_path,
                 n_items, use_linear=False, tau=0.003, nonlinear=False):
        super().__init__()

        backbone = _infer_backbone_from_path(encoder_path)
        self.backbone = backbone
        print(f"[LatentAlignModel] backbone inferred: {backbone}")

        # --- Load classifier and extract components (bypass SentenceTransformer) ---
        classifier_raw = torch.load(classifier_path, weights_only=False, map_location="cpu")

        self.auto_model = classifier_raw.text_encoder[0].auto_model.cpu()
        self.model_name = classifier_raw.model_name
        self.replace_cls = classifier_raw.replace_cls
        self._normalize = classifier_raw.normalize
        self._classifier_tau = classifier_raw.tau

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.linear_tags = classifier_raw.linear_tags.cpu()

        # tag_matrix and tag_weights as buffers
        if hasattr(classifier_raw, 'tag_matrix'):
            self.register_buffer('tag_matrix', classifier_raw.tag_matrix.cpu())
        if hasattr(classifier_raw, 'tag_weights'):
            self.register_buffer('tag_weights', classifier_raw.tag_weights.cpu())

        # item_table as buffer
        if hasattr(classifier_raw, 'item_table'):
            self.register_buffer('item_table', classifier_raw.item_table.cpu())

        del classifier_raw

        # --- Backbone: deep AE (nn.Sequential) vs linear ---
        if backbone == "deep_ae":
            # Saved as nn.Sequential modules — load directly, dims come from checkpoint
            self.encoder = torch.load(encoder_path, weights_only=False, map_location="cpu")
            self.decoder = torch.load(decoder_path, weights_only=False, map_location="cpu")
        else:
            enc_raw = torch.load(encoder_path, weights_only=False, map_location="cpu")
            dec_raw = torch.load(decoder_path, weights_only=False, map_location="cpu")

            if not isinstance(enc_raw, nn.Linear):
                enc_weight = enc_raw
                dec_weight = dec_raw
                self.encoder = nn.Linear(enc_weight.shape[0], enc_weight.shape[1], bias=False)
                self.encoder.weight = nn.Parameter(enc_weight.T)
                self.decoder = nn.Linear(dec_weight.shape[1], dec_weight.shape[0], bias=False)
                self.decoder.weight = nn.Parameter(dec_weight)
            else:
                self.encoder = enc_raw.cpu()
                self.decoder = dec_raw.cpu()

        self.decoder_text = copy.deepcopy(self.decoder)
        self.use_linear = use_linear
        self.tau = tau
        print(self.encoder)
        print(self.decoder)

        # Latent size of the CF backbone (output of the last Linear in the
        # encoder). The mapper projects the 1024-d text embedding to this size,
        # so it works for any backbone rank, not only 1024.
        latent_dim = [m for m in self.encoder.modules() if isinstance(m, nn.Linear)][-1].out_features

        if nonlinear:
            self.mapper = nn.Sequential(
                nn.Linear(1024, 1024, bias=False), nn.SiLU(),
                nn.Linear(1024, latent_dim, bias=False),
            )
        else:
            self.mapper = nn.Linear(1024, latent_dim, bias=False)

        self.bias = nn.Parameter(torch.zeros(n_items))
        print(f"LatentAlignModel | tau={self.tau} | use_linear={self.use_linear}")

    def _encode_text(self, summary, x=None):
        device = self.bias.device
        features = self.tokenizer(summary, max_length=1024, padding=True,
                                  truncation=True, return_tensors='pt')
        input_ids = features["input_ids"].to(device)
        attn_mask = features["attention_mask"].to(device)

        inputs_embeds = self.auto_model.get_input_embeddings()(input_ids)

        if self.replace_cls and x is not None:
            item_embeddings = mean_clicked_item_emb(x, self.item_table)
            if self.training:
                cls_mask = torch.rand(x.size(0), device=device) > 0.5
                inputs_embeds[:, 0] = torch.where(
                    cls_mask.unsqueeze(-1), item_embeddings, inputs_embeds[:, 0]
                )
            else:
                inputs_embeds[:, 0] = item_embeddings

        outputs = self.auto_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        pooled_output = pool(outputs.last_hidden_state, self.model_name, attn_mask)
        return F.normalize(pooled_output, dim=-1)

    def forward(self, x, summary, p=0.5):
        x_label = x.clone()
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - p))
            x = x * mask

        z_text = self._encode_text(summary, x)

        y_tags_pred = self.linear_tags(z_text) / self._classifier_tau
        y_tags_true = x_label @ self.tag_matrix
        if self._normalize:
            y_tags_true = F.normalize(y_tags_true, p=1, dim=-1)

        if self.use_linear:
            z_text = F.normalize(self.mapper(z_text), dim=-1)

        # Collaborative branch — works for both Linear and Sequential encoder
        z_colab = F.normalize(self.encoder(x), dim=-1)

        alpha = 0.5
        z_joint = F.normalize(alpha * z_text + (1 - alpha) * z_colab, dim=-1)

        y_text = self.decoder_text(z_text) / self.tau + self.bias
        y_colab = self.decoder(z_colab) / self.tau + self.bias
        y_joint = self.decoder(z_joint) / self.tau + self.bias

        return (z_colab, z_text, z_joint,
                y_colab, y_text, y_joint,
                y_tags_true, y_tags_pred)

    @torch.no_grad()
    def decode_text(self, text):
        if not isinstance(text, list):
            text = [text]

        z_text = self._encode_text(text)
        y_tags_pred = self.linear_tags(z_text) / self._classifier_tau

        if self.use_linear:
            z_text = F.normalize(self.mapper(z_text), dim=-1)

        y_text = self.decoder_text(z_text) / self.tau + self.bias
        return y_text, y_tags_pred