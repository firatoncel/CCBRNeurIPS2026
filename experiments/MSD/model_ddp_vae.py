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


# ============================================================
# Backbone inference from filename suffix
# ============================================================
def _infer_backbone_from_path(path):
    """
    Infer backbone type from checkpoint filename suffix.
      *_deep.pt        -> non-linear MLP autoencoder (Multi-DAE), nn.Sequential
      *_deep_4096.pt   -> deep_ae (alternative dim variant)
      *_mult_vae.pt    -> Multi-VAE
      *_macrid_vae.pt  -> MacridVAE
      otherwise        -> linear (EASE / EDLAE / Linear-DAE)
    """
    if path is None:
        return "linear"
    name = os.path.basename(path).lower()
    if name.endswith("_mult_vae.pt"):
        return "vae"
    if name.endswith("_macrid_vae.pt"):
        return "macrid"
    if name.endswith("_deep.pt") or name.endswith("_deep_4096.pt"):
        return "deep_ae"
    return "linear"


# ============================================================
# LatentAlignModel — DDP compatible (linear / deep_ae / vae / macrid)
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

        # --- Backbone-specific loading ---
        latent_dim = self._build_backbone(encoder_path, decoder_path)
        print(f"[LatentAlignModel] latent_dim = {latent_dim}")

        self.use_linear = use_linear
        self.tau = tau

        # Mapper: 1024 -> latent_dim
        # NOTE: For VAE/Macrid, mapper output is RAW (not normalized) because
        # the alignment target (mu) lives in Gaussian space, not on a hypersphere.
        if nonlinear:
            self.mapper = nn.Sequential(
                nn.Linear(1024, 1024, bias=False), nn.SiLU(),
                nn.Linear(1024, latent_dim, bias=False),
            )
        else:
            self.mapper = nn.Linear(1024, latent_dim, bias=False)

        self.bias = nn.Parameter(torch.zeros(n_items))
        print(f"LatentAlignModel | tau={self.tau} | use_linear={self.use_linear} | backbone={self.backbone}")

    # --------------------------------------------------------
    # Backbone construction (split per type for clarity)
    # --------------------------------------------------------
    def _build_backbone(self, encoder_path, decoder_path):
        """Build encoder/decoder based on self.backbone. Returns latent_dim."""
        if self.backbone == "vae":
            return self._build_vae(encoder_path, decoder_path)
        if self.backbone == "macrid":
            return self._build_macrid(encoder_path)
        if self.backbone == "deep_ae":
            return self._build_deep_ae(encoder_path, decoder_path)
        return self._build_linear(encoder_path, decoder_path)

    def _build_linear(self, encoder_path, decoder_path):
        enc_raw = torch.load(encoder_path, weights_only=False, map_location="cpu")
        dec_raw = torch.load(decoder_path, weights_only=False, map_location="cpu")

        if not isinstance(enc_raw, nn.Linear):
            enc_weight = enc_raw
            dec_weight = dec_raw
            print(enc_weight.shape, dec_weight.shape)
            self.encoder = nn.Linear(enc_weight.shape[0], enc_weight.shape[1], bias=False)
            self.encoder.weight = nn.Parameter(enc_weight.T)
            self.decoder = nn.Linear(dec_weight.shape[1], dec_weight.shape[0], bias=False)
            self.decoder.weight = nn.Parameter(dec_weight.T)
            latent_dim = enc_weight.shape[1]
        else:
            self.encoder = enc_raw.cpu()
            self.decoder = dec_raw.cpu()
            latent_dim = 1024

        print(self.encoder, self.decoder)
        self.decoder_text = copy.deepcopy(self.decoder)
        return latent_dim

    def _build_deep_ae(self, encoder_path, decoder_path):
        # Saved as full nn.Sequential modules
        self.encoder = torch.load(encoder_path, weights_only=False, map_location="cpu")
        self.decoder = torch.load(decoder_path, weights_only=False, map_location="cpu")
        # Infer latent_dim from last Linear in encoder Sequential
        latent_dim = None
        for m in reversed(list(self.encoder.modules())):
            if isinstance(m, nn.Linear):
                latent_dim = m.out_features
                break
        if latent_dim is None:
            latent_dim = 4096

        print(self.encoder, self.decoder)
        self.decoder_text = copy.deepcopy(self.decoder)
        return latent_dim

    def _build_vae(self, encoder_path, decoder_path):
        # Encoder ckpt: dict {'encoder', 'mu_layer', 'logvar_layer'} state_dicts
        # Decoder ckpt: decoder state_dict
        enc_ckpt = torch.load(encoder_path, weights_only=False, map_location="cpu")
        dec_state = torch.load(decoder_path, weights_only=False, map_location="cpu")

        # Infer dimensions from state_dict shapes
        # encoder trunk: Sequential(Linear(num_items, hidden_dim), Tanh())
        enc_w = enc_ckpt['encoder']['0.weight']  # (hidden_dim, num_items)
        hidden_dim = enc_w.shape[0]
        num_items_ckpt = enc_w.shape[1]
        # mu_layer: Linear(hidden_dim, latent_dim)
        latent_dim = enc_ckpt['mu_layer']['weight'].shape[0]

        # Build architecture
        self.vae_trunk = nn.Sequential(
            nn.Linear(num_items_ckpt, hidden_dim),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items_ckpt),
        )

        # Load weights
        self.vae_trunk.load_state_dict(enc_ckpt['encoder'])
        self.mu_layer.load_state_dict(enc_ckpt['mu_layer'])
        self.logvar_layer.load_state_dict(enc_ckpt['logvar_layer'])
        self.decoder.load_state_dict(dec_state)

        # Alias for any external code expecting `self.encoder`
        self.encoder = self.vae_trunk

        print(self.vae_trunk, self.mu_layer, self.logvar_layer, self.decoder)
        self.decoder_text = copy.deepcopy(self.decoder)
        return latent_dim

    def _build_macrid(self, encoder_path):
        # Encoder ckpt: dict {'encoder', 'items', 'cores'}
        # The decoder checkpoint is not needed: items/cores are in the encoder checkpoint
        enc_ckpt = torch.load(encoder_path, weights_only=False, map_location="cpu")

        # Infer dimensions
        enc_first_w = enc_ckpt['encoder']['0.weight']  # (hidden_dim, num_items)
        hidden_dim = enc_first_w.shape[0]
        num_items_ckpt = enc_first_w.shape[1]
        # encoder last layer: Linear(hidden_dim, 2*latent_dim)
        enc_last_w = enc_ckpt['encoder']['2.weight']
        two_D = enc_last_w.shape[0]
        latent_dim = two_D // 2

        items = enc_ckpt['items']
        cores = enc_ckpt['cores']
        n_concepts = cores.shape[0]

        # Macrid encoder: Linear -> Tanh -> Linear(., 2D)
        self.macrid_encoder = nn.Sequential(
            nn.Linear(num_items_ckpt, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.macrid_encoder.load_state_dict(enc_ckpt['encoder'])

        # items, cores as Parameters (not in the optimizer, so they stay fixed)
        self.items = nn.Parameter(items.clone())
        self.cores = nn.Parameter(cores.clone())
        self.n_concepts = n_concepts
        self.macrid_concept_tau = 1.0  # Macrid's internal cluster-softmax tau (default)

        # Text branch: standard decoder, init from scratch
        # (Macrid's geometry doesn't transfer cleanly to a Sequential text decoder)
        self.decoder_text = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items_ckpt),
        )

        # `encoder` points to the Macrid encoder; Macrid has no decoder module
        self.encoder = self.macrid_encoder
        self.decoder = None  # Macrid decodes via items + cores, no separate module

        print(self.macrid_encoder, "items", self.items.shape, "cores", self.cores.shape)
        return latent_dim

    # --------------------------------------------------------
    # Text encoding (shared across backbones)
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Backbone-specific encoding
    # --------------------------------------------------------
    def _encode_collab_vae(self, x):
        h = self.vae_trunk(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        # z_colab = mu (raw, NO normalize) — alignment target in Gaussian space
        return mu, z  # mu for alignment, z for reconstruction

    def _encode_collab_macrid(self, x):
        # cluster
        items_n = F.normalize(self.items, dim=1)
        cores_n = F.normalize(self.cores, dim=1)
        cates = F.softmax(items_n @ cores_n.t() / self.macrid_concept_tau, dim=1)  # (M, K)

        n = x.shape[0]
        K = self.n_concepts
        M = self.items.shape[0]
        D = self.items.shape[1]

        # per-concept input split
        x_k = x.unsqueeze(1) * cates.t().unsqueeze(0)  # (n, K, M)
        x_k = x_k.reshape(n * K, M)

        h = self.macrid_encoder(x_k)  # (n*K, 2D)
        mu = h[:, :D]
        logvar = h[:, D:]

        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        # Concept-weighted average over K -> single per-user latent for alignment
        # User-level concept weights via x interactions
        x_sum = x.sum(1, keepdim=True).clamp_min(1.0)
        user_concept_w = (x @ cates) / x_sum  # (n, K)
        user_concept_w = F.normalize(user_concept_w, p=1, dim=1)  # ensure sum to 1

        mu_per_user = mu.view(n, K, D)  # (n, K, D)
        mu_global = (mu_per_user * user_concept_w.unsqueeze(-1)).sum(dim=1)  # (n, D)

        return mu_global, z, items_n, cates

    def _decode_macrid(self, z, items_n, cates):
        """MacridVAE decode: log of the concept-weighted sum of exp(z . item / tau)."""
        n = z.shape[0] // self.n_concepts
        K = self.n_concepts
        M = items_n.shape[0]

        logits = torch.mm(z, items_n.t()) / self.macrid_concept_tau  # (n*K, M)
        probs = torch.exp(logits)
        probs = probs.view(n, K, M) * cates.t().unsqueeze(0)  # (n, K, M)
        probs = probs.sum(dim=1)  # (n, M)
        logits_out = torch.log(probs + 1e-10)
        return logits_out

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x, summary, p=0.5):
        x_label = x.clone()
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - p))
            x = x * mask

        z_text_raw = self._encode_text(summary, x)

        y_tags_pred = self.linear_tags(z_text_raw) / self._classifier_tau
        y_tags_true = x_label @ self.tag_matrix
        if self._normalize:
            y_tags_true = F.normalize(y_tags_true, p=1, dim=-1)

        # ---- VAE / Macrid branch (Gaussian-space alignment) ----
        if self.backbone in ("vae", "macrid"):
            # Mapper output is RAW (no normalize) — Gaussian space
            if self.use_linear:
                z_text = self.mapper(z_text_raw)
            else:
                z_text = z_text_raw  # only meaningful if dim already matches latent

            if self.backbone == "vae":
                mu, z_recon = self._encode_collab_vae(x)
                z_colab = mu  # alignment target

                y_colab = self.decoder(z_recon) / self.tau + self.bias
                y_text = self.decoder_text(z_text) / self.tau + self.bias
            else:  # macrid
                mu_global, z_recon, items_n, cates = self._encode_collab_macrid(x)
                z_colab = mu_global  # alignment target

                # Macrid's own decode for y_colab — no /tau, no bias (Macrid's geometry)
                y_colab = self._decode_macrid(z_recon, items_n, cates)
                # y_text uses standard text decoder
                y_text = self.decoder_text(z_text) / self.tau + self.bias

            # No joint branch in VAE/Macrid mode — return None placeholders
            # so main loop can keep a uniform unpacking pattern.
            z_joint = None
            y_joint = None
            return (z_colab, z_text, z_joint,
                    y_colab, y_text, y_joint,
                    y_tags_true, y_tags_pred)

        # ---- Linear / deep_ae branch (original logic) ----
        if self.use_linear:
            z_text = F.normalize(self.mapper(z_text_raw), dim=-1)
        else:
            z_text = z_text_raw

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

        z_text_raw = self._encode_text(text)
        y_tags_pred = self.linear_tags(z_text_raw) / self._classifier_tau

        if self.backbone in ("vae", "macrid"):
            if self.use_linear:
                z_text = self.mapper(z_text_raw)
            else:
                z_text = z_text_raw
            y_text = self.decoder_text(z_text) / self.tau + self.bias
            return y_text, y_tags_pred

        # linear / deep_ae
        if self.use_linear:
            z_text = F.normalize(self.mapper(z_text_raw), dim=-1)
        else:
            z_text = z_text_raw
        y_text = self.decoder_text(z_text) / self.tau + self.bias
        return y_text, y_tags_pred