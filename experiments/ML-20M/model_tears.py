"""
model_tears.py — TEARS-style alignment model.

Adapted from Penaloza et al. (2024), "TEARS: Textual Representations for
Scrutable Recommendations". Differences from the original TEARS:

  - Uses BGE-M3 (BAAI/bge-m3) as the text encoder instead of T5.
  - Loads our pretrained Multi-VAE / MacridVAE checkpoints (saved via the
    `_mult_vae.pt` / `_macrid_vae.pt` convention).
  - Only the alignment + 3-branch reconstruction core is kept here; genre
    interventions are evaluated by intervene_tears.py.

Forward returns:
    logits_merged, logits_rec, logits_text,    # 3 reconstructions (shared decoder)
    text_mu, text_logvar,                       # text branch posterior
    rec_mu,  rec_logvar                         # collab branch posterior

This signature matches TEARS' OT_loss expectation exactly.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# Pooling (shared with our existing pipeline)
# ============================================================
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
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


# ============================================================
# Backbone inference (same convention as the rest of the pipeline)
# ============================================================
def _infer_backbone_from_path(path):
    """*_mult_vae.pt -> vae | *_macrid_vae.pt -> macrid"""
    if path is None:
        return None
    name = os.path.basename(path).lower()
    if name.endswith("_mult_vae.pt"):
        return "vae"
    if name.endswith("_macrid_vae.pt"):
        return "macrid"
    raise ValueError(
        f"TEARS model only supports VAE backbones. Got: {name}. "
        f"Use a checkpoint ending in _mult_vae.pt or _macrid_vae.pt."
    )


# ============================================================
# VAE backbone wrappers
# These wrap our pretrained MultVAE / MacridVAE checkpoints with a unified
# interface: encode(x) -> (mu, logvar) ; reparameterize ; decode(z) -> logits.
# ============================================================

class _VAEBackbone(nn.Module):
    """Wraps our pretrained Multi-VAE checkpoint.

    Checkpoint layout (matches our MultVAE save convention):
        encoder_path -> {'encoder': sd, 'mu_layer': sd, 'logvar_layer': sd}
        decoder_path -> decoder state_dict
    """
    def __init__(self, encoder_path, decoder_path):
        super().__init__()
        enc_ckpt = torch.load(encoder_path, weights_only=False, map_location="cpu")
        dec_state = torch.load(decoder_path, weights_only=False, map_location="cpu")

        # Infer dims from state_dict shapes
        enc_w = enc_ckpt['encoder']['0.weight']           # (hidden, num_items)
        hidden_dim = enc_w.shape[0]
        num_items = enc_w.shape[1]
        latent_dim = enc_ckpt['mu_layer']['weight'].shape[0]

        self.num_items = num_items
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items),
        )

        self.encoder.load_state_dict(enc_ckpt['encoder'])
        self.mu_layer.load_state_dict(enc_ckpt['mu_layer'])
        self.logvar_layer.load_state_dict(enc_ckpt['logvar_layer'])
        self.decoder.load_state_dict(dec_state)

        print(f"[VAE backbone] num_items={num_items}, hidden={hidden_dim}, latent={latent_dim}")

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z):
        return self.decoder(z)


class _MacridBackbone(nn.Module):
    """Wraps our pretrained MacridVAE checkpoint.

    Checkpoint layout (matches our MacridVAE save convention):
        encoder_path -> {'encoder': sd, 'items': tensor, 'cores': tensor}

    encode/decode follow MacridVAE in autoencoders.py, with the concept
    temperature set to `concept_tau`.
    """
    def __init__(self, encoder_path, concept_tau=1.0):
        super().__init__()
        enc_ckpt = torch.load(encoder_path, weights_only=False, map_location="cpu")

        enc_first_w = enc_ckpt['encoder']['0.weight']      # (hidden, num_items)
        enc_last_w = enc_ckpt['encoder']['2.weight']       # (2*latent, hidden)
        hidden_dim = enc_first_w.shape[0]
        num_items = enc_first_w.shape[1]
        latent_dim = enc_last_w.shape[0] // 2

        items = enc_ckpt['items']
        cores = enc_ckpt['cores']
        n_concepts = cores.shape[0]

        self.num_items = num_items
        self.latent_dim = latent_dim
        self.n_concepts = n_concepts
        self.concept_tau = concept_tau

        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.encoder.load_state_dict(enc_ckpt['encoder'])

        self.items = nn.Parameter(items.clone())
        self.cores = nn.Parameter(cores.clone())

        print(f"[Macrid backbone] num_items={num_items}, latent={latent_dim}, "
              f"n_concepts={n_concepts}")

    def _cluster(self):
        items_n = F.normalize(self.items, dim=1)
        cores_n = F.normalize(self.cores, dim=1)
        cates = F.softmax(items_n @ cores_n.t() / self.concept_tau, dim=1)  # (M, K)
        return items_n, cores_n, cates

    def encode(self, x):
        """Returns flattened (mu, logvar) of shape (B, K*D) so that downstream
        code (TEARS forward, OT loss) can treat them like a regular VAE."""
        items_n, cores_n, cates = self._cluster()
        n = x.shape[0]
        K, M, D = self.n_concepts, self.num_items, self.latent_dim

        x_k = x.unsqueeze(1) * cates.t().unsqueeze(0)   # (n, K, M)
        x_k = x_k.reshape(n * K, M)
        h = self.encoder(x_k)                            # (n*K, 2D)
        mu = h[:, :D].reshape(n, K * D)                     # flatten across concepts
        logvar = h[:, D:].reshape(n, K * D)

        # Cache for decode (same forward pass)
        self._cache = (items_n, cates, K, D, M)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z):
        """z: (B, K*D). Returns the log of the concept-weighted item scores, shape (B, M)."""
        items_n, cates, K, D, M = self._cache
        n = z.shape[0]
        z_k = z.reshape(n * K, D) if z.shape == (n, K * D) else z.reshape(n, K, D).reshape(n * K, D)

        logits = (z_k @ items_n.t()) / self.concept_tau     # (n*K, M)
        probs = torch.exp(logits)
        probs = probs.reshape(n, K, M) * cates.t().unsqueeze(0)
        probs = probs.sum(dim=1)                             # (n, M)
        return torch.log(probs + 1e-10)


# ============================================================
# Main TEARS model
# ============================================================
class TearsAlignModel(nn.Module):
    """TEARS-style alignment model.

    Architecture:
        - Frozen pretrained Multi-VAE / MacridVAE backbone
        - Text encoder (BGE-M3) wrapped in LoRA
        - text_to_gaussian (linear): text_dim -> (mu_text, logvar_text)
        - 3 branches all share the VAE decoder:
            z_text   -> decoder -> logits_text
            z_rec    -> decoder -> logits_rec
            z_merged -> decoder -> logits_merged   (mixing of z_text and z_rec)
    """
    def __init__(self, encoder_path, decoder_path,
                 alpha=0.5,
                 lora=True, lora_r=64, lora_alpha=16, lora_dropout=0.1,
                 text_dim=1024):
        super().__init__()

        # --- Backbone (Multi-VAE or MacridVAE) ---
        backbone_kind = _infer_backbone_from_path(encoder_path)
        self.backbone_kind = backbone_kind
        if backbone_kind == "vae":
            self.vae = _VAEBackbone(encoder_path, decoder_path)
        else:  # macrid
            self.vae = _MacridBackbone(encoder_path)

        latent_dim = self.vae.latent_dim
        if backbone_kind == "macrid":
            latent_dim = latent_dim * self.vae.n_concepts  # flattened K*D
        self.latent_dim = latent_dim

        # --- Text encoder loaded directly from HuggingFace ---
        self.model_name = "BAAI/bge-m3"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.auto_model = AutoModel.from_pretrained(self.model_name)

        # Wrap auto_model in LoRA so we only train low-rank adapters on it
        if lora:
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["query", "key", "value", "dense"],
                lora_dropout=lora_dropout, bias="none",
            )
            self.auto_model = get_peft_model(self.auto_model, lora_cfg)
            print("[TEARS] Wrapped text encoder with LoRA")

        # --- Text -> (mu_text, logvar_text) head ---
        # Project pooled text embedding (text_dim) to 2 * latent_dim, split.
        self.text_to_gaussian = nn.Sequential(
            nn.Linear(text_dim, 2 * latent_dim, bias=False),
        )
        for layer in self.text_to_gaussian:
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.size(1)
                fan_out = layer.weight.size(0)
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)

        # --- Mixing coefficient of the merged branch ---
        self.alpha = alpha

        # --- Freeze the pretrained backbone ---
        self._freeze_pretrained_components()

        print(f"[TearsAlignModel] backbone={backbone_kind} | latent_dim={latent_dim} | "
              f"text_dim={text_dim} | alpha={alpha}")

    # ------------------------------------------------------------
    def _freeze_pretrained_components(self):
        """Freeze all Multi-VAE weights, or the MacridVAE encoder and
        item/prototype embeddings."""
        if self.backbone_kind == "vae":
            for p in self.vae.parameters():
                p.requires_grad = False
        else:  # macrid
            for p in self.vae.encoder.parameters():
                p.requires_grad = False
            self.vae.items.requires_grad = False
            self.vae.cores.requires_grad = False

    # ------------------------------------------------------------
    def _encode_text(self, summary):
        device = next(self.parameters()).device
        features = self.tokenizer(summary, max_length=1024, padding=True,
                                  truncation=True, return_tensors='pt')
        input_ids = features["input_ids"].to(device)
        attn_mask = features["attention_mask"].to(device)

        outputs = self.auto_model(input_ids=input_ids, attention_mask=attn_mask)
        pooled = pool(outputs.last_hidden_state, self.model_name, attn_mask)
        # NOTE: NO L2-normalize here — TEARS keeps the raw pooled embedding
        # since the gaussian head expects raw inputs.
        return pooled

    # ------------------------------------------------------------
    def forward(self, x, summary, alpha=None):
        """Returns:
            logits_merged, logits_rec, logits_text,
            text_mu, text_logvar, rec_mu, rec_logvar
        """
        if alpha is None:
            alpha = self.alpha

        # --- Text branch: text encoder -> (mu_text, logvar_text) -> z_text
        text_h = self._encode_text(summary)              # (B, text_dim)
        gauss = self.text_to_gaussian(text_h)            # (B, 2*latent_dim)
        text_mu, text_logvar = gauss[:, :self.latent_dim], gauss[:, self.latent_dim:]
        text_logvar = torch.clamp(text_logvar, min=-10.0, max=10.0)
        z_text = self.vae.reparameterize(text_mu, text_logvar)

        # --- Collab branch: VAE encode -> (rec_mu, rec_logvar) -> z_rec
        rec_mu, rec_logvar = self.vae.encode(x)
        z_rec = self.vae.reparameterize(rec_mu, rec_logvar)

        # Macrid: L2-normalise each concept's latent in z_text and z_rec
        if self.backbone_kind == "macrid":
            K, D = self.vae.n_concepts, self.vae.latent_dim
            z_text = F.normalize(z_text.view(-1, K, D), dim=-1).view(-1, K * D)
            z_rec = F.normalize(z_rec.view(-1, K, D), dim=-1).view(-1, K * D)

        # --- Mixed branch: convex combination
        z_merged = (1.0 - alpha) * z_text + alpha * z_rec

        if self.backbone_kind == "macrid":
            K, D = self.vae.n_concepts, self.vae.latent_dim
            z_merged = F.normalize(z_merged.view(-1, K, D), dim=-1).view(-1, K * D)

        # --- Three reconstructions, all shared decoder
        logits_merged = self.vae.decode(z_merged)
        logits_rec = self.vae.decode(z_rec)
        logits_text = self.vae.decode(z_text)

        return (logits_merged, logits_rec, logits_text,
                text_mu, text_logvar, rec_mu, rec_logvar)


# ============================================================
# OT loss — Bures-Wasserstein distance between two diagonal Gaussians
# ============================================================
class TearsOTLoss(nn.Module):
    """TEARS' alignment loss: 3-branch multinomial recon + KL + Bures-Wasserstein.

    Args (constructor):
        recon_tau, text_tau, rec_tau:  per-branch reconstruction weights
        backbone_kind:                 'vae' or 'macrid' (changes KL formula)
        kfac, split:                   only used when backbone_kind == 'macrid'
    """
    def __init__(self, recon_tau=1.0, text_tau=1.0, rec_tau=1.0,
                 backbone_kind="vae", kfac=None, split=None):
        super().__init__()
        self.recon_tau = recon_tau
        self.text_tau = text_tau
        self.rec_tau = rec_tau
        self.backbone_kind = backbone_kind
        self.kfac = kfac
        self.split = split

    @staticmethod
    def bures_wasserstein_diag(mu1, sigma1_diag, mu2, sigma2_diag):
        """Bures-Wasserstein squared distance between two diagonal Gaussians.
        Closed form when both covariances are diagonal:
            W^2 = ||mu1 - mu2||^2 + sum( (sqrt(s1) - sqrt(s2))^2 )
        """
        mean_term = ((mu1 - mu2) ** 2).sum(dim=1)
        cov_term = ((sigma1_diag.sqrt() - sigma2_diag.sqrt()) ** 2).sum(dim=1)
        return mean_term + cov_term

    def forward(self, logits_merged, logits_rec, logits_text, x,
                text_mu, text_logvar, rec_mu, rec_logvar,
                anneal=1.0, epsilon=1.0):
        # ---- Three multinomial recon losses, all on the same target x ----
        BCE_merged = -torch.mean(torch.mean(F.log_softmax(logits_merged, 1) * x, -1))
        BCE_text = -torch.mean(torch.mean(F.log_softmax(logits_text, 1) * x, -1))
        BCE_rec = -torch.mean(torch.mean(F.log_softmax(logits_rec, 1) * x, -1))

        BCE = (1.0 / 3.0) * (
            BCE_merged * self.recon_tau
            + BCE_text * self.text_tau
            + BCE_rec * self.rec_tau
        )

        # ---- KL on the text-branch posterior ----
        if self.backbone_kind == "macrid":
            # Macrid KL: per-factor, mu term dropped (factors are unit-norm)
            logvar_v = text_logvar.view(-1, self.kfac, self.split)
            KLD = -0.5 * torch.mean(torch.mean(1 + logvar_v - logvar_v.exp(), dim=-1))
        else:
            KLD = -0.5 * torch.mean(
                torch.mean(1 + text_logvar - text_mu.pow(2) - text_logvar.exp(), dim=1)
            )

        # ---- Bures-Wasserstein between text posterior and collab posterior ----
        sigma2_text = torch.exp(text_logvar)
        sigma2_rec = torch.exp(rec_logvar)
        wasserstein = self.bures_wasserstein_diag(
            text_mu, sigma2_text, rec_mu, sigma2_rec
        ).mean()

        total = BCE + anneal * KLD + epsilon * wasserstein

        return total, BCE, wasserstein, BCE_rec, BCE_text, BCE_merged
