"""Trimodal spoilage model and its ablation variants.

One class, ``TrimodalModel``, builds the full model (notebook 04) and every
variant of the ablation study (notebook 05) from a variant dict in
``config.VARIANTS``. Parameter names match the notebooks, so checkpoints saved
by the notebooks (``model_state.pt``) load directly into this class.

    sRGB  -> ConvNeXt-Tiny (+CBAM)    -\
    IR    -> EfficientNet-B0 (+CBAM)   -> per-day fusion -> temporal model -> heads
    gas   -> GRU-D (or mean-impute GRU)-/
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import IMAGE_SIZE, VARIANTS


# ----------------------------------------------------------------------------- visual encoders
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, channels, 1, bias=False))

    def forward(self, x):
        return torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.conv(torch.cat([x.mean(dim=1, keepdim=True),
                                                  x.max(dim=1, keepdim=True)[0]], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention, then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_attn(x)
        sa = self.spatial_attn(x)
        return x * sa, sa


def build_backbone(name: str, pretrained: bool = True, stochastic_depth: bool = True):
    from torchvision.models import (ConvNeXt_Tiny_Weights, EfficientNet_B0_Weights, convnext_tiny,
                                    efficientnet_b0)
    extra = {} if stochastic_depth else {"stochastic_depth_prob": 0.0}   # torchvision defaults otherwise
    if name == "efficientnet_b0":
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None, **extra)
        return base.features, 1280
    if name == "convnext_tiny":
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None, **extra)
        return base.features, 768
    raise ValueError(f"Unknown backbone: {name}")


class VisualEncoder(nn.Module):
    """CNN backbone (+ optional CBAM) applied to every day of a sequence -> (B, T, C)."""

    def __init__(self, backbone_name: str, use_cbam: bool = True, pretrained: bool = True, chunk_size: int = 8,
                 stochastic_depth: bool = True):
        super().__init__()
        self.use_cbam = use_cbam
        self.chunk_size = chunk_size
        self.backbone, self.feature_dim = build_backbone(backbone_name, pretrained, stochastic_depth)
        if use_cbam:
            self.cbam = CBAM(self.feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_n_layers(self, n: int):
        self.freeze_backbone()
        for child in list(self.backbone.children())[-n:]:
            for p in child.parameters():
                p.requires_grad = True

    def forward(self, seq):
        B, T, C, H, W = seq.shape
        flat = seq.reshape(B * T, C, H, W)
        use_ckpt = torch.is_grad_enabled() and any(p.requires_grad for p in self.backbone.parameters())
        feats = []
        for start in range(0, flat.shape[0], self.chunk_size):
            chunk = flat[start:start + self.chunk_size]
            fmap = checkpoint(self.backbone, chunk, use_reentrant=False) if use_ckpt else self.backbone(chunk)
            if self.use_cbam:
                fmap, _ = self.cbam(fmap)
            feats.append(self.pool(fmap).flatten(1))
        return torch.cat(feats, dim=0).view(B, T, -1)


# ----------------------------------------------------------------------------- gas encoders
class GRUDCell(nn.Module):
    """GRU-D cell (Che et al., 2018): decays the last observation and the hidden state with time since
    the last reading, so a missing day is not the same as a zero reading."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.W_gamma_x = nn.Linear(1, input_dim)
        self.W_gamma_h = nn.Linear(1, hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim * 2, hidden_dim)

    def forward(self, x_t, m_t, delta_t, x_last, h_prev):
        gamma_x = torch.exp(-torch.clamp(self.W_gamma_x(delta_t), min=0.0))
        x_bar = self.x_mean.unsqueeze(0).expand_as(x_t)
        x_hat = m_t * x_t + (1 - m_t) * (gamma_x * x_last + (1 - gamma_x) * x_bar)
        gamma_h = torch.exp(-torch.clamp(self.W_gamma_h(delta_t), min=0.0))
        return self.gru_cell(torch.cat([x_hat, m_t], dim=-1), gamma_h * h_prev), x_hat


class GasGRUD(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = GRUDCell(input_dim, hidden_dim)

    def forward(self, gas_seq, gas_mask, gas_delta, pad_mask):
        B, T, D = gas_seq.shape
        h = gas_seq.new_zeros(B, self.hidden_dim)
        x_last = gas_seq.new_zeros(B, D)
        out = []
        for t in range(T):
            x_t = gas_seq[:, t]
            m_t = gas_mask[:, t].unsqueeze(-1).expand(-1, D)
            valid = (~pad_mask[:, t]).float().unsqueeze(-1)
            h_new, _ = self.cell(x_t, m_t, gas_delta[:, t].unsqueeze(-1), x_last, h)
            h = valid * h_new + (1 - valid) * h
            x_last = torch.where(m_t.bool(), x_t, x_last)
            out.append(h)
        return torch.stack(out, dim=1)


class GasMeanImpute(nn.Module):
    """Ablation: missing readings set to zero (the mean) and a plain GRU over [reading, mask]."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim + 1, hidden_dim, batch_first=True)

    def forward(self, gas_seq, gas_mask, gas_delta, pad_mask):
        x = torch.cat([gas_seq * gas_mask.unsqueeze(-1), gas_mask.unsqueeze(-1)], dim=-1)
        lengths = (~pad_mask).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=gas_seq.shape[1])
        return out


# ----------------------------------------------------------------------------- fusion and temporal
class CrossModalFusion(nn.Module):
    """Per-day fusion of the three modality tokens.

    With ``use_cross_attn`` the tokens attend to each other; missing modalities are excluded from the
    keys and from the average, and a day with nothing observed falls back to attending over all three.
    Without it, the tokens are concatenated and projected.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1, use_cross_attn: bool = True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
        else:
            self.concat_proj = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model),
                                             nn.ReLU(), nn.Dropout(dropout))

    def forward(self, rgb, ir, gas, rgb_av, ir_av, gas_av):
        if not self.use_cross_attn:
            return self.concat_proj(torch.cat([rgb, ir, gas], dim=-1)), None
        tokens = torch.stack([rgb, ir, gas], dim=1)
        avail = torch.stack([rgb_av, ir_av, gas_av], dim=1)
        kpm = ~avail
        none = kpm.all(dim=1)
        if none.any():
            kpm, avail = kpm.clone(), avail.clone()
            kpm[none], avail[none] = False, True
        attended, weights = self.attn(tokens, tokens, tokens, key_padding_mask=kpm,
                                      need_weights=True, average_attn_weights=True)
        out = self.norm(tokens + attended)
        a = avail.unsqueeze(-1).float()
        return (out * a).sum(dim=1) / a.sum(dim=1).clamp(min=1.0), weights


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, z, pad_mask):
        return self.encoder(self.pos_enc(z), src_key_padding_mask=pad_mask)


class TemporalLSTM(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.out_proj = nn.Identity()

    def forward(self, z, pad_mask):
        lengths = (~pad_mask).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=z.shape[1])
        return self.out_proj(out)


def gather_last_valid(H, pad_mask):
    last_idx = ((~pad_mask).sum(dim=1) - 1).clamp(min=0)
    return H[torch.arange(H.shape[0], device=H.device), last_idx], last_idx


def _proj(d_in, d_model, dropout):
    return nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(dropout))


def _heads(d_model, dropout):
    cls = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))
    reg = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1), nn.ReLU())
    return cls, reg


# ----------------------------------------------------------------------------- the model
class TrimodalModel(nn.Module):
    """Spoilage classifier + days-until-spoilage regressor over a day-level trajectory.

    Returns ``cls_logits`` (B, 2), ``reg_output`` (B, 1), ``gas_recon`` (B, 6) and ``last_idx`` (B,).
    """

    def __init__(self, variant: str | dict = "full", d_model: int = 64, dropout: float = 0.5,
                 gas_hidden: int = 64, num_heads: int = 4, temporal_layers: int = 2, pretrained: bool = True,
                 rgb_backbone: str = "convnext_tiny", ir_backbone: str = "efficientnet_b0",
                 stochastic_depth: bool = True):
        super().__init__()
        cfg = VARIANTS[variant] if isinstance(variant, str) else dict(variant)
        self.variant = cfg
        self.fusion_type = cfg["fusion"]

        self.rgb_encoder = VisualEncoder(rgb_backbone, cfg["use_cbam"], pretrained, stochastic_depth=stochastic_depth)
        self.ir_encoder = VisualEncoder(ir_backbone, cfg["use_cbam"], pretrained, stochastic_depth=stochastic_depth)
        self.gas_encoder = (GasGRUD if cfg["gas"] == "grud" else GasMeanImpute)(6, gas_hidden)

        self.rgb_proj = _proj(self.rgb_encoder.feature_dim, d_model, dropout)
        self.ir_proj = _proj(self.ir_encoder.feature_dim, d_model, dropout)
        self.gas_proj = _proj(gas_hidden, d_model, dropout)

        def temporal():
            if cfg["temporal"] == "transformer":
                return TemporalTransformer(d_model, num_heads, temporal_layers, dropout=dropout)
            return TemporalLSTM(d_model)

        if self.fusion_type == "intermediate":
            self.fusion = CrossModalFusion(d_model, num_heads, dropout, cfg["use_cross_attn"])
            self.temporal = temporal()
            self.cls_head, self.reg_head = _heads(d_model, dropout)
        elif self.fusion_type == "early":
            self.early_proj = _proj(d_model * 3, d_model, dropout)
            self.temporal = temporal()
            self.cls_head, self.reg_head = _heads(d_model, dropout)
        elif self.fusion_type == "late":
            for m in ("rgb", "ir", "gas"):
                setattr(self, f"temporal_{m}", temporal())
                cls, reg = _heads(d_model, dropout)
                setattr(self, f"cls_head_{m}", cls)
                setattr(self, f"reg_head_{m}", reg)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        self.gas_recon_head = nn.Sequential(nn.Linear(d_model * 2, 128), nn.ReLU(), nn.Dropout(dropout),
                                            nn.Linear(128, 6))

    # backbone schedule ------------------------------------------------------
    def freeze_visual_backbones(self):
        self.rgb_encoder.freeze_backbone()
        self.ir_encoder.freeze_backbone()

    def unfreeze_visual_last_n(self, n: int):
        self.rgb_encoder.unfreeze_last_n_layers(n)
        self.ir_encoder.unfreeze_last_n_layers(n)

    def unfreeze_visual_full(self):
        self.rgb_encoder.unfreeze_backbone()
        self.ir_encoder.unfreeze_backbone()

    # forward ----------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        device = next(self.parameters()).device
        b = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        rgb_av, ir_av, gas_mask, pad_mask = b["rgb_avail"], b["ir_avail"], b["gas_mask"], b["pad_mask"]
        B, T = rgb_av.shape
        rows = torch.arange(B, device=device)

        rgb = self.rgb_proj(self.rgb_encoder(b["rgb_seq"])) * rgb_av.unsqueeze(-1).float()
        ir = self.ir_proj(self.ir_encoder(b["ir_seq"])) * ir_av.unsqueeze(-1).float()
        gas = self.gas_proj(self.gas_encoder(b["gas_seq"], gas_mask, b["gas_delta"], pad_mask)) * gas_mask.unsqueeze(-1)

        if self.fusion_type == "intermediate":
            z, _ = self.fusion(rgb.reshape(B * T, -1), ir.reshape(B * T, -1), gas.reshape(B * T, -1),
                               rgb_av.reshape(B * T), ir_av.reshape(B * T), gas_mask.reshape(B * T).bool())
            H_T, last_idx = gather_last_valid(self.temporal(z.reshape(B, T, -1), pad_mask), pad_mask)
            cls_logits, reg_output = self.cls_head(H_T), self.reg_head(H_T)
        elif self.fusion_type == "early":
            z = self.early_proj(torch.cat([rgb, ir, gas], dim=-1))
            H_T, last_idx = gather_last_valid(self.temporal(z, pad_mask), pad_mask)
            cls_logits, reg_output = self.cls_head(H_T), self.reg_head(H_T)
        else:  # late: one temporal model + heads per modality, averaged over modalities seen on the last day
            logits, regs = [], []
            for m, seq in (("rgb", rgb), ("ir", ir), ("gas", gas)):
                H_T, last_idx = gather_last_valid(getattr(self, f"temporal_{m}")(seq, pad_mask), pad_mask)
                logits.append(getattr(self, f"cls_head_{m}")(H_T))
                regs.append(getattr(self, f"reg_head_{m}")(H_T))
            avail = torch.stack([rgb_av[rows, last_idx].float(), ir_av[rows, last_idx].float(),
                                 gas_mask[rows, last_idx]], dim=1)
            denom = avail.sum(dim=1, keepdim=True).clamp(min=1.0)
            cls_logits = (torch.stack(logits, dim=1) * avail.unsqueeze(-1)).sum(dim=1) / denom
            reg_output = ((torch.stack(regs, dim=1).squeeze(-1) * avail).sum(dim=1) / denom.squeeze(-1)).unsqueeze(-1)

        gas_recon = self.gas_recon_head(torch.cat([rgb[rows, last_idx], ir[rows, last_idx]], dim=1))
        return {"cls_logits": cls_logits, "reg_output": reg_output, "gas_recon": gas_recon, "last_idx": last_idx}


class UncertaintyWeightedLoss(nn.Module):
    """Kendall et al. (2018): sum_i 0.5 * exp(-2 log sigma_i) * L_i + log sigma_i, with log sigma learned."""

    def __init__(self, n_tasks: int = 3):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        return sum(0.5 * torch.exp(-2 * self.log_sigma[i]) * L + self.log_sigma[i] for i, L in enumerate(losses))

    def sigmas(self):
        return torch.exp(self.log_sigma).detach().cpu().tolist()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, variant="full", device="cpu", **kwargs) -> TrimodalModel:
    """Build a model of the given variant and load a ``model_state.pt`` saved by the notebooks or scripts."""
    model = TrimodalModel(variant, pretrained=False, **kwargs)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def dummy_batch(B: int = 2, T: int = 3) -> dict:
    """A random batch with the same keys as ``data.collate`` (for smoke tests and profiling)."""
    return {
        "rgb_seq": torch.randn(B, T, 3, IMAGE_SIZE, IMAGE_SIZE),
        "ir_seq": torch.randn(B, T, 3, IMAGE_SIZE, IMAGE_SIZE),
        "rgb_avail": torch.ones(B, T, dtype=torch.bool),
        "ir_avail": torch.ones(B, T, dtype=torch.bool),
        "gas_seq": torch.randn(B, T, 6),
        "gas_mask": torch.ones(B, T),
        "gas_delta": torch.zeros(B, T),
        "pad_mask": torch.zeros(B, T, dtype=torch.bool),
        "fruit_idx": torch.zeros(B, dtype=torch.long),
        "label": torch.tensor([i % 2 for i in range(B)]),
        "days_until": torch.zeros(B),
    }
