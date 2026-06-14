"""
Lesion-Aware Spatial Fusion — Novel Contributions for DermaFusion-AI
=====================================================================
This module contains the two ARCHITECTURE novelties of the paper. Both operate
on *spatial token grids* (not pooled vectors), which also repairs the prior
degenerate fusion where each branch was a single (B, 1, D) token (so the
attention softmax was over one key → always 1.0 → no real attention).

Both modules are pure-torch and take plain tensors, so they are unit-testable
without any pretrained backbone.

------------------------------------------------------------------------------
NOVELTY 2 — Boundary-Uncertainty-Gated Cross-Attention (BUG-Attn)
------------------------------------------------------------------------------
The Swin-UNet emits a soft lesion-probability map p ∈ [0,1]. Prior work
hard-thresholds it (apply_mask) and throws the soft signal away. We instead form
the per-location *diagnostic uncertainty* as the normalised Bernoulli variance

        u(x) = 4 · p(x) · (1 − p(x))   ∈ [0, 1]

which peaks at p = 0.5 — exactly the lesion BORDER, where the ABCDE rule's
Asymmetry and Border-irregularity criteria live. We then run real spatial
cross-attention (EVA tokens query, ConvNeXt tokens key/value) with an additive,
learnable key bias that steers attention toward border-uncertain regions and
away from confident background:

        score_{q,k} = (Q_q · K_k)/√d  +  γ · u_k  −  δ · (1 − p_k)

γ, δ are learnable scalars. Nearest prior work uses (a) the plain dual-branch
original+segmented cross-attention (arXiv:2510.17773) — which we ablate against
by setting γ=δ=0 — and (b) generic entropy-guided attention; neither derives an
attention bias from a *separate segmentation network's Bernoulli variance* to
couple an original/segmented dual stream.

------------------------------------------------------------------------------
NOVELTY 3 — Mirror-Asymmetry Attention (MAA)
------------------------------------------------------------------------------
Dermoscopic asymmetry (ABCDE "A") is the single most diagnostic feature, yet
self-attention is permutation-equivariant and blind to it. We center the feature
grid on the lesion centroid (computed from the mask) and measure feature
asymmetry as the discrepancy between the grid and its horizontal/vertical mirror
images, restricted to the lesion interior:

        a(x) = ‖f(x) − f_mirror(x)‖₂ · p(x)

This asymmetry map gates the features, injecting a reflection-asymmetry inductive
bias tied directly to the ABCDE "A" criterion. We found no prior work using a
segmentation-derived lesion frame to drive reflection-asymmetry attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
#         NOVELTY 2 — BOUNDARY-UNCERTAINTY-GATED CROSS-ATTENTION               #
# =========================================================================== #

class BoundaryUncertaintyGatedAttention(nn.Module):
    """
    Bidirectional-ready cross-attention with a segmentation-uncertainty key bias.

    Args:
        dim:        token dimension (both branches must share it)
        num_heads:  attention heads
        dropout:    dropout in attention + FFN
        enable_bias: if False, γ=δ are forced inactive (the ablation baseline =
                     plain spatial cross-attention, i.e. the arXiv:2510.17773 design)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1,
                 enable_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim         = dim
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.scale       = self.head_dim ** -0.5
        self.enable_bias = enable_bias

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Learnable bias strengths. γ boosts border-uncertain keys; δ suppresses
        # confident-background keys. Init: γ=0.5 (mild boost), δ=0.5.
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(0.5))

        # Pre-LN FFN
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor,
                p: torch.Tensor):
        """
        Args:
            q_tokens:  (B, Nq, D) query branch tokens (EVA / original image)
            kv_tokens: (B, Nk, D) key/value branch tokens (ConvNeXt / segmented)
            p:         (B, Nk) soft lesion probability at each key location, in [0,1]
        Returns:
            out:  (B, Nq, D) fused tokens
            attn: (B, Nq, Nk) attention averaged over heads (for visualisation)
        """
        B, Nq, D = q_tokens.shape
        Nk = kv_tokens.shape[1]

        qn = self.norm_q(q_tokens)
        kn = self.norm_kv(kv_tokens)

        q = self.q_proj(qn).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kn).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kn).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale          # (B, H, Nq, Nk)

        if self.enable_bias:
            p_c = p.clamp(0.0, 1.0)
            u = 4.0 * p_c * (1.0 - p_c)                          # (B, Nk) Bernoulli var
            bias = self.gamma * u - self.delta * (1.0 - p_c)     # (B, Nk)
            scores = scores + bias[:, None, None, :]             # broadcast over H, Nq

        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                            # (B, H, Nq, head_dim)
        out = out.transpose(1, 2).reshape(B, Nq, D)
        out = self.proj_drop(self.out_proj(out))

        # Residual (Pre-LN) + FFN
        out = q_tokens + out
        out = out + self.ffn(self.ffn_norm(out))

        attn_avg = attn.mean(dim=1)                               # (B, Nq, Nk)
        return out, attn_avg


# =========================================================================== #
#                NOVELTY 3 — MIRROR-ASYMMETRY ATTENTION                        #
# =========================================================================== #

class MirrorAsymmetryAttention(nn.Module):
    """
    Encodes the ABCDE "A" (asymmetry) criterion as a feature-space inductive bias.

    Operates on a square spatial grid. The lesion centroid is estimated from the
    mask and the grid is rolled so the lesion sits at the center; feature
    asymmetry is the discrepancy between the grid and its H/V mirrors, restricted
    to the lesion interior; this gates the features. Returns a pooled vector.

    Args:
        dim:    channel dimension of the feature grid
        lambda_init: initial strength of the asymmetry gate
    """

    def __init__(self, dim: int, lambda_init: float = 1.0):
        super().__init__()
        self.dim = dim
        # 1x1 projection of the (scalar) asymmetry map into a per-channel gate
        self.asym_proj = nn.Conv2d(1, dim, kernel_size=1)
        self.lam = nn.Parameter(torch.tensor(float(lambda_init)))
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _centroid_shift(pmap: torch.Tensor):
        """
        Integer (dh, dw) per-sample shift that moves the mask centroid to grid
        center. Shift amount is a (non-differentiable) function of the mask —
        fine, because features still receive gradients through the rolled tensor.
        pmap: (B, 1, G, G) → returns LongTensors (B,), (B,)
        """
        B, _, G, Gw = pmap.shape
        device = pmap.device
        ys = torch.arange(G, device=device).float()
        xs = torch.arange(Gw, device=device).float()
        w = pmap[:, 0]                                            # (B, G, G)
        denom = w.sum(dim=(1, 2)).clamp(min=1e-6)                 # (B,)
        cy = (w.sum(dim=2) * ys[None, :]).sum(dim=1) / denom      # (B,)
        cx = (w.sum(dim=1) * xs[None, :]).sum(dim=1) / denom      # (B,)
        # Center on the MIRROR axis (G-1)/2 — the fixed point of torch.flip
        # (index i ↦ G-1-i) — NOT G//2. On even grids these differ by 0.5, and
        # using G//2 forces a spurious 1-px roll that fabricates asymmetry even
        # for a perfectly centered, symmetric lesion.
        target_y = (G - 1) / 2.0
        target_x = (Gw - 1) / 2.0
        dh = (torch.full_like(cy, target_y) - cy).round().long()
        dw = (torch.full_like(cx, target_x) - cx).round().long()
        return dh, dw

    def forward(self, feat: torch.Tensor, pmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, D, G, G) feature grid (segmented/texture branch)
            pmap: (B, 1, G, G) soft lesion probability on the same grid
        Returns:
            (B, D) asymmetry-gated, mask-pooled feature vector
        """
        B, D, G, Gw = feat.shape
        dh, dw = self._centroid_shift(pmap)

        # Roll each sample so its lesion centroid is centered (per-sample shift).
        centered = torch.stack([
            torch.roll(feat[b], shifts=(int(dh[b]), int(dw[b])), dims=(1, 2))
            for b in range(B)
        ], dim=0)
        pcent = torch.stack([
            torch.roll(pmap[b], shifts=(int(dh[b]), int(dw[b])), dims=(1, 2))
            for b in range(B)
        ], dim=0)

        # Asymmetry vs horizontal and vertical mirrors (exact, differentiable).
        # NOTE: the ε inside sqrt is essential — without it, a locally symmetric
        # region gives an exactly-zero argument and d/dx √x = ∞ there, producing
        # NaN gradients (and NaN training loss) the moment features are symmetric.
        eps = 1e-6
        f_h = torch.flip(centered, dims=[-1])
        f_v = torch.flip(centered, dims=[-2])
        a_h = (centered - f_h).pow(2).mean(dim=1, keepdim=True).add(eps).sqrt()   # (B,1,G,G)
        a_v = (centered - f_v).pow(2).mean(dim=1, keepdim=True).add(eps).sqrt()
        asym = 0.5 * (a_h + a_v) * pcent                                 # lesion-interior only

        # Gate the centered features by the asymmetry signal, then mask-pool.
        gate = torch.sigmoid(self.asym_proj(asym))                       # (B, D, G, G)
        gated = centered * (1.0 + self.lam * gate)

        # Mask-weighted global pooling → (B, D)
        wsum = (gated * pcent).sum(dim=(2, 3))                           # (B, D)
        denom = pcent.sum(dim=(2, 3)).clamp(min=1e-6)                    # (B, 1)
        pooled = wsum / denom
        return self.norm(pooled)
