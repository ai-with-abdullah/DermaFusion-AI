"""
Dual-Branch Fusion Classifier
==============================
Combines Vision Mamba V2 (global/sequential) and ConvNeXt V3 (local/texture)
branches with multi-head cross-attention fusion and gated feature combination.

Pipeline:
  img_orig  → VisionMambaV2   → feat_mamba  (B, D)
  img_seg   → ConvNeXtV3      → feat_conv   (B, D)
                                    ↓
              MultiScaleCrossAttention(feat_mamba, feat_conv) → fused (B, D)
                                    ↓
              GatedResidualFusion(fused, feat_mamba, feat_conv) → combined (B, D)
                                    ↓
              ClassifierHead → logits (B, num_classes)

Maintains forward(img_orig, img_seg) → (logits, attn_weights) API
for drop-in compatibility with existing training and evaluation code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
#                      MULTI-HEAD CROSS ATTENTION FUSION                       #
# =========================================================================== #

class MultiScaleCrossAttention(nn.Module):
    """
    Bidirectional cross-attention fusion between two feature streams.
    
    Branch A queries Branch B (A attends to B's features) and vice versa.
    The two attended outputs are combined via a learned projection.
    
    Args:
        embed_dim:  Feature dimension of both branches
        num_heads:  Number of attention heads
        dropout:    Dropout rate
    """

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim

        # Pre-norms
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

        # Cross-attention: A → B (A queries, B is key/value)
        self.cross_attn_a2b = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: B → A (B queries, A is key/value)
        self.cross_attn_b2a = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)

        # Post-attention projection: concat A→B and B→A outputs → fused
        self.post_norm = nn.LayerNorm(embed_dim * 2)
        self.post_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # FFN after fusion
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        """
        Args:
            feat_a: (B, D) - Mamba branch features
            feat_b: (B, D) - ConvNeXt branch features
        Returns:
            fused: (B, D) - Cross-attention fused features
            attn_weights: (B, 1, 1) - Average attention weights for visualization
        """
        # Reshape to sequence format: (B, 1, D)
        q_a = feat_a.unsqueeze(1)
        q_b = feat_b.unsqueeze(1)

        # Normalize
        q_a_norm = self.norm_a(q_a)
        q_b_norm = self.norm_b(q_b)

        # A attends to B
        attn_a2b, weights_a2b = self.cross_attn_a2b(
            query=q_a_norm, key=q_b_norm, value=q_b_norm
        )
        attended_a = q_a + self.drop(attn_a2b)  # Residual

        # B attends to A
        attn_b2a, weights_b2a = self.cross_attn_b2a(
            query=q_b_norm, key=q_a_norm, value=q_a_norm
        )
        attended_b = q_b + self.drop(attn_b2a)  # Residual

        # Squeeze back to (B, D)
        attended_a = attended_a.squeeze(1)
        attended_b = attended_b.squeeze(1)

        # Concatenate and project: (B, 2D) → (B, D)
        concat = torch.cat([attended_a, attended_b], dim=-1)  # (B, 2D)
        concat = self.post_norm(concat)
        fused = self.post_proj(concat)  # (B, D)

        # FFN with residual
        fused_residual = fused
        fused = self.ffn_norm(fused)
        fused = fused_residual + self.ffn_drop(self.ffn(fused))

        # Average attention weights for visualization
        attn_weights = (weights_a2b + weights_b2a) / 2.0

        return fused, attn_weights


# =========================================================================== #
#                        GATED RESIDUAL FUSION                                 #
# =========================================================================== #

class DualBranchGatedFusion(nn.Module):
    """
    Learnable gating mechanism that dynamically weights contributions from
    the cross-attention fused features, the Mamba branch, and the ConvNeXt branch.
    
    gate_m = σ(W_m · [fused; feat_mamba])
    gate_c = σ(W_c · [fused; feat_conv])
    output = fused + gate_m * feat_mamba + gate_c * feat_conv
    
    This allows the model to adaptively emphasize global (Mamba) or local
    (ConvNeXt) features depending on the input.
    """

    def __init__(self, embed_dim=512, dropout=0.1):
        super().__init__()

        # Gate for Mamba features
        self.gate_mamba = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        # Gate for ConvNeXt features  
        self.gate_conv = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, fused, feat_mamba, feat_conv):
        """
        Args:
            fused:      (B, D) - Cross-attention fused features
            feat_mamba: (B, D) - Mamba branch features
            feat_conv:  (B, D) - ConvNeXt branch features
        Returns:
            combined: (B, D) - Final gated combination
        """
        # Compute gates
        g_m = self.gate_mamba(torch.cat([fused, feat_mamba], dim=-1))  # (B, D)
        g_c = self.gate_conv(torch.cat([fused, feat_conv], dim=-1))   # (B, D)

        # Gated residual combination
        combined = fused + g_m * feat_mamba + g_c * feat_conv
        combined = self.norm(combined)
        combined = self.dropout(combined)

        return combined


# =========================================================================== #
#                        CLASSIFIER HEAD                                       #
# =========================================================================== #

class ClassifierHead(nn.Module):
    """
    Two-layer classification head with LayerNorm and dropout.
    """
    def __init__(self, in_features=512, hidden_features=256, num_classes=7, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# =========================================================================== #
#                    DUAL-BRANCH FUSION CLASSIFIER                             #
# =========================================================================== #

class DualBranchFusionClassifier(nn.Module):
    """
    Dual-Branch Fusion model combining EVA-02 and ConvNeXt V2 Base.
    
    Branch A (EVA-02):   Captures long-range global/contextual patterns
                         from the ORIGINAL image.
    Branch B (ConvNeXt): Captures local texture/spatial features
                         from the SEGMENTED image.
    
    The features are fused via bidirectional cross-attention, combined
    with learnable gating, and classified.
    
    Args:
        eva02_name:      timm model name for EVA-02 backbone
        eva02_pretrained: Whether to load pretrained EVA-02 weights
        convnext_name:   timm model name for ConvNeXt backbone
        convnext_pretrained: Whether to load pretrained ConvNeXt weights
        fusion_dim:      Shared dimension for fusion (all features projected to this)
        num_heads:       Number of attention heads in cross-attention fusion
        num_classes:     Number of output classes
        dropout:         Global dropout rate
    """

    def __init__(
        self,
        eva02_name: str = "eva02_small_patch14_336.mim_in22k_ft_in1k",
        eva02_pretrained: bool = True,
        convnext_name: str = "convnextv2_base.fcmae_ft_in22k_in1k_384",
        convnext_pretrained: bool = True,
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim

        # ----- Branch A: EVA-02 (Global/Contextual) ----- #
        from models.sota25_backbone import Sota2025Backbone
        self.branch_eva = Sota2025Backbone(
            model_name=eva02_name,
            pretrained=eva02_pretrained
        )
        eva_dim = self.branch_eva.num_features

        # ----- Branch B: ConvNeXt V3 (Local/Texture) ----- #
        from models.convnextv3_backbone import ConvNeXtV3Backbone
        self.branch_conv = ConvNeXtV3Backbone(
            model_name=convnext_name,
            pretrained=convnext_pretrained,
            embed_dim=fusion_dim,
            dropout=dropout,
            multi_scale=True,   # ↑ Enabled — richer multi-scale local texture features
        )
        conv_dim = self.branch_conv.num_features

        # ----- Feature Projections ----- #
        # Project EVA-02 features to fusion_dim if needed
        self.proj_eva = (
            nn.Linear(eva_dim, fusion_dim) if eva_dim != fusion_dim
            else nn.Identity()
        )
        # ConvNeXt already projects to fusion_dim via its FeatureProjector
        self.proj_conv = (
            nn.Linear(conv_dim, fusion_dim) if conv_dim != fusion_dim
            else nn.Identity()
        )

        # ----- Cross-Attention Fusion ----- #
        self.fusion = MultiScaleCrossAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ----- Gated Residual Combination ----- #
        self.gate = DualBranchGatedFusion(
            embed_dim=fusion_dim,
            dropout=dropout,
        )

        # ----- Classifier Head ----- #
        self.classifier = ClassifierHead(
            in_features=fusion_dim,
            hidden_features=fusion_dim // 2,
            num_classes=num_classes,
            dropout=dropout + 0.1,  # Slightly higher dropout in head
        )

        # Log total params
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print(f"\n{'='*70}")
        print(f"[DualBranchFusion] Model Initialized")
        print(f"  Branch A (EVA-02):   dim={eva_dim} → projected to {fusion_dim}")
        print(f"  Branch B (ConvNeXt): dim={conv_dim} → projected to {fusion_dim}")
        print(f"  Fusion dim:          {fusion_dim}")
        print(f"  Total params:        {total_params:.1f}M")
        print(f"  Trainable params:    {trainable_params:.1f}M")
        print(f"  Num classes:         {num_classes}")
        print(f"{'='*70}\n")

    def forward(self, img_orig: torch.Tensor, img_seg: torch.Tensor):
        """
        Forward pass through both branches with fusion.
        
        Args:
            img_orig: Original images (B, 3, H, W) — fed to Mamba branch
            img_seg:  Segmented images (B, 3, H, W) — fed to ConvNeXt branch
            
        Returns:
            logits:       (B, num_classes) classification logits
            attn_weights: Attention weights from cross-attention (for visualization)
        """
        # Branch A: EVA-02 on original image (global dependencies)
        feat_eva = self.branch_eva(img_orig)     # (B, eva_dim)
        feat_eva = self.proj_eva(feat_eva)      # (B, fusion_dim)

        # Branch B: ConvNeXt V3 on segmented image (local textures)
        feat_conv = self.branch_conv(img_seg)        # (B, fusion_dim)
        feat_conv = self.proj_conv(feat_conv)         # (B, fusion_dim)

        # Cross-attention fusion
        fused, attn_weights = self.fusion(feat_eva, feat_conv)  # (B, fusion_dim)

        # Gated residual combination
        combined = self.gate(fused, feat_eva, feat_conv)  # (B, fusion_dim)

        # Classification
        logits = self.classifier(combined)  # (B, num_classes)

        return logits, attn_weights

    def get_eva_params(self):
        """Returns EVA-02 backbone parameters only (for slow differential LR)."""
        return list(self.branch_eva.parameters())

    def get_convnext_params(self):
        """Returns ConvNeXt backbone parameters only (for slow differential LR)."""
        return list(self.branch_conv.parameters())

    def get_head_params(self):
        """Returns fusion + projection + classifier parameters (higher LR).
        
        proj_eva and proj_conv are randomly initialized — they need head LR (1e-4)
        not backbone LR (2e-5 / 1e-5) to learn effectively.
        """
        params = (
            list(self.proj_eva.parameters()    if isinstance(self.proj_eva,  nn.Linear) else [])
            + list(self.proj_conv.parameters() if isinstance(self.proj_conv, nn.Linear) else [])
            + list(self.fusion.parameters())
            + list(self.gate.parameters())
            + list(self.classifier.parameters())
        )
        return params
