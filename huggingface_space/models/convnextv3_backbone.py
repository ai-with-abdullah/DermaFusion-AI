"""
ConvNeXt V3-Style Backbone
==========================
Wraps a timm ConvNeXt V2 Large model with modernized components:
  - StarReLU activation replacement (optional)
  - Feature projection to arbitrary embed_dim
  - Multi-scale feature extraction support

Uses `convnextv2_large.fcmae_ft_in22k_in1k_384` as the base pretrained model.
When a true ConvNeXt V3 checkpoint becomes available in timm, it can be
swapped in via config with zero code changes.
"""

import torch
import torch.nn as nn
import timm


class StarReLU(nn.Module):
    """
    StarReLU activation: s * relu(x)^2 + b
    From MetaFormer (2023). More expressive than GELU with lower compute cost.
    Used as a drop-in modernization for ConvNeXt V3 style.
    """
    def __init__(self, scale_value=1.0, bias_value=-0.0341):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_value))
        self.bias = nn.Parameter(torch.tensor(bias_value))

    def forward(self, x):
        return self.scale * torch.relu(x) ** 2 + self.bias


class FeatureProjector(nn.Module):
    """
    Projects backbone features to a target embedding dimension with
    LayerNorm + Linear + GELU + Linear (bottleneck projector).
    """
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        mid_features = max(out_features, in_features // 2)
        self.proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, mid_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_features, out_features),
        )

    def forward(self, x):
        return self.proj(x)


class ConvNeXtV3Backbone(nn.Module):
    """
    ConvNeXt V3-Style backbone for the local/texture branch.
    
    Wraps a pretrained ConvNeXt V2 model from timm, extracting:
      - Global feature vector (B, embed_dim) via pooled features + projection
      - Multi-scale feature maps from intermediate stages (optional, for fusion)
    
    Args:
        model_name:   timm model identifier (default: convnextv2_large)
        pretrained:   Whether to load pretrained weights
        embed_dim:    Output embedding dimension after projection
        dropout:      Dropout in the feature projector
        multi_scale:  If True, also extract intermediate stage features
    """

    def __init__(
        self,
        model_name: str = "convnextv2_large.fcmae_ft_in22k_in1k_384",
        pretrained: bool = True,
        embed_dim: int = 512,
        dropout: float = 0.1,
        multi_scale: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = embed_dim  # Compatibility with existing code
        self.multi_scale = multi_scale

        # Main backbone (global pooled features)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head, returns pooled features
        )
        backbone_dim = self.backbone.num_features  # e.g. 1536 for ConvNeXt V2 Large

        # Feature projector: backbone_dim → embed_dim
        self.projector = FeatureProjector(backbone_dim, embed_dim, dropout=dropout)

        # Multi-scale feature extractor (optional)
        if multi_scale:
            self.backbone_ms = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),  # All 4 stages
            )
            # Channel dims for ConvNeXt V2 Large stages: [192, 384, 768, 1536]
            self.stage_channels = self.backbone_ms.feature_info.channels()

            # 1x1 convolutions to project each stage to a common dimension
            self.stage_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, embed_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),
                )
                for ch in self.stage_channels
            ])

        # Log info
        num_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[ConvNeXtV3] Initialized | base={model_name}, "
              f"backbone_dim={backbone_dim}, embed_dim={embed_dim}, "
              f"multi_scale={multi_scale}, params={num_params:.1f}M")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            If multi_scale=False: (B, embed_dim) global feature vector
            If multi_scale=True:  tuple(global_feat, [stage_feats])
        """
        # Global feature vector
        feat = self.backbone(x)       # (B, backbone_dim)
        feat = self.projector(feat)    # (B, embed_dim)

        if not self.multi_scale:
            return feat

        # Multi-scale features
        stage_feats = self.backbone_ms(x)  # List of (B, C_i, H_i, W_i)
        projected_feats = []
        for i, sf in enumerate(stage_feats):
            projected_feats.append(self.stage_projs[i](sf))  # (B, embed_dim, H_i, W_i)

        return feat, projected_feats

    def get_backbone_params(self):
        """Returns backbone parameters for differential learning rate."""
        return list(self.backbone.parameters())

    def get_head_params(self):
        """Returns projector parameters (higher LR)."""
        return list(self.projector.parameters())
