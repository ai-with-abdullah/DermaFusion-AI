import torch
import torch.nn as nn
import timm

class Sota2025Backbone(nn.Module):
    """
    2025 Vision Backbone (Supports ConvNeXt-V2, EVA-02, and Mamba derivatives).
    Extracts high-resolution global and local texture features.

    Supported variants:
        EVA-02 Small: eva02_small_patch14_336  → input 336×336, dim=384
        EVA-02 Large: eva02_large_patch14_448  → input 448×448, dim=1024  ← ACTIVE
    """
    def __init__(self, model_name="eva02_large_patch14_448.mim_in22k_ft_in22k_in1k", pretrained=True):
        super().__init__()

        # Instantiate the backbone; num_classes=0 removes the classifier head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # timm exposes num_features automatically for EVA-02 and ConvNeXt V2
        self.num_features = self.backbone.num_features

        # Adaptive pooling fallback for CNN-style backbones that return (B,C,H,W)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Input: (B, 3, H, W) — auto-resized to the model's native resolution.
               EVA-02 Small: 336×336  |  EVA-02 Large: 448×448
        Output: (B, EmbeddingDim) vector.
               EVA-02 Small: dim=384  |  EVA-02 Large: dim=1024
        """
        # Dynamically read target size from model config so this works for
        # BOTH eva02_small_patch14_336 (336px) AND eva02_large_patch14_448 (448px)
        if "eva02" in self.backbone.default_cfg['architecture']:
            target_size = self.backbone.default_cfg['input_size'][-2:]  # e.g. (448, 448)
            if x.shape[-2:] != target_size:
                x = torch.nn.functional.interpolate(
                    x, size=target_size, mode='bicubic', align_corners=False
                )

        # timm forward returns (B, SeqLen, C) for ViTs or (B, C, H, W) for CNNs.
        # With num_classes=0, timm applies global pool automatically for ViTs -> (B, D)
        features = self.backbone(x)
        return features

    # NOTE: Feature projection (backbone_dim → fusion_dim) is handled by
    # DualBranchFusionClassifier.proj_eva using a trained nn.Linear.
    # project_to_dim() has been removed — it created a new untrained Linear
    # on every call and discarded it, which would apply a random projection.
