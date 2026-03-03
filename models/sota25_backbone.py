import torch
import torch.nn as nn
import timm

class Sota2025Backbone(nn.Module):
    """
    2025 Vision Backbone (Supports ConvNeXt-V2, EVA-02, and Mamba derivatives).
    Extracts high-resolution global and local texture features.
    """
    def __init__(self, model_name="convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=True):
        super().__init__()
        
        # We instantiate a highly optimized 2025 SOTA backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,       # Remove classifier head
        )
        
        # In ConvNeXtV2 or EVA02, timm usually provides num_features
        self.num_features = self.backbone.num_features
        
        # Explicit adaptive pooling in case we use varying 2025 architectures
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """
        Input: (B, 3, 384, 384) 
        Output: (B, EmbeddingDim) vector
        """
        # EVA-02 specific patch size handling
        if "eva02" in self.backbone.default_cfg['architecture']:
            # Resize image to match EVA-02 required input size (usually 336x336)
            target_size = self.backbone.default_cfg['input_size'][-2:]
            if x.shape[-2:] != target_size:
                x = torch.nn.functional.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
                
        # timm forward_features returns either (B, C, H, W) for CNNs 
        # or (B, SeqLen, C) for ViTs. We rely on timm's generic forward without head
        features = self.backbone(x)
        return features

    def project_to_dim(self, features, dim):
        if features.shape[1] != dim:
             proj = nn.Linear(features.shape[1], dim).to(features.device)
             return proj(features)
        return features
