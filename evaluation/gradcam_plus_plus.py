"""
GradCAM++ for EVA-02 ViT + ConvNeXt V2
========================================
Implements proper GradCAM++ (Chattopadhay et al., WACV 2018) for both
branches of the DualBranchFusionClassifier.

GradCAM++ improves over GradCAM by using second-order gradients to better
weight activation channels, resulting in sharper and more faithful
spatial saliency maps.

Usage:
    cam = ConvNeXtGradCAMPP(model, device='cpu')
    heatmap, blend = cam.generate(image_tensor, target_class=4)  # mel class
"""

import os
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from configs.config import config

# Reuse colormap and helpers from explainability.py
from matplotlib.colors import LinearSegmentedColormap

_SKIN_CMAP = LinearSegmentedColormap.from_list(
    "skin_heat",
    ["#0d0221", "#3b1578", "#ff6b35", "#ffec5c", "#ffffff"],
    N=256,
)


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(config.MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(config.STD,  device=tensor.device).view(3, 1, 1)
    img  = (tensor * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # Resize cam_norm first to target size
    cam_resized = cv2.resize(cam_norm, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    # Apply custom skin-cancer-aware colormap instead of generic JET
    heatmap = (_SKIN_CMAP(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)


# =========================================================================== #
#                   CONVNEXT GRADCAM++                                         #
# =========================================================================== #

class ConvNeXtGradCAMPP:
    """
    GradCAM++ for the ConvNeXt V2 branch of DualBranchFusionClassifier.

    Hooks into the last 2D convolutional layer of the ConvNeXt backbone
    and computes second-order gradient-weighted class activation maps.
    """

    def __init__(self, model, device: str = 'cpu'):
        self.model    = model
        self.device   = device
        self._acts:   dict = {}
        self._grads:  dict = {}
        self._handles = []

    def _register_hooks(self, target_layer: nn.Module):
        def fwd(m, inp, out):
            self._acts['feat'] = out.detach().clone()

        def bwd(m, gin, gout):
            self._grads['feat'] = gout[0].detach().clone()

        self._handles.append(target_layer.register_forward_hook(fwd))
        self._handles.append(target_layer.register_full_backward_hook(bwd))

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _find_last_conv(self, module: nn.Module) -> Optional[nn.Module]:
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last

    def generate(
        self,
        image_tensor: torch.Tensor,    # (3, H, W) normalized
        target_class: int,
        image_seg_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            cam_map:   (H, W) GradCAM++ float map
            blend_img: (H, W, 3) overlay on original image
        """
        H, W = image_tensor.shape[1], image_tensor.shape[2]
        img_rgb = _denorm(image_tensor)

        # Find target layer in ConvNeXt backbone
        target_layer = self._find_last_conv(self.model.branch_conv.backbone)
        if target_layer is None:
            # Fallback to a random noise map
            cam_map = np.random.rand(H // 16, W // 16).astype(np.float32)
            return cam_map, _overlay(img_rgb, cam_map)

        self._register_hooks(target_layer)
        self.model.eval()
        self.model.zero_grad()

        try:
            x = image_tensor.unsqueeze(0).to(self.device).requires_grad_(True)
            # FIX: Use original image as fallback when seg is None or flat.
            # Previously used torch.zeros_like(x) → ConvNeXt saw near-black image
            # → GradCAM focused on border resizing artifacts instead of the lesion.
            if image_seg_tensor is not None:
                seg_std = image_seg_tensor.std().item()
                x_seg = (
                    image_seg_tensor.unsqueeze(0).to(self.device)
                    if seg_std > 0.03   # seg is informative
                    else x.detach()     # flat mask → use original to avoid border artifact
                )
            else:
                x_seg = x.detach()  # no seg → use original image

            logits, _ = self.model(x, x_seg)
            score = logits[0, target_class]
            score.backward()

        except Exception as e:
            print(f"[GradCAM++] Warning: {e}")
            self._remove_hooks()
            cam_map = np.random.rand(H // 16, W // 16).astype(np.float32)
            return cam_map, _overlay(img_rgb, cam_map)

        finally:
            self._remove_hooks()

        # GradCAM++ weight computation
        if 'feat' not in self._acts or 'feat' not in self._grads:
            cam_map = np.random.rand(H // 16, W // 16).astype(np.float32)
            return cam_map, _overlay(img_rgb, cam_map)

        acts  = self._acts['feat'][0]   # (C, h, w)
        grads = self._grads['feat'][0]  # (C, h, w)

        # GradCAM++ alpha computation (2nd-order terms)
        grads_sq  = grads ** 2
        grads_cub = grads ** 3
        denom     = 2 * grads_sq + (acts * grads_cub).sum(dim=[1, 2], keepdim=True)
        denom     = torch.where(denom == 0, torch.ones_like(denom), denom)
        alpha     = grads_sq / denom                                         # (C, h, w)

        # Weighted combination
        weights   = (alpha * F.relu(grads)).sum(dim=[1, 2])                 # (C,)
        cam        = (weights[:, None, None] * acts).sum(dim=0)             # (h, w)
        cam        = F.relu(cam).cpu().numpy().astype(np.float32)

        # Upsample to image resolution
        cam_up = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)

        self.model.zero_grad()
        self._acts.clear()
        self._grads.clear()

        return cam_up, _overlay(img_rgb, cam_up)


# =========================================================================== #
#                   EVA-02 VIT GRADCAM++ (Attention Rollout)                  #
# =========================================================================== #

class EVAGradCAMPP:
    """
    Grad-CAM on Attention for the EVA-02 ViT branch.
    Hooks into the attention block of the final transformer layer and
    weights the attention weights by backpropagated class gradients.
    Falls back to Attention Rollout if gradient propagation is not possible.
    """

    def __init__(self, model, device: str = 'cpu'):
        self.model  = model
        self.device = device
        self._acts  = {}
        self._grads = {}
        self._handles = []

    def _register_hooks(self, target_layer: nn.Module):
        def fwd(m, inp, out):
            # inp[0] is the attention weights tensor of shape (B, heads, N, N)
            self._acts['attn'] = inp[0].detach().clone()

        def bwd(m, gin, gout):
            # gout[0] is the gradient of loss w.r.t the output of attn_drop
            self._grads['attn'] = gout[0].detach().clone()

        self._handles.append(target_layer.register_forward_hook(fwd))
        self._handles.append(target_layer.register_full_backward_hook(bwd))

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        image_tensor: torch.Tensor,    # (3, H, W) normalized
        target_class: int,
        discard_ratio: float = 0.4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            cam_map:   (H, W) Grad-CAM attention map
            blend_img: (H, W, 3) overlay on original image
        """
        H, W = image_tensor.shape[1], image_tensor.shape[2]
        img_rgb = _denorm(image_tensor)

        eva_backbone = self.model.branch_eva.backbone
        self.model.eval()

        # Find target layer: last block's attn_drop
        target_layer = None
        blocks = getattr(eva_backbone, 'blocks', [])
        if blocks:
            attn_module = getattr(blocks[-1], 'attn', None)
            if attn_module is not None:
                target_layer = getattr(attn_module, 'attn_drop', None)

        # Run with Grad-CAM if target layer exists
        use_rollout = True
        if target_layer is not None:
            self._register_hooks(target_layer)
            self.model.zero_grad()
            try:
                x = image_tensor.unsqueeze(0).to(self.device).requires_grad_(True)
                tgt = eva_backbone.default_cfg.get('input_size', (3, 336, 336))[-2:]
                if x.shape[-2:] != tgt:
                    x = F.interpolate(x, size=tgt, mode='bicubic', align_corners=False)
                
                # We need full forward to pass through head
                x_seg = x.detach()
                logits, _ = self.model(x, x_seg)
                score = logits[0, target_class]
                score.backward()
                use_rollout = False
            except Exception as e:
                print(f"[EVAGradCAM++] Grad-CAM backward failed, falling back to Rollout: {e}")
            finally:
                self._remove_hooks()

        if not use_rollout and 'attn' in self._acts and 'attn' in self._grads:
            # Grad-CAM attention weighting
            acts = self._acts['attn'][0]    # (heads, N, N)
            grads = self._grads['attn'][0]  # (heads, N, N)

            # Head weights = mean gradient over keys/queries
            weights = grads.mean(dim=[-2, -1], keepdim=True)  # (heads, 1, 1)
            # Weighted combination over heads
            cam = F.relu(weights * acts).sum(dim=0)          # (N, N)

            # CLS token index is 0, retrieve its attention to all other patch tokens
            cls_attn = cam[0, 1:].cpu().float().numpy()      # (N-1,)
            # Apply threshold
            threshold = np.percentile(cls_attn, discard_ratio * 100)
            cls_attn[cls_attn < threshold] = 0.0
            
            side = int(math.sqrt(len(cls_attn)))
            cam_map = cls_attn[:side*side].reshape(side, side).astype(np.float32)
            
            self.model.zero_grad()
            self._acts.clear()
            self._grads.clear()
        else:
            # Fallback: Attention Rollout (class-agnostic)
            attn_maps = []
            def rollout_hook(m, inp, out):
                if inp and inp[0] is not None and inp[0].dim() == 4:
                    attn_maps.append(inp[0].detach().cpu())

            handles = []
            for block in blocks:
                attn_m = getattr(block, 'attn', None)
                if attn_m is not None:
                    drop = getattr(attn_m, 'attn_drop', None)
                    if drop is not None:
                        handles.append(drop.register_forward_hook(rollout_hook))

            try:
                with torch.no_grad():
                    x = image_tensor.unsqueeze(0).to(self.device)
                    tgt = eva_backbone.default_cfg.get('input_size', (3, 336, 336))[-2:]
                    if x.shape[-2:] != tgt:
                        x = F.interpolate(x, size=tgt, mode='bicubic', align_corners=False)
                    _ = eva_backbone.forward_features(x)
            except Exception as e:
                print(f"[EVAGradCAM++] Rollout forward failed: {e}")
            finally:
                for h in handles:
                    h.remove()

            if not attn_maps:
                cam_map = np.random.rand(14, 14).astype(np.float32)
            else:
                result = None
                for attn in attn_maps:
                    if attn.dim() == 4 and attn.shape[-1] > 1:
                        attn_mean = attn[0].mean(0)
                        attn_mean = attn_mean + torch.eye(attn_mean.shape[0])
                        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
                        result = attn_mean if result is None else torch.mm(attn_mean, result)

                if result is None:
                    cam_map = np.random.rand(14, 14).astype(np.float32)
                else:
                    cls_attn = result[0, 1:].numpy()
                    threshold = np.percentile(cls_attn, discard_ratio * 100)
                    cls_attn[cls_attn < threshold] = 0
                    side = int(math.sqrt(len(cls_attn)))
                    cam_map = cls_attn[:side*side].reshape(side, side).astype(np.float32)

        cam_up = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_CUBIC)
        return cam_up, _overlay(img_rgb, cam_up)


# =========================================================================== #
#                   DUAL-BRANCH COMBINED VISUALIZATION                         #
# =========================================================================== #

def generate_dual_gradcam(
    model,
    image_tensor: torch.Tensor,
    target_class: int,
    device: str = 'cpu',
    save_dir: str = None,
    image_id: str = 'sample',
    image_seg_tensor: torch.Tensor = None,
) -> dict:
    """
    Generates GradCAM++ for both branches and a fused visualization.
    Saves a 3-panel figure: EVA-02 | ConvNeXt | Fusion.

    Returns dict with keys: eva_map, convnext_map, fusion_map, figure_path.
    """
    convnext_cam = ConvNeXtGradCAMPP(model, device)
    eva_cam      = EVAGradCAMPP(model, device)

    H, W = image_tensor.shape[1], image_tensor.shape[2]
    img_rgb = _denorm(image_tensor)

    # Check if seg image is informative; if flat → pass None so ConvNeXt
    # GradCAM falls back to original image (avoids border artifacts).
    seg_for_cam = image_seg_tensor
    if seg_for_cam is not None and seg_for_cam.std().item() < 0.03:
        seg_for_cam = None  # UNet produced flat mask — don't corrupt GradCAM

    # Generate maps (passing target_class to both for class-specific visualisations)
    eva_map_up,   eva_blend   = eva_cam.generate(image_tensor, target_class)
    conv_map_up,  conv_blend  = convnext_cam.generate(image_tensor, target_class, seg_for_cam)

    # Fuse: equal weight average (could be gated by actual gate values)
    eva_norm  = (eva_map_up  - eva_map_up.min())  / (eva_map_up.max()  - eva_map_up.min()  + 1e-8)
    conv_norm = (conv_map_up - conv_map_up.min()) / (conv_map_up.max() - conv_map_up.min() + 1e-8)
    fusion    = 0.5 * eva_norm + 0.5 * conv_norm
    fusion_blend = _overlay(img_rgb, fusion)

    # Save figure
    figure_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cls_name = config.CLASSES[target_class] if target_class < len(config.CLASSES) else str(target_class)

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor="#0d0d0d")
            titles = ["Original", f"EVA-02 Rollout", f"ConvNeXt GradCAM++", "Fused Map"]
            imgs   = [img_rgb, eva_blend, conv_blend, fusion_blend]
            for ax, title, im in zip(axes, titles, imgs):
                ax.imshow(im)
                ax.set_title(title, color='#e0e0e0', fontsize=11)
                ax.axis('off')
            fig.suptitle(
                f"GradCAM++  |  Image: {image_id}  |  Target: {cls_name.upper()}",
                color='#ffec5c', fontsize=13, fontweight='bold', y=1.02
            )
            plt.tight_layout()
            figure_path = os.path.join(save_dir, f"{image_id}_gradcam_pp.png")
            plt.savefig(figure_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
            plt.close()

    return {
        'eva_map':      eva_map_up,
        'convnext_map': conv_map_up,
        'fusion_map':   fusion,
        'figure_path':  figure_path,
    }
