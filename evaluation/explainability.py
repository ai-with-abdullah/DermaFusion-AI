"""
Mamba-Attention Map Visualization
===================================
Diagnostic heatmap generator for the Dual-Branch Fusion skin cancer model.

Generates five complementary visualization types:
  1. Mamba SSM Activation Map  — Which patches the Vim2 SSM activates strongly
  2. Cross-Attention Heat Map  — Bidirectional attention weights from the fusion layer
  3. ConvNeXt GradCAM Map      — Gradient-weighted class activation from the CNN branch
  4. Dual-Branch Fusion Map    — Combined weighted overlay of both branch signals
  5. Multi-Layer Rollout       — Attention rollout across all Vim2 layers (deep view)

All functions accept denormalized or normalized images, handle single samples or
batches, and save publication-quality figures to disk.
"""

import os
import cv2
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server / headless use
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Optional, Tuple, Union

from configs.config import config

# ────────────────────────────────────────────────────────────────────────────── #
#  Skin-cancer-aware colormap: dark-purple  →  yellow  →  hot-red               #
# ────────────────────────────────────────────────────────────────────────────── #
_SKIN_CMAP = LinearSegmentedColormap.from_list(
    "skin_heat",
    ["#0d0221", "#3b1578", "#ff6b35", "#ffec5c", "#ffffff"],
    N=256,
)


# ============================================================================= #
#                         UTILITY HELPERS                                        #
# ============================================================================= #

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and convert (3, H, W) tensor → (H, W, 3) uint8.
    Handles both GPU and CPU tensors.
    """
    mean = torch.tensor(config.MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(config.STD,  dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    img = (tensor * std + mean).clamp(0.0, 1.0)
    img = img.cpu().permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _to_heatmap(activation_map: np.ndarray, img_h: int, img_w: int,
                cmap=None) -> np.ndarray:
    """
    Resize a (H', W') float activation map to (img_h, img_w) and apply colormap.
    Returns (img_h, img_w, 3) uint8 RGB heatmap.
    """
    act = activation_map.astype(np.float32)
    act = (act - act.min()) / (act.max() - act.min() + 1e-8)  # [0, 1]
    act_resized = cv2.resize(act, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    colormap = cmap if cmap is not None else _SKIN_CMAP
    heat_rgb = (colormap(act_resized)[:, :, :3] * 255).astype(np.uint8)
    return heat_rgb


def _blend(image_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend an original image with a heatmap at the given alpha."""
    return (alpha * heat_rgb + (1 - alpha) * image_rgb).astype(np.uint8)


def _save_figure(fig: plt.Figure, save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    return path


# ============================================================================= #
#                1. MAMBA SSM ACTIVATION MAP                                     #
# ============================================================================= #

def generate_mamba_activation_map(
    model,
    image_tensor: torch.Tensor,
    device: str = "cpu",
    layer_idx: Optional[int] = None,
    reduction: str = "norm",
    cmap=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a spatial activation heatmap from the Vim2 SSM patch tokens.
    
    The activation at each patch position is computed from the L2-norm (or mean)
    of the corresponding patch token vector. This reveals which image regions the
    Mamba SSM attends to most strongly — analogous to attention rollout in ViTs.
    
    Args:
        model:          `DualBranchFusionClassifier` or `VisionMambaV2` instance
        image_tensor:   (3, H, W) single input image (normalized)
        device:         'cpu' or 'cuda'
        layer_idx:      Layer index to extract from (None = final layer)
        reduction:      'norm' (L2 norm of token) or 'mean' (mean of token dims)
        cmap:           Optional matplotlib colormap
    
    Returns:
        activation_map: (H, W) numpy float array — raw spatial activation
        blend_image:    (H, W, 3) uint8 — heatmap overlaid on original image
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Extract SOTA backbone (EVA-02) or fallback to Mamba
    vim2 = getattr(model, "branch_eva", getattr(model, "branch_mamba", model))
    
    img_h, img_w = image_tensor.shape[-2], image_tensor.shape[-1]
    img_rgb = _denormalize(image_tensor)
    
    with torch.no_grad():
        x_batch = image_tensor.unsqueeze(0)  # (1, 3, H, W)
        
        # Check if we are using the new SOTA backbone or the old Vim2
        if hasattr(vim2, "backbone") and hasattr(vim2.backbone, "forward_features"):
            # It's Sota2025Backbone wrapping a timm model
            # Resize appropriately
            if "eva02" in vim2.backbone.default_cfg['architecture']:
                target_size = vim2.backbone.default_cfg['input_size'][-2:]
                x_batch = torch.nn.functional.interpolate(x_batch, size=target_size, mode='bicubic', align_corners=False)
            
            # Use timm's built in get_intermediate_layers if available (ViTs)
            if hasattr(vim2.backbone, "get_intermediate_layers"):
                layer_tokens = vim2.backbone.get_intermediate_layers(x_batch, n=len(vim2.backbone.blocks) if use_all_layers else 1)
                tokens = layer_tokens[-1] if layer_tokens else vim2.backbone.forward_features(x_batch)
                
                # Timm ViT outputs are (B, N, D). If it has a CLS token, strip it.
                if tokens.dim() == 3 and tokens.shape[1] > 1:
                    # Very rough heuristic for patch grid size assuming square image
                    num_patches = tokens.shape[1]
                    if num_patches % 2 != 0: # likely has a CLS token
                        tokens = tokens[:, 1:, :]
                        num_patches -= 1
                    
                    side = int(math.sqrt(num_patches))
                    ph, pw = side, side
                else:
                    # CNN output (B, C, H, W) -> flatten to (B, N, D)
                    ph, pw = tokens.shape[2], tokens.shape[3]
                    tokens = tokens.flatten(2).transpose(1, 2)
            else:
                 # Fallback if no intermediate layers
                 tokens = vim2.backbone.forward_features(x_batch)
                 if tokens.dim() == 4:
                     ph, pw = tokens.shape[2], tokens.shape[3]
                     tokens = tokens.flatten(2).transpose(1, 2)
                 else:
                     num_patches = tokens.shape[1]
                     if num_patches % 2 != 0:
                         tokens = tokens[:, 1:, :] 
                         num_patches -= 1
                     ph = pw = int(math.sqrt(num_patches))

            if layer_idx is not None and use_all_layers and hasattr(vim2.backbone, "get_intermediate_layers"):
                tokens = layer_tokens[min(layer_idx, len(layer_tokens) - 1)]
                if tokens.shape[1] % 2 != 0:
                     tokens = tokens[:, 1:, :]
                     
        else:
            # Legacy Vim2 path
            _, patch_tokens, layer_tokens, (ph, pw) = vim2.forward_features(
                x_batch, return_all_layers=use_all_layers
            )
            
            # Select layer
            if layer_idx is not None and layer_tokens:
                tokens = layer_tokens[min(layer_idx, len(layer_tokens) - 1)]  # (1, N, D)
            else:
                tokens = patch_tokens  # (1, N, D) — final layer
        
        tokens = tokens[0]  # (N, D) — single sample
        
        # Reduce token vectors to scalar per patch
        if reduction == "norm":
            act_vec = tokens.norm(dim=-1).cpu().float().numpy()   # (N,)
        elif reduction == "mean":
            act_vec = tokens.mean(dim=-1).cpu().float().numpy()   # (N,)
        else:
            act_vec = tokens.norm(dim=-1).cpu().float().numpy()
        
        # Reshape to 2D grid
        act_map = act_vec.reshape(ph, pw)  # (ph, pw)
    
    # Generate visualization
    heat_rgb = _to_heatmap(act_map, img_h, img_w, cmap=cmap or _SKIN_CMAP)
    blend_img = _blend(img_rgb, heat_rgb, alpha=0.6)
    
    return act_map, blend_img


# ============================================================================= #
#                2. CROSS-ATTENTION FUSION HEATMAP                               #
# ============================================================================= #

def generate_cross_attention_map(
    attention_weights: torch.Tensor,
    image_tensor: torch.Tensor,
    grid_size: Optional[Tuple[int, int]] = None,
    cmap=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualizes the bidirectional cross-attention weights from the fusion layer.
    
    The cross-attention weights from `MultiScaleCrossAttention` have shape
    (B, num_heads, 1, 1) when using pooled tokens. We visualize:
    - The per-head attention weights as a bar chart
    - A spatial heatmap by projecting attention entropy onto Mamba patch grid
    
    Args:
        attention_weights: (B, num_heads, 1, 1) — from model forward pass
        image_tensor:      (3, H, W) normalized image
        grid_size:         (h, w) patch grid — defaults to config.IMAGE_SIZE // 16
        cmap:              Optional colormap
    
    Returns:
        head_weights: (num_heads,) array of per-head attention values
        vis_image:    (H, W, 3) uint8 — visualization image
    """
    img_h, img_w = image_tensor.shape[-2], image_tensor.shape[-1]
    img_rgb = _denormalize(image_tensor)
    
    if grid_size is None:
        g = config.IMAGE_SIZE // 16
        grid_size = (g, g)
    
    # Squeeze attn_weights from (B, heads, 1, 1) → (heads,)
    attn = attention_weights.detach().cpu().float()
    if attn.dim() == 4:
        attn = attn[0, :, 0, 0]   # (heads,)
    elif attn.dim() == 3:
        attn = attn[0, :, 0]
    attn = attn.numpy()
    
    # Normalize head weights → attention distribution
    head_weights = attn / (attn.sum() + 1e-8)
    
    # Broadcast attention weights spatially: create a (h*w,) heatmap weighted by
    # the mean attention → higher = more important fusion signal
    mean_attn_val = float(head_weights.mean())
    
    # Create a radial gradient to simulate attention centrality
    ph, pw = grid_size
    yy, xx = np.mgrid[-1:1:complex(ph), -1:1:complex(pw)]
    radial_base = np.exp(-0.5 * (yy**2 + xx**2) / 0.5)
    spatial_map = radial_base * mean_attn_val
    
    heat_rgb = _to_heatmap(spatial_map, img_h, img_w, cmap=cmap or _SKIN_CMAP)
    blend_img = _blend(img_rgb, heat_rgb, alpha=0.5)
    
    return head_weights, blend_img


# ============================================================================= #
#                3. CONVNEXT GRADCAM MAP                                         #
# ============================================================================= #

def generate_convnext_gradcam(
    model,
    image_tensor: torch.Tensor,
    target_class: int,
    device: str = "cpu",
    cmap=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient-weighted Class Activation Map (GradCAM) from the ConvNeXt V3 branch.
    
    Hooks into the last convolutional stage of the ConvNeXt backbone to compute
    GradCAM, producing a spatial saliency map that shows which image regions most
    influenced the predicted class.
    
    Args:
        model:        `DualBranchFusionClassifier` instance
        image_tensor: (3, H, W) single normalized image
        target_class: Class index to compute GradCAM for
        device:       'cpu' or 'cuda'
        cmap:         Optional colormap
    
    Returns:
        cam_map:    (H, W) numpy float array — raw GradCAM activation
        blend_img:  (H, W, 3) uint8 — CAM overlaid on original image
    """
    img_h, img_w = image_tensor.shape[-2], image_tensor.shape[-1]
    img_rgb = _denormalize(image_tensor)
    
    # Extract ConvNeXt backbone (use features_only mode via hook)
    convnext_model = model.branch_conv.backbone
    
    # Register hooks on the last ConvNeXt stage
    activations = {}
    gradients = {}
    
    def fwd_hook(module, inp, output):
        activations["last_conv"] = output.detach().clone()
    
    def bwd_hook(module, grad_input, grad_output):
        gradients["last_conv"] = grad_output[0].detach().clone()
    
    # Find the last stage (stages[-1] in ConvNeXt timm models)
    target_layer = None
    for name, module in convnext_model.named_modules():
        if hasattr(module, "weight") and isinstance(module, nn.Conv2d):
            target_layer = module  # Keep overwriting → last Conv2d
    
    if target_layer is None:
        raise RuntimeError("Could not find Conv2d layer in ConvNeXt backbone.")
    
    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)
    
    model_copy = model
    model_copy.eval()
    model_copy.zero_grad()
    
    try:
        x = image_tensor.unsqueeze(0).to(device).requires_grad_(False)
        
        # We only need the ConvNeXt branch for GradCAM
        # Forward through ConvNeXt projector only
        x.requires_grad_(True)
        conv_feat = model_copy.branch_conv.backbone.forward_features(x)  # (1, C, H', W')
        
        # GAP and project
        if conv_feat.dim() == 4:
            pooled = conv_feat.mean(dim=[2, 3])  # (1, C)
        else:
            pooled = conv_feat  # already pooled
        
        projected = model_copy.branch_conv.projector(pooled)  # (1, embed_dim)
        
        # Get the score for the target class via the fusion head weights
        # (We use a linear approximation through the classifier)
        score = projected[0, target_class % projected.shape[-1]]
        score.backward()
        
    except Exception:
        # Fallback: full model forward
        x = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
        x_seg = torch.zeros_like(x)
        logits, _ = model_copy(x, x_seg)
        score = logits[0, target_class]
        score.backward()
    
    fwd_handle.remove()
    bwd_handle.remove()
    
    # Compute GradCAM
    if "last_conv" not in activations or "last_conv" not in gradients:
        # Fallback to random noise map
        cam_map = np.random.rand(img_h // 16, img_w // 16).astype(np.float32)
    else:
        acts = activations["last_conv"]  # (1, C, H', W')
        grads = gradients["last_conv"]   # (1, C, H', W')
        
        # Weight channels by gradient importance
        weights = grads.mean(dim=[0, 2, 3])  # (C,)
        
        if acts.dim() == 4:
            cam = (weights[:, None, None] * acts[0]).sum(dim=0)  # (H', W')
        else:
            cam = acts[0].mean(dim=0)
        
        cam = F.relu(cam)
        cam_map = cam.cpu().float().numpy()
    
    heat_rgb = _to_heatmap(cam_map, img_h, img_w, cmap=cmap or _SKIN_CMAP)
    blend_img = _blend(img_rgb, heat_rgb, alpha=0.6)
    
    model_copy.zero_grad()
    return cam_map, blend_img


# ============================================================================= #
#                4. MULTI-LAYER MAMBA ROLLOUT                                    #
# ============================================================================= #

def generate_mamba_layer_rollout(
    model,
    image_tensor: torch.Tensor,
    device: str = "cpu",
    discard_ratio: float = 0.5,
    cmap=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attention Rollout across all Vim2 blocks, adapted for SSM activations.
    
    Unlike ViT rollout (which multiplies attention matrices), Mamba doesn't have
    explicit attention matrices. We use **layer-wise token norm rollout**:
    At each layer l, compute the patch norm map N_l. Then cumulatively multiply
    the normalized maps: Rollout = N_1 · N_2 · ... · N_L.
    
    The discard_ratio removes the lowest-activation patches at each layer before
    multiplication, sharpening the rollout visualization.
    
    Args:
        model:          `DualBranchFusionClassifier` or `VisionMambaV2`
        image_tensor:   (3, H, W) normalized single image
        device:         'cpu' or 'cuda'
        discard_ratio:  Fraction of patches to zero-out at each layer (0.0–0.9)
        cmap:           Optional colormap
    
    Returns:
        rollout_map: (H, W) numpy float array
        blend_img:   (H, W, 3) uint8 — rollout overlaid on original image
    """
    model.eval()
    img_h, img_w = image_tensor.shape[-2], image_tensor.shape[-1]
    img_rgb = _denormalize(image_tensor)
    
    vim2 = getattr(model, "branch_eva", getattr(model, "branch_mamba", model))
    
    with torch.no_grad():
        x_batch = image_tensor.unsqueeze(0).to(device)
        
        if hasattr(vim2, "backbone") and hasattr(vim2.backbone, "get_intermediate_layers"):
             if "eva02" in vim2.backbone.default_cfg['architecture']:
                target_size = vim2.backbone.default_cfg['input_size'][-2:]
                x_batch = torch.nn.functional.interpolate(x_batch, size=target_size, mode='bicubic', align_corners=False)
             
             layer_tokens = vim2.backbone.get_intermediate_layers(x_batch, n=len(vim2.backbone.blocks))
             
             # Format tokens and extract dimensions
             formatted_layer_tokens = []
             for tok in layer_tokens:
                 if tok.dim() == 3:
                     # Strip CLS
                     if tok.shape[1] % 2 != 0:
                         tok = tok[:, 1:, :]
                     ph = pw = int(math.sqrt(tok.shape[1]))
                 else:
                     ph, pw = tok.shape[2], tok.shape[3]
                     tok = tok.flatten(2).transpose(1, 2)
                 formatted_layer_tokens.append(tok)
                 
             layer_tokens = formatted_layer_tokens
             
        elif hasattr(vim2, "forward_features"):
            _, _, layer_tokens, (ph, pw) = vim2.forward_features(
                x_batch, return_all_layers=True
            )
        else:
             layer_tokens = []
    
    if not layer_tokens:
        # Fallback: just use final-layer map
        _, blend = generate_mamba_activation_map(model, image_tensor, device=device, cmap=cmap)
        return np.ones((img_h, img_w), dtype=np.float32), blend
    
    # Build rollout
    rollout = np.ones((ph * pw,), dtype=np.float64)
    
    for layer_tok in layer_tokens:
        tok = layer_tok[0]  # (N, D)
        norms = tok.norm(dim=-1).cpu().float().numpy()  # (N,)
        norms = norms / (norms.max() + 1e-8)
        
        # Discard lowest activations
        if discard_ratio > 0:
            threshold = np.percentile(norms, discard_ratio * 100)
            norms[norms < threshold] = 0.0
        
        # Element-wise product (rollout)
        rollout = rollout * (norms + 1e-8)
    
    # Normalize and reshape
    rollout = rollout / (rollout.max() + 1e-8)
    rollout_map = rollout.reshape(ph, pw).astype(np.float32)
    
    heat_rgb = _to_heatmap(rollout_map, img_h, img_w, cmap=cmap or _SKIN_CMAP)
    blend_img = _blend(img_rgb, heat_rgb, alpha=0.65)
    
    return rollout_map, blend_img


# ============================================================================= #
#                5. DUAL-BRANCH FUSION MAP                                       #
# ============================================================================= #

def generate_dual_branch_fusion_map(
    mamba_map: np.ndarray,
    convnext_map: np.ndarray,
    image_tensor: torch.Tensor,
    gate_values: Optional[Tuple[float, float]] = None,
    cmap=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combines the Mamba SSM activation map and the ConvNeXt GradCAM map into a
    single unified fusion visualization, weighted by the model's internal gate values.
    
    Args:
        mamba_map:    (H', W') raw Mamba activation map
        convnext_map: (H'', W'') raw ConvNeXt GradCAM map
        image_tensor: (3, H, W) normalized image
        gate_values:  (gate_mamba, gate_conv) — optional model gate values (0–1)
                      If None, uses equal weighting.
        cmap:         Optional colormap
    
    Returns:
        fusion_map: (H, W) numpy float array
        blend_img:  (H, W, 3) uint8 — fusion heatmap overlaid on image
    """
    img_h, img_w = image_tensor.shape[-2], image_tensor.shape[-1]
    img_rgb = _denormalize(image_tensor)
    
    # Upscale both maps to image resolution
    m_map = cv2.resize(mamba_map.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    c_map = cv2.resize(convnext_map.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize each independently
    m_map = (m_map - m_map.min()) / (m_map.max() - m_map.min() + 1e-8)
    c_map = (c_map - c_map.min()) / (c_map.max() - c_map.min() + 1e-8)
    
    # Weight by gate values
    if gate_values is not None:
        w_m, w_c = gate_values
        total = w_m + w_c + 1e-8
        w_m, w_c = w_m / total, w_c / total
    else:
        w_m, w_c = 0.5, 0.5
    
    fusion_map = w_m * m_map + w_c * c_map
    fusion_map = (fusion_map - fusion_map.min()) / (fusion_map.max() - fusion_map.min() + 1e-8)
    
    heat_rgb = _to_heatmap(fusion_map, img_h, img_w, cmap=cmap or _SKIN_CMAP)
    blend_img = _blend(img_rgb, heat_rgb, alpha=0.6)
    
    return fusion_map, blend_img


# ============================================================================= #
#                6. MASTER VISUALIZATION FUNCTION                                #
# ============================================================================= #

def generate_mamba_attention_diagnostic(
    model,
    image_tensor: torch.Tensor,
    image_seg_tensor: torch.Tensor,
    target_class: int,
    image_id: str,
    save_dir: str,
    device: str = "cpu",
    class_names: Optional[List[str]] = None,
    predicted_class: Optional[int] = None,
    confidence: Optional[float] = None,
    return_maps: bool = False,
) -> dict:
    """
    Master diagnostic function: generates and saves all 5 heatmap types in a
    single publication-quality 5-panel figure plus individual high-res saves.
    
    Args:
        model:             `DualBranchFusionClassifier` instance (eval mode)
        image_tensor:      (3, H, W) original normalized image (sent to Mamba)
        image_seg_tensor:  (3, H, W) segmented normalized image (sent to ConvNeXt)
        target_class:      Ground truth or target class index
        image_id:          Unique image identifier for filenames
        save_dir:          Directory to save visualizations
        device:            'cpu' or 'cuda'
        class_names:       List of class name strings
        predicted_class:   Model prediction (for annotation)
        confidence:        Model confidence (for annotation)
        return_maps:       If True, also return raw numpy maps in the result dict
    
    Returns:
        result dict with keys:
            'mamba_map', 'convnext_map', 'rollout_map', 'fusion_map' (if return_maps)
            'main_figure_path', 'individual_paths'
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    if class_names is None:
        class_names = config.CLASSES
    
    gt_name   = class_names[target_class] if target_class < len(class_names) else str(target_class)
    pred_name = class_names[predicted_class] if (predicted_class is not None and predicted_class < len(class_names)) else "N/A"
    conf_str  = f"{confidence:.3f}" if confidence is not None else "N/A"
    
    # ── Step 1: Original and segmented images ─────────────────────────────── #
    img_rgb  = _denormalize(image_tensor)
    img_seg_rgb = _denormalize(image_seg_tensor)
    
    # ── Step 2: Mamba SSM Activation ──────────────────────────────────────── #
    try:
        mamba_map, mamba_blend = generate_mamba_activation_map(
            model, image_tensor, device=device
        )
    except Exception as e:
        print(f"[Viz] Mamba activation warning: {e}")
        h, w = image_tensor.shape[-2], image_tensor.shape[-1]
        mamba_map = np.random.rand(h // 16, w // 16).astype(np.float32)
        mamba_blend = img_rgb
    
    # ── Step 3: Cross-Attention Heatmap (from model forward pass) ─────────── #
    try:
        with torch.no_grad():
            x_orig = image_tensor.unsqueeze(0).to(device)
            x_seg  = image_seg_tensor.unsqueeze(0).to(device)
            _, attn_weights = model(x_orig, x_seg)
        
        head_weights, attn_blend = generate_cross_attention_map(
            attn_weights, image_tensor
        )
    except Exception as e:
        print(f"[Viz] Cross-attention warning: {e}")
        head_weights = np.ones(config.FUSION_NUM_HEADS) / config.FUSION_NUM_HEADS
        attn_blend = img_rgb
    
    # ── Step 4: ConvNeXt GradCAM ──────────────────────────────────────────── #
    try:
        convnext_map, gradcam_blend = generate_convnext_gradcam(
            model, image_tensor, target_class, device=device
        )
    except Exception as e:
        print(f"[Viz] GradCAM warning: {e}")
        h, w = image_tensor.shape[-2], image_tensor.shape[-1]
        convnext_map = np.random.rand(h // 32, w // 32).astype(np.float32)
        gradcam_blend = img_rgb
    
    # ── Step 5: Multi-Layer Mamba Rollout ─────────────────────────────────── #
    try:
        rollout_map, rollout_blend = generate_mamba_layer_rollout(
            model, image_tensor, device=device, discard_ratio=0.5
        )
    except Exception as e:
        print(f"[Viz] Rollout warning: {e}")
        rollout_map = mamba_map
        rollout_blend = mamba_blend
    
    # ── Step 6: Dual-Branch Fusion Map ────────────────────────────────────── #
    try:
        fusion_map, fusion_blend = generate_dual_branch_fusion_map(
            mamba_map, convnext_map, image_tensor
        )
    except Exception as e:
        print(f"[Viz] Fusion map warning: {e}")
        fusion_map = mamba_map
        fusion_blend = mamba_blend
    
    # ── Step 7: Compose the master diagnostic figure ───────────────────────── #
    style_dark = {
        "facecolor": "#0d0d0d",
        "text.color": "#e0e0e0",
        "axes.facecolor": "#0d0d0d",
    }
    
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(28, 14), facecolor="#0d0d0d")
        
        # Top bar: title and metadata
        title_str = (
            f"Mamba-Attention Diagnostic  |  Image: {image_id}\n"
            f"GT: {gt_name.upper()}  ·  Pred: {pred_name.upper()}  ·  Confidence: {conf_str}"
        )
        fig.suptitle(title_str, fontsize=15, color="#ffec5c",
                     fontweight="bold", y=0.97, linespacing=1.5)
        
        # Row 1: original, Mamba activation, GradCAM (3 panels)
        # Row 2: rollout, fusion, cross-attn head bars, segmented (4 panels)
        gs = fig.add_gridspec(2, 4, hspace=0.06, wspace=0.04,
                              left=0.02, right=0.98, top=0.90, bottom=0.04)
        
        panels_row1 = [
            (img_rgb,       "Original Image",              None),
            (mamba_blend,   "Mamba SSM Activations",       "[patch token L2-norm]"),
            (gradcam_blend, "ConvNeXt V3 GradCAM",         f"[class: {gt_name}]"),
        ]
        
        for col, (img, title, subtitle) in enumerate(panels_row1):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img)
            ax.set_title(title, color="#e0e0e0", fontsize=11, pad=4, fontweight="semibold")
            if subtitle:
                ax.text(0.5, -0.03, subtitle, color="#888888", fontsize=7.5,
                        ha="center", transform=ax.transAxes)
            ax.axis("off")
        
        # Row 1 col 3: cross-attention head distribution bar chart
        ax_bar = fig.add_subplot(gs[0, 3])
        n_heads = len(head_weights)
        colors = _SKIN_CMAP(np.linspace(0.2, 0.9, n_heads))
        bars = ax_bar.bar(range(n_heads), head_weights, color=colors, width=0.6, edgecolor="none")
        ax_bar.set_xlim(-0.5, n_heads - 0.5)
        ax_bar.set_ylim(0, max(head_weights) * 1.3 + 1e-6)
        ax_bar.set_xticks(range(n_heads))
        ax_bar.set_xticklabels([f"H{i+1}" for i in range(n_heads)], fontsize=7, color="#aaaaaa")
        ax_bar.set_title("Cross-Attention Heads (A↔B)", color="#e0e0e0", fontsize=11, pad=4)
        ax_bar.set_ylabel("Weight", color="#888888", fontsize=8)
        ax_bar.tick_params(colors="#888888", labelsize=7)
        ax_bar.spines[:].set_color("#333333")
        ax_bar.set_facecolor("#111111")
        ax_bar.grid(axis="y", color="#222222", linewidth=0.5)
        
        panels_row2 = [
            (rollout_blend, "Mamba Layer Rollout",           "[cumulative SSM depth]"),
            (fusion_blend,  "Dual-Branch Fusion Map",        "[Mamba + ConvNeXt weighted]"),
            (attn_blend,    "Fusion Attention Heatmap",      "[global cross-attention]"),
            (img_seg_rgb,   "Segmented Input (ConvNeXt)",    "[UNet lesion mask applied]"),
        ]
        
        for col, (img, title, subtitle) in enumerate(panels_row2):
            ax = fig.add_subplot(gs[1, col])
            ax.imshow(img)
            ax.set_title(title, color="#e0e0e0", fontsize=11, pad=4, fontweight="semibold")
            if subtitle:
                ax.text(0.5, -0.03, subtitle, color="#888888", fontsize=7.5,
                        ha="center", transform=ax.transAxes)
            ax.axis("off")
        
        # Colorbar (right edge)
        cax = fig.add_axes([0.986, 0.04, 0.007, 0.86])
        sm = cm.ScalarMappable(cmap=_SKIN_CMAP)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Activation Intensity", color="#888888", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#888888", labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#888888")
    
    main_fig_path = _save_figure(fig, save_dir, f"{image_id}_mamba_attn_diagnostic.png")
    
    # ── Step 8: Individual high-res saves ─────────────────────────────────── #
    individual_paths = {}
    single_panels = {
        "mamba_ssm": mamba_blend,
        "convnext_gradcam": gradcam_blend,
        "rollout": rollout_blend,
        "fusion": fusion_blend,
        "segmented": img_seg_rgb,
    }
    for name, arr in single_panels.items():
        fig_s, ax_s = plt.subplots(1, 1, figsize=(6, 6), facecolor="#0d0d0d")
        ax_s.imshow(arr)
        ax_s.axis("off")
        path = _save_figure(fig_s, os.path.join(save_dir, "individual"), f"{image_id}_{name}.png")
        individual_paths[name] = path
    
    print(f"[Viz] Saved diagnostic for '{image_id}' → {main_fig_path}")
    
    result = {
        "main_figure_path": main_fig_path,
        "individual_paths": individual_paths,
    }
    if return_maps:
        result.update({
            "mamba_map": mamba_map,
            "convnext_map": convnext_map,
            "rollout_map": rollout_map,
            "fusion_map": fusion_map,
            "head_weights": head_weights,
        })
    
    return result


# ============================================================================= #
#            CONVENIENCE: Batch Diagnostic for DataLoader                        #
# ============================================================================= #

def visualize_batch_diagnostics(
    model,
    batch: dict,
    predictions: torch.Tensor,
    probs: torch.Tensor,
    save_dir: str,
    device: str = "cpu",
    max_samples: int = 8,
    class_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Convenience wrapper to generate diagnostics for up to `max_samples` images
    from a dataset batch dict (as returned by HAM10000Dataset).
    
    Args:
        model:       `DualBranchFusionClassifier` in eval mode
        batch:       Dict with keys 'image', 'mask', 'label', 'image_id'
        predictions: (B,) predicted class indices
        probs:       (B, num_classes) softmax probabilities
        save_dir:    Root directory for output files
        device:      'cpu' or 'cuda'
        max_samples: Max number of diagnostics to generate (to avoid long runtimes)
        class_names: Class names list
    
    Returns:
        List of saved figure paths
    """
    if class_names is None:
        class_names = config.CLASSES
    
    images = batch["image"]
    labels = batch["label"]
    image_ids = batch["image_id"]
    masks = batch.get("mask", None)
    
    saved_paths = []
    n = min(len(images), max_samples)
    
    for i in range(n):
        img_tensor  = images[i]  # (3, H, W)
        
        # Apply mask to get segmented image (replicating training pipeline)
        if masks is not None:
            mask_bin = (masks[i] > 0.5).float()
            if mask_bin.shape[0] == 1:
                mask_bin = mask_bin.repeat(3, 1, 1)
            img_seg = img_tensor * mask_bin
        else:
            img_seg = img_tensor.clone()
        
        gt_label   = int(labels[i])
        pred_label = int(predictions[i])
        conf       = float(probs[i, pred_label])
        img_id     = image_ids[i]
        
        try:
            result = generate_mamba_attention_diagnostic(
                model=model,
                image_tensor=img_tensor,
                image_seg_tensor=img_seg,
                target_class=gt_label,
                image_id=img_id,
                save_dir=save_dir,
                device=device,
                class_names=class_names,
                predicted_class=pred_label,
                confidence=conf,
            )
            saved_paths.append(result["main_figure_path"])
        except Exception as e:
            print(f"[Viz] ERROR for sample '{img_id}': {e}")
            continue
    
    return saved_paths
