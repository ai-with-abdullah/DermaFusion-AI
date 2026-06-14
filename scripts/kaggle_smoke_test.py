"""
Kaggle Smoke Test — run BEFORE any full training run.
======================================================
Exercises the parts that cannot be tested without timm + a GPU:
  • EVA-02 / ConvNeXt spatial token extraction (forward_tokens → grids)
  • Full DualBranchFusionClassifier forward with the soft mask (BUG-Attn + MAA)
  • One segmentation train step (Swin-UNet + LearnableTverskyBCELoss)
  • One classifier train step (SALA + mixup-style mask mixing)
  • NaN / shape sanity on every output and gradient

Usage (Kaggle / any GPU box):
    python scripts/kaggle_smoke_test.py

It uses pretrained=False (no downloads) and batch=2 random tensors, so it runs in
~1–2 minutes. A clean "ALL SMOKE TESTS PASSED" means shapes/grads are wired
correctly and a full run will not crash on a shape/NaN bug. It does NOT validate
accuracy — only mechanical correctness.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F

from configs.config import config
from models.dual_branch_fusion import DualBranchFusionClassifier
from models.transformer_unet import SwinTransformerUNet, LearnableTverskyBCELoss
from training.train_utils import apply_mask
from training.losses import get_combined_class_loss, get_sala_loss
from datasets.unified_dataset import get_source_class_priors, SkinLesionRecord, NUM_SOURCES


def banner(msg): print("\n" + "=" * 70 + f"\n  {msg}\n" + "=" * 70)


def main():
    dev = config.DEVICE
    B, S = 2, config.IMAGE_SIZE
    print(f"Device={dev}  batch={B}  image_size={S}  grid={config.FUSION_GRID}")

    # ── Segmentation: model + learnable-Tversky loss, one step ──────────── #
    banner("1) Segmentation step (Swin-UNet + LearnableTverskyBCELoss)")
    unet = SwinTransformerUNet(pretrained=False).to(dev)
    seg_crit = LearnableTverskyBCELoss().to(dev)
    seg_opt = torch.optim.AdamW(
        list(unet.parameters()) + list(seg_crit.parameters()), lr=1e-4)
    imgs = torch.randn(B, 3, S, S, device=dev)
    masks = (torch.rand(B, 1, S, S, device=dev) > 0.5).float()
    seg_logits = unet(imgs)
    assert seg_logits.shape == (B, 1, S, S), seg_logits.shape
    seg_loss = seg_crit(seg_logits.float(), masks)
    seg_loss.backward()
    assert torch.isfinite(seg_loss), "seg loss is NaN/Inf"
    assert seg_crit.tversky.raw.grad is not None, "Tversky α,β got no grad"
    seg_opt.step()
    print(f"   seg_logits {tuple(seg_logits.shape)} | loss {seg_loss.item():.4f} "
          f"| learned {seg_crit.get_learned_params()}  ✓")

    # ── Classifier: full spatial model forward + SALA, one step ─────────── #
    banner("2) Classifier step (spatial fusion + SALA)")
    model = DualBranchFusionClassifier(
        eva02_name=config.EVA02_BACKBONE, eva02_pretrained=False,
        convnext_name=config.CONVNEXT_BACKBONE, convnext_pretrained=False,
        fusion_dim=config.FUSION_EMBED_DIM, num_heads=config.FUSION_NUM_HEADS,
        num_classes=config.NUM_CLASSES, dropout=config.FUSION_DROPOUT,
        use_spatial_fusion=config.USE_SPATIAL_FUSION, fusion_grid=config.FUSION_GRID,
    ).to(dev)

    # Spatial token extraction (the timm-specific part untestable off-Kaggle)
    with torch.no_grad():
        eva_grid = model.branch_eva.forward_tokens(imgs)
        conv_grid = model.branch_conv.forward_tokens(imgs)
    print(f"   EVA tokens grid:     {tuple(eva_grid.shape)}")
    print(f"   ConvNeXt tokens grid:{tuple(conv_grid.shape)}")
    assert eva_grid.dim() == 4 and conv_grid.dim() == 4, "forward_tokens must return (B,C,H,W)"

    # Build SALA on a tiny synthetic record set
    recs = ([SkinLesionRecord('a.jpg', 5, 'p', 'HAM10000')] * 6
            + [SkinLesionRecord('b.jpg', 4, 'p', 'ISIC2020')] * 4)
    priors = get_source_class_priors(recs).to(dev)
    base = get_combined_class_loss(None, dev, config.NUM_CLASSES, config.LABEL_SMOOTHING)
    sala = get_sala_loss(priors, base, tau=config.SALA_TAU, learnable=config.SALA_LEARNABLE)
    cls_opt = torch.optim.AdamW(
        list(model.get_head_params()) + [sala.margin], lr=1e-4)

    labels = torch.randint(0, config.NUM_CLASSES, (B,), device=dev)
    src_ids = torch.randint(0, NUM_SOURCES, (B,), device=dev)
    with torch.no_grad():
        seg_logits = unet(imgs)
        imgs_seg = apply_mask(imgs, seg_logits)
        mask_prob = torch.sigmoid(seg_logits.float())

    logits, attn = model(imgs, imgs_seg, mask_prob)
    assert logits.shape == (B, config.NUM_CLASSES), logits.shape
    N = config.FUSION_GRID ** 2
    assert attn.shape == (B, N, N), attn.shape
    loss = sala(logits, labels, src_ids)
    loss.backward()
    assert torch.isfinite(loss), "classifier loss is NaN/Inf"
    for nm, p in [('bug_attn.gamma', model.bug_attn.gamma),
                  ('mirror.lam', model.mirror_attn.lam),
                  ('asym_eta', model.asym_eta),
                  ('SALA.margin', sala.margin)]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad: {nm}"
    cls_opt.step()
    print(f"   logits {tuple(logits.shape)} | attn {tuple(attn.shape)} | "
          f"loss {loss.item():.4f}  ✓")

    # ── Ablation flags must change the output ───────────────────────────── #
    banner("3) Ablation flags wired")
    model.eval()
    with torch.no_grad():
        full, _   = model(imgs, imgs_seg, mask_prob)
        no_unc, _ = model(imgs, imgs_seg, mask_prob, disable_uncertainty_bias=True)
        no_asym, _= model(imgs, imgs_seg, mask_prob, disable_asymmetry=True)
    assert not torch.allclose(full, no_unc), "uncertainty-bias ablation no effect"
    assert not torch.allclose(full, no_asym), "asymmetry ablation no effect"
    print("   disable_uncertainty_bias and disable_asymmetry both change output  ✓")

    print("\n" + "#" * 70 + "\n#  ALL SMOKE TESTS PASSED — safe to launch full training\n" + "#" * 70)


if __name__ == "__main__":
    main()
