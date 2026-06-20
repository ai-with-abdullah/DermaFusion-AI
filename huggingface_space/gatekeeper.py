"""
Skin-Image Gatekeeper — cheap CLIP zero-shot pre-filter
=======================================================
Runs BEFORE the heavy dual-branch model. Decides whether an uploaded image is a
plausible skin / dermoscopic lesion image. Non-skin inputs (animals, objects,
screenshots, X-rays, faces, landscapes, documents) are rejected early so the
405M-param classifier + Swin-UNet + 5-view TTA never run on them.

Why CLIP zero-shot:
  - No training, no labelled negative dataset required.
  - A single small ViT-B/32 forward — a tiny fraction of the full pipeline's cost,
    so we are NOT "running the whole model every time" just to reject junk.
  - Loaded lazily and cached; the per-image cost is one image encode.

Safety: if open_clip is unavailable for any reason, the gate degrades to "admit all"
(never hard-fails the app). Tune the decision boundary with the GATE_THRESHOLD env
var (default 0.55 = the prompt-softmax mass that must fall on the skin prompts).
"""
import os

# ── Zero-shot prompt sets ─────────────────────────────────────────────────── #
# "skin" prompts describe valid inputs; "other" prompts cover the common junk a
# user might upload. The gate score = softmax mass on the skin prompts.
_SKIN_PROMPTS = [
    "a dermoscopic photograph of a skin lesion",
    "a close-up medical photo of a mole on human skin",
    "a clinical photograph of a skin spot, rash, or growth",
    "a macro photograph of human skin showing a lesion",
]
_OTHER_PROMPTS = [
    "a photograph of an animal",
    "a photograph of an everyday object",
    "an outdoor landscape or scenery photo",
    "a screenshot of a computer or phone screen",
    "a photo of a person's face",
    "a document, chart, diagram, or piece of text",
    "an x-ray, CT, or radiology scan",
]

_THRESHOLD = float(os.environ.get("GATE_THRESHOLD", "0.55"))

# ── Lazy global state ─────────────────────────────────────────────────────── #
_model = None
_preprocess = None
_text_feats = None
_device = None
_backend = None            # None = uninitialised, "open_clip", or "none" (disabled)
_n_skin = len(_SKIN_PROMPTS)


def _lazy_init():
    """Load CLIP + precompute text features once. Sets _backend to 'none' on failure."""
    global _model, _preprocess, _text_feats, _device, _backend
    if _backend is not None:
        return
    try:
        import torch
        import open_clip

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _model = _model.to(_device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        prompts = _SKIN_PROMPTS + _OTHER_PROMPTS
        with torch.no_grad():
            toks = tokenizer(prompts).to(_device)
            tf = _model.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        _text_feats = tf
        _backend = "open_clip"
        print(f"[Gatekeeper] CLIP gate ready (ViT-B-32, threshold={_THRESHOLD:.0%}).")
    except Exception as e:  # noqa: BLE001 — never let the gate crash the app
        print(f"[Gatekeeper] CLIP unavailable ({e}); gate DISABLED (admitting all images).")
        _backend = "none"


def check_skin_image(pil_img):
    """
    Decide whether `pil_img` is a plausible skin/dermoscopic image.

    Returns:
        is_skin : bool   — True if the image should proceed to the diagnostic model
        score   : float  — aggregated softmax mass on the skin prompts, in [0, 1]
        detail  : str    — short human-readable explanation

    If CLIP is unavailable, admits everything (is_skin=True, score=1.0).
    """
    _lazy_init()
    if _backend == "none":
        return True, 1.0, "input gate disabled (CLIP unavailable)"

    import torch

    with torch.no_grad():
        img = _preprocess(pil_img.convert("RGB")).unsqueeze(0).to(_device)
        feat = _model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * feat @ _text_feats.T).softmax(dim=-1)[0]   # (n_prompts,)
        skin_score = float(probs[:_n_skin].sum().item())

    is_skin = skin_score >= _THRESHOLD
    detail = f"skin-likeness {skin_score:.0%} (threshold {_THRESHOLD:.0%})"
    return is_skin, skin_score, detail
