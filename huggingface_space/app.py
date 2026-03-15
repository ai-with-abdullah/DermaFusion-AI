"""
DermaFusion-AI — Hugging Face Gradio Demo
==========================================
Dual-Branch EVA-02 + ConvNeXt V2 Skin Lesion Classifier
Author: Muhammad Abdullah | The Islamia University of Bahawalpur
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
import time

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesion",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma ⚠️",
    "nv":    "Melanocytic Nevus (Mole)",
    "vasc":  "Vascular Lesion",
}
CLASS_COLORS = {
    "akiec": "#f59e0b",
    "bcc":   "#ef4444",
    "bkl":   "#10b981",
    "df":    "#3b82f6",
    "mel":   "#dc2626",
    "nv":    "#6366f1",
    "vasc":  "#ec4899",
}
RISK_LEVEL = {
    "akiec": ("⚠️ Pre-malignant", "#f59e0b"),
    "bcc":   ("🔴 Malignant", "#ef4444"),
    "bkl":   ("✅ Benign", "#10b981"),
    "df":    ("✅ Benign", "#10b981"),
    "mel":   ("🔴 Malignant — Seek immediate medical attention", "#dc2626"),
    "nv":    ("✅ Benign", "#10b981"),
    "vasc":  ("🟡 Monitor", "#f59e0b"),
}

MODEL_LOADED = False
model = None
unet = None
device = None

# ─── Model Loading ─────────────────────────────────────────────────────────────
def load_models():
    global MODEL_LOADED, model, unet, device
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use HF token from env if available (set as Space secret)
        hf_token = os.environ.get("HF_TOKEN", None)
        from huggingface_hub import hf_hub_download

        REPO_ID = "ai-with-abdullah/DermaFusion-AI"

        # Add parent path for model imports
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.path.abspath(os.getcwd())
        sys.path.insert(0, base_dir)

        from models.dual_branch_fusion import DualBranchFusionClassifier
        from models.transformer_unet import SwinTransformerUNet

        # Load UNet
        print("Downloading UNet weights...")
        unet_path = hf_hub_download(REPO_ID, "best_unet.pth", token=hf_token)
        unet = SwinTransformerUNet(n_classes=1, pretrained=False).to(device)
        unet.load_state_dict(torch.load(unet_path, map_location=device), strict=False)
        unet.eval()
        print("UNet loaded.")

        # Load main model — MUST use eva02_large_patch14_448 (dim=1024) to match checkpoint
        print("Downloading main model weights...")
        model_path = hf_hub_download(REPO_ID, "best_dual_branch_fusion.pth", token=hf_token)

        model = DualBranchFusionClassifier(
            eva02_name="eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
            eva02_pretrained=False,        # we load our own weights below
            convnext_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
            convnext_pretrained=False,     # we load our own weights below
            num_classes=7,
        ).to(device)

        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:3]}...")
        model.eval()

        MODEL_LOADED = True
        print(f"✅ Models loaded on {device}")

    except Exception as e:
        print(f"⚠️  Model not loaded (demo mode): {e}")
        MODEL_LOADED = False

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(pil_img, size=448):
    import torch
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil_img.convert("RGB")).unsqueeze(0).to(device)


# ─── GradCAM++ for ConvNeXt branch ────────────────────────────────────────────
def compute_gradcam(model_branch, feature_layer, inp, class_idx):
    import torch
    import torch.nn.functional as F

    gradients, activations = [], []

    def fwd_hook(m, i, o):
        activations.append(o.detach())

    def bwd_hook(m, gi, go):
        gradients.append(go[0].detach())

    fh = feature_layer.register_forward_hook(fwd_hook)
    bh = feature_layer.register_backward_hook(bwd_hook)

    try:
        inp.requires_grad_(True)
        out = model_branch(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
        score = out[0, class_idx]
        model_branch.zero_grad()
        score.backward()

        grad = gradients[0]          # (1, C, H, W)
        act  = activations[0]        # (1, C, H, W)

        # GradCAM++ weights
        grad2 = grad ** 2
        grad3 = grad ** 3
        alpha_num = grad2
        alpha_den = 2 * grad2 + (grad3 * act).sum(dim=(2, 3), keepdim=True) + 1e-7
        alpha = alpha_num / alpha_den
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, (448, 448), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    except Exception:
        return None
    finally:
        fh.remove()
        bh.remove()


# ─── Generate heatmap overlay ──────────────────────────────────────────────────
def overlay_heatmap(pil_img, cam, alpha=0.5, colormap="jet"):
    img_np = np.array(pil_img.convert("RGB").resize((448, 448)))
    cmap = plt.get_cmap(colormap)
    heatmap = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    overlaid = (img_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return Image.fromarray(overlaid)


# ─── Demo predictions (used when model weights not loaded) ────────────────────
def demo_prediction(pil_img):
    """Return plausible-looking demo output for showcase purposes."""
    np.random.seed(hash(str(pil_img.size)) % 2**31)
    raw = np.random.dirichlet(np.ones(7) * 0.5) * 100
    # Make nv most likely for a realistic demo
    raw[CLASS_NAMES.index("nv")] += 30
    raw = raw / raw.sum() * 100
    probs = {cn: float(f"{raw[i]:.1f}") for i, cn in enumerate(CLASS_NAMES)}
    pred_cls = max(probs, key=probs.get)
    # Fake heatmap
    w, h = 448, 448
    cam = np.zeros((h, w))
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    cam = np.exp(-dist ** 2 / (2 * (h // 4) ** 2))
    heatmap_img = overlay_heatmap(pil_img, cam)
    return pred_cls, probs, heatmap_img


# ─── Main Inference ────────────────────────────────────────────────────────────
def run_inference(pil_img):
    import torch
    import torch.nn.functional as F

    if not MODEL_LOADED or model is None:
        return demo_prediction(pil_img)

    with torch.no_grad():
        inp_eva   = preprocess_image(pil_img, 448)
        inp_conv  = preprocess_image(pil_img, 384)

        # Segment
        mask_logits = unet(inp_eva)
        seg_mask = (torch.sigmoid(mask_logits) > 0.5).float()
        seg_mask_resized = F.interpolate(seg_mask, (384, 384), mode="nearest")
        inp_seg = inp_conv * seg_mask_resized

        logits, _ = model(inp_eva, inp_seg)
        probs_tensor = F.softmax(logits, dim=1)[0]
        probs = {cn: float(f"{probs_tensor[i].item() * 100:.1f}")
                 for i, cn in enumerate(CLASS_NAMES)}
        pred_cls = max(probs, key=probs.get)

    # GradCAM on ConvNeXt branch (use stage[-2] for 24x24 resolution, better than last stage 12x12)
    try:
        inp_conv_grad = preprocess_image(pil_img, 384)
        pred_idx = CLASS_NAMES.index(pred_cls)
        last_stage = list(model.branch_conv.backbone.stages.children())[-2]
        cam = compute_gradcam(model.branch_conv, last_stage,
                              inp_conv_grad, pred_idx)
        heatmap_img = overlay_heatmap(pil_img, cam) if cam is not None else pil_img
    except Exception:
        heatmap_img = pil_img

    return pred_cls, probs, heatmap_img


# ─── Confidence Bar Chart (matplotlib → PIL) ──────────────────────────────────
def make_confidence_chart(probs, pred_cls):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    classes   = CLASS_NAMES
    values    = [probs[c] for c in classes]
    bar_colors = [CLASS_COLORS[c] if c == pred_cls else "#334155" for c in classes]

    bars = ax.barh(classes, values, color=bar_colors, height=0.6,
                   edgecolor="none")

    for bar, val, cls in zip(bars, values, classes):
        label_name = CLASS_LABELS[cls].split("/")[0].strip()
        ax.text(min(val + 1.5, 105), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", ha="left", color="white", fontsize=9,
                fontweight="bold" if cls == pred_cls else "normal")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[:].set_color("#334155")
    for label in ax.get_yticklabels():
        label.set_color("#f8fafc" if label.get_text() == pred_cls else "#94a3b8")
        if label.get_text() == pred_cls:
            label.set_fontweight("bold")

    ax.set_title("Class Confidence Scores", color="#f8fafc",
                 fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ─── Process function exposed to Gradio ───────────────────────────────────────
def process(image):
    if image is None:
        return (
            None,
            None,
            "⬆️ Please upload a dermoscopy image to begin analysis.",
            "",
            "",
        )

    pil_img = Image.fromarray(image).convert("RGB")
    start   = time.time()
    pred_cls, probs, heatmap_img = run_inference(pil_img)
    elapsed = time.time() - start

    chart = make_confidence_chart(probs, pred_cls)
    top_conf = probs[pred_cls]
    risk_text, risk_color = RISK_LEVEL[pred_cls]
    full_label = CLASS_LABELS[pred_cls]

    result_html = f"""
<div style="
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid {risk_color}40;
    border-left: 4px solid {risk_color};
    border-radius: 12px;
    padding: 20px 24px;
    font-family: 'Inter', system-ui, sans-serif;
    margin-top: 4px;
">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
        <div style="
            background:{CLASS_COLORS[pred_cls]}22;
            border:2px solid {CLASS_COLORS[pred_cls]};
            border-radius:50%;
            width:52px; height:52px;
            display:flex; align-items:center; justify-content:center;
            font-size:1.6rem; flex-shrink:0;
        ">🔬</div>
        <div>
            <div style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;">
                Primary Diagnosis
            </div>
            <div style="color:#f8fafc; font-size:1.25rem; font-weight:700; line-height:1.3;">
                {full_label}
            </div>
        </div>
    </div>

    <div style="display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px;">
        <div style="
            background:#ffffff0d; border-radius:8px; padding:10px 16px; flex:1; min-width:120px;
        ">
            <div style="color:#64748b; font-size:0.7rem; text-transform:uppercase; margin-bottom:3px;">
                Confidence
            </div>
            <div style="color:{CLASS_COLORS[pred_cls]}; font-size:1.6rem; font-weight:800;">
                {top_conf:.1f}%
            </div>
        </div>
        <div style="
            background:#ffffff0d; border-radius:8px; padding:10px 16px; flex:1; min-width:120px;
        ">
            <div style="color:#64748b; font-size:0.7rem; text-transform:uppercase; margin-bottom:3px;">
                Risk Level
            </div>
            <div style="color:{risk_color}; font-size:0.95rem; font-weight:700;">
                {risk_text}
            </div>
        </div>
        <div style="
            background:#ffffff0d; border-radius:8px; padding:10px 16px; flex:1; min-width:120px;
        ">
            <div style="color:#64748b; font-size:0.7rem; text-transform:uppercase; margin-bottom:3px;">
                Inference Time
            </div>
            <div style="color:#f8fafc; font-size:0.95rem; font-weight:700;">
                {elapsed:.2f}s
            </div>
        </div>
    </div>

    <div style="
        background:#dc262615; border:1px solid #dc262630; border-radius:8px;
        padding:10px 14px; font-size:0.78rem; color:#fca5a5;
    ">
        ⚠️ <strong>Important:</strong> This is a research tool. Always consult a qualified dermatologist for clinical diagnosis.
    </div>
</div>
"""

    mode_badge = (
        "🟢 Model Active (GPU)"  if MODEL_LOADED and device and str(device).startswith("cuda")
        else "🟡 Model Active (CPU)" if MODEL_LOADED
        else "🔵 Demo Mode (example output)"
    )

    return heatmap_img, chart, result_html, mode_badge, ""


# ─── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root ── */
:root {
    --bg-primary:   #020817;
    --bg-secondary: #0f172a;
    --bg-card:      #1e293b;
    --border:       #1e293b;
    --accent:       #6366f1;
    --accent-glow:  #6366f180;
    --text-primary: #f8fafc;
    --text-muted:   #94a3b8;
    --danger:       #ef4444;
    --success:      #10b981;
}

body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Header ── */
.header-section {
    text-align: center;
    padding: 40px 20px 24px;
    background: linear-gradient(180deg, #0a0f2e 0%, #020817 100%);
    border-bottom: 1px solid #1e293b;
    position: relative;
    overflow: hidden;
}
.header-section::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 400px; height: 200px;
    background: radial-gradient(ellipse, #6366f130 0%, transparent 70%);
    pointer-events: none;
}
.header-title {
    font-size: clamp(1.8rem, 5vw, 2.8rem);
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    line-height: 1.2;
}
.header-sub {
    color: #64748b;
    font-size: 0.95rem;
    max-width: 600px;
    margin: 0 auto 20px;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-bottom: 8px;
}
.badge {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 500;
}
.badge-accent {
    border-color: #6366f150;
    color: #a5b4fc;
    background: #6366f115;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    padding: 20px;
    max-width: 700px;
    margin: 0 auto;
}
.stat-card {
    flex: 1; min-width: 140px;
    background: linear-gradient(135deg, #1e293b, #162032);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    color: #64748b;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Upload area ── */
.upload-zone {
    border: 2px dashed #334155 !important;
    border-radius: 16px !important;
    background: #0f172a !important;
    transition: all 0.3s ease !important;
    min-height: 260px !important;
}
.upload-zone:hover {
    border-color: #6366f1 !important;
    background: #6366f108 !important;
    box-shadow: 0 0 24px #6366f120 !important;
}
.upload-zone .icon-wrap { font-size: 2.5rem !important; }

/* ── Buttons ── */
.analyze-btn {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px #6366f140 !important;
    width: 100% !important;
}
.analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px #6366f160 !important;
    background: linear-gradient(135deg, #818cf8, #6366f1) !important;
}
.clear-btn {
    background: transparent !important;
    color: #64748b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    font-size: 0.9rem !important;
    width: 100% !important;
}
.clear-btn:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
}

/* ── Output panels ── */
.output-image-panel, .chart-panel {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}
.result-panel {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Section labels ── */
.section-label {
    color: #64748b;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 8px;
}

/* ── Mode badge ── */
.mode-badge {
    display: inline-block;
    font-size: 0.75rem;
    color: #64748b;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 20px;
    padding: 4px 12px;
}

/* ── Warning banner ── */
.warning-banner {
    background: linear-gradient(135deg, #78350f20, #92400e15);
    border: 1px solid #d9770640;
    border-radius: 10px;
    padding: 12px 16px;
    color: #fbbf24;
    font-size: 0.82rem;
    text-align: center;
}

/* ── Examples ── */
.examples-section {
    border-top: 1px solid #1e293b;
    padding-top: 16px;
}

/* ── Footer ── */
.footer-section {
    border-top: 1px solid #1e293b;
    padding: 24px 20px;
    text-align: center;
    color: #475569;
    font-size: 0.82rem;
}
.footer-links a {
    color: #6366f1;
    text-decoration: none;
}
.footer-links a:hover { text-decoration: underline; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .stats-row { padding: 16px 12px; gap: 8px; }
    .stat-card { min-width: 110px; }
    .stat-value { font-size: 1.3rem; }
}
@media (max-width: 480px) {
    .header-section { padding: 24px 16px 16px; }
    .badge { font-size: 0.72rem; padding: 3px 10px; }
}

/* ── Tab overrides ── */
.tabs { background: transparent !important; border: none !important; }
.tab-nav button {
    background: #0f172a !important;
    color: #64748b !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}
.tab-nav button.selected {
    background: #6366f115 !important;
    color: #a5b4fc !important;
    border-color: #6366f150 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
"""

# ─── Build UI ──────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.indigo,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        title="DermaFusion-AI | Skin Lesion Classifier",
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div class="header-section">
            <div class="header-title">🔬 DermaFusion-AI</div>
            <div class="header-sub">
                Research-grade skin lesion classification powered by dual-branch
                EVA-02 Large + ConvNeXt V2 fusion with GradCAM++ explainability
            </div>
            <div class="badge-row">
                <span class="badge badge-accent">EVA-02 Large 307M</span>
                <span class="badge badge-accent">ConvNeXt V2 Base</span>
                <span class="badge badge-accent">Bidirectional Cross-Attention</span>
                <span class="badge">7 Lesion Classes</span>
                <span class="badge">GradCAM++ XAI</span>
            </div>
        </div>

        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">0.9908</div>
                <div class="stat-label">Macro AUC</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">85.6%</div>
                <div class="stat-label">Balanced Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">92.2%</div>
                <div class="stat-label">MEL Sensitivity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">401M</div>
                <div class="stat-label">Parameters</div>
            </div>
        </div>
        """)

        # ── Main Layout ──
        with gr.Row(equal_height=False):

            # ─ Left column: Upload + controls ─
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<div class="section-label">Upload Dermoscopy Image</div>')
                image_input = gr.Image(
                    label="",
                    type="numpy",
                    elem_classes=["upload-zone"],
                    height=300,
                    sources=["upload", "clipboard"],
                )

                gr.HTML("""
                <div class="warning-banner" style="margin: 12px 0;">
                    📷 For best results, use <strong>dermoscopy images</strong>.
                    Smartphone photos may have lower accuracy.
                </div>
                """)

                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Analyze Lesion",
                        elem_classes=["analyze-btn"],
                        variant="primary",
                    )
                    clear_btn = gr.ClearButton(
                        [image_input],
                        value="✕ Clear",
                        elem_classes=["clear-btn"],
                    )

                mode_display = gr.Textbox(
                    value="⬆️ Upload an image to begin",
                    label="",
                    interactive=False,
                    show_label=False,
                    elem_classes=["mode-badge"],
                    container=False,
                )

                # Class legend
                gr.HTML("""
                <div style="margin-top:20px;">
                    <div class="section-label">Detectable Classes</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:8px;">
                        <div style="display:flex;align-items:center;gap:6px;font-size:0.78rem;color:#94a3b8;">
                            <span style="width:8px;height:8px;border-radius:2px;background:#f59e0b;flex-shrink:0;"></span>
                            Actinic Keratosis
                        </div>
                        <div style="display:flex;align-items:center;gap:6px;font-size:0.78rem;color:#94a3b8;">
                            <span style="width:8px;height:8px;border-radius:2px;background:#ef4444;flex-shrink:0;"></span>
                            Basal Cell Carcinoma
                        </div>
                        <div style="display:flex;align-items:center;gap:6px;font-size:0.78rem;color:#94a3b8;">
                            <span style="width:8px;height:8px;border-radius:2px;background:#10b981;flex-shrink:0;"></span>
                            Benign Keratosis
                        </div>
                        <div style="display:flex;align-items:center;gap:6px;font-size:0.78rem;color:#94a3b8;">
                            <span style="width:8px;height:8px;border-radius:2px;background:#3b82f6;flex-shrink:0;"></span>
                            Dermatofibroma
                        </div>
                        <div style="display:flex;align-items:center;gap:6px;font-size:0.78rem;color:#94a3b8;">
                            <span style="width:8px;height:8px;border-radius:2px;background:#dc2626;flex-shrink:0;"></span>
                            <strong style="color:#fca5a5;">Melanoma ⚠️</strong>
                        </div>
                        <div style="display:flex;align-items:center;gap:6px;font-size:0.78rem;color:#94a3b8;">
                            <span style="width:8px;height:8px;border-radius:2px;background:#6366f1;flex-shrink:0;"></span>
                            Melanocytic Nevus
                        </div>
                        <div style="display:flex;align-items:center;gap:6px;font-size:0.78rem;color:#94a3b8;">
                            <span style="width:8px;height:8px;border-radius:2px;background:#ec4899;flex-shrink:0;"></span>
                            Vascular Lesion
                        </div>
                    </div>
                </div>
                """)

            # ─ Right column: Results ─
            with gr.Column(scale=2, min_width=400):

                with gr.Tabs():

                    with gr.Tab("📊 Results"):
                        result_html = gr.HTML(
                            value='<div style="color:#475569;text-align:center;padding:40px;font-size:0.9rem;">Upload an image and click Analyze Lesion</div>',
                            elem_classes=["result-panel"],
                        )
                        chart_output = gr.Image(
                            label="Confidence Scores",
                            type="pil",
                            elem_classes=["chart-panel"],
                            height=300,
                            show_label=False,
                        )

                    with gr.Tab("🔥 GradCAM++ Heatmap"):
                        gr.HTML("""
                        <div style="color:#64748b;font-size:0.82rem;padding:8px 0 12px;">
                            GradCAM++ highlights the image regions that most influenced the model's decision.
                            Warmer colors (red/yellow) = higher importance.
                        </div>
                        """)
                        heatmap_output = gr.Image(
                            label="",
                            type="pil",
                            elem_classes=["output-image-panel"],
                            height=380,
                            show_label=False,
                        )

                    with gr.Tab("ℹ️ About"):
                        gr.HTML("""
                        <div style="color:#94a3b8; font-size:0.88rem; line-height:1.8; padding:8px 0;">
                            <h3 style="color:#f8fafc; font-size:1rem; margin-bottom:12px;">DermaFusion-AI Architecture</h3>
                            <p>A dual-branch deep learning system combining two complementary visual AI models:</p>
                            <ul style="margin:10px 0 16px; padding-left:20px;">
                                <li><strong style="color:#a5b4fc;">Branch A — EVA-02 Large (307M params):</strong>
                                    Processes the full 448×448 image to capture global context — lesion shape, borders, surrounding skin patterns.</li>
                                <li style="margin-top:8px;"><strong style="color:#a5b4fc;">Branch B — ConvNeXt V2 Base (88.5M params):</strong>
                                    Processes a segmentation-masked image (lesion only, background zeroed) for fine texture analysis.</li>
                                <li style="margin-top:8px;"><strong style="color:#a5b4fc;">Fusion — Bidirectional Cross-Attention:</strong>
                                    Both branches communicate symmetrically; neither dominates. A sigmoid gate learns the clinical relevance of each feature.</li>
                            </ul>
                            <h3 style="color:#f8fafc; font-size:1rem; margin:16px 0 10px;">Training</h3>
                            <p>Trained on 460,000+ images across 5 datasets: HAM10000, ISIC 2019/2020/2024, PH2 — with patient-aware splitting to prevent data leakage.</p>
                            <h3 style="color:#f8fafc; font-size:1rem; margin:16px 0 10px;">Performance</h3>
                            <table style="width:100%; border-collapse:collapse; font-size:0.83rem;">
                                <tr style="border-bottom:1px solid #1e293b;">
                                    <td style="padding:6px 0; color:#64748b;">Macro AUC</td>
                                    <td style="padding:6px 0; color:#a5b4fc; font-weight:700;">0.9908</td>
                                </tr>
                                <tr style="border-bottom:1px solid #1e293b;">
                                    <td style="padding:6px 0; color:#64748b;">Balanced Accuracy</td>
                                    <td style="padding:6px 0; color:#a5b4fc; font-weight:700;">85.6%</td>
                                </tr>
                                <tr style="border-bottom:1px solid #1e293b;">
                                    <td style="padding:6px 0; color:#64748b;">Melanoma Sensitivity</td>
                                    <td style="padding:6px 0; color:#ef4444; font-weight:700;">92.2% (vs 86% avg. dermatologist)</td>
                                </tr>
                                <tr>
                                    <td style="padding:6px 0; color:#64748b;">DERM7PT AUC (unseen)</td>
                                    <td style="padding:6px 0; color:#a5b4fc; font-weight:700;">0.872</td>
                                </tr>
                            </table>
                            <div style="margin-top:16px; padding:12px; background:#1e293b; border-radius:8px; font-size:0.8rem; color:#64748b;">
                                ⚠️ <strong style="color:#fbbf24;">Medical Disclaimer:</strong>
                                DermaFusion-AI is a research prototype. It has not been clinically validated or approved by any medical regulatory body.
                                Do not use this for self-diagnosis. Always consult a qualified dermatologist.
                            </div>
                        </div>
                        """)

        # ── Event wiring ──
        analyze_btn.click(
            fn=process,
            inputs=[image_input],
            outputs=[heatmap_output, chart_output, result_html, mode_display, gr.Textbox(visible=False)],
            show_progress="full",
        )
        image_input.change(
            fn=lambda img: ("", "") if img is None else (gr.update(), gr.update()),
            inputs=[image_input],
            outputs=[result_html, mode_display] if False else [],  # no-op on change
        )

        # ── Footer ──
        gr.HTML("""
        <div class="footer-section">
            <div class="footer-links" style="margin-bottom:8px;">
                <a href="https://github.com/ai-with-abdullah/DermaFusion-AI" target="_blank">GitHub</a>
                &nbsp;·&nbsp;
                <a href="https://arxiv.org" target="_blank">Paper (arXiv — Coming Soon)</a>
                &nbsp;·&nbsp;
                <a href="https://www.isic-archive.com" target="_blank">ISIC Dataset</a>
            </div>
            <div>
                Muhammad Abdullah · The Islamia University of Bahawalpur, Pakistan · 2026<br>
                <span style="font-size:0.75rem;">
                    Trained on HAM10000, ISIC 2019/2020/2024, PH2 · 460,000+ images · Patient-aware splits
                </span>
            </div>
        </div>
        """)

    return demo


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading DermaFusion-AI models...")
    load_models()
    print(f"Model loaded: {MODEL_LOADED}")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=10000,
        share=False,
        ssr_mode=False,
        show_error=True,
    )



























# """
# DermaFusion-AI — Hugging Face Gradio Demo (ZeroGPU Edition)
# ============================================================
# Dual-Branch EVA-02 + ConvNeXt V2 Skin Lesion Classifier
# Author: Muhammad Abdullah | The Islamia University of Bahawalpur
# """

# import os
# import sys
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import gradio as gr
# from PIL import Image
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import io
# import time
# import spaces  # HF ZeroGPU

# # ─── Constants ────────────────────────────────────────────────────────────────
# CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# CLASS_LABELS = {
#     "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
#     "bcc":   "Basal Cell Carcinoma",
#     "bkl":   "Benign Keratosis-like Lesion",
#     "df":    "Dermatofibroma",
#     "mel":   "Melanoma ⚠️",
#     "nv":    "Melanocytic Nevus (Mole)",
#     "vasc":  "Vascular Lesion",
# }
# CLASS_COLORS = {
#     "akiec": "#f59e0b",
#     "bcc":   "#ef4444",
#     "bkl":   "#10b981",
#     "df":    "#3b82f6",
#     "mel":   "#dc2626",
#     "nv":    "#6366f1",
#     "vasc":  "#ec4899",
# }
# RISK_LEVEL = {
#     "akiec": ("⚠️ Pre-malignant", "#f59e0b"),
#     "bcc":   ("🔴 Malignant", "#ef4444"),
#     "bkl":   ("✅ Benign", "#10b981"),
#     "df":    ("✅ Benign", "#10b981"),
#     "mel":   ("🔴 Malignant — Seek immediate medical attention", "#dc2626"),
#     "nv":    ("✅ Benign", "#10b981"),
#     "vasc":  ("🟡 Monitor", "#f59e0b"),
# }

# # ─── Global model references (loaded once at startup) ─────────────────────────
# MODEL_LOADED = False
# _model = None
# _unet  = None

# # ─── Startup: load models onto CPU, ZeroGPU will move to GPU per-request ──────
# def load_models():
#     global MODEL_LOADED, _model, _unet
#     try:
#         import torch
#         from huggingface_hub import hf_hub_download

#         REPO_ID  = "ai-with-abdullah/DermaFusion-AI"
#         hf_token = os.environ.get("HF_TOKEN", None)

#         sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#         from models.dual_branch_fusion import DualBranchFusionClassifier
#         from models.transformer_unet   import SwinTransformerUNet

#         # UNet
#         print("Loading UNet...")
#         unet_path = hf_hub_download(REPO_ID, "best_unet.pth", token=hf_token)
#         _unet = SwinTransformerUNet(n_classes=1, pretrained=False)
#         _unet.load_state_dict(torch.load(unet_path, map_location="cpu"), strict=False)
#         _unet.eval()

#         # DualBranchFusion – EVA-02 Large
#         print("Loading DualBranchFusion...")
#         model_path = hf_hub_download(REPO_ID, "best_dual_branch_fusion.pth", token=hf_token)
#         _model = DualBranchFusionClassifier(
#             eva02_name="eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
#             eva02_pretrained=False,
#             convnext_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
#             convnext_pretrained=False,
#             num_classes=7,
#         )
#         ckpt  = torch.load(model_path, map_location="cpu")
#         state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
#         _model.load_state_dict(state, strict=False)
#         _model.eval()

#         MODEL_LOADED = True
#         print("✅ Models ready.")
#     except Exception as e:
#         print(f"⚠️ Model load failed (demo mode): {e}")
#         MODEL_LOADED = False


# # ─── Preprocessing ─────────────────────────────────────────────────────────────
# def _preprocess(pil_img, size, device):
#     import torch, torchvision.transforms as T
#     tf = T.Compose([
#         T.Resize((size, size)),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
#     return tf(pil_img.convert("RGB")).unsqueeze(0).to(device)


# # ─── Heatmap overlay ──────────────────────────────────────────────────────────
# def _overlay(pil_img, cam):
#     img_np  = np.array(pil_img.convert("RGB").resize((448, 448)))
#     cmap    = plt.get_cmap("jet")
#     heatmap = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
#     blend   = (img_np * 0.5 + heatmap * 0.5).astype(np.uint8)
#     return Image.fromarray(blend)


# def _demo_cam(pil_img):
#     """Gaussian blob centered on image — used in demo mode."""
#     h, w = 448, 448
#     Y, X = np.ogrid[:h, :w]
#     cam  = np.exp(-((X - w//2)**2 + (Y - h//2)**2) / (2 * (h//4)**2))
#     return _overlay(pil_img, cam)


# # ─── Core inference (decorated with @spaces.GPU for ZeroGPU) ──────────────────
# @spaces.GPU(duration=120)
# def _infer_gpu(pil_img):
#     import torch
#     import torch.nn.functional as F

#     device = torch.device("cuda")
#     model  = _model.to(device)
#     unet   = _unet.to(device)

#     with torch.no_grad():
#         inp_eva  = _preprocess(pil_img, 448, device)
#         inp_conv = _preprocess(pil_img, 384, device)

#         mask   = torch.sigmoid(unet(inp_eva)) > 0.5
#         mask_r = F.interpolate(mask.float(), (384, 384), mode="nearest")
#         inp_seg = inp_conv * mask_r

#         logits, _ = model(inp_eva, inp_seg)
#         probs_t   = F.softmax(logits, dim=1)[0]

#     probs   = {cn: float(f"{probs_t[i].item()*100:.1f}") for i, cn in enumerate(CLASS_NAMES)}
#     pred    = max(probs, key=probs.get)

#     # GradCAM++ on last ConvNeXt stage
#     cam_result = None
#     try:
#         grads, acts = [], []
#         last_stage  = list(model.branch_conv.backbone.stages.children())[-1]
#         fh = last_stage.register_forward_hook(lambda m,i,o: acts.append(o.detach()))
#         bh = last_stage.register_backward_hook(lambda m,gi,go: grads.append(go[0].detach()))

#         inp2   = _preprocess(pil_img, 384, device).requires_grad_(True)
#         out2   = model.branch_conv(inp2)
#         score  = (out2[0] if isinstance(out2, (list,tuple)) else out2)[0, CLASS_NAMES.index(pred)]
#         model.branch_conv.zero_grad()
#         score.backward()

#         fh.remove(); bh.remove()
#         g, a  = grads[0], acts[0]
#         g2    = g**2; g3 = g**3
#         alpha = g2 / (2*g2 + (g3*a).sum((2,3), keepdim=True) + 1e-7)
#         w     = (alpha * F.relu(g)).sum((2,3), keepdim=True)
#         cam   = F.relu((w*a).sum(1, keepdim=True))
#         cam   = F.interpolate(cam, (448,448), mode="bilinear", align_corners=False)
#         cam   = cam.squeeze().cpu().numpy()
#         cam   = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#         cam_result = cam
#     except Exception:
#         pass

#     heatmap = _overlay(pil_img, cam_result) if cam_result is not None else _demo_cam(pil_img)
#     return pred, probs, heatmap


# def _infer_demo(pil_img):
#     np.random.seed(42)
#     raw = np.random.dirichlet(np.ones(7)*0.5)*100
#     raw[CLASS_NAMES.index("nv")] += 30
#     raw = raw / raw.sum() * 100
#     probs = {cn: float(f"{raw[i]:.1f}") for i, cn in enumerate(CLASS_NAMES)}
#     return max(probs, key=probs.get), probs, _demo_cam(pil_img)


# # ─── Confidence bar chart ──────────────────────────────────────────────────────
# def _chart(probs, pred):
#     fig, ax = plt.subplots(figsize=(7, 4))
#     fig.patch.set_facecolor("#0d1117")
#     ax.set_facecolor("#161b22")
#     vals   = [probs[c] for c in CLASS_NAMES]
#     colors = [CLASS_COLORS[c] if c == pred else "#334155" for c in CLASS_NAMES]
#     ax.barh(CLASS_NAMES, vals, color=colors, height=0.6, edgecolor="none")
#     for i, (v, c) in enumerate(zip(vals, CLASS_NAMES)):
#         ax.text(min(v+1.5,110), i, f"{v:.1f}%", va="center", ha="left",
#                 color="white", fontsize=9,
#                 fontweight="bold" if c == pred else "normal")
#     ax.set_xlim(0, 118)
#     ax.set_xlabel("Confidence (%)", color="#94a3b8", fontsize=9)
#     ax.tick_params(colors="#94a3b8", labelsize=9)
#     ax.spines[:].set_color("#334155")
#     for lbl in ax.get_yticklabels():
#         lbl.set_color("#f8fafc" if lbl.get_text() == pred else "#94a3b8")
#         if lbl.get_text() == pred: lbl.set_fontweight("bold")
#     ax.set_title("Class Confidence Scores", color="#f8fafc", fontsize=11,
#                  fontweight="bold", pad=10)
#     plt.tight_layout()
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
#                 facecolor=fig.get_facecolor())
#     plt.close(fig)
#     buf.seek(0)
#     return Image.open(buf).copy()


# # ─── Gradio process function ───────────────────────────────────────────────────
# def process(image):
#     if image is None:
#         return None, None, '<div style="color:#475569;text-align:center;padding:40px;">Upload an image and click Analyze Lesion</div>', "⬆️ Upload an image to begin"

#     pil_img = Image.fromarray(image).convert("RGB")
#     t0 = time.time()

#     if MODEL_LOADED:
#         pred, probs, heatmap = _infer_gpu(pil_img)
#         mode = "🟢 Model Active — ZeroGPU (A100)"
#     else:
#         pred, probs, heatmap = _infer_demo(pil_img)
#         mode = "🔵 Demo Mode — example output only"

#     elapsed   = time.time() - t0
#     chart     = _chart(probs, pred)
#     top_conf  = probs[pred]
#     risk, risk_color = RISK_LEVEL[pred]
#     full_label = CLASS_LABELS[pred]
#     label_color = CLASS_COLORS[pred]

#     result_html = f"""
# <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid {risk_color}40;
# border-left:4px solid {risk_color};border-radius:12px;padding:20px 24px;font-family:'Inter',system-ui,sans-serif;">
#   <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
#     <div style="background:{label_color}22;border:2px solid {label_color};border-radius:50%;
#     width:52px;height:52px;display:flex;align-items:center;justify-content:center;font-size:1.6rem;">🔬</div>
#     <div>
#       <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:.1em;">Primary Diagnosis</div>
#       <div style="color:#f8fafc;font-size:1.25rem;font-weight:700;">{full_label}</div>
#     </div>
#   </div>
#   <div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;">
#     <div style="background:#ffffff0d;border-radius:8px;padding:10px 16px;flex:1;min-width:110px;">
#       <div style="color:#64748b;font-size:.7rem;text-transform:uppercase;margin-bottom:3px;">Confidence</div>
#       <div style="color:{label_color};font-size:1.6rem;font-weight:800;">{top_conf:.1f}%</div>
#     </div>
#     <div style="background:#ffffff0d;border-radius:8px;padding:10px 16px;flex:1;min-width:110px;">
#       <div style="color:#64748b;font-size:.7rem;text-transform:uppercase;margin-bottom:3px;">Risk Level</div>
#       <div style="color:{risk_color};font-size:.95rem;font-weight:700;">{risk}</div>
#     </div>
#     <div style="background:#ffffff0d;border-radius:8px;padding:10px 16px;flex:1;min-width:110px;">
#       <div style="color:#64748b;font-size:.7rem;text-transform:uppercase;margin-bottom:3px;">Inference Time</div>
#       <div style="color:#f8fafc;font-size:.95rem;font-weight:700;">{elapsed:.2f}s</div>
#     </div>
#   </div>
#   <div style="background:#dc262615;border:1px solid #dc262630;border-radius:8px;padding:10px 14px;font-size:.78rem;color:#fca5a5;">
#     ⚠️ <strong>Research tool only.</strong> Always consult a qualified dermatologist for clinical decisions.
#   </div>
# </div>"""
#     return heatmap, chart, result_html, mode


# # ─── CSS ──────────────────────────────────────────────────────────────────────
# CSS = """
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
# body,.gradio-container{font-family:'Inter',system-ui,sans-serif!important;background:#020817!important;color:#f8fafc!important;}
# .header-section{text-align:center;padding:40px 20px 24px;background:linear-gradient(180deg,#0a0f2e,#020817);border-bottom:1px solid #1e293b;position:relative;overflow:hidden;}
# .header-section::before{content:'';position:absolute;top:-60px;left:50%;transform:translateX(-50%);width:400px;height:200px;background:radial-gradient(ellipse,#6366f130,transparent 70%);pointer-events:none;}
# .header-title{font-size:clamp(1.8rem,5vw,2.8rem);font-weight:800;background:linear-gradient(135deg,#a5b4fc,#818cf8,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px;}
# .header-sub{color:#64748b;font-size:.95rem;max-width:600px;margin:0 auto 20px;}
# .badge-row{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:8px;}
# .badge{background:#1e293b;border:1px solid #334155;border-radius:20px;padding:4px 14px;font-size:.78rem;color:#94a3b8;}
# .badge-accent{border-color:#6366f150;color:#a5b4fc;background:#6366f115;}
# .stats-row{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;padding:20px;max-width:700px;margin:0 auto;}
# .stat-card{flex:1;min-width:130px;background:linear-gradient(135deg,#1e293b,#162032);border:1px solid #334155;border-radius:12px;padding:16px;text-align:center;}
# .stat-value{font-size:1.6rem;font-weight:800;background:linear-gradient(135deg,#a5b4fc,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
# .stat-label{color:#64748b;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;}
# .upload-zone{border:2px dashed #334155!important;border-radius:16px!important;background:#0f172a!important;transition:all .3s!important;min-height:260px!important;}
# .upload-zone:hover{border-color:#6366f1!important;box-shadow:0 0 24px #6366f120!important;}
# .analyze-btn{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:white!important;font-weight:700!important;font-size:1rem!important;border:none!important;border-radius:12px!important;padding:14px 32px!important;width:100%!important;box-shadow:0 4px 20px #6366f140!important;transition:all .3s!important;}
# .analyze-btn:hover{transform:translateY(-2px)!important;box-shadow:0 8px 30px #6366f160!important;}
# .clear-btn{background:transparent!important;color:#64748b!important;border:1px solid #334155!important;border-radius:12px!important;width:100%!important;}
# .clear-btn:hover{border-color:#ef4444!important;color:#ef4444!important;}
# .output-image-panel,.chart-panel{background:#0f172a!important;border:1px solid #1e293b!important;border-radius:16px!important;overflow:hidden!important;}
# .section-label{color:#64748b;font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;font-weight:600;margin-bottom:8px;}
# .warning-banner{background:linear-gradient(135deg,#78350f20,#92400e15);border:1px solid #d9770640;border-radius:10px;padding:12px 16px;color:#fbbf24;font-size:.82rem;text-align:center;}
# .footer-section{border-top:1px solid #1e293b;padding:24px 20px;text-align:center;color:#475569;font-size:.82rem;}
# .footer-section a{color:#6366f1;text-decoration:none;}
# .tabs{background:transparent!important;border:none!important;}
# .tab-nav button{background:#0f172a!important;color:#64748b!important;border:1px solid #1e293b!important;border-radius:8px!important;font-size:.85rem!important;}
# .tab-nav button.selected{background:#6366f115!important;color:#a5b4fc!important;border-color:#6366f150!important;}
# @media(max-width:768px){.stats-row{padding:16px 12px;gap:8px;}.stat-card{min-width:100px;}.stat-value{font-size:1.2rem;}}
# ::-webkit-scrollbar{width:6px;}::-webkit-scrollbar-track{background:#0f172a;}::-webkit-scrollbar-thumb{background:#334155;border-radius:3px;}
# """

# # ─── UI ───────────────────────────────────────────────────────────────────────
# def build_ui():
#     with gr.Blocks(css=CSS,
#                    theme=gr.themes.Base(primary_hue=gr.themes.colors.indigo,
#                                         neutral_hue=gr.themes.colors.slate,
#                                         font=gr.themes.GoogleFont("Inter")),
#                    title="DermaFusion-AI | Skin Lesion Classifier") as demo:

#         gr.HTML("""
#         <div class="header-section">
#           <div class="header-title">🔬 DermaFusion-AI</div>
#           <div class="header-sub">Research-grade skin lesion classification — EVA-02 Large + ConvNeXt V2 fusion with GradCAM++ explainability</div>
#           <div class="badge-row">
#             <span class="badge badge-accent">EVA-02 Large 307M</span>
#             <span class="badge badge-accent">ConvNeXt V2 Base</span>
#             <span class="badge badge-accent">Bidirectional Cross-Attention</span>
#             <span class="badge">7 Lesion Classes</span>
#             <span class="badge">ZeroGPU A100</span>
#             <span class="badge">GradCAM++ XAI</span>
#           </div>
#         </div>
#         <div class="stats-row">
#           <div class="stat-card"><div class="stat-value">0.9908</div><div class="stat-label">Macro AUC</div></div>
#           <div class="stat-card"><div class="stat-value">85.6%</div><div class="stat-label">Balanced Accuracy</div></div>
#           <div class="stat-card"><div class="stat-value">92.2%</div><div class="stat-label">MEL Sensitivity</div></div>
#           <div class="stat-card"><div class="stat-value">401M</div><div class="stat-label">Parameters</div></div>
#         </div>""")

#         with gr.Row(equal_height=False):
#             with gr.Column(scale=1, min_width=300):
#                 gr.HTML('<div class="section-label">Upload Dermoscopy Image</div>')
#                 image_input = gr.Image(label="", type="numpy", elem_classes=["upload-zone"], height=300, sources=["upload","clipboard"])
#                 gr.HTML('<div class="warning-banner" style="margin:12px 0;">📷 For best results use <strong>dermoscopy images</strong>. Smartphone photos may have lower accuracy.</div>')
#                 with gr.Row():
#                     analyze_btn = gr.Button("🔍 Analyze Lesion", elem_classes=["analyze-btn"], variant="primary")
#                     gr.ClearButton([image_input], value="✕ Clear", elem_classes=["clear-btn"])
#                 mode_display = gr.Textbox(value="⬆️ Upload an image to begin", label="", interactive=False, show_label=False, container=False)
#                 gr.HTML("""
#                 <div style="margin-top:20px;">
#                   <div class="section-label">Detectable Classes</div>
#                   <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;font-size:.78rem;color:#94a3b8;">
#                     <div style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#f59e0b;flex-shrink:0;"></span>Actinic Keratosis</div>
#                     <div style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#ef4444;flex-shrink:0;"></span>Basal Cell Carcinoma</div>
#                     <div style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#10b981;flex-shrink:0;"></span>Benign Keratosis</div>
#                     <div style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#3b82f6;flex-shrink:0;"></span>Dermatofibroma</div>
#                     <div style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#dc2626;flex-shrink:0;"></span><strong style="color:#fca5a5;">Melanoma ⚠️</strong></div>
#                     <div style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#6366f1;flex-shrink:0;"></span>Melanocytic Nevus</div>
#                     <div style="display:flex;align-items:center;gap:6px;"><span style="width:8px;height:8px;border-radius:2px;background:#ec4899;flex-shrink:0;"></span>Vascular Lesion</div>
#                   </div>
#                 </div>""")

#             with gr.Column(scale=2, min_width=400):
#                 with gr.Tabs():
#                     with gr.Tab("📊 Results"):
#                         result_html   = gr.HTML(value='<div style="color:#475569;text-align:center;padding:40px;font-size:.9rem;">Upload an image and click Analyze Lesion</div>')
#                         chart_output  = gr.Image(label="Confidence Scores", type="pil", elem_classes=["chart-panel"], height=300, show_label=False)
#                     with gr.Tab("🔥 GradCAM++ Heatmap"):
#                         gr.HTML('<div style="color:#64748b;font-size:.82rem;padding:8px 0 12px;">Warmer colors (red/yellow) = regions the model focused on most.</div>')
#                         heatmap_output = gr.Image(label="", type="pil", elem_classes=["output-image-panel"], height=380, show_label=False)
#                     with gr.Tab("ℹ️ About"):
#                         gr.HTML("""
#                         <div style="color:#94a3b8;font-size:.88rem;line-height:1.8;padding:8px 0;">
#                           <h3 style="color:#f8fafc;font-size:1rem;margin-bottom:10px;">Architecture</h3>
#                           <ul style="padding-left:18px;">
#                             <li><strong style="color:#a5b4fc;">Branch A — EVA-02 Large (307M):</strong> Processes full 448×448 image for global context</li>
#                             <li style="margin-top:6px;"><strong style="color:#a5b4fc;">Branch B — ConvNeXt V2 Base (88.5M):</strong> Processes segmented lesion for texture detail</li>
#                             <li style="margin-top:6px;"><strong style="color:#a5b4fc;">Fusion — Bidirectional Cross-Attention:</strong> Both branches communicate symmetrically</li>
#                           </ul>
#                           <h3 style="color:#f8fafc;font-size:1rem;margin:14px 0 8px;">Training</h3>
#                           <p>460,000+ images across 5 datasets: HAM10000, ISIC 2019/2020/2024, PH2 — patient-aware splits</p>
#                           <h3 style="color:#f8fafc;font-size:1rem;margin:14px 0 8px;">Key Results</h3>
#                           <table style="width:100%;border-collapse:collapse;font-size:.83rem;">
#                             <tr style="border-bottom:1px solid #1e293b;"><td style="padding:5px 0;color:#64748b;">Macro AUC</td><td style="color:#a5b4fc;font-weight:700;">0.9908</td></tr>
#                             <tr style="border-bottom:1px solid #1e293b;"><td style="padding:5px 0;color:#64748b;">Balanced Accuracy</td><td style="color:#a5b4fc;font-weight:700;">85.6%</td></tr>
#                             <tr style="border-bottom:1px solid #1e293b;"><td style="padding:5px 0;color:#64748b;">MEL Sensitivity</td><td style="color:#ef4444;font-weight:700;">92.2% (vs 86% avg. dermatologist)</td></tr>
#                             <tr><td style="padding:5px 0;color:#64748b;">DERM7PT AUC (unseen)</td><td style="color:#a5b4fc;font-weight:700;">0.872</td></tr>
#                           </table>
#                           <div style="margin-top:14px;padding:10px;background:#1e293b;border-radius:8px;font-size:.79rem;color:#64748b;">
#                             ⚠️ <strong style="color:#fbbf24;">Medical Disclaimer:</strong> Research prototype only. Not clinically validated. Consult a dermatologist.
#                           </div>
#                         </div>""")

#         analyze_btn.click(fn=process, inputs=[image_input],
#                           outputs=[heatmap_output, chart_output, result_html, mode_display],
#                           show_progress="full")

#         gr.HTML("""
#         <div class="footer-section">
#           <div style="margin-bottom:8px;">
#             <a href="https://github.com/ai-with-abdullah/DermaFusion-AI">GitHub</a> &nbsp;·&nbsp;
#             <a href="#">Paper (arXiv — Coming Soon)</a> &nbsp;·&nbsp;
#             <a href="https://www.isic-archive.com">ISIC Dataset</a>
#           </div>
#           Muhammad Abdullah · The Islamia University of Bahawalpur, Pakistan · 2026<br>
#           <span style="font-size:.75rem;">Trained on HAM10000, ISIC 2019/2020/2024, PH2 · 460,000+ images · Patient-aware splits</span>
#         </div>""")

#     return demo


# # ─── Entry ────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     load_models()
#     build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
