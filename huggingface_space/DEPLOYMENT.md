# Deploying DermaFusion-AI on Hugging Face Spaces
# =================================================

## Step 1 — Upload Model Weights to Hugging Face Hub

The demo needs your 2 model weight files uploaded to HF Hub so the Space can
download them at startup. Do this ONCE from the terminal or Google Colab.

### From Terminal (Mac):

```bash
pip install huggingface_hub

python3 -c "
from huggingface_hub import HfApi, login

# Login with your HF token (get from huggingface.co/settings/tokens)
login(token='YOUR_HF_TOKEN_HERE')

api = HfApi()

# Create model repository
api.create_repo('ai-with-abdullah/DermaFusion-AI', repo_type='model', exist_ok=True)

# Upload weights
api.upload_file(
    path_or_fileobj='outputs/weights/best_dual_branch_fusion.pth',
    path_in_repo='best_dual_branch_fusion.pth',
    repo_id='ai-with-abdullah/DermaFusion-AI',
    repo_type='model',
)
api.upload_file(
    path_or_fileobj='outputs/weights/best_unet.pth',
    path_in_repo='best_unet.pth',
    repo_id='ai-with-abdullah/DermaFusion-AI',
    repo_type='model',
)
print('Weights uploaded successfully!')
"
```

### From Google Colab (if weights are on Google Drive):

```python
!pip install huggingface_hub

from huggingface_hub import HfApi, login
login(token='YOUR_HF_TOKEN_HERE')

api = HfApi()
api.create_repo('ai-with-abdullah/DermaFusion-AI', repo_type='model', exist_ok=True)

api.upload_file(
    path_or_fileobj='/content/drive/MyDrive/YOUR_PATH/best_dual_branch_fusion.pth',
    path_in_repo='best_dual_branch_fusion.pth',
    repo_id='ai-with-abdullah/DermaFusion-AI',
    repo_type='model',
)
api.upload_file(
    path_or_fileobj='/content/drive/MyDrive/YOUR_PATH/best_unet.pth',
    path_in_repo='best_unet.pth',
    repo_id='ai-with-abdullah/DermaFusion-AI',
    repo_type='model',
)
print('Done!')
```

---

## Step 2 — Create the Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Name: `DermaFusion-AI`
4. SDK: **Gradio**
5. Visibility: **Public**
6. Hardware: **T4 GPU (Small)** — $0.60/hr, only runs when someone uses it
   - Free CPU tier also works but inference will be slow (~15–30 seconds)
7. Click **Create Space**

---

## Step 3 — Upload Space Files

Upload the 3 files from `huggingface_space/` folder:

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload Space files
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
SPACE_ID = 'ai-with-abdullah/DermaFusion-AI'

# Upload app files to the Space (not model repo)
for fname in ['app.py', 'requirements.txt', 'README.md']:
    api.upload_file(
        path_or_fileobj=f'huggingface_space/{fname}',
        path_in_repo=fname,
        repo_id=SPACE_ID,
        repo_type='space',
    )
    print(f'Uploaded {fname}')

print('Space files uploaded! Check: https://huggingface.co/spaces/ai-with-abdullah/DermaFusion-AI')
"
```

---

## Step 4 — Add HF Token as Space Secret

The Space needs to download model weights from your private model repo.

1. Go to your Space → Settings → **Repository Secrets**
2. Add: Name = `HF_TOKEN`, Value = your HF token
3. The `app.py` will automatically use `HF_TOKEN` env var

---

## Step 5 — Add Models Folder to Space

The Space also needs the `models/` Python files to load the architecture.
Upload the models folder:

```bash
python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi()

for fname in os.listdir('models'):
    if fname.endswith('.py'):
        api.upload_file(
            path_or_fileobj=f'models/{fname}',
            path_in_repo=f'models/{fname}',
            repo_id='ai-with-abdullah/DermaFusion-AI',
            repo_type='space',
        )
        print(f'Uploaded models/{fname}')
"
```

---

## Step 6 — Your Space is Live!

URL: `https://huggingface.co/spaces/ai-with-abdullah/DermaFusion-AI`

Add this URL to:
- [ ] MIDL 2026 paper
- [ ] arXiv submission
- [ ] GitHub README.md
- [ ] LinkedIn profile
- [ ] CV / scholarship applications

---

## Local Testing (Before Deploying)

Test the demo on your Mac first:

```bash
cd "/Users/macintosh/Desktop/Computer Vision Project"
pip install gradio torch torchvision timm pillow matplotlib
python3 huggingface_space/app.py
# Open: http://localhost:10000
```

In demo mode (without weights loaded), the app still runs with example outputs.

---

## Hardware Cost Estimate

| Hardware | Cost | Speed | Recommendation |
|---|---|---|---|
| CPU (free) | Free | ~25–40 sec/image | OK for demo |
| T4 GPU Small | $0.60/hr | ~2–4 sec/image | Best for paper demo |
| T4 GPU Medium | $0.90/hr | ~1–2 sec/image | Optional |

HF Spaces only charges when requests are actually being served.
If no one uses it for 30 min, it hibernates (free).

---
*DermaFusion-AI | Muhammad Abdullah | March 2026*
