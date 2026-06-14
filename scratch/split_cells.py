import json

notebook_path = "/Users/macintosh/Desktop/DermaFusion/train_on_kaggle.ipynb"

# Read the notebook
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find the combined cell
target_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and any("PHASE 1" in line for line in cell["source"]) and any("PHASE 2" in line for line in cell["source"]):
        target_idx = i
        break

if target_idx != -1:
    # We found the combined cell. Now we will remove the combined markdown + code cell
    # and replace them with separate cells.
    
    # 1. Step 5 Markdown
    step5_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 5: Phase 1 — Segmentation Training (Swin-UNet)\n",
            "Trains the Swin-Transformer U-Net to perform lesion masking. Saves checkpoints after each epoch to `/kaggle/working/DermaFusion-AI/outputs/weights/resume_seg_checkpoint.pth`. If the session stops, re-running this cell will automatically resume from the last epoch."
        ]
    }
    
    # 2. Step 5 Code
    step5_code = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "%cd /kaggle/working/DermaFusion-AI\n",
            "print('=' * 60)\n",
            "print('🚀 PHASE 1 — Segmentation Training (25 epochs)')\n",
            "print('   Swin-Tiny UNet | SEG_BATCH_SIZE=8 | 2× T4 GPUs')\n",
            "print('=' * 60)\n",
            "!PYTHONPATH=. python train_segmentation.py 2>&1 | tee /kaggle/working/train_segmentation.log\n",
            "print('\\n✅ Phase 1 complete!')"
        ],
        "execution_count": None,
        "outputs": []
    }
    
    # 3. Step 6 Markdown
    step6_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6: Phase 2 — Classifier Training (EVA-02 + ConvNeXt V2)\n",
            "Trains the dual-branch fusion classification model. Automatically loads the trained Swin-UNet weights. Saves checkpoints after each epoch to `/kaggle/working/DermaFusion-AI/outputs/weights/resume_checkpoint.pth`. Re-running this cell will automatically resume from the last epoch."
        ]
    }
    
    # 4. Step 6 Code
    step6_code = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "%cd /kaggle/working/DermaFusion-AI\n",
            "print('=' * 60)\n",
            "print('🚀 PHASE 2 — Classifier Training (25 epochs)')\n",
            "print('   EVA-02 Large + ConvNeXt V2 | BATCH_SIZE=2 | 2× T4 GPUs')\n",
            "print('=' * 60)\n",
            "!PYTHONPATH=. python train_classifier.py 2>&1 | tee /kaggle/working/train_classifier.log\n",
            "print('\\n✅ Phase 2 complete!')"
        ],
        "execution_count": None,
        "outputs": []
    }
    
    # Replace the combined markdown cell (at target_idx - 1) and the code cell (at target_idx)
    # with the 4 new cells
    nb["cells"] = nb["cells"][:target_idx - 1] + [step5_md, step5_code, step6_md, step6_code] + nb["cells"][target_idx + 1:]
    print("Successfully split combined training cells!")

# Write back
with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=2)

print("✅ Notebook successfully updated!")
