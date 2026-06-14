import json

with open("train_on_kaggle.ipynb", "r") as f:
    nb = json.load(f)

new_cells = []
for cell in nb["cells"]:
    # Locate Step 6 markdown cell
    if cell["cell_type"] == "markdown" and any("## Step 6: Phase 2" in line for line in cell["source"]):
        # Insert new markdown and code cells instead of the original Step 6
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Phase 2 \u2014 Classifier Training & Ablations\n",
                "To run the Ablation Study by training each model from scratch, you can execute the cells below one-by-one. Each cell trains a specific configuration and saves its weights separately so they do not overwrite each other."
            ]
        })
        
        # Ablation 1
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6a: Ablation 1 \u2014 No TTA Training (Uses standard training but saves separately)\n",
                "Estimated training time: ~2 hours per epoch. Trains the full model structure but saves to `best_classifier_no_tta.pth`."
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python train_classifier.py --ablation no_tta 2>&1 | tee /kaggle/working/train_no_tta.log\n",
                "print('\\n\u2705 Ablation 1 training complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Ablation 2
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6b: Ablation 2 \u2014 ConvNeXt Only Training (Bypasses EVA-02 and fusion)\n",
                "Estimated training time: ~30 minutes per epoch (much faster as it is a single backbone!). Saves to `best_classifier_convnext_only.pth`."
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python train_classifier.py --ablation convnext_only 2>&1 | tee /kaggle/working/train_convnext_only.log\n",
                "print('\\n\u2705 Ablation 2 training complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Ablation 3
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6c: Ablation 3 \u2014 EVA-02 Only Training (Bypasses ConvNeXt and fusion)\n",
                "Estimated training time: ~1.2 hours per epoch. Saves to `best_classifier_eva_only.pth`."
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python train_classifier.py --ablation eva_only 2>&1 | tee /kaggle/working/train_eva_only.log\n",
                "print('\\n\u2705 Ablation 3 training complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Ablation 4
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6d: Ablation 4 \u2014 No Cross-Attention Training (Fuses via simple average)\n",
                "Estimated training time: ~2 hours per epoch. Saves to `best_classifier_no_attention.pth`."
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python train_classifier.py --ablation no_attention 2>&1 | tee /kaggle/working/train_no_attention.log\n",
                "print('\\n\u2705 Ablation 4 training complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Ablation 5
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6e: Ablation 5 \u2014 No Segmentation Training (Original image to ConvNeXt)\n",
                "Estimated training time: ~2 hours per epoch. Saves to `best_classifier_no_segmentation.pth`."
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python train_classifier.py --ablation no_segmentation 2>&1 | tee /kaggle/working/train_no_segmentation.log\n",
                "print('\\n\u2705 Ablation 5 training complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Full Model
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6f: Full Model Training (Dual-Branch + Segmentation + Cross-Attention + TTA)\n",
                "Estimated training time: ~2 hours per epoch. This is your main final model. Saves to `best_dual_branch_fusion.pth`."
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python train_classifier.py 2>&1 | tee /kaggle/working/train_classifier.log\n",
                "print('\\n\u2705 Full Model training complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })
        
        continue

    # Skip original Step 6 code cell
    if cell["cell_type"] == "code" and any("!PYTHONPATH=. python train_classifier.py 2>&1" in line for line in cell["source"]) and not any("--ablation" in line for line in cell["source"]):
        continue

    new_cells.append(cell)

nb["cells"] = new_cells
with open("train_on_kaggle.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
print("Notebook updated successfully with separate ablation training cells!")
