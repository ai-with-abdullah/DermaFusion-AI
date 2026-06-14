import json

with open("train_on_kaggle.ipynb", "r") as f:
    nb = json.load(f)

new_cells = []
for cell in nb["cells"]:
    # Locate Step 8 markdown cell
    if cell["cell_type"] == "markdown" and any("## Step 8: Full Evaluation" in line for line in cell["source"]):
        # Insert new markdown and code cells instead of the original Step 8
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 8: Evaluation & Ablation Study Results\n",
                "You can evaluate each of your trained ablation configurations individually using the cells below. Make sure you have completed the corresponding training step first so that the weights exist."
            ]
        })
        
        # Eval 1
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8a: Evaluate Ablation 1 \u2014 No TTA (Evaluates the standard weights without TTA)"
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python evaluate.py --ablation no_tta 2>&1 | tee /kaggle/working/evaluate_no_tta.log\n",
                "print('\\n\u2705 Ablation 1 evaluation complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Eval 2
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8b: Evaluate Ablation 2 \u2014 ConvNeXt Only (Bypasses EVA-02)"
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python evaluate.py --ablation convnext_only 2>&1 | tee /kaggle/working/evaluate_convnext_only.log\n",
                "print('\\n\u2705 Ablation 2 evaluation complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Eval 3
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8c: Evaluate Ablation 3 \u2014 EVA-02 Only (Bypasses ConvNeXt)"
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python evaluate.py --ablation eva_only 2>&1 | tee /kaggle/working/evaluate_eva_only.log\n",
                "print('\\n\u2705 Ablation 3 evaluation complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Eval 4
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8d: Evaluate Ablation 4 \u2014 No Cross-Attention (Fuses via simple average)"
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python evaluate.py --ablation no_attention 2>&1 | tee /kaggle/working/evaluate_no_attention.log\n",
                "print('\\n\u2705 Ablation 4 evaluation complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Eval 5
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8e: Evaluate Ablation 5 \u2014 No Segmentation (Original image passed to ConvNeXt)"
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python evaluate.py --ablation no_segmentation 2>&1 | tee /kaggle/working/evaluate_no_segmentation.log\n",
                "print('\\n\u2705 Ablation 5 evaluation complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })

        # Full Model Eval
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8f: Evaluate Full Model (Main Dual-Branch + TTA)\n",
                "Running this cell evaluates the final model, runs the Ablation Study comparison script, and generates the final dashboard."
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": [
                "%cd /kaggle/working/DermaFusion-AI\n",
                "!PYTHONPATH=. python evaluate.py 2>&1 | tee /kaggle/working/evaluate_full.log\n",
                "print('\\n\u2705 Full Model evaluation and summary dashboard complete!')"
            ],
            "execution_count": None,
            "outputs": []
        })
        
        continue

    # Skip original Step 8 code cell
    if cell["cell_type"] == "code" and any("!PYTHONPATH=. python evaluate.py 2>&1" in line for line in cell["source"]) and not any("--ablation" in line for line in cell["source"]):
        continue

    new_cells.append(cell)

nb["cells"] = new_cells
with open("train_on_kaggle.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
print("Notebook updated successfully with separate ablation evaluation cells!")
