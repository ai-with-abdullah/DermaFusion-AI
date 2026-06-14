import json

with open("train_on_kaggle.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    # Modify Step 7 markdown
    if cell["cell_type"] == "markdown" and any("## Step 7: Save Weights" in line for line in cell["source"]):
        cell["source"] = [
            "## Step 7: Export Weights, Results and Plots to Output Tab\n",
            "Copies the trained model weights, the **Ablation Study results CSV**, the **Evaluation Dashboard**, and all individual plots to `/kaggle/working/` so they appear in the Kaggle Output panel on the right for easy download."
        ]
    
    # Modify Step 7 code
    if cell["cell_type"] == "code" and any("weights_src = " in line for line in cell["source"]) and any("shutil.copy" in line for line in cell["source"]):
        cell["source"] = [
            "import shutil, os\n",
            "\n",
            "# 1. Export Weights\n",
            "weights_src = '/kaggle/working/DermaFusion-AI/outputs/weights/'\n",
            "if os.path.exists(weights_src):\n",
            "    for f in os.listdir(weights_src):\n",
            "        if f.endswith('.pth'):\n",
            "            dst = f'/kaggle/working/{f}'\n",
            "            shutil.copy(os.path.join(weights_src, f), dst)\n",
            "            print(f'\u2705 Weights: {f}  ({os.path.getsize(dst)/1e6:.0f} MB)')\n",
            "else:\n",
            "    print('\u26a0\ufe0f No weights folder found (yet)')\n",
            "\n",
            "# 2. Export Ablation CSV Results\n",
            "csv_path = '/kaggle/working/DermaFusion-AI/outputs/ablation_study_results.csv'\n",
            "if os.path.exists(csv_path):\n",
            "    shutil.copy(csv_path, '/kaggle/working/ablation_study_results.csv')\n",
            "    print('\u2705 Results: ablation_study_results.csv')\n",
            "\n",
            "# 3. Export Plots and Dashboard\n",
            "plots_src = '/kaggle/working/DermaFusion-AI/outputs/plots/'\n",
            "if os.path.exists(plots_src):\n",
            "    for f in os.listdir(plots_src):\n",
            "        if f.endswith(('.png', '.jpg', '.jpeg')):\n",
            "            shutil.copy(os.path.join(plots_src, f), f'/kaggle/working/{f}')\n",
            "            print(f'\u2705 Plot: {f}')\n",
            "\n",
            "print('\\n\ud83d\udce6 Files exported to /kaggle/working/ and ready for download!')"
        ]
        
    # Modify Step 8 markdown
    if cell["cell_type"] == "markdown" and any("## Step 8: Full Evaluation" in line for line in cell["source"]):
        cell["source"] = [
            "## Step 8: Full Evaluation & Ablation Study\n",
            "\n",
            "### \ud83d\udcc5 Exam Preparation Mode (Run Ablation Study NOW):\n",
            "If you want to run the ablation study **now** using your existing weights (to avoid waiting 24+ hours for training during exams):\n",
            "1. Skip Step 5 and Step 6 cells.\n",
            "2. Copy your existing model weights to `/kaggle/working/DermaFusion-AI/outputs/weights/best_unet.pth` and `best_dual_branch_fusion.pth` (or let the script load them from your active persistent inputs).\n",
            "3. Run this cell below. It will run the full evaluation, generate separate confusion matrices, run the Ablation Study configurations 1-6, and generate the final dashboard.\n",
            "4. Run **Step 7** afterwards to make sure the plots and CSV are copied to the Output tab for downloading.\n",
            "\n",
            "### \ud83c\udfc1 Training from Scratch (After Exams in 2 weeks):\n",
            "1. Clear all old weights: `!rm -rf /kaggle/working/DermaFusion-AI/outputs/weights/*` in a cell.\n",
            "2. Run Step 5 (Segmentation) and Step 6 (Classifier) to train fresh from zero.\n",
            "3. Run this cell to evaluate your new final models."
        ]

with open("train_on_kaggle.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
print("Notebook updated successfully!")
