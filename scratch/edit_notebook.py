import json
import os

notebook_path = "/Users/macintosh/Desktop/DermaFusion/train_on_kaggle.ipynb"

# Read the notebook
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find and update Step 4 (linking datasets)
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("isic-2024" in line for line in cell["source"]):
        source = cell["source"]
        for j, line in enumerate(source):
            if "isic-2024" in line:
                # Insert pad-ufes linking after isic-2024
                source.insert(j + 1, 'link(find_folder([\'pad-ufes\', \'padufes\']),                         \'pad_ufes\',  \'PAD-UFES-20\')\n')
                # Insert release_v0 linking
                source.insert(j + 2, '\n')
                source.insert(j + 3, 'derm7pt_src = find_folder([\'derm7pt\', \'release-v0\', \'release_v0\'])\n')
                source.insert(j + 4, 'if derm7pt_src:\n')
                source.insert(j + 5, '    dst = \'/kaggle/working/DermaFusion-AI/release_v0\'\n')
                source.insert(j + 6, '    if not os.path.exists(dst):\n')
                source.insert(j + 7, '        os.symlink(derm7pt_src, dst)\n')
                source.insert(j + 8, '        print(f\'✅ DERM7PT: linked to {dst}\')\n')
                break
        break

# Append Step 9 cells
step9_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 9: Advanced Paper Evaluations\n",
        "Runs calibration, bootstrap confidence intervals, McNemar significance tests, inference benchmarking, 5-fold CV, and external smartphone/dermoscopy evaluations (PAD-UFES-20 and DERM7PT)."
    ]
}

step9_code = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "%cd /kaggle/working/DermaFusion-AI\n",
        "\n",
        "print('=== 1. Probability Calibration (Temperature Scaling) ===')\n",
        "!PYTHONPATH=. python -m evaluation.run_temperature_scaling\n",
        "\n",
        "print('\\n=== 2. Statistical Significance (McNemar\\'s Test) ===')\n",
        "!PYTHONPATH=. python -m evaluation.run_statistical_tests\n",
        "\n",
        "print('\\n=== 3. Statistical Confidence Intervals (Bootstrap) ===')\n",
        "!PYTHONPATH=. python -m evaluation.run_confidence_intervals\n",
        "\n",
        "print('\\n=== 4. Inference Latency & Speed Benchmark ===')\n",
        "!PYTHONPATH=. python -m evaluation.run_inference_benchmark\n",
        "\n",
        "print('\\n=== 5. 5-Fold Cross-Validation (EVA-02 Small) ===')\n",
        "!PYTHONPATH=. python -m evaluation.run_cross_validation\n",
        "\n",
        "print('\\n=== 6. Smartphone External Test (PAD-UFES-20) ===')\n",
        "!PYTHONPATH=. python test_padufes.py\n",
        "\n",
        "print('\\n=== 7. Smartphone Head Fine-Tuning ===')\n",
        "!PYTHONPATH=. python finetune_padufes.py\n",
        "\n",
        "print('\\n=== 8. DERM7PT External Test (4 combinations) ===')\n",
        "!PYTHONPATH=. python test_both_weights.py\n",
        "\n",
        "print('\\n✅ All advanced paper evaluations are complete!')"
    ],
    "execution_count": None,
    "outputs": []
}

# Only append if Step 9 is not already there
has_step9 = any("Step 9:" in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
if not has_step9:
    nb["cells"].append(step9_md)
    nb["cells"].append(step9_code)

# Write the notebook back
with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=2)

print("✅ Notebook successfully updated!")
