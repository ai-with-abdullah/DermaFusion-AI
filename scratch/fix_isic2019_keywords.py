import json

notebook_path = "/Users/macintosh/Desktop/DermaFusion/train_on_kaggle.ipynb"

# Read the notebook
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find and update Step 4 (linking datasets)
updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("isic_2019" in line for line in cell["source"]):
        source = cell["source"]
        for j, line in enumerate(source):
            if "isic-2019" in line and "andrewmvd" not in line:
                source[j] = line.replace(
                    "['isic-2019', 'isic2019', 'skin-lesion-images']",
                    "['isic-2019', 'isic2019', 'skin-lesion-images', 'andrewmvd', 'salviohexia']"
                )
                updated = True
                print(f"Updated line in notebook: {source[j].strip()}")
                break
        if updated:
            break

if updated:
    # Write the notebook back
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=2)
    print("✅ Notebook successfully updated with ISIC 2019 fallbacks!")
else:
    print("⚠️ No updates made (keywords might already be updated or cell not found).")
