import json
import os

notebook_path = "/Users/macintosh/Desktop/DermaFusion/train_on_kaggle.ipynb"

# Read the notebook
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find and update Step 4 (linking datasets)
updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("pad_ufes" in line for line in cell["source"]):
        source = cell["source"]
        for j, line in enumerate(source):
            if "pad_ufes" in line:
                # Insert Stanford DDI linking after pad-ufes linking
                # Check if it's already there
                if any("ddi" in l for l in source):
                    print("DDI link already exists in notebook cell.")
                    break
                source.insert(j + 1, "link(find_folder(['ddi', 'diverse-dermatology']),                  'ddi',       'Stanford DDI')\n")
                updated = True
                break
        if updated:
            break

if updated:
    # Write the notebook back
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=2)
    print("✅ Notebook successfully updated with Stanford DDI link!")
else:
    print("❌ Could not find the dataset linking cell or DDI already linked.")
