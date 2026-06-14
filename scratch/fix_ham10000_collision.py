import json

notebook_path = "/Users/macintosh/Desktop/DermaFusion/train_on_kaggle.ipynb"

# Read the notebook
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find and update the ham10000 linking call
updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("link(find_folder" in line for line in cell["source"]):
        source = cell["source"]
        for j, line in enumerate(source):
            if "link(find_folder(" in line and "'ham10000'" in line and "avoid_keywords" in line:
                source[j] = "link(find_folder(['skin-cancer-mnist', 'mnist-ham10000'], avoid_keywords=['part_', 'part1', 'part2', 'images', 'masks', 'segmentation', 'lesion-segmentations']),               'ham10000',  'HAM10000')\n"
                updated = True
                print(f"Updated line in notebook: {source[j].strip()}")
                break
        if updated:
            break

if updated:
    # Write the notebook back
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=2)
    print("✅ Notebook successfully updated with resolved HAM10000 collision keywords!")
else:
    print("⚠️ HAM10000 link call not found or already updated.")
