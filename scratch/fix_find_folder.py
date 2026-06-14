import json

notebook_path = "/Users/macintosh/Desktop/DermaFusion/train_on_kaggle.ipynb"

# Read the notebook
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Define the new robust find_folder code lines
new_find_folder_lines = [
    "def find_folder(keywords):\n",
    "    # Search up to 3 levels deep in /kaggle/input\n",
    "    paths_to_check = ['/kaggle/input']\n",
    "    checked_paths = set()\n",
    "    for depth in range(3):\n",
    "        next_paths = []\n",
    "        for p in paths_to_check:\n",
    "            if not os.path.exists(p) or p in checked_paths:\n",
    "                continue\n",
    "            checked_paths.add(p)\n",
    "            try:\n",
    "                for item in os.listdir(p):\n",
    "                    full = os.path.join(p, item)\n",
    "                    if os.path.isdir(full):\n",
    "                        if any(kw in item.lower() for kw in keywords):\n",
    "                            return full\n",
    "                        next_paths.append(full)\n",
    "            except Exception:\n",
    "                pass\n",
    "        paths_to_check = next_paths\n",
    "    return None\n"
]

# Find and replace the find_folder definition
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("def find_folder" in line for line in cell["source"]):
        source = cell["source"]
        
        # Find start and end indices of the old find_folder definition
        start_idx = -1
        end_idx = -1
        for idx, line in enumerate(source):
            if "def find_folder" in line:
                start_idx = idx
            if start_idx != -1 and "return None" in line:
                end_idx = idx + 1
                break
                
        if start_idx != -1 and end_idx != -1:
            # Replace the old lines with new lines
            cell["source"] = source[:start_idx] + new_find_folder_lines + source[end_idx:]
            print("Found and replaced find_folder function!")
            break

# Write back
with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=2)

print("✅ Notebook successfully updated with robust search function!")
