import json

notebook_path = "/Users/macintosh/Desktop/DermaFusion/train_on_kaggle.ipynb"

# Read the notebook
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find and update the find_folder cell
updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("def find_folder(keywords):" in line for line in cell["source"]):
        source = cell["source"]
        
        # We will replace the find_folder and calls
        new_source = []
        skip = False
        for line in source:
            if "def find_folder(keywords):" in line:
                new_source.append("def find_folder(keywords, avoid_keywords=None):\n")
                new_source.append("    # Search up to 3 levels deep in /kaggle/input\n")
                new_source.append("    paths_to_check = ['/kaggle/input']\n")
                new_source.append("    checked_paths = set()\n")
                new_source.append("    for depth in range(3):\n")
                new_source.append("        next_paths = []\n")
                new_source.append("        for p in paths_to_check:\n")
                new_source.append("            if not os.path.exists(p) or p in checked_paths:\n")
                new_source.append("                continue\n")
                new_source.append("            checked_paths.add(p)\n")
                new_source.append("            try:\n")
                new_source.append("                for item in os.listdir(p):\n")
                new_source.append("                    full = os.path.join(p, item)\n")
                new_source.append("                    if os.path.isdir(full):\n")
                new_source.append("                        if any(kw in item.lower() for kw in keywords):\n")
                new_source.append("                            if avoid_keywords and any(akw in item.lower() for akw in avoid_keywords):\n")
                new_source.append("                                continue\n")
                new_source.append("                            # Check for a deeper subfolder that matches\n")
                new_source.append("                            try:\n")
                new_source.append("                                for sub in os.listdir(full):\n")
                new_source.append("                                    sub_full = os.path.join(full, sub)\n")
                new_source.append("                                    if os.path.isdir(sub_full) and any(kw in sub.lower() for kw in keywords):\n")
                new_source.append("                                        if avoid_keywords and any(akw in sub.lower() for akw in avoid_keywords):\n")
                new_source.append("                                            continue\n")
                new_source.append("                                        return sub_full\n")
                new_source.append("                            except Exception:\n")
                new_source.append("                                pass\n")
                new_source.append("                            return full\n")
                new_source.append("                        next_paths.append(full)\n")
                new_source.append("            except Exception:\n")
                new_source.append("                pass\n")
                new_source.append("        paths_to_check = next_paths\n")
                new_source.append("    return None\n")
                skip = True
            elif skip:
                # Skip the old implementation of find_folder
                if "def link(" in line:
                    skip = False
                    new_source.append(line)
            elif "link(find_folder(['ham10000', 'skin-cancer-mnist'])," in line:
                new_source.append("link(find_folder(['ham10000', 'skin-cancer-mnist'], avoid_keywords=['part_', 'part1', 'part2', 'images']),               'ham10000',  'HAM10000')\n")
            else:
                new_source.append(line)
        
        cell["source"] = new_source
        updated = True
        break

if updated:
    # Write the notebook back
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=2)
    print("✅ Notebook successfully updated with avoid_keywords in find_folder!")
else:
    print("⚠️ find_folder cell not found in notebook.")
