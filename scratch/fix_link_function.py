import json

notebook_path = "/Users/macintosh/Desktop/DermaFusion/train_on_kaggle.ipynb"

# Read the notebook
with open(notebook_path, "r") as f:
    nb = json.load(f)

# Find and update the link function cell
updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("def link(src, dst_name, label):" in line for line in cell["source"]):
        source = cell["source"]
        
        # We will replace the link function implementation
        new_source = []
        skip = False
        for line in source:
            if "def link(src, dst_name, label):" in line:
                new_source.append("def link(src, dst_name, label):\n")
                new_source.append("    dst = os.path.join(data_dir, dst_name)\n")
                new_source.append("    if os.path.lexists(dst):\n")
                new_source.append("        try:\n")
                new_source.append("            if os.path.islink(dst):\n")
                new_source.append("                os.unlink(dst)\n")
                new_source.append("            elif os.path.isdir(dst):\n")
                new_source.append("                import shutil\n")
                new_source.append("                shutil.rmtree(dst)\n")
                new_source.append("            else:\n")
                new_source.append("                os.remove(dst)\n")
                new_source.append("        except Exception:\n")
                new_source.append("            pass\n")
                new_source.append("    if src:\n")
                new_source.append("        os.symlink(src, dst)\n")
                new_source.append("        print(f'✅ {label}: linked from {src}')\n")
                new_source.append("    else:\n")
                new_source.append("        print(f'❌ {label}: NOT FOUND')\n")
                skip = True
            elif skip:
                # Skip the old implementation until we reach the calls
                if "link(find_folder(" in line or "derm7pt_src =" in line:
                    skip = False
                    new_source.append(line)
            else:
                new_source.append(line)
        
        cell["source"] = new_source
        updated = True
        break

if updated:
    # Write the notebook back
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=2)
    print("✅ Notebook successfully updated with robust link function!")
else:
    print("⚠️ Link function cell not found in notebook.")
