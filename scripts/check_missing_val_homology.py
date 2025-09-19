import os

STRUCTURE_DIR = "PDB"
LABEL_FILE = "datasets/PPBS/labels_validation_homology.txt"
OUTPUT_LOG = "missing_val_homology.log"

# Common structure file extensions to check
EXTENSIONS = [".bioent", ".cif", ".ent", ".pdb"]

# Read PDB IDs
with open(LABEL_FILE, "r") as f:
    lines = f.readlines()

pdb_ids = [line.strip().split()[0] for line in lines]
missing = []

for pid in pdb_ids:
    structure_id = pid.split("_")[0].lower()
    found = False
    for ext in EXTENSIONS:
        if os.path.exists(os.path.join(STRUCTURE_DIR, f"{structure_id}{ext}")) \
           or os.path.exists(os.path.join(STRUCTURE_DIR, f"pdb{structure_id}{ext}")):
            found = True
            break
    if not found:
        missing.append(pid)

# Write results
with open(OUTPUT_LOG, "w") as f:
    for pid in missing:
        f.write(pid + "\n")

print(f"[Done] Checked {len(pdb_ids)} entries. Missing: {len(missing)} — see {OUTPUT_LOG}")
