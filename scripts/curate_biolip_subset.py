import os
import csv
import requests
from Bio.PDB import PDBParser, Select, PDBIO
import numpy as np
from collections import defaultdict

# === CONFIG ===
biolip_txt_path = "datasets/BioLip/BioLiP_nr.txt"
output_pdb_dir = "datasets/BioLip/structures"
os.makedirs(output_pdb_dir, exist_ok=True)

# Filters
min_residues = 50
max_residues = 1000
allowed_exp_method = "X-RAY"
max_resolution = 2.5
exclude_ligands = {"HOH", "SO4", "CA", "NA", "MG", "ZN", "K"}

# === Storage ===
origins, resids_list, labels_list = [], [], []

# === Helpers ===
def download_pdb(pdb_id, out_path):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url)
    if r.status_code == 200:
        with open(out_path, "w") as f:
            f.write(r.text)
        return True
    return False

class ChainSelect(Select):
    def __init__(self, target_chain):
        self.target_chain = target_chain
    def accept_chain(self, chain):
        return chain.id == self.target_chain

def has_complete_ca(chain):
    return all("CA" in res for res in chain.get_residues() if res.id[0] == " ")

def extract_binding_mask(chain, binding_residues):
    res_ids = []
    labels = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        seq_id = res.get_id()[1]
        res_ids.append(seq_id)
        labels.append(1 if seq_id in binding_residues else 0)
    return res_ids, labels

# === Main ===
with open(biolip_txt_path) as f:
   print("✅ Opened BioLiP_nr.txt")
   reader = csv.reader(f, delimiter='\t')
   row0 = next(reader)
   print(f"🔍 First row: {row0}")
   f.seek(0)
   reader = csv.reader(f, delimiter="\t")

   for i, row in enumerate(reader):
       try:
           pdb_id, chain_id = row[0], row[1]
           ligand = row[4]
           exp_method = row[10]
           res_str = row[11]

           origin_id = f"{pdb_id.lower()}_0-{chain_id}"
           print(f"[Check] {origin_id} | ligand={ligand}, exp={exp_method}, res={res_str}")
           if i>50:
               break

           if ligand in exclude_ligands or exp_method != allowed_exp_method:
               continue
           try:
               resolution = float(res_str)
               if resolution > max_resolution:
                   continue
           except ValueError:
               continue
           origin_id = f"{pdb_id}_0-{chain_id}"
           pdb_path = os.path.join(output_pdb_dir, f"{origin_id}.pdb")
           if not os.path.exists(pdb_path):
               if not download_pdb(pdb_id.upper(), pdb_path):
                   continue

           parser = PDBParser(QUIET=True)
           structure = parser.get_structure(pdb_id, pdb_path)
           chain = structure[0][chain_id]
           if not has_complete_ca(chain):
               continue

           # Parse ligand-binding residues
           try:
               binding_residues = set(map(int, row[13].split(",")))
           except Exception as e:
               print(f"[Bad Binding Residue List] {origin_id}: {row[13]} — {e}")
               continue


           # Create per-residue label
           res_ids, bin_labels = extract_binding_mask(chain, binding_residues)
           if len(res_ids) < min_residues or len(res_ids) > max_residues:
               continue

           origins.append(origin_id)
           resids_list.append(np.array(res_ids))
           labels_list.append(np.array(bin_labels))

           if len(origins) % 100 == 0:
               print(f"[✓] Processed {len(origins)} proteins")

           if len(origins) >= 2500:
               print(f"🛑 Stopping at 2500 entries.")
               break

       except Exception as e:
           print(f"[Skip] {row[0]} {row[1]} - {e}")
           continue

# === Save for preprocess_datasets.py ===
import pickle
with open("datasets/BioLip/biolip_subset.pkl", "wb") as f:
    pickle.dump({
        "origins": origins,
        "resids": resids_list,
        "labels": labels_list
    }, f)

print(f"\n✅ Saved {len(origins)} filtered BioLiP entries.")
print("➡ Run `preprocess_split()` using the `biolip_subset.pkl` to generate .data + .pkl.")
