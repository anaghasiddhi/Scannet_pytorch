#!/usr/bin/env python
import os, glob, time
import numpy as np
from Bio.PDB import MMCIFParser
from tqdm import tqdm

from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.datasets import ScanNetDataset

# --- Config ---
SPLIT = "test"  # which split to sample from
N = 100         # how many chains to enrich
DATA_DIR = "data"
STRUCTURES_DIR = os.environ.get("BIOLIP_STRUCTURES",
                                os.path.join("datasets", "BioLip", "structures"))

IN_DIR = os.path.join(DATA_DIR, f"{SPLIT}.data")
OUT_DIR = os.path.join(DATA_DIR, f"{SPLIT}.atom_enriched")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Collect sample npys ---
files = sorted(glob.glob(os.path.join(IN_DIR, "*.npy")))[:N]
print(f"Found {len(files)} files to enrich from {IN_DIR}")

parser = MMCIFParser(QUIET=True)

start = time.time()
processed = 0
for f in tqdm(files, desc="Enriching"):
    data = np.load(f, allow_pickle=True).item()
    pdb_id = os.path.basename(f).split(".")[0]
    base, chain_id = pdb_id.split("_", 1)
    cif = os.path.join(STRUCTURES_DIR, f"{base.lower()}.cif")
    if not os.path.exists(cif):
        print("Skip (no CIF):", pdb_id)
        continue
    try:
        structure = parser.get_structure(base, cif)
        if chain_id not in structure[0]:
            print("Skip (no chain):", pdb_id)
            continue
        chain = structure[0][chain_id]

        coords, atom_types, res_idx = [], [], []
        for residue in chain:
            for atom in residue:
                coords.append(atom.coord)
                atom_types.append(atom.element)
                res_idx.append(residue.id[1])

        data["coord_atom"] = np.array(coords, dtype=np.float32)
        data["indices_atom"] = np.array(res_idx, dtype=np.int32)
        data["attr_atom"] = np.array(atom_types)  # keep as strings for now

        np.save(os.path.join(OUT_DIR, f"{pdb_id}.npy"), data, allow_pickle=True)
        processed += 1
    except Exception as e:
        print("Skip:", pdb_id, e)

elapsed = time.time() - start
print(f"✅ Enriched {processed}/{len(files)} in {elapsed:.2f} s "
      f"(avg {elapsed/max(processed,1):.3f} s per entry)")

# --- Extrapolate full runtime ---
TOTAL = 126692  # all chains across splits
if processed > 0:
    est_total = (TOTAL / processed) * elapsed
    print(f"⏱ Estimated full enrichment (single-core) = {est_total/3600:.2f} h")

# --- Dry run check ---
print("\n🔍 Dry-run loading with pipeline…")
label_pkl = os.path.join(DATA_DIR, "BioLip_final_test.pkl")
import pickle
with open(label_pkl, "rb") as f:
    labels = pickle.load(f)

subset_ids = [os.path.basename(f).replace(".npy","") for f in files]
pipeline = ScanNetPipeline(atom_features="valency",
                           aa_features="sequence",
                           aa_frames="triplet_sidechain",
                           Beff=500)
dataset = ScanNetDataset(subset_ids, labels, pipeline, OUT_DIR)
ok = 0
for i in range(min(5, len(dataset))):
    out = dataset[i]
    if out is not None:
        ok += 1
print(f"Dry run loaded {ok}/{min(5,len(dataset))} examples successfully ✅")
