import os
import traceback
import numpy as np
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.utilities import io_utils
from Bio.PDB import PDBParser
from scannet_pytorch.preprocessing.PDBio import load_chains

# === Set the correct absolute path to labels ===
train_labels_path = os.path.abspath("datasets/PPBS/labels_train.txt")

# === Initialize pipeline ===
pipeline = ScanNetPipeline(
    atom_features="valency",
    aa_features="sequence",
    aa_frames="triplet_sidechain",
    Beff=500
)

# === Read label file ===
origins, _, resids, labels_raw = io_utils.read_labels(train_labels_path)
print(f"Total training examples: {len(origins)}")

# === Normalize resids ===
def normalize_resids(raw_resids):
    # raw_resids: array of shape (N, 2), e.g., [['A', '0'], ['A', '1']]
    # Output: list of tuples: [('A', 0, ' '), ('A', 1, ' '), ...]
    return [(chain, int(resseq), ' ') for chain, resseq in raw_resids]

resids = [normalize_resids(r) for r in resids]

# === Load and parse first PDB chain using .bioent loader ===
pdb_id_chain = origins[0]  # e.g. "13gs_0-A"
pdb_id = pdb_id_chain.split("_")[0]  # "13gs"
chain_id = pdb_id_chain.split("-")[1]  # "A"

# === Use loader that supports .bioent ===
structure, chains = load_chains(pdb_id=pdb_id, chain_ids='all')

# === Debug structure ===
print(f"Returned structure ID: {structure.id}")
print(f"Chain IDs available: {[c.id for c in chains]}")

# === Extract correct chain ===
chain_obj = next((c for c in chains if c.id == chain_id), None)

if chain_obj is None:
    raise ValueError(f"❌ Chain {chain_id} not found in structure {pdb_id}")

# === Try processing the first example ===
try:
    print(f"Resid type: {type(resids[0])}")
    print(f"Resid sample (first 3): {resids[0][:3]}")
    print(f"→ Attempting: {origins[0]}")
    print(f"Label type: {type(labels_raw[0])}")
    print(f"Label content (first few items): {labels_raw[0][:10]}")
    result = pipeline.process_example(
        chain_obj=chain_obj,
        labels=labels_raw[0]
    )


    if result is None:
        print("[❌] process_example() returned None — no usable output")
    else:
        x, y = result
        print("[✅] process_example() succeeded.")
        print(f"→ x type: {type(x)}")
        print(f"→ y type: {type(y)}")

        if isinstance(x, list):
            print(f"→ x is a list with {len(x)} parts")

            for i, x_part in enumerate(x):
                shape = x_part.shape
                dtype = x_part.dtype
                mean = x_part.mean()
                std = x_part.std()
                print(f"  x[{i}] → shape: {shape}, dtype: {dtype}, mean: {mean:.4f}, std: {std:.4f}")

        if isinstance(y, np.ndarray):
            print(f"→ y shape: {y.shape}, dtype: {y.dtype}")
            print(f"→ y unique values: {np.unique(y, return_counts=True)}")
            print(f"→ Positive labels: {y.sum()}")

except Exception as e:
    print(f"[💥] Failed with exception:\n{e}")
    traceback.print_exc()
