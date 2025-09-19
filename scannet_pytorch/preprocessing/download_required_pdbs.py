import warnings

# Suppress known harmless warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*reflected list.*")
warnings.filterwarnings("ignore", message=".*scheduled for deprecation.*")


import os
import pickle
import torch
import numpy as np
import time
from tqdm import tqdm

# Setup module path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("🚀 Script started...", flush=True)

from scannet_pytorch.datasets import ScanNetDataset
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.utilities import paths

# === Constants ===
MAX_LENGTH = 800

label_dict_paths = [
    "data/train_label_dict.pkl",
    "data/val_70_label_dict.pkl",
    "data/val_none_label_dict.pkl",
    "data/val_topology_label_dict.pkl",
    "data/val_homology_label_dict.pkl"
]

# === Create pipeline object ===

pipeline = ScanNetPipeline(
    atom_features="valency",
    aa_features="sequence",
    aa_frames="triplet_sidechain",
    Beff=500
)

print("ScanNetPipeline Initialized")
# === Ensure folder exists ===
os.makedirs(paths.pipeline_folder, exist_ok=True)

seen_ids = set()
print("📂 Loading label dictionaries...")
for pkl_path in label_dict_paths:
    if not os.path.exists(pkl_path):
        print(f"❌ ERROR: Missing file: {pkl_path}")
        continue

    try:
        with open(pkl_path, "rb") as f:
            label_dict = pickle.load(f)
            print(f"✅ Loaded {pkl_path} with {len(label_dict)} chains")
            seen_ids.update(label_dict.keys())
    except Exception as e:
        print(f"❌ Failed to load {pkl_path}: {e}")

print(f"\n🧬 Total unique chain IDs collected: {len(seen_ids)}")
if len(seen_ids) == 0:
    raise RuntimeError("❌ No chains found. Check your label dict files.")

# Optional: Preview the first few
print("🔍 Sample chain IDs:", sorted(list(seen_ids))[:5])

# === Create dummy dataset and force loading ===
print("\n📦 Creating ScanNetDataset with dummy labels...", flush=True)
try:
    dataset = ScanNetDataset(
        pdb_ids=sorted(seen_ids),
        label_dict={id_: np.zeros(MAX_LENGTH, dtype=np.int64) for id_ in seen_ids},
        pipeline=pipeline,
        pdb_folder=paths.structures_folder,
        max_length=MAX_LENGTH
    )
    print(f"✅ Dataset created with {len(dataset)} examples", flush=True)
except Exception as e:
    raise RuntimeError(f"❌ Dataset creation failed: {e}")

print("🔄 Forcing PDB file loads via __getitem__...\n", flush= True)

skipped = []
start_time = time.time()

for i in tqdm(range(len(dataset))):
    pdb_id = dataset.pdb_ids[i]
    try:
        print(f"[{i+1}/{len(dataset)}] Loading {pdb_id}...", flush=True)
        _ = dataset[i]
    except Exception as e:
        print(f"❌ Skipped {pdb_id}: {e}", flush=True)
        skipped.append((pdb_id, str(e)))

elapsed = time.time() - start_time
print(f"\n✅ Done. Time taken: {elapsed:.2f} seconds")
print(f"⛔ Total skipped: {len(skipped)}")

# Optional: Save skipped chains for retry
if skipped:
    with open("skipped_pdbs.txt", "w") as f:
        for pid, msg in skipped:
            f.write(f"{pid}\t{msg}\n")

    print("📄 Saved skipped PDB IDs to skipped_pdbs.txt")
    print("🔍 Sample skipped:")
    for x in skipped[:5]:
        print(" -", x)
