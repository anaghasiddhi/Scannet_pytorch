import os, sys
#sys.stderr = open(os.devnull, 'w')
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))  # Adds /ScanNet to path

import torch
import numpy as np
from scannet_pytorch.datasets import ScanNetDataset
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.utilities import io_utils, paths


# === Config ===
label_file = os.path.abspath(os.path.join(paths.library_folder, "..", "datasets", "PPBS", "labels_train.txt"))
pdb_folder = paths.structures_folder  # typically PDB/ or structures/
max_length = 800
target_id = "2q03_0-A"  # You can change this

# === Step 1: Load label data ===
print("Loading lable data\n")
all_ids, _, all_resids, all_labels = io_utils.read_labels(label_file)
#print("[DEBUG] Available IDs:", all_ids[:20])
target_index = all_ids.index(target_id)

#print("[DEBUG] Available IDs:", list(label_dict.keys())[:20])  # Or use len() to get the count
pdb_id = all_ids[target_index]
label = all_labels[target_index]
resids = all_resids[target_index]

# Sanity
# print(f"[DEBUG] Selected ID: {pdb_id}")
# print(f"[DEBUG] Label shape: {label.shape}")
# print(f"[DEBUG] Resids shape: {resids.shape}")

# Reshape label
if label.ndim == 1:
    label = label[:, None]

# === Step 2: Build pipeline ===
print("Building Pipeline\n")
pipeline = ScanNetPipeline(
    atom_features="valency",
    aa_features="sequence",
    aa_frames="triplet_sidechain",
    Beff=500
)

# === Step 3: Process through pipeline ===
try:
    print("Fetching processed dataset\n")
    inputs, outputs, _, loaded = pipeline.build_processed_dataset(
        dataset_name="debug-single",
        list_origins=[pdb_id],
        list_resids=[resids],
        list_labels=[label],
        fresh=True,
        save=False,
        permissive=True
    )

    print("✅ build_processed_dataset succeeded.")
except Exception as e:
    print(f"❌ Failed during build_processed_dataset: {e}")
    raise

# === Optional: Try __getitem__ on dataset ===
print("Trying ScannetDataset")
dataset = ScanNetDataset(
    pdb_ids=[pdb_id],
    label_dict={pdb_id: label},
    pipeline=pipeline,
    pdb_folder=pdb_folder,
    max_length=max_length
)

print(f"\n[Dataset] Loaded: {dataset.loaded}, Skipped: {dataset.skipped}")
print(f"[Dataset] Skipped IDs: {dataset.skipped_ids}")

try:
    result = dataset[0]
    if result is None:
        print("❌ __getitem__ returned None (likely due to skipped structure or processing error).")
    else:
        sample_input, sample_target = result
        print("✅ __getitem__ succeeded.")
        print("Input shapes:", [x.shape if isinstance(x, torch.Tensor) else type(x) for x in sample_input])
        print("Target shape:", sample_target.shape)
except Exception as e:
    print(f"❌ __getitem__ failed: {e}")
