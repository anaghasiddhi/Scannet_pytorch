#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*Numba.*")
warnings.filterwarnings("ignore", message=".*reflected list.*")

"""
Preprocesses ScanNet datasets (PPBS) and saves:
- .data files for model input
- .pkl label dictionaries for mapping residue labels to structures

Run this script ONCE before training with train_pytorch.py
"""

import os
import pickle
import time
from scannet_pytorch.utilities import paths, io_utils
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline


# === Constants ===
label_dir = os.path.abspath(os.path.join(paths.library_folder, "..", "datasets", "PPBS"))
label_files = {
    "train": os.path.join(label_dir, "labels_train.txt"),
    "val_topology": os.path.join(label_dir, "labels_validation_topology.txt"),
    "val_none": os.path.join(label_dir, "labels_validation_none.txt"),
    "val_70": os.path.join(label_dir, "labels_validation_70.txt"),
    "val_homology": os.path.join(label_dir, "labels_validation_homology.txt"),
}


# === Utility ===
def ensure_2d(labels_raw):
    """Ensure label arrays are 2D (N, 1) for BCEWithLogits."""
    return [label[:, None] if label.ndim == 1 else label for label in labels_raw]


def preprocess_split(split_name, origins, resids, labels_raw, pipeline, fresh=True):
    """Preprocess and save a dataset split."""
    labels_raw = ensure_2d(labels_raw)
    print(f"→ Processing {split_name} | Examples: {len(origins)}")
    
    # Process & save .data file
    pipeline.build_processed_dataset(
        dataset_name=f"PPBS_{split_name}",
        list_origins=origins,
        list_resids=resids,
        list_labels=labels_raw,
        fresh=fresh,
        save=True
    )

    # Save label dict (.pkl)
    label_dict = dict(zip(origins, labels_raw))
    os.makedirs(paths.pipeline_folder, exist_ok=True)
    with open(os.path.join(paths.pipeline_folder, f"{split_name}_label_dict.pkl"), "wb") as f:
        pickle.dump(label_dict, f)

    print(f"✅ Saved {split_name} label dict → {split_name}_label_dict.pkl\n")


def main():
    start_time = time.time()

    print("📦 Initializing preprocessing pipeline...")
    pipeline = ScanNetPipeline(
        atom_features="valency",
        aa_features="sequence",
        aa_frames="triplet_sidechain",
        Beff=500
    )

    # === Train Split ===
    print("\n📦 Preprocessing training split...")
    train_origins, _, train_resids, train_labels_raw = io_utils.read_labels(label_files["train"])
    preprocess_split("train", train_origins, train_resids, train_labels_raw, pipeline, fresh=True)

    # === Validation Splits ===
    for split in ["val_topology", "val_none", "val_70", "val_homology"]:
        print(f"\n📦 Preprocessing validation split: {split}")
        val_origins, _, val_resids, val_labels_raw = io_utils.read_labels(label_files[split])
        preprocess_split(split, val_origins, val_resids, val_labels_raw, pipeline, fresh=True)

    print(f"\n🎉 Preprocessing complete. Total time: {round(time.time() - start_time, 2)} seconds")
    print(f"🧃 All .data and .pkl files saved under: {paths.pipeline_folder}")


if __name__ == "__main__":
    main()
