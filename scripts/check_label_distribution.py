#!/usr/bin/env python3

import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scannet_pytorch.datasets import ScanNetDataset
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.utilities import io_utils, paths
skip_log_path = "skipped_examples.txt"
skip_log = open(skip_log_path, "w")


def check_label_distribution(split_name, split_txt_path, label_dict_path, pdb_folder):
    # === Load PDB IDs ===
    with open(split_txt_path) as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    print(f"[📂] Loaded {len(pdb_ids)} PDB-chain IDs from {split_txt_path}")

    # === Load Label Dictionary ===
    with open(label_dict_path, "rb") as f:
        label_dict = pickle.load(f)
    print(f"[📊] Loaded label dict with {len(label_dict)} entries from {label_dict_path}")

    # === Setup Pipeline and Dataset ===
    pipeline = ScanNetPipeline(
        atom_features="valency",
        aa_features="sequence",
        aa_frames="triplet_sidechain",
        Beff=500,
    )

    dataset = ScanNetDataset(
        pdb_ids=pdb_ids,
        label_dict=label_dict,
        pipeline=pipeline,
        pdb_folder=pdb_folder,
        max_length=None,
    )

    # === Loop through dataset and count 0s / 1s ===
    total_0, total_1, skipped = 0, 0, 0
    skipped_ids = []

    print(f"\n🔍 Checking label distribution for: {split_name}\n")

    for i in tqdm(range(len(dataset)), desc=f"{split_name}"):
        try:
            _, label = dataset[i]
            if label is None:
                raise ValueError("Label is None")
            label = label.flatten()
            total_0 += np.sum(label == 0)
            total_1 += np.sum(label == 1)
        except Exception as e:
            skipped += 1
            skipped_ids.append(dataset.pdb_ids[i])
            skip_log.write(f"Missing labels for {dataset.pdb_ids[i]}\n")
            skip_log.write(f"Skipped {dataset.pdb_ids[i]} due to error: {e}\n")

    # === Report ===
    print("\n=== Label Distribution Report ===")
    print(f"Split: {split_name}")
    print(f"Total usable examples: {len(dataset) - skipped}")
    print(f"Total skipped examples: {skipped}")
    print(f"Residue labels → 0s: {total_0} | 1s: {total_1}")
    if total_0 + total_1 > 0:
        ratio = total_1 / (total_0 + total_1)
        print(f"Class 1 ratio (binding): {ratio:.4f}")
    else:
        print("No valid labels found.")

    if skipped_ids:
        print("\n⚠️ Skipped PDB IDs:")
        for sid in skipped_ids:
            print(sid)


if __name__ == "__main__":
    # === Modify these for the split you want to check ===
    split_name = "train"
    split_txt_path = "datasets/PPBS/labels_train.txt"
    label_dict_path = "data/train_label_dict.pkl"
    pdb_folder = os.path.abspath("PDB")  # or set manually if needed

    check_label_distribution(split_name, split_txt_path, label_dict_path, pdb_folder)
    skip_log.close()
