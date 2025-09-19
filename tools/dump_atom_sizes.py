#!/usr/bin/env python3
import os
import argparse
import csv
from collections import Counter

import numpy as np

# --- ScanNet bits ---
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.datasets import ScanNetDataset
from scannet_pytorch.utilities.io_utils import load_pickle


def count_atoms(coords):
    """
    Return a robust atom count for various shapes:
      - np.ndarray with shape (N, 3)
      - list/tuple of arrays
      - fallback to len()
    """
    try:
        if isinstance(coords, np.ndarray):
            if coords.ndim == 2:
                return int(coords.shape[0])
            if coords.ndim == 1:
                # rare degenerate case: try to reshape
                return int(len(coords) // 3)
        if isinstance(coords, (list, tuple)):
            total = 0
            for x in coords:
                if isinstance(x, np.ndarray):
                    if x.ndim == 2:
                        total += int(x.shape[0])
                    elif x.ndim == 1:
                        total += int(len(x) // 3)
                else:
                    try:
                        total += int(len(x))
                    except Exception:
                        pass
            if total > 0:
                return total
        # last resort
        return int(len(coords))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description="Dump atom counts for a list of IDs.")
    ap.add_argument("ids_path", type=str, help="Path to a text file of IDs (one per line).")
    ap.add_argument("--data_root", type=str, default="data",
                    help="Root folder that contains train/val/test .atom_enriched and BioLip_*.pkl")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="train",
                    help="Which split to read from (controls labels pkl and *.atom_enriched path).")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Optional CSV to write: id,atoms")
    ap.add_argument("--threshold", type=int, default=None,
                    help="If set, also write <ids_path> filtered to IDs with atoms <= threshold.")
    args = ap.parse_args()

    # --- read IDs ---
    with open(args.ids_path) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if not ids:
        raise RuntimeError(f"❌ No IDs found in {args.ids_path}")

    # --- label dict (match your train code’s precedence) ---
    pkl_name = {
        "train": ["BioLip_final_train.pkl", "BioLip_train.pkl"],
        "val":   ["BioLip_final_val.pkl",   "BioLip_val.pkl"],
        "test":  ["BioLip_final_test.pkl",  "BioLip_test.pkl"],
    }[args.split]

    label_pkl = None
    for nm in pkl_name:
        cand = os.path.join(args.data_root, nm)
        if os.path.exists(cand):
            label_pkl = cand
            break
    if label_pkl is None:
        raise FileNotFoundError(f"❌ Could not find any of {pkl_name} under {args.data_root}")

    label_dict = load_pickle(label_pkl)

    # --- pipeline like training ---
    pipeline = ScanNetPipeline(
        atom_features="valency",
        aa_features="sequence",
        aa_frames="triplet_sidechain",
        Beff=500,
    )

    pdb_folder = os.path.join(args.data_root, f"{args.split}.atom_enriched")
    if not os.path.isdir(pdb_folder):
        raise FileNotFoundError(f"❌ {pdb_folder} not found (expected materialized *.atom_enriched folder)")

    # --- dataset pointed at atom_enriched with only these IDs ---
    ds = ScanNetDataset(
        pdb_ids=ids,
        label_dict=label_dict,
        pipeline=pipeline,
        pdb_folder=pdb_folder,
        max_length=800,
    )

    counts = []
    kept = []
    skipped_examples = 0

    for i in range(len(ds)):
        try:
            sample = ds[i]
            if sample is None:
                skipped_examples += 1
                continue
            coords = sample[0]
            n_atoms = count_atoms(coords)
            counts.append((ds.ids[i] if hasattr(ds, "ids") else ids[i], n_atoms))
        except Exception as e:
            skipped_examples += 1
            # Keep going; this mirrors your loader’s tolerant behavior.

    # --- basic stats ---
    only_counts = [c for _, c in counts]
    if not only_counts:
        print("❌ No usable samples to report.")
        return

    total = len(only_counts)
    print(f"✅ usable samples: {total} | skipped during load: {skipped_examples}")
    for thr in [3000, 4000, 5000, 8000, 10000, 20000, 30000]:
        pct = 100.0 * sum(c > thr for c in only_counts) / total
        print(f"  > atoms > {thr:>5}: {pct:6.2f}%  ({sum(c > thr for c in only_counts)}/{total})")

    q = [5, 10, 25, 50, 75, 90, 95, 99]
    qs = np.percentile(only_counts, q).astype(int)
    print("  quantiles:")
    for qq, val in zip(q, qs):
        print(f"    {qq:>2}th: {val}")

    print(f"  min={min(only_counts)}, max={max(only_counts)}, mean={np.mean(only_counts):.1f}, median={np.median(only_counts):.1f}")

    # --- write CSV if requested ---
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "atoms"])
            w.writerows(counts)
        print(f"📝 wrote CSV → {args.out_csv}")

    # --- optionally emit filtered ID list ---
    if args.threshold is not None:
        out_path = args.ids_path.rsplit(".", 1)[0] + f"_le_{args.threshold}.txt"
        with open(out_path, "w") as f:
            for cid, n in counts:
                if n <= args.threshold:
                    f.write(f"{cid}\n")
        kept_n = sum(1 for _, n in counts if n <= args.threshold)
        print(f"🧹 wrote filtered IDs ≤ {args.threshold} atoms → {out_path}  (kept {kept_n}/{total})")


if __name__ == "__main__":
    main()
