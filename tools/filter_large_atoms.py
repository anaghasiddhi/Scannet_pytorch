# tools/filter_large_atoms.py
import os
import sys
import json
import argparse
from typing import Iterable, Tuple, Optional

import numpy as np

# local imports
from scannet_pytorch.datasets import ScanNetDataset
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline

def safe_len_coords(coords) -> int:
    """
    Robustly count atoms from many possible coord formats:
    - np.ndarray of shape (L, 3) or (L, >=3)
    - list/tuple of np.ndarray segments
    - ragged object arrays
    - dicts containing 'coords' or similar
    """
    if coords is None:
        return 0

    # unwrap dict-like
    if isinstance(coords, dict):
        for k in ("coords", "coord", "atom_coordinate", "atoms", "positions"):
            if k in coords:
                return safe_len_coords(coords[k])
        # fall-through: try values
        total = 0
        for v in coords.values():
            total += safe_len_coords(v)
        return total

    # plain ndarray
    if isinstance(coords, np.ndarray):
        if coords.ndim == 2:
            # (L, 3) or (L, F)
            return int(coords.shape[0])
        elif coords.ndim == 1 and coords.dtype == object:
            # ragged object array (e.g., array of segments)
            total = 0
            for seg in coords:
                try:
                    total += safe_len_coords(seg)
                except Exception:
                    continue
            return total
        else:
            # unexpected shape; best effort
            return int(coords.shape[0]) if coords.size > 0 else 0

    # list / tuple of segments
    if isinstance(coords, (list, tuple)):
        total = 0
        for seg in coords:
            try:
                total += safe_len_coords(seg)
            except Exception:
                continue
        return total

    # anything else: can’t count
    return 0


def main():
    ap = argparse.ArgumentParser(description="Filter out very large-atom chains to avoid OOM")
    ap.add_argument("ids_in", help="Path to input IDs file (one chain/pdb id per line)")
    ap.add_argument("ids_out", help="Path to write filtered IDs")
    ap.add_argument("data_root", help="Root folder that contains train/val/test .atom_enriched dirs")
    ap.add_argument("max_atoms", type=int, help="Max atoms allowed (chains above this are skipped)")
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test"],
                    help="Which split’s .atom_enriched folder to use for loading")
    ap.add_argument("--max_len", type=int, default=800, help="Dataset max_length (same as training)")
    args = ap.parse_args()

    # Build a minimal pipeline consistent with training
    pipeline = ScanNetPipeline(
        atom_features="valency",
        aa_features="sequence",
        aa_frames="triplet_sidechain",
        Beff=500,
    )

    # Read ids
    with open(args.ids_in) as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    # The dataset only needs pdb_ids + pdb_folder; labels aren’t required for counting coords
    pdb_folder = os.path.join(args.data_root, f"{args.split}.atom_enriched")

    # Fake label dict with 0/1 per id just to satisfy constructor (values aren’t used)
    label_dict = {cid: [0] for cid in ids}

    ds = ScanNetDataset(
        pdb_ids=ids,
        label_dict=label_dict,
        pipeline=pipeline,
        pdb_folder=pdb_folder,
        max_length=args.max_len,
    )
    print(f"[ScanNetDataset] Initialized with {len(ds)} IDs. Cache dir={os.path.abspath(os.path.join(args.data_root, '..', 'npz_cache'))}")

    kept, skipped = [], []
    for i in range(len(ds)):
        try:
            sample = ds[i]
            if sample is None:
                # dataset may skip unusable examples
                continue
            # dataset returns (coords, feats, labels, ...) — first item is coords
            coords = sample[0]
            n_atoms = safe_len_coords(coords)
            if n_atoms <= args.max_atoms and n_atoms > 0:
                kept.append(ds.ids[i] if hasattr(ds, "ids") else ids[i])
            else:
                skipped.append((ds.ids[i] if hasattr(ds, "ids") else ids[i], n_atoms))
        except Exception as e:
            print(f"⚠️  skipping idx={i}: {e}")

    with open(args.ids_out, "w") as f:
        for cid in kept:
            f.write(cid + "\n")

    print(f"✅ wrote {len(kept)} IDs → {args.ids_out}")
    print(f"⏭️ skipped {len(skipped)} giants (> {args.max_atoms} atoms)")
    for cid, n in skipped[:10]:
        print(f"   {cid}: {n} atoms")


if __name__ == "__main__":
    main()
