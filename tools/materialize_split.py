#!/usr/bin/env python3
import argparse, os, shutil, sys, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_ids(p):
    with open(p) as f:
        return [line.strip() for line in f if line.strip()]

def find_candidate(id_, roots):
    # Look for file in each root. Accept either direct file or symlink that resolves.
    for r in roots:
        p = Path(r) / f"{id_}.npy"
        if p.exists():
            try:
                return p.resolve(strict=True)
            except Exception:
                # broken symlink inside root; skip
                pass
        # also allow nested “*.data/*/id.npy” cases
        for hit in Path(r).rglob(f"{id_}.npy"):
            try:
                return hit.resolve(strict=True)
            except Exception:
                continue
    return None

def copy_one(id_, roots, dst_dir, check_dim, overwrite):
    dst = Path(dst_dir) / f"{id_}.npy"
    if dst.exists() and not overwrite:
        return ("skipped", id_, None)
    src = find_candidate(id_, roots)
    if src is None:
        return ("missing", id_, None)
    # optional feature-dim check
    try:
        arr = np.load(src, allow_pickle=True)
        if check_dim is not None:
            if arr.ndim == 1 and arr.shape[-1] != check_dim:
                return ("bad_dim", id_, arr.shape)
            if arr.ndim > 1 and arr.shape[-1] != check_dim:
                return ("bad_dim", id_, arr.shape)
    except Exception as e:
        return ("bad_load", id_, str(e))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return ("copied", id_, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="final_{split}_ids.txt")
    ap.add_argument("--source-root", action="append", required=True,
                    help="Root(s) containing the true .npy pool(s). Repeatable.")
    ap.add_argument("--dst", required=True, help="data/{split}.solid.data")
    ap.add_argument("--check-dim", type=int, default=20)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ids = read_ids(args.ids)
    roots = [Path(r) for r in args.source_root]
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    stats = {"copied":0, "skipped":0, "missing":0, "bad_dim":0, "bad_load":0}
    examples = {"missing":[], "bad_dim":[], "bad_load":[]}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(copy_one, i, roots, dst, args.check_dim, args.overwrite): i for i in ids}
        for fut in as_completed(futs):
            kind, i, info = fut.result()
            stats[kind] += 1
            if kind in examples and len(examples[kind]) < 10:
                examples[kind].append((i, info))

    total = len(ids)
    print(f"\nDone → {dst}")
    for k,v in stats.items():
        print(f"{k:>8}: {v}")
    print(f"  total: {total}")
    if stats["missing"]:
        print("\nFirst missing examples:")
        for x in examples["missing"]: print("  ", x[0])
    if stats["bad_dim"]:
        print("\nFirst bad_dim examples (id, shape):")
        for x in examples["bad_dim"]: print("  ", x)
    if stats["bad_load"]:
        print("\nFirst bad_load examples (id, err):")
        for x in examples["bad_load"]: print("  ", x)

if __name__ == "__main__":
    sys.exit(main())
