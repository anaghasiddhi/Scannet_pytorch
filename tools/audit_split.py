#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True)
ap.add_argument("--dir", required=True)
args = ap.parse_args()

ids = [x.strip() for x in open(args.ids) if x.strip()]
root = Path(args.dir)

missing = []
bad_load = []
for idx, i in enumerate(ids, 1):
    p = root / f"{i}.npy"
    if not p.exists():
        missing.append(i)
        continue
    try:
        _ = np.load(p, allow_pickle=True)  # just try loading
    except Exception as e:
        bad_load.append((i, str(e)))

    if idx % 5000 == 0:
        print(f"...checked {idx}/{len(ids)}")

print(f"\nChecked {len(ids)} IDs in {root}")
print(f"missing={len(missing)}, bad_load={len(bad_load)}")
if missing:
    print(" first missing:", missing[:10])
if bad_load:
    print(" first bad_load:", bad_load[:10])
