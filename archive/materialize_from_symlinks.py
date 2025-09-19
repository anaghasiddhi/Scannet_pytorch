#!/usr/bin/env python3
import os, shutil, numpy as np

DATA = "data"
SRC_DIRS = {
    "train": os.path.join(DATA, "train.final.data"),
    "val":   os.path.join(DATA, "val.final.data"),
    "test":  os.path.join(DATA, "test.final.data"),
}
DST_DIRS = {
    "train": os.path.join(DATA, "train.solid.data"),
    "val":   os.path.join(DATA, "val.solid.data"),
    "test":  os.path.join(DATA, "test.solid.data"),
}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def ok_dim20(npy_path):
    try:
        d = np.load(npy_path, allow_pickle=True).item()
        a = d.get("attr_aa", None)
        return a is not None and getattr(a, "ndim", 0) >= 2 and a.shape[1] == 20
    except Exception:
        return False

def main():
    for split, src in SRC_DIRS.items():
        dst = DST_DIRS[split]
        ensure_dir(dst)
        if not os.path.isdir(src):
            print(f"[{split}] ⚠️ source dir missing: {src}"); continue

        names = [f for f in os.listdir(src) if f.endswith(".npy")]
        copied = missing = bad = 0
        for i, name in enumerate(names, 1):
            link_path = os.path.join(src, name)
            # resolve symlink to real path
            target = os.path.realpath(link_path)
            if not os.path.isfile(target):
                missing += 1
                continue
            # optional: enforce attr_aa dim=20 (skip weird files)
            if not ok_dim20(target):
                bad += 1
                continue
            dst_path = os.path.join(dst, name)
            if not os.path.exists(dst_path):
                shutil.copy2(target, dst_path)
                copied += 1
            if i % 10000 == 0:
                print(f"[{split}] {i}/{len(names)} processed … copied={copied}, bad={bad}, missing={missing}")

        print(f"[{split}] done: copied={copied}, bad={bad}, missing={missing} → {dst}")

if __name__ == "__main__":
    main()
