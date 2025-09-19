#!/usr/bin/env python3
import os, sys

DATA = "data"
SRC_DIRS = [os.path.join(DATA, "train.data"),
            os.path.join(DATA, "val.data"),
            os.path.join(DATA, "test.data")]

SPLITS = {
    "train": os.path.join(DATA, "final_train_ids.txt"),
    "val":   os.path.join(DATA, "final_val_ids.txt"),
    "test":  os.path.join(DATA, "final_test_ids.txt"),
}

TARGET_DIRS = {
    "train": os.path.join(DATA, "train.final.data"),
    "val":   os.path.join(DATA, "val.final.data"),
    "test":  os.path.join(DATA, "test.final.data"),
}

def find_src_path(pid):
    fn = f"{pid}.npy"
    for d in SRC_DIRS:
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            return p
    return None

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def rel_symlink(src, dst):
    # Remove if existing wrong link/file
    if os.path.lexists(dst):
        if os.path.islink(dst) and os.readlink(dst) == src:
            return
        try:
            os.remove(dst)
        except OSError:
            pass
    # Make link with relative path for portability
    rel = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel, dst)

def main():
    for split, idfile in SPLITS.items():
        tdir = TARGET_DIRS[split]
        ensure_dir(tdir)
        with open(idfile) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        have = miss = 0
        for i, pid in enumerate(ids, 1):
            src = find_src_path(pid)
            if src is None:
                miss += 1
                continue
            dst = os.path.join(tdir, f"{pid}.npy")
            rel_symlink(src, dst)
            have += 1
            if i % 10000 == 0:
                print(f"[{split}] {i}/{len(ids)} processed … (linked={have}, missing={miss})")
        print(f"[{split}] done: linked={have}, missing={miss} → {tdir}")

if __name__ == "__main__":
    main()
