#!/usr/bin/env python3
import os, pickle, random

DATA = "data"

# --- Load unified labels ---
labels = {}
for sp in ("train","val","test"):
    p = os.path.join(DATA, f"BioLip_{sp}.pkl")
    if os.path.exists(p):
        d = pickle.load(open(p,"rb"))
        labels.update(d)
print(f"[labels] unified: {len(labels)} entries")

# --- Load all .npy IDs actually present ---
with open(os.path.join(DATA, "all_ids_clean.txt")) as f:
    npy_ids = [ln.strip() for ln in f if ln.strip()]
print(f"[features] npy files: {len(npy_ids)}")

# --- Intersection: aligned pool ---
aligned_ids = [pid for pid in npy_ids if pid in labels]
print(f"[aligned] usable pairs: {len(aligned_ids)}")

# --- Shuffle + split 70/15/15 ---
random.seed(42)
random.shuffle(aligned_ids)

n = len(aligned_ids)
a = int(0.70 * n)
b = int(0.85 * n)

splits = {
    "final_train_ids.txt": aligned_ids[:a],
    "final_val_ids.txt":   aligned_ids[a:b],
    "final_test_ids.txt":  aligned_ids[b:]
}

for name, ids in splits.items():
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"{name}: {len(ids)}")

# --- Write split-specific label PKLs ---
for sp, fname in [("train","final_train_ids.txt"),
                  ("val","final_val_ids.txt"),
                  ("test","final_test_ids.txt")]:
    ids = [ln.strip() for ln in open(os.path.join(DATA, fname))]
    kept = {pid: labels[pid] for pid in ids if pid in labels}
    out_pkl = os.path.join(DATA, f"BioLip_final_{sp}.pkl")
    pickle.dump(kept, open(out_pkl, "wb"))
    print(f"{sp}: wrote {len(kept)} labels → {out_pkl}")
