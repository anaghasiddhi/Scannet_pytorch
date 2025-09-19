import os, pickle, sys

DATA = "data"
splits = {
    "train": "final_train_ids.txt",
    "val":   "final_val_ids.txt",
    "test":  "final_test_ids.txt",
}

# unified label source
src_all = os.path.join(DATA, "BioLip_all_labels.pkl")
if os.path.exists(src_all):
    unified = pickle.load(open(src_all,"rb"))
    print(f"[labels] using {src_all} with {len(unified)} entries")
else:
    merged = {}
    for sp in ("train","val","test"):
        p = os.path.join(DATA, f"BioLip_{sp}.pkl")
        if os.path.exists(p):
            d = pickle.load(open(p,"rb"))
            merged.update(d)
    if not merged:
        sys.exit("❌ No unified label source found")
    unified = merged
    print(f"[labels] merged {len(unified)} entries from split PKLs")

# intersect with each final split
for sp, fname in splits.items():
    ids = [ln.strip() for ln in open(os.path.join(DATA, fname))]
    kept, miss = {}, []
    for pid in ids:
        val = unified.get(pid)
        if val is not None:
            kept[pid] = val
        else:
            miss.append(pid)
    out_pkl = os.path.join(DATA, f"BioLip_final_{sp}.pkl")
    pickle.dump(kept, open(out_pkl, "wb"))
    print(f"[{sp}] ids: {len(ids)} → kept {len(kept)}, missing {len(miss)}")
    if miss:
        print(f"  e.g. missing: {miss[:10]}")
