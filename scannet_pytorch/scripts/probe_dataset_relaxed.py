import os, numpy as np

SPLIT_FILES = {
    "train": "data/final_train_ids.txt",
    "val":   "data/final_val_ids.txt",
    "test":  "data/final_test_ids.txt",
}

bad = []

def length(x):
    try:
        return int(x.shape[0])
    except Exception:
        return len(x)

for split, id_file in SPLIT_FILES.items():
    if not os.path.exists(id_file):
        print(f"⚠️  Missing {id_file}, skipping {split}")
        continue

    with open(id_file) as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    npy_paths = [os.path.join(f"data/{split}.data", f"{pid}.npy") for pid in ids]
    print(f"🔎 Probing (relaxed) {len(npy_paths)} files in {split} ...")

    for i, p in enumerate(npy_paths, 1):
        try:
            d = np.load(p, allow_pickle=True).item()
            if "attr_aa" not in d:
                bad.append((split, os.path.basename(p), "missing_attr"))
                continue

            L = length(d["attr_aa"])

            # per-residue tensors must match attr_aa
            for k in ("triplets_aa", "indices_aa"):
                if k in d and length(d[k]) != L:
                    bad.append((split, os.path.basename(p), f"{k}_len", length(d[k]), L))
                    break
            else:
                # per-atom coords must be >= per-residue
                if "coord_aa" in d and length(d["coord_aa"]) < L:
                    bad.append((split, os.path.basename(p), "coord_lt_attr", length(d["coord_aa"]), L))
        except Exception as e:
            bad.append((split, os.path.basename(p), "load_error", str(e)))

        if i % 10000 == 0:
            print(f"  ... {i}/{len(npy_paths)} checked in {split}")

print("\n✅ Probe (relaxed) complete.")
if not bad:
    print("🎉 all files pass relaxed checks")
else:
    print(f"❌ {len(bad)} files failed relaxed checks")
    out = "data/bad_ids_relaxed_final.txt"
    with open(out, "w") as f:
        for b in bad:
            f.write("\t".join(map(str, b)) + "\n")
    print(f"➡️ wrote details to {out} (split, id, reason, len, attr_len)")
