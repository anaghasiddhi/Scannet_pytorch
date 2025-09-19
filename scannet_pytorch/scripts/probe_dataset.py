import os, glob, numpy as np

pool = "data/train.data"
bad = []

files = glob.glob(f"{pool}/*.npy")
print(f"🔎 Probing {len(files)} files in {pool} ...")

for i, p in enumerate(files, 1):
    try:
        d = np.load(p, allow_pickle=True).item()
        if "attr_aa" not in d:
            bad.append((os.path.basename(p), "missing_attr"))
            continue

        L = len(d["attr_aa"])
        for k in ("coord_aa", "triplets_aa", "indices_aa"):
            if k not in d: 
                continue
            if len(d[k]) != L:
                bad.append((os.path.basename(p), k, len(d[k]), L))
                break
    except Exception as e:
        bad.append((os.path.basename(p), "load_error", str(e)))

    if i % 10000 == 0:
        print(f"  ... {i}/{len(files)} checked")

print(f"\n✅ Probe complete. total={len(files)} files")
if not bad:
    print("🎉 all files aligned cleanly")
else:
    print(f"❌ {len(bad)} misaligned or errored files")
    with open("data/bad_ids.txt", "w") as f:
        for b in bad:
            f.write("\t".join(map(str, b)) + "\n")
    print("➡️ written details to data/bad_ids.txt (id, key, bad_len, attr_len)")
