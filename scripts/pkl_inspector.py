# audit_labels.py
import pickle, os, numpy as np, sys, glob

SPLIT="train"  # change to val/test as needed
ROOT="/home/scratch1/asiddhi/benchmarking_model2/ScanNet"  # adjust if different
LBL = os.path.join(ROOT, "data", f"{SPLIT}_label_dict.pkl")
ALT = os.path.join(ROOT, "data", f"BioLip_{SPLIT}.pkl")
DATA_DIR = os.path.join(ROOT, "data", f"{SPLIT}.data")

def load_pkl(p):
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

label_dict = load_pkl(LBL) or load_pkl(ALT)
assert label_dict is not None, f"Neither {LBL} nor {ALT} found."

print(f"Loaded labels: {len(label_dict)} entries from",
      (LBL if os.path.exists(LBL) else ALT))

# label schema probe
flat, nested = 0, 0
nonempty, empty = 0, 0
len_dist = []
pos_count = []
bad_types = 0

def flatten(d):
    if isinstance(d, dict):
        for k,v in d.items():
            if isinstance(v, dict):
                for ck, cv in v.items():
                    yield f"{k}_{ck}", cv
            else:
                yield k, v
    else:
        raise TypeError

for key, val in flatten(label_dict):
    if isinstance(val, np.ndarray):
        L = len(val)
        P = int(val.sum()) if val.dtype != bool else int(val.sum())
    elif isinstance(val, (list, tuple)):
        L = len(val)
        P = int(np.sum(val))
    else:
        bad_types += 1
        continue
    if L == 0:
        empty += 1
    else:
        nonempty += 1
        len_dist.append(L)
        pos_count.append(P)

print(f"non-empty: {nonempty} | empty: {empty} | bad-types: {bad_types}")
if nonempty:
    import statistics as st
    print(f"len(labels): min={min(len_dist)} med={int(st.median(len_dist))} "
          f"mean={int(st.mean(len_dist))} max={max(len_dist)}")
    print(f"pos per seq: min={min(pos_count)} med={int(st.median(pos_count))} "
          f"mean={round(sum(pos_count)/len(pos_count),2)} max={max(pos_count)}")

# feature existence check (very common failure)
# assume features saved per key as {DATA_DIR}/{key}.npy (adjust if different)
missing_feat = []
mismatch_case = []
sampled = 0

for key, val in list(flatten(label_dict))[:5000]:  # sample check
    key_path = os.path.join(DATA_DIR, f"{key}.npy")
    if not os.path.exists(key_path):
        # retry common variations
        alt1 = os.path.join(DATA_DIR, f"{key.upper()}.npy")
        alt2 = os.path.join(DATA_DIR, f"{key.lower()}.npy")
        if os.path.exists(alt1) or os.path.exists(alt2):
            mismatch_case.append(key)
        else:
            missing_feat.append(key)
    sampled += 1

print(f"[FEATURE CHECK] sampled={sampled}, missing={len(missing_feat)}, "
      f"case-mismatch={len(mismatch_case)}")
if missing_feat[:10]:
    print("Examples missing:", missing_feat[:10])
if mismatch_case[:10]:
    print("Case-mismatch examples:", mismatch_case[:10])
