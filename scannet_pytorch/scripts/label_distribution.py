#!/usr/bin/env python3
import os, argparse, pickle, numpy as np

PREFERRED = ["BioLip_final_{split}.pkl", "BioLip_{split}.pkl", "{split}_labels.pkl"]

def pick_label_file(data_dir, split):
    for pat in PREFERRED:
        cand = os.path.join(data_dir, pat.format(split=split))
        if os.path.exists(cand):
            return cand
    return None

def load_labels_dict(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    # d: { "pdb_chain": np.ndarray([...0/1...]) }
    return d

def summarize_split(data_dir, split):
    fpath = pick_label_file(data_dir, split)
    if not fpath:
        return {"split": split, "found": False, "message": f"No label file found for {split} in {data_dir}"}

    d = load_labels_dict(fpath)
    arrays = [np.asarray(v, dtype=np.float32).ravel() for v in d.values() if v is not None]
    if not arrays:
        return {"split": split, "found": True, "file": fpath, "n_residues": 0, "n_chains": 0,
                "pos_rate": 0.0, "PR_AUC_random": 0.0, "AUROC_random": 0.5,
                "F1_allpos": 0.0, "Precision_allpos": 0.0, "Recall_allpos": 1.0}

    y = np.concatenate(arrays)
    n = int(y.size)
    pos = float((y == 1).sum())
    pi = pos / n if n > 0 else 0.0

    # Baselines that depend only on label distribution
    pr_auc_random = pi          # expected PR-AUC for random scores
    auroc_random  = 0.5
    precision_allpos = pi       # if you predict all 1s
    recall_allpos = 1.0
    f1_allpos = (2 * precision_allpos * recall_allpos /
                 (precision_allpos + recall_allpos)) if (precision_allpos + recall_allpos) > 0 else 0.0

    return {
        "split": split,
        "found": True,
        "file": fpath,
        "n_chains": len(d),
        "n_residues": n,
        "positives": int(pos),
        "pos_rate": pi,
        "PR_AUC_random": pr_auc_random,
        "AUROC_random": auroc_random,
        "F1_allpos": f1_allpos,
        "Precision_allpos": precision_allpos,
        "Recall_allpos": recall_allpos
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data",
                    help="Folder containing BioLiP label PKLs (default: ./data)")
    ap.add_argument("--splits", type=str, default="train,val,test",
                    help="Comma-separated splits to summarize")
    args = ap.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        stats = summarize_split(args.data_dir, split)
        if not stats.get("found", False):
            print(f"[{split}] ❌ {stats['message']}")
            continue
        print(f"\n[{split}] ✅ {stats['file']}")
        print(f"- chains:     {stats['n_chains']}")
        print(f"- residues:   {stats['n_residues']:,}")
        print(f"- positives:  {stats['positives']:,}  (pos_rate = {stats['pos_rate']:.4%})")
        print(f"- PR-AUC (random): {stats['PR_AUC_random']:.6f}")
        print(f"- AUROC (random):  {stats['AUROC_random']:.3f}")
        print(f"- All-positives:   Precision={stats['Precision_allpos']:.4f}, "
              f"Recall={stats['Recall_allpos']:.4f}, F1={stats['F1_allpos']:.4f}")

if __name__ == "__main__":
    main()
