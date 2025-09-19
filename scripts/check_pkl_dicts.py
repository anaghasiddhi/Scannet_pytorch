#!/usr/bin/env python3
"""
check_pkl_dicts.py

Sanity-checks *_label_dict.pkl files produced by your preprocessing run.

It will:
  1) Summarize each PKL (how many chains, how many empty labels, how many all-zero labels).
  2) Show top chains by number of positive residues, plus a sample of all-zero (non-empty) chains.
  3) (Optional) Cross-check label length against actual AA residue count in mmCIF for a sample.
  4) (Optional) Compare label lengths with the ground-truth labels_biolip.txt (via its .index).

Usage (typical):
  python check_pkl_dicts.py \
    --data-dir /home/scratch1/asiddhi/benchmarking_model2/ScanNet/scannet_pytorch/../data \
    --structures-dir /home/scratch1/asiddhi/benchmarking_model2/ScanNet/datasets/BioLip/structures \
    --labels-txt /home/scratch1/asiddhi/benchmarking_model2/ScanNet/labels_biolip.txt \
    --sample 1000

Tips:
- If Biopython is not installed in your env, install it or skip --structures-dir checks.
- If you only want summaries of PKLs, just pass --data-dir and nothing else.
"""

import argparse
import glob
import os
import pickle
import struct
import sys
from typing import Dict, Tuple, List

import numpy as np

# Biopython is optional (only needed for residue length checks)
try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    HAVE_BIOPYTHON = True
except Exception:
    HAVE_BIOPYTHON = False


def find_label_pkls(data_dir: str) -> List[str]:
    pattern = os.path.join(os.path.abspath(data_dir), "*_label_dict.pkl")
    return sorted(glob.glob(pattern))


def load_label_dict(pkl_path: str) -> Dict[str, np.ndarray]:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    # Normalize to dict[str, np.ndarray]
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            cleaned[k] = v
        else:
            try:
                cleaned[k] = np.asarray(v, dtype=np.float32)
            except Exception:
                cleaned[k] = np.array([], dtype=np.float32)
    return cleaned


def summarize_dict(name: str, d: Dict[str, np.ndarray]) -> Dict[str, float]:
    n = len(d)
    lens = np.fromiter((v.size for v in d.values()), dtype=np.int64, count=n)
    sums = np.fromiter(((int(v.sum()) if v.size > 0 else 0) for v in d.values()),
                       dtype=np.int64, count=n)

    length_eq_0 = int((lens == 0).sum())
    length_gt0_sum_eq_0 = int(((lens > 0) & (sums == 0)).sum())
    sum_gt_0 = int((sums > 0).sum())

    stats = {
        "chains": n,
        "length_eq_0": length_eq_0,
        "length_eq_0_pct": (100.0 * length_eq_0 / n) if n else 0.0,
        "length_gt0_sum_eq_0": length_gt0_sum_eq_0,
        "length_gt0_sum_eq_0_pct": (100.0 * length_gt0_sum_eq_0 / n) if n else 0.0,
        "sum_gt_0": sum_gt_0,
        "sum_gt_0_pct": (100.0 * sum_gt_0 / n) if n else 0.0,
        "median_len_nonempty": int(np.median(lens[lens > 0])) if (lens > 0).any() else 0,
        "p90_len_nonempty": int(np.percentile(lens[lens > 0], 90)) if (lens > 0).any() else 0,
    }

    print(f"\n=== {name} ===")
    print(f"chains: {stats['chains']}")
    print(f"length==0 : {stats['length_eq_0']} ({stats['length_eq_0_pct']:.2f}%)")
    print(f"length>0 & sum==0 : {stats['length_gt0_sum_eq_0']} ({stats['length_gt0_sum_eq_0_pct']:.2f}%)")
    print(f"sum>0 : {stats['sum_gt_0']} ({stats['sum_gt_0_pct']:.2f}%)")
    print(f"labels per chain (non-empty): median={stats['median_len_nonempty']} p90={stats['p90_len_nonempty']}")
    return stats


def show_examples(name: str, d: Dict[str, np.ndarray], top_k: int = 10) -> None:
    rows = []
    for k, v in d.items():
        rows.append((k, int(v.size), int(v.sum())))
    rows.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_k} chains by positives in {name}:")
    for r in rows[:top_k]:
        print(f"  {r[0]}  len={r[1]}  pos={r[2]}")

    print(f"\nSample up to {top_k} all-zero (but non-empty) chains in {name}:")
    c = 0
    for k, L, S in rows:
        if L > 0 and S == 0:
            print(f"  {k}  len={L}  pos=0")
            c += 1
            if c >= top_k:
                break


def count_valid_residues_mmcif(mmCIF_path: str, pdb_id: str, chain_id: str) -> int:
    """Return AA residue count with coordinates for a given chain, or negative on error."""
    if not HAVE_BIOPYTHON:
        return -9
    if not os.path.exists(mmCIF_path):
        return -1
    try:
        parser = MMCIFParser(QUIET=True)
        model = parser.get_structure(pdb_id, mmCIF_path)[0]
        chain = model[chain_id] if chain_id in model else None
        if chain is None:
            # loose fallback: try case-insensitive match
            for ch in model:
                if getattr(ch, "id", "").upper() == chain_id.upper():
                    chain = ch
                    break
        if chain is None:
            return -2
        count = 0
        for res in chain:
            if is_aa(res) and any(a.coord is not None for a in res.get_atoms()):
                count += 1
        return count
    except Exception:
        return -3


def residue_length_check_one_dict(name: str,
                                  d: Dict[str, np.ndarray],
                                  structures_dir: str,
                                  sample: int) -> None:
    """Sample some entries and compare labels length vs actual residue count."""
    if not HAVE_BIOPYTHON:
        print("\n[Residue check skipped] Biopython not available in this environment.")
        return

    items = list(d.items())
    if not items:
        print("\n[Residue check skipped] Empty dict.")
        return

    np.random.seed(0)
    np.random.shuffle(items)
    items = items[:min(sample, len(items))]

    mismatches = []
    empties = 0
    ok = 0

    for origin, labels in items:
        if not isinstance(labels, np.ndarray) or labels.size == 0:
            empties += 1
            continue
        try:
            pdb_id, chain_id = origin.split("_", 1)
        except ValueError:
            mismatches.append((origin, "bad_origin_format"))
            continue
        cif_path = os.path.join(structures_dir, f"{pdb_id.lower()}.cif")
        nres = count_valid_residues_mmcif(cif_path, pdb_id, chain_id)
        if nres > 0 and nres == labels.size:
            ok += 1
        else:
            mismatches.append((origin, nres, labels.size))

    print(f"\n[Residue length check] {name}")
    print(f"  sampled: {len(items)}  ok_match: {ok}  empty_arrays: {empties}  mismatches: {len(mismatches)}")
    if mismatches:
        print("  sample mismatches (up to 10):")
        for row in mismatches[:10]:
            print(f"    {row}")


def load_labels_index(index_path: str) -> List[Tuple[int, int]]:
    idx = []
    with open(index_path, "rb") as f:
        while True:
            b = f.read(12)
            if not b:
                break
            offset = struct.unpack('Q', b[:8])[0]
            length = struct.unpack('I', b[8:12])[0]
            idx.append((offset, length))
    return idx


def compare_with_ground_truth(our_dict: Dict[str, np.ndarray],
                              labels_txt_path: str,
                              sample: int = 1000) -> None:
    """Compare label lengths against labels_biolip.txt for a sample."""
    index_path = labels_txt_path + ".index"
    if not os.path.exists(labels_txt_path):
        print("\n[GT compare skipped] labels_biolip.txt not found.")
        return
    if not os.path.exists(index_path):
        print("\n[GT compare skipped] .index file not found (expected labels_biolip.txt.index).")
        return

    idx = load_labels_index(index_path)
    np.random.seed(1)
    if len(idx) > sample:
        sel = np.random.choice(len(idx), size=sample, replace=False)
        offsets = [idx[i] for i in sel]
    else:
        offsets = idx

    same_len = 0
    total = 0
    missing_in_our = 0

    with open(labels_txt_path, "rb") as lf:
        for off, ln in offsets:
            lf.seek(off)
            line = lf.read(ln).decode("utf-8", errors="replace").strip()
            parts = line.split()
            if not parts:
                continue
            origin = parts[0]
            gt = np.array([int(x) for x in parts[1:]], dtype=np.float32)
            if origin not in our_dict:
                missing_in_our += 1
                continue
            v = our_dict[origin]
            if isinstance(v, np.ndarray) and v.size > 0:
                total += 1
                if v.size == gt.size:
                    same_len += 1

    print("\n[Ground-truth length compare]")
    print(f"  sampled GT entries: {len(offsets)}")
    print(f"  present in our dict: {total}, same length: {same_len}")
    if total:
        print(f"  same length %: {100.0 * same_len / total:.2f}%")
    print(f"  missing in our dict: {missing_in_our}")


def main():
    parser = argparse.ArgumentParser(description="Sanity-check *_label_dict.pkl outputs.")
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing *_label_dict.pkl files (e.g., ../data).")
    parser.add_argument("--structures-dir", default=None,
                        help="Directory with BioLip mmCIF files to validate residue counts (optional).")
    parser.add_argument("--labels-txt", default=None,
                        help="Path to labels_biolip.txt for GT length comparison (optional).")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Sample size for residue/GT checks (default: 1000).")
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many examples to show for top positives and zero-label samples.")
    args = parser.parse_args()

    label_pkls = find_label_pkls(args.data_dir)
    if not label_pkls:
        print(f"No *_label_dict.pkl files found under {args.data_dir}")
        sys.exit(1)

    print("Found PKLs:")
    for p in label_pkls:
        print("  ", p)

    # Summaries + examples for each PKL
    dicts = []
    for p in label_pkls:
        d = load_label_dict(p)
        dicts.append((os.path.basename(p), d))
        summarize_dict(os.path.basename(p), d)
        show_examples(os.path.basename(p), d, top_k=args.top_k)

    # Residue length checks (optional)
    if args.structures_dir:
        if not HAVE_BIOPYTHON:
            print("\n[Skip] Biopython not available; cannot check mmCIF residue counts.")
        else:
            for name, d in dicts:
                residue_length_check_one_dict(name, d, args.structures_dir, sample=args.sample)

    # Ground-truth comparison (optional)
    if args.labels_txt:
        # Merge all dicts for convenience
        merged = {}
        for _, d in dicts:
            merged.update(d)
        compare_with_ground_truth(merged, args.labels_txt, sample=args.sample)

    print("\nDone.")


if __name__ == "__main__":
    main()
