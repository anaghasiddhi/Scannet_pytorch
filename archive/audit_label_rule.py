#!/usr/bin/env python3
"""
audit_label_rule_fast.py

Fast audit of your per-chain labels (.npy) vs. ScanNet paper's rule:
A residue is positive if ANY heavy atom is within CUTOFF Å of ANY heavy atom
from ANY OTHER chain in the same model.

Uses scipy.spatial.cKDTree for ~100–1000× speedup vs brute force.
Shows per-chain progress and a short table + summary at the end.

Example:
  python audit_label_rule_fast.py \
    --labels-dir data/test.atom_enriched \
    --ids-file   data/test.atom_enriched/processed_ids.txt \
    --struct-dir datasets/BioLip/structures \
    --sample 30 --cutoff 4.0 --progress
"""
import argparse, sys, os, random
from pathlib import Path
import numpy as np
from typing import Optional

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--ids-file",   required=True, type=Path)
    ap.add_argument("--struct-dir", required=True, type=Path)
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path, default=None, help="Optional: write per-chain results to CSV")
    ap.add_argument("--progress", action="store_true", help="Print progress per chain")
    return ap.parse_args()

# ---------- load labels ----------
def load_labels_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        v = arr
    else:
        try:
            d = arr.item() if hasattr(arr, "item") else {}
        except Exception:
            d = {}
        v = None
        for k in ("labels","y","target","targets","res_labels","binding_labels","y_true","label"):
            if k in d:
                v = d[k]; break
        if v is None and isinstance(d, dict):
            for val in d.values():
                try:
                    vv = np.asarray(val).squeeze()
                    if vv.ndim == 1 and vv.size > 0:
                        v = vv; break
                except Exception:
                    pass
        if v is None:
            raise ValueError(f"Couldn't find labels in {path.name}")
    v = np.asarray(v).squeeze()
    if v.ndim > 1: v = v.reshape(-1)
    if not set(np.unique(v)).issubset({0,1,0.0,1.0}):
        v = (v > 0.5).astype(np.int8)
    return v.astype(np.int8)

# ---------- KD-tree paper-rule ----------
def get_chain_from_model(model, chain_id: str):
    if chain_id in model: return model[chain_id]
    for cid in (chain_id.upper(), chain_id.lower()):
        if cid in model: return model[cid]
    for ch in model:
        if getattr(ch, "id", None) == chain_id: return ch
    return None

def paper_rule_labels_fast(cif_path: Path, chain_id: str, cutoff: float) -> Optional[np.ndarray]:
    try:
        from Bio.PDB import MMCIFParser, is_aa
        from scipy.spatial import cKDTree
    except Exception as e:
        print("Need biopython + scipy. Try: pip install biopython scipy", file=sys.stderr)
        raise

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("x", str(cif_path))
    try:
        model = next(iter(structure))  # first model
    except StopIteration:
        return None

    target_chain = get_chain_from_model(model, chain_id)
    if target_chain is None:
        return None

    target_residues = [r for r in target_chain if is_aa(r)]
    if not target_residues:
        return None

    # Collect heavy atom coords for "other" chains
    other_coords = []
    for ch in model:
        if ch.id == target_chain.id: continue
        for r in ch:
            if not is_aa(r): continue
            for a in r.get_atoms():
                name = str(a.get_name()).upper()
                if getattr(a, "element", None) != "H" and not name.startswith("H"):
                    other_coords.append(a.coord)
    if not other_coords:
        # No partners in the asymmetric unit; many PPIs are across symmetry mates.
        # In that case, paper-rule positives will be zero here.
        return np.zeros(len(target_residues), dtype=np.int8)

    other_coords = np.asarray(other_coords, dtype=np.float32)
    tree = cKDTree(other_coords, balanced_tree=True, compact_nodes=True)

    # Build array of heavy-atom coords per residue (and an index to slice)
    res_offsets = [0]
    targ_coords = []
    for r in target_residues:
        for a in r.get_atoms():
            name = str(a.get_name()).upper()
            if getattr(a, "element", None) != "H" and not name.startswith("H"):
                targ_coords.append(a.coord)
        res_offsets.append(len(targ_coords))
    targ_coords = np.asarray(targ_coords, dtype=np.float32)

    out = np.zeros(len(target_residues), dtype=np.int8)
    if targ_coords.size == 0:
        return out

    # For speed, query neighbors for all target atoms in one shot
    # neighbors is a list of lists; any hit within a residue => positive
    neigh = tree.query_ball_point(targ_coords, r=cutoff)
    for i in range(len(target_residues)):
        s, e = res_offsets[i], res_offsets[i+1]
        if e > s:
            # did any atom in this residue have neighbors?
            hit = any(len(neigh[j]) > 0 for j in range(s, e))
            out[i] = 1 if hit else 0
    return out

# ---------- main ----------
def main():
    args = parse_args()
    if not args.labels_dir.exists(): sys.exit(f"[err] labels dir not found: {args.labels_dir}")
    if not args.ids_file.exists():   sys.exit(f"[err] ids file not found: {args.ids_file}")
    if not args.struct_dir.exists(): sys.exit(f"[err] struct dir not found: {args.struct_dir}")

    ids = [ln.strip() for ln in args.ids_file.read_text().splitlines() if ln.strip()]
    if not ids: sys.exit("[err] ids file is empty")

    random.seed(args.seed)
    sample = random.sample(ids, min(args.sample, len(ids)))

    rows = []
    for idx, key in enumerate(sample, 1):
        pdb, chain = key.split("_", 1) if "_" in key else (key[:4], key[5:])
        npy = args.labels_dir / f"{key}.npy"
        cif = args.struct_dir / f"{pdb}.cif"

        if not npy.exists():
            if args.progress: print(f"[{idx}/{len(sample)}] skip (missing labels): {npy}")
            continue
        if not cif.exists():
            # try .pdb as a fallback
            pdb_path = args.struct_dir / f"{pdb}.pdb"
            if pdb_path.exists():
                cif = pdb_path
            else:
                if args.progress: print(f"[{idx}/{len(sample)}] skip (missing structure): {cif}")
                continue

        try:
            y_yours = load_labels_npy(npy)
        except Exception as e:
            if args.progress: print(f"[{idx}/{len(sample)}] skip {npy.name}: {e}")
            continue

        try:
            y_paper = paper_rule_labels_fast(cif, chain, cutoff=args.cutoff)
        except Exception as e:
            if args.progress: print(f"[{idx}/{len(sample)}] paper-rule failed for {key}: {e}")
            continue
        if y_paper is None:
            if args.progress: print(f"[{idx}/{len(sample)}] skip (no target chain): {key}")
            continue

        n = min(len(y_yours), len(y_paper))
        if n == 0:
            if args.progress: print(f"[{idx}/{len(sample)}] skip (zero-length): {key}")
            continue
        y1, y2 = y_yours[:n].astype(np.int8), y_paper[:n].astype(np.int8)

        inter = np.logical_and(y1==1, y2==1).sum()
        union = np.logical_or(y1==1, y2==1).sum()
        jacc = (inter/union) if union else 1.0

        rows.append({
            "key": key,
            "yours_p": float(y1.mean()),
            "paper_p": float(y2.mean()),
            "jaccard": float(jacc),
            "n_res": int(n),
        })

        if args.progress:
            print(f"[{idx}/{len(sample)}] {key}  yours_p={y1.mean():.4f}  paper_p={y2.mean():.4f}  jacc={jacc:.3f}")

    if not rows:
        sys.exit("No comparable chains processed. Double-check paths and file formats (.cif/.pdb).")

    rows.sort(key=lambda r: r["jaccard"])

    # print a small table
    print("\nkey\tyours_p\tpaper_p\tjaccard\tn")
    for r in rows[:10]:
        print(f"{r['key']}\t{r['yours_p']:.4f}\t{r['paper_p']:.4f}\t{r['jaccard']:.3f}\t{r['n_res']}")

    avg_yours = float(np.mean([r["yours_p"] for r in rows]))
    avg_paper = float(np.mean([r["paper_p"] for r in rows]))
    avg_jacc  = float(np.mean([r["jaccard"] for r in rows]))

    print("\nSummary on sample:")
    print(f"- mean(yours_p) = {avg_yours:.4f}")
    print(f"- mean(paper_p) = {avg_paper:.4f}")
    print(f"- mean(jaccard) = {avg_jacc:.3f}")
    print(f"- sample size   = {len(rows)} chains")

    # optional CSV
    from csv import DictWriter
    csv_path = args.csv
    if csv_path:
        with csv_path.open("w", newline="") as f:
            w = DictWriter(f, fieldnames=["key","yours_p","paper_p","jaccard","n_res"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote CSV: {csv_path}")

if __name__ == "__main__":
    main()
