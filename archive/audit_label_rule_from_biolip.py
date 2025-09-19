#!/usr/bin/env python3
"""
audit_label_rule_from_biolip.py

Rebuild your labels from a BioLiP annotation file and per-chain feature .npy files
(containing `indices_aa`), then compare to PPBS-style "paper-rule" labels computed
on-the-fly: residue is positive if ANY heavy atom is within CUTOFF Å of ANY heavy
atom from ANY OTHER chain in the same model (asymmetric unit only).

Supports three BioLiP annotation formats:
  - rows : CSV with columns [pdb, chain, resi|resnum|residue_index]
  - list : CSV with columns [key=<pdb>_<chain>, res_list="..."]
  - colon: plain text lines "<pdb>_<...>_<chain>:<partner>:<tokens like t40 k43 ...>"

Example:
  python audit_label_rule_from_biolip.py \
    --labels-dir data/test.atom_enriched \
    --ids-file   data/test.atom_enriched/processed_ids.txt \
    --struct-dir datasets/BioLip/structures \
    --biolip-csv datasets/BioLip/BioLiP_annotation.csv \
    --format colon \
    --sample 10 --cutoff 4.0 --progress
"""
import argparse, csv, os, random, re, sys
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, Any, List
import numpy as np

# ------------------------- args -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Audit BioLiP-derived labels vs paper 4Å rule")
    ap.add_argument("--labels-dir", required=True, type=Path, help="Dir with <PDB>_<CHAIN>.npy containing indices_aa")
    ap.add_argument("--ids-file",   required=True, type=Path, help="Text file with one ID per line (e.g., 6ow3_B)")
    ap.add_argument("--struct-dir", required=True, type=Path, help="Dir with structures <pdb>.cif (or .pdb)")
    ap.add_argument("--biolip-csv", required=True, type=Path, help="BioLiP annotation file (rows/list/colon formats)")
    ap.add_argument("--cutoff", type=float, default=4.0, help="Distance cutoff Å for paper rule (default 4.0)")
    ap.add_argument("--sample", type=int, default=30, help="How many chains to sample")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--progress", action="store_true", help="Print progress per chain")
    ap.add_argument("--csv", type=Path, default=None, help="Optional: write per-chain results to CSV")
    ap.add_argument("--format", choices=["rows", "list", "colon"], default=None,
                    help="Force annotation format; use 'colon' for lines like '101m_1_a:...:t40 k43 ...'")
    return ap.parse_args()

# --------------------- util helpers ----------------------
def _digits_tail_to_int(tok: str) -> Optional[int]:
    m = re.search(r'(\d+)$', tok.strip())
    return int(m.group(1)) if m else None

def _split_pdb_chain_from_key(key: str) -> Optional[Tuple[str,str]]:
    """
    Accept keys like:
      '101m_1_a'  -> ('101m','a')
      '7vxu_L'    -> ('7vxu','L')
      '6ow3_B'    -> ('6ow3','B')
    Strategy: split by '_', take first as pdb, last as chain.
    """
    parts = key.strip().split("_")
    if len(parts) < 2: return None
    pdb = parts[0].lower()
    chain = parts[-1]
    if not pdb or not chain: return None
    return pdb, chain

# ----------------------- BioLiP read ----------------------
def read_biolip_rows(path: Path) -> Dict[Tuple[str,str], Set[int]]:
    pos: Dict[Tuple[str,str], Set[int]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = [c.lower().strip() for c in (reader.fieldnames or [])]
        if not cols or "pdb" not in cols or "chain" not in cols:
            raise ValueError("rows-format requires header with columns: pdb, chain, resi|resnum|residue_index")
        for r in reader:
            d = {k.lower().strip(): v for k,v in r.items()}
            pdb = (d.get("pdb") or "").lower().strip()
            ch  = (d.get("chain") or "").strip()
            res = d.get("resi") or d.get("resnum") or d.get("residue_index")
            if not (pdb and ch and res): continue
            try:
                i = int(str(res).strip())
            except Exception:
                continue
            pos.setdefault((pdb, ch), set()).add(i)
    return pos

def read_biolip_list(path: Path) -> Dict[Tuple[str,str], Set[int]]:
    pos: Dict[Tuple[str,str], Set[int]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = [c.lower().strip() for c in (reader.fieldnames or [])]
        if not cols or "key" not in cols:
            raise ValueError("list-format requires header with columns: key, res_list|resis|residue_list|residue_indices")
        for r in reader:
            d = {k.lower().strip(): v for k,v in r.items()}
            key = (d.get("key") or "").strip()
            if "_" not in key: continue
            split = _split_pdb_chain_from_key(key)
            if not split: continue
            pdb, ch = split
            lst = d.get("res_list") or d.get("resis") or d.get("residue_list") or d.get("residue_indices") or ""
            items = [x for tok in lst.replace(";", " ").replace(",", " ").split() for x in [tok.strip()] if x]
            vals = []
            for x in items:
                try: vals.append(int(x))
                except: pass
            pos.setdefault((pdb, ch), set()).update(vals)
    return pos

def read_biolip_colon(path: Path) -> Dict[Tuple[str,str], Set[int]]:
    """
    Each line: "<qkey>:<partner>:<tokens>"
      qkey    = something like '101m_1_a' or '7vxu_L'
      tokens  = space-separated items like 't40', 'k43', 'F139', ...
    We ignore the partner; we parse residue numbers from tokens (digits tail).
    """
    pos: Dict[Tuple[str,str], Set[int]] = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        parts = ln.split(":")
        if len(parts) < 3:  # not our expected line
            continue
        qkey = parts[0].strip()
        toks = parts[-1].strip().split()
        split = _split_pdb_chain_from_key(qkey)
        if not split: continue
        pdb, ch = split
        resi = []
        for t in toks:
            n = _digits_tail_to_int(t)
            if n is not None:
                resi.append(n)
        if resi:
            pos.setdefault((pdb, ch), set()).update(resi)
    return pos

def read_biolip_any(path: Path, forced_format: Optional[str]) -> Dict[Tuple[str,str], Set[int]]:
    if forced_format == "rows":
        return read_biolip_rows(path)
    if forced_format == "list":
        return read_biolip_list(path)
    if forced_format == "colon":
        return read_biolip_colon(path)
    # try to auto-detect (very light)
    first = ""
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                first = ln
                break
    if ":" in first and "," not in first and "\t" not in first:
        return read_biolip_colon(path)
    # fall back to CSV readers (assumes header present)
    try:
        return read_biolip_rows(path)
    except Exception:
        return read_biolip_list(path)

# --------------------- load indices_aa --------------------
def load_indices_aa(npy_path: Path) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        raise ValueError(f"{npy_path.name} looks like a raw array; expected a dict with 'indices_aa'")
    try:
        d = arr.item()
    except Exception:
        raise ValueError(f"{npy_path.name} not a dict-like .npy")
    if "indices_aa" not in d:
        raise KeyError(f"'indices_aa' not found in {npy_path.name}; keys={list(d.keys())}")
    return np.asarray(d["indices_aa"]).astype(int).reshape(-1)

# --------------- paper-rule labels (KD-tree) --------------
def get_chain_from_model(model, chain_id: str):
    if chain_id in model: return model[chain_id]
    for cid in (chain_id.upper(), chain_id.lower()):
        if cid in model: return model[cid]
    for ch in model:
        if getattr(ch, "id", None) == chain_id: return ch
    return None

def paper_rule_labels_fast(struct_path: Path, chain_id: str, cutoff: float) -> Optional[np.ndarray]:
    try:
        from Bio.PDB import MMCIFParser, PDBParser, is_aa
        from scipy.spatial import cKDTree
    except Exception:
        print("Need biopython + scipy (pip install biopython scipy)", file=sys.stderr); raise
    # parse structure
    if struct_path.suffix.lower() == ".cif":
        structure = MMCIFParser(QUIET=True).get_structure("x", str(struct_path))
    else:
        structure = PDBParser(QUIET=True).get_structure("x", str(struct_path))
    try:
        model = next(iter(structure))
    except StopIteration:
        return None
    target_chain = get_chain_from_model(model, chain_id)
    if target_chain is None:
        return None
    target_res = [r for r in target_chain if is_aa(r)]
    if not target_res:
        return None
    # other chains' heavy atoms
    other_coords = []
    for ch in model:
        if ch.id == target_chain.id: continue
        for r in ch:
            if not is_aa(r): continue
            for a in r.get_atoms():
                name = str(a.get_name()).upper()
                if getattr(a, "element", None) != "H" and not name.startswith("H"):
                    other_coords.append(a.coord)
    out = np.zeros(len(target_res), dtype=np.int8)
    if not other_coords:
        return out
    tree = cKDTree(np.asarray(other_coords, dtype=np.float32))
    # target heavy atoms grouped per residue
    res_offsets = [0]; tcoords: List[np.ndarray] = []
    for r in target_res:
        for a in r.get_atoms():
            name = str(a.get_name()).upper()
            if getattr(a, "element", None) != "H" and not name.startswith("H"):
                tcoords.append(a.coord)
        res_offsets.append(len(tcoords))
    if not tcoords:
        return out
    tcoords = np.asarray(tcoords, dtype=np.float32)
    neigh = tree.query_ball_point(tcoords, r=cutoff)
    for i in range(len(target_res)):
        s, e = res_offsets[i], res_offsets[i+1]
        out[i] = 1 if e > s and any(len(neigh[j]) > 0 for j in range(s, e)) else 0
    return out

# --------------------------- main -------------------------
def main():
    args = parse_args()
    for p in [args.labels_dir, args.ids_file, args.struct_dir, args.biolip_csv]:
        if not p.exists():
            sys.exit(f"[err] not found: {p}")

    # Load BioLiP positives
    try:
        biolip_pos = read_biolip_any(args.biolip_csv, forced_format=args.format)
    except Exception as e:
        sys.exit(f"[err] reading BioLiP annotations: {e}")

    # Load IDs
    ids = [ln.strip() for ln in args.ids_file.read_text().splitlines() if ln.strip()]
    if not ids: sys.exit(f"[err] no IDs in {args.ids_file}")

    random.seed(args.seed)
    sample = random.sample(ids, min(args.sample, len(ids)))

    rows = []
    for i, key in enumerate(sample, 1):
        if "_" not in key:
            if args.progress: print(f"[{i}/{len(sample)}] skip malformed id: {key}")
            continue
        pdb, ch = key.split("_", 1)
        pdb_low = pdb.lower()
        npy = args.labels_dir / f"{key}.npy"
        cif = args.struct_dir / f"{pdb_low}.cif"
        pdb_fallback = args.struct_dir / f"{pdb_low}.pdb"

        if not npy.exists():
            if args.progress: print(f"[{i}/{len(sample)}] missing labels npy: {npy}")
            continue
        struct_path = cif if cif.exists() else (pdb_fallback if pdb_fallback.exists() else None)
        if struct_path is None:
            if args.progress: print(f"[{i}/{len(sample)}] missing structure: {cif} / {pdb_fallback}")
            continue

        # indices_aa from your .npy
        try:
            idx_aa = load_indices_aa(npy)
        except Exception as e:
            if args.progress: print(f"[{i}/{len(sample)}] can't read indices_aa in {npy.name}: {e}")
            continue

        # rebuild YOUR labels from BioLiP
        pos_set = biolip_pos.get((pdb_low, ch), set())
        mine = np.isin(idx_aa, np.fromiter(pos_set, dtype=int)) if pos_set else np.zeros_like(idx_aa, dtype=bool)
        y_yours = mine.astype(np.int8)

        # PAPER labels by 4Å to other chains (asymmetric unit)
        try:
            y_paper = paper_rule_labels_fast(struct_path, ch, cutoff=args.cutoff)
        except Exception as e:
            if args.progress: print(f"[{i}/{len(sample)}] paper-rule failed for {key}: {e}")
            continue
        if y_paper is None:
            if args.progress: print(f"[{i}/{len(sample)}] no residues for {key}")
            continue

        n = min(len(y_yours), len(y_paper))
        if n == 0:
            if args.progress: print(f"[{i}/{len(sample)}] zero-length {key}")
            continue
        y1, y2 = y_yours[:n], y_paper[:n]

        inter = int(np.logical_and(y1==1, y2==1).sum())
        union = int(np.logical_or(y1==1, y2==1).sum())
        jacc = (inter/union) if union else 1.0

        rows.append({
            "key": key,
            "yours_p": float(y1.mean()),
            "paper_p": float(y2.mean()),
            "jaccard": float(jacc),
            "n_res": int(n),
        })

        if args.progress:
            print(f"[{i}/{len(sample)}] {key}  yours_p={y1.mean():.4f}  paper_p={y2.mean():.4f}  jacc={jacc:.3f}")

    if not rows:
        sys.exit("No comparable chains processed. Check --biolip-csv format and paths.")

    rows.sort(key=lambda r: r["jaccard"])  # worst overlaps first

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

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["key","yours_p","paper_p","jaccard","n_res"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote CSV: {args.csv}")

if __name__ == "__main__":
    main()
