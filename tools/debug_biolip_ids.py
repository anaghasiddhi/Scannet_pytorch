#!/usr/bin/env python3
import os, re, glob, pickle, random
from collections import Counter

STRUCT_DIR = "datasets/BioLip/structures"
PKL_PATH   = "data/biolip_subset.pkl"            # adjust if needed
ORIGINS_GLOB = "data/*_origins.txt"              # adjust if needed

PDB_RE = re.compile(r'([0-9][A-Za-z0-9]{3})')    # first 4-char PDB-like token

def norm_pdb_id(s: str) -> str:
    """Return canonical 4-char pdb_id (lowercase). Robust to extras like 8aa2_1_A."""
    s = s.strip()
    m = PDB_RE.search(s)
    if not m:
        return ""
    return m.group(1).lower()

def norm_chain(s: str) -> str:
    """
    Extract a single chain ID if present.
    - Accepts '8aa2_1_A', '8aa2_A', 'A', 'a'
    - If multiple chains exist (e.g., 'A,B'), take the first (warn).
    """
    # pick last underscore token if any
    tok = s.strip().split("_")[-1]
    # split on commas if present
    tok = tok.split(",")[0]
    # only keep simple alnum, usually 1 char
    tok = re.sub(r'[^A-Za-z0-9]', '', tok)
    return tok.upper() if tok else ""

def norm_origin_to_pdb_chain(origin: str) -> str:
    pdb = norm_pdb_id(origin)
    chn = norm_chain(origin)
    return f"{pdb}_{chn}" if pdb and chn else pdb

def list_cif_pdbs(struct_dir):
    pdbs = set()
    for f in os.listdir(struct_dir):
        if f.lower().endswith(".cif"):
            base = f[:-4]  # strip .cif
            pdbs.add(base.lower())
    return pdbs

def read_origins(globpat):
    origins = []
    for p in glob.glob(globpat):
        with open(p, "r") as fh:
            for line in fh:
                t = line.strip()
                if t:
                    origins.append(t)
    return origins

def read_biolip_subset(pkl_path):
    try:
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
    except FileNotFoundError:
        return []
    # obj could be list/dict; normalize to list of strings we compare against
    items = []
    if isinstance(obj, dict):
        items = list(obj.keys())
    elif isinstance(obj, (list, tuple, set)):
        items = list(obj)
    else:
        # attempt generic extraction
        try:
            items = list(obj)
        except Exception:
            pass
    return [str(x) for x in items]

def main():
    cif_pdbs = list_cif_pdbs(STRUCT_DIR)                       # e.g., {'8aa2', ...}
    origins  = read_origins(ORIGINS_GLOB)                      # e.g., ['8aa2_1_A', ...]
    pkl_items = read_biolip_subset(PKL_PATH)                   # e.g., ['8aa2_A', ...] or similar

    # Normalize
    origins_pdbs   = {norm_pdb_id(o) for o in origins}
    origins_pairs  = {norm_origin_to_pdb_chain(o) for o in origins if norm_origin_to_pdb_chain(o)}

    pkl_pdbs = {norm_pdb_id(x) for x in pkl_items}
    pkl_pairs = set()
    for x in pkl_items:
        # if items are already like '8aa2_A', keep; else fallback to pdb only
        if "_" in x:
            pkl_pairs.add(f"{norm_pdb_id(x)}_{norm_chain(x)}")
        else:
            pkl_pairs.add(norm_pdb_id(x))

    print("\n=== SAMPLE CHECKS ===")
    print(f"CIF pdb count: {len(cif_pdbs)}  | sample: {sorted(list(cif_pdbs))[:5]}")
    print(f"Origins lines: {len(origins)}   | sample: {origins[:5]}")
    print(f"biolip_subset items: {len(pkl_items)} | sample: {pkl_items[:5]}")

    print("\n=== DIFFS (PDB only) ===")
    missing_in_cif_from_origins = sorted(list(origins_pdbs - cif_pdbs))[:25]
    print(f"PDBs referenced by origins but no CIF on disk (first 25): {missing_in_cif_from_origins}")

    missing_in_cif_from_pkl = sorted(list(pkl_pdbs - cif_pdbs))[:25]
    print(f"PDBs referenced in pickle but no CIF on disk (first 25): {missing_in_cif_from_pkl}")

    print("\n=== DIFFS (PDB_CHAIN) ===")
    # Build a set of available PDB_CHAIN derived only from CIF + any chain we see in origins
    # This tells us if the parser is introducing spurious middle tokens.
    cif_plus_chain_from_origins = {f"{p}_{norm_chain(o)}"
                                   for o in origins for p in [norm_pdb_id(o)]
                                   if p and norm_chain(o)}
    missing_pairs_from_cif = sorted(list((origins_pairs | pkl_pairs) - cif_plus_chain_from_origins))[:25]
    print(f"PDB_CHAIN combos referenced but not seen in origins→CIF mapping (first 25): {missing_pairs_from_cif}")

    # Frequent raw formats in origins (to spot `_1_`)
    pat = Counter()
    for o in origins:
        if "_1_" in o:
            pat["_1_"] += 1
        elif re.search(r'_[A-Za-z0-9]$', o):
            pat["_CHAIN_SUFFIX"] += 1
        else:
            pat["other"] += 1
    print("\nOrigins pattern summary:", pat)

    # Final verdict hint
    if missing_in_cif_from_origins or missing_in_cif_from_pkl:
        print("\nLikely cause: ID normalization drift (e.g., extra `_1_` token).")
    else:
        print("\nStructure presence looks OK; focus on how process_chain does its lookups.")

if __name__ == "__main__":
    main()
