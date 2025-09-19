#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioLiP preprocessing for ScanNet (dict quartet output).
Supports sequential mode (--num-workers 0) and spawn-safe workers.
"""

import os, sys, re, time, pickle, random, signal, argparse, traceback
from typing import Dict, List, Sequence, Tuple
import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa

# --- Repo imports ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, BASE_DIR)

from scannet_pytorch.utilities import paths
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.utilities.id_normalization import norm_pdb_id, norm_chain

# --- Paths & Config ---
BIOLIP_DIR     = os.environ.get("BIOLIP_DIR", os.path.join(BASE_DIR, "datasets", "BioLip"))
STRUCTURES_DIR = os.environ.get("BIOLIP_STRUCTURES", os.path.join(BIOLIP_DIR, "structures"))
BIOLIP_SUBSET  = os.environ.get("BIOLIP_SUBSET", os.path.join(BIOLIP_DIR, "biolip_subset.pkl"))
BIOLIP_ANN_CSV = os.environ.get("BIOLIP_ANN", os.path.join(BIOLIP_DIR, "BioLiP_annotation.csv"))

SPLIT_DIR = os.path.join(BIOLIP_DIR, "splits"); os.makedirs(SPLIT_DIR, exist_ok=True)
DATA_ROOT = os.path.abspath(os.path.join(paths.library_folder, "..", "data")); os.makedirs(DATA_ROOT, exist_ok=True)

NUM_WORKERS          = int(os.environ.get("BIOPRE_NUM_WORKERS", 4))
TIMEOUT_PER_CHAIN    = int(os.environ.get("BIOPRE_TIMEOUT", 30))
MAX_STRUCTURE_SIZE_MB= int(os.environ.get("BIOPRE_MAX_CIF_MB", 100))
MAX_STRUCTURE_SIZE   = MAX_STRUCTURE_SIZE_MB * 1024**2
MAX_RESIDUES_PER_CHAIN=int(os.environ.get("BIOPRE_MAX_RES", 2000))
RANDOM_SEED          = int(os.environ.get("BIOPRE_SEED", 42))

# --- Globals ---
_EXISTING_PDBS: set = set()
_LARGE_PDBS: set    = set()
_PIPELINE = None

# unify names
STRUCTURES_DIR = os.environ.get("BIOLIP_STRUCTURES", os.path.join(BIOLIP_DIR, "structures"))
_STRUCTURES_DIR = STRUCTURES_DIR

# --- Helpers ---
def _origin_to_pair(origin: str) -> Tuple[str,str]:
    parts = origin.strip().split("_")
    if len(parts)==3: pdb_raw,_,ch_raw=parts
    elif len(parts)==2: pdb_raw,ch_raw=parts
    else: raise ValueError(f"Bad origin: {origin}")
    return norm_pdb_id(pdb_raw), norm_chain(ch_raw)

def to_pdb_chain(origin: str) -> str:
    p,c=_origin_to_pair(origin); return f"{p}_{c}"

def _timeout(seconds:int):
    class Alarm:
        def __enter__(self):
            if seconds>0:
                signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(seconds)
        def __exit__(self,a,b,c): signal.alarm(0); return False
    return Alarm()

def _probe_npy(split_dir: str, max_checks: int = 5):
    """
    Quick sanity check: load a few .npy files from split_dir
    and verify they contain the dict quartet with aligned shapes.
    """
    import numpy as np, os, random
    files = [f for f in os.listdir(split_dir) if f.endswith(".npy")]
    if not files:
        print(f"⚠️ Probe: no .npy files found in {split_dir}")
        return
    sample = random.sample(files, min(max_checks, len(files)))
    for f in sample:
        path = os.path.join(split_dir, f)
        try:
            d = np.load(path, allow_pickle=True).item()
            keys = list(d.keys())
            L = d["attr_aa"].shape[0]
            assert all(d[k].shape[0] == L for k in ["coord_aa","triplets_aa","indices_aa"])
            print(f"✅ Probe {f}: keys={keys}, L={L}")
        except Exception as e:
            print(f"❌ Probe failed for {f}: {type(e).__name__} {e}")
    print(f"🔎 Probe complete for {split_dir} ({len(files)} total files).")

# --- Trim + align helper used by both worker & sequential paths ---
def _trim_and_align(feats, aligned):
    """
    Normalize, trim to common length, cast, and NaN-guard the quartet + labels.
    Returns (sample_dict, aligned) or None if invalid.
    """
    import numpy as _np

    # Unpack
    try:
        aa_trip, aa_attr, aa_idx, aa_clouds = feats[:4]
    except Exception:
        return None

    # Length helper (robust to lists/arrays/None)
    def _len0(x):
        if x is None:
            return 0
        try:
            return int(getattr(x, "shape", [0])[0])
        except Exception:
            x_arr = _np.array(x)
            return int(x_arr.shape[0]) if x_arr.ndim >= 1 else 0

    lengths = [
        _len0(aa_attr),
        _len0(aa_trip),
        _len0(aa_idx),
        _len0(aa_clouds),
        int(getattr(aligned, "size", 0)),
    ]
    if any(L <= 0 for L in lengths):
        return None

    L = min(lengths)

    # Convert + trim + cast
    aa_trip   = _np.array(aa_trip)  [:L].astype(_np.float32, copy=False)
    aa_attr   = _np.array(aa_attr)  [:L].astype(_np.float32, copy=False)
    aa_idx    = _np.array(aa_idx)   [:L].astype(_np.int32,   copy=False)
    aa_clouds = _np.array(aa_clouds)[:L].astype(_np.float32, copy=False)
    aligned   = _np.array(aligned)  [:L].astype(_np.float32, copy=False)

    # NaN guard for float arrays
    for arr in (aa_trip, aa_attr, aa_clouds):
        if _np.isnan(arr).any():
            arr[_np.isnan(arr)] = 0.0

    sample = _build_sample_dict(aa_trip, aa_attr, aa_idx, aa_clouds)
    return sample, aligned



# --- Build dict quartet (stable dtypes & keys) ---
def _build_sample_dict(aa_triplets, aa_attr, aa_idx, aa_clouds) -> Dict[str, np.ndarray]:
    L = int(np.asarray(aa_attr).shape[0])
    if aa_triplets is None:
        aa_triplets = np.zeros((L, 0), dtype=np.float32)
    return {
        "coord_aa":    np.array(aa_clouds, dtype=np.float32, copy=False),
        "attr_aa":     np.array(aa_attr,   dtype=np.float32, copy=False),
        "triplets_aa": np.array(aa_triplets, dtype=np.float32, copy=False),
        "indices_aa":  np.array(aa_idx,    dtype=np.int32,   copy=False),
    }

# --- BioLiP annotations ---
def _load_biolip_annotations(path:str)->Dict[str,set]:
    ann={}; 
    with open(path) as f:
        for line in f:
            if ":" not in line: continue
            left,right=line.strip().split(":",1)
            key=to_pdb_chain(left); residues=set()
            for tok in right.split():
                m=re.search(r"(\d+)",tok); 
                if m: residues.add(int(m.group(1)))
            ann.setdefault(key,set()).update(residues)
    return ann

BIOLIP_ANN=_load_biolip_annotations(BIOLIP_ANN_CSV)

def _generate_binding_labels(pdb,chain)->np.ndarray:
    key=f"{pdb}_{chain}"
    if key not in BIOLIP_ANN: return np.zeros(0,dtype=np.float32)
    cif=os.path.join(STRUCTURES_DIR,f"{pdb}.cif")
    if not os.path.exists(cif): return np.zeros(0,dtype=np.float32)
    structure=MMCIFParser(QUIET=True).get_structure(pdb,cif)
    if chain not in structure[0]: return np.zeros(0,dtype=np.float32)
    residues=[r for r in structure[0][chain] if is_aa(r)]
    if len(residues)>MAX_RESIDUES_PER_CHAIN: return np.zeros(0,dtype=np.float32)
    labels=np.zeros(len(residues),dtype=np.float32)
    for i,res in enumerate(residues):
        if res.id[1] in BIOLIP_ANN[key]: labels[i]=1.0
    return labels

# --- Pre-scan CIFs ---
def pre_scan_structures(pairs:Sequence):
    global _EXISTING_PDBS,_LARGE_PDBS
    _EXISTING_PDBS.clear(); _LARGE_PDBS.clear()
    pdbs={norm_pdb_id(p if isinstance(p,str) else p[0]) for p in pairs}
    for pdb in pdbs:
        cif=os.path.join(STRUCTURES_DIR,f"{pdb}.cif")
        if not os.path.exists(cif): continue
        try: size=os.path.getsize(cif)
        except: continue
        if size>MAX_STRUCTURE_SIZE: _LARGE_PDBS.add(pdb); continue
        _EXISTING_PDBS.add(pdb)
    print(f"✅ Pre-scan CIFs: usable={len(_EXISTING_PDBS)} | too_large={len(_LARGE_PDBS)}")

# --- Worker init & worker ---
def _worker_feature_init(structures_dir:str,pipeline_kwargs:Dict):
    global _PIPELINE,_STRUCTURES_DIR
    _STRUCTURES_DIR=structures_dir
    _PIPELINE=ScanNetPipeline(**pipeline_kwargs)
    print(f"[Worker init] PID={os.getpid()} pipeline ready",flush=True)

# --- Worker path (spawn-safe) ---
def _worker_process_one(origin: str):
    """
    Process a single origin inside a worker.
    Expects worker initializer to set globals: _PIPELINE, _STRUCTURES_DIR.
    Returns: (key, sample_dict_or_None, aligned_or_None, status_str)
    """
    try:
        pdb, ch = _origin_to_pair(origin)
        key = f"{pdb}_{ch}"

        # Ensure CIF exists
        cif = os.path.join(_STRUCTURES_DIR, f"{pdb}.cif")
        if not os.path.exists(cif):
            return key, None, None, "no_cif"

        # Labels (skip empty)
        labels = _generate_binding_labels(pdb, ch)
        if not isinstance(labels, np.ndarray) or labels.size == 0:
            return key, None, None, "empty_labels"

        # Parse structure and fetch chain
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb, cif)
        try:
            chain_obj = structure[0][ch]
        except Exception:
            # Normalize chain id gracefully (e.g., whitespace/case)
            chain_obj = structure[0][ch.strip()]

        # Run pipeline → (feats, aligned)
        out = _PIPELINE.process_example(chain_obj=chain_obj, labels=labels)
        if not (isinstance(out, (list, tuple)) and len(out) == 2):
            return key, None, None, "pipeline_return_error"

        feats, aligned = out
        if aligned is None or int(getattr(aligned, "size", 0)) == 0:
            return key, None, None, "empty_aligned"

        trimmed = _trim_and_align(feats, aligned)
        if trimmed is None:
            return key, None, None, "len_mismatch"

        sample, aligned = trimmed
        return key, sample, aligned, "ok"

    except TimeoutError:
        return origin, None, None, "timeout"
    except FileNotFoundError:
        return origin, None, None, "file_not_found"
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return origin, None, None, f"error:{type(e).__name__}"



# --- Main split processor (sequential + multiprocessing) ---
def _process_split(name: str, origins: Sequence[str], pipeline: ScanNetPipeline):
    """
    Process a split and write .npy dict quartets + labels .pkl.
    - Sequential mode when NUM_WORKERS <= 0
    - ProcessPool with spawn-safe initializer when NUM_WORKERS > 0
    """
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor, as_completed

    split_dir = os.path.join(DATA_ROOT, f"{name}.data")
    os.makedirs(split_dir, exist_ok=True)

    processed_ids_file = os.path.join(DATA_ROOT, f"{name}_processed_ids.txt")
    already = set()
    if os.path.exists(processed_ids_file):
        with open(processed_ids_file) as f:
            already = {ln.strip() for ln in f if ln.strip()}

    # Filter out already processed
    want = [o for o in origins if to_pdb_chain(o) not in already]

    # Where we accumulate labels aligned to each saved .npy
    labels_out: Dict[str, np.ndarray] = {}
    wrote = 0
    skipped: Dict[str, str] = {}

    # Worker initializer will set globals used by _worker_process_one
    def _worker_feature_init(structures_dir, pipeline_state):
        global _STRUCTURES_DIR, _PIPELINE
        _STRUCTURES_DIR = structures_dir
        _PIPELINE = pipeline_state

    # ---------------- SEQUENTIAL MODE ----------------
    if NUM_WORKERS <= 0:
        print("🔧 Running in sequential mode")
        for i, o in enumerate(want, 1):
            try:
                pdb, ch = _origin_to_pair(o)
                key = f"{pdb}_{ch}"

                cif = os.path.join(_STRUCTURES_DIR, f"{pdb}.cif")
                if not os.path.exists(cif):
                    skipped[o] = "no_cif"
                    continue

                labels = _generate_binding_labels(pdb, ch)
                if not isinstance(labels, np.ndarray) or labels.size == 0:
                    skipped[o] = "empty_labels"
                    continue

                structure = MMCIFParser(QUIET=True).get_structure(pdb, cif)
                chain_obj = structure[0][ch]

                out = pipeline.process_example(chain_obj=chain_obj, labels=labels)
                if not (isinstance(out, (list, tuple)) and len(out) == 2):
                    skipped[o] = "pipeline_return_error"
                    continue

                feats, aligned = out
                trimmed = _trim_and_align(feats, aligned)
                if trimmed is None:
                    skipped[o] = "len_mismatch"
                    continue

                sample, aligned = trimmed

                # Save
                np.save(os.path.join(split_dir, f"{key}.npy"), sample, allow_pickle=True)
                labels_out[key] = aligned.astype(np.float32, copy=False)
                with open(processed_ids_file, "a") as pf:
                    pf.write(key + "\n")
                wrote += 1

                if i % 1000 == 0:
                    print(f"… {i} processed; wrote={wrote}; skipped={len(skipped)}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                tb = traceback.format_exc()
                print(f"❌ Error for {o}: {e}\n{tb}")
                skipped[o] = f"error:{type(e).__name__}"

    # ---------------- MULTIPROCESS MODE ----------------
    else:
        print(f"🧵 Spawning {NUM_WORKERS} workers")
        CHUNK_SIZE = 2000  # submissions per chunk
        # Pipeline object is assumed to be picklable or has state for workers
        with ProcessPoolExecutor(
            max_workers=NUM_WORKERS,
            initializer=_worker_feature_init,
            initargs=(_STRUCTURES_DIR, pipeline),
        ) as ex:
            for start in range(0, len(want), CHUNK_SIZE):
                chunk = want[start:start + CHUNK_SIZE]
                futures = {ex.submit(_worker_process_one, o): o for o in chunk}
                for fut in as_completed(futures):
                    o = futures[fut]
                    try:
                        key, sample, aligned, status = fut.result()
                        if status == "ok":
                            np.save(os.path.join(split_dir, f"{key}.npy"), sample, allow_pickle=True)
                            labels_out[key] = aligned.astype(np.float32, copy=False)
                            with open(processed_ids_file, "a") as pf:
                                pf.write(key + "\n")
                            wrote += 1
                        else:
                            skipped[o] = status
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        skipped[o] = f"error:{type(e).__name__}"

    # Persist labels (single pickle per split)
    labels_pkl = os.path.join(DATA_ROOT, f"{name}_labels.pkl")
    with open(labels_pkl, "wb") as f:
        pickle.dump(labels_out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Split '{name}': wrote={wrote}, skipped={len(skipped)} → {split_dir}")
    if skipped:
        skipped_log = os.path.join(DATA_ROOT, f"{name}_skipped.txt")
        with open(skipped_log, "w") as f:
            for k, v in sorted(skipped.items()):
                f.write(f"{k}\t{v}\n")
        print(f"⚠️  Skipped list → {skipped_log}")


# --- Main ---
def main():
    global STRUCTURES_DIR, _STRUCTURES_DIR, NUM_WORKERS, DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-list", type=str, default=None,
                    help="File with explicit PDB_chain IDs to process (batch mode)")
    ap.add_argument("--skip-list", type=str, default=None,
                    help="File with IDs to skip (legacy mode)")
    ap.add_argument("--cif-root", type=str, required=True,
                    help="Directory with .cif structure files")
    ap.add_argument("--output-dir", type=str, default="data",
                    help="Directory to save outputs")
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                    help="Number of worker processes")
    args = ap.parse_args()

    # Configure globals
    STRUCTURES_DIR = os.path.abspath(args.cif_root)
    DATA_ROOT = os.path.abspath(args.output_dir)
    NUM_WORKERS = args.num_workers
    _STRUCTURES_DIR = STRUCTURES_DIR
    print(f"🔧 CIF root: {STRUCTURES_DIR}")
    print(f"🔧 NUM_WORKERS={NUM_WORKERS}")
    print(f"📂 Output dir: {DATA_ROOT}")
    os.makedirs(DATA_ROOT, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Batch mode: process only IDs in input-list ---
    if args.input_list:
        with open(args.input_list) as f:
            origins = [ln.strip() for ln in f if ln.strip()]
        split_name = f"recovery_{os.path.basename(args.input_list)}"
        pre_scan_structures([_origin_to_pair(o) for o in origins if "_" in o])
        pipeline = ScanNetPipeline(
            atom_features="valency",
            aa_features="sequence",
            aa_frames="triplet_sidechain",
            Beff=500,
        )
        _process_split(split_name, origins, pipeline)
        return

    # --- Skip-list mode: process all except those in skip-list ---
    if args.skip_list:
        split = os.path.basename(args.skip_list).split("_")[0]
        with open(args.skip_list) as f:
            pairs = [_origin_to_pair(l) for l in f if "_" in l]
        origins = [f"{p}_{c}" for p, c in pairs]
        pre_scan_structures(pairs)
        pipeline = ScanNetPipeline(
            atom_features="valency",
            aa_features="sequence",
            aa_frames="triplet_sidechain",
            Beff=500,
        )
        _process_split(f"reproc_{split}_{int(time.time())}", origins, pipeline)
        return

    # --- Full mode: process train/val/test splits ---
    with open(BIOLIP_SUBSET, "rb") as f:
        raw = pickle.load(f)
    pairs = [
        _origin_to_pair(str(k)) if not isinstance(k, tuple)
        else (norm_pdb_id(k[0]), norm_chain(k[1]))
        for k in (raw if isinstance(raw, list) else raw.keys())
    ]
    pairs = [(p, c) for p, c in pairs if p and c]
    print(f"✅ Loaded {len(pairs)} chains from subset")
    pre_scan_structures(pairs)

    random.shuffle(pairs)
    n = len(pairs)
    train, val, test = (
        pairs[:int(.7 * n)],
        pairs[int(.7 * n):int(.85 * n)],
        pairs[int(.85 * n):],
    )
    for nm, sub in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(SPLIT_DIR, f"{nm}_origins.txt"), "w") as f:
            for p, c in sub:
                f.write(f"{p}_{c}\n")
    pipeline = ScanNetPipeline(
        atom_features="valency",
        aa_features="sequence",
        aa_frames="triplet_sidechain",
        Beff=500,
    )
    _process_split("train", [f"{p}_{c}" for p, c in train], pipeline)
    _process_split("val",   [f"{p}_{c}" for p, c in val],   pipeline)
    _process_split("test",  [f"{p}_{c}" for p, c in test],  pipeline)
    print("🎉 Full preprocessing finished")


if __name__ == "__main__":
    main()
