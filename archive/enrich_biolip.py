#!/usr/bin/env python
import os, glob, time, argparse, sys
import numpy as np
from Bio.PDB import MMCIFParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

def process_one(args):
    f, structures_dir, out_dir = args
    try:
        data = np.load(f, allow_pickle=True).item()
        pdb_id = os.path.basename(f).split(".")[0]
        base, chain_id = pdb_id.split("_", 1)

        cif = os.path.join(structures_dir, f"{base.lower()}.cif")
        if not os.path.exists(cif):
            return pdb_id, "no_cif"

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(base, cif)
        if chain_id not in structure[0]:
            return pdb_id, "no_chain"

        chain = structure[0][chain_id]
        coords, atom_types, res_idx = [], [], []
        for residue in chain:
            for atom in residue:
                coords.append(atom.coord)
                atom_types.append(atom.element)
                res_idx.append(residue.id[1])

        data["coord_atom"] = np.array(coords, dtype=np.float32)
        data["indices_atom"] = np.array(res_idx, dtype=np.int32)
        data["attr_atom"] = np.array(atom_types)

        out_path = os.path.join(out_dir, f"{pdb_id}.npy")
        np.save(out_path, data, allow_pickle=True)
        return pdb_id, "ok"
    except Exception as e:
        return os.path.basename(f), f"error:{type(e).__name__}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, required=True,
                    choices=["train", "val", "test"])
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--structures-dir", type=str,
                    default=os.environ.get("BIOLIP_STRUCTURES",
                                           os.path.join("datasets","BioLip","structures")))
    ap.add_argument("--id-list", type=str,
                    help="Optional file containing subset of IDs to process (one per line, without .npy)")

    args = ap.parse_args()

    # log setup
    os.makedirs("logs/enrich", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs/enrich", f"{args.split}_{ts}.log")
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout
    print(f"📓 Logging to {log_path}\n")

    in_dir = os.path.join(args.data_dir, f"{args.split}.data")
    out_dir = os.path.join(args.data_dir, f"{args.split}.atom_enriched")
    os.makedirs(out_dir, exist_ok=True)

    # ✅ apply id-list if provided
    if args.id_list:
        with open(args.id_list) as f:
            subset_ids = {ln.strip() for ln in f if ln.strip()}
        files = [os.path.join(in_dir, f"{pid}.npy") for pid in subset_ids]
    else:
        files = sorted(glob.glob(os.path.join(in_dir, "*.npy")))

    print(f"🔧 Split={args.split} | {len(files)} npy files in {in_dir}")
    print(f"💾 Output → {out_dir}")
    print(f"⚙️ Workers = {args.n_workers}")

    # checkpoint
    processed_file = os.path.join(out_dir, "processed_ids.txt")
    already = set()
    if os.path.exists(processed_file):
        with open(processed_file) as f:
            already = {ln.strip() for ln in f if ln.strip()}
    todo = [f for f in files if os.path.basename(f).split(".")[0] not in already]
    print(f"Resuming: {len(already)} already processed, {len(todo)} left")

    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(process_one, (f, args.structures_dir, out_dir)): f for f in todo}
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Enriching")):
            results.append(fut.result())
            if (i+1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i+1) / elapsed
                remaining = (len(futures)-(i+1)) / max(rate, 1e-6)
                print(f"[Progress] {i+1}/{len(futures)} | {rate:.2f} chains/s | "
                      f"ETA {remaining/60:.1f} min", flush=True)

    elapsed = time.time() - start
    ok = [pid for pid, s in results if s == "ok"]
    skipped = [(pid, s) for pid, s in results if s != "ok"]

    if ok:
        with open(processed_file, "a") as f:
            for pid in ok:
                f.write(pid + "\n")

    print(f"\n✅ {args.split}: enriched={len(ok)}, skipped={len(skipped)} "
          f"in {elapsed/60:.2f} min "
          f"(avg {elapsed/max(len(ok),1):.3f} s/entry)")

    if skipped:
        log_skipped = os.path.join(out_dir, f"{args.split}_skipped.txt")
        with open(log_skipped, "a") as f:
            for pid, status in skipped:
                f.write(f"{pid}\t{status}\n")
        print(f"⚠️ Skipped list → {log_skipped}")

if __name__ == "__main__":
    main()
