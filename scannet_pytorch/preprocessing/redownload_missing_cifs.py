#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import argparse
import pathlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

LINE_RE = re.compile(r"^\s*([0-9A-Za-z]{4})_([A-Za-z0-9])\s+\t?\s*no_cif\s*$")

def parse_skip_file(path: pathlib.Path):
    pairs = []
    with path.open() as fh:
        for line in fh:
            m = LINE_RE.match(line.strip())
            if m:
                pdb, chain = m.group(1).lower(), m.group(2)
                pairs.append((pdb, chain))
    return pairs

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def urlretrieve_with_retry(url: str, dst: pathlib.Path, retries=3, backoff=1.5) -> bool:
    for i in range(retries):
        try:
            urllib.request.urlretrieve(url, dst.as_posix())
            return True
        except Exception:
            if i == retries - 1:
                return False
            time.sleep(backoff ** i)
    return False

def fetch_one(pdb_id: str, chain_id: str, out_dir: pathlib.Path) -> dict:
    """Try CIF first, then PDB; skip validation for speed."""
    result = {
        "pdb_id": pdb_id, "chain_id": chain_id,
        "saved_path": None, "status": None, "detail": None
    }
    ensure_dir(out_dir)

    cif_path = out_dir / f"{pdb_id}.cif"
    pdb_path = out_dir / f"{pdb_id}.pdb"

    # Instant skip if file exists & >0 bytes
    for p in (cif_path, pdb_path):
        if p.exists() and p.stat().st_size > 0:
            result.update({"saved_path": p.as_posix(), "status": "already_ok", "detail": "skip_no_validation"})
            return result

    # Try CIF
    if urlretrieve_with_retry(f"https://files.rcsb.org/download/{pdb_id}.cif", cif_path):
        result.update({"saved_path": cif_path.as_posix(), "status": "downloaded", "detail": "cif_ok"})
        return result

    # Try PDB
    if urlretrieve_with_retry(f"https://files.rcsb.org/download/{pdb_id}.pdb", pdb_path):
        result.update({"saved_path": pdb_path.as_posix(), "status": "downloaded", "detail": "pdb_ok"})
        return result

    result.update({"status": "failed", "detail": "download_failed"})
    return result

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    ap = argparse.ArgumentParser(description="Re-download missing mmCIF/PDB files for skipped BioLip chains (fast + resume mode).")
    ap.add_argument("--skips", nargs="+", required=True, help="Paths to *_skipped.txt files.")
    ap.add_argument("--out", default="cifs", help="Directory to save structures.")
    ap.add_argument("--workers", type=int, default=8, help="Parallel downloads.")
    ap.add_argument("--report", default="redownload_report.json", help="JSON report path.")
    ap.add_argument("--goodlist", default="recovered_chains.txt", help="File for recovered/valid chains.")
    ap.add_argument("--batch-size", type=int, default=20000, help="How many IDs to process per batch.")
    ap.add_argument("--heartbeat", type=int, default=1000, help="Print progress every N downloads.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing report file.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    ensure_dir(out_dir)

    # Collect unique pairs from skip files
    pairs = []
    for s in args.skips:
        pairs.extend(parse_skip_file(pathlib.Path(s)))
    pairs = sorted(set(pairs))

    if not pairs:
        print("No pairs parsed from skip files.")
        sys.exit(1)

    # If resume mode, filter out already completed ones
    results = []
    if args.resume and os.path.exists(args.report):
        with open(args.report) as f:
            prev_report = json.load(f)
        done_ids = {(r["pdb_id"], r["chain_id"]) for r in prev_report["results"] if r["status"] in ("downloaded", "already_ok")}
        pairs = [p for p in pairs if p not in done_ids]
        results.extend(prev_report["results"])
        print(f"Resuming: Skipped {len(done_ids)} already done IDs.")

    total_ids = len(pairs) + len(results)
    print(f"Remaining to fetch: {len(pairs)} / {total_ids}")

    total_done = len(results)
    start_time = time.time()

    for batch_num, batch in enumerate(chunk_list(pairs, args.batch_size), 1):
        print(f"\n=== Starting batch {batch_num} ({len(batch)} IDs) ===", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one, pdb, ch, out_dir): (pdb, ch) for pdb, ch in batch}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                total_done += 1

                if total_done % args.heartbeat == 0:
                    elapsed = time.time() - start_time
                    print(f"[{total_done}/{total_ids}] done in {elapsed:.1f}s", flush=True)

        # Save progress after each batch
        with open(args.report, "w") as f:
            json.dump({
                "totals": defaultdict(int, {r["status"]: sum(1 for x in results if x["status"] == r["status"]) for r in results}),
                "out_dir": out_dir.as_posix(),
                "results": results
            }, f, indent=2)

    # Final output files
    recovered = [f"{r['pdb_id']}_{r['chain_id']}" for r in results if r["status"] in ("downloaded","already_ok")]
    with open(args.goodlist, "w") as f:
        f.write("\n".join(recovered))

    still_bad = [f"{r['pdb_id']}_{r['chain_id']}\t{r['detail']}" for r in results if r["status"] == "failed"]
    with open("still_missing.txt", "w") as f:
        f.write("\n".join(still_bad))

    print(f"\nDone.\nRecovered/valid: {len(recovered)}"
          f"\nFailed: {len(still_bad)}"
          f"\nReport: {args.report}"
          f"\nGood list: {args.goodlist}"
          f"\nStill missing: still_missing.txt")

if __name__ == "__main__":
    main()
