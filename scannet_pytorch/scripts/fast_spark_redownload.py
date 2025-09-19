from pyspark.sql import SparkSession
import os
import requests

# ---------------- CONFIG ---------------- #
SKIP_FILES = [
    "data/train_skipped.txt",
    "data/val_skipped.txt",
    "data/test_skipped.txt"
]
OUT_DIR = "/home/scratch1/asiddhi/benchmarking_model2/ScanNet/PDB"
WORKERS = 32  # used in --master local[32] when running spark-submit
TIMEOUT = 15  # seconds for HTTP requests
# ---------------------------------------- #

def load_skips():
    ids = set()
    for path in SKIP_FILES:
        with open(path) as f:
            for line in f:
                pdb_chain = line.strip().split()[0]  # format: 1abc_A
                pdb_id = pdb_chain.split("_")[0].lower()
                ids.add(pdb_id)
    return sorted(ids)

def file_exists(pdb_id):
    return (
        os.path.exists(os.path.join(OUT_DIR, f"{pdb_id}.cif")) or
        os.path.exists(os.path.join(OUT_DIR, f"{pdb_id}.pdb"))
    )

def download_pdb(pdb_id):
    """Download CIF or PDB, return status string."""
    try:
        cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        cif_path = os.path.join(OUT_DIR, f"{pdb_id}.cif")
        pdb_path = os.path.join(OUT_DIR, f"{pdb_id}.pdb")

        # Try CIF first
        r = requests.get(cif_url, timeout=TIMEOUT)
        if r.status_code == 200:
            with open(cif_path, "wb") as f:
                f.write(r.content)
            return f"{pdb_id} OK (cif)"

        # Try PDB if CIF failed
        r = requests.get(pdb_url, timeout=TIMEOUT)
        if r.status_code == 200:
            with open(pdb_path, "wb") as f:
                f.write(r.content)
            return f"{pdb_id} OK (pdb)"

        return f"{pdb_id} FAIL: not found"

    except Exception as e:
        return f"{pdb_id} FAIL: {str(e)}"

def main():
    spark = (
        SparkSession.builder
        .appName("FastSparkPDBDownloader")
        .getOrCreate()
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    all_ids = load_skips()
    missing_ids = [pid for pid in all_ids if not file_exists(pid)]

    print(f"Total PDB IDs in skips: {len(all_ids)}")
    print(f"Already present locally: {len(all_ids) - len(missing_ids)}")
    print(f"To download: {len(missing_ids)}")

    rdd = spark.sparkContext.parallelize(missing_ids, numSlices=len(missing_ids))

    results = rdd.map(download_pdb).collect()

    # Save log
    log_path = os.path.join(OUT_DIR, "fast_spark_redownload.log")
    with open(log_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Done. Log saved to {log_path}")

if __name__ == "__main__":
    main()
