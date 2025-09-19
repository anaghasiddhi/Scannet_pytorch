import os
import pickle
import concurrent.futures
from pathlib import Path
from Bio.PDB import PDBParser, PPBuilder

# === Config ===
LABEL_PICKLES = {
    "train": "train_label_dict.pkl",
    "val_none": "val_none_label_dict.pkl",
    "val_topology": "val_topology_label_dict.pkl",
    "val_70": "val_70_label_dict.pkl",
    "val_homology": "val_homology_label_dict.pkl",
}
PDB_DIR = "/home/scratch1/asiddhi/benchmarking_model2/ScanNet/PDB"
LABEL_DIR = "/home/scratch1/asiddhi/benchmarking_model2/ScanNet/data"
OUT_FASTA_DIR = "fasta_splits"
Path(OUT_FASTA_DIR).mkdir(exist_ok=True)

def extract_chain_sequence(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                ppb = PPBuilder()
                peptides = ppb.build_peptides(chain)
                if peptides:
                    return str(peptides[0].get_sequence())
    return None

def extract_entry(pdb_id):
    try:
        chain_id = pdb_id.split("-")[-1]
        raw_pdb_code = pdb_id.split("-")[0].lower()
        if "_" in raw_pdb_code:
            raw_pdb_code = raw_pdb_code.split("_")[0]
        pdb_file = Path(PDB_DIR) / f"pdb{raw_pdb_code}.bioent"
        if not pdb_file.exists():
            return None

        seq = extract_chain_sequence(pdb_file, chain_id)
        if seq and len(seq) >= 10:
            print(f"[✓] Saved sequence for {pdb_id} (len={len(seq)})")
            return f">{pdb_id}\n{seq}"
    except Exception:
        return None
    return None


def process_split(split, label_file):
    with open(label_file, "rb") as f:
        label_dict = pickle.load(f)
    pdb_ids = list(label_dict.keys())

    out_fasta = Path(OUT_FASTA_DIR) / f"scan_{split}.fasta"
    print(f"[⋯] Processing {split} with {len(pdb_ids)} entries...")

    # Use a worker pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(extract_entry, pdb_ids))

    valid_entries = [r for r in results if r is not None]
    with open(out_fasta, "w") as f_out:
        f_out.write("\n".join(valid_entries) + "\n")

    print(f"[✓] {split} → {len(valid_entries)} sequences written to {out_fasta}")

# === Run all splits ===
for split, pickle_file in LABEL_PICKLES.items():
    full_pickle_path = os.path.join(LABEL_DIR, pickle_file)
    if not os.path.exists(full_pickle_path):
        print(f"[!] Missing label file: {full_pickle_path}")
        continue
    process_split(split, full_pickle_path)
