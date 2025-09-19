import os, sys
#sys.stderr = open(os.devnull, 'w')
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))  # Adds /ScanNet to path
import traceback
import torch
import numpy as np
from scannet_pytorch.datasets import ScanNetDataset
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.utilities import io_utils, paths

from Bio import PDB
from scannet_pytorch.preprocessing.PDBio import getPDB
from scannet_pytorch.preprocessing.PDB_processing import process_chain
from scannet_pytorch.preprocessing.protein_chemistry import has_complete_ca

def load_single_label_and_resids(label_file, target_id, label_type='int'):
    import numpy as np  # Ensure numpy is available inside the function

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].lower() == target_id.lower():
                label_values = [int(x) if label_type == 'int' else float(x) for x in parts[1:]]
                resids = [[target_id.split('_')[-1], str(i+1)] for i in range(len(label_values))]  # [chain, resnum]
                return target_id, resids, np.array(label_values)
    raise ValueError(f"[Error] Label for {target_id} not found in {label_file}")


# === Mini structure sanity check ===
manual_pdb_id = "101m"
manual_chain_id = "A"
full_id = f"{manual_pdb_id}_1_{manual_chain_id}"  # becomes 1ubq_1_A
structure_id = f"{manual_pdb_id}_{manual_chain_id}"  # e.g. "101m_A"
structure_path = os.path.join("PDB", f"{manual_pdb_id}.cif")
structures_folder = "PDB"  # this is the folder containing the .cif files
label_file = "labels_biolip_cleaned.txt"

print(f"\n=== Testing manual structure: {manual_pdb_id}_{manual_chain_id} ===")


try:
    location, _ = getPDB(manual_pdb_id, biounit=False, compressed=False, warn=True, structures_folder=structures_folder)
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure(manual_pdb_id, location)

    model = next(structure.get_models())  # Use iterator
    available_chains = [chain.id for chain in model.get_chains()]
    print("Available chains:", available_chains)

    if manual_chain_id not in available_chains:
        print(f"❌ Chain {manual_chain_id} not found in structure.")
    elif not has_complete_ca(structure, manual_chain_id):
        print(f"❌ Chain {manual_chain_id} is missing Cα atoms.")
    else:
        chain = model[manual_chain_id]
        print(f"✅ Found chain {manual_chain_id}")
        print("✅ Chain has complete Cα trace")
        features = process_chain(chain)
        print("✅ process_chain() succeeded")

except Exception as e:
    print(f"❌ Failed during parsing or processing: {e}")
    traceback.print_exc()

# === Step 1: Load label data ===
print("Loading lable data\n")
pdb_id, resids, label = load_single_label_and_resids(label_file, full_id)

# Sanity
print(f"[DEBUG] Selected ID: {pdb_id}")
print(f"[DEBUG] Label shape: {label.shape}")
print(f"[DEBUG] Resids shape: {len(resids)}")

# Reshape label
if label.ndim == 1:
    label = label[:, None]

#=== Step 2: Build pipeline ===
print("Building Pipeline\n")
pipeline = ScanNetPipeline(
    atom_features="valency",
    aa_features="sequence",
    aa_frames="triplet_sidechain",
    Beff=500
)

# === Step 3: Process through pipeline ===
try:
    print("Fetching processed dataset\n")
    inputs, outputs, _, loaded = pipeline.build_processed_dataset(
        dataset_name="debug_biolip_test",
        list_origins=[pdb_id],
        list_resids=[resids],
        list_labels=[label],
        fresh=True,
        save=False,
        permissive=True
    )

    print("✅ build_processed_dataset succeeded.")
except Exception as e:
    print(f"❌ Failed during build_processed_dataset: {e}")
    traceback.print_exc()
    raise

# === Optional: Try __getitem__ on dataset ===
print("Trying ScannetDataset")
dataset = ScanNetDataset(
    pdb_ids=[pdb_id],
    label_dict={pdb_id: label},
    pipeline=pipeline,
    pdb_folder=paths.structures_folder,
    max_length=800
)

print(f"\n[Dataset] Loaded: {dataset.loaded}, Skipped: {dataset.skipped}")
print(f"[Dataset] Skipped IDs: {dataset.skipped_ids}")

try:
    result = dataset[0]
    if result is None:
        print("❌ __getitem__ returned None (likely due to skipped structure or processing error).")
    else:
        sample_input, sample_target = result
        print("✅ __getitem__ succeeded.")
        print("Input shapes:", [x.shape if isinstance(x, torch.Tensor) else type(x) for x in sample_input])
        print("Target shape:", sample_target.shape)
except Exception as e:
    print(f"❌ __getitem__ failed: {e}")

