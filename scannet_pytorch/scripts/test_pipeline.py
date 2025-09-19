import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

import os
import traceback
import pickle
import numpy as np
from scannet_pytorch.preprocessing import PDBio, pipelines
from scannet_pytorch.utilities import io_utils

# === Setup (overridden by .pkl debug dump) ===
pkl_path = "debug_dump_1a0s_0-P.pkl"  # Change this path if testing another file

# Comment out or remove these (no longer needed)
# pdb_id = '1x87_0-A'
# chain_id = 'A'
# label_file = os.path.join("datasets", "PPBS", "labels_train.txt")

# === Load debug dump instead of reading labels and chains ===

pkl_path = "debug_dump_1a0s_0-P.pkl"  # Change this to whichever file failed

with open(pkl_path, "rb") as f:
    debug = pickle.load(f)

pdb_id = debug["pdb_id"]
labels = debug["labels"]
chains = debug["chains"]

# Optional: print chain IDs if you're unsure which one to select
print(f"🔍 Chains available in debug dump for {pdb_id}: {[c.id for c in chains]}")

# Guess chain ID from PDB ID (same logic as Dataset)
chain_id = pdb_id.split("-")[-1]
try:
    chain_obj = next(c for c in chains if c.id == chain_id)
except StopIteration:
    raise ValueError(f"[ERROR] Chain {chain_id} not found in debug dump for {pdb_id}")

# Reshape label if needed
if labels.ndim == 1:
    labels = labels[:, None]
print(f"✅ Loaded labels from .pkl: shape = {labels.shape}, dtype = {labels.dtype}")



# Initialize ScanNet pipeline
pipeline = pipelines.ScanNetPipeline(
    with_aa=True,
    with_atom=True,
    aa_features='sequence',
    atom_features='valency',
    padded=False,
    Lmax_aa=800
)

# Run processing
try:
    print("Running process_example on real data...")
    print("🧪 About to call process_example()")
    result = pipeline.process_example(chain_obj=chain_obj, labels=target_labels)

    if result is None:
        print(f"❌ process_example returned None for {pdb_id}_{chain_id}")
        exit(1)

    inputs, labels = result
    print("✅ process_example returned without crashing")
    print("inputs type:", type(inputs))
    for i, item in enumerate(inputs):
        print(f"  inputs[{i}] shape:", None if item is None else getattr(item, "shape", type(item)))
    print("labels type:", type(labels))
    print("labels shape:", None if labels is None else getattr(labels, "shape", None))

except Exception as e:
    print(f"❌ Unexpected exception for {pdb_id}_{chain_id}")
    traceback.print_exc()
