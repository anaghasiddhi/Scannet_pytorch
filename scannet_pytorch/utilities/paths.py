import os

# ====== ROOT PROJECT FOLDER ======
library_folder = os.getenv("SCANNET_LIB", "/home/scratch1/asiddhi/benchmarking_model2/ScanNet/scannet_pytorch")

# ====== CORE DATA + MODEL PATHS ======
structures_folder = os.path.join(library_folder, "..", "PDB")  # one level up
predictions_folder = os.path.join(library_folder, "..", "predictions")
model_folder = os.path.join(library_folder, "..", "models")
pipeline_folder = os.path.join(library_folder, "..", "data")
initial_values_folder = os.path.join(model_folder, "initial_values")
visualization_folder = os.path.join(library_folder, "..", "visualizations")

# ====== MSA PATHS (OPTIONAL FOR ScanNet_noMSA) ======
MSA_folder = os.path.join(library_folder, "..", "MSA")
path2hhblits = os.getenv("HHBLITS_PATH", None)
path2sequence_database = os.getenv("HHBLITS_DB", None)

# ====== BASELINES ======
path_to_dssp = os.getenv("DSSP_PATH", "/path/to/mkdssp")
path_to_msms = os.getenv("MSMS_PATH", "/path/to/msms.x86_64Linux2.2.6.1")
homology_folder = os.path.join(library_folder, "..", "baselines", "homology")
path_to_multiprot = os.path.join(homology_folder, "multiprot.Linux")

# ====== EXPORTS ======
__all__ = [
    "library_folder",
    "structures_folder",
    "predictions_folder",
    "model_folder",
    "pipeline_folder",
    "initial_values_folder",
    "visualization_folder",
    "MSA_folder",
    "path2hhblits",
    "path2sequence_database",
    "path_to_dssp",
    "path_to_msms",
    "homology_folder",
    "path_to_multiprot"
]
