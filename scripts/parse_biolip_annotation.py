import os
import pickle
from scannet_pytorch.preprocessing.protein_chemistry import has_complete_ca
from scannet_pytorch.preprocessing.PDBio import getPDB, parse_structure
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from Bio.PDB import MMCIFParser, PDBParser
from scannet_pytorch.preprocessing import PDB_processing


biolip_dir = os.path.join("datasets", "BioLip")
input_file = os.path.join(biolip_dir, "BioLiP_annotation.csv")
output_file = os.path.join(biolip_dir, "biolip_subset.pkl")

if not os.path.exists(input_file):
    raise FileNotFoundError(f"{input_file} not found!")

output_data = []
skipped = 0

with open(input_file, "r") as f:
    for line in f:
        if ":" not in line:
            print("[SKIP] Malformed line")
            continue

        full_id, binding_str = line.rsplit(":", 1)
        binding_res = binding_str.strip().split()

        pdb_id_chain = full_id.split("_")
        pdb_id = pdb_id_chain[0]
        chain_id = pdb_id_chain[-1]
 
        if len(chain_id) != 1 or not chain_id.isalpha():
            print(f"[SKIP] Skipping invalid chain ID: {chain_id}")
            skipped += 1
            continue

        # Download and parse structure
        structure_path, _ = getPDB(pdb_id, biounit=True, structures_folder=os.path.join(biolip_dir, "structures"))
        with open(structure_path, "r") as f:
            content = f.read()
            if "_atom_site" not in content:
                print(f"[SKIP] Corrupt or incomplete CIF: {pdb_id}")
                os.remove(structure_path)  # so it redownloads clean next time
                skipped += 1
                continue
        if not structure_path or not os.path.exists(structure_path):
            print(f"[SKIP] Missing PDB file for {pdb_id}")
            skipped += 1
            continue
        parser = MMCIFParser(QUIET=True) if structure_path.endswith(".cif") else PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, structure_path)

        if chain_id not in structure[0]:
            print(f"[SKIP] Chain {chain_id} not found")
            skipped += 1
            continue
        chain_obj = structure[0][chain_id]
     
        try:
            sequence, backbone_coordinates, atomic_coordinates, atom_ids, atom_types = (
                PDB_processing.process_chain(chain_obj)
             )
        except Exception as e:
            print(f"[SKIP] process_chain failed: {e}")
            skipped += 1
            continue


        if chain_id not in structure[0]:
            print(f"[SKIP] Chain {chain_id} not in {pdb_id}")
            skipped += 1
            continue

        chain = structure[0][chain_id]
        if len(chain) > 1000:
            print(f"[SKIP] Chain too long in {pdb_id}")
            skipped += 1
            continue

        if not has_complete_ca(structure, chain_id):
            print(f"[SKIP] Missing CA atoms in {pdb_id}")
            skipped += 1
            continue

        # ✅ Valid entry
        print(f"[Saved] pdb id: {pdb_id}")
        output_data.append((pdb_id, chain_id, binding_res))

# Save final list
with open(output_file, "wb") as f:
    pickle.dump(output_data, f)

print(f"\n✅ Saved {len(output_data)} BioLiP entries → {output_file}")
print(f"⚠️ Skipped {skipped} entries due to issues\n")
