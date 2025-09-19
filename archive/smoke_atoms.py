import numpy as np, glob, time, os
from Bio.PDB import MMCIFParser

STRUCTURES_DIR = os.environ.get("BIOLIP_STRUCTURES",
                                os.path.join("datasets", "BioLip", "structures"))

files = sorted(glob.glob("data/test.data/*.npy"))[:100]
parser = MMCIFParser(QUIET=True)

start = time.time()
processed = 0
for f in files:
    data = np.load(f, allow_pickle=True).item()
    pdb_id = f.split("/")[-1].split(".")[0]  # e.g. "1a01_A"
    base = pdb_id.split("_")[0].lower()      # normalize, matches CIF name

    cif = os.path.join(STRUCTURES_DIR, f"{base}.cif")
    if not os.path.exists(cif):
        print("Skip (no CIF):", pdb_id)
        continue

    try:
        structure = parser.get_structure(base, cif)
        chain_id = pdb_id.split("_")[1]
        if chain_id not in structure[0]:
            print("Skip (no chain):", pdb_id)
            continue

        chain = structure[0][chain_id]
        coords, atom_types, res_idx = [], [], []
        for residue in chain:
            for atom in residue:
                coords.append(atom.coord)
                atom_types.append(atom.element)
                res_idx.append(residue.id[1])

        data["coord_atom"] = np.array(coords, dtype=np.float32)
        data["indices_atom"] = np.array(res_idx, dtype=np.int32)
        processed += 1
    except Exception as e:
        print("Skip:", pdb_id, e)

end = time.time()
if processed:
    print(f"Processed {processed} entries in {end - start:.2f} s")
    print(f"Average per entry: {(end - start)/processed:.3f} s")
else:
    print("No entries processed.")
