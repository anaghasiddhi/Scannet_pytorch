import os
import sys
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

# Configuration
PDB_DIR = "/home/scratch1/asiddhi/benchmarking_model2/ScanNet/datasets/BioLip/structures"
DISTANCE_CUTOFF = 4.5
METAL_ATOMS = {'LI','BE','NA','MG','AL','K','CA','SC','TI','V','CR','MN','FE','CO','NI','CU','ZN','GA','RB','SR','Y','ZR','MO','TC','RU','RH','PD','AG','CD','IN','SN','SB','CS','BA','LA','CE','PR','ND','PM','SM','EU','GD','TB','DY','HO','ER','TM','YB','LU','HF','TA','W','RE','OS','IR','PT','AU','HG','TL','PB','BI','PO','FR','RA','AC','TH','PA','U','NP','PU','AM','CM','BK','CF','ES','FM','MD','NO','LR','RF','DB','SG','BH','HS','MT','DS','RG','CN','NH','FL','MC','LV','TS','OG'}

def process_chain(chain, ligand_ns, metal_ns):
    """Generate binary labels for a single chain"""
    labels = []
    residue_count = 0
    aa_count = 0
    
    for residue in chain:
        residue_count += 1
        # Skip non-amino acid residues
        if not is_aa(residue, standard=True):
            continue
            
        aa_count += 1
        residue_bound = False
        for atom in residue:
            # Check ligand proximity
            if ligand_ns and ligand_ns.search(atom.coord, DISTANCE_CUTOFF, level="A"):
                residue_bound = True
                break
            # Check metal proximity
            if metal_ns and metal_ns.search(atom.coord, DISTANCE_CUTOFF, level="A"):
                residue_bound = True
                break
        labels.append("1" if residue_bound else "0")
    
    return " ".join(labels), residue_count, aa_count

def process_chain_id(chain_id):
    """Process a single chain ID with detailed diagnostics"""
    parts = chain_id.split('_')
    if len(parts) < 3:
        return f"{chain_id}\tERROR: Invalid chain ID format"
    
    pdb_id = parts[0].lower()
    model_id = parts[1]
    chain_code = parts[2]
    cif_file = os.path.join(PDB_DIR, f"{pdb_id}.cif")
    
    # 1. Check file existence
    if not os.path.exists(cif_file):
        return f"{chain_id}\tMISSING_CIF"
    
    try:
        # 2. Parse structure
        with open(cif_file, 'r') as cif_handle:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(pdb_id, cif_handle)
            
            # 3. Find correct model
            model = None
            model_candidates = [
                model_id,               # Original ID
                str(model_id),           # String representation
                int(model_id) if model_id.isdigit() else None,  # Integer
                0, '0',                 # Common fallbacks
                1, '1'
            ]
            
            # Try all candidates
            for candidate in model_candidates:
                if candidate is not None and candidate in structure:
                    model = structure[candidate]
                    break
            
            # Fallback to first model
            if model is None and len(structure) > 0:
                model = next(structure.get_models())
                print(f"  WARNING: Using first model (ID={model.id}) for {chain_id}")
            
            if model is None:
                return f"{chain_id}\tERROR: No model found"
            
            # 4. Find correct chain
            chain = None
            chain_candidates = [
                chain_code,             # Original code
                chain_code.upper(),     # Uppercase
                chain_code.lower(),     # Lowercase
                'a' + chain_code,       # 7dr2-style
                'a' + chain_code.upper(),
                'a' + chain_code.lower()
            ]
            
            # Try all candidates
            for candidate in chain_candidates:
                if candidate in model:
                    chain = model[candidate]
                    break
            
            if chain is None:
                # Try partial match (chain_code in actual ID)
                for actual_id in model.child_dict.keys():
                    if chain_code in actual_id:
                        chain = model[actual_id]
                        print(f"  INFO: Using partial match {actual_id} for {chain_code}")
                        break
            
            if chain is None:
                available_chains = list(model.child_dict.keys())
                return f"{chain_id}\tERROR: Chain not found. Available: {available_chains}"
            
            # 5. Collect ligands/metals
            ligand_atoms = []
            metal_atoms = []
            for residue in model.get_residues():
                resname = residue.get_resname().strip()
                if resname in ["HOH", "WAT", "H2O", "DOD"]:
                    continue
                    
                for atom in residue:
                    if atom.element.upper() in METAL_ATOMS:
                        metal_atoms.append(atom)
                    elif residue.id[0] != " ":
                        ligand_atoms.append(atom)
            
            # 6. Generate labels
            ligand_ns = NeighborSearch(ligand_atoms) if ligand_atoms else None
            metal_ns = NeighborSearch(metal_atoms) if metal_atoms else None
            
            binary_str, residue_count, aa_count = process_chain(chain, ligand_ns, metal_ns)
            
            return (f"{chain_id} {binary_str}",
                    f"  Success: {aa_count} AA residues ({residue_count} total residues)")
    
    except Exception as e:
        return f"{chain_id}\tERROR: {type(e).__name__}: {str(e)}"

if __name__ == "__main__":
    # Entries to test (from labels_biolip.txt)
    chains_to_process = [
        "6u4m_1_A",
        "6u4n_1_B",
        "6xqj_1_A",
        "6xv2_1_A",
        "6xyv_1_A",
        "6y94_1_A",
    ]

    print(f"Testing {len(chains_to_process)} previously labeled chains for validation\n")

    for chain_id in chains_to_process:
        print(f"\nProcessing {chain_id}:")
        result = process_chain_id(chain_id)
        if isinstance(result, tuple):
            print(result[0])
            print(result[1])
        else:
            print(result)
