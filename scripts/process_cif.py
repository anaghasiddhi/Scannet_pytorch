import os
import traceback
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

# Configuration
CIF_FILE = "/home/scratch1/asiddhi/benchmarking_model2/ScanNet/datasets/BioLip/structures/7dr2.cif"
CHAINS_TO_PROCESS = ["7dr2_1_A", "7dr2_1_B"]
DISTANCE_CUTOFF = 4.5
METAL_ATOMS = {'LI','BE','NA','MG','AL','K','CA','SC','TI','V','CR','MN','FE','CO','NI','CU','ZN','GA','RB','SR','Y','ZR','MO','TC','RU','RH','PD','AG','CD','IN','SN','SB','CS','BA','LA','CE','PR','ND','PM','SM','EU','GD','TB','DY','HO','ER','TM','YB','LU','HF','TA','W','RE','OS','IR','PT','AU','HG','TL','PB','BI','PO','FR','RA','AC','TH','PA','U','NP','PU','AM','CM','BK','CF','ES','FM','MD','NO','LR','RF','DB','SG','BH','HS','MT','DS','RG','CN','NH','FL','MC','LV','TS','OG'}

def process_chain(chain, ligand_ns, metal_ns):
    """Generate binary labels for a single chain"""
    labels = []
    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
            
        residue_bound = False
        for atom in residue:
            if ligand_ns and ligand_ns.search(atom.coord, DISTANCE_CUTOFF, level="A"):
                residue_bound = True
                break
            if metal_ns and metal_ns.search(atom.coord, DISTANCE_CUTOFF, level="A"):
                residue_bound = True
                break
        labels.append("1" if residue_bound else "0")
    return " ".join(labels)

print(f"Testing CIF parsing: {CIF_FILE}")
print(f"File size: {os.path.getsize(CIF_FILE)/1e6:.1f} MB")

# Attempt 1: Try standard parsing
try:
    print("\nAttempt 1: Standard MMCIFParser")
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("7dr2", CIF_FILE)
    print("✅ Successfully parsed structure")
    
    # Process model 1
    if 1 in structure:
        model = structure[1]
        print(f"Model 1 contains {len(model)} chains")
        
        # Process chains
        for chain_id in CHAINS_TO_PROCESS:
            _, _, chain_code = chain_id.split('_')
            if chain_code in model:
                chain = model[chain_code]
                print(f"\nProcessing chain {chain_code}:")
                print(f"  Residues: {len(list(chain.get_residues()))}")
                print(f"  Amino acids: {len([r for r in chain if is_aa(r, standard=True)])}")
            else:
                print(f"⚠️ Chain {chain_code} not found in model 1")
    else:
        print("⚠️ Model 1 not found in structure")

except Exception as e:
    print(f"❌ Parser failed: {type(e).__name__}: {str(e)}")
    traceback.print_exc()

# Attempt 2: Try with explicit file handle
try:
    print("\nAttempt 2: MMCIFParser with file handle")
    with open(CIF_FILE, 'r') as cif_handle:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("7dr2", cif_handle)
        print("✅ Successfully parsed with file handle")
except Exception as e:
    print(f"❌ File handle parsing failed: {type(e).__name__}: {str(e)}")
    traceback.print_exc()

# Attempt 3: Try alternative parser (MMCIF2)
try:
    print("\nAttempt 3: MMCIF2 parser (if available)")
    from Bio.PDB.MMCIF2 import MMCIF2
    parser = MMCIF2(QUIET=True)
    structure = parser.get_structure("7dr2", CIF_FILE)
    print("✅ Successfully parsed with MMCIF2")
except ImportError:
    print("⚠️ MMCIF2 not available in this BioPython version")
except Exception as e:
    print(f"❌ MMCIF2 parser failed: {type(e).__name__}: {str(e)}")
    traceback.print_exc()

# Attempt 4: Full processing simulation
print("\nAttempt 4: Full processing simulation")
try:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("7dr2", CIF_FILE)
    model = structure[1]
    
    # Collect ligands/metals
    ligand_atoms = []
    metal_atoms = []
    for residue in model.get_residues():
        resname = residue.get_resname().strip()
        if resname in ["HOH", "WAT"]:
            continue
        for atom in residue:
            if atom.element.upper() in METAL_ATOMS:
                metal_atoms.append(atom)
            elif residue.id[0] != " ":
                ligand_atoms.append(atom)
    
    # Create neighbor searches
    ligand_ns = NeighborSearch(ligand_atoms) if ligand_atoms else None
    metal_ns = NeighborSearch(metal_atoms) if metal_atoms else None
    
    # Process chains
    for chain_id in CHAINS_TO_PROCESS:
        _, _, chain_code = chain_id.split('_')
        if chain_code not in model:
            print(f"⚠️ Chain {chain_code} not found")
            continue
        
        chain = model[chain_code]
        binary_str = process_chain(chain, ligand_ns, metal_ns)
        print(f"Processed {chain_id}: {binary_str[:50]}... (length: {len(binary_str.split())})")
        
except Exception as e:
    print(f"❌ Full processing failed: {type(e).__name__}: {str(e)}")
    traceback.print_exc()

print("\nTest completed.")
