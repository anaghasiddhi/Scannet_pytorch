import os
import urllib.request
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

# ===== CONFIGURATION =====
PDB_ID = "101m"  # Known test case from your BioLiP example
CHAIN_ID = "A"   # Known chain ID from your example
STRUCTURES_DIR = "test_structures"
MAX_RESIDUES = 1000
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"  # Start with CIF
# =========================

def has_complete_ca(structure, chain_id):
    """Check if chain has complete Cα trace for all amino acid residues"""
    model = structure[0]  # First model
    if chain_id not in model:
        print(f"❌ Chain {chain_id} not found in structure")
        return False
        
    chain = model[chain_id]
    missing_ca = 0
    total_residues = 0
    
    for residue in chain:
        # Skip non-amino acid residues (heteroatoms/water)
        if residue.get_id()[0] != " ":
            continue
            
        total_residues += 1
        # Check for Cα atom
        if not residue.has_id("CA"):
            missing_ca += 1
            print(f"  ❌ Missing Cα in residue {residue.id[1]} ({residue.resname})")
            
    if missing_ca > 0:
        print(f"❌ Missing Cα in {missing_ca}/{total_residues} residues")
        return False
        
    print(f"✅ Complete Cα trace: {total_residues} residues")
    return True

def download_structure(pdb_id, format="cif"):
    """Download structure from RCSB"""
    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    filename = f"{pdb_id}.{format}"
    filepath = os.path.join(STRUCTURES_DIR, filename)
    
    print(f"\n=== Downloading {pdb_id} ({format.upper()} format) ===")
    try:
        urllib.request.urlretrieve(
            RCSB_URL.format(pdb_id=pdb_id).replace(".cif", f".{format}"),
            filepath
        )
        print(f"✅ Downloaded to: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return None

def validate_structure(pdb_id, chain_id):
    """Validate a single PDB chain against all criteria"""
    print(f"\n{'='*50}")
    print(f"Validating: {pdb_id} chain {chain_id}")
    print(f"{'='*50}")
    
    # Validate chain ID format
    if len(chain_id) != 1 or not chain_id.isalpha():
        print(f"❌ Invalid chain ID: '{chain_id}'")
        return False
    
    # First try CIF format
    cif_file = download_structure(pdb_id, "cif")
    if not cif_file:
        return False
    
    # Try parsing directly instead of checking content
    print("\n=== Attempting to parse CIF file ===")
    try:
        cif_parser = MMCIFParser(QUIET=False)
        cif_structure = cif_parser.get_structure(pdb_id, cif_file)
        print(f"✅ CIF parsed successfully: {len(cif_structure)} models")
        return cif_structure
    except KeyError as e:
        if "_atom_site" in str(e):
            print(f"⚠️ CIF parsing failed: {e}")
            print("This usually indicates missing atomic coordinates")
        else:
            print(f"❌ CIF parsing failed: {e}")
    except PDBConstructionException as e:
        print(f"❌ CIF parsing failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected CIF parsing error: {e}")
    
    # If CIF parsing failed, try PDB format
    print("\n=== Trying PDB format ===")
    pdb_file = download_structure(pdb_id, "pdb")
    if not pdb_file:
        return None
    
    print("\n=== Attempting to parse PDB file ===")
    try:
        pdb_parser = PDBParser(QUIET=False)
        pdb_structure = pdb_parser.get_structure(pdb_id, pdb_file)
        print(f"✅ PDB parsed successfully: {len(pdb_structure)} models")
        return pdb_structure
    except PDBConstructionException as e:
        print(f"❌ PDB parsing failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected PDB parsing error: {e}")
    
    return None

def validate_chain(structure, pdb_id, chain_id):
    """Validate chain meets all requirements"""
    if not structure:
        return False
    
    print("\n=== Validating chain ===")
    # Check chain exists
    model = structure[0]
    if chain_id not in model:
        print(f"❌ Chain {chain_id} not found in structure")
        print(f"Available chains: {[chain.id for chain in model.get_chains()]}")
        return False
    print(f"✅ Chain found: {chain_id}")
    
    # Check chain length
    chain = model[chain_id]
    residues = list(chain.get_residues())
    print(f"ℹ️ Chain length: {len(residues)} residues")
    
    if len(residues) > MAX_RESIDUES:
        print(f"❌ Chain too long ({len(residues)} > {MAX_RESIDUES} residues)")
        return False
    print("✅ Chain length acceptable")
    
    # Check complete Cα trace
    if not has_complete_ca(structure, chain_id):
        return False
    
    print("\n✅✅✅ ALL VALIDATION CHECKS PASSED ✅✅✅")
    return True

def main():
    print(f"\n{'*'*60}")
    print(f"VALIDATION SCRIPT FOR {PDB_ID}_{CHAIN_ID}")
    print(f"{'*'*60}")
    
    # Step 1: Download and validate structure
    structure = validate_structure(PDB_ID, CHAIN_ID)
    
    # Step 2: Validate specific chain
    if structure:
        is_valid = validate_chain(structure, PDB_ID, CHAIN_ID)
        if is_valid:
            print(f"\nSUCCESS: {PDB_ID}_{CHAIN_ID} is VALID")
        else:
            print(f"\nFAILURE: {PDB_ID}_{CHAIN_ID} is INVALID")
    else:
        print("\nFAILURE: Could not obtain valid structure")

if __name__ == "__main__":
    main()
