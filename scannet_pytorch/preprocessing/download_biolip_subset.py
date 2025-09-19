import os
import csv
import urllib.request
import pickle
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
import sys
from tqdm import tqdm  # For progress bars

# ===== CONFIGURATION =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BIOLIP_DIR = os.path.join(PROJECT_ROOT, "datasets", "BioLip")
STRUCTURES_DIR = os.path.join(BIOLIP_DIR, "structures")
OUTPUT_FILE = os.path.join(BIOLIP_DIR, "biolip_subset.pkl")
MAX_RESIDUES = 1000
# =========================

def has_complete_ca(structure, chain_id):
    """Check if chain has complete Cα trace for all amino acid residues"""
    model = structure[0]  # First model
    if chain_id not in model:
        return False
        
    chain = model[chain_id]
    
    for residue in chain:
        # Skip non-amino acid residues (heteroatoms/water)
        if residue.get_id()[0] != " ":
            continue
            
        # Check for Cα atom
        if not residue.has_id("CA"):
            return False
            
    return True

def download_structure(pdb_id, format="cif"):
    """Download structure from RCSB"""
    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    filename = f"{pdb_id}.{format}"
    filepath = os.path.join(STRUCTURES_DIR, filename)
    
    # Skip download if file already exists
    if os.path.exists(filepath):
        return filepath
    
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.{format}"
        urllib.request.urlretrieve(url, filepath)
        return filepath
    except Exception as e:
        print(f"❌ Download failed for {pdb_id}: {str(e)}")
        return None

def is_valid_chain(chain_id):
    """Validate chain ID format"""
    return len(chain_id) == 1 and chain_id.isalpha()

def process_pdb(pdb_id, chain_id):
    """Process a single PDB chain and validate against all criteria"""
    # Validate chain ID format
    if not is_valid_chain(chain_id):
        print(f"Skipping {pdb_id}_{chain_id}: Invalid chain ID")
        return False

    # First try CIF format
    cif_file = download_structure(pdb_id, "cif")
    structure = None
    
    if cif_file:
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(pdb_id, cif_file)
        except Exception:
            # CIF parsing failed, try PDB format
            pdb_file = download_structure(pdb_id, "pdb")
            if pdb_file:
                try:
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure(pdb_id, pdb_file)
                except Exception as e:
                    print(f"❌ PDB parsing failed: {pdb_id} - {str(e)}")
    
    if not structure:
        print(f"❌ Failed to parse structure: {pdb_id}")
        return False

    # Check chain exists
    model = structure[0]
    if chain_id not in model:
        print(f"❌ Chain not found: {pdb_id}_{chain_id}")
        return False

    # Check chain length
    chain = model[chain_id]
    residues = list(chain.get_residues())
    if len(residues) > MAX_RESIDUES:
        print(f"❌ Chain too long: {pdb_id}_{chain_id} ({len(residues)} residues)")
        return False

    # Check complete Cα trace
    if not has_complete_ca(structure, chain_id):
        print(f"❌ Incomplete Cα trace: {pdb_id}_{chain_id}")
        return False

    return True

def main():
    # Create directories if they don't exist
    os.makedirs(BIOLIP_DIR, exist_ok=True)
    os.makedirs(STRUCTURES_DIR, exist_ok=True)
    
    # Find annotation file
    annotation_path = os.path.join(BIOLIP_DIR, "BioLiP_annotation.csv")
    
    # Verify annotation file exists
    if not os.path.exists(annotation_path):
        # Print directory contents for debugging
        print(f"Directory contents of {BIOLIP_DIR}:")
        for f in os.listdir(BIOLIP_DIR):
            print(f" - {f}")
            
        raise FileNotFoundError(
            f"Annotation file not found in {BIOLIP_DIR}\n"
            f"Please ensure your BioLiP annotation file exists in this directory."
        )

    print(f"Using annotation file: {annotation_path}")

    # Read and parse annotation file
    valid_pairs = []
    seen_pairs = set()
    total_entries = 0
    valid_count = 0
    error_count = 0

    # First count total lines for progress bar
    with open(annotation_path, "r", encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1  # Subtract header row
        
    print(f"\nProcessing {total_lines} entries from BioLiP annotation file...")
    
    with open(annotation_path, "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        # Initialize progress bar
        pbar = tqdm(total=total_lines, desc="Processing entries", unit="entry")
        
        for row in reader:
            total_entries += 1
            pbar.update(1)
            
            # Skip empty rows
            if not row or not row[0].strip():
                continue
                
            # Each row has only one column with the complex structure
            complex_str = row[0].strip()
            
            # Extract PDB ID and chain ID from the first part (before colon)
            if ':' in complex_str:
                complex_id = complex_str.split(':', 1)[0]
            else:
                complex_id = complex_str
                
            # Split the complex ID by underscores
            parts = complex_id.split('_')
            if len(parts) < 3:
                error_count += 1
                continue
                
            # PDB ID is the first part (4 characters)
            pdb_id = parts[0][:4].lower()
            # Chain ID is the last part
            chain_id = parts[-1]
            
            # Skip invalid chain IDs
            if not is_valid_chain(chain_id):
                error_count += 1
                continue
                
            pair = (pdb_id, chain_id)
            
            # Process each unique pair only once
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            try:
                if process_pdb(pdb_id, chain_id):
                    valid_pairs.append(pair)
                    valid_count += 1
            except Exception as e:
                print(f"❌ Error processing {pdb_id}_{chain_id}: {str(e)}")
                error_count += 1
                
        pbar.close()

    # Save valid pairs
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(valid_pairs, f)
        
    # Print comprehensive summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total entries processed:   {total_entries}")
    print(f"Unique (PDB, chain) pairs: {len(seen_pairs)}")
    print(f"Valid chains found:        {valid_count}")
    print(f"Entries with errors:       {error_count}")
    print(f"Validation success rate:   {valid_count/len(seen_pairs)*100:.2f}%")
    print(f"\nResults saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
