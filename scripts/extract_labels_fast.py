import argparse
import os
import gzip
import csv
import multiprocessing
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

# Configuration
MAX_WORKERS = 8  # Optimal for your 64-core system
CHUNK_SIZE = 50  # Chains per batch
OUTPUT_FILE = "binding_labels.txt"
ERROR_FILE = "processing_errors.log"

def process_chunk(args):
    """Process a batch of chains independently"""
    chunk, pdb_dir = args
    parser = MMCIFParser(QUIET=True)
    results = []
    
    for chain_id in chunk:
        try:
            pdb_id, chain_letter = chain_id.split('_')[0].lower(), chain_id.split('_')[2][0]
            cif_file = f"{pdb_dir}/{pdb_id}.cif"
            
            if not os.path.exists(cif_file):
                results.append((chain_id, "MISSING_CIF"))
                continue

            structure = parser.get_structure("cif", cif_file)
            # ... (rest of your processing logic) ...
            
            results.append((chain_id, binary_str))
        except Exception as e:
            results.append((chain_id, f"ERROR:{str(e)}"))
        finally:
            parser = None  # Force cleanup
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_csv', required=True)
    parser.add_argument('--pdb_dir', required=True)
    args = parser.parse_args()

    # Prepare task queue
    with open(args.annotation_csv) as f:
        all_chains = [row[0] for row in csv.reader(f, delimiter=':') if row]
    
    # Split into batches
    chunks = [all_chains[i:i + CHUNK_SIZE] 
             for i in range(0, len(all_chains), CHUNK_SIZE)]
    
    # Process with worker pool
    with multiprocessing.Pool(MAX_WORKERS) as pool:
        results = pool.imap_unordered(
            process_chunk, 
            [(chunk, args.pdb_dir) for chunk in chunks],
            chunksize=10
        )
        
        with open(OUTPUT_FILE, 'a') as out, open(ERROR_FILE, 'a') as err:
            for batch in results:
                for chain_id, result in batch:
                    if result.startswith("ERROR") or result == "MISSING_CIF":
                        err.write(f"{chain_id}\t{result}\n")
                    else:
                        out.write(f"{chain_id} {result}\n")
                out.flush()
                err.flush()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
