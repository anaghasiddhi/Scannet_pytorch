import os
import sys
import csv
import time
import argparse
import logging
import signal
import gc
from functools import lru_cache
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from logging.handlers import RotatingFileHandler
import concurrent.futures
import psutil

# Configuration
MAX_WORKERS = 16
CHUNK_SIZE = 100
OUTPUT_FILE = "labels_biolip.txt"
ERROR_FILE = "processing_errors.log"
DONE_DIR = "processed_chains"
DISTANCE_CUTOFF = 4.5
METAL_CUTOFF = 4.0
MAX_CHUNK_TIME = 600  # 5 minutes per chunk timeout
METAL_ATOMS = {'LI','BE','NA','MG','AL','K','CA','SC','TI','V','CR','MN','FE','CO','NI','CU','ZN','GA','RB','SR','Y','ZR','MO','TC','RU','RH','PD','AG','CD','IN','SN','SB','CS','BA','LA','CE','PR','ND','PM','SM','EU','GD','TB','DY','HO','ER','TM','YB','LU','HF','TA','W','RE','OS','IR','PT','AU','HG','TL','PB','BI','PO','AT','RN','FR','RA','AC','TH','PA','U','NP','PU','AM','CM','BK','CF','ES','FM','MD','NO','LR'}
PTM_WHITELIST = {'MSE','SEP','TPO','PTR','CSO','PCA','KCX','CME','CSD','OCS','ALY','MLY','FME','HYP','LLP','M3L','TYS'}

# Setup logging
logger = logging.getLogger("pdb_processor")
logger.setLevel(logging.INFO)
error_handler = RotatingFileHandler(ERROR_FILE, maxBytes=10*1024*1024, backupCount=5)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(error_handler)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Processing timed out")

@lru_cache(maxsize=100)
def parse_cif_structure(cif_file, pdb_id):
    """Parse CIF file with caching and timeout"""
    parser = MMCIFParser(QUIET=True)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60-second timeout
    try:
        structure = parser.get_structure(pdb_id, cif_file)
        signal.alarm(0)  # Disable alarm
        return structure
    except TimeoutException:
        signal.alarm(0)
        logger.error(f"Timeout processing {cif_file}")
        raise
    except Exception as e:
        signal.alarm(0)
        logger.error(f"Error parsing {cif_file}: {str(e)}")
        raise
    finally:
        signal.alarm(0)

def load_processed_chains():
    """Load already processed chains from done markers"""
    processed = set()
    if not os.path.exists(DONE_DIR):
        return processed
        
    for fname in os.listdir(DONE_DIR):
        if fname.endswith(".done"):
            processed.add(fname[:-5])
    return processed

def process_chain(chain, ligand_ns, metal_ns, ligand_cutoff, metal_cutoff):
    """Generate binary labels with strict residue order"""
    labels = []
    for residue in chain:
        if not is_aa(residue, standard=True):
            continue

        residue_bound = False
        for atom in residue:
            if ligand_ns and ligand_ns.search(atom.coord, ligand_cutoff, level="A"):
                residue_bound = True
                break
            if metal_ns and metal_ns.search(atom.coord, metal_cutoff, level="A"):
                residue_bound = True
                break
        labels.append("1" if residue_bound else "0")
    return " ".join(labels)

def process_chunk(args):
    """Process a batch of chains with all fixes implemented"""
    try:
        chunk, pdb_dir, ligand_cutoff, metal_cutoff = args
        logger.info(f"Starting chunk with {len(chunk)} chains: {chunk[0]}..{chunk[-1]}")
        results = []
        pdb_model_groups = {}

        for chain_id in chunk:
            parts = chain_id.split('_')
            if len(parts) < 3:
                results.append((chain_id, f"ERROR:Invalid chain_id format '{chain_id}'"))
                continue

            pdb_id = parts[0].lower()
            model_id = parts[1]
            chain_code = parts[2]
            key = (pdb_id, model_id)
            
            if key not in pdb_model_groups:
                pdb_model_groups[key] = []
            pdb_model_groups[key].append((chain_id, chain_code))

        # Process each PDB/model group
        for (pdb_id, model_id), chains in pdb_model_groups.items():
            cif_file = None
            for ext in [f"{pdb_id}.cif", f"{pdb_id.upper()}.cif"]:
                candidate = os.path.join(pdb_dir, ext)
                if os.path.exists(candidate):
                    cif_file = candidate
                    break

            if not cif_file:
                for chain_id, _ in chains:
                    results.append((chain_id, f"MISSING_CIF: {pdb_id}"))
                continue

            try:
                structure = parse_cif_structure(cif_file, pdb_id)
                model = None
                model_candidates = [model_id, str(model_id)]
                
                for candidate in model_candidates:
                    if candidate in structure:
                        model = structure[candidate]
                        break
                        
                if model is None and len(structure) > 0:
                    model = next(structure.get_models())
                    logger.warning(f"Using first model (ID={model.id}) for {pdb_id}_{model_id}")

                if model is None:
                    for chain_id, _ in chains:
                        results.append((chain_id, f"MISSING_MODEL: {model_id}"))
                    continue

                # Collect ligands and metals
                ligand_atoms = []
                metal_atoms = []
                for residue in model.get_residues():
                    resname = residue.get_resname().strip()
                    if resname in ["HOH", "WAT", "H2O", "DOD"]:
                        continue

                    for atom in residue:
                        element = atom.element.upper()
                        if element in METAL_ATOMS:
                            metal_atoms.append(atom)
                        elif residue.id[0] != " " and resname not in PTM_WHITELIST:
                            ligand_atoms.append(atom)

                ligand_ns = NeighborSearch(ligand_atoms) if ligand_atoms else None
                metal_ns = NeighborSearch(metal_atoms) if metal_atoms else None

                # Process chains
                for chain_id, chain_code in chains:
                    chain = None
                    chain_candidates = [
                        chain_code,
                        chain_code.upper(),
                        chain_code.lower(),
                        'a' + chain_code,
                        'a' + chain_code.upper(),
                        'a' + chain_code.lower(),
                        'b' + chain_code,
                        'b' + chain_code.upper(),
                        'b' + chain_code.lower()
                    ]
                    
                    for candidate in chain_candidates:
                        if candidate in model:
                            chain = model[candidate]
                            break
                    
                    if chain is None:
                        available_chains = list(model.child_dict.keys())
                        results.append((chain_id, 
                                    f"MISSING_CHAIN: Requested '{chain_code}' | Available: {available_chains}"))
                        continue
                    
                    # Check for amino acids
                    if not any(is_aa(res) for res in chain):
                        results.append((chain_id, "ERROR: No amino acid residues found"))
                        continue
                    
                    try:
                        binary_str = process_chain(chain, ligand_ns, metal_ns, ligand_cutoff, metal_cutoff)
                        results.append((chain_id, binary_str))
                    except Exception as e:
                        results.append((chain_id, f"ERROR:{type(e).__name__}: {str(e)}"))
                
            except Exception as e:
                for chain_id, _ in chains:
                    results.append((chain_id, f"ERROR:{type(e).__name__}: {str(e)}"))
        
        logger.info(f"Completed chunk: {chunk[0]}..{chunk[-1]}")
        return results
    except Exception as e:
        # Catch-all for any unhandled exceptions in the chunk
        return [(chain_id, f"FATAL:{type(e).__name__}:{str(e)}") for chain_id in chunk]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_csv', required=True, help='Path to BioLiP annotation CSV file')
    parser.add_argument('--pdb_dir', required=True, help='Directory containing PDB/CIF files')
    parser.add_argument('--resume', action='store_true', help='Skip already processed chains')
    parser.add_argument('--ligand_cutoff', type=float, default=DISTANCE_CUTOFF, help='Ligand binding cutoff distance')
    parser.add_argument('--metal_cutoff', type=float, default=METAL_CUTOFF, help='Metal binding cutoff distance')
    args = parser.parse_args()

    os.makedirs(DONE_DIR, exist_ok=True)
    start_time = time.time()
    processed_chains = load_processed_chains() if args.resume else set()

    # Read chains from annotation file
    with open(args.annotation_csv) as f:
        all_chains = [row[0].strip() for row in csv.reader(f, delimiter=':') if row]

    if args.resume:
        initial_count = len(all_chains)
        all_chains = [chain for chain in all_chains if chain not in processed_chains]
        print(f"Resuming: {len(all_chains)} chains remaining ({initial_count - len(all_chains)} skipped)")

    if not all_chains:
        print("No chains to process!")
        return

    # Create chunks
    chunks = [all_chains[i:i + CHUNK_SIZE] for i in range(0, len(all_chains), CHUNK_SIZE)]
    total_chains = len(all_chains)
    print(f"Processing {total_chains} chains in {len(chunks)} chunks...")

    # Prepare file handles
    file_mode = 'a' if args.resume and os.path.exists(OUTPUT_FILE) else 'w'
    processed_count = 0
    last_report_time = start_time

    # Use ProcessPoolExecutor for per-chunk timeouts
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all chunks to the executor
        future_to_chunk = {
            executor.submit(process_chunk, (chunk, args.pdb_dir, args.ligand_cutoff, args.metal_cutoff)): chunk
            for chunk in chunks
        }
        
        with open(OUTPUT_FILE, file_mode) as out, open(ERROR_FILE, 'a' if args.resume else 'w') as err:
            # Process completed futures
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                current_time = time.time()
                
                try:
                    # Get results with timeout protection
                    batch = future.result(timeout=MAX_CHUNK_TIME)
                except concurrent.futures.TimeoutError:
                    # Handle chunk-level timeout
                    logger.error(f"Chunk timed out: {chunk[0]}..{chunk[-1]}")
                    batch = [(chain_id, "TIMEOUT: Chunk processing exceeded time limit") for chain_id in chunk]
                except Exception as e:
                    # Handle any other executor exceptions
                    logger.error(f"Executor error: {str(e)}")
                    batch = [(chain_id, f"EXECUTOR_ERROR: {str(e)}") for chain_id in chunk]
                
                # Process results
                for chain_id, result in batch:
                    if result.startswith(("ERROR", "MISSING", "TIMEOUT", "FATAL", "EXECUTOR")):
                        err.write(f"{chain_id}\t{result}\n")
                    else:
                        out.write(f"{chain_id} {result}\n")
                    
                    # Create done marker
                    open(os.path.join(DONE_DIR, f"{chain_id}.done"), 'w').close()

                # Update counters and report progress
                batch_size = len(batch)
                processed_count += batch_size
                
                if current_time - last_report_time >= 30:  # Report more frequently
                    elapsed = current_time - start_time
                    chains_per_sec = processed_count / elapsed
                    remaining = total_chains - processed_count
                    eta = remaining / chains_per_sec if chains_per_sec > 0 else float('inf')
                    
                    # Add memory usage to report
                    mem = psutil.virtual_memory()
                    print(f"Processed: {processed_count}/{total_chains} "
                          f"({processed_count/total_chains*100:.1f}%) | "
                          f"Rate: {chains_per_sec:.2f} chains/sec | "
                          f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))} | "
                          f"Mem: {mem.percent}% used")
                    last_report_time = current_time
                
                # Memory management
                gc.collect()

    # Sort output files
    for fname in [OUTPUT_FILE, ERROR_FILE]:
        if os.path.exists(fname):
            with open(fname) as f:
                lines = sorted(f.readlines(), key=lambda x: x.split('\t')[0])
            with open(fname, 'w') as f:
                f.writelines(lines)

    # Final report
    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time/3600:.2f} hours")
    print(f"Results: {OUTPUT_FILE}")
    print(f"Errors: {ERROR_FILE}")

if __name__ == "__main__":
    # Windows-safe freeze support
    if sys.platform.startswith('win'):
        from multiprocessing import freeze_support
        freeze_support()
    main()
