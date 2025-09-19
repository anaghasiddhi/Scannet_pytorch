import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # CRITICAL FIX - MUST BE FIRST
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import pickle
from Bio.PDB import MMCIFParser, PDBParser
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import urllib.request
import socket
import time
import logging
import traceback
from collections import defaultdict
import threading
import gzip
import shutil
import multiprocessing
import psutil
import signal

# ===== CONFIGURATION =====
NUM_DOWNLOAD_WORKERS = 16
NUM_PROCESSING_WORKERS = 4  # Reduced for stability
MAX_RESIDUES = 1000
STRUCTURES_DIR = "datasets/BioLip/structures"
ANNOTATION_PATH = "datasets/BioLip/BioLiP_annotation.csv"
OUTPUT_FILE = "datasets/BioLip/biolip_subset.pkl"
LOG_FILE = "datasets/BioLip/processing.log"
PROCESSING_TIMEOUT = 600  # 10 minutes per structure
WATCHDOG_INTERVAL = 1200  # 20 minutes
CHUNK_SIZE = 200  # Smaller chunks for memory stability
PROGRESS_INTERVAL = 100
# =========================

# Enhanced logging setup
class ProgressFilter(logging.Filter):
    def filter(self, record):
        record.progress = f"{progress_completed}/{progress_total}" if progress_total > 0 else "0/0"
        return True

logger = logging.getLogger("BioLiPProcessor")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(progress)s] - %(levelname)s - %(message)s'
))
file_handler.addFilter(ProgressFilter())
logger.addHandler(file_handler)

# Console handler with colors
class ColorFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.INFO: '\033[92m',  # Green
        logging.WARNING: '\033[93m',  # Yellow
        logging.ERROR: '\033[91m',  # Red
        logging.CRITICAL: '\033[91m\033[1m',  # Bold red
    }
    RESET_CODE = '\033[0m'

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, '')
        message = super().format(record)
        return f"{color}{message}{self.RESET_CODE}"

console = logging.StreamHandler()
console.setFormatter(ColorFormatter(
    '[%(progress)s] %(levelname)s: %(message)s'
))
console.addFilter(ProgressFilter())
logger.addHandler(console)

# Global progress tracking
progress_completed = 0
progress_total = 0
progress_lock = threading.Lock()
last_activity_time = time.time()
watchdog_active = True

def log_success(message):
    logger.info(f"✅ {message}")

def log_warning(message):
    logger.warning(f"⚠️ {message}")

def log_error(message):
    logger.error(f"❌ {message}")

def update_progress(increment=1):
    global progress_completed, last_activity_time
    with progress_lock:
        progress_completed += increment
        last_activity_time = time.time()

def update_activity():
    global last_activity_time
    with progress_lock:
        last_activity_time = time.time()

def deadlock_watchdog():
    """Monitor and detect deadlocks with extended timeout"""
    while watchdog_active:
        time.sleep(60)
        current_time = time.time()
        
        with progress_lock:
            last_act = last_activity_time
        elapsed = current_time - last_act

        if elapsed > WATCHDOG_INTERVAL:
            logger.critical(f"DEADLOCK DETECTED! No progress for {elapsed:.0f} seconds.")
            try:
                logger.critical("Attempting to dump stack traces...")
                import faulthandler
                faulthandler.dump_traceback()
            except Exception:
                logger.critical("Failed to dump stack traces")
            os.kill(os.getpid(), signal.SIGTERM)
            return

# Timeout handler for downloads
socket.setdefaulttimeout(60)

# ---- Robust Downloader ----
def download_structure(pdb_id):
    """Download structure with timeout handling and retries"""
    cif_path = os.path.join(STRUCTURES_DIR, f"{pdb_id}.cif")
    pdb_path = os.path.join(STRUCTURES_DIR, f"{pdb_id}.pdb")
    gz_cif_path = f"{cif_path}.gz"

    # Skip if exists in any format
    if os.path.exists(cif_path) or os.path.exists(pdb_path) or os.path.exists(gz_cif_path):
        return True

    try:
        # Try compressed CIF first
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id}.cif.gz",
            gz_cif_path,
            reporthook=lambda count, size, total: update_activity()
        )

        # Decompress in memory
        with gzip.open(gz_cif_path, 'rb') as f_in:
            with open(cif_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_cif_path)
        log_success(f"Downloaded CIF: {pdb_id}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            try:
                # Try PDB format for older structures
                urllib.request.urlretrieve(
                    f"https://files.rcsb.org/download/{pdb_id}.pdb",
                    pdb_path,
                    reporthook=lambda count, size, total: update_activity()
                )
                log_success(f"Downloaded PDB (fallback): {pdb_id}")
                return True
            except Exception as e:
                log_error(f"Both formats failed for {pdb_id}: {str(e)}")
                return False
        else:
            log_error(f"HTTP error for {pdb_id}: {str(e)}")
            return False
    except Exception as e:
        log_error(f"Download failed for {pdb_id}: {str(e)}")
        return False

# ---- Structure Processing ----
class StructureParser:
    """Optimized structure parser with format detection"""
    def __init__(self):
        self.cif_parser = MMCIFParser(QUIET=True)
        self.pdb_parser = PDBParser(QUIET=True)

    def get_structure(self, pdb_id):
        cif_path = os.path.join(STRUCTURES_DIR, f"{pdb_id}.cif")
        pdb_path = os.path.join(STRUCTURES_DIR, f"{pdb_id}.pdb")

        try:
            if os.path.exists(cif_path):
                return self.cif_parser.get_structure(pdb_id, cif_path)
            elif os.path.exists(pdb_path):
                return self.pdb_parser.get_structure(pdb_id, pdb_path)
            return None
        except Exception as e:
            log_error(f"Parsing failed for {pdb_id}: {str(e)}")
            return None

def validate_chain(chain, max_residues):
    """Efficient chain validation with early termination"""
    residue_count = 0
    for residue in chain:
        # Skip hetero residues
        if residue.id[0] != " ":
            continue

        residue_count += 1
        # Early termination for long chains
        if residue_count > max_residues:
            return False, residue_count

        # Check for CA atom
        if "CA" not in residue:
            return False, residue_count

    return True, residue_count

def process_pdb_chains(pdb_item):
    """Process all chains for a single PDB ID with activity updates"""
    pdb_id, chain_ids = pdb_item
    valid_chains = []
    error_counts = defaultdict(int)
    parser = StructureParser()

    try:
        update_activity()  # Initial activity update
        
        structure = parser.get_structure(pdb_id)
        if not structure:
            return valid_chains, {"no_structure": len(chain_ids)}

        model = structure[0]
        for chain_id in chain_ids:
            try:
                update_activity()  # Update before each chain processing
                
                if chain_id not in model:
                    error_counts["chain_not_found"] += 1
                    continue

                chain = model[chain_id]
                valid, residue_count = validate_chain(chain, MAX_RESIDUES)

                if residue_count > MAX_RESIDUES:
                    log_warning(f"Chain too long: {pdb_id}_{chain_id} ({residue_count} residues)")
                    error_counts["chain_too_long"] += 1
                elif not valid:
                    log_warning(f"Incomplete Cα trace: {pdb_id}_{chain_id}")
                    error_counts["incomplete_ca"] += 1
                else:
                    log_success(f"Valid chain: {pdb_id}_{chain_id}")
                    valid_chains.append((pdb_id, chain_id))

            except Exception as e:
                error_counts["validation_error"] += 1
                log_error(f"Validation error for {pdb_id}_{chain_id}: {str(e)}")
        
        update_activity()  # Final update after processing

    except Exception as e:
        log_error(f"Global error for {pdb_id}: {str(e)}")
        error_counts["global_error"] = len(chain_ids)

    return valid_chains, error_counts

# ---- Resource Monitoring ----
def log_resource_usage():
    try:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.Process().memory_info().rss / (1024 ** 2)
        logger.info(f"Resource Usage: CPU={cpu:.1f}%, MEM={mem:.1f}MB")
    except Exception as e:
        logger.warning(f"Resource monitoring failed: {str(e)}")

# ---- Main Processing ----
def main():
    global progress_total, watchdog_active

    try:
        # Start deadlock watchdog
        watchdog_thread = threading.Thread(target=deadlock_watchdog, daemon=True)
        watchdog_thread.start()
        update_activity()

        # Step 1: Load and preprocess annotation data
        logger.info("Loading BioLiP annotation data...")
        df = pd.read_csv(ANNOTATION_PATH, header=None, names=["complex_str"])

        # Efficient parsing
        logger.info("Parsing complexes...")
        complex_parts = df["complex_str"].str.split(":")
        df["complex_id"] = complex_parts.str[0]
        pdb_chain = df["complex_id"].str.split("_")
        df["pdb_id"] = pdb_chain.str[0].str[:4].str.lower()
        df["chain_id"] = pdb_chain.str[-1]

        # Filter valid chain IDs
        df = df[df["chain_id"].str.match(r"^[A-Za-z]$")].copy()

        # Group chains by PDB ID
        logger.info("Grouping chains by PDB ID...")
        pdb_chain_groups = defaultdict(list)
        for pdb_id, chain_id in zip(df["pdb_id"], df["chain_id"]):
            pdb_chain_groups[pdb_id].append(chain_id)

        unique_pdbs = list(pdb_chain_groups.keys())
        total_chains = sum(len(chains) for chains in pdb_chain_groups.values())
        global progress_total
        progress_total = total_chains

        logger.info(f"Found {len(unique_pdbs)} unique PDB IDs")
        logger.info(f"Found {total_chains} unique chain entries")

        # Step 2: Download structures in parallel
        os.makedirs(STRUCTURES_DIR, exist_ok=True)
        logger.info(f"Downloading structures with {NUM_DOWNLOAD_WORKERS} workers...")
        log_resource_usage()

        # Thread-based downloader with progress tracking
        with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as executor:
            futures = {executor.submit(download_structure, pdb_id): pdb_id for pdb_id in unique_pdbs}
            processed = 0
            
            for future in as_completed(futures):
                pdb_id = futures[future]
                try:
                    success = future.result()
                    if not success:
                        log_error(f"Permanent download failure: {pdb_id}")
                except Exception as e:
                    log_error(f"Unhandled download error for {pdb_id}: {str(e)}")
                finally:
                    processed += 1
                    update_activity()
                    
                    if processed % PROGRESS_INTERVAL == 0:
                        logger.info(f"Downloaded {processed}/{len(unique_pdbs)} PDBs")
                        log_resource_usage()
            
            logger.info(f"Download complete: {len(unique_pdbs)} structures processed")

        # Step 3: Process chains with enhanced stability
        logger.info(f"Processing chains with {NUM_PROCESSING_WORKERS} workers...")
        valid_pairs = []
        error_counts = defaultdict(int)
        pdb_items = list(pdb_chain_groups.items())
        total_items = len(pdb_items)
        log_resource_usage()

        # Use spawn context for better compatibility
        ctx = multiprocessing.get_context('spawn')
        
        with ProcessPoolExecutor(
            max_workers=NUM_PROCESSING_WORKERS,
            mp_context=ctx
        ) as executor:
            processed_count = 0
            chunk_size = CHUNK_SIZE
            
            for chunk_start in range(0, total_items, chunk_size):
                update_activity()
                chunk_end = min(chunk_start + chunk_size, total_items)
                chunk = pdb_items[chunk_start:chunk_end]
                futures = {executor.submit(process_pdb_chains, item): item for item in chunk}
                
                for future in as_completed(futures):
                    item = futures[future]
                    pdb_id, chain_ids = item
                    try:
                        # Individual timeout per future
                        pdb_valid_chains, pdb_errors = future.result(timeout=PROCESSING_TIMEOUT)
                        valid_pairs.extend(pdb_valid_chains)
                        for error_type, count in pdb_errors.items():
                            error_counts[error_type] += count
                    except FuturesTimeoutError:
                        log_error(f"Timeout processing {pdb_id} (chains: {chain_ids})")
                        error_counts["timeout"] += len(chain_ids)
                    except Exception as e:
                        log_error(f"Unhandled processing error for {pdb_id}: {str(e)}")
                        error_counts["unhandled_error"] += len(chain_ids)
                    finally:
                        processed_count += 1
                        update_progress(len(chain_ids))
                        
                        if processed_count % PROGRESS_INTERVAL == 0:
                            logger.info(f"Processed {processed_count}/{total_items} PDBs ({len(valid_pairs)} valid chains)")
                            log_resource_usage()
                
                logger.info(f"Completed chunk: {processed_count}/{total_items} PDBs processed")

            logger.info(f"Processing complete: {total_items} PDBs processed")

        # Save results and report
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total chains processed:     {total_chains}")
        logger.info(f"Valid chains found:         {len(valid_pairs)}")
        logger.info(f"Success rate:               {len(valid_pairs)/total_chains*100:.2f}%")
        logger.info("\nError Breakdown:")
        for error_type, count in sorted(error_counts.items()):
            logger.info(f"- {error_type.replace('_', ' ').title()}: {count}")

        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(valid_pairs, f)
        logger.info(f"\nResults saved to {OUTPUT_FILE}")
        logger.info(f"Detailed log saved to {LOG_FILE}")

    except Exception as e:
        logger.critical(f"Fatal error in main process: {str(e)}")
        logger.critical(traceback.format_exc())
    finally:
        # Clean up before exit
        watchdog_active = False
        if watchdog_thread.is_alive():
            watchdog_thread.join(timeout=5)

if __name__ == "__main__":
    # Check numpy importability before multiprocessing
    try:
        import numpy as np
        logger.info("NumPy import verified successfully")
    except ImportError as e:
        logger.critical(f"NumPy import failed: {str(e)}")
        raise
    
    main()
