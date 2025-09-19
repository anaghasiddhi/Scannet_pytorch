#!/usr/bin/env python3
import os, shutil, sys, subprocess

BATCH_DIR = "data/unprocessed_batches"
LOG_DIR = "logs/reproc_unprocessed"
RECOVERY_DIR = "data/recovery"
MAIN_DATA_DIR = "data"

CIF_ROOT = "/tmp/structures_local"
NUM_WORKERS = 4
MAX_PARALLEL = 4   # how many batches at once

def launch_batches():
    """Run recovery on unprocessed batches, throttled to MAX_PARALLEL, with progress info."""
    batches = sorted(os.listdir(BATCH_DIR))
    total = len(batches)
    running = []
    completed = 0

    for f in batches:
        batch_file = os.path.join(BATCH_DIR, f)
        log_file = os.path.join(LOG_DIR, f"{f}.log")

        cmd = [
            "python", "-u", "scannet_pytorch/preprocess_biolip.py",
            "--input-list", batch_file,
            "--cif-root", CIF_ROOT,
            "--output-dir", RECOVERY_DIR,
            "--num-workers", str(NUM_WORKERS)
        ]
        print(f"Launching {f} ({completed}/{total} done, {len(running)} active) → logs/{f}.log")
        lf = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=lf)
        running.append((f, p, lf))

        # throttle
        if len(running) >= MAX_PARALLEL:
            done = False
            while not done:
                for entry in list(running):
                    fname, proc, logf = entry
                    if proc.poll() is not None:
                        logf.close()
                        running.remove(entry)
                        completed += 1
                        print(f"✅ Finished {fname} ({completed}/{total})")
                        done = True
                        break

    # wait for any stragglers
    for fname, proc, logf in running:
        proc.wait()
        logf.close()
        completed += 1
        print(f"✅ Finished {fname} ({completed}/{total})")

    print(f"🎉 All {total} batches finished.")

def merge_recovered():
    """Copy recovered .npy + update labels into main split dirs."""
    for sub in os.listdir(RECOVERY_DIR):
        if not sub.endswith(".data"): 
            continue
        split = "train" if "train" in sub else "val" if "val" in sub else "test"
        src_dir = os.path.join(RECOVERY_DIR, sub)
        dst_dir = os.path.join(MAIN_DATA_DIR, f"{split}.data")
        os.makedirs(dst_dir, exist_ok=True)

        print(f"➡️ Merging {src_dir} → {dst_dir}")
        for fname in os.listdir(src_dir):
            if fname.endswith(".npy"):
                shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

        # merge labels pkl
        src_labels = os.path.join(RECOVERY_DIR, sub.replace(".data", "_labels.pkl"))
        dst_labels = os.path.join(MAIN_DATA_DIR, f"{split}_labels.pkl")
        if os.path.exists(src_labels):
            # naive append: just rename recovery labels so you can merge later in Python
            merged_copy = dst_labels.replace(".pkl", "_with_recovery.pkl")
            print(f"⚠️ Copying labels separately: {src_labels} → {merged_copy}")
            shutil.copy2(src_labels, merged_copy)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge_recovered()
    else:
        launch_batches()
