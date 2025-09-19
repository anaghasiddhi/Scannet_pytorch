import os
import pickle
from torch.utils.data import DataLoader
from scannet_pytorch.datasets import ScanNetDataset
from scannet_pytorch.utilities.io_utils import to_tensor
from scannet_pytorch.utilities.train_utils import filter_none, dry_run_dataset
from scannet_pytorch.utilities import paths

def get_train_loader(pipeline, batch_size=1, max_length=800, num_workers=32, subset_ids=None):
    """
    Load and return the BioLip training DataLoader.
    Prefer final aligned PKLs/ID lists when available.
    """

    splits_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    # --- Prefer aligned labels ---
    train_label_path = os.path.join(splits_dir, "BioLip_final_train.pkl")
    if not os.path.exists(train_label_path):
        train_label_path = os.path.join(splits_dir, "BioLip_train.pkl")

    if not os.path.exists(train_label_path):
        raise FileNotFoundError(f"❌ Expected {train_label_path} not found. Did you run BioLip preprocessing?")

    with open(train_label_path, "rb") as f:
        train_label_dict = pickle.load(f)

    # --- Prefer aligned ID list ---
    if subset_ids is None:
        processed_path = os.path.join(splits_dir, "final_train_ids.txt")
        if not os.path.exists(processed_path):
            processed_path = os.path.join(splits_dir, "train_processed_ids.txt")
        if os.path.exists(processed_path):
            with open(processed_path) as f:
                subset_ids = set([ln.strip() for ln in f if ln.strip()])
            print(f"[Train] Using ID list: {processed_path} (n={len(subset_ids)})")

    train_origins = list(train_label_dict.keys())
    if subset_ids is not None:
        subset_ids = set(subset_ids)
        train_origins = [pid for pid in train_origins if pid in subset_ids]
        if len(train_origins) == 0:
            raise RuntimeError("❌ No matching IDs found in train set.")

    train_dataset = ScanNetDataset(
        pdb_ids=train_origins,
        label_dict=train_label_dict,
        pipeline=pipeline,
        pdb_folder=os.path.join(splits_dir, "train.atom_enriched"),  # your materialized symlink
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=filter_none,
        num_workers=num_workers
    )

    # Dry run
    strict = os.environ.get("SCANNET_STRICT_DRYRUN", "0") == "1"
    ok = False
    try:
        ok = dry_run_dataset(train_loader.dataset)
    except Exception as e:
        print(f"⚠️  DryRun exception: {e}")
    if not ok:
        print(f"[DryRun] Loaded={getattr(train_loader.dataset,'loaded',0)}, Skipped={getattr(train_loader.dataset,'skipped',0)}")
        print(f"[DryRun] First few skips: {getattr(train_loader.dataset,'skipped_ids',[])[:5]}")
        if strict:
            raise RuntimeError("❌ Dry run failed: No usable examples found in BioLip training set. (strict mode)")
        else:
            print("⚠️  Dry run failed — continuing for smoke test (set SCANNET_STRICT_DRYRUN=1 to enforce).")

    return train_loader


def get_val_loader(pipeline, batch_size=1, max_length=800, num_workers=16, subset_ids=None):
    """
    Load and return the BioLip validation DataLoader.
    Prefer final aligned PKL/ID lists when available.
    """
    splits_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    # --- Prefer aligned labels ---
    val_label_path = os.path.join(splits_dir, "BioLip_final_val.pkl")
    if not os.path.exists(val_label_path):
        val_label_path = os.path.join(splits_dir, "BioLip_val.pkl")
    if not os.path.exists(val_label_path):
        raise FileNotFoundError(f"❌ Expected {val_label_path} not found. Did you run BioLip preprocessing?")

    with open(val_label_path, "rb") as f:
        val_label_dict = pickle.load(f)

    # --- Prefer aligned ID list ---
    if subset_ids is None:
        processed_path = os.path.join(splits_dir, "final_val_ids.txt")
        if not os.path.exists(processed_path):
            processed_path = os.path.join(splits_dir, "val_processed_ids.txt")
        if os.path.exists(processed_path):
            with open(processed_path) as f:
                subset_ids = set([ln.strip() for ln in f if ln.strip()])
            print(f"[Val] Using ID list: {processed_path} (n={len(subset_ids)})")

    val_origins = list(val_label_dict.keys())
    if subset_ids is not None:
        subset_ids = set(subset_ids)
        val_origins = [pid for pid in val_origins if pid in subset_ids]
        if len(val_origins) == 0:
            raise RuntimeError("❌ No matching IDs found in validation set.")

    val_dataset = ScanNetDataset(
        pdb_ids=val_origins,
        label_dict=val_label_dict,
        pipeline=pipeline,
        pdb_folder=os.path.join(splits_dir, "val.atom_enriched"),   # symlink points to final pool
        max_length=max_length
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=filter_none,
        num_workers=num_workers
    )

    print(f"[Val] Dataset size: {len(val_dataset)}")
    return val_loader


def get_test_loader(pipeline, batch_size=1, max_length=800, num_workers=16, subset_ids=None):
    """
    Load and return the BioLip test DataLoader.
    Prefer final aligned PKL/ID lists when available.
    """
    splits_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    # --- Prefer aligned labels ---
    test_label_path = os.path.join(splits_dir, "BioLip_final_test.pkl")
    if not os.path.exists(test_label_path):
        test_label_path = os.path.join(splits_dir, "BioLip_test.pkl")
    if not os.path.exists(test_label_path):
        raise FileNotFoundError(f"❌ Expected {test_label_path} not found. Did you run BioLip preprocessing?")

    with open(test_label_path, "rb") as f:
        test_label_dict = pickle.load(f)

    # --- Prefer aligned ID list ---
    if subset_ids is None:
        processed_path = os.path.join(splits_dir, "final_test_ids.txt")
        if not os.path.exists(processed_path):
            processed_path = os.path.join(splits_dir, "test_processed_ids.txt")
        if os.path.exists(processed_path):
            with open(processed_path) as f:
                subset_ids = set([ln.strip() for ln in f if ln.strip()])
            print(f"[Test] Using ID list: {processed_path} (n={len(subset_ids)})")

    test_origins = list(test_label_dict.keys())
    if subset_ids is not None:
        subset_ids = set(subset_ids)
        test_origins = [pid for pid in test_origins if pid in subset_ids]
        if len(test_origins) == 0:
            raise RuntimeError("❌ No matching IDs found in test set.")

    test_dataset = ScanNetDataset(
        pdb_ids=test_origins,
        label_dict=test_label_dict,
        pipeline=pipeline,
        pdb_folder=os.path.join(splits_dir, "test.atom_enriched"),  # symlink points to final pool
        max_length=max_length
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=filter_none,
        num_workers=num_workers
    )

    print(f"[Test] Dataset size: {len(test_dataset)}")
    return test_loader
