#!/usr/bin/env python
import os
import sys
import torch
import argparse
from tqdm import tqdm

# Ensure local imports work (add parent dir to PYTHONPATH)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scannet_pytorch.train_pytorch import init_model, _unpack_batch
from scannet_pytorch.utilities.train_utils import process_inputs
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.utilities.data_loaders import (
    get_train_loader,
    get_val_loader,
    get_test_loader,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt", type=str, required=False,
                        default="checkpoints/phaseD_atoms_labels_enriched_20250905_1949/best_model_val.pt")
    args = parser.parse_args()

    device = torch.device(args.device)

    # === Model ===
    model = init_model(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # === Data loader ===
    pipeline = ScanNetPipeline(
        atom_features="valency",
        aa_features="sequence",
        aa_frames="triplet_sidechain",
        Beff=500
    )
    if args.split == "train":
        loader = get_train_loader(pipeline, batch_size=1, num_workers=4)
    elif args.split == "val":
        loader = get_val_loader(pipeline, batch_size=1, num_workers=2)
    else:  # test
        loader = get_test_loader(pipeline, batch_size=1, num_workers=2)

    out_dir = f"/home/scratch1/asiddhi/fusion_dumps/scannet/{args.split}"
    os.makedirs(out_dir, exist_ok=True)

    # === Dump logits ===
    skipped = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Dumping ScanNet {args.split}")):
            if batch is None:
                skipped += 1
                continue
            try:
                inputs, targets = _unpack_batch(batch, device)
            except ValueError:
                skipped += 1
                continue

            packed = process_inputs(inputs, device)
            logits = (
                model(**packed)
                if isinstance(packed, dict)
                else model(*packed)
                if isinstance(packed, (list, tuple))
                else model(packed)
            )

            if logits.ndim > targets.ndim and logits.size(-1) == 1:
                logits = logits.squeeze(-1)

            pdb_id = loader.dataset.pdb_ids[i]  # ✅ real PDB ID

            save_obj = {
                "id": pdb_id,
                "z_s": logits.squeeze(0).cpu(),   # shape [L]
                "y": targets.squeeze(0).cpu(),    # shape [L]
                "res_ids": list(range(logits.shape[-1]))
            }
            torch.save(save_obj, os.path.join(out_dir, f"{pdb_id}.pt"))

    print(f"✅ Finished {args.split} split. Skipped {skipped} proteins due to None batches.")


if __name__ == "__main__":
    main()
