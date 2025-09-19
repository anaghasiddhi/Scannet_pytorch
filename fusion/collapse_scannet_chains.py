#!/usr/bin/env python
import os
import sys
import argparse
import torch
from tqdm import tqdm
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Collapse ScanNet chain-level dumps into PDB-level dumps")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"],
                        help="Which split to collapse (train, val, test)")
    parser.add_argument("--in_dir", type=str, default="/home/scratch1/asiddhi/fusion_dumps/scannet",
                        help="Input dir containing chain-level dumps")
    parser.add_argument("--out_dir", type=str, default="/home/scratch1/asiddhi/fusion_dumps/scannet_collapsed",
                        help="Output dir for collapsed PDB-level dumps")
    return parser.parse_args()


def main():
    args = parse_args()

    in_dir = os.path.join(args.in_dir, args.split)
    out_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    print(f"📂 Collapsing ScanNet chain dumps from: {in_dir}")
    print(f"💾 Saving collapsed PDB-level dumps to: {out_dir}")

    # group chain files by PDB id (strip chain suffix after "_")
    groups = defaultdict(list)
    for fname in os.listdir(in_dir):
        if not fname.endswith(".pt"):
            continue
        pdb_id = fname.split("_")[0]  # e.g. "1a0d_A.pt" → "1a0d"
        groups[pdb_id].append(os.path.join(in_dir, fname))

    print(f"🔎 Found {len(groups)} unique PDB IDs in split={args.split}")

    for pdb_id, file_list in tqdm(groups.items(), desc=f"Collapsing {args.split}", ncols=100):
        z_s_list, y_list, res_ids_list = [], [], []
        for fpath in sorted(file_list):  # sort chains alphabetically
            obj = torch.load(fpath, map_location="cpu")

            z_s = obj["z_s"]
            y = obj["y"]

            # ensure 1-D
            if z_s.ndim == 0:
                z_s = z_s.unsqueeze(0)
            if y.ndim == 0:
                y = y.unsqueeze(0)

            z_s_list.append(z_s)
            y_list.append(y)
            res_ids_list.append(obj["res_ids"])

        # concatenate across chains
        try:
            z_s = torch.cat(z_s_list, dim=0)
            y = torch.cat(y_list, dim=0)
        except Exception as e:
            print(f"⚠️ Skipping {pdb_id} due to concat error: {e}")
            continue

        res_ids = sum(res_ids_list, [])

        dump = {
            "id": pdb_id,
            "z_s": z_s,
            "y": y,
            "res_ids": res_ids
        }

        save_path = os.path.join(out_dir, f"{pdb_id}.pt")
        torch.save(dump, save_path)

    print(f"✅ Finished collapsing {args.split}. Output in {out_dir}")


if __name__ == "__main__":
    main()
