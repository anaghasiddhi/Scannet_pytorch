import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPerformanceWarning

# Suppress specific Numba warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

import os
import torch
import pickle
import time
import numpy as np
import traceback
from torch.utils.data import Dataset
from scannet_pytorch.preprocessing import PDBio
from scannet_pytorch.preprocessing.PDBio import parse_str


import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class ScanNetDataset(Dataset):
    def __init__(self, split: str, data_root: str):
        """
        Dataset for preprocessed BioLip data.

        Args:
            split (str): one of {"train", "val", "test"}.
            data_root (str): path to ScanNet/data folder where
                             BioLip_{split}.pkl and {split}.data/ live.
        """
        self.split = split
        self.data_root = data_root

        # === Load labels ===
        label_path = os.path.join(data_root, f"BioLip_{split}.pkl")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        with open(label_path, "rb") as f:
            self.label_dict = pickle.load(f)

        # === List of origins (keys like "2XNT_J") ===
        self.origins = list(self.label_dict.keys())

        # === Feature directory ===
        self.feature_dir = os.path.join(data_root, f"{split}.data")
        if not os.path.isdir(self.feature_dir):
            raise FileNotFoundError(f"Feature dir not found: {self.feature_dir}")

        # Stats
        print(f"[ScanNetDataset] Split={split}, loaded {len(self.origins)} label entries")

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        origin = self.origins[idx]  # e.g. "2XNT_J"

        # --- Load labels ---
        labels = self.label_dict.get(origin, None)
        if labels is None or labels.size == 0:
            return None  # skip unusable

        # Ensure labels are 2D (N, 1)
        if labels.ndim == 1:
            labels = labels[:, np.newaxis]
        labels = torch.tensor(labels, dtype=torch.float32)

        # --- Load precomputed features ---
        feat_path = os.path.join(self.feature_dir, f"{origin}.npy")
        if not os.path.exists(feat_path):
            print(f"[⚠️ Missing feature file] {feat_path}")
            return None

        feats = np.load(feat_path)
        feats = torch.tensor(feats, dtype=torch.float32)

        # --- Sanity check ---
        if feats.shape[0] != labels.shape[0]:
            print(f"[⚠️ Length mismatch] {origin}: feats={feats.shape}, labels={labels.shape}")
            return None

        return feats, labels, origin
