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


class ScanNetDataset(Dataset):

    def __init__(self, pdb_ids, label_dict, pipeline, pdb_folder, max_length=None):
        self.pdb_ids = pdb_ids
        self.label_dict = label_dict
        self.pipeline = pipeline
        self.pdb_folder = pdb_folder
        self.max_length = max_length  # If set, we'll truncate rather than skip
        self.loaded = 0
        self.skipped = 0
        self.skipped_ids = []  # Track which examples were skipped

        # Dataset-managed cache directory (do NOT rely on pipeline attributes)
        from pathlib import Path
        default_cache = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "npz_cache")
        )
        self.npz_dir = Path(
            getattr(pipeline, "npz_dir", None)
            or os.environ.get("SCANNET_NPZ_DIR", default_cache)
        )
        self.npz_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ScanNetDataset] Initialized with {len(self.pdb_ids)} IDs. Cache dir={self.npz_dir}")

    def __len__(self):
        return len(self.pdb_ids)

    # --- add: small helper to choose first available key ---
    def _pick(self, d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None


    def _get_labels(self, pdb_id):
        """
        Safe label lookup: try exact, lower, upper, and base-id forms.
        Ensures labels are always returned as a NumPy array (even if scalar).
        """
        base = str(pdb_id).split("_")[0]
        labels = None
        for cand in (pdb_id, pdb_id.lower(), pdb_id.upper(),
                     base, base.lower(), base.upper()):
            val = self.label_dict.get(cand)
            if val is not None:
                labels = val
                break

        if labels is None:
            return None

        labels = np.asarray(labels, dtype=np.float32)

        # 🔧 normalize: if scalar, expand to shape (1,)
        if labels.ndim == 0:
            labels = labels[None]

        return labels


    def _load_precomputed_from_npy(self, pdb_id):
        import os, numpy as np

        # try exact, lower, upper filenames
        chosen = None
        for name in (f"{pdb_id}.npy", f"{pdb_id.lower()}.npy", f"{pdb_id.upper()}.npy"):
            fpath = os.path.join(self.pdb_folder, name)
            if os.path.exists(fpath):
                chosen = fpath; break
        if chosen is None:
            return None

        arr = np.load(chosen, allow_pickle=True)
        if isinstance(arr, dict):
            sample = arr
        elif isinstance(arr, np.ndarray) and arr.dtype == object:
            try:
                sample = arr.item()
            except Exception:
                #print(f"[WHY] {pdb_id}: object array couldn’t .item() from {chosen}")
                return None
        else:
            #print(f"[WHY] {pdb_id}: unsupported npy format in {chosen} (type={type(arr)})")
            return None

        # Show keys once
        def _first_nonnull(d, opts):
            for k in opts:
                if k in d and d[k] is not None:
                    return k, d[k]
            return None, None

        k_coord, coord_aa       = _first_nonnull(sample, ["coord_aa","aa_coord","aa_xyz","xyz","coords_aa"])
        k_attr,  attr_aa        = _first_nonnull(sample, ["attr_aa","aa_features","features"])
        k_trip,  triplets_aa    = _first_nonnull(sample, ["triplets_aa","aa_triplets","aa_frames","frames"])
        k_idx,   indices_aa     = _first_nonnull(sample, ["indices_aa","aa_indices","indices","neighbors","knn_indices"])

        missing = [name for name,val in [("coord_aa",coord_aa),("attr_aa",attr_aa),
                                         ("triplets_aa",triplets_aa),("indices_aa",indices_aa)] if val is None]
        if missing:
            #print(f"[WHY] {pdb_id}: missing keys in {os.path.basename(chosen)} → {missing}")
            #print(f"[WHY] available keys: {list(sample.keys())[:20]}")
            return None

        # shape guard
        if getattr(attr_aa, "ndim", None) == 1:
            attr_aa = attr_aa[:, None]
        # labels — try exact/lower/upper
        labels = self._get_labels(pdb_id)
        if labels is None:
            #print(f"[WHY] {pdb_id}: labels not found in label_dict (tried exact/lower/upper)")
            return None
        labels = np.asarray(labels, dtype=np.float32)

        # 🔧 expand scalar or single-element labels to residue-length
        if labels.shape[0] == 1 and coord_aa is not None:
            labels = np.repeat(labels, coord_aa.shape[0]).astype(np.float32)

        # show shapes once
        try:
            shapes = (getattr(coord_aa, "shape", None),
                      getattr(attr_aa, "shape", None),
                      getattr(triplets_aa, "shape", None),
                      getattr(indices_aa, "shape", None),
                      labels.shape)
        except Exception:
            shapes = ("?",) * 5
        #print(f"[WHY] {pdb_id}: quartet shapes {shapes}")

        L = min(coord_aa.shape[0], attr_aa.shape[0],
                triplets_aa.shape[0], indices_aa.shape[0],
                labels.shape[0])
        if L < 1:
            #print(f"[WHY] {pdb_id}: min length after trim is {L} (one of arrays is empty)")
            return None

        inputs = [coord_aa[:L], attr_aa[:L],
                  triplets_aa[:L], indices_aa[:L]]


        # optional atom-level quartet
        kA, coord_atom    = _first_nonnull(sample, ["coord_atom","atom_coord","atom_xyz"])
        kB, attr_atom     = _first_nonnull(sample, ["attr_atom","atom_features"])
        kC, triplets_atom = _first_nonnull(sample, ["triplets_atom","atom_triplets","atom_frames"])
        kD, indices_atom  = _first_nonnull(sample, ["indices_atom","atom_indices","knn_indices_atom"])

        # 🔑 map element symbols to ints if needed
        element_map = {"C":0,"N":1,"O":2,"S":3,"H":4,"P":5,"UNK":6}
        if attr_atom is not None and getattr(attr_atom, "dtype", None) is not None:
            if attr_atom.dtype.kind in {"U","S"}:
                attr_atom = np.array([element_map.get(str(e), element_map["UNK"]) for e in attr_atom],
                                     dtype=np.int64)

        # 🔧 ensure indices_atom has (Na, K>=1) with int64
        if indices_atom is not None:
            indices_atom = np.asarray(indices_atom)
            if indices_atom.ndim == 1:
                indices_atom = indices_atom[:, None]  # (Na,) -> (Na,1)
            if indices_atom.dtype != np.int64:
                indices_atom = indices_atom.astype(np.int64, copy=False)

        if (coord_atom is not None) and (attr_atom is not None) and (indices_atom is not None):
            Na = min(coord_atom.shape[0], attr_atom.shape[0], indices_atom.shape[0])

            # if missing, create dummy triplets
            if triplets_atom is None:
                triplets_atom = np.zeros((Na, 0), dtype=np.float32)

            La = min(Na, triplets_atom.shape[0])
            if La > 0:
                inputs += [coord_atom[:La], attr_atom[:La], triplets_atom[:La], indices_atom[:La]]

        mask = next((sample[k] for k in ["mask","aa_mask"] if k in sample and sample[k] is not None), None)
        if mask is not None:
            inputs.append(mask[:L])

        return inputs, labels

    def __getitem__(self, idx):
        """
        Prefer precomputed .npy in pdb_folder; only fall back to local .cif/.pdb.
        Returns (inputs_list, labels_np) or None if unusable.
        """
        pdb_id = self.pdb_ids[idx]

        # try precomputed .npy via helper
        pre = self._load_precomputed_from_npy(pdb_id)
        if pre is not None:
            inputs, labels = pre
            if getattr(inputs[1], "ndim", None) == 1:
                inputs[1] = inputs[1][:, None]
            self.loaded += 1
            return inputs, labels

        # --------- FALLBACK: try local structure ----------
        try:
            pdb_code, chain_id = pdb_id.split("_", 1)
            pdb_code = pdb_code.strip().lower()
            chain_id = chain_id.strip()
        except Exception:
            self.skipped += 1
            return None

        local_names = (
            f"{pdb_code}.cif", f"{pdb_code}.pdb", f"{pdb_code}.ent",
            f"pdb{pdb_code}.cif", f"pdb{pdb_code}.pdb", f"pdb{pdb_code}.ent",
        )
        local_path = next((os.path.join(self.pdb_folder, n) for n in local_names
                           if os.path.exists(os.path.join(self.pdb_folder, n))), None)
        if local_path is None:
            self.skipped += 1
            return None

        try:
            chains = PDBio.load_chains(
                pdb_code, [chain_id], biounit=True, structures_folder=self.pdb_folder,
            )
        except TypeError:
            chains = PDBio.load_chains(pdb_code, [chain_id])

        if not chains:
            self.skipped += 1
            return None

        proc = self.pipeline.process_example(chains[0], pdb_id)
        if proc is None:
            self.skipped += 1
            return None

        # canonical AA quartet
        coord_aa    = self._pick(proc, ["coord_aa", "aa_coord", "aa_xyz", "xyz"])
        attr_aa     = self._pick(proc, ["attr_aa", "aa_features", "features"])
        triplets_aa = self._pick(proc, ["triplets_aa", "aa_triplets", "aa_frames", "frames"])
        indices_aa  = self._pick(proc, ["indices_aa", "aa_indices", "indices"])
        if any(x is None for x in (coord_aa, attr_aa, triplets_aa, indices_aa)):
            self.skipped += 1
            return None

        if getattr(attr_aa, "ndim", None) == 1:
            attr_aa = attr_aa[:, None]

        # safe label lookup
        labels = self._get_labels(pdb_id)
        if labels is None:
            self.skipped += 1
            #print(f"[WHY] {pdb_id}: labels not found in label_dict (tried exact/lower/upper/base)")
            return None
        labels = np.asarray(labels, dtype=np.float32)


        # 🔧 normalize labels shape
        # - scalars -> (1,)
        if labels.ndim == 0:
            labels = labels[None]
        # - (L,1) -> (L,)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]

        # 🔧 expand scalar or single-element labels to residue-length
        if labels.shape[0] == 1 and coord_aa is not None:
            labels = np.repeat(labels, coord_aa.shape[0]).astype(np.float32)


        # trim to common length
        L = min(coord_aa.shape[0], attr_aa.shape[0],
                triplets_aa.shape[0], indices_aa.shape[0], labels.shape[0])
        if L < 1:
            self.skipped += 1
            return None

        coord_aa, attr_aa, triplets_aa, indices_aa, labels = \
            coord_aa[:L], attr_aa[:L], triplets_aa[:L], indices_aa[:L], labels[:L]
        inputs = [coord_aa, attr_aa, triplets_aa, indices_aa]

        # optional atom-level quartet
        coord_atom    = self._pick(proc, ["coord_atom", "atom_coord", "atom_xyz"])
        attr_atom     = self._pick(proc, ["attr_atom", "atom_features"])
        triplets_atom = self._pick(proc, ["triplets_atom", "atom_triplets", "atom_frames"])
        indices_atom  = self._pick(proc, ["indices_atom", "atom_indices"])
        if all(x is not None for x in (coord_atom, attr_atom, triplets_atom, indices_atom)):
            La = min(coord_atom.shape[0], attr_atom.shape[0],
                     triplets_atom.shape[0], indices_atom.shape[0])
            inputs += [coord_atom[:La], attr_atom[:La],
                       triplets_atom[:La], indices_atom[:La]]

        mask = self._pick(proc, ["mask", "aa_mask"])
        if mask is not None:
            inputs.append(mask[:L])

        self.loaded += 1
        return inputs, labels


    def _find_structure_file(self, pdb_id):
        # Extract base PDB code (e.g., "1ag9" from "1ag9_0-A")
        pdb_code = pdb_id.split("_")[0].lower()

        names = [
            f"{pdb_code}.pdb",
            f"{pdb_code}.ent",
            f"{pdb_code}.cif",
            f"pdb{pdb_code}.pdb",
            f"pdb{pdb_code}.ent",
            f"pdb{pdb_code}.cif",
            f"{pdb_code}.bioent",
            f"pdb{pdb_code}.bioent"
        ]

        for name in names:
            path = os.path.join(self.pdb_folder, name)
            if os.path.exists(path):
                return path
        return None

    def _download_structure_if_missing(self, pdb_id):
        try:
            for attempt in range(2):
                location, _ = PDBio.getPDB(pdb_id, biounit=True, structures_folder=self.pdb_folder)
                time.sleep(1)
                if location and os.path.exists(location):
                    return location
        except Exception as e:
            print(f"[❌ Download error] {e}")
        return None
