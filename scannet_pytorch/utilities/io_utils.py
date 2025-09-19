import h5py
import numpy as np
import pickle
import os
import yaml

import json
import numpy as np
from pathlib import Path

def save_labels_probs(path, labels, probs_or_logits, are_logits=False):
    """
    path: Path to a .json file to write.
    labels: 1D array-like of 0/1.
    probs_or_logits: 1D array-like of floats (probabilities in [0,1] or raw logits).
    are_logits: if True, will store under 'logits', else 'probs'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=int).tolist()
    scores = np.asarray(probs_or_logits, dtype=float).tolist()
    payload = {
        "labels": labels,
        ("logits" if are_logits else "probs"): scores
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def get_from_dataset(f, key, verbose=True):
    if verbose:
        print(f'Loading {key}')

    isList = f[f"{key}/isList"][()]
    isDictionary = f[f"{key}/isDictionary"][()]
    isArray = f[f"{key}/isArray"][()]

    if isDictionary:
        keys_raw = f[f"{key}/listKeys"][()]
        keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in keys_raw]
        out = {}
        for key_ in keys:
            out[key_] = get_from_dataset(f, f"{key}/{key_}", verbose=verbose)
    elif isArray:
        itemType = f[f"{key}/itemType"][()]
        if isinstance(itemType, bytes):
            itemType = itemType.decode('utf-8')

        if itemType == 'object':
            lenItem = f[f"{key}/lenItem"][()]
            out = np.array([
                get_from_dataset(f, f"{key}/subitems/{k}", verbose=verbose)
                for k in range(lenItem)
            ])
        else:
            out = f[f"{key}/data"][()].astype(itemType)

        if isList:
            out = list(out)
    else:
        out = f[f"{key}/data"][()]

    return out


def add_to_dataset(f, key, item, verbose=True):
    isDictionary = isinstance(item, dict)
    isList = isinstance(item, list) or isinstance(item, tuple)

    if isList:
        item = np.array(item)

    isArray = isinstance(item, np.ndarray)

    f[f"{key}/isList"] = isList
    f[f"{key}/isDictionary"] = isDictionary
    f[f"{key}/isArray"] = isArray

    if isDictionary:
        keys = list(item.keys())
        f[f"{key}/listKeys"] = np.array(keys, dtype='S')
        for key_ in keys:
            add_to_dataset(f, f"{key}/{key_}", item[key_], verbose=verbose)

    elif isArray:
        itemType = str(item.dtype)
        f[f"{key}/itemType"] = itemType

        if itemType == 'object':
            lenItem = len(item)
            f[f"{key}/lenItem"] = lenItem
            if verbose:
                print(f"{key} is of type object, saving subitems")
            for k, item_ in enumerate(item):
                add_to_dataset(f, f"{key}/subitems/{k}", item_, verbose=verbose)
        else:
            if 'U' in itemType:
                item = item.astype('S')
            if verbose:
                print(f"Adding {key} to dataset")
            f.create_dataset(f"{key}/data", data=item)

    else:
        if verbose:
            print(f"Adding {key} to dataset")
        f.create_dataset(f"{key}/data", data=item)

    return


def save_h5py(env, filename, verbose=True, subset=None, exclude=None):
    keys = subset if subset is not None else list(env.keys())

    if exclude is not None:
        keys = [key for key in keys if key not in exclude]

    with h5py.File(filename, 'w', libver='latest') as f:
        add_to_dataset(f, 'listKeys', keys, verbose=verbose)
        for key in keys:
            add_to_dataset(f, key, env[key], verbose=verbose)


def load_h5py(filename, verbose=True, subset=None, exclude=None):
    env = {}

    with h5py.File(filename, 'r', libver='latest') as f:
        keys = subset if subset is not None else get_from_dataset(f, 'listKeys', verbose=verbose)

        if exclude is not None:
            keys = [key for key in keys if key not in exclude]

        for key in keys:
            env[key] = get_from_dataset(f, key, verbose=verbose)

    return env


def load_pickle(filename, subset=None, exclude=None):
    with open(filename, 'rb') as f:
        env = pickle.load(f)

    if subset is not None:
        keys = subset
    else:
        keys = list(env.keys())

    if exclude is not None:
        keys = [key for key in keys if key not in exclude]

    return {key: env[key] for key in keys} if (subset or exclude) else env


def save_pickle(env, filename, subset=None, exclude=None, protocol=4):
    keys = subset if subset is not None else list(env.keys())

    if exclude is not None:
        keys = [key for key in keys if key not in exclude]

    env_ = {key: env[key] for key in keys} if (subset or exclude) else env

    with open(filename, 'wb') as f:
        pickle.dump(env_, f, protocol=protocol)

def load_pickle_splitted(filename, nsubsets, subset_indexes=None, subset=None, exclude=None):
    assert nsubsets > 1

    if subset_indexes is None:
        subset_indexes = list(range(1, nsubsets + 1))
    elif not isinstance(subset_indexes, list):
        subset_indexes = [subset_indexes]

    base, _ = os.path.splitext(filename)
    list_filenames = [f"{base}_{i}_{nsubsets}.data" for i in subset_indexes]

    # Load the first subset
    env = load_pickle(list_filenames[0], subset=subset, exclude=exclude)

    # Merge remaining subsets
    for fname in list_filenames[1:]:
        env2 = load_pickle(fname, subset=subset, exclude=exclude)
        for key in env.keys():
            if isinstance(env[key], list):
                env[key] += env2.get(key, [])
            elif isinstance(env[key], np.ndarray):
                env[key] = np.concatenate((env[key], env2.get(key, np.array([]))), axis=0)
            elif isinstance(env[key], dict):
                for subkey in env[key]:
                    val1 = env[key][subkey]
                    val2 = env2.get(key, {}).get(subkey, [] if isinstance(val1, list) else np.array([]))
                    if isinstance(val1, list):
                        env[key][subkey] += val2
                    elif isinstance(val1, np.ndarray):
                        env[key][subkey] = np.concatenate((val1, val2), axis=0)

    return env
def write_labels(list_origins, list_sequences, list_resids, list_labels, output_file):
    nexamples = len(list_origins)
    with open(output_file, 'w') as f:
        for n in range(nexamples):
            origin = list_origins[n]
            sequence = list_sequences[n]
            label = list_labels[n]
            resids = list_resids[n]
            L = len(sequence)
            f.write('>%s\n' % origin)
            for l in range(L):
                if label.dtype == np.float:
                    f.write('%s %s %s %.4f\n' % (resids[l, 0], resids[l, 1], sequence[l], label[l]))
                else:
                    f.write('%s %s %s %s\n' % (resids[l, 0], resids[l, 1], sequence[l], label[l]))
    return output_file

def read_labels(input_file, nmax=None, label_type='int'):
    list_origins = []
    list_sequences = []
    list_labels = []
    list_resids = []

    with open(input_file, 'r') as f:
        count = 0
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith('>'):
                if nmax is not None and count == nmax:
                    break
                if count > 0:
                    list_origins.append(origin)
                    list_sequences.append(sequence)
                    list_labels.append(np.array(labels))
                    list_resids.append(np.array(resids))
                origin = line[1:].strip()
                sequence = ''
                labels = []
                resids = []
                count += 1
            else:
                parts = line.split()
                if len(parts) < 3:
                    print(f"[Warning] Skipping malformed line {line_num}: {line}")
                    continue
                try:
                    resids.append(parts[:2])
                    sequence += parts[2]
                    if label_type == 'int':
                        labels.append(int(parts[-1]))
                    else:
                        labels.append(float(parts[-1]))
                except Exception as e:
                    print(f"[Error] Could not parse line {line_num}: {line}\n{e}")
                    continue

    # Append the last record
    if count > 0:
        list_origins.append(origin)
        list_sequences.append(sequence)
        list_labels.append(np.array(labels))
        list_resids.append(np.array(resids))

    return list_origins, list_sequences, list_resids, list_labels


def read_labels_flat(input_file, label_type='int'):
    list_origins = []
    list_labels = []

    with open(input_file, 'r') as f:
        for lineno, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"[Warning] Skipping malformed line {lineno}: {line}")
                continue
            origin = parts[0]
            try:
                labels = [int(x) if label_type == 'int' else float(x) for x in parts[1:]]
            except ValueError:
                print(f"[Error] Could not parse labels on line {lineno}: {line}")
                continue
            list_origins.append(origin)
            list_labels.append(np.array(labels))

    return list_origins, None, None, list_labels

def load_single_label_flat(label_file, target_id, label_type='int'):
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].lower() == target_id.lower():
                labels = [int(x) if label_type == 'int' else float(x) for x in parts[1:]]
                return labels
    raise ValueError(f"[Error] Label for {target_id} not found in {label_file}")

def load_single_label_and_resids(label_file, target_id, label_type='int'):
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].lower() == target_id.lower():
                label_values = [int(x) if label_type == 'int' else float(x) for x in parts[1:]]
                resids = [[target_id.split('_')[-1], str(i+1)] for i in range(len(label_values))]  # [chain, resnum]
                return target_id, resids, np.array(label_values)
    raise ValueError(f"[Error] Label for {target_id} not found in {label_file}")


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def to_tensor(x, device):
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(device)
    try:
        return torch.tensor(x, dtype=torch.float32, device=device)
    except Exception as e:
        print(f"[ERROR] to_tensor failed on input of type {type(x)}: {e}")
        raise
