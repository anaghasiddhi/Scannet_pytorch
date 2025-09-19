import os
import sys
import time
import numpy as np
import pandas as pd
from time import sleep
from numba import prange, njit
from scipy.interpolate import interp1d

from scannet_pytorch.utilities.paths import path2hhblits, path2sequence_database

# Define numeric types used throughout the module
curr_float = np.float32
curr_int = np.int16

# Define amino acid vocabulary
aa = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-'
]

# Map amino acids to indices
aadict = {aa[k]: k for k in range(len(aa))}

# Add ambiguity and special character mappings
for ambiguous in ['X', 'B', 'Z', 'O', 'U']:
    aadict[ambiguous] = len(aa)

# Map lowercase equivalents
lowercase_aa = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']
for k, key in enumerate(lowercase_aa):
    aadict[key] = aadict[aa[k]]

# Handle special symbols
aadict['x'] = len(aa)
aadict['b'] = len(aa)
aadict['z'] = -1
aadict['.'] = -1

def seq2num(string):
    if isinstance(string, (str, np.str_)):
        return np.array([aadict.get(x, len(aa)) for x in string], dtype=curr_int)[np.newaxis, :]
    elif isinstance(string, (list, np.ndarray)):
        return np.array([[aadict.get(x, len(aa)) for x in s] for s in string], dtype=curr_int)
    else:
        raise TypeError(f"Unsupported input type for seq2num: {type(string)}")


def num2seq(num):
    if num.ndim == 1:
        return ''.join([aa[min(int(x), len(aa) - 1)] for x in num])
    elif num.ndim == 2:
        return [''.join([aa[min(int(x), len(aa) - 1)] for x in row]) for row in num]
    else:
        raise ValueError(f"Input to num2seq must be 1D or 2D, got {num.ndim}D")


def load_FASTA(
    filename,
    with_labels=True,
    numerical=True,
    remove_insertions=True,
    drop_duplicates=True
):
    all_seqs = []
    all_labels = [] if with_labels else None
    current_seq = ''

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    all_seqs.append(current_seq)
                    current_seq = ''
                if with_labels:
                    all_labels.append(line[1:])
            else:
                current_seq += line
                if remove_insertions:
                    current_seq = ''.join([x for x in current_seq if not (x.islower() or x == '.')])

        # Append final sequence
        if current_seq:
            all_seqs.append(current_seq)

    # Remove header line's empty string if present
    all_seqs = all_seqs[1:] if all_seqs and all_seqs[0] == '' else all_seqs

    if numerical:
        all_seqs = np.array([[aadict.get(res, len(aa)) for res in seq] for seq in all_seqs], dtype=curr_int, order='C')
    else:
        all_seqs = np.array(all_seqs)

    if drop_duplicates:
        all_seqs_df = pd.DataFrame(all_seqs)
        all_seqs_df = all_seqs_df.drop_duplicates()
        if with_labels:
            all_labels = np.array(all_labels)[all_seqs_df.index]
        all_seqs = all_seqs_df.to_numpy()

    if with_labels:
        return all_seqs, np.array(all_labels)
    else:
        return all_seqs


@njit(parallel=False, cache=True, nogil=False)
def weighted_average(config, weights, q):
    B, N = config.shape
    out = np.zeros((N, q), dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            out[n, config[b, n]] += weights[b]
    out /= weights.sum()
    return out

@njit(parallel=True, cache=True)
def count_neighbours(MSA, threshold=0.1, remove_gaps=False):
    B = MSA.shape[0]
    num_neighbours = np.ones(B, dtype=curr_int)

    for b1 in prange(B):
        for b2 in range(b1 + 1, B):
            if remove_gaps:
                valid_mask = (MSA[b1] != 20) * (MSA[b2] != 20)
                num_diff = (MSA[b1] != MSA[b2]) * valid_mask
                similarity = num_diff.sum() / max(valid_mask.sum(), 1)
            else:
                similarity = (MSA[b1] != MSA[b2]).mean()

            if similarity < threshold:
                num_neighbours[b1] += 1
                num_neighbours[b2] += 1

    return num_neighbours


def get_focusing_weights(all_sequences, all_weights, WT, targets_Beff, step=0.5):
    homology = 1.0 - (all_sequences == all_sequences[WT]).mean(-1)
    Beff = all_weights.sum()
    targets_Beff = np.array(targets_Beff)
    Beff_min = targets_Beff.min()

    all_focusing_weights = np.ones((len(all_weights), len(targets_Beff)), dtype=np.float32)

    if Beff_min < Beff:
        # Determine maximum focusing factor
        Beff_current = Beff
        focusing = 0.0
        all_focusings = [0.0]
        all_Beff = [Beff]

        while Beff_current > Beff_min:
            focusing += step
            focusing_weights = np.exp(-focusing * homology)
            Beff_current = (all_weights * focusing_weights).sum()
            all_focusings.append(focusing)
            all_Beff.append(Beff_current)

        # Interpolate to find per-target focusing weights
        f = interp1d(all_Beff, all_focusings, bounds_error=False)
        target_focusings = f(targets_Beff)
        target_focusings[targets_Beff > Beff] = 0.0

        for l, target_focusing in enumerate(target_focusings):
            all_focusing_weights[:, l] = np.exp(-target_focusing * homology)

    return all_focusing_weights


def conservation_score(PWM, Beff, Bvirtual=5):
    eps = Bvirtual / (Bvirtual + Beff * (1 - PWM[:, -1]))
    PWM = PWM[:, :-1].copy()
    PWM /= PWM.sum(-1)[:, np.newaxis]
    PWM = eps[:, np.newaxis] / 20 + (1 - eps[:, np.newaxis]) * PWM
    conservation = np.log(20) - (-np.log(PWM) * PWM).sum(-1)
    return conservation


def compute_PWM(location, gap_threshold=0.3,
                neighbours_threshold=0.1, Beff=500, WT=0, nmax=10000, scaled=False):
    
    if not isinstance(Beff, list):
        Beff = [Beff]

    nBeff = len(Beff)

    # Load sequences and labels
    all_sequences, all_labels = load_FASTA(
        location,
        remove_insertions=True,
        with_labels=True,
        drop_duplicates=True
    )

    # Filter out sequences with too many gaps
    sequences_with_few_gaps = (all_sequences == 20).mean(-1) < gap_threshold
    all_sequences = all_sequences[sequences_with_few_gaps]
    all_labels = all_labels[sequences_with_few_gaps]

    # Cap max number of sequences
    if len(all_sequences) >= nmax:
        d2wt = (all_sequences != all_sequences[WT:WT + 1]).mean(-1)
        subset = np.argsort(d2wt)[:nmax]
        all_sequences = all_sequences[subset]
        all_labels = all_labels[subset]

    # Compute sequence weights via neighborhood collapse
    all_weights = 1.0 / count_neighbours(all_sequences, threshold=neighbours_threshold)

    # Handle ambiguous residues
    ambiguous_residues = np.nonzero(all_sequences == 21)
    if len(ambiguous_residues[0]) > 0:
        all_sequences[ambiguous_residues[0], ambiguous_residues[1]] = 20
        PWM_temp = weighted_average(all_sequences, all_weights.astype(curr_float), 21)
        consensus = np.argmax(PWM_temp, axis=-1)
        all_sequences[ambiguous_residues[0], ambiguous_residues[1]] = consensus[ambiguous_residues[1]]

    # Compute focusing weights
    all_focusing_weights = get_focusing_weights(
        all_sequences, all_weights, WT, Beff
    )

    # Apply focusing
    all_weights_focused = all_weights[:, np.newaxis] * all_focusing_weights
    all_weights_focused /= all_weights_focused.mean(0)

    # Initialize PWM
    PWM = np.zeros((all_sequences.shape[-1], 21, nBeff), dtype=curr_float)

    # Compute PWM for each target Beff
    for n in range(nBeff):
        PWM[:, :, n] = weighted_average(
            all_sequences,
            all_weights_focused[:, n].astype(curr_float),
            21
        )

    # Optionally apply conservation-based scaling
    if scaled:
        Beff_actual = all_weights.sum()
        for n in range(nBeff):
            conservation = conservation_score(PWM[:, :, n], Beff_actual, Bvirtual=5)
            PWM[:, :, n] *= conservation[:, np.newaxis]

    if nBeff == 1:
        return PWM[:, :, 0]
    return PWM



def call_hhblits(
    sequence,
    output_alignment,
    path2hhblits=path2hhblits,
    path2sequence_database=path2sequence_database,
    overwrite=True,
    cores=6,
    iterations=4,
    MSA=None
):

    query_file = output_alignment[:-6] + '_query.fasta'
    output_file = output_alignment[:-6] + 'metadata.txt'

    # Skip computation if files already exist and overwrite is False
    if not overwrite:
        if os.path.exists(output_alignment) and os.path.exists(output_file):
            print(f"File {output_alignment} already exists. Not recomputing.")
            return output_alignment

    # If MSA is provided, copy it; otherwise, write FASTA from sequence
    if MSA is not None:
        os.system(f'scp {MSA} {query_file}')
    else:
        with open(query_file, 'w') as f:
            f.write('>WT\n')
            f.write(sequence + '\n')

    # Escape spaces in file paths
    query_file_escaped = query_file.replace(" ", "\\ ")
    output_file_escaped = output_file.replace(" ", "\\ ")
    output_alignment_escaped = output_alignment.replace(" ", "\\ ")
    path2sequence_database_escaped = path2sequence_database.replace(" ", "\\ ")

    # Construct command
    cmd = (
        f"{path2hhblits} -cpu {cores} -all -n {iterations} "
        f"-i {query_file_escaped} "
        f"-o {output_file_escaped} "
        f"-oa3m {output_alignment_escaped} "
        f"-d {path2sequence_database_escaped}"
    )

    # Run hhblits and time it
    t = time.time()
    os.system(cmd)
    os.system(f"rm {query_file_escaped}")
    print(f"Called hhblits finished: Duration {time.time() - t:.2f} s")

    return output_alignment
