from scannet_pytorch.preprocessing.protein_chemistry import (
    dictionary_covalent_bonds_numba,
    atom_type_mass,
    list_atoms,
    aa_to_index,
)
from numba import njit, types
from numba.typed import List, Dict
import numpy as np


def get_atom_frameCloud(sequence, atom_coordinates, atom_ids):
    atom_clouds = np.concatenate(atom_coordinates, axis=0)
    atom_attributes = np.concatenate(atom_ids, axis=-1)
    atom_triplets = np.array(
        _get_atom_triplets(sequence, List(atom_ids), dictionary_covalent_bonds_numba),
        dtype=np.int32
    )
    atom_indices = np.concatenate(
        [np.ones(len(atom_ids[l]), dtype=np.int32) * l for l in range(len(sequence))],
        axis=-1
    )[:, np.newaxis]
    return atom_clouds, atom_triplets, atom_attributes, atom_indices

@njit(parallel=False, cache=False)
def _get_atom_triplets(sequence, atom_ids, dictionary_covalent_bonds_numba):
    L = len(sequence)
    atom_triplets = List()
    all_keys = List(dictionary_covalent_bonds_numba.keys())
    current_natoms = 0

    for l in range(L):
        aa = sequence[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)

        for n in range(natoms):
            id = atom_id[n]

            if id == 17:  # N, special case, bound to C of previous aa
                if l > 0 and 0 in atom_ids[l - 1]:
                    previous = current_natoms - len(atom_ids[l - 1]) + atom_ids[l - 1].index(0)
                else:
                    previous = -1
                next = current_natoms + atom_id.index(1) if 1 in atom_id else -1

            elif id == 0:  # C, special case, bound to N of next aa
                previous = current_natoms + atom_id.index(1) if 1 in atom_id else -1
                if l < L - 1 and 17 in atom_ids[l + 1]:
                    next = current_natoms + natoms + atom_ids[l + 1].index(17)
                else:
                    next = -1

            else:
                key = aa + '_' + str(id)
                if key in all_keys:
                    previous_id, next_id, _ = dictionary_covalent_bonds_numba[key]
                else:
#                     print('Strange atom', key)
                    previous_id = -1
                    next_id = -1

                previous = current_natoms + atom_id.index(previous_id) if previous_id in atom_id else -1
                next = current_natoms + atom_id.index(next_id) if next_id in atom_id else -1

            atom_triplets.append((current_natoms + n, previous, next))

        current_natoms += natoms

    return atom_triplets


def get_aa_frameCloud(atom_coordinates, atom_ids, verbose=True, method='triplet_backbone'):
    if method == 'triplet_backbone':
        get_aa_frameCloud_ = _get_aa_frameCloud_triplet_backbone
    elif method == 'triplet_sidechain':
        get_aa_frameCloud_ = _get_aa_frameCloud_triplet_sidechain
    elif method == 'triplet_cbeta':
        get_aa_frameCloud_ = _get_aa_frameCloud_triplet_cbeta
    elif method == 'quadruplet':
        get_aa_frameCloud_ = _get_aa_frameCloud_quadruplet
    else:
        raise ValueError(f"Unknown method '{method}' passed to get_aa_frameCloud")

    if atom_coordinates is None or atom_ids is None:
        if verbose:
            print("⚠️ Skipping chain: atom_coordinates or atom_ids is None")
        return None, None, None
    if len(atom_coordinates) == 0 or len(atom_ids) == 0:
        if verbose:
            print("⚠️ Skipping chain: atom_coordinates or atom_ids is empty")
        return None, None, None
    try:

        aa_clouds, aa_triplets = get_aa_frameCloud_(List(atom_coordinates), List(atom_ids), verbose=verbose)
        aa_indices = np.arange(len(atom_coordinates), dtype=np.int32)[:, np.newaxis]
        aa_clouds = np.array(aa_clouds)
        aa_triplets = np.array(aa_triplets, dtype=np.int32)

        return aa_clouds, aa_triplets, aa_indices

    except Exception as e:
        if verbose:
            print(f"⚠️ Skipping chain due to error in frameCloud: {e}")
        return None, None, None


@njit(cache=True, parallel=False)
def _get_aa_frameCloud_triplet_backbone(atom_coordinates, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_clouds = List()
    aa_triplets = List()

    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)

        if 1 in atom_id:
            calpha_coordinate = atom_coordinate[atom_id.index(1)]
        else:
            calpha_coordinate = atom_coordinate[0]  # fallback: use first atom

        aa_clouds.append(calpha_coordinate)

    # Add virtual calpha at beginning and end
    aa_clouds.append(aa_clouds[0] + (aa_clouds[1] - aa_clouds[2]))       # index L
    aa_clouds.append(aa_clouds[L - 1] + (aa_clouds[L - 2] - aa_clouds[L - 3]))  # index L+1

    for l in range(L):
        center = l
        previous = L if l == 0 else l - 1
        next = L + 1 if l == L - 1 else l + 1
        aa_triplets.append((center, previous, next))

    return aa_clouds, aa_triplets

@njit(cache=True, parallel=False)
def _get_aa_frameCloud_triplet_sidechain(atom_coordinates, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_clouds = List()
    aa_triplets = List()
    count = 0

    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)

        # Get Calpha coordinate
        if 1 in atom_id:
            calpha_coordinate = atom_coordinate[atom_id.index(1)]
        else:
            calpha_coordinate = atom_coordinate[0]

        center = count
        aa_clouds.append(calpha_coordinate)
        count += 1

        if count > 1:
            previous = aa_triplets[-1][0]
        else:
            # Virtual Calpha (prepending)
            virtual_calpha_coordinate = 2 * calpha_coordinate - atom_coordinates[1][0]
            aa_clouds.append(virtual_calpha_coordinate)
            previous = count
            count += 1

        # Compute side chain center of mass
        sidechain_CoM = np.zeros(3, dtype=np.float32)
        sidechain_mass = 0.0

        for n in range(natoms):
            if atom_id[n] not in [0, 1, 17, 26, 34]:
                mass = atom_type_mass[atom_id[n]]
                sidechain_CoM += mass * atom_coordinate[n]
                sidechain_mass += mass

        if sidechain_mass > 0:
            sidechain_CoM /= sidechain_mass
        else:
            # Special case for Glycine or missing sidechains
            if l > 0:
                if (0 in atom_id) and (1 in atom_id) and (17 in atom_ids[l - 1]):
                    sidechain_CoM = (
                        3 * atom_coordinate[atom_id.index(1)]
                        - atom_coordinates[l - 1][atom_ids[l - 1].index(17)]
                        - atom_coordinate[atom_id.index(0)]
                    )
                else:
                    #if verbose:
                    #    print("Warning: missing side chain and backbone atoms at position", l)
                    sidechain_CoM = atom_coordinate[-1]
            else:
                #if verbose:
                 #   print("Warning: missing side chain and backbone atoms at position", l)
                sidechain_CoM = atom_coordinate[-1]

        aa_clouds.append(sidechain_CoM)
        next = count
        count += 1

        aa_triplets.append((center, previous, next))

    return aa_clouds, aa_triplets


@njit(cache=True, parallel=False)
def _get_aa_frameCloud_triplet_cbeta(atom_coordinates, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_clouds = List()
    aa_triplets = List()
    count = 0

    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)

        # Calpha
        if 1 in atom_id:
            calpha_coordinate = atom_coordinate[atom_id.index(1)]
        else:
            calpha_coordinate = atom_coordinate[0]

        # Cbeta
        if 2 in atom_id:
            cbeta_coordinate = atom_coordinate[atom_id.index(2)]
        else:
            if (0 in atom_id) and (1 in atom_id) and (17 in atom_id):
                cbeta_coordinate = (
                    3 * atom_coordinate[atom_id.index(1)]
                    - atom_coordinate[atom_id.index(17)]
                    - atom_coordinate[atom_id.index(0)]
                )
            else:
                cbeta_coordinate = atom_coordinate[-1]

        center = count
        aa_clouds.append(calpha_coordinate)
        count += 1

        if count > 1:
            previous = aa_triplets[-1][0]
        else:
            virtual_calpha_coordinate = 2 * calpha_coordinate - atom_coordinates[1][0]
            aa_clouds.append(virtual_calpha_coordinate)
            previous = count
            count += 1

        aa_clouds.append(cbeta_coordinate)
        next = count
        count += 1

        aa_triplets.append((center, previous, next))

    return aa_clouds, aa_triplets

@njit(cache=True, parallel=False)
def _get_aa_frameCloud_quadruplet(atom_coordinates, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_clouds = List()
    aa_triplets = List()

    # Step 1: Place C-alpha coordinates
    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]

        if 1 in atom_id:
            calpha_coordinate = atom_coordinate[atom_id.index(1)]
        else:
            calpha_coordinate = atom_coordinate[0]

        aa_clouds.append(calpha_coordinate)

    # Step 2: Add virtual Calpha atoms at start and end
    aa_clouds.append(aa_clouds[0] + (aa_clouds[1] - aa_clouds[2]))
    aa_clouds.append(aa_clouds[L - 1] + (aa_clouds[L - 2] - aa_clouds[L - 3]))

    count = L + 2  # Offset for new atoms being appended

    # Step 3: Add sidechain center-of-mass (dipole) and create quadruplet tuples
    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)

        sidechain_CoM = np.zeros(3, dtype=np.float32)
        sidechain_mass = 0.0

        for n in range(natoms):
            if atom_id[n] not in [0, 1, 17, 26, 34]:
                mass = atom_type_mass[atom_id[n]]
                sidechain_CoM += mass * atom_coordinate[n]
                sidechain_mass += mass

        if sidechain_mass > 0:
            sidechain_CoM /= sidechain_mass
        else:
            if (0 in atom_id) and (1 in atom_id) and (17 in atom_id):
                sidechain_CoM = (
                    3 * atom_coordinate[atom_id.index(1)]
                    - atom_coordinate[atom_id.index(17)]
                    - atom_coordinate[atom_id.index(0)]
                )
            else:
                sidechain_CoM = at_


def add_virtual_atoms(atom_clouds, atom_triplets, verbose=True):
    # Call numba-accelerated function to generate virtual atoms
    virtual_atom_clouds, atom_triplets = _add_virtual_atoms(atom_clouds, atom_triplets, verbose=verbose)

    if len(virtual_atom_clouds) > 0:
        virtual_atom_clouds = np.array(virtual_atom_clouds)

        # Numba occasionally produces bizarre values (e.g., >1e8)
        if np.abs(virtual_atom_clouds).max() > 1e8:
            #print("umba bug detected: large values in virtual_atom_clouds")

            # Identify affected indices
            weird_indices = np.nonzero(np.abs(virtual_atom_clouds).max(-1) > 1e8)[0]
            #print(f"Fixing {len(weird_indices)} corrupted virtual atoms")

            # Find which original atom each broken virtual atom connects to
            original_atom_indices = np.array([
                np.nonzero((atom_triplets[:, 1:] == len(atom_triplets) + idx).max(-1))[0][0]
                for idx in weird_indices
            ])

            for weird_idx, original_idx in zip(weird_indices, original_atom_indices):
                # Copy over a nearby valid position
                virtual_atom_clouds[weird_idx] = atom_clouds[original_idx]

                # Perturb slightly to maintain uniqueness (debugging workaround)
                if atom_triplets[original_idx, 1] == weird_idx:
                    virtual_atom_clouds[weird_idx][0] += 1
                else:
                    virtual_atom_clouds[weird_idx][2] += 1

        # Append fixed virtual atoms to the cloud
        atom_clouds = np.concatenate([atom_clouds, virtual_atom_clouds], axis=0)

    return atom_clouds, atom_triplets
@njit(cache=False)
def _add_virtual_atoms(atom_clouds, atom_triplets, verbose=True):
    natoms = len(atom_triplets)
    virtual_atom_clouds = List()
    count_virtual_atoms = 0
    centers = list(atom_triplets[:, 0])

    for n in range(natoms):
        triplet = atom_triplets[n]
        case1 = (triplet[1] >= 0) and (triplet[2] >= 0)
        case2 = (triplet[1] < 0) and (triplet[2] >= 0)
        case3 = (triplet[1] >= 0) and (triplet[2] < 0)
        case4 = (triplet[1] < 0) and (triplet[2] < 0)

        if case1:
            continue  # Atom has both previous and next covalent bonds

        elif case2:
            # Previous missing, next exists
            next_triplet = atom_triplets[centers.index(triplet[2])]
            if next_triplet[2] >= 0:
                virtual_atom = atom_clouds[next_triplet[0]] - atom_clouds[next_triplet[2]] + atom_clouds[triplet[0]]
            else:
                virtual_atom = atom_clouds[triplet[0]] + np.array([1, 0, 0])
            virtual_atom_clouds.append(virtual_atom)
            triplet[1] = natoms + count_virtual_atoms
            count_virtual_atoms += 1

        elif case3:
            # Next missing, previous exists
            prev_triplet = atom_triplets[centers.index(triplet[1])]
            if prev_triplet[1] >= 0:
                virtual_atom = atom_clouds[prev_triplet[0]] - atom_clouds[prev_triplet[1]] + atom_clouds[triplet[0]]
            else:
                virtual_atom = atom_clouds[triplet[0]] + np.array([0, 0, 1])
            virtual_atom_clouds.append(virtual_atom)
            triplet[2] = natoms + count_virtual_atoms
            count_virtual_atoms += 1

        elif case4:
            # Fully disconnected atom — both bonds missing
            #print("Pathological: atom has no bonds at all", triplet[0])
            virtual_prev = atom_clouds[triplet[0]] + np.array([1, 0, 0])
            virtual_next = atom_clouds[triplet[0]] + np.array([0, 0, 1])
            virtual_atom_clouds.append(virtual_prev)
            virtual_atom_clouds.append(virtual_next)
            triplet[1] = natoms + count_virtual_atoms
            triplet[2] = natoms + count_virtual_atoms + 1
            count_virtual_atoms += 2

    return virtual_atom_clouds, atom_triplets


if __name__ == '__main__':
    import numpy as np
    from Bio.PDB import PDBList
    from preprocessing import PDBio, PDB_processing
    from preprocessing.protein_frames import get_atom_frameCloud, add_virtual_atoms, get_aa_frameCloud
    from preprocessing.protein_chemistry import list_atoms, aa_to_index

    PDB_folder = '/home/scratch1/asiddhi/benchmarking_model2/ScanNet/PDB/'
    pdb = '2kho'
    chain = 'A'

    pdblist = PDBList()
    pdblist.retrieve_pdb_file(pdb, pdir=PDB_folder, file_format='mmCif')
    struct, chains = PDBio.load_chains(pdb_id=pdb, chain_ids=[(0, chain)], file=f"{PDB_folder}{pdb}.cif")

    sequence, backbone_coordinates, atom_coordinates, atom_ids, atom_types = PDB_processing.process_chain(chains)

    atom_clouds, atom_triplets, atom_attributes, atom_indices = get_atom_frameCloud(
        sequence, atom_coordinates, atom_ids
    )

    print("First 20 atom triplets:")
    for i in range(min(20, len(atom_triplets))):
        tmp = atom_triplets[i]
        center = list_atoms[atom_attributes[tmp[0]]]
        previous = list_atoms[atom_attributes[tmp[1]]] if tmp[1] >= 0 else 'NONE'
        next = list_atoms[atom_attributes[tmp[2]]] if tmp[2] >= 0 else 'NONE'
        print(f"{i:2d} | Center: {center:3s} | Prev: {previous:5s} | Next: {next:5s}")

    atom_clouds_filled, atom_triplets_filled = add_virtual_atoms(atom_clouds, atom_triplets, verbose=True)

    aa_clouds, aa_triplets, aa_indices = get_aa_frameCloud(atom_coordinates, atom_ids, verbose=True)
    aa_attributes = np.array([aa_to_index[aa] for aa in sequence], dtype=np.int32)

    inputs2network = [
        aa_triplets,
        aa_indices,
        aa_clouds,
        aa_attributes,
        atom_triplets_filled,
        atom_indices,
        atom_clouds_filled,
    ]

    print("\nNetwork input tensor shapes and types:")
    for i, input_tensor in enumerate(inputs2network):
        if isinstance(input_tensor, np.ndarray):
            print(f"Input {i}: shape={input_tensor.shape}, dtype={input_tensor.dtype}")
        else:
            print(f"Input {i}: type={type(input_tensor)}")
