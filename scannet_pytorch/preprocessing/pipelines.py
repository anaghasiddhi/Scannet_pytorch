import warnings
from numba.core.errors import NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
import os
import time
import numpy as np
import traceback
from numpy import float32, int16
from functools import partial
from multiprocessing import Pool

from scannet_pytorch.utilities import io_utils
from scannet_pytorch.utilities.paths import pipeline_folder, MSA_folder, structures_folder
from scannet_pytorch.utilities.dataset_utils import align_labels
from scannet_pytorch.preprocessing import protein_frames
from scannet_pytorch.preprocessing import protein_chemistry
from scannet_pytorch.preprocessing import PDB_processing
from scannet_pytorch.preprocessing import sequence_utils
from scannet_pytorch.preprocessing import PDBio

try:
    from numba import njit
except ImportError:
    print('Failed to import numba. Some speedups may be disabled.')

# === Database Paths ===
database_locations = {
    'dockground': pipeline_folder + 'dockground_database_processed.data',
    'SabDab': pipeline_folder + 'SabDab_database_processed.data',
    'disprot': pipeline_folder + 'disprot_database_processed.data'
}

database_nchunks = {
    'dockground': 40,
    'disprot': 1,
    'SabDab': 1
}

# === Datatypes ===
curr_float = float32
curr_int = int16

# === DSSP Mapping ===
dict_dssp2num = {
    'H': 0, 'B': 1, 'E': 2,
    'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7
}
dict_dssp2name = {
    'H': 'Alpha helix (4-12)',
    'B': 'Isolated beta-bridge residue',
    'E': 'Strand',
    'G': '3-10 helix',
    'I': 'Pi helix',
    'T': 'Turn',
    'S': 'Bend',
    '-': 'None'
}
dict_num2name = [dict_dssp2name[key] for key in dict_dssp2name]

# === Beff Targets ===
targets_Beff = [10, 50, 100, 200, 300, 400, 500, 1000, 2000, float('inf')]

# === Global Medians ===
median_asa = 0.24390244
median_backbone_depth = 2.6421
median_sidechain_depth = 2.21609
median_halfsphere_excess_up = -0.1429
median_coordination = 30
median_volumeindex = [-0.6115, -0.4397, -0.2983]


def padd_matrix(matrix, padded_matrix=None, Lmax=800, padding_value=-1):
    L = matrix.shape[0]

    if padded_matrix is None:
        ndim = matrix.ndim
        if ndim == 1:
            shape = (Lmax,)
        else:
            shape = (Lmax,) + matrix.shape[1:]
        padded_matrix = np.full(shape, padding_value, dtype=matrix.dtype)
    else:
        Lmax = padded_matrix.shape[0]

    if L > Lmax:
        padded_matrix[:] = matrix[:Lmax]
    else:
        padded_matrix[:L] = matrix
        padded_matrix[L:] = padding_value

    padded_matrix[np.isnan(padded_matrix)] = padding_value
    return padded_matrix


def remove_nan(matrix, padding_value=0.0):
    mask = np.isnan(matrix).any(axis=-1)
    matrix[mask] = padding_value
    return matrix


def binarize_variable(matrix, thresholds):
    thresholds = np.array([-np.inf] + list(thresholds) + [np.inf])
    left = matrix[:, np.newaxis] > thresholds[np.newaxis, :-1]
    right = matrix[:, np.newaxis] <= thresholds[np.newaxis, 1:]
    return left & right


def categorize_variable(matrix, mini, maxi, n_classes):
    return np.floor((matrix - mini) / ((maxi - mini) / n_classes)).astype(curr_int)


def binarize_categorical(matrix, n_classes, out=None):
    L = matrix.shape[0]
    matrix = matrix.astype(int)

    # ✅ Ensure 2D shape
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    if out is None:
        out = np.zeros((L, n_classes), dtype=bool)
    subset = (matrix >= 0) & (matrix < n_classes)
    out[np.arange(L)[subset[:, 0]], matrix[subset[:, 0], 0]] = 1  # 🔥 Full fix
    return out


def secondary_structure2num(secondary_structure):
    L = len(secondary_structure)
    out = np.zeros(L, dtype=curr_int)
    for i, ss in enumerate(secondary_structure):
        out[i] = dict_dssp2num.get(ss, -1)
    return out


def binarize_padd_residue_label(residue_labels, n_classes, Lmax=800):
    B = len(residue_labels)
    Ls = np.array([len(r) for r in residue_labels])
    output = np.zeros((B, Lmax, n_classes), dtype=curr_float)

    for b in range(B):
        for l in range(Ls[b]):
            label = residue_labels[b][l]
            if 0 <= label < n_classes:
                output[b, l, label] += 1

    return output


class Pipeline:
    def __init__(self, pipeline_name, pipeline_folder=pipeline_folder, **kwargs):
        self.pipeline_name = pipeline_name
        self.pipeline_folder = pipeline_folder
        self.requirements = []
        self.padded = False

    def build_processed_dataset(self,
                                dataset_name,
                                list_origins=None,
                                list_resids=None,
                                list_labels=None,
                                biounit=True,
                                structures_folder=structures_folder,
                                MSA_folder=MSA_folder,
                                pipeline_folder=pipeline_folder,
                                verbose=True,
                                fresh=False,
                                save=True,
                                permissive=True,
                                overwrite=False,
                                ncores=1):
        location_processed_dataset = pipeline_folder + dataset_name + f'_{self.pipeline_name}.data'
        loaded_samples = []
        found = False
        if not fresh:
            try:
                env = io_utils.load_pickle(location_processed_dataset)
                inputs = env['inputs']
                outputs = env['outputs']
                failed_samples = env['failed_samples']
                loaded_samples = []  # No samples were freshly processed
                found = True
            except Exception:
                found = False

        if not found:
            if verbose:
                print('Processed dataset not found, building it...')

            t = time.time()
            location_raw_dataset = pipeline_folder + dataset_name + '_raw.data'
            condition1 = os.path.exists(location_raw_dataset)
            condition2 = False

            if condition1:
                if verbose:
                    print(f'Raw dataset {dataset_name} found')
                env = io_utils.load_pickle(location_raw_dataset)
                if all(f'all_{req}s' in env for req in self.requirements):
                    if verbose:
                        print(f'Dataset {dataset_name} found with all required fields')
                    condition2 = True

            if condition1 and condition2:
                inputs, outputs, failed_samples, loaded_samples = self.process_dataset(
                    env,
                    label_name='all_labels' if 'all_labels' in env else None,
                    permissive=permissive
                )
            else:
                if verbose:
                    print(f'Building and processing dataset {dataset_name}')
                #print(f"🧪 build_processed_dataset: number of labels = {len(list_labels)}")
                #print(f"    Example label shape: {list_labels[0].shape if list_labels else 'None'}")

                inputs, outputs, failed_samples, loaded_samples = self.build_and_process_dataset(
                    list_origins,
                    list_resids=list_resids,
                    list_labels=list_labels,
                    biounit=biounit,
                    structures_folder=structures_folder,
                    MSA_folder=MSA_folder,
                    verbose=verbose,
                    permissive=permissive,
                    overwrite=overwrite,
                    ncores=ncores
                )

            print(f'Processed dataset built... (t={time.time() - t:.0f} s)')

            if save:
                print('Saving processed dataset...')
                #print(f'[DEBUG] Saving processed dataset to: {location_processed_dataset}')
                #print(f'[DEBUG] Directory exists: {os.path.exists(os.path.dirname(location_processed_dataset))}')
                t = time.time()

                env = {'inputs': inputs, 'outputs': outputs, 'failed_samples': failed_samples}

                # Ensure directory exists
                os.makedirs(os.path.dirname(location_processed_dataset), exist_ok=True)

                # Try saving with error handling
                try:
                    io_utils.save_pickle(env, location_processed_dataset)
                    print(f'Processed dataset saved (t={time.time() - t:.0f} s)')
                except Exception as e:
                    print(f'[ERROR] Failed to save processed dataset: {e}')

        
        return inputs, outputs, failed_samples, loaded_samples
    def build_and_process_dataset(self,
                                   list_origins,
                                   list_resids=None,
                                   list_labels=None,
                                   biounit=True,
                                   structures_folder=structures_folder,
                                   MSA_folder=MSA_folder,
                                   verbose=True,
                                   overwrite=False,
                                   permissive=True,
                                   ncores=1):
        B = len(list_origins)

        has_labels = list_labels is not None
        if has_labels:
            assert len(list_labels) == B

        has_resids = list_resids is not None
        if has_resids:
            assert len(list_resids) == B

        if ncores > 1:
            ncores = min(ncores, B)
            pool = Pool(ncores)
            batch_size = int(np.ceil(B / ncores))
            batch_list_origins = [list_origins[k * batch_size: min((k + 1) * batch_size, B)] for k in range(ncores)]
            batch_list_labels = [list_labels[k * batch_size: min((k + 1) * batch_size, B)] for k in range(ncores)] if has_labels else [None] * ncores
            batch_list_resids = [list_resids[k * batch_size: min((k + 1) * batch_size, B)] for k in range(ncores)] if has_resids else [None] * ncores

            _build = partial(self.build_and_process_dataset,
                             biounit=biounit,
                             structures_folder=structures_folder,
                             MSA_folder=MSA_folder,
                             verbose=verbose,
                             overwrite=overwrite,
                             permissive=permissive,
                             ncores=1)

            batch_outputs = pool.starmap(_build, zip(batch_list_origins, batch_list_resids, batch_list_labels))
            pool.close()

            # Determine shape of output
            input_is_list, output_is_list = False, False
            ninputs, noutputs = 1, 1
            for out in batch_outputs:
                if out and out[0]:
                    input_is_list = isinstance(out[0], list)
                    ninputs = len(out[0]) if input_is_list else 1
                    if has_labels:
                        output_is_list = isinstance(out[1], list)
                        noutputs = len(out[1]) if output_is_list else 1
                    break

            # Merge inputs
            if input_is_list:
                inputs = [[] for _ in range(ninputs)]
                for batch in batch_outputs:
                    if batch[0]:
                        for i in range(ninputs):
                            inputs[i].extend(batch[0][i])
                inputs = [np.array(x) for x in inputs]
            else:
                inputs = []
                for batch in batch_outputs:
                    if batch[0]:
                        inputs.extend(batch[0])
                inputs = np.array(inputs)

            # Merge outputs
            if has_labels:
                if output_is_list:
                    outputs = [[] for _ in range(noutputs)]
                    for batch in batch_outputs:
                        if batch[1]:
                            for i in range(noutputs):
                                outputs[i].extend(batch[1][i])
                    outputs = [np.array(x) for x in outputs]
                else:
                    outputs = []
                    for batch in batch_outputs:
                        if batch[1]:
                            outputs.extend(batch[1])
                    outputs = np.array(outputs)
            else:
                outputs = None

            failed_samples = list(np.concatenate([
                np.array(batch_outputs[k][2], dtype=int) + k * batch_size
                for k in range(ncores)
            ]))
            return inputs, outputs, failed_samples

        else:
            # Single-core fallback
            inputs, outputs, failed_samples = [], [], []
            loaded_samples =[]

            for b, origin in enumerate(list_origins):
                if verbose:
                    print(f'Processing example {origin} ({b}/{B})')
                try:
                    pdb_id = origin.split("_")[0]  # Only extract "101m" from "101m_1_A"
                    pdbfile, chain_ids = PDBio.getPDB(pdb_id, biounit=biounit, structures_folder=structures_folder)
                    if isinstance(chain_ids[0], tuple) and chain_ids[0][1] == 'all':
                        flat_chain_ids = 'all'
                    else:
                        flat_chain_ids = chain_ids
                    print(f"🔍 Requested chain_ids for {origin}: {flat_chain_ids}")
                    struct, chain_objs = PDBio.load_chains(file=pdbfile, chain_ids=flat_chain_ids)
                    print(f"[DEBUG] chain_objs types: {[type(c) for c in chain_objs]}")


                    if not chain_objs:
                        print(f"⚠️ Requested chains {chain_ids} not found in {origin}. Falling back to all chains.")
                        struct, chain_objs = PDBio.load_chains(file=pdbfile, chain_ids=None)
                        if not chain_objs:
                            print(f"❌ Still no chains found after fallback — skipping {origin}")
                            raise ValueError(f"Empty chain_objs — skipping {origin}")

                    if 'PWM' in self.requirements or 'conservation' in self.requirements:
                        sequences = [PDB_processing.process_chain(chain)[0] for chain in chain_objs]
                        msa_outfiles = [
                            MSA_folder + f'MSA_{PDBio.parse_str(origin)[0].split("/")[-1].split(".")[0]}_{mid}_{cid}.fasta'
                            for (mid, cid) in chain_ids
                        ]
                        MSA_files = [
                            sequence_utils.call_hhblits(seq, out, overwrite=overwrite)
                            for seq, out in zip(sequences, msa_outfiles)
                        ]
                        if len(MSA_files) == 1:
                            MSA_files = MSA_files[0]
                    else:
                        MSA_files = None

                    if has_labels:
                        labels = list_labels[b]

                        # 🔍 DEBUG: Print label shape
                        #print(f"🧪 Label check [{origin}]: shape = {labels.shape}, type = {type(labels)}")
                        
                        if labels.ndim == 1:
                            #print(f"⚠️ Label is 1D! Example values: {labels[:10]}")
                            labels = labels[:, None]  # ✅ FIX: convert to 2D shape
                        
                        if isinstance(chain_objs, list):
                            if len(chain_objs) == 0:
                                raise ValueError(f"[{origin}] Empty chain_objs — this should have been caught earlier.")
                            chain_objs = chain_objs[0]
                        
 
                        pdb_resids = PDB_processing.get_PDB_indices(chain_objs, return_model=True, return_chain=True)
                        aligned_labels = align_labels(
                            labels,
                            pdb_resids,
                            label_resids=list_resids[b] if has_resids else None
                        )
                        #print(f"🧪 aligned_labels shape after alignment = {aligned_labels.shape}, type = {type(aligned_labels)}")
                    else:
                        aligned_labels = None

                    #print(f"🧪 Calling process_example on {origin} | labels.shape = {aligned_labels.shape if aligned_labels is not None else 'None'}")
                    #print(f"   chain_obj type: {type(chain_objs)} | MSA_file type: {type(MSA_files)}")


                    input_data, output_data = self.process_example(
                        chain_obj=chain_objs,
                        MSA_file=MSA_files,
                        labels=aligned_labels
                    )

                    inputs.append(input_data)
                    if has_labels:
                        outputs.append(output_data)
                    loaded_samples.append(origin)

                except Exception as e:
                    print(f'Failed to process example {origin} ({b}/{B}), Error: {e}')
                    print(f"[Exception]: {e}")
                    print("[Traceback]:")
                    traceback.print_exc()
                    if permissive:
                        failed_samples.append(b)
                        continue
                    else:
                        raise

            if not inputs:
                outputs_return = [] if has_labels else None
                return [], outputs_return, failed_samples, []

            ninputs = len(inputs[0]) if isinstance(inputs[0], list) else 1

            if self.padded:
                if ninputs > 1:
                    inputs = [np.stack([x[i] for x in inputs], axis=0) for i in range(ninputs)]
                else:
                    inputs = np.stack(inputs, axis=0)
                if has_labels:
                    outputs = np.stack(outputs, axis=0)
            else:
                if ninputs > 1:
                    inputs = [np.array([x[i] for x in inputs]) for i in range(ninputs)]
                else:
                    inputs = np.array(inputs)
                if has_labels:
                    outputs = np.array(outputs)

            if has_labels:
                return inputs, outputs, failed_samples, loaded_samples
            else:
                return inputs, None, failed_samples, loaded_samples



class HandcraftedFeaturesPipeline(Pipeline):
    def __init__(self,
                 with_gaps=True,
                 Beff=500,
                 pipeline_folder=pipeline_folder,
                 feature_list=None):
        if feature_list is None:
            feature_list = [
                'primary',
                'secondary',
                'conservation',
                'pwm',
                'asa',
                'residue_depth',
                'volume_index',
                'half_sphere',
                'coordination'
            ]

        self.feature_list = feature_list
        self.Beff = Beff
        self.with_gaps = with_gaps

        features = ''.join(feature[0] for feature in feature_list)
        pipeline_name = f'pipeline_Handcrafted_features-{features}_gaps-{with_gaps}_Beff-{Beff}'

        super().__init__(pipeline_name, pipeline_folder=pipeline_folder)

        self.features_dimension = 0
        self.feature_names = []

        if 'primary' in self.feature_list:
            print('primary')
            self.requirements.append('sequence')
            self.feature_names += [f'AA {aa}' for aa in sequence_utils.aa[:20]]
            self.features_dimension += 20

        if 'secondary' in self.feature_list:
            print('secondary')
            self.requirements.append('secondary_structure')
            self.feature_names += [f'SS {name}' for _, name in dict_dssp2name.items()]
            self.features_dimension += 8

        if 'conservation' in self.feature_list:
            print('conservation')
            self.requirements.append('conservation_score')
            self.feature_names.append('Conservation')
            self.features_dimension += 1

        if 'asa' in self.feature_list:
            print('Accessible surface area')
            self.requirements.append('accessible_surface_area')
            self.feature_names.append('Accessible Surface Area')
            self.features_dimension += 1


        if 'residue_depth' in self.feature_list:
            print('Residue Depth')
            self.requirements += ['backbone_depth', 'sidechain_depth']
            self.feature_names += ['Residue Backbone Depth', 'Residue SideChain Depth']
            self.features_dimension += 2

        if 'volume_index' in self.feature_list:
            print('Volume index')
            self.requirements.append('volume_index')
            self.feature_names += [f'Volume Index {i}' for i in range(3)]
            self.features_dimension += 3


        if 'half_sphere' in self.feature_list:
            print('Half sphere exposure')
            self.requirements.append('halfsphere_excess_up')
            self.feature_names.append('Half Sphere Excess up')
            self.features_dimension += 1


        if 'coordination' in self.feature_list:
            print('Coordination')
            self.requirements.append('coordination')
            self.feature_names.append('Coordination')
            self.features_dimension += 1


        if 'pwm' in self.feature_list:
            print('pwm')
            self.requirements.append('PWM')
            if self.with_gaps:
                self.feature_names += [f'PWM {aa}' for aa in sequence_utils.aa]
                self.features_dimension += 21
            else:
                self.feature_names += [f'PWM {aa}' for aa in sequence_utils.aa[:20]]
                self.features_dimension += 20


        if 'npwm' in self.feature_list:
            print('normed pwm')
            self.requirements += ['PWM', 'conservation']
            if self.with_gaps:
                self.feature_names += [f'nPWM {aa}' for aa in sequence_utils.aa]
                self.features_dimension += 21
            else:
                self.feature_names += [f'nPWM {aa}' for aa in sequence_utils.aa[:20]]
                self.features_dimension += 20

    def process_example(self,
                        chain_obj=None,
                        sequence=None,
                        secondary_structure=None,
                        accessible_surface_area=None,
                        backbone_depth=None,
                        sidechain_depth=None,
                        halfsphere_excess_up=None,
                        coordination=None,
                        volume_index=None,
                        MSA_file=None,
                        PWM=None,
                        conservation_score=None,
                        backbone_coordinates=None,
                        atomic_coordinates=None,
                        labels=None):

        #print("🧪 DEBUG: Entered process_example")
        
        missing_features = {
            'sequence': sequence is None and 'primary' in self.feature_list,
            'secondary_structure': secondary_structure is None and 'secondary' in self.feature_list,
            'conservation': conservation_score is None and ('conservation' in self.feature_list or 'npwm' in self.feature_list),
            'asa': accessible_surface_area is None and 'asa' in self.feature_list,
            'residue_depth': (backbone_depth is None or sidechain_depth is None) and 'residue_depth' in self.feature_list,
            'half_sphere': halfsphere_excess_up is None and 'half_sphere' in self.feature_list,
            'coordination': coordination is None and 'coordination' in self.feature_list,
            'volume_index': volume_index is None and 'volume_index' in self.feature_list,
            'pwm': PWM is None and 'pwm' in self.feature_list,
            'npwm': PWM is None and 'npwm' in self.feature_list,
        }
        missing_features['backbone_coordinates'] = (
            backbone_coordinates is None and
            (missing_features['residue_depth'] or
             missing_features['half_sphere'] or
             missing_features['coordination'] or
             missing_features['volume_index'])
        )

        missing_features['atomic_coordinates'] = (
            atomic_coordinates is None and
            (missing_features['residue_depth'] or
             missing_features['half_sphere'] or
             missing_features['coordination'] or
             missing_features['volume_index'])
        )


        if any(flag for flag in missing_features.values()) and chain_obj is None:
            print(missing_features)
            raise ValueError('Missing features and chain not provided')

        try:
            sequence, backbone_coordinates, atomic_coordinates, _, _ = PDB_processing.process_chain(chain_obj)
            #print("✅ PDB_processing.process_chain succeeded")
            #print("sequence:", None if sequence is None else f"{type(sequence)}, len = {len(sequence)}")
            #print("backbone_coordinates shape:", None if backbone_coordinates is None else backbone_coordinates.shape)
            #print("atomic_coordinates shape:", None if atomic_coordinates is None else atomic_coordinates.shape)

        except Exception as e:
            import traceback
            print("❌ Error in process_chain")
            traceback.print_exc()
            raise e


        try:
            secondary_structure, accessible_surface_area = PDB_processing.apply_DSSP(chain_obj)
            #print("✅ DSSP succeeded")
            #print("secondary_structure:", None if secondary_structure is None else f"{type(secondary_structure)}, len = {len(secondary_structure)}")
            #print("accessible_surface_area shape:", None if accessible_surface_area is None else accessible_surface_area.shape)

        except Exception as e:
            import traceback
            print("❌ Error in apply_DSSP")
            traceback.print_exc()
            raise e


        if missing_features['pwm'] or missing_features['npwm']:
            if MSA_file is not None:
                if not isinstance(MSA_file, list):
                    PWM = sequence_utils.compute_PWM(
                        MSA_file,
                        gap_threshold=0.3,
                        neighbours_threshold=0.1,
                        Beff=self.Beff,
                        WT=0,
                        scaled=False
                    )
                else:
                    PWM = []
                    for msa in MSA_file:
                        pwm_chunk = sequence_utils.compute_PWM(
                            msa,
                            gap_threshold=0.3,
                            neighbours_threshold=0.1,
                            Beff=self.Beff,
                            WT=0,
                            scaled=False
                        )
                        PWM.append(pwm_chunk)
                    PWM = np.concatenate(PWM, axis=0)
            else:
                raise ValueError('Missing PWM or MSA')

        if missing_features['conservation']:
            conservation_score = sequence_utils.conservation_score(PWM, 1, Bvirtual=1e-4)

        if missing_features['residue_depth'] or missing_features['volume_index']:
            backbone_depth, sidechain_depth, volume_index = PDB_processing.analyze_surface(
                chain_obj, atomic_coordinates
            )

        if missing_features['half_sphere'] or missing_features['coordination']:
            halfsphere_excess_up, coordination = PDB_processing.ComputeResidueHSE(backbone_coordinates)

        L = len(sequence)
        input_features = np.zeros((L, self.features_dimension), dtype=curr_float)
        index_start = 0

        if 'primary' in self.feature_list:
            binarize_categorical(
                sequence_utils.seq2num(sequence)[0],
                20,
                out=input_features[:, index_start:index_start + 20]
            )
            index_start += 20

        if 'secondary' in self.feature_list:
            binarize_categorical(
                secondary_structure2num(secondary_structure),
                8,
                out=input_features[:, index_start:index_start + 8]
            )
            index_start += 8

        if 'conservation' in self.feature_list:
            if conservation_score.ndim == 1:
                conservation_score = conservation_score.reshape(-1)
            input_features[:, index_start] = conservation_score
            index_start += 1

        if 'asa' in self.feature_list:
            accessible_surface_area[np.isnan(accessible_surface_area)] = median_asa
            input_features[:, index_start] = accessible_surface_area
            index_start += 1

        if 'residue_depth' in self.feature_list:
            backbone_depth[np.isnan(backbone_depth) | (backbone_depth > 20)] = median_backbone_depth
            sidechain_depth[np.isnan(sidechain_depth) | (sidechain_depth > 20)] = median_sidechain_depth
            input_features[:, index_start] = backbone_depth
            input_features[:, index_start + 1] = sidechain_depth
            index_start += 2

        if 'half_sphere' in self.feature_list:
            halfsphere_excess_up[np.isnan(halfsphere_excess_up)] = median_halfsphere_excess_up
            input_features[:, index_start] = halfsphere_excess_up
            index_start += 1

        if 'coordination' in self.feature_list:
            coordination[np.isnan(coordination)] = median_coordination
            input_features[:, index_start] = coordination
            index_start += 1

        if 'volume_index' in self.feature_list:
            for i in range(volume_index.shape[-1]):
                volume_index[np.isnan(volume_index[:, i]), i] = median_volumeindex[i]
            input_features[:, index_start:index_start + volume_index.shape[-1]] = volume_index
            index_start += volume_index.shape[-1]

        if 'pwm' in self.feature_list:
            if PWM.ndim == 1:
                PWM = PWM.reshape(-1, 21)
            if self.with_gaps:
                input_features[:, index_start:index_start + 21] = PWM
                index_start += 21
            else:
                input_features[:, index_start:index_start + 20] = PWM[:, :-1]
                index_start += 20

        if 'npwm' in self.feature_list:
            if PWM.ndim == 1:
                PWM = PWM.reshape(-1, 21)
            if conservation_score.ndim == 1:
                conservation_score = conservation_score.reshape(-1)
            if self.with_gaps:
                input_features[:, index_start:index_start + 21] = PWM * conservation_score[:, np.newaxis]
                index_start += 21
            else:
                input_features[:, index_start:index_start + 20] = PWM[:, :-1] * conservation_score[:, np.newaxis]
                index_start += 20

        inputs = input_features
        outputs = labels
        return inputs, outputs


    def process_dataset(self, env, label_name=None, permissive=True):
        outputs = env[label_name] if label_name is not None else None
        failed_samples = []

        B = len(env['all_origins'])
        all_input_features = []

        for b in range(B):
            inputs = {
                requirement: env[f'all_{requirement}s'][b]
                for requirement in self.requirements
            }

            try:
                input_features, _ = self.process_example(**inputs)
                all_input_features.append(input_features)
            except Exception as error:
                print(f'Failed to parse example ({b}/{B}), Error: {error}')
                if permissive:
                    failed_samples.append(b)
                    continue
                else:
                    raise ValueError('Failed in non-permissive mode')

        inputs = np.array(all_input_features)

        if outputs is not None:
            outputs = np.array([
                outputs[b] for b in range(B) if b not in failed_samples
            ])

        return inputs, outputs, failed_samples



class ScanNetPipeline(Pipeline):
    '''
    from preprocessing import PDB_processing,PDBio,pipelines

    pdb = '1a3x'
    chains = 'all'
    struct, chains = PDBio.load_chains(pdb_id=pdb ,chain_ids=chains)

    pipeline = pipelines.ScanNetPipeline(
                     with_aa=True,
                     with_atom=True,
                     aa_features='sequence',
                     atom_features='type',
                     padded=False,
                     Lmax_aa=800
                     )
    [atom_clouds,atom_triplets,atom_attributes,atom_indices,aa_clouds, aa_triplets, aa_attributes, aa_indices] = pipeline.process_example(chains)
    '''

    def __init__(self,
                 pipeline_folder=pipeline_folder,
                 with_aa=True,
                 with_atom=True,
                 aa_features='sequence',
                 atom_features='valency',
                 Beff=500,
                 aa_frames='triplet_sidechain',
                 padded=False,
                 Lmax_aa=800,
                 Lmax_aa_points=None,
                 Lmax_atom=None,
                 Lmax_atom_points=None):

        pipeline_name = (
            f'pipeline_ScanNet_aa-{aa_features if with_aa else "none"}'
            f'_atom-{atom_features if with_atom else "none"}'
            f'_frames-{aa_frames}_Beff-{Beff}'
        )

        if padded:
            pipeline_name += f'_padded-{Lmax_aa}'

        super().__init__(pipeline_name, pipeline_folder=pipeline_folder)

        self.with_aa = with_aa
        self.with_atom = with_atom
        self.aa_features = aa_features
        self.atom_features = atom_features
        self.Beff = Beff
        self.aa_frames = aa_frames
        self.padded = padded

        self.Lmax_aa = Lmax_aa
        self.Lmax_aa_points = Lmax_aa_points
        self.Lmax_atom = Lmax_atom
        self.Lmax_atom_points = Lmax_atom_points


        super().__init__(pipeline_name, pipeline_folder=pipeline_folder)

        self.with_aa = with_aa
        self.with_atom = with_atom

        if Lmax_atom is None:
            Lmax_atom = 9 * Lmax_aa

        if Lmax_aa_points is None:
            if aa_frames == 'triplet_backbone':
                Lmax_aa_points = Lmax_aa + 2
            elif aa_frames in ['triplet_sidechain', 'triplet_cbeta']:
                Lmax_aa_points = 2 * Lmax_aa + 1
            elif aa_frames == 'quadruplet':
                Lmax_aa_points = 2 * Lmax_aa + 2

        if Lmax_atom_points is None:
            Lmax_atom_points = 11 * Lmax_aa

        self.Lmax_aa = Lmax_aa
        self.Lmax_atom = Lmax_atom
        self.Lmax_aa_points = Lmax_aa_points
        self.Lmax_atom_points = Lmax_atom_points

        self.aa_features = aa_features
        self.atom_features = atom_features

        assert aa_frames in [
            'triplet_sidechain',
            'triplet_cbeta',
            'triplet_backbone',
            'quadruplet'
        ]
        self.aa_frames = aa_frames
        self.Beff = Beff
        self.padded = padded

        self.requirements = ['atom_coordinate', 'atom_id', 'sequence']

        if self.with_aa and self.aa_features in ['pwm', 'both']:
            self.requirements.append('PWM')

        
    def process_example(self,
                        chain_obj=None,
                        sequence=None,
                        MSA_file=None,
                        atomic_coordinates=None,
                        atom_ids=None,
                        PWM=None,
                        labels=None,
                        *kwargs):

        #print("🧪 Entered real process_example in ScanNetPipeline")
        if chain_obj is not None:
            sequence, backbone_coordinates, atomic_coordinates, atom_ids, atom_types = (
                PDB_processing.process_chain(chain_obj)
            )

        if self.with_aa and self.aa_features in ['pwm', 'both']:
            PWM_is_none = (PWM is None) if not isinstance(PWM, list) else (PWM[0] is None)
            if PWM_is_none:
                if MSA_file is not None:
                    if not isinstance(MSA_file, list):
                        PWM = sequence_utils.compute_PWM(
                            MSA_file,
                            gap_threshold=0.3,
                            neighbours_threshold=0.1,
                            Beff=self.Beff,
                            WT=0,
                            scaled=False
                        )
                    else:
                        PWM = [
                            sequence_utils.compute_PWM(
                                msa_file,
                                gap_threshold=0.3,
                                neighbours_threshold=0.1,
                                Beff=self.Beff,
                                WT=0,
                                scaled=False
                            )
                            for msa_file in MSA_file
                        ]
                        PWM = np.concatenate(PWM, axis=0)
                        #print("✅ PWM from MSA_file concatenated")
                        #print("PWM shape:", PWM.shape)
                else:
                    raise ValueError('Missing PWM or MSA')
            elif isinstance(PWM, list):
                PWM = np.concatenate(PWM, axis=0)
                #print("✅ PWM from preloaded list concatenated")
                #print("PWM shape after concat:", PWM.shape)
        if self.with_aa:    
            try:
               #print("🧪 Entering AA feature construction")

               aa_clouds, aa_triplets, aa_indices = protein_frames.get_aa_frameCloud(
                    atomic_coordinates, atom_ids,
                    verbose=True, method=self.aa_frames
               )
               #print("✅ get_aa_frameCloud succeeded")
               #print("  aa_clouds:", None if aa_clouds is None else aa_clouds.shape)
               #print("  aa_triplets:", None if aa_triplets is None else aa_triplets.shape)
               #print("  aa_indices:", None if aa_indices is None else aa_indices.shape)

               #print("🧪 Constructing aa_attributes...")
               if self.aa_features == 'sequence':
                   #print("  using sequence mode")
                   aa_seq_numeric = sequence_utils.seq2num(sequence)[0]
                   aa_attributes = binarize_categorical(aa_seq_numeric, 20)
               elif self.aa_features == 'pwm':
                   #print("  using PWM mode")
                   assert PWM is not None, "PWM is None"
                   aa_attributes = PWM
               elif self.aa_features == 'both':
                   #print("  using both mode")
                   aa_seq_numeric = sequence_utils.seq2num(sequence)[0]
                   assert PWM is not None, "PWM is None for both"
                   aa_attributes = np.concatenate(
                       [binarize_categorical(aa_seq_numeric, 20), PWM],
                        axis=1
                   )
               else:
                   raise ValueError(f"Unsupported aa_features mode: {self.aa_features}")

               #print("✅ aa_attributes shape:", aa_attributes.shape)
               aa_attributes = aa_attributes.astype(curr_float)

               if self.padded:
                   #print("🧪 Padding AA tensors...")
                   aa_clouds = padd_matrix(aa_clouds, padding_value=0, Lmax=self.Lmax_aa_points)
                   aa_triplets = padd_matrix(aa_triplets, padding_value=-1, Lmax=self.Lmax_aa)
                   aa_attributes = padd_matrix(aa_attributes, padding_value=0, Lmax=self.Lmax_aa)
                   aa_indices = padd_matrix(aa_indices, padding_value=-1, Lmax=self.Lmax_aa)
               else:
                   #print("🧪 Removing NaNs from AA tensors...")
                   aa_clouds = remove_nan(aa_clouds, padding_value=0.)
                   aa_triplets = remove_nan(aa_triplets, padding_value=-1)
                   aa_attributes = remove_nan(aa_attributes, padding_value=0.)
                   aa_indices = remove_nan(aa_indices, padding_value=-1)

               #print("✅ AA feature construction complete")

            except Exception as e:
                print("❌ Error during AA input construction:", e)
                return None

        if self.with_atom:
            try:
                #print("🧪 Entering atom feature construction")

                atom_clouds, atom_triplets, atom_attributes, atom_indices = (
                    protein_frames.get_atom_frameCloud(sequence, atomic_coordinates, atom_ids)
                )
                #print("✅ get_atom_frameCloud succeeded")
                #print("  atom_clouds:", atom_clouds.shape)
                #print("  atom_triplets:", atom_triplets.shape)
                #print("  atom_attributes:", atom_attributes.shape)
                #print("  atom_indices:", atom_indices.shape)

                #print("🧪 Adding virtual atoms...")
                atom_clouds, atom_triplets = protein_frames.add_virtual_atoms(
                    atom_clouds, atom_triplets, verbose=True
                )
                #print("✅ Virtual atoms added")
                #print("  atom_clouds (post-virtual):", atom_clouds.shape)
                #print("  atom_triplets (post-virtual):", atom_triplets.shape)

                #print("🧪 Constructing atom_attributes with mode:", self.atom_features)
                if self.atom_features == 'type':
                    atom_attributes = protein_chemistry.index_to_type[atom_attributes]
                elif self.atom_features == 'valency':
                    aa_indices_flat = atom_indices[:, 0]
                    #print("  aa_indices_flat shape:", aa_indices_flat.shape)
                    seq_ids = sequence_utils.seq2num(sequence)[0]
                    #print("  seq_ids shape:", seq_ids.shape)
                    atom_attributes = protein_chemistry.index_to_valency[
                        seq_ids[aa_indices_flat], atom_attributes
                    ]
                atom_attributes += 1  # shift IDs to avoid 0 confusion
                atom_attributes = atom_attributes.astype(curr_float)
                #print("✅ atom_attributes processed:", atom_attributes.shape)

                if self.padded:
                    #print("🧪 Padding atom tensors...")
                    atom_clouds = padd_matrix(atom_clouds, padding_value=0, Lmax=self.Lmax_atom_points)
                    atom_triplets = padd_matrix(atom_triplets, padding_value=-1, Lmax=self.Lmax_atom)
                    atom_attributes = padd_matrix(atom_attributes, padding_value=-1, Lmax=self.Lmax_atom)
                    atom_indices = padd_matrix(atom_indices, padding_value=-1, Lmax=self.Lmax_atom)
                else:
                    #print("🧪 Removing NaNs from atom tensors...")
                    atom_clouds = remove_nan(atom_clouds, padding_value=0.)
                    atom_triplets = remove_nan(atom_triplets, padding_value=-1)
                    atom_attributes = remove_nan(atom_attributes, padding_value=-1)
                    atom_indices = remove_nan(atom_indices, padding_value=-1)

                #print("✅ Atom feature construction complete")

            except Exception as e:
                print("❌ Error during atom input construction:", e)
                traceback.print_exc()
                return None
        try:
            if labels is not None:
                #print("✅ Entering label processing block")
                #print("  Label dtype:", labels.dtype)
                #print("  Label shape before reshape:", labels.shape)

                if labels.ndim == 1:
                    labels = labels[:, None]
                    #print("✅ Label reshaped to:", labels.shape)

                if labels.dtype in [bool, int, np.bool_, np.int32, np.int64]:
                    #print("🧪 Calling binarize_categorical...")
                    outputs = binarize_categorical(labels, 2).astype(curr_int)
                    #print("✅ binarize_categorical succeeded:", outputs.shape)
                else:
                    #print("⚠️ Labels not int/bool — skipping binarize_categorical()")
                    outputs = labels

                if self.padded:
                    #print("🧪 Padding outputs...")
                    outputs = padd_matrix(outputs, Lmax=self.Lmax_aa, padding_value=0)[np.newaxis]
                else:
                    #print("🧪 Removing NaNs from outputs...")
                    outputs = remove_nan(outputs, padding_value=0.)

                #print("✅ Output processing complete:", outputs.shape if outputs is not None else None)

            else:
#                 print("⚠️ No labels provided — outputs will be None")
                outputs = None

            # Final input assembly before return
            inputs = []
            if self.with_aa:
                inputs += [aa_triplets, aa_attributes, aa_indices, aa_clouds]
            if self.with_atom:
                inputs += [atom_triplets, atom_attributes, atom_indices, atom_clouds]

            #print("🧪 Final input assembly:")
            #for i, item in enumerate(inputs):
            #    print(f"    inputs[{i}]:", None if item is None else getattr(item, 'shape', type(item)))
            #print("  outputs:", outputs.shape if outputs is not None else None)

            return inputs, outputs

        except Exception as e:
            print(f"❌ Error during final steps in process_example: {e}")
            traceback.print_exc()
            return None


    def process_dataset(self, env, label_name=None, permissive=True):
        all_sequences = env['all_sequences']
        all_atom_coordinates = env['all_atom_coordinates']
        all_atom_ids = env['all_atom_ids']

        all_labels = env[label_name] if label_name is not None else None

        if 'PWM' in self.requirements:
            index = targets_Beff.index(self.Beff)
            all_PWMs = np.array([PWM[:, :, index] for PWM in env['all_PWMs']])

        inputs = []
        outputs = []
        failed_samples = []

        B = len(all_sequences)
        for b in range(B):
            self.print_progress(b)

            input_args = {
                'atomic_coordinates': all_atom_coordinates[b],
                'atom_ids': all_atom_ids[b],
                'sequence': all_sequences[b],
            }

            if 'PWM' in self.requirements:
                input_args['PWM'] = all_PWMs[b]

            if all_labels is not None:
                input_args['labels'] = all_labels[b]

            try:
                input_, output_ = self.process_example(**input_args)
                inputs.append(input_)
                outputs.append(output_)
            except Exception as error:
                print(f'Failed to parse example ({b}/{B}), Error: {error}')
                if permissive:
                    failed_samples.append(b)
                    continue
                else:
                    raise ValueError('Failed in non-permissive mode')

        ninputs = len(inputs[0])

        if self.padded:
            inputs = [
                np.concatenate([example[k] for example in inputs], axis=0)
                for k in range(ninputs)
            ]
            outputs = np.concatenate(outputs)
        else:
            inputs = [
                np.array([example[k] for example in inputs])
                for k in range(ninputs)
            ]
            outputs = np.array(outputs)

        return inputs, outputs, failed_samples
