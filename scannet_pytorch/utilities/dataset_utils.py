import os
import sys
import numpy as np

from scannet_pytorch.preprocessing import PDBio
from scannet_pytorch.preprocessing import PDB_processing
from scannet_pytorch.preprocessing import sequence_utils

from scannet_pytorch.utilities.paths import MSA_folder, structures_folder, pipeline_folder
from scannet_pytorch.utilities import io_utils



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
                    f.write('%s %s %s %.4f\n' % (resids[l, -2], resids[l, -1], sequence[l], label[l]))
                else:
                    f.write('%s %s %s %s\n' % (resids[l, -2], resids[l, -1], sequence[l], label[l]))
    return output_file



def read_labels(input_file, nmax=np.inf, label_type='int'):
    list_origins = []
    list_sequences = []
    list_labels = []
    list_resids = []

    with open(input_file, 'r') as f:
        count = 0
        for line in f:
            if (line[0] == '>'):
                if count == nmax:
                    break
                if count > 0:
                    list_origins.append(origin)
                    list_sequences.append(sequence)
                    list_labels.append(np.array(labels))
                    list_resids.append(np.array(resids))

                origin = line[1:-1]
                sequence = ''
                labels = []
                resids = []
                count += 1
            else:
                line_splitted = line[:-1].split(' ')
                resids.append(line_splitted[:2])
                sequence += line_splitted[2]
                if label_type == 'int':
                    labels.append(int(line_splitted[-1]))
                else:
                    labels.append(float(line_splitted[-1]))

    list_origins.append(origin)
    list_sequences.append(sequence)
    list_labels.append(np.array(labels))
    list_resids.append(np.array(resids))

    list_origins = np.array(list_origins)
    list_sequences = np.array(list_sequences)
    list_labels = np.array(list_labels)
    list_resids = np.array(list_resids)
    return list_origins, list_sequences, list_resids, list_labels





def align_labels(labels, pdb_resids, label_resids=None, format="missing"):
    #print("💥  USING **PATCHED** align_labels")
    
    if pdb_resids is None or len(pdb_resids) == 0:
        print("❌ align_labels skipped: empty pdb_resids")
        return None

    if labels.ndim == 1:
        labels = labels[:, None]  # (N, 1)

    label_resids = np.array(label_resids)
    label_len, feat_dim = labels.shape
    seq_len = len(pdb_resids)

    if label_resids is not None:
        # Trim PDB id fields to match label_resids shape
        #print("⚠️ Before slicing pdb_resids:")
        #print("  pdb_resids:", type(pdb_resids), getattr(pdb_resids, 'shape', 'N/A'))
        #print("  label_resids:", type(label_resids), getattr(label_resids, 'shape', 'N/A'))


        if label_resids.ndim >= 2 and label_resids.shape[-1] == 2:
            pdb_resids = pdb_resids[:, -2:]
        elif label_resids.ndim >= 2 and label_resids.shape[-1] == 1:
            pdb_resids = pdb_resids[:, -1:]
        else:
            print(f"⚠️ [align_labels] Skipping due to unexpected label_resids shape: {label_resids.shape}")
            return None


        # 🔍 Diagnostics: print the shape and type before the comparison
        #print("   pdb_resids.shape:", pdb_resids.shape)
        #print("   label_resids.shape:", label_resids.shape)

        pdb_str = np.array(['_'.join(map(str, r)) for r in pdb_resids])
        label_str = np.array(['_'.join(map(str, r)) for r in label_resids])

        #print("   pdb_str.shape:", pdb_str.shape, "| Sample:", pdb_str[:5])
        #print("   label_str.shape:", label_str.shape, "| Sample:", label_str[:5])

        try:
            idx_pdb, idx_lab = np.where(pdb_str[:, None] == label_str[None, :])
        except Exception as e:
            print("❌ align_labels failed during np.where")
            #print("   pdb_str:", pdb_str[:5])
            #print("   label_str:", label_str[:5])
            print("   Error:", e)
            import traceback
            traceback.print_exc()
            raise

        # Initialise output
        if format == "sparse":
            aligned = np.zeros((seq_len, feat_dim), dtype=labels.dtype)
        elif format == "missing":
            fill = -1 if np.issubdtype(labels.dtype, np.integer) else np.nan
            aligned = np.full((seq_len, feat_dim), fill, dtype=labels.dtype)
        else:
            raise ValueError("format not supported")

        # Assign
        aligned[idx_pdb, 0] = labels[idx_lab, 0]

    else:
        assert label_len == seq_len, (
            f"Label length ({label_len}) ≠ sequence length ({seq_len})")
        aligned = labels

    return aligned



def build_dataset(
                  dataset_name,
                  list_origins,
                  list_resids = None,
                  list_labels=None,
                  biounit=True,
                  structures_folder=structures_folder,
                  MSA_folder=MSA_folder,
                  pipeline_folder=pipeline_folder,
                  requirements = ['sequence','atom_coordinate','atom_id','accessible_surface_area','secondary_structure'],
                  verbose=True,
                  Beff=500,
                  nchunks=1
                  ):
    '''
    Use this function to download PDB files, parse them and cache the calculated files.
    '''

    B = len(list_origins)
    if (list_labels is not None):
        assert len(list_labels) == B
        has_labels = True
    else:
        has_labels = False

    if (list_resids is not None):
        assert len(list_resids) == B
        has_resids = True
    else:
        has_resids = False



    fields = [
        'resid',
        'sequence',
    ]

    all_resids = []
    all_sequences = []
    if 'backbone_coordinate' in requirements:
        all_backbone_coordinates = []
        fields.append('backbone_coordinate')
    if 'atom_coordinate' in requirements:
        all_atom_coordinates = []
        fields.append('atom_coordinate')
    if 'atom_id' in requirements:
        all_atom_ids = []
        fields.append('atom_id')
    if 'atom_type' in requirements:
        all_atom_types = []
        fields.append('atom_type')
    if ('PWM' in requirements) | ('conservation' in requirements):
        all_PWMs = []
        all_conservations = []
        all_MSAs = []
        fields.append('PWM')
        fields.append('conservation')
        fields.append('MSA')

    if ('secondary_structure' in requirements) | ('accessible_surface_area' in requirements):
        all_secondary_structures = []
        all_accessible_surface_areas = []
        fields.append('secondary_structure')
        fields.append('accessible_surface_area')

    if any([x in requirements for x in ['halfsphere_excess_up', 'coordination']]):
        all_halfsphere_excess_ups = []
        all_coordinations = []
        fields.append('halfsphere_excess_up')
        fields.append('coordination')

    if any([x in requirements for x in ['backbone_depth', 'sidechain_depth', 'volume_index']]):
        all_backbone_depths = []
        all_sidechain_depths = []
        all_volume_indexes = []
        fields.append('backbone_depth')
        fields.append('sidechain_depth')
        fields.append('volume_index')

    if has_labels:
        all_labels = []
        fields.append('label')

    for b, origin in enumerate(list_origins):
        if verbose:
            print('Processing example %s (%s/%s)' % (origin, b, B))
        pdbfile, chain_ids = PDBio.getPDB(origin, biounit=biounit, structures_folder=structures_folder)
        struct, chain_objs = PDBio.load_chains(file=pdbfile, chain_ids=chain_ids)
        sequences = []
        backbone_coordinates = []
        atom_coordinates = []
        atom_ids = []
        atom_types = []
        for chain_obj in chain_objs:
            sequence,backbone_coordinate,atom_coordinate,atom_id,atom_type=PDB_processing.process_chain(chain_obj)
            sequences.append(sequence)
            backbone_coordinates.append(backbone_coordinate)
            atom_coordinates.append(atom_coordinate)
            atom_ids.append(atom_id)
            atom_types.append(atom_type)

        if ('PWM' in requirements) | ('conservation' in requirements):
            sequences = [PDB_processing.process_chain(chain_obj)[0] for chain_obj in chain_objs]
            output_alignments = [MSA_folder + 'MSA_' + '%s_%s_%s' % (
            pdbfile.split('/')[-1].split('.')[0], chain_id[0], chain_id[1]) + '.fasta' for chain_id in chain_ids]
            MSA_files = [sequence_utils.call_hhblits(sequence, output_alignment) for sequence, output_alignment in
                         zip(sequences, output_alignments)]

            PWMs = [sequence_utils.compute_PWM(MSA_file,Beff=Beff) for MSA_file in MSA_files]
            if 'conservation' in requirements:
                conservations = [sequence_utils.conservation_score(PWM, Beff, Bvirtual=5) for PWM in PWMs]

            if len(MSA_files) == 1:
                MSA_files = MSA_files[0]

        all_sequences.append(''.join(sequences))
        all_resids.append(PDB_processing.get_PDB_indices(chain_objs,return_model=True,return_chain=True))

        if 'backbone_coordinate' in requirements:
            all_backbone_coordinates.append(np.concatenate(backbone_coordinates))
        if 'atom_coordinate' in requirements:
            all_atom_coordinates.append(np.concatenate(atom_coordinates))
        if 'atom_id' in requirements:
            all_atom_ids.append(np.concatenate(atom_ids))
        if 'atom_type' in requirements:
            all_atom_types.append(np.concatenate(atom_types))
        if 'PWM' in requirements:
            all_PWMs.append(np.concatenate(PWMs))

        if 'conservation' in requirements:
            all_conservations.append(np.concatenate(conservations))
            if not 'PWM' in requirements:
                all_MSAs.append(MSA_files)

        if ('secondary_structure' in requirements) | ('accessible_surface_area' in requirements):
            secondary_structure, accessible_surface_area = PDB_processing.apply_DSSP(chain_objs)
            all_secondary_structures.append(secondary_structure)
            all_accessible_surface_areas.append(accessible_surface_area)

        if any([x in requirements for x in ['halfsphere_excess_up', 'coordination']]):
            halfsphere_excess_up, coordination = PDB_processing.ComputeResidueHSE(np.concatenate(backbone_coordinates) )
            all_halfsphere_excess_ups.append(halfsphere_excess_up)
            all_coordinations.append(coordination)

        if any([x in requirements for x in ['backbone_depth', 'sidechain_depth', 'volume_index']]):
            backbone_depth, sidechain_depth, volume_index = PDB_processing.analyze_surface(chain_obj,
                                                                                           np.concatenate(atom_coordinates))
            all_backbone_depths.append(backbone_depth)
            all_sidechain_depths.append(sidechain_depth)
            all_volume_indexes.append(volume_index)

        if has_labels:
            labels = list_labels[b]
            label_resids = list_resids[b] if list_resids is not None else None
            pdb_resids = all_resids[-1]
            all_labels.append(align_labels(labels,pdb_resids,label_resids=label_resids))

    env = {}
    for field in fields:
        env['all_%ss'%field] = np.array(locals()['all_%ss'%field],dtype=np.object)
    env['all_origins'] = np.array(list_origins)
    location = pipeline_folder + dataset_name + '_raw.data'
    io_utils.save_pickle(env,location)
    return location


if __name__ == '__main__':

    database = '/Users/jerometubiana/Downloads/Databases/PIDPI_interaction_database_disprot/70/interface_labels_cv1.txt'
    list_origins, list_sequences, list_resids, list_labels = read_labels(database)

    i = 0
    origin = list_origins[i]
    labels = list_labels[i]
    sequence = list_sequences[i]
    label_resids = list_resids[i]

    pdbfile,chain = PDBio.getPDB(origin,biounit=False)
    structure,chains = PDBio.load_chains(file=pdbfile,chain_ids=chain)
    pdb_resids = PDB_processing.get_PDB_indices(chains,return_model=True,return_chain=True)

    aligned_labels = align_labels(labels[:50], pdb_resids, label_resids=label_resids[:50], format='sparse')

    location = build_dataset('test',
                             list_origins,
                             list_resids=list_resids,
                             list_labels=list_labels,
                             biounit=False,
                             requirements = [
                                 'sequence',
                                 'backbone_coordinate',
                                 'atom_coordinate',
                                 'atom_id',
                                 'atom_type',
                                 'accessible_surface_area',
                                 'secondary_structure',
                                 'halfsphere_excess_up',
                                 'coordination',
                                 # 'PWM',
                                 # 'conservation',
                                 # 'backbone_depth',
                                 # 'sidechain_depth',
                                 # 'volume_index'
                             ]
                             )

    env = io_utils.load_pickle(location)
