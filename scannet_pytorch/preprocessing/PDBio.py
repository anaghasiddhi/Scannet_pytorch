import warnings
from numba.core.errors import NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
import os
import gzip
import warnings
from urllib.request import urlretrieve, urlcleanup
import shutil

# Bio.PDB stays as-is
import Bio.PDB
from Bio.PDB import PDBParser, MMCIFParser

# Update to PyTorch-style project structure
from scannet_pytorch.utilities.paths import structures_folder

def is_PDB_identifier(str):
    return ( (len(str) == 4) and str.isalnum() )

def is_UniProt_identifier(str):
    L = len(str)
    correct_length = L in [6,10]
    if not correct_length:
        return False
    only_alnum = str.isalnum()
    only_upper = (str.upper() == str)
    first_is_letter = str[0].isalpha()
    six_is_digit = str[5].isnumeric()

    valid_uniprot_id = correct_length and only_alnum and only_upper and first_is_letter and six_is_digit
    if L == 10:
        seven_is_letter = str[6].isalpha()
        last_is_digit = str[1].isnumeric()
        valid_uniprot_id = valid_uniprot_id and seven_is_letter and last_is_digit
    return valid_uniprot_id


def parse_structure(pdb_path):
    """
    Parses a PDB or CIF structure using Biopython.

    Args:
        pdb_path (str): Path to the structure file.

    Returns:
        structure (Bio.PDB.Structure.Structure): Parsed structure object.
    """
    if pdb_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif pdb_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format for: {pdb_path}")

    structure_id = os.path.basename(pdb_path).split('.')[0]
    return parser.get_structure(structure_id, pdb_path)


#%% Function for downloading biounit files.

class myPDBList(Bio.PDB.PDBList):
    PDB_REF = """
    The Protein Data Bank: a computer-based archival file for macromolecular structures.
    F.C.Bernstein, T.F.Koetzle, G.J.B.Williams, E.F.Meyer Jr, M.D.Brice, J.R.Rodgers, O.Kennard, T.Shimanouchi, M.Tasumi
    J. Mol. Biol. 112 pp. 535-542 (1977)
    http://www.pdb.org/.
    """
    def __init__(self,*args, **kwargs):
        kwargs['pdb'] = structures_folder
        super().__init__(*args,**kwargs)
        self.alphafold_server = 'https://alphafold.ebi.ac.uk/' # entry/Q13469
        self.pdb_server = 'https://files.wwpdb.org'
        self.flat_tree = True

    def retrieve_pdb_file(self, code, obsolete=False, pdir=None, file_format=None, overwrite=False):
        """Fetch PDB structure file from PDB server, and store it locally.

        The PDB structure's file name is returned as a single string.
        If obsolete ``==`` True, the file will be saved in a special file tree.

        NOTE. The default download format has changed from PDB to PDBx/mmCif

        :param code: pdb or uniprot ID
         PDB code: 4-symbols structure Id from PDB (e.g. 3J92).
         Uniprot ID: 6 or 10 symbols (e.g. Q8WZ42).

        :type code: string

        :param file_format:
            File format. Available options:

            * "mmCif" (default, PDBx/mmCif file),
            * "pdb" (format PDB),
            * "xml" (PDBML/XML format),
            * "mmtf" (highly compressed),
            * "bundle" (PDB formatted archive for large structure}
            * 'biounit' (format PDB)

        :type file_format: string

        :param overwrite: if set to True, existing structure files will be overwritten. Default: False
        :type overwrite: bool

        :param obsolete:
            Has a meaning only for obsolete structures. If True, download the obsolete structure
            to 'obsolete' folder, otherwise download won't be performed.
            This option doesn't work for mmtf format as obsoleted structures aren't stored in mmtf.
            Also doesn't have meaning when parameter pdir is specified.
            Note: make sure that you are about to download the really obsolete structure.
            Trying to download non-obsolete structure into obsolete folder will not work
            and you face the "structure doesn't exists" error.
            Default: False

        :type obsolete: bool

        :param pdir: put the file in this directory (default: create a PDB-style directory tree)
        :type pdir: string

        :return: filename
        :rtype: string
        """
        file_format = self._print_default_format_warning(
            file_format)  # Deprecation warning
        file_format = file_format.lower()

        is_pdb = is_PDB_identifier(code)
        is_uniprot = is_UniProt_identifier(code)
        if not (is_pdb or is_uniprot):
            raise ValueError("Specified file_format %s doesn't exist or is not supported. Maybe a typo. Please use one of: mmCif, pdb, xml, mmtf, bundle, biounit" % file_format)

        if is_pdb:
            code = code.lower()


        if is_pdb:
            # Get the compressed PDB structure
            archive = {'pdb': 'pdb%s.ent.gz', 'mmCif': '%s.cif.gz', 'xml': '%s.xml.gz', 'mmtf': '%s',
                       'bundle': '%s-pdb-bundle.tar.gz', 'biounit': '%s.pdb1.gz', 'biounit_mmCif': '%s-assembly1.cif.gz'}
            archive_fn = archive[file_format] % code

            if file_format not in archive.keys():
                raise("Specified file_format %s doesn't exists or is not supported. Maybe a typo. "
                      "Please, use one of the following: mmCif, pdb, xml, mmtf, bundle, biounit" % file_format)

            if file_format in ('pdb', 'mmCif', 'xml'):
                pdb_dir = "divided" if not obsolete else "obsolete"
                file_type = "pdb" if file_format == "pdb" else "mmCIF" if file_format == "mmCif" else "XML"
                url = (self.pdb_server + '/pub/pdb/data/structures/%s/%s/%s/%s' %
                       (pdb_dir, file_type, code[1:3], archive_fn))
            elif file_format == 'bundle':
                url = (self.pdb_server + '/pub/pdb/compatible/pdb_bundle/%s/%s/%s' %
                       (code[1:3], code, archive_fn))
            elif file_format == 'biounit':
                url = (self.pdb_server + '/pub/pdb/data/biounit/PDB/divided/%s/%s' %
                       (code[1:3], archive_fn))
            elif file_format == 'biounit_mmCif':
                url = (self.pdb_server + '/pub/pdb/data/assemblies/mmCIF/divided/%s/%s' %
                       (code[1:3], archive_fn))
            else:
                url = ('http://mmtf.rcsb.org/v1.0/full/%s' % code)

        elif is_uniprot:
            assert file_format in ['pdb','mmCif']
            url = self.alphafold_server + '/files/AF-%s-F1-model_v2%s'%(code, '.pdb' if file_format == 'pdb' else '.cif')
            archive_fn = url.split('/')[-1]
        else:
            return


        # Where does the final PDB file get saved?
        if pdir is None:
            path = self.local_pdb if not obsolete else self.obsolete_pdb
            if not self.flat_tree:  # Put in PDB-style directory tree
                path = os.path.join(path, code[1:3])
        else:  # Put in specified directory
            path = pdir
        if not os.access(path, os.F_OK):
            os.makedirs(path)
        filename = os.path.join(path, archive_fn)
        if is_pdb:
            final = {'pdb': 'pdb%s.ent', 'mmCif': '%s.cif', 'xml': '%s.xml',
                     'mmtf': '%s.mmtf', 'bundle': '%s-pdb-bundle.tar', 'biounit': 'pdb%s.bioent', 'biounit_mmCif': '%s_bioentry.cif'}
        elif is_uniprot:
            final = {'pdb':'AF_%s.pdb','mmCif':'AF_%s.cif'}
        else:
            return
        final_file = os.path.join(path, final[file_format] % code)

        # Skip download if the file already exists
        if not overwrite:
            if os.path.exists(final_file):
                if self._verbose:
                    print("Structure exists: '%s' " % final_file)
                return final_file

        # Retrieve the file
        if self._verbose:
            print("Downloading PDB structure '%s'..." % code)
        try:
            urlcleanup()
            urlretrieve(url, filename)
        except IOError:
            if self._verbose:
                print("Desired structure doesn't exists")
            return
        else:
            if is_pdb:
                with gzip.open(filename, 'rb') as gz:
                    with open(final_file, 'wb') as out:
                        out.writelines(gz)
                os.remove(filename)
            else:
                os.rename(filename,final_file)
        return final_file



class ChainSelect(Bio.PDB.Select):
    def __init__(self,selected_chains,*args,**kwargs):
        self.selected_chains = selected_chains
        super().__init__(*args,**kwargs)
    def accept_model(self,model):
        if self.selected_chains in ['upper','lower','all']:
            return 1
        elif model.id in [x[0] for x in self.selected_chains]:
            return 1
        else:
            return 0
    def accept_chain(self, chain):
        if self.selected_chains == 'all':
            return 1
        elif self.selected_chains == 'upper':
            return int( (chain.get_full_id()[2].isupper() or  (chain.get_full_id()[2]==' ') ) )
        elif self.selected_chains == 'lower':
            return int(chain.get_full_id()[2].islower())
        elif (chain.get_full_id()[1],chain.get_full_id()[2]) in self.selected_chains:
            return 1
        else:
            return 0


def parse_str(identifier):
    str_split = identifier.split('_')

    if len(str_split) == 1:
        structure_identifier = identifier
        chains = None
    else:
        if '.' in str_split[-1]:  # Path-like string with an underscore
            structure_identifier = identifier
            chains = None
        else:
            structure_identifier = '_'.join(str_split[:-1])
            chains = str_split[-1]

    if chains is not None:
        if ('+' in chains) or ('-' in chains):
            chain_identifiers = chains.split('+')
            chain_identifiers = [
                (int(x.split('-')[0]), x.split('-')[1]) if '-' in x else (0, x)
                for x in chain_identifiers
            ]
        else:
            chain_identifiers = [(0, x) for x in chains]
    else:
        chain_identifiers = 'all'

    return structure_identifier, chain_identifiers


def format_chain_id(chain_list):
    return '+'.join(f"{model}-{chain}" for model, chain in chain_list)

def standardize_structure_filename(location, structure_id, ext=".cif"):
    if location and os.path.exists(location):
        new_location = os.path.join(os.path.dirname(location), f"{structure_id}{ext}")
        shutil.move(location, new_location)
        return new_location
    return location

def getPDB(identifier_string, biounit=False, compressed=False, warn=True, verbose=False, structures_folder=None):
    """
    Download mmCIF from RCSB (for PDB IDs) or AlphaFold (for UniProt IDs)
    Returns file path and chain specification as [(model_id, chain_id)]
    """
    if biounit:
        print(f"[Warning] biounit=True passed to getPDB(), but this version ignores it.")

    structure_id, chain = parse_str(identifier_string)
    is_pdb = is_PDB_identifier(structure_id)
    is_uniprot = is_UniProt_identifier(structure_id)

    if is_pdb:
        pdb_id = structure_id.lower().split("_")[0]
        location = os.path.join(structures_folder, f"{pdb_id}.cif")
        if not os.path.exists(location):
            if verbose: 
                print(f"Downloading {pdb_id} from RCSB...")
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
            os.makedirs(structures_folder, exist_ok=True)
            os.system(f"wget -q {url} -O {location}")

    elif is_uniprot:
        uniprot_id = structure_id
        location = os.path.join(structures_folder, f"AF_{uniprot_id}.cif")
        if not os.path.exists(location):
            if verbose: 
                print(f"Downloading AlphaFold {uniprot_id}...")
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v2.cif"
            os.makedirs(structures_folder, exist_ok=True)
            os.system(f"wget -q {url} -O {location}")

    else:  # Handle local file paths
        location = structure_id
        assert os.path.exists(location), f"File not found: {location}"

    if verbose:
        print(f"[getPDB] Resolved {identifier_string} → {location}")
    
    # Normalize chain output to [(0, chain)] or list of tuples
    chain_out = [(0, chain)] if not isinstance(chain, (list, tuple)) else chain
    return location, chain_out


def extract_chains(location, chains, final_location):
    if chains == 'all':
        shutil.copyfile(location, final_location)
    else:
        with warnings.catch_warnings(record=True) as w:
            if location.endswith('.cif'):
                parser = Bio.PDB.MMCIFParser()
            else:
                parser = Bio.PDB.PDBParser()

            struct = parser.get_structure('name', location)
            io = Bio.PDB.PDBIO()

            if isinstance(chains, list) and len(chains) == 1:
                model, chain = chains[0]
                chain_obj = struct[model][chain]
                if len(chain) > 1:
                    chain_obj.id = chain[0]  # Truncate multi-char chain ID
                io.set_structure(chain_obj)
                for atom in Bio.PDB.Selection.unfold_entities(io.structure, 'A'):
                    atom.disordered_flag = 0
                io.save(final_location)
            else:
                io.set_structure(struct)
                io.save(final_location, ChainSelect(chains))

    return final_location



def load_chains(
    pdb_id=None,
    chain_ids='all',
    file=None,
    pdbparser=None,
    mmcifparser=None,
    structures_folder=structures_folder,
    dockground_indexing=False,
    biounit=True,
    verbose=True
):
    if pdbparser is None:
        pdbparser = Bio.PDB.PDBParser(QUIET=True)
    if mmcifparser is None:
        mmcifparser = Bio.PDB.MMCIFParser()

    assert (file is not None) or (pdb_id is not None)

    if file is None and pdb_id is not None:
        file = getPDB(pdb_id, biounit=biounit, structures_folder=structures_folder)[0]
    else:
        pdb_id = file.split('/')[-1].split('.')[0][-4:]

    parser = mmcifparser if file.endswith('.cif') else pdbparser

    if verbose:
        pass #print(f"📦 Parsing structure from: {file}")

    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(pdb_id, file)

    chain_objs = []

    for model in structure:
        for chain in model:
            #print(f"🔍 Found chain: model={model.id}, id={chain.id}")

            if chain_ids == 'all':
                chain_objs.append(chain)

            elif chain_ids == 'lower' and chain.id.islower():
                chain_objs.append(chain)

            elif chain_ids == 'upper' and (chain.id.isupper() or chain.id == ' '):
                chain_objs.append(chain)

            elif isinstance(chain_ids, list):
                # Case: [(model_id, chain_id)]
                if isinstance(chain_ids[0], tuple):
                    if (model.id, chain.id) in chain_ids:
                        #print(f"✅ Selected (model={model.id}, chain={chain.id})")
                        chain_objs.append(chain)
                else:
                    # Case: ['A', 'B']
                    if chain.id in chain_ids:
                        #print(f"✅ Selected chain id={chain.id}")
                        chain_objs.append(chain)

    #print(f"Total selected chains: {len(chain_objs)}")
    #print(f"[DEBUG] file = {file}")
    #print(f"[DEBUG] pdb_id = {pdb_id}")
    #print(f"[DEBUG] All chains in model: {[chain.id for chain in structure[0]]}")
    #print(f"[DEBUG] Selected chain_ids: {chain_ids}")
    #print(f"[DEBUG] Selected chain_objs: {[chain.id for chain in chain_objs]}")

    return structure, chain_objs
