# scannet_pytorch/utilities/id_normalization.py
import os, re
PDB_RE = re.compile(r'([0-9][A-Za-z0-9]{3})')

def norm_pdb_id(s: str) -> str:
    """Extract canonical 4-char pdb id (lowercase) from things like '2xnt_J' or '2xnt.cif'."""
    s = os.path.basename(str(s)).lower().replace(".cif", "")
    m = PDB_RE.search(s)
    return m.group(1) if m else ""

def norm_chain(s: str) -> str:
    """Get chain id from '2xnt_J' → 'J'. If none, return ''."""
    tok = str(s).strip().split("_")[-1]
    tok = tok.split(",")[0]
    tok = re.sub(r'[^A-Za-z0-9]', '', tok)
    return tok.upper() if tok else ""
