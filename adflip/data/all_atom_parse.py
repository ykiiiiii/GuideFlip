from typing import List
import torch
import numpy as np
import dataclasses
import os
import gzip

from collections import OrderedDict
from Bio.Data import IUPACData
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.spatial import cKDTree


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "<UNK>",
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}


# Tokenization goes from proteins to nucleic acids, to single atoms
residue_tokens = OrderedDict()

residue_tokens["<PAD>"] = len(residue_tokens)
residue_tokens["<MASK>"] = len(residue_tokens)


def add_flag_tokens(flag):
    # Residue names came from prody.flagDefinition. They are a fixed table, and the
    # order they arrive in decides every token index the checkpoint was trained with,
    # so the table is kept literally in residue_config rather than recomputed.
    from adflip.data.residue_config import FLAG_RESIDUES
    for residue in FLAG_RESIDUES[flag]:
        if residue not in residue_tokens:
            residue_tokens[residue] = len(residue_tokens)


add_flag_tokens("stdaa")
residue_tokens["<UNK>"] = len(residue_tokens)

# Extra tokens for ligands
add_flag_tokens("nucleotide")
# add_flag_tokens("nonstdaa")
# add_flag_tokens("nucleic")
# add_flag_tokens("ion")
# residue_tokens["<GLYCAN>"] = len(residue_tokens)


token_to_index = {token: i for i, token in enumerate(residue_tokens)}
index_to_token = {i: token for i, token in enumerate(residue_tokens)}
num_residue_tokens = len(residue_tokens)

# num_protein_tokens = residue_tokens["XAA"] + 1


# Residue type sets — imported from centralized config
from adflip.data.residue_config import get_protein_residues, NUCLEOTIDE, ION
nucleotide_residues = NUCLEOTIDE
ion_residues = ION


# Element tokens
elements = list(IUPACData.atom_weights.keys())
_ = elements.pop(elements.index("H"))  # Remove hydrogen
elements.append("<ATOM_UNK>")
elements.append("<ATOM_PAD>")
elements = [e.upper() for e in elements]

element_to_index = {element: i for i, element in enumerate(elements)}
num_element_tokens = len(elements)


# Data tables ship beside this module. Opened relative to it, not to the working
# directory, so importing the package does not depend on where python was started.
_MISC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "misc")

# AF3 crystallization aids
af3_crystallization_aids = [
    x.strip() for x in open(os.path.join(_MISC, "af3_crystallization_aids.txt")).readlines()
]
af3_crystallization_aids = set(af3_crystallization_aids)


# AF3 ligand exclusion list
af3_ligands_excluded = [
    x.strip() for x in open(os.path.join(_MISC, "af3_ligands_excluded.txt")).readlines()
]
af3_ligands_excluded = set(af3_ligands_excluded)

# Glycans
af3_glycans = [x.strip() for x in open(os.path.join(_MISC, "af3_glycans.txt")).readlines()]
af3_glycans = set(af3_glycans)


def get_token_index(token):
    return token_to_index.get(token, token_to_index["<UNK>"])


def get_element_index(element):
    return element_to_index.get(element, element_to_index["<ATOM_UNK>"])


@dataclasses.dataclass
class StructureData:
    residue_token: np.ndarray
    residue_index: np.ndarray
    residue_atom_index: np.ndarray
    occupancy: np.ndarray
    bfactor: np.ndarray
    chain_id: np.ndarray
    position: np.ndarray
    element_index: np.ndarray
    # atom_name: np.ndarray
    is_ion: np.ndarray
    is_protein: np.ndarray
    is_nucleotide: np.ndarray
    is_center: np.ndarray
    is_backbone: np.ndarray
    backbone_mask: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    not_pad_mask: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    interact_non_protein_res: np.ndarray = np.nan
    interact_ion_res: np.ndarray = np.nan
    interact_nucleotide_res: np.ndarray = np.nan
    interact_molecule_res: np.ndarray = np.nan
    is_mask: np.ndarray = np.nan
    noisy_residue_token: np.ndarray = np.nan
    time_step: np.ndarray = np.nan

    def __len__(self):
        return len(self.residue_token)

    def num_residues(self):
        return len(np.unique(self.residue_index))


@dataclasses.dataclass
class BatchStructureData:
    residue_token: np.ndarray
    residue_index: np.ndarray
    residue_atom_index: np.ndarray
    occupancy: np.ndarray
    bfactor: np.ndarray
    batch_index: np.ndarray
    chain_id: np.ndarray
    position: np.ndarray
    element_index: np.ndarray
    mask_chain: np.ndarray
    # atom_name: np.ndarray
    is_ion: np.ndarray
    is_protein: np.ndarray
    is_nucleotide: np.ndarray
    is_center: np.ndarray
    is_backbone: np.ndarray
    backbone_mask: np.ndarray
    not_pad_mask: np.ndarray
    interact_non_protein_res: np.ndarray
    interact_ion_res: np.ndarray
    interact_nucleotide_res: np.ndarray
    interact_molecule_res: np.ndarray
    noisy_residue_token: np.ndarray
    time_step: np.ndarray

    def __len__(self):
        return len(self.residue_token)


structure_data_fields = StructureData.__dataclass_fields__


def slice_structure_data(
    structure_data: StructureData,
    slice_array: np.array,
) -> StructureData:
    new_data = {}
    for key in structure_data_fields:
        key_obj = getattr(structure_data, key)
        if hasattr(key_obj, "shape") and len(key_obj.shape) > 0:
            if key_obj.shape[0] == slice_array.shape[0]:
                new_data[key] = getattr(structure_data, key)[slice_array]
        else:
            new_data[key] = getattr(structure_data, key)
        if key == "residue_index":
            # new_data[key] -= np.min(structure_data.residue_index)
            new_data[key] = np.unique(new_data[key], return_inverse=True)[1]
    return StructureData(**new_data)


def init_struct_data_dict():
    struct_data = {}
    struct_data["residue_token"] = []
    struct_data["residue_index"] = []
    struct_data["residue_atom_index"] = []
    struct_data["occupancy"] = []
    struct_data["bfactor"] = []
    struct_data["chain_id"] = []
    struct_data["position"] = []
    struct_data["element_index"] = []
    struct_data["is_ion"] = []
    struct_data["is_protein"] = []
    struct_data["is_nucleotide"] = []
    struct_data["is_center"] = []
    struct_data["is_backbone"] = []
    struct_data["backbone_mask"] = []
    struct_data["mask_chain"] = []
    struct_data["not_pad_mask"] = []
    return struct_data


def extend_struct_data_dict(
    struct_data_dict, extension_data_dict, ligand_center=False, ion_center=False
) -> bool:
    # Check if ligand or ion
    if len(extension_data_dict["position"]) == 0:
        return False
    if not (
        extension_data_dict["is_protein"][-1] or
        extension_data_dict["is_nucleotide"][-1] or
        extension_data_dict["is_ion"][-1]
    ): #for ligand
        positions = np.array(extension_data_dict["position"])
        center_idx = np.argmin(
            np.linalg.norm(
                positions - np.mean(positions, axis=0, keepdims=True),
                axis=-1
            ),
            axis=0
        )
        if ligand_center:
            extension_data_dict["is_center"] = [
                i == center_idx for i in range(len(extension_data_dict["position"]))
            ]
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return True
        else:
            extension_data_dict["is_center"] = [
                False for i in range(len(extension_data_dict["position"]))
            ]
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return False
    elif extension_data_dict["is_ion"][-1]:
        if ion_center:
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return True
        else:
            extension_data_dict["is_center"] = [
                False for i in range(len(extension_data_dict["position"]))
            ]
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return False
    else:
        if any(extension_data_dict["is_center"]):
            for key in struct_data_dict:
                struct_data_dict[key].extend(extension_data_dict[key])
            return True
    return False




def _normalize_chain_id(chainid: str):
    chainid = str(chainid).strip()
    if not chainid or chainid in (".", "?"):
        return None
    return chainid


def _parse_mmcif_assemblies(path_or_name: str):
    if not os.path.isfile(path_or_name):
        return None

    if path_or_name.endswith(".gz"):
        opener = lambda p: gzip.open(p, "rt", errors="ignore")
    else:
        opener = lambda p: open(p, "rt", errors="ignore")

    try:
        with opener(path_or_name) as f:
            cif = MMCIF2Dict(f)
    except Exception:
        return None

    def _to_list(val, n=None):
        if val is None:
            return [] if n is None else ["" for _ in range(n)]
        if isinstance(val, str):
            return [val]
        return list(val)

    asmb_ids = _to_list(cif.get('_pdbx_struct_assembly_gen.assembly_id'))
    if not asmb_ids:
        return None

    asmb_chains = _to_list(cif.get('_pdbx_struct_assembly_gen.asym_id_list'), len(asmb_ids))

    return {
        'asmb_ids': asmb_ids,
        'asmb_chains': asmb_chains,
    }

def parse_structure(path_or_name: str):
    if ".pdb" in path_or_name:
        parser = PDBParser(QUIET=True)
        return parser.get_structure("structure", path_or_name)
    parser = MMCIFParser(QUIET=True, auth_chains=False)
    if path_or_name.endswith(".gz"):
        with gzip.open(path_or_name, "rt", errors="ignore") as f:
            return parser.get_structure("structure", f)
    return parser.get_structure("structure", path_or_name)


class _NonWaterHydrogenSelect(Select):
    """PDBIO selector that skips water and hydrogen, consistent with parse_mmcif_to_structure_data."""
    _water = {"HOH", "WAT", "DOD"}

    def accept_residue(self, residue):
        return residue.get_resname().strip() not in self._water

    def accept_atom(self, atom):
        element = (atom.element or "").strip().upper()
        if element in {"H", "D"} or atom.get_name().startswith("H"):
            return 0
        return 1


def cif_to_pdb(cif_path, pdb_path):
    """Convert mmCIF to PDB using Biopython, consistent with parse_structure.

    Uses the same MMCIFParser(auth_chains=False) so the resulting PDB
    contains exactly the same chains/residues that parse_mmcif_to_structure_data sees.

    Handles:
    - Multi-character chain IDs -> remapped to single-character PDB chain IDs
    - Water-only chains -> removed (reduces chain count for PDB format limit)
    - Long residue names (>3 chars) -> truncated to 3 chars for PDB compatibility
    """
    structure = parse_structure(cif_path)

    # Single-char chain IDs: A-Z, 0-9, a-z (total 62)
    pdb_chain_chars = [chr(i) for i in range(ord('A'), ord('Z')+1)] + \
                      [chr(i) for i in range(ord('0'), ord('9')+1)] + \
                      [chr(i) for i in range(ord('a'), ord('z')+1)]

    model = next(structure.get_models())
    chains = list(model.get_chains())

    # Detach all chains first to avoid ID collision during renaming
    for chain in chains:
        model.detach_child(chain.id)

    # Filter out water-only chains (they are skipped by _NonWaterHydrogenSelect anyway)
    water_resnames = {"HOH", "WAT", "DOD"}
    non_water_chains = []
    for chain in chains:
        has_non_water = any(
            r.get_resname().strip() not in water_resnames
            for r in chain.get_residues()
        )
        if has_non_water:
            non_water_chains.append(chain)

    if len(non_water_chains) > len(pdb_chain_chars):
        raise ValueError(
            f"Structure has {len(non_water_chains)} non-water chains, "
            f"exceeding PDB format limit of {len(pdb_chain_chars)} single-char chain IDs."
        )

    # Re-add with single-char IDs; also fix long residue names
    for i, chain in enumerate(non_water_chains):
        chain.id = pdb_chain_chars[i]
        for res in chain:
            # Truncate residue names > 3 chars (PDB format limit)
            resname = res.resname.strip()
            if len(resname) > 3:
                res.resname = resname[:3]
        model.add(chain)

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path, _NonWaterHydrogenSelect())


def parse_mmcif_to_structure_data(path_or_name,parser_chain_id = None,ion_center = False,ligand_center=False) -> StructureData:
    """
    Parse a mmCIF file and return a StructureData object.
    """
    protein_residues = get_protein_residues()

   # ------------ load & pre-filter atoms ------------
    structure = parse_structure(path_or_name)
    model = next(structure.get_models())

    # ------------ init containers ------------
    struct_data = init_struct_data_dict()
    internal_residue_index = 0
    internal_chain_index = -1
    chain_id_map = {}

    water_resnames = {"HOH", "WAT", "DOD"}

    for chain in model:
        chainid = _normalize_chain_id(chain.id)
        if chainid is None:
            continue
        if parser_chain_id is not None and chainid not in parser_chain_id:
            continue

        internal_chain_index += 1
        chain_id_map[chainid] = internal_chain_index

        for residue in chain:
            resname = residue.get_resname().strip()
            if resname in water_resnames:
                continue
            if resname in af3_crystallization_aids:
                continue

            temp_residue_data = init_struct_data_dict()
            residue_atom_count = 0
            backbone_atoms_present = set()

            is_ion_res = resname in ion_residues
            is_protein_res = resname in protein_residues
            is_nucleotide_res = resname in nucleotide_residues

            for atom in residue:
                element = (atom.element or "").strip().upper()
                atom_name = atom.get_name()
                if element in {"H", "D"} or atom_name.startswith("H"):
                    continue

                if resname in af3_glycans:
                    resname_use = "<GLYCAN>"
                else:
                    resname_use = resname

                temp_residue_data["residue_token"].append(get_token_index(resname_use))
                temp_residue_data["residue_index"].append(internal_residue_index)
                temp_residue_data["residue_atom_index"].append(residue_atom_count)
                temp_residue_data["occupancy"].append(atom.get_occupancy() or 0.0)
                temp_residue_data["bfactor"].append(atom.get_bfactor() or 0.0)
                temp_residue_data["chain_id"].append(internal_chain_index)
                temp_residue_data["position"].append(atom.get_coord())
                temp_residue_data["element_index"].append(get_element_index(element))
                temp_residue_data["is_ion"].append(is_ion_res)
                temp_residue_data["is_protein"].append(is_protein_res)
                temp_residue_data["is_nucleotide"].append(is_nucleotide_res)
                temp_residue_data["not_pad_mask"].append(True)
                temp_residue_data["is_backbone"].append(
                    is_protein_res and atom_name in ["N", "CA", "C", "O"]
                )

                if is_protein_res and atom_name in ["N", "CA", "C", "O"]:
                    backbone_atoms_present.add(atom_name)

                if is_protein_res:
                    temp_residue_data["is_center"].append(atom_name == "CA")
                elif is_nucleotide_res:
                    temp_residue_data["is_center"].append(atom_name == "C1'")
                elif is_ion_res:
                    temp_residue_data["is_center"].append(
                        ion_center and residue_atom_count == 0
                    )
                else:
                    temp_residue_data["is_center"].append(
                        ligand_center and residue_atom_count == 0
                    )

                residue_atom_count += 1

            if len(temp_residue_data["position"]) == 0:
                continue

            backbone_complete = is_protein_res and all(
                x in backbone_atoms_present for x in ["N", "CA", "C", "O"]
            )
            temp_residue_data["backbone_mask"] = [
                backbone_complete for _ in range(len(temp_residue_data["position"]))
            ]

            if extend_struct_data_dict(
                struct_data,
                temp_residue_data,
                ligand_center=ligand_center,
                ion_center=ion_center,
            ):
                internal_residue_index += 1

    for key in struct_data:
        struct_data[key] = np.asarray(struct_data[key])

    struct_data["residue_index"] -= struct_data["residue_index"].min()

    if struct_data['mask_chain'].shape[0] == 0:
        del struct_data['mask_chain']

    data = StructureData(**struct_data)

    assembly_info = _parse_mmcif_assemblies(path_or_name)
    if assembly_info is not None:
        asmb_chain_indices = []
        for chains in assembly_info.get("asmb_chains", []):
            idxs = []
            for c in str(chains).split(","):
                c_norm = _normalize_chain_id(c)
                if c_norm is None:
                    continue
                if c_norm in chain_id_map:
                    idxs.append(chain_id_map[c_norm])
            asmb_chain_indices.append(idxs)

        data.asmb_ids = assembly_info.get("asmb_ids", [])
        data.asmb_chains = assembly_info.get("asmb_chains", [])
        data.asmb_chain_indices = asmb_chain_indices
        data.asym_id_to_chain_index = chain_id_map

    return data


def mask_structure_data(struct_data: StructureData, mask_array: np.ndarray):
    """
    Mask the structure data with a boolean array.
    """
    # Masking the data consists of putting in <MASK> tokens for the protein residues
    # And removing the non-backbone atoms for the masked residues

    # The mask array is the same length as the numbe of residues
    # We need to calculate the atom_mask_array
    # This is a boolean array of the same length as the number of atoms

    masked_residues = np.nonzero(mask_array)[0]
    atom_mask_array = np.isin(struct_data.residue_index, masked_residues)
    atom_mask_array = np.logical_and(
        atom_mask_array, np.logical_not(struct_data.is_backbone)
    )
    atom_keep_array = np.logical_not(atom_mask_array)

    struct_data.residue_token[atom_mask_array] = token_to_index["<MASK>"]

    masked_data = {}
    for key in structure_data_fields:
        masked_data[key] = struct_data.__dict__[key][atom_keep_array]
    return StructureData(**masked_data)


def get_closest_n_residues(
    struct_data: StructureData, position: np.ndarray, n: int
) -> StructureData:
    """
    Get the n closest residues to a position.
    """
    # Going to pick residues by center atom
    center_atoms = struct_data.position[struct_data.is_center]
    kdtree = cKDTree(center_atoms)
    _, closest_residue_indices = kdtree.query(position, n)
    closest_residue_indices = np.sort(closest_residue_indices)
    # Now remove residues not selected
    keep_array = np.isin(struct_data.residue_index, closest_residue_indices)

    return slice_structure_data(struct_data, keep_array)


def get_closest_n_atoms(
    struct_data: StructureData,
    position: np.ndarray,
    n: int,
    remove_incomplete_residues: bool = True,
) -> StructureData:
    """
    Get the n closest atoms to a position.
    Throw out atoms that are partially picked from a residue
    """
    kdtree = cKDTree(struct_data.position)
    _, closest_atom_indices = kdtree.query(position, n)
    closest_atom_indices = np.sort(closest_atom_indices)
    closest_atom_mask = np.zeros(len(struct_data.position), dtype=bool)
    closest_atom_mask[closest_atom_indices] = True
    if remove_incomplete_residues:
        residues_to_throw = np.unique(struct_data.residue_index[~closest_atom_mask])
        atoms_to_throw = np.isin(struct_data.residue_index, residues_to_throw)
        closest_atom_mask[atoms_to_throw] = False

    return slice_structure_data(struct_data, closest_atom_mask)


pad_constants = {
    "residue_token": token_to_index["<PAD>"],
    "residue_index": -1,
    "residue_atom_index": -1,
    "occupancy": 0.0,
    "bfactor": 0.0,
    "chain_id": -1,
    "position": 0.0,
    "element_index": element_to_index["<ATOM_PAD>"],
    "atom_name": "",
    "is_ion": False,
    "is_protein": False,
    "is_nucleotide": False,
    "is_center": False,
    "is_backbone": False,
    "backbone_mask": False,
    "not_pad_mask": False,
    "is_mask": False,
    'interact_non_protein_res': False,
    'interact_ion_res': False,
    'interact_nucleotide_res': False,
    'interact_molecule_res': False,
    "noisy_residue_token": token_to_index["<PAD>"],
    "time_step": -1,
}


def pad_structure_data(struct_data: StructureData, pad_length: int):
    """
    Pad a structure data to a certain length.
    """
    pad_data = {}
    for key in structure_data_fields:
        if key != "position" and key != "time_step":
            pad_data[key] = np.pad(
                struct_data.__dict__[key],
                (0, pad_length - len(struct_data)),
                mode="constant",
                constant_values=pad_constants[key],
            )
        elif key == "time_step":
            pad_data[key] = struct_data.__dict__[key]
        else:
            pad_values = np.zeros((pad_length - len(struct_data), 3))
            pad_data[key] = np.concatenate(
                [struct_data.__dict__[key], pad_values], axis=0
            )
    if 'mask_chain' in struct_data.__dict__.keys():
        pad_mask_chain = np.pad(
            struct_data.__dict__["mask_chain"],
            (0, pad_length - len(struct_data)),
            mode="constant",
            constant_values=pad_constants["not_pad_mask"],
        )
    data = StructureData(**pad_data)
    data.mask_chain = pad_mask_chain
    return data

def from_numpy(x_array: np.array) -> torch.Tensor:
    if x_array.dtype == np.float64:
        x_array = x_array.astype(np.float32)
    return torch.from_numpy(x_array)


def batch_structure_data_list(
    struct_data_list: List[StructureData],
    pad_length: int = None,
    to_torch: bool = False,
) -> BatchStructureData:
    """
    Batch a list of structure data.
    """
    try:
        struct_data_list = [x for x in struct_data_list if x is not None]
        if len(struct_data_list) == 0:
            return None
        # struct_data_list = [struct_data for struct_data in struct_data_list if struct_data.is_center.sum() <= 512 and np.unique(struct_data.residue_index).shape[0] == struct_data.is_center.sum()]

        stack_fn = np.stack if not to_torch else torch.stack
        full_fn = np.full if not to_torch else torch.full
        convert_fn = (lambda x: x) if not to_torch else from_numpy

        if pad_length is None:
            pad_length = max(len(struct_data) for struct_data in struct_data_list)

        batch_data = {}
        new_struct_data_list = []
        for struct_data in struct_data_list:
            if len(struct_data) < pad_length:
                new_struct_data_list.append(pad_structure_data(struct_data, pad_length))
            else:
                new_struct_data_list.append(struct_data)

        for key in structure_data_fields:
            if key not in BatchStructureData.__dataclass_fields__ or key in [
                "time_step"
            ]:
                continue
            batch_data[key] = stack_fn(
                [
                    convert_fn(struct_data.__dict__[key])
                    for struct_data in new_struct_data_list
                ]
            )
        batch_data["mask_chain"] = stack_fn(
            [
                convert_fn(struct_data.mask_chain)
                for struct_data in new_struct_data_list
            ])

        batch_data["batch_index"] = stack_fn(
            [full_fn((pad_length,), i) for i in range(len(new_struct_data_list))]
        )
        batch_data["time_step"] = stack_fn(
            [convert_fn(struct_data.time_step) for struct_data in new_struct_data_list]
        )
        return BatchStructureData(**batch_data)

    except Exception as e:
        print(e)
        return None


def dump_structure_data(struct_data: StructureData, path: str):
    """
    Dump a structure data to a file.
    """
    def _to_saveable(value):
        if isinstance(value, dict):
            return np.array([value], dtype=object)
        if isinstance(value, list):
            try:
                with np.warnings.catch_warnings():
                    np.warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
                    arr = np.array(value)
                return arr
            except Exception:
                return np.array(value, dtype=object)
        return value

    data = {k: _to_saveable(v) for k, v in struct_data.__dict__.items()}
    np.savez_compressed(path, **data)


def load_structure_data(path: str) -> StructureData:
    """
    Load a structure data to a file.
    """
    struct_data = np.load(path, allow_pickle=True)
    field_names = set(structure_data_fields.keys())
    data_kwargs = {k: struct_data[k] for k in struct_data.files if k in field_names}
    # Backward compatibility: older .npz may not have backbone_mask
    if "backbone_mask" not in data_kwargs:
        if "residue_index" in data_kwargs and "is_backbone" in data_kwargs:
            residue_index = data_kwargs["residue_index"]
            is_backbone = data_kwargs["is_backbone"]
            backbone_mask = np.zeros_like(is_backbone, dtype=bool)
            for rid in np.unique(residue_index):
                sel = residue_index == rid
                # residue is complete if it has all 4 backbone atoms
                if np.sum(is_backbone[sel]) >= 4:
                    backbone_mask[sel] = True
            data_kwargs["backbone_mask"] = backbone_mask
        elif "is_backbone" in data_kwargs:
            data_kwargs["backbone_mask"] = np.zeros_like(
                data_kwargs["is_backbone"], dtype=bool
            )
    data = StructureData(**data_kwargs)
    for k in struct_data.files:
        if k not in field_names:
            val = struct_data[k]
            if isinstance(val, np.ndarray) and val.dtype.kind == "O" and val.size == 1:
                val = val.item()
            setattr(data, k, val)
    return data


def parse_or_load_mmcif(path_or_name,parser_chain_id = None,ion_center = False,ligand_center = False) -> StructureData:
    if os.path.splitext(path_or_name)[1] == ".npz":
        return load_structure_data(path_or_name)
    else:
        return parse_mmcif_to_structure_data(path_or_name,parser_chain_id,ion_center = ion_center,ligand_center=ligand_center)


def get_example_batch():
    struct_data1 = parse_mmcif_to_structure_data("5a00")
    struct_data2 = parse_mmcif_to_structure_data("7npm")
    struct_data3 = parse_mmcif_to_structure_data("dataset/test_nucleotide/1bc7.pdb")
    struct_data_list = [struct_data1, struct_data2,struct_data3]
    batch_data = batch_structure_data_list(struct_data_list)
    return batch_data



def propagate_mask_vectorized(mask, index):
    # Ensure inputs are tensors
    mask = torch.as_tensor(mask, dtype=torch.bool)
    index = torch.as_tensor(index)

    # Get unique indices and their inverse mapping
    unique_indices, inverse_indices = torch.unique(index, return_inverse=True)

    # Create a tensor to hold the "any True" status for each unique index
    any_true = torch.zeros_like(unique_indices, dtype=torch.bool)

    # Use scatter_reduce to check if any mask is True for each unique index
    any_true.scatter_reduce_(0, inverse_indices, mask, reduce='amax')

    # Expand the any_true tensor to match the original shape
    return any_true[inverse_indices]


def interact_residue(data):
    protein_pos = np.expand_dims(data.position[data.is_protein],axis=1)
    non_protein_pos = np.expand_dims(data.position[~data.is_protein],axis=0)
    ion_pos = np.expand_dims(data.position[data.is_ion],axis=0)
    nucleotide_pos = np.expand_dims(data.position[data.is_nucleotide],axis=0)
    molecule_pos = np.expand_dims(data.position[~data.is_protein&~data.is_ion&~data.is_nucleotide],axis=0)

    dist_non_protein = np.sqrt((np.sum((protein_pos-non_protein_pos)**2,axis=-1)))
    interact_non_protein = np.any(dist_non_protein<5,axis=1)
    dist_ion = np.sqrt((np.sum((protein_pos-ion_pos)**2,axis=-1)))
    interact_ion = np.any(dist_ion<5,axis=1)
    dist_nucleotide = np.sqrt((np.sum((protein_pos-nucleotide_pos)**2,axis=-1)))
    interact_nucleotide = np.any(dist_nucleotide<5,axis=1)
    dist_molecule = np.sqrt((np.sum((protein_pos-molecule_pos)**2,axis=-1)))
    interact_molecule = np.any(dist_molecule<5,axis=1)

    interact_non_protein_res = propagate_mask_vectorized(interact_non_protein,data.residue_index[data.is_protein])
    interact_ion_res = propagate_mask_vectorized(interact_ion,data.residue_index[data.is_protein])
    interact_nucleotide_res = propagate_mask_vectorized(interact_nucleotide,data.residue_index[data.is_protein])
    interact_molecule_res = propagate_mask_vectorized(interact_molecule,data.residue_index[data.is_protein])


    data.interact_non_protein_res = interact_non_protein_res.numpy()
    data.interact_ion_res = interact_ion_res.numpy()
    data.interact_nucleotide_res = interact_nucleotide_res.numpy()
    data.interact_molecule_res = interact_molecule_res.numpy()

    return data

def pdb2data(pdb_file,device = 'cpu',ligand_center=False,ion_center=False):
    data = parse_mmcif_to_structure_data(pdb_file,ligand_center=ligand_center,ion_center=ion_center)
    data = interact_residue(data)
    result = {}
    for k, v in data.__dict__.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v).to(device)
        elif isinstance(v, (int, float)):
            result[k] = torch.Tensor([v])
        # skip non-tensor fields (assembly info lists/dicts)
    result['batch_index'] = torch.zeros_like(result['residue_index'])
    result = {
        k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else v for k, v in result.items()
    }
    return result



def switch_pdb_order_add_ligand(new_pdb_path, original_pdb_path, confidence=None):
    new_lines = []
    temp_line = None
    ligand_lines = []

    # Read the original PDB to extract ligand information
    with open(original_pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                ligand_lines.append(line)

    # Process the new PDB file
    current_residue = None
    current_chain = None
    confidence_index = -1
    with open(new_pdb_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                residue_number = int(line[22:26].strip().rstrip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') or '0')
                chain_id = line[21]
                if residue_number != current_residue or chain_id != current_chain:
                    current_residue = residue_number
                    current_chain = chain_id
                    if confidence is not None:
                        confidence_index = (confidence_index + 1) % len(confidence)

                atom_name = line[12:16].strip()
                if atom_name == "CB":
                    atom_index = int(line[6:11])
                    temp_line = line[:6] + f"{atom_index + 1:5}" + line[11:]
                    if confidence is not None:
                        temp_line = temp_line[:60] + f"{confidence[confidence_index]:6.2f}" + temp_line[66:]
                elif atom_name == "O":
                    atom_index = int(line[6:11])
                    new_line = line[:6] + f"{atom_index - 1:5}" + line[11:]
                    if confidence is not None:
                        new_line = new_line[:60] + f"{confidence[confidence_index]:6.2f}" + new_line[66:]
                    new_lines.append(new_line)
                    if temp_line is not None:
                        new_lines.append(temp_line)
                    temp_line = None
                else:
                    new_line = line
                    if confidence is not None:
                        new_line = new_line[:60] + f"{confidence[confidence_index]:6.2f}" + new_line[66:]
                    new_lines.append(new_line)
            elif line.startswith('TER'):
                new_lines.append(line)
            elif line.startswith('END'):
                new_lines.append(line)
                new_lines.extend(ligand_lines)
                break
            else:
                new_lines.append(line)

    # Write the new PDB file
    with open(new_pdb_path, 'w') as f:
        f.writelines(new_lines)

