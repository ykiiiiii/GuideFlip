"""PDB complexes, residue mapping, and interface geometry.

Chain A is the target and chain B is the binder. The reference interface is fixed
once; later predictions are aligned against those same contacting residues.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dockq_metrics(model_path, reference_path) -> dict:
    """Official DockQ 2.1.3 iRMSD against the first saved design model.

    Residue positions are known from the pipeline, even when the design PDB was
    decoded from soft sequence probabilities. Renumber both chains by position
    before DockQ's numbering alignment; never sequence-align away substitutions.
    AF2 multi-model predictions are scored individually then averaged.
    """
    import math
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from importlib.metadata import version
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO
    from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

    def parse(path):
        parser = MMCIFParser(QUIET=True) if str(path).endswith('.cif') else PDBParser(QUIET=True)
        return list(parser.get_structure('complex', str(path)))

    references, models = parse(reference_path), parse(model_path)
    reference = references[0]
    expected = {chain: len(list(reference[chain])) for chain in ('A', 'B')}

    def save(model, path):
        if {chain.id for chain in model} != {'A', 'B'}:
            raise ValueError('DockQ requires target A and binder B')
        for chain in model:
            residues = list(chain)
            if len(residues) != expected[chain.id] or any('CA' not in r for r in residues):
                raise ValueError('DockQ residue positions do not match the design reference')
            for i, residue in enumerate(residues):
                residue.id = (' ', 10000 + i, ' ')
            for i, residue in enumerate(residues, 1):
                residue.id = (' ', i, ' ')
        writer = PDBIO()
        writer.set_structure(model)
        writer.save(str(path))

    per_model = []
    with TemporaryDirectory(prefix='guideflip-dockq-') as temporary:
        root = Path(temporary)
        save(reference, root / 'reference.pdb')
        native = load_PDB(str(root / 'reference.pdb'))
        for i, model in enumerate(models):
            path = root / f'model_{i}.pdb'
            save(model, path)
            scores, _ = run_on_all_native_interfaces(load_PDB(str(path)), native,
                                                     chain_map={'A': 'A', 'B': 'B'}, no_align=True)
            score = scores.get('AB', {}).get('iRMSD')
            if score is None or not math.isfinite(score):
                raise ValueError('DockQ could not calculate a finite interface RMSD')
            per_model.append(float(score))
    return dict(irmsd=float(np.mean(per_model)), irmsd_per_model=per_model,
                irmsd_units='angstrom', irmsd_method='DockQ', dockq_version=version('DockQ'),
                irmsd_reference='first_saved_design_model',
                irmsd_mapping='A:A,B:B; residue_position; no_align',
                irmsd_aggregation='model_mean')


#: Amino-acid order used by AlphaFold, and therefore by every array in this package.
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
NUM_RESTYPES = len(RESTYPES)

#: Token value for a position whose identity has not been decided.
UNDECIDED = -1

TARGET_CHAIN = "A"
BINDER_CHAIN = "B"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def read_chain(path: str, chain: str) -> list[str]:
    """The ATOM records of one chain, in file order."""
    with open(path) as handle:
        return [line for line in handle
                if line.startswith("ATOM") and len(line) > 21 and line[21] == chain]


def sequence_of(lines) -> str:
    """The one-letter sequence of some ATOM records, one letter per CA.

    Takes records rather than a structure because that is what both callers have: a
    scaffold read off disk with `read_chain`, and a chain of a predicted complex.
    Residues with no three-letter code here become X.
    """
    return "".join(THREE_TO_ONE.get(line[17:20].strip(), "X")
                   for line in lines if line[12:16].strip() == "CA")


def assemble(target: list[str], binder: list[str] | None = None) -> str:
    """One PDB, target as chain A and binder as chain B.

    ColabDesign reads both chains from one PDB string. GuideFlip withholds the
    interchain template (`rm_template_ic=True`), so these structures need no
    pre-docking. Only chain labels change; coordinates and residue numbers do not.
    """
    lines = [line[:21] + TARGET_CHAIN + line[22:] for line in target]
    if binder:
        lines += [line[:21] + BINDER_CHAIN + line[22:] for line in binder]
    return "".join(lines) + "END\n"


@dataclass(frozen=True)
class Complex:
    """A target-binder complex, held as PDB text."""

    pdb: str
    target_len: int
    binder_len: int

    def first_model(self) -> str:
        """The first model's lines.

        A prediction made with several AlphaFold models is written as several MODEL
        records in one file. Reading straight through would return each residue once
        per model, which silently doubles every chain and makes a per-residue index
        mean something different depending on how many models a design happened to
        run.
        """
        lines = []
        for line in self.pdb.splitlines():
            if line.startswith("ENDMDL"):
                break
            lines.append(line)
        return "\n".join(lines)

    def _ca_lines(self, chain: str):
        return (line for line in self.first_model().splitlines()
                if line.startswith("ATOM") and line[12:16].strip() == "CA"
                and line[21] == chain)

    def ca_coords(self, chain: str) -> np.ndarray:
        """CA coordinates of one chain, in residue order, as an (N, 3) array."""
        return np.asarray(
            [(float(line[30:38]), float(line[38:46]), float(line[46:54]))
             for line in self._ca_lines(chain)],
            dtype=np.float64,
        ).reshape(-1, 3)

    def write(self, path: str) -> str:
        with open(path, "w") as handle:
            handle.write(self.pdb)
        return path


class NoInterface(ValueError):
    """The two chains are not in contact, so there is no interface to measure."""


def _heavy_atoms_by_residue(pdb: str, chain: str) -> tuple[np.ndarray, np.ndarray]:
    """Heavy-atom coordinates of one chain, with the residue each atom belongs to.

    Residues are numbered by the order they appear, which is the same order
    `Complex.ca_coords` returns and therefore the order every index here refers to.
    """
    coordinates: list[tuple[float, float, float]] = []
    residue_of_atom: list[int] = []
    seen: dict[str, int] = {}
    for line in pdb.splitlines():
        if not line.startswith("ATOM") or len(line) < 54 or line[21] != chain:
            continue
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if element == "H":
            continue
        key = line[22:27]                      # residue number plus insertion code
        if key not in seen:
            seen[key] = len(seen)
        coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        residue_of_atom.append(seen[key])
    return (np.asarray(coordinates, dtype=np.float64).reshape(-1, 3),
            np.asarray(residue_of_atom, dtype=int))


def _contacting_residues(pdb: str, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Residues of each chain with any heavy atom within `cutoff` of the other chain."""
    target, target_residue = _heavy_atoms_by_residue(pdb, TARGET_CHAIN)
    binder, binder_residue = _heavy_atoms_by_residue(pdb, BINDER_CHAIN)
    if len(target) == 0 or len(binder) == 0:
        raise NoInterface("a chain is missing from the structure")

    close = np.zeros((target_residue.max() + 1, binder_residue.max() + 1), dtype=bool)
    for start in range(0, len(target), 2048):          # chunked to bound memory
        block = target[start:start + 2048]
        separation = np.linalg.norm(block[:, None, :] - binder[None, :, :], axis=-1)
        touching = separation < cutoff
        if touching.any():
            rows = target_residue[start:start + 2048]
            for row, contacts in zip(rows, touching):
                close[row] |= np.bincount(binder_residue[contacts],
                                          minlength=close.shape[1]).astype(bool)
    return np.flatnonzero(close.any(axis=1)), np.flatnonzero(close.any(axis=0))


def _kabsch(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotation taking `mobile` onto `reference`, both already centred."""
    correlation = mobile.T @ reference
    left, _, right = np.linalg.svd(correlation)
    chirality = np.sign(np.linalg.det(right.T @ left.T))
    return left @ np.diag([1.0, 1.0, chirality]) @ right


def binder_monomer_rmsd(model_path, reference_path, sequence: str) -> dict:
    """Align each monomer's Cα atoms to binder B of the first design model.

    Residues match by position, since the soft design prediction can encode
    different amino-acid identities from the final decoded sequence.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    reference = next(iter(parser.get_structure('design', str(reference_path))))
    residues = list(reference['B'])
    if len(residues) != len(sequence) or any('CA' not in r for r in residues):
        raise ValueError('design binder does not match the monomer sequence length')
    fixed = np.asarray([r['CA'].coord for r in residues], dtype=np.float64)
    fixed -= fixed.mean(axis=0)
    values = []
    for model in parser.get_structure('monomer', str(model_path)):
        if [c.id for c in model] != ['A']:
            raise ValueError('AF2 monomer output must contain only binder chain A')
        residues = list(model['A'])
        if (''.join(THREE_TO_ONE.get(r.resname, 'X') for r in residues) != sequence
                or any('CA' not in r for r in residues)):
            raise ValueError('AF2 monomer output does not match the final binder sequence')
        mobile = np.asarray([r['CA'].coord for r in residues], dtype=np.float64)
        mobile -= mobile.mean(axis=0)
        if not np.isfinite(mobile).all() or not np.isfinite(fixed).all():
            raise ValueError('non-finite coordinates in binder RMSD calculation')
        aligned = mobile @ _kabsch(mobile, fixed)
        values.append(float(np.sqrt(np.mean(np.sum((aligned - fixed) ** 2, axis=1)))))
    if not values:
        raise ValueError('no AF2 monomer structures to compare')
    return dict(rmsd=float(np.mean(values)), rmsd_per_model=values,
                rmsd_units='angstrom', rmsd_atoms='CA',
                rmsd_reference='first_saved_design_model_binder_B',
                rmsd_mapping='monomer_A:design_B; residue_position; binder_superposition',
                rmsd_aggregation='five_model_mean')


@dataclass(frozen=True)
class Interface:
    """The residues in contact in a reference complex, and where they were."""

    target: np.ndarray      # indices into the target chain
    binder: np.ndarray      # indices into the binder chain
    reference: np.ndarray   # (n, 3) their CA coordinates in the reference

    @property
    def size(self) -> int:
        return len(self.reference)

    @classmethod
    def of(cls, structure: Complex, *, cutoff: float = 5.0) -> "Interface":
        """Select the residues of each chain with a heavy atom within `cutoff`."""
        target_residues, binder_residues = _contacting_residues(
            structure.first_model(), cutoff)
        if len(target_residues) + len(binder_residues) < 3:
            raise NoInterface(f"fewer than three residues within {cutoff} A")

        target = structure.ca_coords(TARGET_CHAIN)
        binder = structure.ca_coords(BINDER_CHAIN)
        return cls(target=target_residues, binder=binder_residues,
                   reference=np.vstack([target[target_residues],
                                        binder[binder_residues]]))

    def _coordinates_in(self, structure: Complex) -> np.ndarray:
        """Where this interface's residues are in another prediction of the complex."""
        target = structure.ca_coords(TARGET_CHAIN)
        binder = structure.ca_coords(BINDER_CHAIN)
        if len(target) <= self.target.max() or len(binder) <= self.binder.max():
            raise NoInterface("the structure is shorter than the reference")
        return np.vstack([target[self.target], binder[self.binder]])

    def rmsd(self, structure: Complex) -> float:
        """Interface RMSD of `structure` against the reference.

        Both chains' interface residues are superposed together and the deviation is
        measured over the same set, so the number says how much the *arrangement* of
        the interface changed, not how much either chain moved.
        """
        model = self._coordinates_in(structure)
        centred_model = model - model.mean(axis=0)
        centred_reference = self.reference - self.reference.mean(axis=0)
        rotated = centred_model @ _kabsch(centred_model, centred_reference)
        deviation = rotated - centred_reference
        return float(np.sqrt((deviation**2).sum() / len(deviation)))
