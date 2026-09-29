"""The structure-conditioned sequence prior.

ADFlip is an all-atom inverse-folding model: given a structure and a partly specified
sequence it says which amino acids belong at the positions still open. Here it is
asked one question, once per update, so that every answer sees the structure as it
currently stands rather than one fixed at the start. It does not sample; GuideFlip
does that.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np

from ..state import DesignState
from ..structure import BINDER_CHAIN, RESTYPES, Complex

BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})

CHECKPOINT_FORMAT = "backflip-adflip-1"


def backbone_only(pdb: str, chain: str = BINDER_CHAIN,
                  keep: np.ndarray | None = None) -> str:
    """Drop the sidechain atoms of one chain, except at the positions in `keep`.

    At a position the design has not settled, AlphaFold's sidechain is its guess at an
    amino acid that has not been chosen; handing it back would be showing the model
    its own answer, so it goes.

    At a settled position the identity is no longer in question, and what the
    sidechain adds is the atomic environment the neighbours sit in -- which is the
    whole reason for using an all-atom model rather than a backbone one. It also
    matches what the model expects: its own sampler removes sidechains exactly where a
    residue is masked and keeps them everywhere else, so stripping a residue whose
    identity is given presents a combination -- known amino acid, no atoms to show for
    it -- that does not arise in training.

    Sidechains are retained as positions commit. A VHH framework starts committed,
    so it retains its sidechains from the first guided update. The target is untouched.

    `keep` is a boolean mask over the chain's residues in order.
    """
    if keep is not None and not np.any(keep):
        keep = None

    lines, seen, ordinal = [], {}, -1
    for line in pdb.splitlines():
        if not (line.startswith("ATOM") and len(line) > 21 and line[21] == chain):
            lines.append(line)
            continue
        residue = line[22:27]
        if residue not in seen:
            ordinal += 1
            seen[residue] = ordinal
        if line[12:16].strip() in BACKBONE_ATOMS:
            lines.append(line)
        elif keep is not None and seen[residue] < len(keep) and keep[seen[residue]]:
            lines.append(line)
    return "\n".join(lines)


class SequencePrior:
    """ADFlip, held fixed and asked once per update.

    Takes a `backflip-adflip-1` inference checkpoint, supplied as
    `adflip/adflip_inference.pt`. It is loaded with `weights_only=True`; training
    checkpoints that require arbitrary Python unpickling are not supported here.
    """

    def __init__(self, checkpoint: str, *, device: str | None = None) -> None:
        import torch

        from adflip.data.all_atom_parse import restype_1to3, token_to_index
        from adflip.data.residue_config import configure as configure_residues
        from adflip.design import DiscreteFlow_AA, Config, build_denoiser

        self._torch = torch
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{checkpoint} is not a {CHECKPOINT_FORMAT} checkpoint; "
                f"use the bundled adflip/adflip_inference.pt")

        config = Config(saved["config"])
        # Whether non-standard residues count as protein decides what the parser calls
        # designable, so it comes from the checkpoint rather than from whatever this
        # module was last left set to.
        configure_residues(include_nonstd_amino_acids=getattr(
            config.data, "include_nonstd_amino_acids", True))

        flow = DiscreteFlow_AA(config, build_denoiser(config, self._device), min_t=0.0)
        flow.load_state_dict(saved["weights"])
        self._flow = flow.to(self._device)
        self._flow.model.eval()

        from adflip.data.all_atom_parse import residue_tokens
        self._mask_token = residue_tokens["<MASK>"]
        # column j of an ADFlip logit row that corresponds to AlphaFold restype j
        self._restype_columns = np.array(
            [token_to_index[restype_1to3[letter]] for letter in RESTYPES], dtype=np.int64)

    def predict(self, structure: Complex, state: DesignState) -> np.ndarray:
        """Amino-acid logits for the binder, conditioned on `structure`.

        Returns an (L, 20) array in AlphaFold restype order. Positions the design has
        already decided are supplied as context, so the answer for the ones still open
        is made in the sequence environment they will actually sit in.
        """
        import adflip.design as design

        torch = self._torch
        handle, path = tempfile.mkstemp(suffix=".pdb")
        os.close(handle)
        try:
            with open(path, "w") as pdb:
                pdb.write(backbone_only(structure.first_model(), keep=state.committed))
            data = design.pdb2data(path, self._device)
        finally:
            os.unlink(path)

        binder = self._binder_positions(data, state.length)
        samples = self._compose_samples(data, binder, state)
        time = torch.tensor([[state.time]], device=self._device)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
                module=r"torch\.nn\.modules\.transformer",
            )
            _, noisy = self._flow.corrupt_data_by_sample(data, state.time, samples)
            logits, _ = self._flow.model(noisy, time)

        return (logits[binder][:, self._restype_columns]
                .detach().float().cpu().numpy().astype(np.float64))

    def _residue_basis(self, data):
        """The atoms that stand for a residue, which is what the model is given.

        A centre atom marks a residue, but the tokens the model reads cover only those
        that also have a usable backbone. Everything indexed per residue here is
        indexed on that basis, so it is computed once.
        """
        basis = data["is_center"].bool() & data["is_protein"].bool()
        if "backbone_mask" in data:
            basis = basis & data["backbone_mask"].bool()
        return basis

    def _binder_positions(self, data, expected: int):
        """Which of those residues are the binder's.

        The structure is assembled target first and binder second, and the parser
        numbers chains as it meets them, so the binder is the last chain. Asked as a
        chain rather than as a list of residue identifiers, which is why nothing here
        needs the parser to hand back a residue-numbering map.

        The count is checked against the binder's length, because a silent mismatch
        here would put the prior's answer for one residue on another.
        """
        basis = self._residue_basis(data)
        chains = data["chain_id"].squeeze()[basis]
        if chains.numel() == 0:
            raise ValueError("the structure has no protein residues with a backbone")
        binder_chain = int(chains.max())
        if binder_chain == 0:
            raise ValueError(
                "the structure has one chain; the prior needs a target and a binder")
        positions = (chains == binder_chain).nonzero(as_tuple=True)[0]
        if positions.numel() != expected:
            raise ValueError(
                f"the last chain has {positions.numel()} residues and the design has "
                f"{expected}; the structure is not the one being designed")
        return positions

    def _compose_samples(self, data, binder, state: DesignState):
        """The token sequence handed to the model: target as it is, binder partly decided."""
        torch = self._torch
        samples = data["residue_token"][self._residue_basis(data)].clone().to(self._device)
        samples[binder] = self._mask_token
        decided = np.flatnonzero(state.committed)
        if decided.size:
            tokens = self._restype_columns[state.tokens[decided]]
            samples[binder[decided]] = torch.as_tensor(
                tokens, device=self._device, dtype=samples.dtype)
        return samples.unsqueeze(0) if samples.dim() == 1 else samples
