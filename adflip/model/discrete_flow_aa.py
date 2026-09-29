"""The flow's data corruption, which is all GuideFlip asks of it.

Upstream's DiscreteFlow_AA both corrupts data and samples from the model. GuideFlip
samples for itself -- the flow, the guidance and the decisions all live in the
guideflip package -- and asks this class for one thing: given a structure and the
tokens decided so far, prepare the input the denoiser reads. So `corrupt_data_by_sample`
is kept, byte for byte, and the rest is not.

Removed with it: the training loop (`compute_loss`, `test`, `forward`), the samplers
(`sample`, `adaptive_sample`, `_prepare_sampling_io`), and PIPPack sidechain packing
(`load_sc_model`, `sc_packing`). Those carried every heavy import this module had --
PIPPack, prody, matplotlib, yaml, pickle -- and none of them survive.

Upstream: https://github.com/ykiiiiii/ADFLIP at daa01da, model/discrete_flow_aa.py.
"""
import torch
import torch.nn as nn

from adflip.data.all_atom_parse import residue_tokens


class DiscreteFlow_AA(nn.Module):
    """Holds the denoiser and prepares what it is shown.

    The signature keeps `config` and `min_t` because the checkpoint was written against
    a module of this shape and its parameters have to land where they did.
    """

    def __init__(self, config, model, min_t=0.0, **kwargs):
        super().__init__()
        self.config = config
        self.model = model
        self.min_t = min_t
        self.label_smoothing = config.training.label_smoothing

    _CORRUPT_SKIP_FIELDS = frozenset({
        "noisy_residue_token", "interact_non_protein_res", "interact_ion_res",
        "interact_nucleotide_res", "interact_molecule_res", "is_mask",
    })

    def corrupt_data_by_sample(self, data, time, sample):
        """Corrupt data for one sampling step: inject current samples, remove sidechains of masked residues."""
        device = data["residue_token"].device
        noisy_data = {k: v.squeeze().clone() for k, v in data.items()}

        # Step 1: replace designable positions with current sample tokens
        designable_mask = noisy_data["is_center"].bool() & noisy_data["is_protein"].bool()
        if "backbone_mask" in noisy_data:
            designable_mask = designable_mask & noisy_data["backbone_mask"].bool()
        noisy_data["residue_token"][designable_mask] = sample

        # Step 2: which center residues are still <MASK>?
        center_tokens = noisy_data["residue_token"][noisy_data["is_center"].bool()]
        residue_is_masked = (center_tokens == residue_tokens["<MASK>"])

        # Step 3: propagate <MASK> from center to all atoms of masked residues
        is_protein = noisy_data["is_protein"].bool()
        res_idx_protein = noisy_data["residue_index"][is_protein]
        if res_idx_protein.numel() > 0:
            assert center_tokens.shape[0] > res_idx_protein.max().item()
        atom_masked = residue_is_masked[res_idx_protein]
        noisy_data["residue_token"][is_protein] = torch.where(
            atom_masked,
            torch.tensor(residue_tokens["<MASK>"], device=device),
            noisy_data["residue_token"][is_protein],
        )

        # Step 4: build keep_mask — remove sidechain atoms of masked residues
        #   backbone atoms are always kept; sidechain atoms only kept if residue is unmasked
        keep_mask = torch.ones_like(noisy_data["residue_token"], dtype=torch.bool)
        sidechain_protein = ~noisy_data["is_backbone"].bool() & is_protein
        keep_mask[sidechain_protein] = ~residue_is_masked[noisy_data["residue_index"][sidechain_protein]]

        # Step 5: apply keep_mask to all per-atom fields
        time_tensor = torch.ones((1, 1), device=device) * time
        for name, item in noisy_data.items():
            if name == "time_step":
                noisy_data[name] = time_tensor
            elif name not in self._CORRUPT_SKIP_FIELDS:
                noisy_data[name] = item[keep_mask].unsqueeze(0)

        return data, noisy_data


