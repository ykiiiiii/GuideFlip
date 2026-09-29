"""The AlphaFold design objective.

Confidence terms read from AlphaFold's own heads, and structural terms describing what
a binder should look like: contacts within it and across the interface, compactness,
and optionally helicity. Backpropagating this is what turns AlphaFold from a predictor
into a source of design guidance.
"""
from __future__ import annotations

import numpy as np

from ..settings import LossWeights


def attach(model, weights: LossWeights) -> None:
    """Attach the design objective to a prepared AlphaFold binder model.

    Confidence and contact terms go through the model's own options; the rest are
    registered as additional loss callbacks.
    """
    model.opt["weights"].update({
        "plddt": weights.plddt,
        "pae": weights.pae_intra,
        "i_pae": weights.pae_inter,
        "con": weights.contacts_intra,
        "i_con": weights.contacts_inter,
        # Disable reference-structure losses: redesigned loops and the docking pose
        # are not required to reproduce their input coordinates.
        "dgram_cce": 0.0,
        "fape": 0.0,
        "rmsd": 0.0,
    })
    model.opt["con"].update({
        "num": weights.intra_contact_number,
        "cutoff": weights.intra_contact_distance,
        "binary": False,
        "seqsep": 9,
    })
    model.opt["i_con"].update({
        "num": weights.inter_contact_number,
        "cutoff": weights.inter_contact_distance,
        "binary": False,
    })

    if weights.interface_ptm:
        _add_interface_ptm(model, weights.interface_ptm)
    if weights.radius_of_gyration:
        _add_radius_of_gyration(model, weights.radius_of_gyration)
    if weights.helicity:
        _add_helicity(model, weights.helicity)


def _add_interface_ptm(model, weight: float) -> None:
    """Interface pTM, as a loss to be minimised."""
    from colabdesign.af.loss import get_ptm, mask_loss

    def loss_fn(inputs, outputs):
        return {"i_ptm": mask_loss(1 - get_ptm(inputs, outputs, interface=True))}

    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["i_ptm"] = weight


def _add_radius_of_gyration(model, weight: float) -> None:
    """Penalise a binder less compact than a globular protein of its length."""
    import jax
    import jax.numpy as jnp
    from colabdesign.af.alphafold.common import residue_constants

    ca_index = residue_constants.atom_order["CA"]

    def loss_fn(inputs, outputs):
        ca = outputs["structure_module"]["final_atom_positions"][:, ca_index][
            -model._binder_len:]
        radius = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        expected = 2.38 * ca.shape[0] ** 0.365
        return {"rg": jax.nn.elu(radius - expected)}

    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["rg"] = weight


def _add_helicity(model, weight: float) -> None:
    """Reward i to i+3 contacts within the binder, the signature of a helix."""
    import jax.numpy as jnp
    from colabdesign.af.loss import _get_con_loss, get_dgram_bins

    target_len, binder_len = model._target_len, model._binder_len
    binder_mask = np.append(np.zeros(target_len), np.ones(binder_len))
    mask_2d = np.outer(binder_mask, binder_mask)

    def loss_fn(inputs, outputs):
        if "offset" in inputs:
            offset = inputs["offset"]
        else:
            index = inputs["residue_index"].flatten()
            offset = index[:, None] - index[None, :]
        contacts = _get_con_loss(outputs["distogram"]["logits"],
                                 get_dgram_bins(outputs), cutoff=6.0, binary=True)
        mask = jnp.where(mask_2d, offset == 3, 0)
        return {"helix": jnp.where(mask, contacts, 0.0).sum() / (mask.sum() + 1e-8)}

    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["helix"] = weight
