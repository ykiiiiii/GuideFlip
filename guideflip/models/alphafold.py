"""AlphaFold2.3-Multimer as a differentiable scorer of binder sequences.

One call folds a candidate sequence and returns both the predicted complex and the
gradient of the design objective with respect to that sequence. The complex is what
the next update conditions on; the gradient is what steers it. Nothing is computed
twice.
"""
from __future__ import annotations

import dataclasses
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..settings import Design, DesignConfig, Target
from ..structure import NUM_RESTYPES, RESTYPES, Complex
from . import objective


@dataclass(frozen=True)
class FoldResult:
    """What one AlphaFold evaluation yields."""

    complex: Complex
    loss: float
    binder_plddt: float
    interface_ptm: float
    gradient: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)


def withheld_after_folding(target: Target, length: int) -> str | bool:
    """What `rm_target` should be once a structure has been predicted for a sequence.

    The whole chain, because nothing of a prediction should be held onto -- unless the
    caller has already said which part to withhold, in which case that is the answer.
    """
    return target.rm_target or f"A1-{length}"


def fold_target(target: Target, *, directory: str, params_dir: str | None = None,
                num_recycles: int = 3) -> Target:
    """Predict a structure for a target given only its sequence.

    Designing against a sequence is not a separate mode. AlphaFold has to place the
    chain somewhere to fold a binder against it, so a structure is predicted once and
    used as the input -- and then, unless the caller has said otherwise, the whole
    target is marked flexible, so no part of that prediction is held onto and the
    target is free to refold against the binder at every update. What the prediction
    supplies is a starting point, not a constraint.

    A target that already has a structure comes back unchanged.

    This uses the monomer models, not multimer, so it needs `params_model_*_ptm.npz`
    beside the `multimer_v3` parameters the design itself uses. Two sets, and the
    second one is easy to leave out; `guideflip selfcheck` names them.
    """
    if not target.is_predicted:
        return target      # checked before the import: doing nothing needs nothing

    from colabdesign import mk_afdesign_model

    sequence = "".join(target.sequence.split())
    model = mk_afdesign_model(protocol="hallucination", use_multimer=False,
                              num_recycles=num_recycles, data_dir=params_dir,
                              debug=False)
    model.prep_inputs(length=len(sequence))
    model.predict(seq=sequence, num_models=1, verbose=False)

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{target.name}_predicted.pdb")
    model.save_pdb(filename=path, get_best=False)

    return dataclasses.replace(target, structure=path, chain="A",
                               rm_target=withheld_after_folding(target, len(sequence)))


class BinderMonomer:
    """Five AF2-pTM models folding only the final binder sequence."""

    def __init__(self, length: int, *, params_dir: str, num_recycles: int = 3):
        from colabdesign import mk_afdesign_model

        self.length = length
        self.num_recycles = num_recycles
        self.model_names = [f'model_{i}_ptm' for i in range(1, 6)]
        self.model = mk_afdesign_model(
            protocol='hallucination', use_multimer=False, use_templates=False,
            model_names=self.model_names, data_dir=params_dir,
            num_recycles=num_recycles, debug=False)
        if self.model._model_names != self.model_names:
            raise FileNotFoundError('AF2 monomer validation requires all five pTM parameter files')
        self.model.prep_inputs(length=length)

    def predict(self, sequence: str, *, seed: int, path: str) -> dict:
        if len(sequence) != self.length or set(sequence) - set(RESTYPES):
            raise ValueError('invalid final binder sequence for AF2 monomer')
        self.model.predict(seq=sequence, seed=seed, models=self.model_names,
                           num_models=5, sample_models=False, dropout=False,
                           num_recycles=self.num_recycles, verbose=False)
        plddt = np.asarray(self.model.aux['all']['plddt'])
        if (plddt.shape != (5, self.length) or not np.isfinite(plddt).all()
                or np.any((plddt < 0) | (plddt > 1))):
            raise ValueError('AF2 monomer returned invalid pLDDT')
        self.model.save_pdb(filename=path, get_best=False)
        return dict(binder_plddt=float(plddt.mean()),
                    binder_plddt_per_model=plddt.mean(axis=1).tolist(),
                    plddt_scale='0-1', model_count=5, model_names=self.model_names,
                    model_seed=seed, num_recycles=self.num_recycles, dropout=False,
                    msa='none', templates='none', input_chains='binder_only',
                    score_aggregation='five_model_mean')


class AlphaFold:
    """A prepared AlphaFold binder model.

    The design, its templates and the objective are fixed at construction; after that
    the model does exactly one thing: evaluate a binder sequence.
    """

    def __init__(self, design: Design, config: DesignConfig, *,
                 params_dir: str | None = None) -> None:
        from colabdesign import mk_afdesign_model

        if design.target.is_predicted:
            raise ValueError(
                "this target has only a sequence; call fold_target() first to predict "
                "a structure for it")

        self.binder_length = design.binder_length()
        self._config = config
        self._model = mk_afdesign_model(protocol="binder", use_multimer=True,
                                        num_recycles=config.num_recycles,
                                        data_dir=params_dir, best_metric="loss",
                                        debug=False)
        self._model.prep_inputs(**design.prep_arguments(
            omit_amino_acids=config.omit_amino_acids,
            seed=config.seed))
        objective.attach(self._model, config.loss)
        self._target_length = int(self._model._target_len)

        # Hold the scaffold's own positions exactly, using ColabDesign's `fix_pos`,
        # which overwrites them with a one-hot of the scaffold's residues *after* the
        # soft/hard blend rather than trying to survive it.
        if design.binder.is_scaffolded:
            held = design.binder.held_positions()
            if held.size:
                if not hasattr(self._model, "_wt_aatype"):
                    raise RuntimeError(
                        "the model has no reference sequence to hold positions to; "
                        "ColabDesign sets one only for a binder given a chain")
                # Read at trace time, so it has to be in place before the first run.
                self._model.opt["fix_pos"] = held

    @property
    def target_length(self) -> int:
        return self._target_length

    def set_seed(self, seed: int) -> None:
        """Restart the model's own random stream.

        Building the model is expensive enough that a batch builds one and runs every
        trajectory through it, and the numpy generator each trajectory is given does
        not reach this: which two models a run samples, and the dropout inside them,
        come from a jax key that carries on from wherever the previous trajectory left
        it. So the tenth trajectory of one batch and a run of `--seed <its seed>` on
        its own were different experiments sharing a name. Resetting here makes the
        two the same.
        """
        self._model.set_seed(seed)

    def evaluate(self, sequence: np.ndarray, *, mixture: float = 0.0,
                 backprop: bool = True) -> FoldResult:
        """Fold `sequence` and differentiate the design objective through it.

        `sequence` is an (L, 20) array over the binder positions. `mixture` controls
        how far AlphaFold's input is moved from that array towards its softmax: zero
        passes the array through unchanged, which is what the guided flow wants, while
        the AF-only interval uses a fixed positive mixture to keep gradients flowing to raw
        logits.
        """
        if sequence.shape != (self.binder_length, NUM_RESTYPES):
            raise ValueError(f"expected ({self.binder_length}, {NUM_RESTYPES}) "
                             f"sequence, got {sequence.shape}")

        self._model.set_opt(soft=float(mixture), hard=0.0)
        self._model._params = {"seq": np.asarray(sequence, dtype=np.float32)[None]}
        # Preserve the existing phase-specific recycle budgets.
        recycles = (self._config.af_only_recycles if mixture > 0.0
                    else self._config.num_recycles)
        self._model.run(backprop=backprop, num_models=self._config.num_models,
                        sample_models=True, num_recycles=recycles)

        aux = self._model.aux
        gradient = None
        if backprop:
            gradient = np.asarray(aux["grad"]["seq"], dtype=np.float64).reshape(
                self.binder_length, NUM_RESTYPES)

        return FoldResult(complex=self._read_complex(), loss=float(aux["loss"]),
                          binder_plddt=float(1.0 - aux["losses"]["plddt"]),
                          interface_ptm=float(aux["i_ptm"]), gradient=gradient,
                          metrics=dict(aux["log"]))

    def score(self, sequence: str) -> FoldResult:
        """Score a finished sequence with all five AF models and dropout disabled."""
        if not isinstance(sequence, str):
            raise TypeError("final scoring requires an amino-acid sequence string")
        if len(sequence) != self.binder_length:
            raise ValueError(f"expected {self.binder_length} residues, got {len(sequence)}")
        if set(sequence) - set(RESTYPES):
            raise ValueError("final sequence must contain only standard amino acids")
        one_hot = np.zeros((len(sequence), NUM_RESTYPES), dtype=np.float32)
        for position, letter in enumerate(sequence):
            one_hot[position, RESTYPES.index(letter)] = 1.0

        dropout = self._model.opt.get("dropout", True)
        self._model.set_opt(soft=0.0, hard=0.0, dropout=False)
        self._model._params = {"seq": one_hot[None]}
        try:
            self._model.run(backprop=False, model_nums=list(range(5)),
                            num_recycles=self._config.num_recycles)
            aux = self._model.aux
        finally:
            self._model.set_opt(dropout=dropout)
        metrics = dict(aux["log"])
        pae = np.asarray(aux["all"]["pae"])
        n = self._target_length
        minima = np.minimum(pae[:, :n, n:].min(axis=(1, 2)),
                            pae[:, n:, :n].min(axis=(1, 2)))
        metrics.update(ipae_min=float(minima.mean()), ipae_min_per_model=minima.tolist())
        return FoldResult(complex=self._read_complex(), loss=float(aux["loss"]),
                          binder_plddt=float(1.0 - aux["losses"]["plddt"]),
                          interface_ptm=float(aux["i_ptm"]), gradient=None,
                          metrics=metrics)

    def _read_complex(self) -> Complex:
        """Serialise the current prediction into a `Complex`."""
        handle, path = tempfile.mkstemp(suffix=".pdb")
        os.close(handle)
        try:
            self._model.save_pdb(filename=path, get_best=False)
            with open(path) as pdb:
                text = pdb.read()
        finally:
            os.unlink(path)
        return Complex(text, self._target_length, self.binder_length)
