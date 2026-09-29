"""One complete guided-flow trajectory and its updates.

Start at run_design(): AlphaFold guides every update, while lambda_t schedules the
ADFlip contribution. The update, confidence schedule, guidance conversion, and
residue commitment are defined below in the order they are used.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import warnings

import numpy as np

from .settings import Design, DesignConfig
from .state import DesignState
from .structure import RESTYPES, UNDECIDED, Complex, Interface, NoInterface
from .models.alphafold import AlphaFold, FoldResult
from .models.prior import SequencePrior


class DesignFailed(RuntimeError):
    """The flow reached its prior-gate budget without any finite confidence state."""


@dataclass(frozen=True)
class DesignResult:
    """One guided-flow trajectory. AF2 validation refolds `sequence` before writing its final structure."""

    sequence: str
    state: DesignState
    complex: Complex
    prior_enabled_at: int
    retained_step: int
    retained_confidence: float
    interface_guarded: bool
    """False, with a warning, if the retained state has no interface to anchor to."""
    trace: list[dict] = field(default_factory=list)
    retained: DesignState | None = None
    """The retained confidence peak used when the ADFlip contribution is enabled."""

    @property
    def is_complete(self) -> bool:
        return self.state.is_complete


def initial_state(config: DesignConfig, design: Design | None,
                  rng: np.random.Generator) -> DesignState:
    """Where a trajectory begins: noise, or a scaffold with its loops open."""
    if design is not None and design.binder.is_scaffolded:
        return DesignState.initial(
            config.binder_length, rng, scaffold=design.binder.sequence(),
            designable=design.binder.designable_positions(),
            seed_designable=config.scaffold_sequence_init)
    return DesignState.initial(config.binder_length, rng)


def run_design(config: DesignConfig, alphafold: AlphaFold, prior: SequencePrior, *,
               rng: np.random.Generator | None = None, on_update=None,
               design: Design | None = None) -> DesignResult:
    """Generate one sequence and complex in a single guided-flow trajectory.

    AlphaFold guides every update. lambda_t controls the ADFlip contribution;
    the existing confidence schedule chooses when it changes from zero to one.
    """
    rng = rng or np.random.default_rng(config.seed)
    monitor = ConfidenceMonitor(config)
    forbidden = forbidden_mask(config.omit_amino_acids, RESTYPES)
    trace: list[dict] = []

    state = initial_state(config, design, rng)

    lambda_t = 0
    flow_progress = 0
    reference, guarded = None, False
    while True:
        update = advance(state, alphafold, prior, config, lambda_t=lambda_t,
                         forbidden=forbidden, reference=reference)
        smoothed = None
        if lambda_t == 0:
            smoothed = monitor.observe(update.state, update.fold.binder_plddt)
        else:
            flow_progress += 1

        state = update.state
        trace.append(_record(update, step=len(trace) + 1, smoothed=smoothed))
        if on_update is not None:
            on_update(update, trace[-1])

        if lambda_t == 0:
            if monitor.exhausted:
                raise DesignFailed(
                    f"no usable confidence in {config.prior_gate_max_steps} updates; AlphaFold "
                    f"returned nothing finite for this target")
            if monitor.should_switch:
                state = retained = monitor.retained
                reference, guarded = _anchor(state, config)
                prior_enabled_at = len(trace) + 1
                lambda_t = 1
        elif state.is_complete or flow_progress >= config.max_steps:
            break

    sequence = state.decoded(config.prior_temperature, config.retained_prior_weight,
                             forbidden)
    return DesignResult(sequence=sequence, state=state, complex=state.complex,
                        prior_enabled_at=prior_enabled_at, retained_step=monitor.best_step,
                        retained_confidence=monitor.best_confidence,
                        interface_guarded=guarded, trace=trace,
                        retained=retained)


def _anchor(state: DesignState, config: DesignConfig) -> tuple[Interface | None, bool]:
    """The interface every later prediction is measured against, chosen once.

    Choosing it once is the point: a selection taken afresh from each prediction would
    follow a drifting binder instead of catching it.

    When the retained complex has no interface there is nothing to anchor to. That is
    reported rather than absorbed, because a flow that runs without the guard is a
    different experiment from one that runs with it.
    """
    if state.complex is None:
        return None, False
    try:
        return Interface.of(state.complex, cutoff=config.interface_cutoff), True
    except NoInterface as why:
        warnings.warn(
            f"the retained complex has no interface to anchor to ({why}), so the "
            f"drift guard cannot act and this flow runs unguarded",
            RuntimeWarning, stacklevel=3)
        return None, False


def _record(update, *, step: int, smoothed: float | None = None) -> dict:
    """One row of the trajectory."""
    row = {"step": step, "state_step": update.state.step,
           "lambda_t": update.lambda_t, "t": float(update.state.time),
           "loss": update.fold.loss,
           "binder_plddt": update.fold.binder_plddt,
           "interface_ptm": update.fold.interface_ptm,
           "decided": float(update.state.time)}
    if smoothed is not None:
        row["smoothed"] = smoothed
    if update.lambda_t == 1:
        row["settled"] = update.settled
        row["interface_rmsd"] = update.interface_rmsd
        row["structure_kept"] = update.structure_kept
    return row


_DESCENT_EPSILON = 1e-7


def normalise(gradient: np.ndarray) -> np.ndarray:
    """Rescale a gradient so its norm is the square root of the live positions.

    This makes the step size independent of how large the raw gradient happens to be
    and of how long the binder is, which is what keeps one learning rate usable across
    targets.
    """
    live = (np.square(gradient).sum(axis=-1, keepdims=True) > 0).sum(axis=-2, keepdims=True)
    magnitude = np.linalg.norm(gradient, axis=(-1, -2), keepdims=True)
    return gradient * np.sqrt(live) / (magnitude + _DESCENT_EPSILON)


@dataclass(frozen=True)
class Update:
    """The result of one update: a new state and the evaluation that produced it."""

    state: DesignState
    fold: FoldResult
    lambda_t: int
    settled: int = 0
    """Positions that crossed the threshold at this update."""
    interface_rmsd: float | None = None
    """How far the new prediction's interface sits from the one the flow started on."""
    structure_kept: bool = True
    """Whether that new prediction was carried forward."""


def _shown_with_held_positions(state: DesignState,
                               logits: np.ndarray) -> np.ndarray:
    """`logits`, with the positions that may not be designed set to their residue.

    A clean one-hot, which is what those positions mean. What AlphaFold is actually
    shown at them is not this array, though: ColabDesign overwrites them after its own
    blend, from `fix_pos` and the scaffold's residues, so the value written here only
    has to be honest rather than to survive a transform. `AlphaFold.__init__` sets
    the fixed positions before the first evaluation.
    """
    if state.designable.all():
        return logits
    held = np.flatnonzero(~state.designable)
    shown = logits.copy()
    shown[held] = 0.0
    shown[held, state.tokens[held]] = 1.0
    return shown


def _descend(state: DesignState, fold: FoldResult,
             config: DesignConfig) -> np.ndarray:
    """One step of normalised gradient descent on the logits.

    A position that may not be designed is not being designed, so the objective has no
    say over it.
    """
    gradient = np.where(state.designable[:, None], fold.gradient, 0.0)
    return state.logits - config.learning_rate * normalise(gradient)


def advance(state: DesignState, alphafold: AlphaFold, prior: SequencePrior,
            config: DesignConfig, *, lambda_t: int,
            forbidden: np.ndarray | None = None,
            reference: Interface | None = None) -> Update:
    """Advance the same flow using AF gradients and the scheduled ADFlip prior."""
    if lambda_t not in (0, 1):
        raise ValueError("lambda_t must be 0 or 1")
    if lambda_t == 0:
        shown = _shown_with_held_positions(state, state.logits)
        fold = alphafold.evaluate(shown, mixture=config.af_only_mixture, backprop=True)
        return Update(state=state.with_logits(_descend(state, fold, config))
                      .with_complex(fold.complex).advanced(), fold=fold, lambda_t=0)

    if state.complex is None:
        raise ValueError("a guided update needs a structure to condition on")

    # 1. Ask the prior about the structure as it stands. Settled positions go in as
    #    context, so the answer for what is still open is consistent with what is not.
    #    The answer joins a running mean rather than a sum: see DesignState.
    state = state.with_prior(prior.predict(state.complex, state))

    # 2. Read the three sources together, at temperature one. See commit_residues.
    proposal = _proposal(state, config, forbidden)

    # 3. Open positions are probabilities; committed ones are their exact one-hot.
    #    Nothing is rounded merely to give AlphaFold a complete sequence.
    fold = alphafold.evaluate(proposal, mixture=0.0, backprop=True)

    # 4. Map the gradient through the open positions' softmax. At committed one-hot
    #    positions the Jacobian is zero, so their identities receive no new guidance.
    state = state.with_guidance(guidance_logits(
        fold.gradient, proposal, scale=config.guidance_scale,
        forbidden=forbidden))

    # 5. Settle whatever now clears the bar, which rises as the flow proceeds.
    step = commit_residues(
        state.combined(config.prior_temperature, config.retained_prior_weight),
        threshold=commit_threshold(state, config), forbidden=forbidden)
    crossing = step.decided & ~state.committed
    positions = np.flatnonzero(crossing)
    state = state.commit(positions, step.tokens[positions])

    # 6. Carry the new prediction forward, unless the interface has moved so far that
    #    it is no longer the complex being designed.
    structure, rmsd, kept = _feedback(state, fold, config, reference)

    return Update(state=state.with_complex(structure).advanced(), fold=fold,
                  lambda_t=1, settled=len(positions), interface_rmsd=rmsd,
                  structure_kept=kept)


def _proposal(state: DesignState, config: DesignConfig,
              forbidden: np.ndarray | None) -> np.ndarray:
    """What the three sources jointly propose, as a probability distribution.

    Every committed position is exact one-hot: the scaffold residue at a fixed
    position, or the flow's chosen residue at a newly settled position. Open
    positions retain the full combined distribution, with forbidden residues masked.
    """
    combined = state.combined(config.prior_temperature, config.retained_prior_weight)
    if forbidden is not None:
        combined = np.where(forbidden[None, :], -np.inf, combined)
    proposal = softmax(combined, 1.0)
    if state.committed.any():
        settled = np.flatnonzero(state.committed)
        proposal[settled] = 0.0
        proposal[settled, state.tokens[settled]] = 1.0
    return proposal


def _feedback(state: DesignState, fold: FoldResult, config: DesignConfig,
              reference: Interface | None) -> tuple:
    """Decide which structure the next update conditions on, and report the drift."""
    rmsd = None
    if reference is not None:
        try:
            rmsd = reference.rmsd(fold.complex)
        except NoInterface:
            rmsd = float("inf")   # the binder is no longer in contact at all

    if rmsd is not None and rmsd > config.feedback_tolerance:
        return state.complex, rmsd, False
    return fold.complex, rmsd, True


def _interpolate(bounds: tuple[float, float], fraction: float) -> float:
    start, end = bounds
    return start + (end - start) * min(max(fraction, 0.0), 1.0)


@dataclass
class ConfidenceMonitor:
    """Decides when the binder has stopped improving, and holds onto its best moment.

    The AF-only interval ends at the *peak* of confidence, not at the update where the peak is
    recognised: the state at the peak is retained and the flow resumes from it.
    Confidence is smoothed first, because a single update's pLDDT is noisy enough to
    end the AF-only updates early by accident.
    """

    config: DesignConfig
    _window: deque = field(init=False)
    _best: float = field(default=float("-inf"), init=False)
    _best_step: int = field(default=0, init=False)
    _retained: DesignState | None = field(default=None, init=False)
    _switched: bool = field(default=False, init=False)
    _observations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=self.config.prior_gate_window)

    def observe(self, state: DesignState, confidence: float) -> float:
        """Record one update. Returns the smoothed confidence."""
        self._observations += 1
        self._window.append(confidence)
        smoothed = sum(self._window) / len(self._window)

        if self._observations >= self.config.prior_gate_min_steps:
            if smoothed > self._best + 1e-4:
                self._best = smoothed
                self._best_step = self._observations
                self._retained = state
            elif self._observations - self._best_step >= self.config.prior_gate_patience:
                self._switched = True
        if self._observations >= self.config.prior_gate_max_steps:
            self._switched = True
        return smoothed

    @property
    def should_switch(self) -> bool:
        """True once the AF-only interval has found its peak and waited out the patience."""
        return self._switched and self._retained is not None

    @property
    def exhausted(self) -> bool:
        """True if the cap was reached without ever retaining a state.

        Any finite confidence beats the initial -inf, so the first observation past the
        floor retains a state and this stays False. It reports the one case that
        cannot: a confidence that is NaN throughout, which no comparison is ever true
        of. That is AlphaFold having failed rather than the design having gone badly,
        and the two deserve different answers.
        """
        return self._observations >= self.config.prior_gate_max_steps and self._retained is None

    @property
    def retained(self) -> DesignState:
        if self._retained is None:
            raise RuntimeError("no state was retained; the AF-only interval never reached its floor")
        return self._retained

    @property
    def best_confidence(self) -> float:
        return self._best

    @property
    def best_step(self) -> int:
        return self._best_step


def commit_threshold(state: DesignState, config: DesignConfig) -> float:
    """Raise the confidence threshold as the fraction of committed residues grows."""
    return _interpolate(config.commit_threshold, state.time)


_GUIDANCE_EPSILON = 1e-8


def to_logit_gradient(gradient: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Map a gradient with respect to probabilities into logit space.

    The softmax Jacobian is ``dp_k/dl_j = p_k (delta_kj - p_j)``, so a gradient ``g``
    over probabilities becomes ``p * (g - <p, g>)`` over logits. Subtracting the inner
    product removes the component along the constant direction, which softmax cannot
    see and which would otherwise appear as an arbitrary offset.

    These are the same probabilities AlphaFold receives. At committed positions
    the input is exactly one-hot, so the Jacobian and the guidance are both zero.
    """
    inner = (probabilities * gradient).sum(axis=-1, keepdims=True)
    return probabilities * (gradient - inner)


def guidance_logits(gradient: np.ndarray, probabilities: np.ndarray, *,
                    scale: float, forbidden: np.ndarray | None = None) -> np.ndarray:
    """The term AlphaFold contributes to the amino-acid logits.

    The sign is chosen so that following the term lowers the design loss. Excluded
    amino acids are zeroed before normalisation, so the guidance budget is redistributed
    over the ones that remain available rather than spent on ones that cannot be used.
    Normalising by the root-mean-square makes `scale` mean the same thing regardless of
    how large the raw gradient happens to be.
    """
    descent = -to_logit_gradient(gradient, probabilities)
    if forbidden is not None:
        descent = descent.copy()
        descent[:, forbidden] = 0.0
    magnitude = np.sqrt(np.mean(descent**2))
    return scale * descent / (magnitude + _GUIDANCE_EPSILON)


def forbidden_mask(omit: str, restypes: str) -> np.ndarray | None:
    """Boolean mask over amino acids excluded from the design."""
    if not omit:
        return None
    excluded = set(omit.upper())
    return np.array([letter in excluded for letter in restypes], dtype=bool)


@dataclass(frozen=True)
class FlowStep:
    """The outcome of one flow update."""

    tokens: np.ndarray        # (L,) decided amino-acid index, UNDECIDED elsewhere
    decided: np.ndarray       # (L,) bool
    confidence: np.ndarray    # (L,) probability of the leading amino acid
    proposal: np.ndarray      # (L, 20) the distribution itself


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax at a given temperature."""
    scaled = logits / max(temperature, 1e-6)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def commit_residues(logits: np.ndarray, *, threshold: float,
            forbidden: np.ndarray | None = None) -> FlowStep:
    """Decide the positions this distribution is confident about.

    `logits` are the sequence prior and the guidance read together, and they are read
    here at temperature one. That matters: this is the only place a probability is
    compared against anything, so it has to be a real probability. Sharpening first
    would put every position above any threshold at once and the flow would collapse
    into a single decoding -- which is also why the sharpening the prior does need,
    `prior_temperature`, is applied upstream where the two priors are mixed, and not
    again here.

    `forbidden` is an optional (20,) boolean mask of amino acids excluded from the
    design.
    """
    if forbidden is not None:
        logits = np.where(forbidden[None, :], -np.inf, logits)

    proposal = softmax(logits, 1.0)
    confidence = proposal.max(axis=-1)
    decided = confidence > threshold
    tokens = np.where(decided, proposal.argmax(axis=-1), UNDECIDED)
    return FlowStep(tokens, decided, confidence, proposal)
