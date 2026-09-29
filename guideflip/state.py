"""The design being transported, from random initialisation to a decided sequence.

Everything the algorithm needs to resume lives in `DesignState`. Three sources argue
about what each open position should be and they are kept apart, because they are
meant to behave differently as the flow proceeds -- that asymmetry is the method.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .structure import NUM_RESTYPES, RESTYPES, UNDECIDED, Complex


@dataclass(frozen=True)
class DesignState:
    """The binder's sequence distribution and the complex it is designed against.

    Three things argue about each open position:

    - `logits` is what the AF-only updates left behind. The AF-only updates evolve it by gradient
      descent; once the ADFlip contribution is enabled it is held fixed, a standing opinion about
      the sequence a binder of this shape should have.
    - `prior` is the *running mean* of everything the inverse-folding model has said.
      A mean rather than a sum, because the prior is the loudest voice in the room:
      summed, it would grow with every update until nothing else could be heard.
      Averaged, its weight stays constant however long the flow runs -- and since
      each prediction is conditioned on a slightly different structure, averaging
      reads a consensus over structures rather than trusting any single one.
    - `guidance` is the *sum* of what AlphaFold has argued. A sum rather than a mean,
      because this is where disagreement has to be able to build: one update cannot
      overturn a confident prior, but an objection repeated over several updates can.

    That asymmetry -- prior averaged, guidance accumulated -- is what lets a position
    the prior likes and AlphaFold dislikes slowly lose its favourite to another amino
    acid.

    Two boolean arrays say what may happen where, and they answer different
    questions:

    - `designable` is fixed for the run: the positions the optimiser may change at
      all. A binder designed from nothing has every position designable, which is a
      value and not a special case; a nanobody has its loops.
    - `committed` grows during the run: the positions the flow has settled. It starts
      as the complement of `designable`, since a position nobody may change is
      settled before the first update, and only ever adds. So `committed` is always a
      superset of `~designable`, and one test covers both.

    Commitment is monotone. A position is fixed when the combined distribution
    crosses the confidence threshold, not by separate prior/AF agreement tests.
    Revision happens before a position is fixed, not after.
    """

    logits: np.ndarray            # (L, NUM_RESTYPES) -- what the AF-only updates left
    committed: np.ndarray         # (L,) bool
    tokens: np.ndarray            # (L,) int, UNDECIDED where not committed
    designable: np.ndarray        # (L,) bool -- fixed for the run
    complex: Complex | None = None
    step: int = 0
    prior: np.ndarray | None = None       # (L, NUM_RESTYPES) running mean
    prior_samples: int = 0
    guidance: np.ndarray | None = None    # (L, NUM_RESTYPES) accumulated

    def __post_init__(self) -> None:
        length, width = self.logits.shape
        if width != NUM_RESTYPES:
            raise ValueError(f"logits must be (L, {NUM_RESTYPES}), got {self.logits.shape}")
        for name in ("committed", "tokens", "designable"):
            if getattr(self, name).shape != (length,):
                raise ValueError(f"{name} must have shape ({length},), got "
                                 f"{getattr(self, name).shape}")
        if bool(np.any(self.committed & (self.tokens == UNDECIDED))):
            raise ValueError("a committed position must carry a token")
        if bool(np.any(~self.designable & ~self.committed)):
            raise ValueError("a position that may not be designed must already be settled")

    @classmethod
    def initial(
        cls,
        binder_length: int,
        rng: np.random.Generator,
        scale: float = 0.01,
        scaffold: str | None = None,
        designable: np.ndarray | None = None,
        seed_designable: float = 0.0,
    ) -> "DesignState":
        """Random initialisation, or a scaffold with only some positions open.

        With a scaffold, every position outside `designable` starts settled on the
        residue the scaffold already has. Its stored logits favour that residue;
        ColabDesign's `fix_pos` supplies exact one-hot inputs during AF-only updates too.

        `seed_designable` puts that much logit on the scaffold's own residue at the
        open positions too. They stay open -- the bias is small enough that a hundred
        AF-only updates move off it freely -- but the design starts from a real loop
        rather than from nothing, which is what a scaffold is for: the fold is given,
        and asking the optimiser to rediscover a plausible loop from noise while it
        also has to find the pose is two problems where there was meant to be one.
        """
        logits = rng.normal(scale=scale, size=(binder_length, NUM_RESTYPES))
        committed = np.zeros(binder_length, dtype=bool)
        tokens = np.full(binder_length, UNDECIDED, dtype=int)
        open_ = np.ones(binder_length, dtype=bool)

        if scaffold is not None:
            if len(scaffold) != binder_length:
                raise ValueError(
                    f"scaffold is {len(scaffold)} residues, binder is {binder_length}")
            open_ = np.zeros(binder_length, dtype=bool)
            if designable is not None:
                open_[np.asarray(designable, dtype=int)] = True
            held = np.flatnonzero(~open_)
            tokens[held] = [RESTYPES.index(scaffold[i]) for i in held]
            committed[held] = True
            logits[held] = 0.0
            logits[held, tokens[held]] = 10.0
            if seed_designable:
                free = np.flatnonzero(open_)
                logits[free, [RESTYPES.index(scaffold[i]) for i in free]] += seed_designable
        elif designable is not None:
            raise ValueError("designable positions name a scaffold's loops; "
                             "a binder designed from nothing has no residues to hold")

        return cls(logits=logits, committed=committed, tokens=tokens, designable=open_)

    @property
    def length(self) -> int:
        return self.logits.shape[0]

    def combined(self, temperature: float = 1.0, retained_weight: float = 1.0) -> np.ndarray:
        """What the three sources say together, in logit space.

        The two priors are mixed first and sharpened together; the guidance is added
        after. The order matters, and getting it wrong couples knobs that should be
        independent.

        Mixing first means `retained_weight` is a ratio between the retained AF-only logits
        and the inverse-folding model's, and stays that ratio whatever the temperature
        is. Sharpening one of them alone would silently demote the other every time
        the temperature changed.

        The guidance is added afterwards, in the units the sharpened priors are
        already in, so its scale means the same thing whatever temperature the priors
        are read at.
        """
        total = retained_weight * self.logits
        if self.prior is not None:
            total = total + self.prior
        total = total / max(temperature, 1e-6)
        if self.guidance is not None:
            total = total + self.guidance
        return total

    def with_prior(self, prediction: np.ndarray) -> "DesignState":
        """Fold one inverse-folding prediction into the running mean."""
        samples = self.prior_samples + 1
        mean = (prediction if self.prior is None
                else self.prior + (prediction - self.prior) / samples)
        return replace(self, prior=mean, prior_samples=samples)

    def with_guidance(self, direction: np.ndarray) -> "DesignState":
        """Add one update's worth of AlphaFold guidance to the running total."""
        total = direction if self.guidance is None else self.guidance + direction
        return replace(self, guidance=total)

    @property
    def time(self) -> float:
        """Flow time: the fraction of positions that have been decided.

        Counted over the whole binder, which for a scaffold means it starts high --
        a nanobody whose framework is settled before the first update is already most
        of the way through. That is what the inverse-folding model should be told,
        since the sequence really is mostly known, and it is also what the commit
        threshold reads, where it means the bar starts near its strict end. Whether
        that is right for a scaffold has not been measured.
        """
        return float(self.committed.mean())

    @property
    def is_complete(self) -> bool:
        return bool(self.committed.all())

    def sequence(self, undecided: str | None = "X", temperature: float = 1.0,
                 retained_weight: float = 1.0,
                 forbidden: np.ndarray | None = None) -> str:
        """Current sequence; undecided positions are shown as `undecided`.

        An open position falls back to what the three sources currently favour, not to
        what the AF-only updates alone favoured -- during the flow `logits` is only one of them.

        `forbidden` has to be given wherever it was given to the flow. A settled
        position cannot hold an excluded amino acid, because the flow masked them
        before deciding; an open one reads its own argmax, which was never masked. So
        without this a design comes back containing exactly the residue that was
        excluded, at exactly the positions that failed to settle *because* it was --
        the prior wanting a cysteine it may not have is why that position never cleared
        the bar, and reporting a cysteine there is the worst available answer.
        """
        combined = self.combined(temperature, retained_weight)
        if forbidden is not None:
            combined = np.where(forbidden[None, :], -np.inf, combined)
        argmax = combined.argmax(axis=-1)
        return "".join(
            RESTYPES[self.tokens[i]] if self.committed[i] else
            (RESTYPES[argmax[i]] if undecided is None else undecided)
            for i in range(self.length)
        )

    def decoded(self, temperature: float = 1.0, retained_weight: float = 1.0,
                forbidden: np.ndarray | None = None) -> str:
        """The designed sequence. Undecided positions fall back to their argmax."""
        return self.sequence(undecided=None, temperature=temperature,
                             retained_weight=retained_weight, forbidden=forbidden)

    def commit(self, positions: np.ndarray, tokens: np.ndarray) -> "DesignState":
        """Settle `positions` on `tokens`, keeping everything already settled.

        The guided update calls this only for newly confident positions. Settled
        identities become context for the prior and are never reopened by the loop.
        """
        if len(positions) == 0:
            return self
        committed = self.committed.copy()
        settled = self.tokens.copy()
        committed[positions] = True
        settled[positions] = tokens
        return replace(self, committed=committed, tokens=settled)

    def with_logits(self, logits: np.ndarray) -> "DesignState":
        return replace(self, logits=logits)

    def with_complex(self, structure: Complex) -> "DesignState":
        return replace(self, complex=structure)

    def advanced(self) -> "DesignState":
        return replace(self, step=self.step + 1)
