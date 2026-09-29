"""Stand-ins for the two models, so the loop can be tested without either.

Everything above `guideflip.models` is numpy, and these are what let that be true of
the tests as well: a scorer that returns a gradient towards an answer chosen in
advance, and a prior that is confident about the same one. A loop driven by them
reaches a known sequence, so what the tests check is the machinery rather than a
model's opinion.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guideflip.models.alphafold import FoldResult
from guideflip.structure import NUM_RESTYPES, RESTYPES, Complex, assemble
from pdb_records import chain_lines

TARGET_LENGTH = 6


def rising_then_flat(step: int, peak_at: int = 20) -> float:
    """Confidence that climbs to `peak_at` and then stops, during AF-only updates."""
    return 0.9 * min(step, peak_at) / peak_at


def always(value: float):
    return lambda step: value


def complex_of(binder_length: int, apart: float = 4.0) -> Complex:
    """A two-chain structure of the right lengths, the chains close enough to touch."""
    target = chain_lines("A", ["ALA"] * TARGET_LENGTH)
    binder = chain_lines("B", ["GLY"] * binder_length, start=(apart, 0.0, 0.0))
    return Complex(assemble(target, binder), TARGET_LENGTH, binder_length)


class FakeAlphaFold:
    """A scorer that pulls the sequence towards `answer`.

    The gradient is the one that lowers a loss of "how far is this from the answer",
    so a loop that follows it arrives there. `confidence` says what the prior gate observes
    at each step, which is what decides when it stops.
    """

    def __init__(self, binder_length: int, answer: str,
                 confidence=rising_then_flat, apart: float = 4.0):
        self.binder_length = binder_length
        self.target_length = TARGET_LENGTH
        self.answer = answer
        self._confidence = confidence
        self._apart = apart
        self.calls = 0
        self.seen = []

    def _target_one_hot(self) -> np.ndarray:
        one_hot = np.zeros((self.binder_length, NUM_RESTYPES))
        for position, letter in enumerate(self.answer):
            one_hot[position, RESTYPES.index(letter)] = 1.0
        return one_hot

    def evaluate(self, sequence, *, mixture=0.0, backprop=True) -> FoldResult:
        sequence = np.asarray(sequence, dtype=np.float64)
        if sequence.shape != (self.binder_length, NUM_RESTYPES):
            raise ValueError(f"expected ({self.binder_length}, {NUM_RESTYPES}), "
                             f"got {sequence.shape}")
        self.calls += 1
        self.seen.append(sequence.copy())
        confidence = float(self._confidence(self.calls))
        # d/dp of ||p - answer||^2, so descending it moves towards the answer
        gradient = 2.0 * (sequence - self._target_one_hot()) if backprop else None
        return FoldResult(complex=complex_of(self.binder_length, self._apart),
                          loss=float(np.square(sequence - self._target_one_hot()).sum()),
                          binder_plddt=confidence, interface_ptm=confidence,
                          gradient=gradient, metrics={})


class FakePrior:
    """A prior that is certain about `answer` and says so every time."""

    def __init__(self, binder_length: int, answer: str, strength: float = 8.0):
        self.binder_length = binder_length
        self.answer = answer
        self.strength = strength
        self.calls = 0

    def predict(self, structure, state) -> np.ndarray:
        self.calls += 1
        logits = np.zeros((self.binder_length, NUM_RESTYPES))
        for position, letter in enumerate(self.answer):
            logits[position, RESTYPES.index(letter)] = self.strength
        return logits


class UndecidedPrior(FakePrior):
    """A prior with no opinion, so nothing settles without the guidance."""

    def predict(self, structure, state) -> np.ndarray:
        self.calls += 1
        return np.zeros((self.binder_length, NUM_RESTYPES))
