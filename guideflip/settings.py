"""Design inputs, numerical defaults, and validation selection.

Structures are resolved relative to the settings JSON. The same settings feed the
design models, guided flow, and selected validators.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
import math
import os

import numpy as np

from .structure import BINDER_CHAIN, TARGET_CHAIN, assemble, read_chain, sequence_of


@dataclass(frozen=True)
class LossWeights:
    """AlphaFold objective weights; compactness and helicity are optional constraints."""

    plddt: float = 0.1
    pae_intra: float = 0.4
    pae_inter: float = 0.1
    contacts_intra: float = 1.0
    contacts_inter: float = 5.0
    interface_ptm: float = 0.05
    radius_of_gyration: float = 0.0
    helicity: float = 0.0

    intra_contact_number: int = 2
    intra_contact_distance: float = 14.0
    inter_contact_number: int = 1
    inter_contact_distance: float = 20.0


@dataclass(frozen=True)
class DesignConfig:
    """One trajectory: One guided flow with a scheduled ADFlip contribution."""

    binder_length: int

    # ADFlip gate: fixed logits/softmax blend until smoothed binder pLDDT plateaus.
    af_only_mixture: float = 0.5
    prior_gate_min_steps: int = 100
    prior_gate_window: int = 15
    prior_gate_patience: int = 10
    prior_gate_max_steps: int = 500
    learning_rate: float = 0.1

    # With ADFlip enabled: mean prior logits, retained logits and accumulated guidance.
    max_steps: int = 60
    prior_temperature: float = 0.2
    """Temperature applied to the combined AF-only updates and ADFlip priors."""
    retained_prior_weight: float = 0.4
    """Weight of the retained AF-only updates logits relative to the ADFlip mean."""
    guidance_scale: float = 0.3
    """RMS scale of each guidance increment, constant throughout the flow."""
    commit_threshold: tuple[float, float] = (0.90, 0.95)
    """Confidence threshold interpolated over the fraction of committed residues."""

    # Every update considers new structural feedback against the same reference.
    feedback_tolerance: float = 10.0
    """Maximum interface CA-RMSD in angstroms for accepting a new structure."""
    interface_cutoff: float = 5.0
    """Heavy-atom contact distance in angstroms used to define the reference interface."""

    scaffold_sequence_init: float = 0.0
    """Initial logit bias toward the scaffold residue at designable positions."""
    omit_amino_acids: str = "C"

    num_models: int = 2
    """AF models sampled per design update, with dropout enabled."""
    num_recycles: int = 3
    af_only_recycles: int = 1

    seed: int = 0
    loss: LossWeights = field(default_factory=LossWeights)

    def __post_init__(self) -> None:
        if self.binder_length <= 0:
            raise ValueError("binder_length must be positive")
        if not 0.0 <= self.af_only_mixture <= 1.0:
            raise ValueError("af_only_mixture must lie in [0, 1]")
        if not 1 <= self.prior_gate_min_steps <= self.prior_gate_max_steps:
            raise ValueError("prior_gate_min_steps must lie in [1, prior_gate_max_steps]")
        if self.prior_gate_patience < 1 or self.prior_gate_window < 1:
            raise ValueError("prior_gate_patience and prior_gate_window must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive; guided flow cannot be skipped")
        if not all(0.0 < bound < 1.0 for bound in self.commit_threshold):
            raise ValueError("commit_threshold values must lie strictly in (0, 1)")
        if (self.feedback_tolerance is None or not math.isfinite(self.feedback_tolerance)
                or self.feedback_tolerance <= 0):
            raise ValueError("feedback_tolerance must be finite and positive")


def residue_numbers(lines) -> list[str]:
    """The PDB residue identifiers of some ATOM records, in order, once each."""
    seen = []
    for line in lines:
        key = line[22:27].strip()
        if not seen or key != seen[-1]:
            if key in seen:
                raise ValueError(f"residue {key} appears in two places in the chain")
            seen.append(key)
    return seen


def positions(spec: str | None, numbers: list[str], chain: str) -> np.ndarray:
    """Sequence indices for a residue-range string, the way the field writes them.

    "B26-33,B51-57" names PDB residue numbers, and what the design needs is where those
    sit in the chain's own order. Those coincide only while a chain is numbered from
    one without gaps, which is usual and is not guaranteed -- a construct with a
    disordered loop removed is numbered around the hole. So the mapping is done rather
    than assumed, and a number that is not in the chain is an error rather than an
    index quietly one place out.

    `None` or an empty string names every position.
    """
    if spec is None or not spec.strip():
        return np.arange(len(numbers))

    index_of = {number: index for index, number in enumerate(numbers)}
    chosen: list[int] = []
    for part in spec.split(","):
        part = original = part.strip()
        if not part:
            continue
        # A leading letter is a chain only when a residue follows it. Stripping it
        # from anything else turns a typo into a complaint about the wrong thing.
        if part[:1].isalpha() and part[1:2].isdigit():
            if part[0] != chain:
                raise ValueError(f"{original!r} names chain {part[0]}, not {chain}")
            part = part[1:]
        first, dash, last = part.partition("-")
        if not first.isdigit() or (dash and not last.isdigit()):
            raise ValueError(f"{original!r} is not a residue or a range of them")
        wanted = ([first] if not dash
                  else [str(n) for n in range(int(first), int(last) + 1)])
        for number in wanted:
            if number not in index_of:
                raise ValueError(
                    f"residue {number} is not in chain {chain}, which runs "
                    f"{numbers[0]}-{numbers[-1]}")
            chosen.append(index_of[number])
    return np.array(sorted(set(chosen)), dtype=int)


def _ranges(chosen: list[str]) -> list[str]:
    """Consecutive residue numbers written as ranges, so `26,27,...,33` reads `26-33`.

    Only for legibility: a comma-separated list of every residue says the same thing.
    Worth the few lines because this is what ends up in the arguments a design is run
    with, and three nanobody loops are twenty-nine numbers.
    """
    runs: list[list[int]] = []
    for number in sorted(int(each) for each in chosen):
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return [str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs]


@dataclass(frozen=True)
class Target:
    """The protein a binder is designed against.

    Given as a structure, or as a sequence for which one is predicted once and then
    marked flexible throughout -- which is not a different kind of design but the same
    machinery a partly disordered target already uses, applied to all of it.
    """

    structure: str | None = None
    sequence: str | None = None
    chain: str = TARGET_CHAIN
    hotspot: str | None = None
    rm_target: str | bool = False
    name: str = "target"

    def __post_init__(self) -> None:
        if self.structure is None and not self.sequence:
            raise ValueError("a target needs either a structure or a sequence")

    @property
    def is_predicted(self) -> bool:
        """True when a structure has still to be folded for it."""
        return self.structure is None

    def records(self) -> list[str]:
        return read_chain(self.structure, self.chain)


@dataclass(frozen=True)
class Binder:
    """What is being designed.

    From nothing, it is a length and every position is open. On a scaffold it is a
    chain of a structure, with `designable` naming the positions the optimiser may
    change and `rm_binder` the ones whose template is withheld -- two questions, kept
    apart, and for a nanobody answered with the same loops.
    """

    length: int | tuple[int, int] | None = None
    scaffold: str | None = None
    chain: str = BINDER_CHAIN
    designable: str | None = None
    rm_binder: str | bool | None = None
    """Binder residues whose structural template is withheld.

    `None` derives the selection from `designable`. A residue range removes only
    those positions; `True` removes the entire binder template.
    """

    def __post_init__(self) -> None:
        if (self.scaffold is None) == (self.length is None):
            raise ValueError("a binder is either a length or a scaffold, not both "
                             "and not neither")
        if self.scaffold is None and self.designable is not None:
            raise ValueError("designable positions name a scaffold's loops; a binder "
                             "designed from nothing has none to hold")
        if self.length is not None:
            is_range = isinstance(self.length, (list, tuple))
            values = self.length if is_range else (self.length,)
            if ((is_range and len(values) != 2)
                    or any(isinstance(value, bool)
                           or not isinstance(value, (int, np.integer))
                           or value <= 0 for value in values)):
                raise ValueError("binder_len must be a positive integer or "
                                 "[min, max] with two positive integers")
            if is_range:
                if values[0] > values[1]:
                    raise ValueError("binder_len range requires min <= max")
                object.__setattr__(self, "length", tuple(int(n) for n in values))
            else:
                object.__setattr__(self, "length", int(self.length))

    @property
    def is_scaffolded(self) -> bool:
        return self.scaffold is not None

    def records(self) -> list[str]:
        return read_chain(self.scaffold, self.chain)

    def sequence(self) -> str:
        return sequence_of(self.records())

    def designable_positions(self) -> np.ndarray:
        """Sequence indices of the positions the optimiser may change."""
        records = self.records()
        return positions(self.designable, residue_numbers(records), self.chain)

    def held_positions(self) -> np.ndarray:
        """Sequence indices of the positions the optimiser may not change."""
        held = np.ones(len(self.sequence()), dtype=bool)
        held[self.designable_positions()] = False
        return np.flatnonzero(held)

    def withheld_template(self) -> str | bool:
        """`rm_binder`, as ColabDesign reads it: against the assembled complex.

        This is the one translation in the package, and it exists because the two
        fields are read against different things. `designable` is resolved here,
        against the scaffold's own chain -- the file the user is looking at. ColabDesign
        resolves `rm_binder` against the assembled complex, where the binder has been
        relabelled chain B. So the same string means different residues in the two
        fields, and `26-33` in both names the loops in one and a stretch of the
        *target* in the other, silently.

        Both are parsed here instead, once, and emitted in the form ColabDesign reads.
        An unprefixed range means the binder, because in this field there is nothing
        else it could mean; `A26-33` is refused rather than obeyed.
        """
        if self.rm_binder is not None and not isinstance(self.rm_binder, str):
            return self.rm_binder            # a plain bool: all of it, or none
        numbers = residue_numbers(self.records())
        if self.rm_binder is None:
            # Follow the loops. With no loops named, every position is designable and
            # withholding the whole binder's template is what that means.
            if self.designable is None:
                return True
            chosen = self.designable_positions()
        else:
            chosen = positions(self.rm_binder, numbers, BINDER_CHAIN)
        if len(chosen) == len(numbers):
            return True
        if not len(chosen):
            return False
        return ",".join(f"{BINDER_CHAIN}{part}"
                        for part in _ranges([numbers[index] for index in chosen]))


@dataclass(frozen=True)
class Design:
    """A target and a binder, and the one structure AlphaFold is given for them."""

    target: Target
    binder: Binder
    overrides: dict = dataclasses.field(default_factory=dict)
    """Overrides of the numerical defaults in `DesignConfig`."""
    filters: dict = dataclasses.field(default_factory=dict)
    """Optional final-score thresholds; interpreted by the root design.py workflow."""
    trajectories: int = 1
    af3: dict | None = None

    def assembled(self) -> str:
        """Target as chain A and binder as chain B, concatenated and not placed."""
        binder = self.binder.records() if self.binder.is_scaffolded else None
        return assemble(self.target.records(), binder)

    def binder_length(self) -> int:
        if isinstance(self.binder.length, tuple):
            raise ValueError("resolve binder_len with design.for_seed(seed) first")
        return (len(self.binder.sequence()) if self.binder.is_scaffolded
                else int(self.binder.length))

    def for_seed(self, seed: int) -> "Design":
        """Resolve a length range once, before preparing a trajectory's models.

        A separate generator keeps length sampling out of the design's random
        stream. Fixed lengths and scaffolds are returned unchanged.
        """
        if not isinstance(self.binder.length, tuple):
            return self
        lower, upper = self.binder.length
        length = int(np.random.default_rng(seed).integers(lower, upper + 1))
        return dataclasses.replace(self, binder=dataclasses.replace(
            self.binder, length=length))

    def prep_arguments(self, *, omit_amino_acids: str, seed: int) -> dict:
        """What ColabDesign's binder protocol is called with.

        The ranges arrive against the assembled complex, where the target is chain A
        and the binder chain B. Input PDB residue numbers are preserved.
        """
        arguments = dict(
            pdb_filename=self.assembled(),
            chain=TARGET_CHAIN,
            hotspot=self.target.hotspot or None,
            rm_target=self.target.rm_target or False,
            # Omitting nothing is a real choice -- a scaffold whose own loops carry a
            # cysteine cannot be started from if C is forbidden -- but ColabDesign
            # splits this on commas and an empty string yields one empty code, which it
            # then looks up and dies on. None is how you say "none".
            rm_aa=omit_amino_acids or None,
            seed=seed,
        )
        if self.binder.is_scaffolded:
            arguments.update(
                binder_chain=BINDER_CHAIN,
                rm_binder=self.binder.withheld_template(),
                rm_binder_seq=True,
                rm_binder_sc=True,
                rm_template_ic=True,
            )
        else:
            arguments["binder_len"] = self.binder_length()
        return arguments

    @classmethod
    def from_json(cls, path: str) -> "Design":
        """Read a design description.

        Field names are ColabDesign's, so a reader who knows one knows the other.

        Structures are resolved relative to the file that names them, so a settings
        file and the structures beside it move together. An absolute path is left
        alone.
        """
        with open(path) as handle:
            raw = json.load(handle)
        unknown = set(raw) - _KNOWN_FIELDS - {field for field in raw
                                              if field.startswith("_")}
        if unknown:
            raise ValueError(f"{path}: unknown field(s) {sorted(unknown)}; "
                             f"known are {sorted(_KNOWN_FIELDS)}")

        beside = os.path.dirname(os.path.abspath(path))
        overrides = raw.get("config") or {}
        resolve = lambda name: (None if not raw.get(name)
                                else os.path.join(beside, raw[name]))

        target = Target(
            structure=resolve("target"),
            sequence=raw.get("target_sequence"),
            chain=raw.get("target_chain", TARGET_CHAIN),
            hotspot=raw.get("hotspot") or None,
            rm_target=raw.get("rm_target", False),
            name=raw.get("name", "design"),
        )
        binder = Binder(
            length=raw.get("binder_len"),
            scaffold=resolve("binder_scaffold"),
            chain=raw.get("binder_scaffold_chain", BINDER_CHAIN),
            designable=raw.get("designable"),
            rm_binder=raw.get("rm_binder"),
        )
        filters = raw.get("filters", {})
        if not isinstance(filters, dict):
            raise ValueError("filters must be a JSON object; omit it to leave filtering disabled")
        if raw.get("af3") is not None and not isinstance(raw["af3"], dict):
            raise ValueError("af3 runtime settings must be a JSON object")
        counts = {name: raw.get(name, 1)
                  for name in ("trajectories",)}
        for name, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{name} must be a positive integer")
        return cls(target=target, binder=binder, overrides=dict(overrides), filters=filters, af3=raw.get("af3"),
                   **counts)


_KNOWN_FIELDS = {
    "name", "target", "target_sequence", "target_chain", "hotspot", "rm_target",
    "binder_len", "binder_scaffold", "binder_scaffold_chain", "designable", "rm_binder",
    "config", "filters", "trajectories", "af3",
}


@dataclasses.dataclass(frozen=True)
class FilterRules:
    min_interface_ptm: float | None = 0.8
    min_binder_plddt: float | None = None
    max_ipae_min: float | None = 1.5
    max_irmsd: float | None = 2.0

    @classmethod
    def from_settings(cls, values: dict) -> FilterRules:
        if not isinstance(values, dict):
            raise ValueError('filter thresholds must be a JSON object')
        unknown = set(values) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown filter(s): {', '.join(sorted(unknown))}")
        for name, value in values.items():
            if value is not None and (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (name.startswith("min_") and value > 1)):
                unit = "[0, 1]" if name.startswith("min_") else "[0, infinity) angstroms"
                raise ValueError(f'filters.{name} must be a finite number in {unit}, or null')
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class MonomerRules:
    min_binder_plddt: float | None = None
    max_rmsd: float | None = None

    @classmethod
    def from_settings(cls, values: dict) -> MonomerRules:
        unknown = set(values) - {'min_binder_plddt', 'max_rmsd'}
        if unknown:
            raise ValueError(f'unknown AF2 monomer filter(s): {sorted(unknown)}')
        for name, value in values.items():
            if value is not None and (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (name == 'min_binder_plddt' and value > 1)):
                raise ValueError(f'invalid filters.af2_monomer.{name}: {value}')
        return cls(**values)


@dataclasses.dataclass(frozen=True)
class FilterPlan:
    af2: FilterRules | None = None
    af3: FilterRules | None = None
    af2_monomer: MonomerRules | None = None

    @property
    def backends(self):
        return [name for name in ('af2', 'af3', 'af2_monomer') if getattr(self, name) is not None]

    @classmethod
    def from_settings(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError('filters must be a JSON object')
        # Older flat settings and omitted filters keep AF2 validation enabled.
        if not set(raw) & {'af2', 'af3', 'af2_monomer'}:
            return cls(af2=FilterRules.from_settings(raw))
        unknown = set(raw) - {'af2', 'af3', 'af2_monomer'}
        if unknown:
            raise ValueError(f'unknown filter backend(s), or mixed flat/nested filters: {sorted(unknown)}')
        selected = {}
        for name, block in raw.items():
            if not isinstance(block, dict):
                raise ValueError(f'filters.{name} must be a JSON object')
            enabled = block.get('enabled', True)
            if not isinstance(enabled, bool):
                raise ValueError(f'filters.{name}.enabled must be true or false')
            rule_type = MonomerRules if name == 'af2_monomer' else FilterRules
            rules = rule_type.from_settings({k: v for k, v in block.items() if k != 'enabled'})
            selected[name] = rules if enabled else None
        return cls(**selected)
