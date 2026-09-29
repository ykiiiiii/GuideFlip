"""GuideFlip: binder design by AlphaFold-guided discrete flow matching.

The algorithm lives here and is plain numpy. The two models it argues with -- AlphaFold
through ColabDesign, and ADFlip as the sequence prior -- are behind `guideflip.models`,
which is the only place jax or torch is imported. That is why the tests in this package
need neither a GPU nor a set of weights, and it is also why importing this module does
not load either framework: `AlphaFold` and `SequencePrior` are imported from
`guideflip.models` when they are wanted.
"""
from .settings import Binder, Design, DesignConfig, LossWeights, Target
from .flow import DesignResult, DesignFailed, run_design
from .state import DesignState
from .structure import Complex, Interface, NoInterface, assemble

__all__ = [
    "DesignConfig", "LossWeights",
    "Design", "Target", "Binder",
    "DesignState", "Complex", "assemble",
    "Interface", "NoInterface",
    "DesignResult", "DesignFailed", "run_design",
]
