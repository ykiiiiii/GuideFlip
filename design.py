#!/usr/bin/env python3
"""Design binders: settings -> target -> independent trajectories -> selected validation.

Start reading at run(). The numerical algorithm lives in guideflip/flow.py;
command-line handling and result files live in guideflip/cli.py and results.py.
"""
from __future__ import annotations

import dataclasses
import gc
import sys
from pathlib import Path

import numpy as np

from guideflip.cli import build_parser, load_settings, progress_reporter, weights_available
from guideflip.flow import DesignFailed, run_design
from guideflip.models.alphafold import AlphaFold, fold_target
from guideflip.models.prior import SequencePrior
from guideflip.settings import FilterPlan
from guideflip.results import (prepare_output, save_failure, save_generation,
                             save_validation, summarize)


def run(args) -> int:
    from guideflip.validation import run_pipeline
    return run_pipeline(args, design_stage)


def design_stage(args) -> int:
    """Run one complete design per seed, reusing models when the length is unchanged."""
    # 1. Validate inputs and reserve output names before loading model weights.
    design, config = load_settings(args)
    filters = FilterPlan.from_settings(design.filters)
    if not weights_available(args):
        return 2
    seeds = range(args.seed, args.seed + args.trajectories)
    prepare_output(design, args.output, seeds)

    # 2. A sequence-only target needs an initial structure, with its template withheld.
    if design.target.is_predicted:
        print(f"predicting a structure for {design.target.name} from its sequence",
              file=sys.stderr)
        design = dataclasses.replace(design, target=fold_target(
            design.target, directory=str(Path(args.output) / "design"), params_dir=args.alphafold_params))
        print(f"  {design.target.structure}", file=sys.stderr)

    # 3. Load AF2 and the ADFlip sequence prior once for fixed-length designs.
    first_design = design.for_seed(args.seed)
    alphafold = AlphaFold(first_design, config, params_dir=args.alphafold_params)
    prepared_length = config.binder_length
    prior = SequencePrior(args.prior_checkpoint)
    scored = 0

    for seed in seeds:
        trajectory_design = design.for_seed(seed)
        trajectory = dataclasses.replace(
            config, seed=seed, binder_length=trajectory_design.binder_length())
        name = f"{design.target.name}_seed{seed}"

        # Length ranges require rebuilding shape-dependent AF2 inputs and caches.
        if isinstance(design.binder.length, tuple) and args.report_every > 0:
            print(f"{name}\tbinder_len {trajectory.binder_length}",
                  file=sys.stderr, flush=True)
        if trajectory.binder_length != prepared_length:
            import jax

            del alphafold
            jax.clear_caches()
            gc.collect()
            alphafold = AlphaFold(trajectory_design, trajectory,
                                  params_dir=args.alphafold_params)
            prepared_length = trajectory.binder_length

        # 4. Generate one continuous guided-flow trajectory per seed.
        alphafold.set_seed(seed)
        try:
            result = run_design(
                trajectory, alphafold, prior, rng=np.random.default_rng(seed),
                design=trajectory_design,
                on_update=progress_reporter(name, args.report_every))
        except DesignFailed as failure:
            save_failure(args.output, name, seed, failure)
            continue

        # 5. Preserve generation output, then validate the fixed final sequence.
        save_generation(args.output, name, result, trajectory, design, args.settings)
        if filters.af2 is None:
            scored += 1
            continue
        try:
            final = alphafold.score(result.sequence)
            save_validation(args.output, name, result, final, filters.af2)
        except Exception as failure:
            save_failure(args.output, name, seed, failure, stage="validation")
            continue
        scored += 1

    # 6. Summarize all attempts and collect candidates passing all selected filters.
    summarize(args.output)
    return 0 if scored else 1


def main(argv=None) -> int:
    return run(build_parser(direct=True).parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
