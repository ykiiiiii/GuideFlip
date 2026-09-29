"""Command-line settings, validation and environment checks."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

from .settings import Design, DesignConfig, LossWeights

WEIGHTS = {
    "alphafold_params": ("GUIDEFLIP_AF_PARAMS",
                         "directory holding the AlphaFold parameters"),
    "prior_checkpoint": ("GUIDEFLIP_PRIOR_CHECKPOINT",
                         "ADFlip inference checkpoint (bundled adflip/adflip_inference.pt)"),
}


def add_weight_arguments(parser: argparse.ArgumentParser) -> None:
    for name, (variable, help_text) in WEIGHTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}",
                            default=os.environ.get(variable),
                            help=f"{help_text} (or ${variable})")


def add_design_arguments(design: argparse.ArgumentParser, *, direct: bool) -> None:
    if direct:
        design.add_argument("--settings", required=True, help="a design description (JSON)")
    else:
        design.add_argument("settings", help="a design description (JSON)")
    design.add_argument("--output", required=True, help="directory for designs")
    design.add_argument("--trajectories", type=int, default=None,
                        help="independent guided-flow trajectories (settings default: 1)")
    design.add_argument("--seed", type=int, default=0)
    design.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        dest="overrides",
                        help="override a design parameter; repeatable")
    design.add_argument("--report-every", type=int, default=25, metavar="UPDATES",
                        help="print progress this often; 0 for silence")
    design.add_argument("--_design-worker", action="store_true", help=argparse.SUPPRESS)
    design.add_argument("--_monomer-worker", action="store_true", help=argparse.SUPPRESS)
    add_weight_arguments(design)


def build_parser(*, direct: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="design.py" if direct else "guideflip",
        description="Design a binder by guided discrete flow matching.")
    if direct:
        add_design_arguments(parser, direct=True)
        return parser
    commands = parser.add_subparsers(dest="command", required=True)
    design = commands.add_parser("design", help="run a design")
    add_design_arguments(design, direct=False)
    retry = commands.add_parser("validate-af3", help="retry AF3 validation of saved designs")
    retry.add_argument("settings", help="the same design settings JSON")
    retry.add_argument("--output", required=True, help="existing design output directory")
    check = commands.add_parser(
        "selfcheck", help="report whether everything needed is installed and findable")
    add_weight_arguments(check)
    return parser


#: The parameters a design does not take from `--set`, and where each comes from.
NOT_OVERRIDABLE = {"seed": "--seed", "binder_length": "the settings file"}


def coerce(name: str, text: str, declared: str) -> object:
    """One written value, read as the type its field declares.

    Every field is annotated, so the annotation is the only thing that has to be
    consulted -- and consulting it is what stops a value being accepted in a form
    that means something else. The alternative, trying JSON and keeping the raw
    string when it fails, accepts every misspelling silently.
    """
    written = text.strip()
    if declared == "str":
        # For example, the amino-acid exclusion string. Quotes are optional.
        return json.loads(written) if written[:1] in "\"'" else written
    if declared in ("int", "float"):
        try:
            return int(written) if declared == "int" else float(written)
        except ValueError:
            raise ValueError(f"{name} is {declared}, not {text!r}")
    try:
        value = json.loads(written)
    except json.JSONDecodeError as why:
        raise ValueError(f"{name} is {declared}, so it is written as JSON: {why}")
    return tuple(value) if declared.startswith("tuple") else value


def parse_override(text: str) -> tuple[str, object]:
    """One NAME=VALUE, typed the way the field it names is typed."""
    name, separator, value = text.partition("=")
    name = name.strip()
    if not separator:
        raise ValueError(f"--set takes NAME=VALUE, not {text!r}")
    fields = {field.name: field for field in dataclasses.fields(DesignConfig)}
    if name in NOT_OVERRIDABLE:
        raise ValueError(f"{name} comes from {NOT_OVERRIDABLE[name]}, not --set")
    if name not in fields:
        raise ValueError(f"{name!r} is not a design parameter; see --help")
    return name, coerce(name, value, fields[name].type)


def configure(design: Design, seed: int, overrides: list[str]) -> DesignConfig:
    """The settings file's overrides, then the command line's, over the defaults."""
    fields = {field.name: field for field in dataclasses.fields(DesignConfig)}
    settings = dict(design.overrides)
    for name in settings:
        if name in NOT_OVERRIDABLE:
            raise ValueError(f"{name} comes from {NOT_OVERRIDABLE[name]}; "
                             f"a settings file's config block cannot set it")
        if name not in fields:
            raise ValueError(f"{name!r} is not a design parameter")
    for text in overrides:
        name, value = parse_override(text)
        settings[name] = value
    # JSON has no tuples, so a pair written in a settings file arrives as a list.
    for name, value in settings.items():
        if fields[name].type.startswith("tuple") and not isinstance(value, tuple):
            settings[name] = tuple(value)
    # Nor does JSON have a LossWeights. Named weights are merged over the defaults
    # rather than replacing them, so a file that moves one weight keeps the other
    # eleven -- and an unknown name is refused here, where the message can say so,
    # rather than as an attribute error inside the objective an hour later.
    if isinstance(settings.get("loss"), dict):
        weights = {field.name for field in dataclasses.fields(LossWeights)}
        unknown = sorted(set(settings["loss"]) - weights)
        if unknown:
            raise ValueError(f"loss has no weight called {', '.join(unknown)}; "
                             f"the weights are {', '.join(sorted(weights))}")
        settings["loss"] = dataclasses.replace(LossWeights(), **settings["loss"])
    return DesignConfig(binder_length=design.binder_length(), seed=seed, **settings)


def selfcheck(args) -> int:
    """Name every module and every checkpoint, because an install can be short of one.

    pip, conda and uv all report success for an environment that is missing a module,
    and a campaign is where that otherwise surfaces.
    """
    problems = []

    def report(what: str, ok: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'MISSING'}  {what}{'  ' + detail if detail else ''}")
        if not ok:
            problems.append(what)

    print("modules")
    for module, why in (("numpy", ""), ("torch", "the sequence prior"),
                        ("jax", "AlphaFold"), ("colabdesign", "AlphaFold"),
                        ("adflip.design", "the sequence prior")):
        try:
            __import__(module)
            report(module, True, why)
        except Exception as problem:
            report(module, False, f"{type(problem).__name__}: {problem}")

    print("accelerator")
    try:
        import jax
        devices = jax.devices()
        report("jax devices", any(d.platform != "cpu" for d in devices), str(devices))
    except Exception as problem:
        report("jax devices", False, str(problem))

    try:
        import torch
        report("torch CUDA", torch.cuda.is_available(), torch.version.cuda or "CPU build")
    except Exception as problem:
        report("torch CUDA", False, str(problem))

    print("weights")
    params = args.alphafold_params
    if not params or not os.path.isdir(params):
        report("AlphaFold parameters", False, f"{params or 'not given'}")
    else:
        # Two sets, and the monomer one is easy to leave out: the design uses
        # multimer, and folding a target given only its sequence uses monomer.
        #
        # All five multimer models, not the two a trajectory samples from. The
        # finished sequence is scored with every one of them, and that happens at
        # the very end -- so checking only 1 and 2 reported a sound install and
        # then failed after the whole design had been paid for.
        for pattern, numbers, why in (
                ("params_model_{}_multimer_v3.npz", (1, 2, 3, 4, 5),
                 "design, and scoring the finished sequence with all five"),
                ("params_model_{}_ptm.npz", (1, 2, 3, 4, 5),
                 "folding a target given only its sequence")):
            missing = [pattern.format(n) for n in numbers
                       if not os.path.exists(os.path.join(params, pattern.format(n)))]
            report(pattern.format("*"), not missing,
                   why if not missing else f"missing {missing}")

    checkpoint = args.prior_checkpoint
    if not checkpoint or not os.path.exists(checkpoint):
        report("prior checkpoint", False, f"{checkpoint or 'not given'}")
    else:
        try:
            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            report("prior checkpoint", saved.get("format") == "backflip-adflip-1",
                   saved.get("format") or "unknown format; use adflip/adflip_inference.pt")
        except Exception as problem:
            report("prior checkpoint", False,
                   f"{type(problem).__name__}: {problem}; use the bundled "
                   f"adflip/adflip_inference.pt, not a training checkpoint")

    if problems:
        print(f"\n{len(problems)} problem(s): {', '.join(problems)}", file=sys.stderr)
        return 1
    print("\neverything needed is here")
    return 0


def progress_reporter(name: str, every: int):
    """Report one continuous guided-flow trajectory and its ADFlip gate."""
    if every <= 0:
        return None

    def report(update, row):
        if row["step"] % every and row["lambda_t"] == 0:
            return
        drift = row.get("interface_rmsd")
        print(f"  {name} {row['step']:>4}  guided flow  lambda {row['lambda_t']}  "
              f"loss {row['loss']:7.3f}  plddt {row['binder_plddt']:.3f}  "
              f"i_ptm {row['interface_ptm']:.3f}  "
              f"drift {'-' if drift is None else f'{drift:.1f}A'}",
              file=sys.stderr, flush=True)
        if row["lambda_t"] == 1:
            print(f"    sequence  {update.state.sequence(undecided='-')}",
                  file=sys.stderr, flush=True)

    return report


def load_settings(args):
    """Resolve the independent trajectory count and validate before loading models."""
    design = Design.from_json(args.settings)
    if args.trajectories is None:
        args.trajectories = design.trajectories
    if isinstance(args.trajectories, bool) or args.trajectories < 1:
        raise ValueError("trajectories must be a positive integer")
    if args.seed < 0 or args.seed + args.trajectories - 1 > 2**32 - 1:
        raise ValueError("trajectory seeds must be in [0, 2**32-1]")
    config = configure(design.for_seed(args.seed), args.seed, args.overrides)
    return design, config


def weights_available(args):
    """Keep missing-weight diagnostics available without importing GPU frameworks."""
    for name, (variable, _) in WEIGHTS.items():
        if getattr(args, name) is None:
            print(f"error: --{name.replace('_', '-')} is required (or ${variable})",
                  file=sys.stderr)
            return False
    return True


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "selfcheck":
        return selfcheck(args)
    if args.command == "validate-af3":
        from .validation import retry_af3
        return retry_af3(args)
    from design import run
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
