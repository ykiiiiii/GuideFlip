#!/usr/bin/env bash
set -euo pipefail
guideflip_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep the isolated installer here so no separate tools directory is needed.
exec python3 - "$guideflip_root" "$@" <<'PYTHON_INSTALLER'
"""Install GuideFlip in a dedicated Conda environment; no sudo required."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(sys.argv.pop(1)).resolve()
AF_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
AF_FILES = tuple(f"params_model_{number}_{family}.npz"
                 for family in ("multimer_v3", "ptm") for number in range(1, 6))
MARKER = ".guideflip-installer.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="install.sh", description=__doc__)
    result.add_argument("--conda", default=os.environ.get("CONDA_EXE") or "conda",
                        help="Conda executable (default: $CONDA_EXE or conda)")
    environment = result.add_mutually_exclusive_group()
    environment.add_argument("--env-name", default="guideflip",
                             help="dedicated Conda environment name (default: guideflip)")
    environment.add_argument("--prefix", type=Path,
                             help="create a Conda environment at this path instead of by name")
    result.add_argument("--prior-checkpoint", type=Path,
                        default=ROOT / "adflip/adflip_inference.pt",
                        help="existing backflip-adflip-1 inference checkpoint; not downloaded")
    result.add_argument("--alphafold-params", type=Path,
                        default=os.environ.get("GUIDEFLIP_AF_PARAMS") or None,
                        help="reuse an existing complete AF parameter directory; otherwise download")
    result.add_argument("--skip-selfcheck", action="store_true",
                        help="install on a login node; run guideflip selfcheck on the GPU node later")
    result.add_argument("--dry-run", action="store_true",
                        help="show the plan without downloads, installation or file changes")
    return result


def run(command, *, env=None):
    command = [str(arg) for arg in command]
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def missing_parameters(directory: Path) -> list[str]:
    """Reject damaged existing files instead of replacing user-supplied weights."""
    missing = []
    for name in AF_FILES:
        path = directory / name
        if not path.exists():
            missing.append(name)
        elif not path.is_file() or not zipfile.is_zipfile(path):
            raise ValueError(f"invalid AF parameter file: {path}; it was not overwritten")
    return missing


def extract_parameters(archive: Path, directory: Path) -> None:
    """Extract only the ten expected regular files, without trusting tar paths."""
    needed = set(missing_parameters(directory))
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=directory) as staging:
        with tarfile.open(archive) as source:
            found = set()
            for member in source:
                name = member.name.removeprefix("./")
                if name not in needed:
                    continue
                if name in found or not member.isfile():
                    raise ValueError(f"unexpected archive member: {member.name}")
                destination = Path(staging) / name
                with source.extractfile(member) as handle, destination.open("wb") as out:
                    shutil.copyfileobj(handle, out)
                if not zipfile.is_zipfile(destination):
                    raise ValueError(f"download contains an invalid parameter file: {name}")
                found.add(name)
        if found != needed:
            raise ValueError(f"AF archive is missing: {', '.join(sorted(needed - found))}")
        for name in sorted(needed):
            # Publish complete files atomically; refuse an existing destination.
            os.link(Path(staging) / name, directory / name)


def download_parameters(directory: Path) -> None:
    if not missing_parameters(directory):
        print(f"Reusing AF parameters: {directory}")
        return
    archive = ROOT / ".cache/alphafold_params_2022-12-06.tar.part"
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Keep the download to allow retries after either a network or extraction error.
    run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
         "--output", archive, AF_URL])
    extract_parameters(archive, directory)


def environment_path(conda, args, env) -> Path:
    info = json.loads(subprocess.check_output([conda, "info", "--json"], text=True, env=env))
    base = Path(info["root_prefix"]).resolve()
    if args.prefix:
        prefix = args.prefix.expanduser().resolve()
    else:
        directories = [Path(path).expanduser().resolve() for path in info["envs_dirs"]]
        existing = [directory / args.env_name for directory in directories
                    if (directory / args.env_name).exists()]
        if existing:
            prefix = existing[0].resolve()
        else:
            prefix = None
            for directory in directories:
                parent = next(path for path in (directory, *directory.parents) if path.exists())
                if os.access(parent, os.W_OK):
                    prefix = directory / args.env_name
                    break
            if prefix is None:
                raise ValueError("no writable Conda environment directory; choose --prefix")
    if prefix == base:
        raise ValueError("the Conda base environment must not be used for GuideFlip")
    return prefix


def check_environment_path(prefix: Path) -> None:
    """Only reuse environments created here, never take over a user's environment."""
    if not prefix.exists():
        return
    if not prefix.is_dir():
        raise ValueError(f"environment path is not a directory: {prefix}")
    marker = prefix / MARKER
    expected = {"repository": str(ROOT), "format": 2, "manager": "conda"}
    if not marker.exists():
        if any(prefix.iterdir()):
            raise ValueError(f"{prefix} was not created by this installer; "
                             "choose a new --env-name or --prefix")
    elif json.loads(marker.read_text()) != expected:
        raise ValueError(f"{prefix} belongs to a different installation; "
                         "choose a new --env-name or --prefix")


def check_activation_paths(*paths) -> None:
    # Conda stores these values verbatim and later emits shell activation commands.
    if any(any(char in str(path) for char in "'\"$`\n\r") for path in paths):
        raise ValueError("Conda activation paths must not contain quotes, shell expansions or newlines")


def install(args) -> None:
    if not args.prefix and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.env_name)
                            or args.env_name in {"base", "root"}):
        raise ValueError("choose a dedicated --env-name, not base/root or a path")
    checkpoint = args.prior_checkpoint.expanduser().resolve()
    params = (args.alphafold_params or ROOT / ".cache/alphafold").expanduser().resolve()
    check_activation_paths(ROOT, checkpoint, params, args.prefix or "")
    clean_env = dict(os.environ, PYTHONNOUSERSITE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    for variable in ("PYTHONPATH", "PYTHONHOME"):
        clean_env.pop(variable, None)

    conda = shutil.which(args.conda)
    if not args.dry_run:
        if sys.platform != "linux":
            raise ValueError("this installer supports Linux NVIDIA GPU environments only")
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError("use the bundled adflip/adflip_inference.pt, "
                             "or provide --prior-checkpoint /path/to/inference.pt")
        if conda is None:
            raise ValueError("Conda was not found; install Miniforge/Miniconda first, "
                             "or select --conda /path/to/conda (see README.md)")
        prefix = environment_path(conda, args, clean_env)
        check_activation_paths(prefix)
        check_environment_path(prefix)
        missing = missing_parameters(params)
        if args.alphafold_params and missing:
            raise ValueError("the supplied AF directory is incomplete: " + ", ".join(missing))
        if missing and shutil.which("curl") is None:
            raise ValueError("curl is required to download AF parameters; or use --alphafold-params")
        selector = ["--prefix", str(prefix)]
    else:
        conda = conda or args.conda
        selector = (["--prefix", str(args.prefix.expanduser().resolve())] if args.prefix
                    else ["--name", args.env_name])

    create = [conda, "create", "--yes", *selector, "--override-channels", "--channel",
              "conda-forge", "--no-default-packages", "python=3.11", "pip"]
    runner = [conda, "run", "--no-capture-output", *selector, "python"]
    constraints = ROOT / "installation/requirements.txt"
    commands = [
        [*runner, "-m", "pip", "install", "--upgrade", "pip"],
        [*runner, "-m", "pip", "install", "torch==2.8.0",
         "--index-url", "https://download.pytorch.org/whl/cu128"],
        [*runner, "-m", "pip", "install", "-c", constraints, "jax[cuda12]==0.6.0"],
        [*runner, "-m", "pip", "install", "-c", constraints, "-e", f"{ROOT}[models]"],
        [*runner, "-m", "pip", "check"],
    ]
    variables = dict(GUIDEFLIP_AF_PARAMS=str(params), GUIDEFLIP_PRIOR_CHECKPOINT=str(checkpoint),
                     XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONNOUSERSITE="1")
    save_variables = [conda, "env", "config", "vars", "set", *selector,
                      *[f"{key}={value}" for key, value in variables.items()]]
    activate = str(args.prefix.expanduser().resolve()) if args.prefix else args.env_name
    print(f"Environment: {selector[-1]}\nADFlip checkpoint: {checkpoint}\nAF parameters: {params}")
    if args.dry_run:
        print("DRY RUN: no changes or downloads")
        for command in [create, *commands, save_variables]:
            print("+ " + shlex.join(map(str, command)))
        print("Validate the supplied inference checkpoint with weights_only=True")
        print(f"Reuse AF parameters from {params}" if args.alphafold_params
              else f"Download missing AF parameters from {AF_URL}")
        print("Skip GPU selfcheck" if args.skip_selfcheck else "Run guideflip selfcheck")
        print(f"After installation: conda activate {shlex.quote(activate)}")
        return

    if os.environ.get("LD_LIBRARY_PATH"):
        print("Warning: LD_LIBRARY_PATH may override pip CUDA libraries; see README.md",
              file=sys.stderr)
    # Claim a new/empty directory before creating the environment, to allow safe retries.
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / MARKER).write_text(json.dumps(
        {"repository": str(ROOT), "format": 2, "manager": "conda"}) + "\n")
    if not (prefix / "conda-meta/history").is_file():
        if (prefix / "conda-meta").exists():
            raise ValueError("incomplete Conda metadata; choose a new --env-name or --prefix")
        run(create, env=clean_env)
    # Never run 'conda create --yes' on an existing environment: it can remove it.
    run([*runner, "-c",
         "import sys; from pathlib import Path; "
         "prefix = Path(sys.argv[1]).resolve(); "
         "valid = sys.version_info[:2] == (3, 11) "
         "and Path(sys.prefix).resolve() == prefix "
         "and (prefix / 'conda-meta/history').is_file(); "
         "sys.exit(0 if valid else 'Expected a dedicated Python 3.11 Conda environment')",
         prefix], env=clean_env)
    for command in commands:
        run(command, env=clean_env)
    run([*runner, "-c",
         "import sys, torch; saved = torch.load(sys.argv[1], map_location='cpu', weights_only=True); "
         "valid = isinstance(saved, dict) and saved.get('format') == 'backflip-adflip-1'; "
         "sys.exit(0 if valid else 'Expected an inference checkpoint; see README.md')",
         checkpoint], env=clean_env)
    if not args.alphafold_params:
        download_parameters(params)
    run(save_variables, env=clean_env)
    clean_env.update(variables)
    if not args.skip_selfcheck:
        run([*runner, "-m", "guideflip.cli", "selfcheck"], env=clean_env)
    else:
        print("GPU readiness was NOT checked. Run guideflip selfcheck on the GPU node before design.")
    print(f"\nInstalled. In each new shell, run:\nconda activate {shlex.quote(activate)}")
    print("If this environment is already active, deactivate and activate it again to load its settings.")


def main(argv=None) -> int:
    try:
        install(parser().parse_args(argv))
    except (ValueError, OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
        print(f"Installation stopped: {error}\nSee README.md; no existing weights were replaced.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYTHON_INSTALLER
