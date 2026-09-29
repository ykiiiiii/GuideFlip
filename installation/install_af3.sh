#!/usr/bin/env bash
set -euo pipefail
guideflip_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 - "$guideflip_root" "$@" <<'PYTHON_INSTALLER'
"""Install native AF3 for GuideFlip without Docker or administrator privileges."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(sys.argv.pop(1)).resolve()
REVISION = "608edb684db9f6fd0e677fea01c4cefc60f8a8aa"
REPOSITORY = "https://github.com/google-deepmind/alphafold3.git"
# Includes the cold wheel extraction improvements needed for large CUDA wheels.
UV_VERSION = "0.12.19"
MARKER = ".guideflip-af3-installer.json"


def run(command, *, env, cwd=None):
    command = list(map(str, command))
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, env=env, cwd=cwd, check=True)


def check_prefix(prefix):
    expected = {"repository": str(ROOT), "revision": REVISION, "format": 1}
    if prefix.exists():
        marker = prefix / MARKER
        if not prefix.is_dir() or (not marker.is_file() and any(prefix.iterdir())):
            raise ValueError(f"{prefix} is not an installation created here; choose a new --prefix")
        if marker.is_file() and json.loads(marker.read_text()) != expected:
            raise ValueError(f"{prefix} belongs to another installation; choose a new --prefix")
    return expected


def install(args):
    prefix = args.prefix.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    toolchain, source, venv = prefix / "toolchain", prefix / "source", prefix / "venv"
    conda = shutil.which(args.conda)
    env = dict(os.environ, PYTHONNOUSERSITE="1", UV_LINK_MODE="copy",
               UV_PROJECT_ENVIRONMENT=str(venv), UV_CACHE_DIR=str(args.cache_dir),
               CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs), MAKEFLAGS=f"-j{args.jobs}")
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    create = [conda or args.conda, "create", "--yes", "--prefix", toolchain,
              "--override-channels", "--channel", "conda-forge", "--no-default-packages",
              "python=3.12", "pip", "gcc_linux-64=14", "gxx_linux-64=14", "make", "zlib", "git"]
    runner = [conda or args.conda, "run", "--no-capture-output", "--prefix", toolchain]
    uv = toolchain / "bin/uv"
    commands = [
        [*runner, "python", "-m", "pip", "install", f"uv=={UV_VERSION}"],
        [*runner, "git", "init", source],
        [*runner, "git", "-C", source, "fetch", "--depth", "1", REPOSITORY, REVISION],
        [*runner, "git", "-C", source, "checkout", "--detach", "FETCH_HEAD"],
        [*runner, uv, "sync", "--project", source, "--python", toolchain / "bin/python",
         "--frozen", "--no-dev", "--no-editable"],
        [*runner, venv / "bin/build_data"],
    ]
    print(f"AF3 revision: {REVISION}\nInstallation: {prefix}\nWeights: {model_dir}", flush=True)
    if args.dry_run:
        print("DRY RUN: no changes or downloads")
        for command in [create, *commands]:
            print("+ " + shlex.join(map(str, command)))
        print(f"Write runtime wrapper: {prefix / 'python'}")
        print(f"Save paths automatically: {ROOT / 'installation/.guideflip-af3.json'}")
        print("Skip GPU check" if args.skip_gpu_check else "Check JAX GPU visibility")
        return
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("this installer supports Linux x86_64 with an NVIDIA GPU")
    if not conda:
        raise ValueError("install Miniforge/Conda first, or supply --conda /path/to/conda")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if not any(p.is_file() and p.stat().st_size > 0 for p in model_dir.glob("af3*.bin*")):
        raise ValueError("no AF3 weights in --model-dir; obtain them from the official AF3 installation page")
    expected = check_prefix(prefix)
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / MARKER).write_text(json.dumps(expected, indent=2) + "\n")
    if not (toolchain / "conda-meta/history").is_file():
        if toolchain.exists() and any(toolchain.iterdir()):
            raise ValueError("incomplete toolchain; choose a new --prefix instead of overwriting it")
        run(create, env=env)
    if (source / ".git").exists():
        dirty = subprocess.check_output([str(toolchain / "bin/git"), "-C", str(source),
                                         "status", "--porcelain", "--untracked-files=no"], text=True)
        if dirty.strip():
            raise ValueError("AF3 source has local edits; choose a new --prefix")
    for command in commands:
        run(command, env=env)
    # A private launcher avoids shell activation and protects AF3 from AF2's Python paths.
    wrapper = "#!/usr/bin/env bash\nset -euo pipefail\n"
    wrapper += "unset PYTHONPATH PYTHONHOME\nexport PYTHONNOUSERSITE=1\n"
    wrapper += f"export PATH={shlex.quote(str(toolchain / 'bin'))}:\"$PATH\"\n"
    wrapper += f"export LD_LIBRARY_PATH={shlex.quote(str(toolchain / 'lib'))}\n"
    wrapper += 'export XLA_PYTHON_CLIENT_PREALLOCATE=false\n'
    wrapper += 'export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false}"\n'
    wrapper += f"exec {shlex.quote(str(venv / 'bin/python'))} \"$@\"\n"
    launcher = prefix / "python"
    launcher.write_text(wrapper)
    launcher.chmod(0o755)
    run([launcher, "-c", "import alphafold3.cpp; from alphafold3.constants import chemical_components; "
         "assert len(chemical_components.Ccd()) > 0; print('AF3 native extension and CCD: OK')"], env=env)
    if not args.skip_gpu_check:
        run([launcher, "-c", "import jax; devices = jax.devices(); print(devices); "
             "assert any(d.platform == 'gpu' for d in devices), 'No JAX GPU device'"], env=env)
    # Keep the exact resolved toolchain and Python packages for troubleshooting.
    for filename, command in (
        ("conda-explicit.txt", [conda, "list", "--prefix", toolchain, "--explicit"]),
        ("python-packages.txt", [uv, "pip", "freeze", "--python", venv / "bin/python"]),
    ):
        output = subprocess.check_output(list(map(str, command)), text=True, env=env)
        (prefix / filename).write_text(output)
    (prefix / "installation.json").write_text(json.dumps({
        "source": REPOSITORY, "revision": REVISION, "uv": UV_VERSION,
        "uv_lock_sha256": hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest(),
        "gpu_visibility_checked": not args.skip_gpu_check,
        "inference_checked": False, "model_dir": str(model_dir),
        "mode": "msa_free_template_free",
    }, indent=2) + "\n")
    configuration = {"python": str(launcher), "script": str(source / "run_alphafold.py"),
                     "model_dir": str(model_dir)}
    destination = ROOT / "installation/.guideflip-af3.json"
    if destination.exists() and json.loads(destination.read_text()) != configuration:
        if not args.replace_config:
            raise ValueError(f"installed, but {destination} already points elsewhere; "
                             "rerun with --replace-config to select this installation")
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(configuration, indent=2) + "\n")
    temporary.replace(destination)
    print(f"\nInstalled AF3. GuideFlip reads {destination} automatically.")
    print("GPU inference has not been tested by this installer; run the amylin AF3 example.")
    if args.skip_gpu_check:
        print("GPU visibility was not checked (--skip-gpu-check).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=ROOT / ".af3",
                        help="dedicated installation directory (default: this checkout/.af3)")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="directory containing official AF3 model weights (not bundled)")
    parser.add_argument("--conda", default=os.environ.get("CONDA_EXE") or "conda")
    parser.add_argument("--jobs", type=int, default=4, help="parallel C++ build jobs (default: 4)")
    parser.add_argument("--cache-dir", type=Path,
                        help="persistent uv build/download cache (default: temporary local storage)")
    parser.add_argument("--skip-gpu-check", action="store_true", help="install on a CPU/login node")
    parser.add_argument("--replace-config", action="store_true", help="select this installation over a previously configured AF3")
    parser.add_argument("--dry-run", action="store_true")
    try:
        args = parser.parse_args()
        if args.cache_dir:
            args.cache_dir = args.cache_dir.expanduser().resolve()
            install(args)
        elif args.dry_run:
            args.cache_dir = Path(tempfile.gettempdir()) / "guideflip-af3-<temporary>/uv"
            install(args)
        else:
            # Streaming wheel extraction can be very slow on cluster network storage.
            # Keep intermediates local; only the completed environment lives at --prefix.
            with tempfile.TemporaryDirectory(prefix="guideflip-af3-") as scratch:
                args.cache_dir = Path(scratch) / "uv"
                install(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Installation stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYTHON_INSTALLER
