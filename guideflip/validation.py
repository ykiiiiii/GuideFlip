"""Validation orchestration, binder-only AF2 folding and MSA-free AF3 folding.

The design subprocess exits before AF3 starts, releasing its GPU allocations.
AF3 runs through its own Python environment; its code and weights are not bundled.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .cli import load_settings, weights_available
from .settings import Design, FilterPlan
from .structure import RESTYPES
from .results import summarize, prepare_output


def af3_config(design, settings):
    plan = FilterPlan.from_settings(design.filters)
    if plan.af3 is None:
        return None
    raw = dict(design.af3 or {})
    if 'filters' in raw:
        raise ValueError('place AF3 thresholds in filters.af3, not af3.filters')
    config = AF3Config.from_settings(raw, beside=Path(settings).resolve().parent)
    criteria = {}
    if plan.af3.min_interface_ptm is not None:
        criteria['min_iptm'] = plan.af3.min_interface_ptm
    if plan.af3.min_binder_plddt is not None:
        criteria['min_binder_plddt'] = plan.af3.min_binder_plddt * 100
    for name in ('max_ipae_min', 'max_irmsd'):
        value = getattr(plan.af3, name)
        if value is not None:
            criteria[name] = value
    return dataclasses.replace(config, filters=criteria)


def run_pipeline(args, design_stage):
    design, _ = load_settings(args)
    plan = FilterPlan.from_settings(design.filters)
    if getattr(args, '_monomer_worker', False):
        return monomer_candidates(args)
    if (plan.af3 is None and plan.af2_monomer is None) or getattr(args, '_design_worker', False):
        return design_stage(args)
    if not weights_available(args):
        return 2
    config = af3_config(design, args.settings)
    if config is not None:
        config.check_installation()
    if plan.af2_monomer is not None:
        check_monomer_weights(args.alphafold_params)
    prepare_output(design, args.output, range(args.seed, args.seed + args.trajectories))
    # The parent imports neither GPU framework; the child owns all design allocations.
    import design as workflow
    command = [sys.executable, str(Path(workflow.__file__).resolve()), '--_design-worker',
               '--settings', str(Path(args.settings).resolve()), '--output', str(Path(args.output).resolve()),
               '--seed', str(args.seed), '--trajectories', str(args.trajectories),
               '--report-every', str(args.report_every), '--alphafold-params', args.alphafold_params,
               '--prior-checkpoint', args.prior_checkpoint]
    for override in args.overrides:
        command.extend(['--set', override])
    child = subprocess.run(command, check=False)
    folder = Path(args.output)
    if not candidate_records(folder / 'design', backend=None):
        return child.returncode or 1
    monomer_code = 0
    if plan.af2_monomer is not None:
        monomer_command = [arg if arg != '--_design-worker' else '--_monomer-worker'
                           for arg in command]
        log = folder / 'filter/af2_monomer/monomer.log'
        print(f'AF2 monomer: binder-only folding. Log: {log}', flush=True)
        with log.open('w') as handle:
            monomer_code = subprocess.run(monomer_command, stdout=handle,
                                         stderr=subprocess.STDOUT, check=False).returncode
    code, ok = 0, True
    if config is not None:
        code = fold_candidates(folder, config)
        ok = summarize_candidates(folder, config,
                                  runner_error=f'AF3 runner exited with code {code}' if code else None)
    summarize(folder)
    return 0 if child.returncode == 0 and monomer_code == 0 and code == 0 and ok else 1


def check_monomer_weights(params_dir):
    root = Path(params_dir)
    for i in range(1, 6):
        filename = f'params_model_{i}_ptm.npz'
        if not any((p / filename).is_file() for p in (root, root / 'params')):
            raise FileNotFoundError(f'AF2 monomer parameter file is missing: {filename}')


def monomer_candidates(args):
    """An isolated GPU worker; no target coordinates enter the predictor."""
    from .models.alphafold import BinderMonomer
    from .structure import binder_monomer_rmsd
    from .results import filter_monomer, save_failure

    design, config = load_settings(args)
    rules = FilterPlan.from_settings(design.filters).af2_monomer
    if rules is None:
        raise ValueError('AF2 monomer worker requires filters.af2_monomer.enabled=true')
    check_monomer_weights(args.alphafold_params)
    folder = Path(args.output)
    records = candidate_records(folder / 'design', backend='af2_monomer')
    output = folder / 'filter/af2_monomer'
    output.mkdir(parents=True, exist_ok=True)
    model, length, ok = None, None, bool(records)
    for name, record in records.items():
        try:
            sequence = record['sequence']
            if len(sequence) != length:
                if model is not None:
                    import gc
                    import jax
                    del model
                    gc.collect()
                    jax.clear_caches()
                    model, length = None, None
                model = BinderMonomer(len(sequence), params_dir=args.alphafold_params,
                                      num_recycles=config.num_recycles)
                length = len(sequence)
            scores = model.predict(sequence, seed=record['seed'], path=str(output / f'{name}.pdb'))
            scores.update(binder_monomer_rmsd(output / f'{name}.pdb', folder / f'design/{name}.pdb', sequence))
            if len(scores['rmsd_per_model']) != 5:
                raise ValueError('AF2 monomer must save all five model predictions')
            selection = filter_monomer(scores, rules, complete=record.get('design_complete') is True)
            write_json(output / f'{name}.json', dict(
                name=name, sequence=sequence, validation_status='completed',
                evaluator='AF2-Monomer', design_complete=record.get('design_complete', False),
                filter=selection, **scores))
            print(f"{name}\tAF2 monomer pLDDT {scores['binder_plddt']:.3f}"
                  f"\tbinder C-alpha RMSD {scores['rmsd']:.3f} A", flush=True)
        except Exception as failure:
            save_failure(folder, name, record['seed'], failure,
                         stage='validation', backend='af2_monomer')
            ok = False
    return 0 if ok else 1


def retry_af3(args):
    """Reuse complete AF3 samples; run only missing jobs, keeping saved AF2 decisions."""
    design = Design.from_json(args.settings)
    config = af3_config(design, args.settings)
    if config is None:
        raise ValueError('validate-af3 requires filters.af3.enabled=true')
    folder = Path(args.output)
    if not (folder / 'design').is_dir():
        raise FileNotFoundError(f'no saved designs in {folder}')
    config.check_installation()
    code = fold_candidates(folder, config)
    ok = summarize_candidates(folder, config,
                              runner_error=f'AF3 runner exited with code {code}' if code else None)
    summarize(folder)
    return 0 if code == 0 and ok else 1


SAMPLES_PER_SEED = 5  # AlphaFold3 default number of diffusion samples per seed.
INSTALL_CONFIG = Path(__file__).resolve().parents[1] / "installation/.guideflip-af3.json"



def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    """Replace one generated record atomically, so interrupted summaries are retryable."""
    def serializable(value):
        # Encode nonfinite metrics as JSON null.
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(serializable(data), handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def finite(value, label: str, low: float, high: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{label} must be finite and in [{low}, {high}]")
    return float(value)


@dataclasses.dataclass(frozen=True)
class AF3Config:
    python: str
    script: str
    model_dir: str
    model_seeds: tuple[int, ...] = (1,)
    flash_attention_implementation: str | None = "xla"
    filters: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_settings(cls, raw: dict | None, *, beside: Path) -> AF3Config | None:
        if raw is None:
            return None
        unknown = set(raw) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown af3 setting(s): {sorted(unknown)}")
        config_path = Path(os.environ.get("GUIDEFLIP_AF3_CONFIG", INSTALL_CONFIG)).expanduser()
        # Resolve the legacy root-level configuration path to installation/.
        legacy_path = INSTALL_CONFIG.parent.parent / ".guideflip-af3.json"
        if (config_path.resolve() == legacy_path.resolve() and not config_path.is_file()
                and INSTALL_CONFIG.is_file()):
            config_path = INSTALL_CONFIG
        defaults = read_json(config_path) if config_path.is_file() else {}
        if "GUIDEFLIP_AF3_CONFIG" in os.environ and not config_path.is_file():
            raise FileNotFoundError(f"AF3 installation configuration is missing: {config_path}")
        if not isinstance(defaults, dict) or set(defaults) - {"python", "script", "model_dir"}:
            raise ValueError(f"invalid AF3 installation configuration: {config_path}")
        paths = {}
        for name in ("python", "script", "model_dir"):
            value = raw.get(name) or os.environ.get("GUIDEFLIP_AF3_" + name.upper())
            relative_to = beside
            if not value:
                value = defaults.get(name)
                relative_to = config_path.resolve().parent
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"run installation/install_af3.sh, set af3.{name}, or set $GUIDEFLIP_AF3_{name.upper()}")
            path = Path(value).expanduser()
            paths[name] = str(path if path.is_absolute() else relative_to / path)
            paths[name] = os.path.abspath(paths[name])
        seeds = raw.get("model_seeds", [1])
        if (not isinstance(seeds, list) or not seeds
                or any(isinstance(seed, bool) or not isinstance(seed, int)
                       or not 0 <= seed < 2**32 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("af3.model_seeds must be distinct integers in [0, 2**32-1]")
        attention = raw.get("flash_attention_implementation", cls.flash_attention_implementation)
        if attention not in (None, "triton", "cudnn", "xla"):
            raise ValueError("af3.flash_attention_implementation must be triton, cudnn or xla")
        filters = raw.get("filters", {})
        if not isinstance(filters, dict):
            raise ValueError("af3.filters must be a JSON object")
        bounds = {"min_iptm": (0, 1), "min_binder_plddt": (0, 100),
                  "max_ipae_min": (0, float("inf")), "max_irmsd": (0, float("inf"))}
        unknown = set(filters) - set(bounds)
        if unknown:
            raise ValueError(f"unknown AF3 filter(s): {sorted(unknown)}")
        for name, value in filters.items():
            if value is not None:
                finite(value, "af3.filters." + name, *bounds[name])
        active = {key: value for key, value in filters.items() if value is not None}
        return cls(**paths, model_seeds=tuple(seeds), filters=active,
                   flash_attention_implementation=attention)

    def check_installation(self) -> None:
        """Cheap file checks only: not a claim that GPU inference has been tested."""
        if not Path(self.python).is_file() or not os.access(self.python, os.X_OK):
            raise FileNotFoundError(f"AF3 Python executable is missing: {self.python}")
        if not Path(self.script).is_file():
            raise FileNotFoundError(f"AF3 runner is missing: {self.script}")
        if not any(path.is_file() for path in Path(self.model_dir).glob("af3*.bin*")):
            raise FileNotFoundError(f"AF3 weights are missing in {self.model_dir}")

    def prediction_settings(self) -> dict:
        return {"python": self.python, "script": self.script, "model_dir": self.model_dir,
                "model_seeds": list(self.model_seeds), "samples_per_seed": SAMPLES_PER_SEED,
                "msa": "none", "templates": "none", "input_version": 1,
                "flash_attention_implementation": self.flash_attention_implementation}

    def command(self, inputs: Path, outputs: Path) -> list[str]:
        command = [self.python, self.script, "--input_dir", str(inputs.resolve()),
                "--output_dir", str(outputs.resolve()), "--model_dir", self.model_dir,
                "--run_data_pipeline=false", "--run_inference=true"]
        if self.flash_attention_implementation:
            command.append("--flash_attention_implementation=" + self.flash_attention_implementation)
        return command


def candidate_records(folder: Path, backend='af3') -> dict[str, dict]:
    records = {}
    for path in sorted(folder.glob("*.json")):
        record = read_json(path)
        if (isinstance(record, dict) and "sequence" in record
                and (backend is None or backend in record.get("validation_backends", []))):
            records[path.stem] = record
    return records


def protein(sequence: str, chain: str) -> dict:
    if not isinstance(sequence, str) or not sequence or set(sequence) - set(RESTYPES):
        raise ValueError(f"AF3 chain {chain} must be a nonempty standard amino-acid sequence")
    return {"protein": {"id": chain, "sequence": sequence,
                        "unpairedMsa": "", "pairedMsa": "", "templates": []}}


def campaign(folder: Path, config: AF3Config, *, create: bool) -> dict:
    """One immutable sequence/job mapping. Filters may change without refolding."""
    records = candidate_records(folder / "design")
    if not records:
        raise ValueError(f"no generated designs in {folder / 'design'}")
    jobs, seen = [], {}
    for name, record in records.items():
        sequence, target = record["sequence"], record.get("target_sequence")
        chains = [protein(target, "A"), protein(sequence, "B")]
        pair = (target, sequence)
        if pair not in seen:
            job = {"id": f"job_{len(jobs) + 1:05d}", "candidates": [], "sequences": chains}
            jobs.append(job)
            seen[pair] = job
        seen[pair]["candidates"].append(name)
    expected = {"prediction_settings": config.prediction_settings(), "jobs": jobs}
    path = folder / "filter" / "af3" / "jobs.json"
    if path.exists():
        if read_json(path) != expected:
            raise ValueError("AF3 inputs/runner/seeds differ from saved campaign; "
                             "use a new output directory (filters alone may change)")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, expected)
    else:
        raise FileNotFoundError(f"no AF3 campaign at {path}; run a design with filters.af3.enabled=true first")
    return expected


def sample_metrics(path: Path, job: dict, folder: Path) -> dict:
    """Validate chain/sequence mapping, then compute means from full PAE and CA pLDDT."""
    from Bio.PDB import MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionException
    from Bio.SeqUtils import seq1

    suffixes = ('summary_confidences.json', 'confidences.json', 'model.cif')
    # AF3 3.0.1 prefixes sample filenames; older versions used bare names.
    layouts = [[path / (prefix + suffix) for suffix in suffixes]
               for prefix in ('', f"{job['id']}_{path.name}_")]
    complete = [layout for layout in layouts if all(item.is_file() for item in layout)]
    if len(complete) != 1:
        raise ValueError(f'expected one complete AF3 sample file set in {path}; found {len(complete)}')
    summary_path, confidence_path, cif = complete[0]
    summary = read_json(summary_path)
    confidence = read_json(confidence_path)
    try:
        model = MMCIFParser(QUIET=True, auth_chains=False, auth_residues=False).get_structure(
            "af3", str(cif))[0]
    except PDBConstructionException as error:
        raise ValueError(f"invalid AF3 mmCIF: {error}") from error
    if set(chain.id for chain in model) != {"A", "B"}:
        raise ValueError("AF3 model must contain exactly target A and binder B")
    lengths, binder_confidence = {}, None
    for entry in job["sequences"]:
        chain, expected = entry["protein"]["id"], entry["protein"]["sequence"]
        residues = list(model[chain])
        if ("".join(seq1(res.resname) for res in residues) != expected
                or [res.id[1] for res in residues] != list(range(1, len(expected) + 1))
                or any("CA" not in res for res in residues)):
            raise ValueError(f"AF3 chain {chain} sequence/residue mapping does not match its input")
        lengths[chain] = len(expected)
        if chain == "B":
            binder_confidence = [finite(float(res["CA"].bfactor), "AF3 CA pLDDT", 0, 100)
                                 for res in residues]

    chains = np.asarray(confidence["token_chain_ids"])
    ids = confidence["token_res_ids"]
    wanted = {(chain, i) for chain, length in lengths.items() for i in range(1, length + 1)}
    if len(chains) != len(wanted) or len(ids) != len(chains) or set(zip(chains, ids)) != wanted:
        raise ValueError("AF3 PAE token chain/residue mapping does not match its input")
    pae = np.asarray(confidence["pae"], dtype=float)
    if (pae.shape != (len(chains), len(chains)) or not np.isfinite(pae).all()
            or (pae < 0).any()):
        raise ValueError("AF3 PAE must be a finite, nonnegative square token matrix")
    a, b = np.flatnonzero(chains == "A"), np.flatnonzero(chains == "B")
    ab, ba = float(pae[np.ix_(a, b)].mean()), float(pae[np.ix_(b, a)].mean())
    pair_min = np.asarray(summary.get("chain_pair_pae_min"), dtype=float)
    if pair_min.shape != (2, 2) or not np.isfinite(pair_min).all() or (pair_min < 0).any():
        raise ValueError("AF3 chain_pair_pae_min must be a finite nonnegative 2x2 matrix")
    ipae_min = float(min(pair_min[0, 1], pair_min[1, 0]))
    clash = summary.get("has_clash")
    if not isinstance(clash, (bool, int, float)) or clash not in (0, 1):
        raise ValueError("AF3 has_clash must be 0 or 1")
    return {"iptm": finite(summary.get("iptm"), "AF3 ipTM", 0, 1),
            "ranking_score": finite(summary.get("ranking_score"), "AF3 ranking_score",
                                    -float("inf"), float("inf")),
            "has_clash": bool(clash), "binder_plddt": float(np.mean(binder_confidence)),
            "mean_ipae": (ab + ba) / 2, "ipae_min": ipae_min, "pae_a_to_b": ab, "pae_b_to_a": ba,
            "model_cif": str(cif.relative_to(folder)), "_binder_ca_plddt": binder_confidence}


def job_results(folder: Path, job: dict, config: AF3Config) -> tuple[list[dict], str | None]:
    """Use the newest complete attempt, or report the newest partial attempt as failed.

    Never combine samples from different attempts or count top-level best-model
    copies as extra samples. A truncated seed/sample set is not a complete result.
    """
    expected = {(seed, sample) for seed in config.model_seeds
                for sample in range(SAMPLES_PER_SEED)}
    partial = ([], "AF3 output missing")
    for attempt in sorted((folder / "filter" / "af3").glob("run_*/output/" + job["id"]), reverse=True):
        rows, errors = [], []
        for seed, sample in sorted(expected):
            path = attempt / f"seed-{seed}_sample-{sample}"
            try:
                row = sample_metrics(path, job, folder)
                rows.append(dict(row, model_seed=seed, sample=sample))
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                errors.append(f"seed {seed}, sample {sample}: {error}")
        actual = {path.name for path in attempt.glob("seed-*_sample-*") if path.is_dir()}
        wanted = {f"seed-{seed}_sample-{sample}" for seed, sample in expected}
        if actual - wanted:
            errors.append("unexpected seed/sample directories: " + ", ".join(sorted(actual - wanted)))
        if not errors:
            return rows, None
        if partial[1] == "AF3 output missing":
            partial = (rows, "; ".join(errors))
    return partial


def fold_candidates(folder: Path, config: AF3Config) -> int:
    """Batch only unfinished unique sequences into a fresh, non-overwriting attempt."""
    folder = folder.resolve()
    jobs = campaign(folder, config, create=True)["jobs"]
    pending = [job for job in jobs if job_results(folder, job, config)[1] is not None]
    if not pending:
        return 0
    index = 1
    while (folder / "filter" / "af3" / f"run_{index:04d}").exists():
        index += 1
    attempt = folder / "filter" / "af3" / f"run_{index:04d}"
    inputs, outputs = attempt / "input", attempt / "output"
    inputs.mkdir(parents=True)
    for job in pending:
        write_json(inputs / (job["id"] + ".json"),
                   {"name": job["id"], "modelSeeds": list(config.model_seeds),
                    "sequences": job["sequences"], "dialect": "alphafold3", "version": 1})
    command = config.command(inputs, outputs)
    write_json(attempt / "command.json", {"argv": command, "cwd": str(Path(config.script).parent)})
    environment = os.environ.copy()
    # Do not inject GuideFlip's imports into the separate AF3 environment.
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print(f"AF3: {len(pending)} unique sequences, {len(config.model_seeds)} seed(s), "
          f"{SAMPLES_PER_SEED} samples/seed. Log: {attempt / 'af3.log'}", flush=True)
    with (attempt / "af3.log").open("w") as log:
        try:
            result = subprocess.run(command, cwd=Path(config.script).parent,
                                    env=environment, stdout=log, stderr=subprocess.STDOUT,
                                    check=False)
            return result.returncode
        except OSError as error:
            log.write(str(error) + "\n")
            return 1


def summarize_candidates(folder: Path, config: AF3Config, *, runner_error=None) -> bool:
    """Save one AF3 result per candidate, retaining all sample scores and raw outputs."""
    from .settings import FilterRules
    from .results import filter_design, save_failure
    from types import SimpleNamespace

    folder = folder.resolve()
    jobs = campaign(folder, config, create=False)["jobs"]
    records = candidate_records(folder / "design")
    complete = True
    output = folder / "filter/af3"
    rules = FilterRules(min_interface_ptm=config.filters.get('min_iptm'),
                        min_binder_plddt=(config.filters['min_binder_plddt'] / 100
                                         if 'min_binder_plddt' in config.filters else None),
                        max_ipae_min=config.filters.get('max_ipae_min'),
                        max_irmsd=config.filters.get('max_irmsd'))
    with (output / "samples.csv").open("w", newline="") as handle:
        fields = ('job', 'candidates', 'model_seed', 'sample', 'interface_ptm',
                  'binder_plddt', 'mean_ipae', 'ipae_min', 'has_clash', 'ranking_score', 'model_cif')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            rows, error = job_results(folder, job, config)
            error = runner_error or error
            complete = complete and error is None
            for row in rows:
                writer.writerow(dict(job=job['id'], candidates=';'.join(job['candidates']),
                                     interface_ptm=row['iptm'], binder_plddt=row['binder_plddt']/100,
                                     **{k: row[k] for k in fields[2:] if k not in ('interface_ptm', 'binder_plddt')}))
            for name in job['candidates']:
                record = records[name]
                if error:
                    save_failure(folder, name, record['seed'], RuntimeError(error),
                                 stage='validation', backend='af3')
                    continue
                best = max(rows, key=lambda row: row['ranking_score'])
                from .structure import dockq_metrics
                try:
                    geometry = (dockq_metrics(folder / best['model_cif'], folder / f'design/{name}.pdb')
                                if rules.max_irmsd is not None else {})
                except Exception as failure:
                    save_failure(folder, name, record['seed'], failure, stage='validation', backend='af3')
                    complete = False
                    continue
                scores = SimpleNamespace(interface_ptm=best['iptm'], binder_plddt=best['binder_plddt']/100,
                                         ipae_min=best['ipae_min'], irmsd=geometry.get('irmsd'))
                selection = filter_design(scores, rules, complete=record.get('design_complete') is True)
                result = dict(name=name, sequence=record['sequence'], validation_status='completed',
                              evaluator='AF3', same_model_family_as_design=False,
                              interface_ptm=scores.interface_ptm, binder_plddt=scores.binder_plddt,
                              plddt_scale='0-1', mean_ipae=best['mean_ipae'], has_clash=best['has_clash'],
                              ipae_min=best['ipae_min'], ipae_min_units='angstrom', **geometry,
                              ranking_score=best['ranking_score'], model_seed=best['model_seed'],
                              sample=best['sample'], sample_count=len(rows),
                              selection_rule='highest_ranking_score',
                              source_model_cif=best['model_cif'],
                              prediction_settings=config.prediction_settings(),
                              design_complete=record.get('design_complete', False), filter=selection)
                shutil.copy2(folder / best['model_cif'], output / f'{name}.cif')
                write_json(output / f'{name}.json', result)
                filtering = ('filtering disabled' if selection['status'] == 'not_run'
                             else f"filter {selection['status']}")
                print(f'{name}\tAF3 ipTM {scores.interface_ptm:.3f}\tbinder pLDDT {scores.binder_plddt:.3f}'
                      f'\tAF3 validation completed; {filtering}', flush=True)
    return complete
