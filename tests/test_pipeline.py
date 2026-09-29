"""CPU workflow checks with stand-in models, not protein design validation."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import design as workflow
from guideflip.cli import build_parser, configure, main as cli_main
from guideflip.settings import DesignConfig
from guideflip.settings import Design
from guideflip.flow import DesignFailed, run_design
from guideflip.results import FilterRules, filter_design
from guideflip.structure import RESTYPES, sequence_of
from tests.fakes import FakeAlphaFold, FakePrior, always

ROOT = Path(__file__).resolve().parents[1]
FAST = {"prior_gate_min_steps": 3, "prior_gate_patience": 2, "prior_gate_window": 1,
        "prior_gate_max_steps": 8, "max_steps": 4}


class Model(FakeAlphaFold):
    def __init__(self, design, config, **kwargs):
        super().__init__(config.binder_length, "G" * config.binder_length,
                         confidence=always(0.9))
        self.set_seed(config.seed)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        if result.gradient is not None:
            result = dataclasses.replace(result, gradient=result.gradient +
                                         self.rng.normal(size=result.gradient.shape))
        return result

    def score(self, sequence):
        one_hot = np.eye(20)[[RESTYPES.index(aa) for aa in sequence]]
        return dataclasses.replace(self.evaluate(one_hot, backprop=False),
                                   metrics={"ipae_min": 1.0, "ipae_min_per_model": [1.0]*5})


class Prior:
    def predict(self, structure, state):
        return FakePrior(state.length, "G" * state.length).predict(structure, state)


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(workflow, "AlphaFold", Model)
    monkeypatch.setattr(workflow, "SequencePrior", lambda _: Prior())


def arguments(output, seed=0, trajectories=1):
    # CPU workflow tests use AF2 stand-ins; AF3 has separate runner tests.
    settings = output.parent / "amylin_settings.json"
    raw = json.loads((ROOT / "examples/amylin.json").read_text())
    raw["target"] = str(ROOT / "examples/amylin.pdb")
    raw["filters"]["af3"]["enabled"] = False
    raw["filters"]["af2_monomer"]["enabled"] = False
    settings.write_text(json.dumps(raw))
    result = ["--settings", str(settings),
              "--output", str(output), "--seed", str(seed),
              "--trajectories", str(trajectories), "--report-every", "0",
              "--alphafold-params", "unused", "--prior-checkpoint", "unused"]
    for key, value in FAST.items():
        result.extend(["--set", f"{key}={value}"])
    return result


def test_amylin_inputs_and_template_mask():
    pdb = Design.from_json(ROOT / "examples/amylin.json")
    sequence = Design.from_json(ROOT / "examples/asyn.json")
    assert sequence_of(pdb.target.records()) == "KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY"
    assert sequence.target.sequence == "LGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
    assert sequence.target.is_predicted and len(sequence.target.sequence) == 41
    assert pdb.binder_length() == sequence.binder_length() == 105
    assert pdb.filters == sequence.filters
    assert pdb.prep_arguments(omit_amino_acids="C", seed=0)["rm_target"] == "A1-37"


def test_full_and_minimal_examples_resolve_to_same_design():
    from guideflip.settings import FilterPlan
    minimal = Design.from_json(ROOT / 'examples/amylin.json')
    full = Design.from_json(ROOT / 'examples/amylin_full.json')
    assert full.target == minimal.target
    assert full.binder == minimal.binder
    assert full.trajectories == minimal.trajectories == 1
    assert configure(full, 0, []) == configure(minimal, 0, [])
    assert FilterPlan.from_settings(full.filters) == FilterPlan.from_settings(minimal.filters)
    assert full.prep_arguments(omit_amino_acids='C', seed=0) == minimal.prep_arguments(
        omit_amino_acids='C', seed=0)


def test_complete_run_outputs_and_default_acceptance(tmp_path, models):
    assert workflow.main(arguments(tmp_path)) == 0
    record = json.loads((tmp_path / "design" / "amylin_seed0.json").read_text())
    assert record["sequence"] == "G" * 105
    assert record["design_complete"] is True
    assert {row["lambda_t"] for row in record["trace"]} == {0, 1}
    assert [row["step"] for row in record["trace"]] == list(range(1, len(record["trace"]) + 1))
    validation = json.loads((tmp_path / "filter/af2/amylin_seed0.json").read_text())
    assert validation["filter"]["status"] == "passed"
    assert validation["evaluator"] == "AF2-Multimer"
    assert (tmp_path / "design" / "amylin_seed0.pdb").is_file()
    assert "G" * 105 in (tmp_path / "design/sequences.fasta").read_text()
    assert "G" * 105 in (tmp_path / "accept/sequences.fasta").read_text()
    summary = json.loads((tmp_path / "design/run_summary.json").read_text())
    assert summary["validated"] == 1


def test_batched_seeds_match_independent_runs(tmp_path, models):
    batch = tmp_path / "batch"
    assert workflow.main(arguments(batch, trajectories=2)) == 0
    for seed in (0, 1):
        single = tmp_path / f"single_{seed}"
        assert workflow.main(arguments(single, seed=seed)) == 0
        assert (batch / "design" / f"amylin_seed{seed}.json").read_bytes() == (
            single / "design" / f"amylin_seed{seed}.json").read_bytes()
        assert (batch / "design" / f"amylin_seed{seed}.pdb").read_bytes() == (
            single / "design" / f"amylin_seed{seed}.pdb").read_bytes()


def test_existing_results_are_not_overwritten(tmp_path, models):
    workflow.main(arguments(tmp_path))
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        workflow.main(arguments(tmp_path))
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_generation_failure_is_recorded_and_next_seed_runs(tmp_path, models, monkeypatch):
    real = workflow.run_design

    def first_fails(config, *args, **kwargs):
        if config.seed == 0:
            raise DesignFailed("synthetic failure")
        return real(config, *args, **kwargs)

    monkeypatch.setattr(workflow, "run_design", first_fails)
    assert workflow.main(arguments(tmp_path, trajectories=2)) == 0
    assert not (tmp_path / "design" / "amylin_seed0.pdb").exists()
    assert (tmp_path / "design" / "amylin_seed1.json").exists()
    assert json.loads((tmp_path / "design/failures.jsonl").read_text())["seed"] == 0
    assert json.loads((tmp_path / "design/run_summary.json").read_text())["generation_failed"] == 1


def test_all_generation_failed_returns_nonzero(tmp_path, models, monkeypatch):
    def fail(*args, **kwargs):
        raise DesignFailed("synthetic failure")

    monkeypatch.setattr(workflow, "run_design", fail)
    assert workflow.main(arguments(tmp_path)) == 1
    assert (tmp_path / "design/sequences.fasta").read_text() == ""


def test_sequence_target_is_prepared_once(tmp_path, models, monkeypatch):
    calls = []

    def fold(target, **kwargs):
        calls.append(target.sequence)
        from Bio.SeqUtils import seq3
        from tests.pdb_records import chain_lines
        from guideflip.models.alphafold import withheld_after_folding
        path = tmp_path / "initial.pdb"
        path.write_text("".join(chain_lines("A", [seq3(aa).upper() for aa in target.sequence])))
        return dataclasses.replace(target, structure=str(path),
                                   rm_target=withheld_after_folding(target, len(target.sequence)))

    monkeypatch.setattr(workflow, "fold_target", fold)
    args = arguments(tmp_path, trajectories=2)
    args[1] = str(ROOT / "examples/asyn.json")
    # Check target preparation in the design worker; AF3 runs in a separate runtime.
    assert workflow.design_stage(build_parser(direct=True).parse_args(args)) == 0
    assert len(calls) == 1 and len(calls[0]) == 41
    assert (tmp_path / "design" / "asyn_100_140_seed1.json").exists()


def test_length_change_rebuilds_model_without_loading_real_jax(tmp_path, monkeypatch, models):
    lengths, cleared = [], []

    def model(design, config, **kwargs):
        lengths.append(config.binder_length)
        return Model(design, config, **kwargs)

    monkeypatch.setattr(workflow, "AlphaFold", model)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(clear_caches=lambda: cleared.append(True)))
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"name": "variable", "target": str(ROOT / "examples/amylin.pdb"),
                                    "binder_len": [8, 10]}))
    args = arguments(tmp_path / "output", trajectories=2)
    args[1] = str(settings)
    assert workflow.main(args) == 0
    assert lengths == [10, 9] and cleared == [True]


def test_missing_weights_return_diagnostic_without_creating_output(tmp_path, monkeypatch):
    monkeypatch.delenv("GUIDEFLIP_AF_PARAMS", raising=False)
    monkeypatch.delenv("GUIDEFLIP_PRIOR_CHECKPOINT", raising=False)
    output = tmp_path / "not_created"
    assert workflow.main(["--settings", str(ROOT / "examples/amylin.json"),
                          "--output", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize("seed,count", [(-1, 1), (2**32 - 1, 2), (0, 0)])
def test_invalid_run_counts_fail_before_models(tmp_path, seed, count):
    with pytest.raises(ValueError):
        workflow.main(arguments(tmp_path, seed=seed, trajectories=count))


@pytest.mark.parametrize("field,value", [("unknown_backend", {}), ("sequences_per_trajectory", 3)])
def test_deferred_settings_are_rejected_explicitly(tmp_path, field, value):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"target_sequence": "AAA", "binder_len": 8, field: value}))
    with pytest.raises(ValueError, match="unknown field"):
        Design.from_json(path)


def test_cli_and_script_share_the_same_workflow(tmp_path, models):
    args = arguments(tmp_path)
    assert cli_main(["design", args[1], *args[2:]]) == 0


@pytest.mark.parametrize("complete,score,status", [
    (True, 0.9, "passed"), (True, 0.6, "rejected"),
    (False, 0.9, "rejected"), (True, float("nan"), "rejected")])
def test_screening_requires_valid_scores_and_complete_flow(complete, score, status):
    final = SimpleNamespace(interface_ptm=score, binder_plddt=0.9, ipae_min=1.0, irmsd=1.0)
    assert filter_design(final, FilterRules(min_interface_ptm=0.7),
                         complete=complete)["status"] == status


def test_parameter_overrides_preserve_unspecified_loss_weights():
    design = Design.from_json(ROOT / "examples/amylin.json")
    design = dataclasses.replace(design, overrides={"loss": {"plddt": 0.2}})
    config = configure(design, 4, ["max_steps=10"])
    assert config.seed == 4 and config.max_steps == 10
    assert config.loss.plddt == 0.2 and config.loss.contacts_inter == 5.0


def test_single_flow_gate_does_not_call_prior_before_enabling_it():
    cfg = DesignConfig(binder_length=8, **FAST)
    model = Model(None, cfg)
    prior = FakePrior(8, "G" * 8)
    result = run_design(cfg, model, prior)
    assert prior.calls == sum(row["lambda_t"] == 1 for row in result.trace)
    first_prior = next(row["step"] for row in result.trace if row["lambda_t"] == 1)
    assert result.prior_enabled_at == first_prior
    assert result.retained_step < first_prior


def test_direct_help_works_without_model_dependencies():
    proc = subprocess.run([sys.executable, str(ROOT / "design.py"), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--settings" in proc.stdout and "--stage" not in proc.stdout


@pytest.mark.parametrize("invalid", [False, True])
def test_validation_failure_preserves_generation(tmp_path, models, monkeypatch, invalid):
    original = Model.score
    def score(self, sequence):
        if invalid:
            return dataclasses.replace(original(self, sequence), interface_ptm=float("nan"))
        raise RuntimeError("synthetic validation failure")
    monkeypatch.setattr(Model, "score", score)
    assert workflow.main(arguments(tmp_path)) == 1
    assert (tmp_path / "design/amylin_seed0.pdb").is_file()
    assert json.loads((tmp_path / "design/amylin_seed0.json").read_text())["sequence"]
    report = json.loads((tmp_path / "design/run_summary.json").read_text())
    assert report["generated"] == report["validation_failed"] == 1
    assert report["accepted"] == report["validated"] == 0
    assert not (tmp_path / "accept/amylin_seed0.pdb").exists()


def test_accept_folder_contains_only_explicit_passes(tmp_path, models):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"name": "screen", "target": str(ROOT / "examples/amylin.pdb"),
                                   "binder_len": 105, "filters": {"min_binder_plddt": 0.8}}))
    args = arguments(tmp_path / "output")
    args[1] = str(settings)
    assert workflow.main(args) == 0
    root = tmp_path / "output"
    assert (root / "accept/screen_seed0.pdb").read_bytes() == (root / "filter/af2/screen_seed0.pdb").read_bytes()
    assert json.loads((root / "design/run_summary.json").read_text())["accepted"] == 1
