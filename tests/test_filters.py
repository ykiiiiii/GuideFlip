"""Synthetic AF3 adapter tests; these do not perform model inference."""
import csv
import dataclasses
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import design as workflow
from guideflip import validation as af3
from guideflip.settings import FilterPlan
from guideflip.results import summarize
from guideflip.validation import af3_config
from guideflip.settings import Design
from tests.pdb_records import chain_lines
from tests.test_pipeline import models, arguments, Model, ROOT


@pytest.mark.parametrize('changes,passed', [({}, True), ({'interface_ptm': .799}, False),
    ({'ipae_min': 1.501}, False), ({'irmsd': 2.001}, False),
    ({'ipae_min': None}, False), ({'irmsd': float('nan')}, False)])
def test_default_three_thresholds_are_inclusive_and_missing_never_passes(changes, passed):
    from guideflip.settings import FilterRules
    from guideflip.results import filter_design
    values = dict(interface_ptm=.8, binder_plddt=.9, ipae_min=1.5, irmsd=2.)
    values.update(changes)
    assert filter_design(SimpleNamespace(**values), FilterRules.from_settings({}),
                         complete=True)['passed'] == passed


def test_explicit_null_disables_all_default_thresholds():
    from guideflip.settings import FilterRules
    from guideflip.results import filter_design
    rules = FilterRules.from_settings(dict(min_interface_ptm=None, max_ipae_min=None, max_irmsd=None))
    assert filter_design(SimpleNamespace(), rules, complete=True)['status'] == 'not_run'


@pytest.mark.parametrize('raw', [{'max_ipae_min': -1}, {'max_irmsd': float('inf')},
                               {'max_irmsd': True}, {'min_interface_ptm': 1.1}])
def test_invalid_default_threshold_overrides(raw):
    from guideflip.settings import FilterRules
    with pytest.raises(ValueError):
        FilterRules.from_settings(raw)


def test_af2_minimum_is_per_model_bidirectional_then_averaged():
    from guideflip.models.alphafold import AlphaFold
    from guideflip.settings import DesignConfig
    pae = np.full((5, 3, 3), 30.)
    expected = []
    for i in range(5):
        pae[i, 0, 1:] = [i + 1., 20.]
        pae[i, 1:, 0] = [20., i + .5]
        expected.append(i + .5)
    fake = SimpleNamespace(opt={'dropout': True}, aux=dict(loss=1., losses={'plddt': .1},
                           i_ptm=.9, log={}, all={'pae': pae}))
    fake.set_opt = lambda **kwargs: fake.opt.update(kwargs)
    fake.run = lambda **kwargs: None
    af = AlphaFold.__new__(AlphaFold)
    af._model, af.binder_length, af._target_length = fake, 2, 1
    af._config = DesignConfig(binder_length=2)
    af._read_complex = lambda: None
    result = af.score('AA')
    assert result.metrics['ipae_min_per_model'] == expected
    assert result.metrics['ipae_min'] == np.mean(expected)
    assert fake.opt['dropout'] is True


def test_official_dockq_is_invariant_to_rigid_translation_and_residue_numbers(tmp_path):
    from guideflip.structure import dockq_metrics
    from tests.fakes import complex_of
    reference = tmp_path/'reference.pdb'
    reference.write_text(complex_of(6).pdb)
    lines = []
    for line in reference.read_text().splitlines():
        if line.startswith('ATOM'):
            line = line[:22] + f'{int(line[22:26])+100:4d}' + line[26:30] + f'{float(line[30:38])+12:8.3f}' + line[38:]
        lines.append(line)
    model = tmp_path/'model.pdb'
    model.write_text('\n'.join(lines)+'\n')
    result = dockq_metrics(model, reference)
    assert result['irmsd'] < 1e-5 and result['dockq_version'] == '2.1.3'


def candidate(root, name='demo_seed0', backends=('af3',), **changes):
    for directory in ('design', 'filter/af2', 'filter/af3', 'accept'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    record = dict(name=name, seed=0, sequence='GGGG', target_sequence='AAA',
                  design_complete=True, generation_status='completed', validation_backends=list(backends))
    record.update(changes)
    af3.write_json(root / 'design' / f'{name}.json', record)
    return record


def config(root, **changes):
    weights=root/'weights'
    weights.mkdir(exist_ok=True)
    (weights/'af3.bin').write_bytes(b'fixture')
    runner=root/'runner.py'
    runner.touch()
    return af3.AF3Config.from_settings(dict(python=sys.executable, script=str(runner),
                                           model_dir=str(weights), **changes), beside=root)


def write_sample(folder, job, sample, *, attempt=1, seed=1, iptm=0.8, ranking=0.8, plddt=80):
    from Bio.PDB import MMCIFIO, PDBParser
    from Bio.SeqUtils import seq3
    path = folder / "filter/af3" / f"run_{attempt:04d}" / "output" / job["id"] / f"seed-{seed}_sample-{sample}"
    path.mkdir(parents=True, exist_ok=True)
    text, token_chains, token_ids = [], [], []
    for entry in job["sequences"]:
        protein = entry["protein"]
        chain, sequence = protein["id"], protein["sequence"]
        text += chain_lines(chain, [seq3(aa).upper() for aa in sequence], b=plddt)
        token_chains += [chain] * len(sequence)
        token_ids += list(range(1, len(sequence) + 1))
    structure = PDBParser(QUIET=True).get_structure("test", StringIO("".join(text)))
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(path / "model.cif"))
    count = len(token_chains)
    a = np.flatnonzero(np.array(token_chains) == "A")
    b = np.flatnonzero(np.array(token_chains) == "B")
    pae = np.zeros((count, count))
    cross = np.arange(1, len(a) * len(b) + 1).reshape(len(a), len(b))
    pae[np.ix_(a, b)] = cross
    pae[np.ix_(b, a)] = cross.T * 2
    af3.write_json(path / "confidences.json", {"pae": pae.tolist(),
                   "token_chain_ids": token_chains, "token_res_ids": token_ids})
    af3.write_json(path / "summary_confidences.json", {"iptm": iptm, "ranking_score": ranking,
                                                       "has_clash": 0, "chain_pair_pae_min": [[0, 1], [2, 0]]})
    return path


@pytest.mark.parametrize('raw,backends', [({}, ['af2']), ({'min_interface_ptm': .7}, ['af2']),
    ({'af2': {'enabled': False}, 'af3': {'enabled': True}}, ['af3']),
    ({'af2': {}, 'af3': {}}, ['af2', 'af3']),
    ({'af2': {'enabled': False}, 'af3': {'enabled': False}}, [])])
def test_backend_selection(raw, backends):
    assert FilterPlan.from_settings(raw).backends == backends


@pytest.mark.parametrize('raw', [{'af2': {'enabled': 'false'}}, {'af3': {'min_binder_plddt': 80}},
    {'af2': {}, 'min_interface_ptm': .7}, {'af3': {'unknown': 1}}, {'af3': True}])
def test_bad_settings_fail_before_loading_models(raw):
    with pytest.raises(ValueError):
        FilterPlan.from_settings(raw)


def test_af3_config_uses_shared_score_units(tmp_path):
    cfg = config(tmp_path)
    path = tmp_path/'settings.json'
    path.write_text(json.dumps(dict(target_sequence='AAA', binder_len=4,
        filters={'af2': {'enabled': False}, 'af3': {'min_interface_ptm': .7, 'min_binder_plddt': .8}},
        af3={key: getattr(cfg,key) for key in ('python','script','model_dir')})))
    parsed = af3_config(Design.from_json(path), path)
    assert parsed.filters == {'min_iptm': .7, 'min_binder_plddt': 80, 'max_ipae_min': 1.5, 'max_irmsd': 2.0}
    parsed.check_installation()


@pytest.mark.parametrize('settings,expected', [({}, 'xla'),
    ({'flash_attention_implementation': 'xla'}, 'xla'),
    ({'flash_attention_implementation': 'triton'}, 'triton'),
    ({'flash_attention_implementation': 'cudnn'}, 'cudnn'),
    ({'flash_attention_implementation': None}, None)])
def test_af3_attention_defaults_and_overrides_reach_runner(tmp_path, settings, expected):
    cfg = config(tmp_path, **settings)
    flags = [arg for arg in cfg.command(tmp_path/'inputs', tmp_path/'outputs')
             if arg.startswith('--flash_attention_implementation=')]
    assert flags == ([] if expected is None else [f'--flash_attention_implementation={expected}'])
    assert cfg.prediction_settings()['flash_attention_implementation'] == expected


def test_minimal_af3_example_uses_saved_runtime(tmp_path, monkeypatch):
    saved = tmp_path/'installation.json'
    saved.write_text(json.dumps(dict(python='venv/python', script='source/run.py', model_dir='weights')))
    monkeypatch.setenv('GUIDEFLIP_AF3_CONFIG', str(saved))
    for name in ('PYTHON', 'SCRIPT', 'MODEL_DIR'):
        monkeypatch.delenv('GUIDEFLIP_AF3_' + name, raising=False)
    path = ROOT/'examples/amylin.json'
    design = Design.from_json(path)
    cfg = af3_config(design, path)
    assert FilterPlan.from_settings(design.filters).backends == ['af2', 'af3', 'af2_monomer']
    assert cfg.python == str(tmp_path/'venv/python')
    assert cfg.script == str(tmp_path/'source/run.py')
    assert cfg.model_dir == str(tmp_path/'weights')
    assert cfg.model_seeds == (1,) and cfg.filters == {'min_iptm': .8, 'max_ipae_min': 1.5, 'max_irmsd': 2.0}
    assert cfg.flash_attention_implementation == 'xla'


def test_full_af3_example_null_paths_use_saved_runtime(tmp_path, monkeypatch):
    saved = tmp_path/'installation.json'
    saved.write_text(json.dumps(dict(python='venv/python', script='source/run.py', model_dir='weights')))
    monkeypatch.setenv('GUIDEFLIP_AF3_CONFIG', str(saved))
    for name in ('PYTHON', 'SCRIPT', 'MODEL_DIR'):
        monkeypatch.delenv('GUIDEFLIP_AF3_' + name, raising=False)
    path = ROOT/'examples/amylin_full.json'
    design = Design.from_json(path)
    design = dataclasses.replace(design, filters={'af3': {'enabled': True}})
    cfg = af3_config(design, path)
    assert cfg.python == str(tmp_path/'venv/python')
    assert cfg.script == str(tmp_path/'source/run.py')
    assert cfg.model_dir == str(tmp_path/'weights')
    assert cfg.model_seeds == (1,) and cfg.flash_attention_implementation == 'xla'


def test_disabled_validation_never_calls_af2_score(tmp_path, models, monkeypatch):
    def forbidden(*args):
        pytest.fail('disabled AF2 validation must not run')
    monkeypatch.setattr(Model, 'score', forbidden)
    settings = tmp_path/'settings.json'
    settings.write_text(json.dumps(dict(name='disabled', target=str(ROOT/'examples/amylin.pdb'),
        binder_len=105, filters={'af2': {'enabled': False}, 'af3': {'enabled': False}})))
    args=arguments(tmp_path/'out')
    args[1]=str(settings)
    assert workflow.main(args)==0
    root=tmp_path/'out'
    assert not (root/'filter/af2').exists()
    report=json.loads((root/'design/run_summary.json').read_text())
    assert report['generated']==1 and report['validated']==report['accepted']==0


def test_af3_deduplication_and_selected_sample(tmp_path):
    cfg=config(tmp_path, filters={'min_iptm': .7})
    root=tmp_path/'out'
    candidate(root)
    candidate(root, 'demo_seed1')
    job=af3.campaign(root,cfg,create=True)['jobs'][0]
    assert len(job['candidates'])==2
    for entry in job['sequences']:
        assert entry['protein']['unpairedMsa']==entry['protein']['pairedMsa']==''
        assert entry['protein']['templates']==[]
    for sample in range(5):
        write_sample(root,job,sample,iptm=.9 if sample<4 else .5,ranking=sample/5,plddt=80)
    assert af3.summarize_candidates(root,cfg)
    saved=af3.read_json(root/'filter/af3/demo_seed0.json')
    assert saved['sample']==4 and saved['interface_ptm']==.5
    assert saved['binder_plddt']==.8 and saved['mean_ipae']==9.75
    assert saved['filter']['status']=='rejected'
    summarize(root)
    assert af3.read_json(root/'design/run_summary.json')['accepted']==0
    assert len(list(csv.DictReader((root/'filter/af3/samples.csv').open())))==5


@pytest.mark.parametrize('damage', ['missing', 'nan', 'mapping', 'runner'])
def test_invalid_or_failed_af3_cannot_pass(tmp_path, damage):
    cfg=config(tmp_path, filters={'min_iptm': .1})
    root=tmp_path/'out'
    candidate(root)
    job=af3.campaign(root,cfg,create=True)['jobs'][0]
    for sample in range(4 if damage=='missing' else 5):
        path=write_sample(root,job,sample)
    if damage=='nan':
        af3.write_json(path/'summary_confidences.json', dict(iptm=None,ranking_score=.8,has_clash=0))
    if damage=='mapping':
        data=af3.read_json(path/'confidences.json')
        data['token_res_ids'][0]=9
        af3.write_json(path/'confidences.json',data)
    assert not af3.summarize_candidates(root,cfg,runner_error='exit 1' if damage=='runner' else None)
    summarize(root)
    report=af3.read_json(root/'design/run_summary.json')
    assert report['accepted']==report['validated']==0 and report['validation_failed']==1


@pytest.mark.parametrize('af2_pass', [True,False])
def test_both_selected_requires_both_pass_and_uses_af3_structure(tmp_path, af2_pass):
    root=tmp_path/'out'
    candidate(root, backends=('af2','af3'))
    cfg=config(tmp_path, filters={'min_iptm': .7})
    job=af3.campaign(root,cfg,create=True)['jobs'][0]
    for sample in range(5): write_sample(root,job,sample)
    af3.summarize_candidates(root,cfg)
    # AF3 cannot accept a design before the required AF2 validation finishes.
    summarize(root)
    assert af3.read_json(root/'design/run_summary.json')['accepted']==0
    af3.write_json(root/'filter/af2/demo_seed0.json', dict(validation_status='completed',
        filter=dict(criteria={'min_interface_ptm':.7}, passed=af2_pass,
                    status='passed' if af2_pass else 'rejected',reasons=[])))
    summarize(root)
    assert af3.read_json(root/'design/run_summary.json')['accepted']==int(af2_pass)
    assert (root/'accept/demo_seed0.cif').exists()==af2_pass


def test_parent_waits_for_design_exit_before_af3(tmp_path, monkeypatch):
    from guideflip import validation
    root=tmp_path/'out'
    cfg=config(tmp_path)
    path=tmp_path/'settings.json'
    path.write_text(json.dumps(dict(name='demo',target_sequence='AAA',binder_len=4,
        filters={'af3':{}}, af3={key:getattr(cfg,key) for key in ('python','script','model_dir')})))
    args=arguments(root)
    args[1]=str(path)
    events=[]
    def child(command, **kwargs):
        assert '--_design-worker' in command
        candidate(root)
        events.append('child_exited')
        return SimpleNamespace(returncode=0)
    def fold(*args):
        assert events==['child_exited']
        events.append('af3_started')
        return 0
    monkeypatch.setattr(validation.subprocess,'run',child)
    monkeypatch.setattr(af3,'fold_candidates',fold)
    monkeypatch.setattr(af3,'summarize_candidates',lambda *a,**k: True)
    assert workflow.main(args)==0
    assert events==['child_exited','af3_started']


def test_real_external_runner_invocation_and_log(tmp_path):
    cfg=config(tmp_path, filters={'min_iptm': .7})
    root=tmp_path/'out'
    candidate(root)
    job=af3.campaign(root,cfg,create=True)['jobs'][0]
    fixture=tmp_path/'fixture'
    for sample in range(5): write_sample(fixture,job,sample)
    source=fixture/'filter/af3/run_0001/output'/job['id']
    Path(cfg.script).write_text('import sys,shutil\nfrom pathlib import Path\n'
        'args=sys.argv\noutput=Path(args[args.index("--output_dir")+1])\n'
        'assert "--flash_attention_implementation=xla" in args\n'
        f'shutil.copytree({str(source)!r}, output/{job["id"]!r})\nprint("synthetic runner completed")\n')
    assert af3.fold_candidates(root,cfg)==0
    assert af3.summarize_candidates(root,cfg)
    summarize(root)
    assert af3.read_json(root/'design/run_summary.json')['accepted']==1
    assert 'synthetic runner completed' in (root/'filter/af3/run_0001/af3.log').read_text()
    assert af3.fold_candidates(root,cfg)==0
    assert not (root/'filter/af3/run_0002').exists()
    settings=tmp_path/'retry.json'
    settings.write_text(json.dumps(dict(target_sequence='AAA',binder_len=4,
        filters={'af3': {'min_interface_ptm': .7, 'max_ipae_min': None, 'max_irmsd': None}},
        af3={key:getattr(cfg,key) for key in ('python','script','model_dir')})))
    from guideflip.cli import main
    assert main(['validate-af3',str(settings),'--output',str(root)])==0
    assert not (root/'filter/af3/run_0002').exists()


@pytest.mark.parametrize('layout', ['prefixed', 'mixed', 'ambiguous'])
def test_af3_versioned_filenames_are_read_as_one_consistent_set(tmp_path, layout):
    import shutil
    cfg=config(tmp_path)
    root=tmp_path/'out'
    candidate(root)
    job=af3.campaign(root,cfg,create=True)['jobs'][0]
    for sample in range(5):
        path=write_sample(root,job,sample)
        names=('summary_confidences.json','confidences.json','model.cif')
        for name in names if layout!='mixed' else names[:1]:
            old=path/name
            new=path/f"{job['id']}_{path.name}_{name}"
            if layout=='ambiguous': shutil.copy2(old,new)
            else: old.rename(new)
    ok=af3.summarize_candidates(root,cfg)
    assert ok==(layout=='prefixed')
    saved=af3.read_json(root/'filter/af3/demo_seed0.json')
    assert saved['validation_status']==('completed' if ok else 'failed')


def test_monomer_defaults_record_scores_and_reject_interface_thresholds():
    from guideflip.settings import MonomerRules
    from guideflip.results import filter_monomer
    plan = FilterPlan.from_settings({'af2_monomer': {'enabled': True}})
    assert plan.backends == ['af2_monomer']
    assert filter_monomer({'binder_plddt': .9, 'rmsd': 1.}, plan.af2_monomer,
                          complete=True)['status'] == 'not_run'
    for raw in ({'min_interface_ptm': .8}, {'min_binder_plddt': 80},
                {'max_rmsd': -1}, {'max_rmsd': True}, {'max_rmsd': float('nan')}):
        with pytest.raises(ValueError):
            MonomerRules.from_settings(raw)
    rules = MonomerRules(min_binder_plddt=.8, max_rmsd=2.)
    assert filter_monomer({'binder_plddt': .8, 'rmsd': 2.}, rules, complete=True)['passed']
    assert not filter_monomer({'binder_plddt': .9, 'rmsd': 2.001}, rules, complete=True)['passed']


def test_monomer_rmsd_uses_only_binder_and_removes_rigid_motion(tmp_path):
    from guideflip.structure import binder_monomer_rmsd
    from tests.pdb_records import atom
    xyz = np.array([[0., 0., 0.], [4., 0., 0.], [0., 5., 0.], [0., 0., 6.]])
    ref = tmp_path/'design.pdb'
    # Different residue identities in the soft design must not drop positions.
    ref.write_text(atom(1, ' CA ', 'TRP', 'A', 1, [100., 100., 100.]) +
                   ''.join(atom(i+2, ' CA ', 'GLY', 'B', i+11, x) for i,x in enumerate(xyz)))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    mono = tmp_path/'monomer.pdb'
    moved = xyz @ rotation + [12., -7., 22.]
    mono.write_text(''.join(atom(i+1, ' CA ', 'ALA', 'A', i+1, x) for i,x in enumerate(moved)))
    assert binder_monomer_rmsd(mono, ref, 'AAAA')['rmsd'] < 1e-6
    moved[-1] += [2., 3., 4.]
    mono.write_text(''.join(atom(i+1, ' CA ', 'ALA', 'A', i+1, x) for i,x in enumerate(moved)))
    assert binder_monomer_rmsd(mono, ref, 'AAAA')['rmsd'] > .5
    with pytest.raises(ValueError, match='final binder sequence'):
        binder_monomer_rmsd(mono, ref, 'AAAG')
    with mono.open('a') as f:
        f.write(atom(10, ' CA ', 'ALA', 'B', 1, [0., 0., 0.]))
    with pytest.raises(ValueError, match='only binder'):
        binder_monomer_rmsd(mono, ref, 'AAAA')


def test_monomer_predictor_receives_no_target_or_template(tmp_path, monkeypatch):
    from guideflip.models.alphafold import BinderMonomer
    calls = {}
    class Predictor:
        _model_names = [f'model_{i}_ptm' for i in range(1, 6)]
        aux = {'all': {'plddt': np.full((5, 4), .88)}}
        def prep_inputs(self, **kwargs): calls['inputs'] = kwargs
        def predict(self, **kwargs): calls['predict'] = kwargs
        def save_pdb(self, filename, get_best): Path(filename).write_text('prediction')
    def factory(**kwargs):
        calls['constructor'] = kwargs
        return Predictor()
    monkeypatch.setitem(sys.modules, 'colabdesign', SimpleNamespace(mk_afdesign_model=factory))
    model = BinderMonomer(4, params_dir='unused')
    result = model.predict('AAAA', seed=7, path=str(tmp_path/'monomer.pdb'))
    assert calls['inputs'] == {'length': 4}
    assert calls['constructor']['use_multimer'] is False
    assert calls['constructor']['use_templates'] is False
    assert calls['predict']['seq'] == 'AAAA' and calls['predict']['seed'] == 7
    assert calls['predict']['dropout'] is False and calls['predict']['num_models'] == 5
    assert result['binder_plddt'] == pytest.approx(.88)
    assert result['binder_plddt_per_model'] == pytest.approx([.88]*5)


def test_monomer_worker_failure_is_recorded(tmp_path, monkeypatch):
    from guideflip.models import alphafold
    args = workflow.build_parser(direct=True).parse_args(arguments(tmp_path/'out'))
    raw = json.loads(Path(args.settings).read_text())
    raw['filters'] = {'af2_monomer': {'enabled': True}}
    Path(args.settings).write_text(json.dumps(raw))
    root = Path(args.output)
    (root/'design').mkdir(parents=True)
    af3.write_json(root/'design/demo_seed0.json', dict(name='demo_seed0', seed=0,
        sequence='AAAA', validation_backends=['af2_monomer'], design_complete=True))
    class Broken:
        def __init__(self, *a, **k): pass
        def predict(self, *a, **k): raise RuntimeError('synthetic monomer failure')
    monkeypatch.setattr(alphafold, 'BinderMonomer', Broken)
    monkeypatch.setattr(af3, 'check_monomer_weights', lambda *a: None)
    assert af3.monomer_candidates(args) == 1
    result = af3.read_json(root/'filter/af2_monomer/demo_seed0.json')
    assert result['validation_status'] == 'failed' and result['evaluator'] == 'AF2-Monomer'


def test_parent_waits_for_monomer_exit_before_af3(tmp_path, monkeypatch):
    root = tmp_path/'out'
    cfg = config(tmp_path)
    path = tmp_path/'settings.json'
    path.write_text(json.dumps(dict(name='demo', target_sequence='AAA', binder_len=4,
        filters={'af3': {}, 'af2_monomer': {}},
        af3={key:getattr(cfg,key) for key in ('python','script','model_dir')})))
    args = arguments(root); args[1] = str(path)
    events = []
    def child(command, **kwargs):
        if '--_design-worker' in command:
            candidate(root)
            events.append('design_exited')
        else:
            assert '--_monomer-worker' in command
            assert events == ['design_exited']
            events.append('monomer_exited')
        return SimpleNamespace(returncode=0)
    def fold(*args):
        assert events == ['design_exited', 'monomer_exited']
        events.append('af3_started')
        return 0
    monkeypatch.setattr(af3, 'check_monomer_weights', lambda *a: None)
    monkeypatch.setattr(af3.subprocess, 'run', child)
    monkeypatch.setattr(af3, 'fold_candidates', fold)
    monkeypatch.setattr(af3, 'summarize_candidates', lambda *a,**k: True)
    assert workflow.main(args) == 0
    assert events == ['design_exited', 'monomer_exited', 'af3_started']


def test_monomer_summary_records_metrics_and_requires_completion(tmp_path):
    root = tmp_path/'out'
    candidate(root)
    design_path = root/'design/demo_seed0.json'
    record = af3.read_json(design_path)
    record['validation_backends'] = ['af2_monomer']
    af3.write_json(design_path, record)
    (root/'filter/af2_monomer').mkdir(parents=True)
    summarize(root)
    assert af3.read_json(root/'design/run_summary.json')['validated'] == 0
    path = root/'filter/af2_monomer/demo_seed0.json'
    af3.write_json(path, dict(validation_status='completed', evaluator='AF2-Monomer',
        binder_plddt=.87, rmsd=1.23, filter=dict(status='not_run', passed=None, criteria={}, reasons=[])))
    summarize(root)
    summary = af3.read_json(root/'design/run_summary.json')
    assert summary['validated'] == 1 and summary['accepted'] == 0
    row = next(csv.DictReader((root/'design/summary.csv').open()))
    assert row['af2_monomer_status'] == 'completed'
    assert float(row['af2_monomer_binder_plddt']) == .87
    assert float(row['af2_monomer_rmsd']) == 1.23
    assert row['filter_status'] == 'not_run'
