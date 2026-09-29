"""Generation records, AF2 validation and accepted candidates in separate folders."""
from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path
import shutil
import sys

from .settings import FilterRules, FilterPlan
from .structure import BINDER_CHAIN, sequence_of, dockq_metrics


def filter_design(final, rules: FilterRules, *, complete: bool) -> dict:
    criteria = {name: value for name, value in dataclasses.asdict(rules).items()
                if value is not None}
    if not criteria:
        return {"status": "not_run", "passed": None, "criteria": {}, "reasons": []}
    reasons = []
    if not complete:
        reasons.append("flow ended before all designable positions committed")
    for name in ("interface_ptm", "binder_plddt"):
        value, minimum = getattr(final, name), getattr(rules, "min_" + name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            reasons.append(f"{name} is not a finite score in [0, 1]")
        elif minimum is not None and value < minimum:
            reasons.append(f"{name}={value:.6g} < {minimum:.6g}")
    for name in ("ipae_min", "irmsd"):
        maximum = getattr(rules, "max_" + name)
        if maximum is None:
            continue
        value = getattr(final, name, None)
        if value is None or not math.isfinite(value) or value < 0:
            reasons.append(f"{name} is unavailable or invalid")
        elif value > maximum:
            reasons.append(f"{name}={value:.6g} > {maximum:.6g} A")
    return {"status": "rejected" if reasons else "passed", "passed": not reasons,
            "criteria": criteria, "reasons": reasons}


def filter_monomer(scores, rules, *, complete: bool) -> dict:
    criteria = {key: value for key, value in dataclasses.asdict(rules).items() if value is not None}
    if not criteria:
        return dict(status='not_run', passed=None, criteria={}, reasons=[])
    reasons = [] if complete else ['flow ended before all designable positions committed']
    for metric, threshold, minimum in [('binder_plddt', rules.min_binder_plddt, True),
                                       ('rmsd', rules.max_rmsd, False)]:
        value = scores.get(metric)
        if value is None or not math.isfinite(value) or value < 0 or (minimum and value > 1):
            reasons.append(f'{metric} is unavailable or invalid')
        elif threshold is not None and (value < threshold if minimum else value > threshold):
            reasons.append(f"{metric}={value:.6g} {'<' if minimum else '>'} {threshold:.6g}")
    return dict(status='rejected' if reasons else 'passed', passed=not reasons,
                criteria=criteria, reasons=reasons)


def _write_json(path, value):
    # A lost interface is recorded internally as infinity; portable JSON uses null.
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        return item

    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_output(design, folder, seeds):
    """Check every candidate path before creating folders or loading models."""
    label = design.target.name
    if not label or Path(label).name != label or label in ('.', '..'):
        raise ValueError("settings name must be a filename label, not a path")
    output = Path(folder)
    for seed in seeds:
        for directory in ('design', 'filter/af2', 'filter/af3', 'filter/af2_monomer', 'accept'):
            for suffix in ('.pdb', '.cif', '.json'):
                path = output / directory / f'{label}_seed{seed}{suffix}'
                if path.exists():
                    raise FileExistsError(f'refusing to overwrite {path}; choose a new output or seed')
    predicted = output / 'design' / f'{label}_predicted.pdb'
    if design.target.is_predicted and predicted.exists():
        raise FileExistsError(f'refusing to overwrite {predicted}; choose a new output')
    plan = FilterPlan.from_settings(design.filters)
    if (output / 'filter/af3/jobs.json').exists():
        raise FileExistsError('AF3 campaign exists; use a new output directory')
    for directory in ['design', 'accept', *('filter/' + name for name in plan.backends)]:
        (output / directory).mkdir(parents=True, exist_ok=True)


def save_generation(folder, name, result, config, design, settings):
    """Save the last accepted generation prediction before any validation runs.

    This structure was predicted from the flow's sequence probabilities. It need not
    encode every identity in the final decoded sequence; both sequences are recorded.
    """
    output = Path(folder) / 'design'
    pdb = result.complex.first_model().splitlines()
    structure_sequence = sequence_of(line for line in pdb
                                    if line.startswith('ATOM') and line[21] == BINDER_CHAIN)
    record = {
        'name': name, 'sequence': result.sequence, 'seed': config.seed,
        'target_sequence': sequence_of(design.target.records()),
        'validation_backends': FilterPlan.from_settings(design.filters).backends,
        'generation_status': 'completed' if result.is_complete else 'incomplete',
        'design_complete': result.is_complete,
        'prior_enabled_at': result.prior_enabled_at, 'retained_step': result.retained_step,
        'retained_confidence': result.retained_confidence,
        'interface_guarded': result.interface_guarded,
        'config': dataclasses.asdict(config), 'settings': str(Path(settings).resolve()),
        'structure_source': 'last_accepted_generation_prediction',
        'structure_sequence': structure_sequence,
        'structure_matches_final_sequence': structure_sequence == result.sequence,
        'trace': result.trace,
    }
    if isinstance(design.binder.length, tuple):
        record['binder_len_range'] = list(design.binder.length)
    result.complex.write(str(output / f'{name}.pdb'))
    _write_json(output / f'{name}.json', record)


def save_validation(folder, name, result, final, filters):
    """Score the fixed decoded sequence using the same AF2-Multimer design model."""
    for metric in ('interface_ptm', 'binder_plddt'):
        value = getattr(final, metric)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'AF2 validation returned invalid {metric}: {value}')
    if not math.isfinite(final.loss):
        raise ValueError('AF2 validation returned a non-finite loss')
    from types import SimpleNamespace
    output = Path(folder) / "filter/af2"
    final.complex.write(str(output / f"{name}.pdb"))
    geometry = (dockq_metrics(output / f"{name}.pdb", Path(folder) / f"design/{name}.pdb")
                if filters.max_irmsd is not None else {})
    scores = SimpleNamespace(interface_ptm=final.interface_ptm, binder_plddt=final.binder_plddt,
                             ipae_min=final.metrics.get("ipae_min"), irmsd=geometry.get("irmsd"))
    selection = filter_design(scores, filters, complete=result.is_complete)
    record = {
        'name': name, 'sequence': result.sequence, 'validation_status': 'completed',
        'evaluator': 'AF2-Multimer', 'same_model_family_as_design': True,
        'model_count': 5, 'dropout': False, 'score_aggregation': 'five_model_mean',
        'interface_ptm': final.interface_ptm, 'binder_plddt': final.binder_plddt,
        'loss': final.loss, 'design_complete': result.is_complete, 'filter': selection,
        'ipae_min': scores.ipae_min, 'ipae_min_units': 'angstrom',
        'ipae_min_aggregation': 'mean_of_five_model_bidirectional_minima',
        'ipae_min_per_model': final.metrics.get('ipae_min_per_model'),
        **geometry,
    }
    _write_json(output / f'{name}.json', record)
    filtering = ('filtering disabled' if selection['status'] == 'not_run'
                 else f"filter {selection['status']}")
    print(f'{name}\t{result.sequence}\tipTM {final.interface_ptm:.3f}'
          f"\tpLDDT {final.binder_plddt:.3f}\tAF2 validation completed; {filtering}",
          flush=True)


def save_failure(folder, name, seed, failure, *, stage='generation', backend='af2'):
    """Record failures as rows too; an unevaluated design can never be accepted."""
    print(f'{name}\t{stage} failed: {failure}', file=sys.stderr)
    record = {'name': name, 'seed': seed, 'stage': stage,
              'error_type': type(failure).__name__, 'reason': str(failure)}
    if stage == 'generation':
        record.update(generation_status='failed', design_complete=False)
        _write_json(Path(folder) / 'design' / f'{name}.json', record)
    else:
        evaluator = {'af2': 'AF2-Multimer', 'af3': 'AF3', 'af2_monomer': 'AF2-Monomer'}[backend]
        record.update(validation_status='failed', evaluator=evaluator,
                      filter={'status': 'failed', 'passed': False,
                              'criteria': {}, 'reasons': [str(failure)]})
        _write_json(Path(folder) / 'filter' / backend / f'{name}.json', record)
    with (Path(folder) / 'design/failures.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')


def summarize(folder):
    """Aggregate all selected validators; missing/failed outputs never pass."""
    root = Path(folder)
    columns = ('name', 'seed', 'sequence', 'generation_status', 'design_complete',
               'validation_status', 'evaluator', 'interface_ptm', 'binder_plddt', 'ipae_min', 'irmsd', 'loss',
               'filter_status', 'filter_criteria', 'filter_reasons', 'failure_reason',
               'design_pdb', 'validation_pdb', 'accepted_pdb', 'accepted_cif',
               'af2_status', 'af2_interface_ptm', 'af2_binder_plddt', 'af2_ipae_min', 'af2_irmsd',
               'af3_status', 'af3_interface_ptm', 'af3_binder_plddt', 'af3_ipae_min', 'af3_irmsd', 'af3_cif',
               'af2_monomer_status', 'af2_monomer_binder_plddt', 'af2_monomer_rmsd', 'af2_monomer_pdb')
    counts = dict(attempted=0, generated=0, complete=0, generation_failed=0,
                  validated=0, validation_failed=0, accepted=0)
    unique = set()
    with (root / 'design/summary.csv').open('w', newline='') as csv_file, \
            (root / 'design/sequences.fasta').open('w') as fasta, \
            (root / 'accept/sequences.fasta').open('w') as accepted:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for path in sorted((root / 'design').glob('*_seed*.json')):
            record = json.loads(path.read_text())
            name = record['name']
            backends = record.get('validation_backends', ['af2'])
            validations = {}
            for backend in backends:
                saved = root / 'filter' / backend / f'{name}.json'
                validations[backend] = json.loads(saved.read_text()) if saved.exists() else {}
            complete = bool(backends) and all(v.get('validation_status') == 'completed'
                                             for v in validations.values())
            failed = any(v.get('validation_status') == 'failed' for v in validations.values())
            status = 'completed' if complete else 'failed' if failed else 'not_run'
            primary = validations.get('af3', validations.get('af2', validations.get('af2_monomer', {})))
            criteria = {k: v.get('filter', {}).get('criteria', {}) for k, v in validations.items()}
            reasons = []
            for backend, val in validations.items():
                if val.get('validation_status') != 'completed':
                    reasons.append(backend + ': ' + val.get('reason', 'validation not completed'))
                elif val.get('filter', {}).get('criteria') and val['filter'].get('passed') is not True:
                    reasons.extend(backend + ': ' + reason for reason in
                                   (val['filter'].get('reasons') or ['thresholds did not pass']))
            if record.get('design_complete') is not True:
                reasons.append('flow ended before all designable positions committed')
            source_backend = 'af3' if 'af3' in backends else 'af2' if 'af2' in backends else 'af2_monomer'
            source = root / 'filter' / source_backend / (name + ('.cif' if source_backend == 'af3' else '.pdb'))
            passed = complete and any(criteria.values()) and not reasons and source.is_file()
            filter_status = ('failed' if failed else 'not_run' if not any(criteria.values())
                             or not complete else 'passed' if passed else 'rejected')
            selection = dict(status=filter_status, passed=True if passed else None if filter_status == 'not_run' else False,
                             criteria=criteria, reasons=reasons)
            counts['attempted'] += 1
            counts['generation_failed'] += record.get('generation_status') == 'failed'
            counts['complete'] += bool(record.get('design_complete'))
            counts['validated'] += complete
            counts['validation_failed'] += failed
            row = {key: record.get(key, '') for key in columns}
            row.update({key: primary.get(key, '') for key in ('evaluator', 'interface_ptm', 'binder_plddt', 'ipae_min', 'irmsd', 'loss')})
            row.update(validation_status=status, filter_status=filter_status,
                       filter_criteria=json.dumps(criteria, sort_keys=True), filter_reasons='; '.join(reasons),
                       failure_reason=record.get('reason', '; '.join(v.get('reason', '') for v in validations.values() if v.get('reason'))))
            for backend in ('af2', 'af3'):
                val = validations.get(backend, {})
                row[backend + '_status'] = val.get('validation_status', 'not_run' if backend in backends else 'disabled')
                for metric in ('interface_ptm', 'binder_plddt', 'ipae_min', 'irmsd'):
                    row[backend + '_' + metric] = val.get(metric, '')
            monomer = validations.get('af2_monomer', {})
            row['af2_monomer_status'] = monomer.get(
                'validation_status', 'not_run' if 'af2_monomer' in backends else 'disabled')
            for metric in ('binder_plddt', 'rmsd'):
                row['af2_monomer_' + metric] = monomer.get(metric, '')
            for key, relative in (('design_pdb', f'design/{name}.pdb'),
                                  ('validation_pdb', f'filter/af2/{name}.pdb'),
                                  ('af3_cif', f'filter/af3/{name}.cif'),
                                  ('af2_monomer_pdb', f'filter/af2_monomer/{name}.pdb')):
                row[key] = relative if (root / relative).exists() else ''
            if 'sequence' in record:
                counts['generated'] += 1
                unique.add(record['sequence'])
                fasta.write(f">{name}\n{record['sequence']}\n")
            for suffix in ('.pdb', '.cif', '.json'):
                (root / 'accept' / f'{name}{suffix}').unlink(missing_ok=True)
            if passed:
                counts['accepted'] += 1
                shutil.copy2(source, root / 'accept' / source.name)
                _write_json(root / 'accept' / f'{name}.json',
                            dict(name=name, sequence=record['sequence'], filter=selection,
                                 validations=validations))
                accepted.write(f">{name}\n{record['sequence']}\n")
                row['accepted_cif' if source.suffix == '.cif' else 'accepted_pdb'] = f'accept/{source.name}'
            writer.writerow(row)
    counts['unique_sequences'] = len(unique)
    _write_json(root / 'design/run_summary.json', counts)
    print('Summary: ' + json.dumps(counts), file=sys.stderr)
