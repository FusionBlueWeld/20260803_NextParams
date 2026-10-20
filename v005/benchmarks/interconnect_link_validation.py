"""Connect recorded v005 recipes; audit r002 with fixed and varying incoming states.

Physics is used only to create labelled training rows and independently score results.
The bundle2 bridge uses the existing fast_stage backend, not the CLI NN/GP model.
"""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from validation.multistage.electrical_interconnect.pipeline import (
    STAGE_ORDER, STAGE_MANIFESTS, evaluate_line, line_config, FINAL_SPECIFICATIONS)
from validation.multistage.src.checks.electrical_interconnect_validation import jsonable
from validation.multistage.src.checks.interconnect_data import generate_workpieces, train_bundles
from validation.multistage.src.checks.interconnect_metrics import prediction_errors, oracle_predictors
from v005.src.stage_bundle import train_stage_bundle, load_stage_bundle
from r002.src.api import run_search
from r002.src.pipeline import evaluate_pipeline

DEST = ROOT / 'v005/trials/_interconnect_link_20261010'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def log(message):
    print(message, flush=True)

def recipe(seed):
    result, sources = {}, {}
    for stage in STAGE_ORDER:
        trial = ROOT / f'v005/trials/trial_interconnect_stage_{stage}_s{seed}_20261010_focus_on'
        summary = json.loads((trial / 'summary.json').read_text(encoding='utf-8-sig'))
        result.update({f'{stage}.{k}': v for k, v in summary['best_inputs'].items()})
        sources[stage] = {'trial': str(trial), 'summary': summary,
                          'summary_sha256': hashlib.sha256((trial/'summary.json').read_bytes()).hexdigest()}
    return result, sources

def truth(controls):
    actual = evaluate_line(controls)
    # evaluate_line intentionally computes NG downstream for diagnosis. Enforce
    # manifest domains separately: an invalid diagnostic is not an executable line.
    violations = {}
    for stage, inputs in actual['stage_inputs'].items():
        bad = {}
        for name, value in inputs.items():
            lo, hi = STAGE_MANIFESTS[stage]['input_bounds'][name]
            if np.any((np.asarray(value) < lo) | (np.asarray(value) > hi)):
                bad[name] = {'value': value, 'bounds': [lo, hi]}
        violations[stage] = bad
    actual['domain_violations'] = violations
    actual['executable_pass'] = bool(actual['final']['line_feasible']) and not any(violations.values())
    return actual

def fixed_bundles(seed, destination):
    config, predictors = line_config(), {}
    training = {}
    for stage in STAGE_ORDER:
        trial = ROOT / f'v005/trials/trial_interconnect_stage_{stage}_s{seed}_20261010_focus_on'
        fixture = json.loads((trial/'fixture.json').read_text(encoding='utf-8-sig'))['fixed_incoming_context']
        with (trial/'data/experiments.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        m = copy.deepcopy(STAGE_MANIFESTS[stage])
        m['bundle_schema_version'] = '2.0'
        m['provenance'] = {'source_trial': str(trial), 'incoming_state_policy': 'FIXED_NOMINAL',
                           'backend_note': 'bridge_refit_fast_stage_not_original_cli_model'}
        # Several labels have the same names as incoming fields: always source
        # incoming state from fixture, never substitute outgoing CSV labels.
        x = {name: np.array([float(r[name]) for r in rows]) for name in m['controls']}
        x.update({name: np.full(len(rows), fixture[name]) for name in m['incoming_context']})
        y = {name: np.array([float(r[name]) for r in rows]) for name in m['predicted_outputs']}
        save(destination/stage/'training_rows.json', {'inputs': x, 'outputs': y})
        bundle = train_stage_bundle(destination/stage, m, x, y, seed=seed, max_training_samples=256)
        predictors[stage] = load_stage_bundle(bundle)
        for row in config['stages']:
            if row['id'] == stage:
                row['bundle'] = str(bundle)
        training[stage] = {'rows': len(rows), 'varying_incoming_fields': 0}
    return config, predictors, training

def search(config, models, controls, seed, budget=2048):
    conf = copy.deepcopy(config)
    conf['warm_start'] = [controls]
    conf['search'] = {'max_evaluations': budget, 'population_size': 64,
                      'batch_size': 64, 'max_seconds': 120, 'seed': seed, 'support_threshold': .5}
    result = run_search(conf, models, recommendation_count=4)
    scored = [{'candidate': c, 'actual': truth(c['controls'])} for c in result['candidates']]
    return conf, {'search': result, 'independent_scoring': scored}

def main():
    if (DEST/'results.json').exists():
        raise FileExistsError('Completed run exists; preserve it rather than overwrite.')
    started = time.perf_counter()
    records = {'purpose': 'recorded_v005_recipes_to_r002_state_propagation',
               'backend': 'existing_v005_fast_stage_bundle2_bridge_refit',
               'not_original_cli_model': True, 'seeds': [], 'sweeps': []}
    for seed in (42, 0, 1):
        controls, sources = recipe(seed)
        direct = truth(controls)
        log(f'seed {seed}: connected recorded recipes pass={direct["executable_pass"]}')
        conf, fixed, training = fixed_bundles(seed, DEST/f'seed_{seed}/fixed_bundles')
        prediction = evaluate_pipeline(conf, fixed, controls)
        fixed_conf, fixed_search = search(conf, fixed, controls, seed)
        save(DEST/f'seed_{seed}/fixed_pipeline.json', fixed_conf)
        # New linked training follows actual workpiece lineage and stops admitting
        # a downstream row when its parent fails; all admitted labels remain saved.
        dataset = generate_workpieces(8192, seed + 7000)
        save(DEST/f'seed_{seed}/linked_training.json', dataset)
        linked_conf, linked = train_bundles(dataset, DEST/f'seed_{seed}/linked_bundles', 256)
        linked_conf, linked_search = search(linked_conf, linked, controls, seed)
        save(DEST/f'seed_{seed}/linked_pipeline.json', linked_conf)
        holdout = generate_workpieces(4096, seed + 17000)
        save(DEST/f'seed_{seed}/holdout.json', holdout)
        row = {'seed': seed, 'v005_sources': sources, 'controls': controls,
               'direct_connection': direct, 'fixed_training': training,
               'fixed_recipe_prediction': prediction, 'fixed_r002': fixed_search,
               'linked_training_counts': {s: {'admitted': b['admitted_count'], 'passed': b['passed_count']}
                                          for s,b in dataset['blocks'].items()},
               'linked_holdout_errors': prediction_errors(holdout, linked), 'linked_r002': linked_search}
        save(DEST/f'seed_{seed}/results.json', row)
        records['seeds'].append(row)
        log(f'seed {seed}: linked r002 scored {[r["actual"]["executable_pass"] for r in linked_search["independent_scoring"]]}')
        if seed == 0:
            sweep_models, sweep_config, sweep_controls = linked, linked_conf, controls
    # Reachable perturbations of assembly cleaning. Stage outputs of the completed
    # upstream prefix are supplied as observations; only welding controls are free.
    # Physics oracle r002 is an explicit separate reference for isolating model error.
    for dose in np.linspace(0, 1, 7):
        controls = {**sweep_controls, 'assembly.bottom_cleaning_dose': float(dose),
                    'assembly.top_cleaning_dose': float(dose)}
        unchanged = truth(controls)
        conf = copy.deepcopy(sweep_config)
        conf['fixed_controls'] = {k:v for k,v in controls.items() if not k.startswith('welding.')}
        conf['completed_stages'] = list(STAGE_ORDER[:-1])
        conf['observed_outputs'] = {f'{s}.{k}': float(v) for s in STAGE_ORDER[:-1]
                                    for k,v in unchanged['stage_outputs'][s].items()}
        model_conf, recourse = search(conf, sweep_models, controls, 100+len(records['sweeps']), 1024)
        _, reference = search(conf, oracle_predictors(), controls, 100+len(records['sweeps']), 1024)
        entry = {'both_cleaning_dose': float(dose), 'unchanged_welding': unchanged,
                 'learned_recourse': recourse, 'oracle_recourse_reference': reference,
                 'upstream_pass': all(bool(unchanged['local_feasible'][s]) for s in STAGE_ORDER[:-1])
                                   and not any(unchanged['domain_violations'][s] for s in STAGE_ORDER[:-1])}
        records['sweeps'].append(entry)
        save(DEST/f'sweeps/dose_{dose:.4f}.json', entry)
        log(f'cleaning dose {dose:.3f}: final resistance={float(unchanged["final"]["cold_unclamped_resistance_uohm"]):.3f}')
    records['elapsed_seconds'] = time.perf_counter()-started
    paths = [Path(__file__), *ROOT.glob('r002/src/*.py'), *ROOT.glob('v005/src/*.py'),
             *ROOT.glob('validation/multistage/electrical_interconnect/*.py'),
             *ROOT.glob('validation/multistage/src/checks/interconnect_*.py')]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        data = path.read_bytes()
        hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        target = DEST/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    save(DEST/'source_sha256.json', hashes)
    save(DEST/'results.json', records)
    log(f'completed in {records["elapsed_seconds"]:.1f}s: {DEST}')

if __name__ == '__main__':
    main()
