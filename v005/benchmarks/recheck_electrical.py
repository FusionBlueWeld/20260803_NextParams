"""Stdlib-only relay and posthoc comparison for the five electrical trials.

This file never imports or starts the optimizer or simulator. Codex runs their
CLI commands independently between init/export/ingest/finalize commands.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
TRIALS = ROOT / 'v005/trials'
BATCH = TRIALS / '_recheck_20261010'
SOURCES = ['cut_al', 'form_cu', 'form_al', 'assembly', 'interconnect_welding']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def relay():
    spec = importlib.util.spec_from_file_location('historical_file_relay',
        TRIALS / '_batch_20261010/file_relay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_for(stage):
    return TRIALS / f'trial_{stage}_20261010'


def target_for(stage):
    return TRIALS / f'trial_recheck_{stage}_20261010'


def init(stage):
    source, target = source_for(stage), target_for(stage)
    if not target.is_dir() or (target / 'protocol.json').exists():
        raise ValueError('Create a fresh trial with normal CLI --new first; never overwrite a recheck.')
    names = ['problem.csv', 'knowledge_constraints.csv', 'search_regions.csv',
             'stop_settings.csv', 'search_settings.json', 'stage_connection.json',
             'spec.json', 'simulator_contract.json']
    hashes = {}
    for name in names:
        path = source / name
        if path.exists():
            shutil.copy2(path, target / name)
            hashes[name] = digest(path)
    # Preserve original learning/grid budgets and let new algorithm defaults apply.
    settings = read(target / 'search_settings.json')
    settings.update(cpu_threads=1, distance_memory_mb=16)
    dump(target / 'search_settings.json', settings)
    snapshot = source / 'steps/step_000/experiments.csv'
    shutil.copy2(snapshot, target / 'data/experiments.csv')
    first = target / 'steps/step_000'
    first.mkdir(parents=True)
    shutil.copy2(snapshot, first / 'experiments.csv')
    (target / 'exchange').mkdir()
    contract = read(target / 'simulator_contract.json')
    data = rows(snapshot)
    service = relay()
    best = service.best_row(contract, data)
    dump(target / 'ledger.json', [{'step': 0, 'total_measurements': len(data),
        'feasible_measurements': sum(service.good(contract, row) for row in data),
        'best_value': None if best is None else float(best[contract['objective']]),
        'best_id': None if best is None else best['experiment_id']}])
    dump(target / 'protocol.json', {'comparison_source': source.name,
        'initial_conditions': len(data), 'initial_snapshot_sha256': digest(snapshot),
        'recommendation_maximum_per_step': 5, 'max_steps': 20, 'additional_measurement_budget': 100,
        'same_problem_constraints_context_and_initial_observations': True,
        'learning_and_grid_budgets': settings, 'changed_policy': 'v005-efficient-search-2 defaults',
        'source_input_sha256': hashes, 'started_unix': time.time(),
        'isolation': 'separate CLI processes; Codex relays JSON/CSV; no oracle before exploration ends'})
    print(json.dumps({'stage': stage, 'trial': target.name, 'initial_rows': len(data)}))


def finalize(stage):
    target, source = target_for(stage), source_for(stage)
    contract = read(target / 'simulator_contract.json')
    service = relay()
    data = rows(target / 'data/experiments.csv')
    ledger = read(target / 'ledger.json')
    protocol = read(target / 'protocol.json')
    status = read(target / 'output/stopping_status.json')
    audit = read(target / 'posthoc_grid_audit.json')
    old = read(source / 'summary.json')
    best = service.best_row(contract, data)
    value = None if best is None else float(best[contract['objective']])
    optimum = audit['best_value']
    close = lambda x, y: abs(x-y) <= max(1e-10, abs(y)*1e-9)
    first = next((entry for entry in ledger if entry['best_value'] is not None
                  and optimum is not None and close(entry['best_value'], optimum)), None)
    unique = len({tuple(float(row[name]) for name in contract['parameters']) for row in data})
    checks = []
    for entry in ledger[1:]:
        step = entry['step']
        request_path = target / 'exchange' / f'request_{step:03d}.json'
        measurement_path = target / 'exchange' / f'measurement_{step:03d}.json'
        request, measured = read(request_path), read(measurement_path)
        assert measured['request_sha256'] == digest(request_path)
        assert entry['measurement_sha256'] == digest(measurement_path)
        assert len(request['conditions']) == len(measured['results'])
        for condition, actual in zip(request['conditions'], measured['results']):
            assert condition['experiment_id'] == actual['experiment_id']
            assert all(condition[name] == actual['inputs'][name] for name in contract['parameters'])
        checks.append({'step': step, 'actual_batch_count': len(measured['results']),
            'request_measurement_hash_verified': True,
            'passed': sum(actual['feasible'] for actual in measured['results'])})
    assert unique == len(data), 'Duplicate measured input conditions'
    assert digest(target / 'steps/step_000/experiments.csv') == protocol['initial_snapshot_sha256']
    assert all(digest(source / name) == value for name, value in protocol['source_input_sha256'].items())
    externally_limited = status['status']=='CONTINUE' and ledger[-1]['step']==20
    assert status['status'] != 'CONTINUE' or externally_limited, 'Exploration must end before posthoc audit'
    if not externally_limited:
        assert not (target / 'output/recommendations.csv').exists()
    stop_label='EXTERNAL_STEP_LIMIT' if externally_limited else status['status']
    stop_reasons=['比較実行の最大20ステップに到達。CLIの自動収束ではない'] if externally_limited else status['reasons']
    predictions = list((target / 'output/response_spaces').glob('*.csv'))
    timing = []
    for path in sorted((target / 'steps').glob('step_*/search_budget.json')):
        budget = read(path)
        timing.append({'step': int(path.parent.name.split('_')[1]),
            'wall_seconds': budget.get('wall_seconds'),
            'learning_seconds': budget.get('learning', {}).get('seconds'),
            'prediction_seconds': budget.get('prediction_seconds'),
            'learning_skipped': budget.get('learning_skipped', False),
            'prediction_grid_passes': budget.get('prediction_grid_passes'),
            'runtime': budget.get('runtime')})
    result = {'trial': target.name, 'source_trial': source.name, 'title': contract['spec']['title'],
        'steps': ledger[-1]['step'], 'measurements': len(data),
        'additional_measurements': len(data)-len(rows(target/'steps/step_000/experiments.csv')),
        'unique_conditions': unique, 'feasible_measurements': sum(service.good(contract, row) for row in data),
        'best_value': value, 'oracle_grid_best': optimum,
        'objective_gap': None if value is None or optimum is None else
            (value-optimum if contract['direction']=='minimize' else optimum-value),
        'first_optimum_step': None if first is None else first['step'],
        'stop_status': stop_label, 'engine_stop_status':status['status'],
        'stop_reasons': stop_reasons, 'stop_metrics': status['metrics'],
        'converged': status['status']=='STOP_RECOMMENDED',
        'optimum_retained': value is not None and optimum is not None and close(value,optimum),
        'old_steps': old['steps'], 'old_measurements': old['measurements'],
        'old_first_optimum_step': old['first_optimum_step'], 'old_best_value': old['best_value'],
        'old_stop_status': old['stop_status'], 'old_stop_reasons': old['stop_reasons'],
        'saved_measurements': old['measurements']-len(data),
        'prediction_files': len(predictions), 'process_checks': checks, 'run_timing': timing,
        'elapsed_including_relay_seconds': time.time()-protocol['started_unix'],
        'scope': 'fixed incoming context, identical initial data, noiseless discrete-grid scenario; one seed',
        'initial_source_unchanged': True, 'source_inputs_unchanged': True}
    dump(target / 'summary.json', result)
    BATCH.mkdir(exist_ok=True)
    dump(BATCH / f'{stage}.json', result)
    (target / 'REPORT.md').write_text('\n'.join([
        f"# {result['title']}：改善後の再探索", '',
        f"前回 {old['steps']}ステップ／{old['measurements']}条件 → 今回 {result['steps']}ステップ／{len(data)}条件。",
        f"停止：{stop_label}。理由：{stop_reasons}。",
        f"実測最良：{value}。探索後の独立格子最良：{optimum}。差：{result['objective_gap']}。",
        f"最適値初到達：{result['first_optimum_step']}ステップ。",
        '', '初期実測24条件・問題・制約・流入状態・学習予算は前回と同じ。',
        '推薦上限5件、追加100条件／最大20ステップ。件数は改善後の方針で可変。',
        'v005とシミュレータは別プロセスで実行し、CodexがJSON/CSVを受け渡した。',
        '全格子の真値は停止後にのみ評価し、optimizerの入力に含めていない。',
        'REVIEW_REQUIREDは自動収束・最適性の証明と区別する。',
        '', 'GUIでこのtrialを選択すると各ステップの予測・実測を表示できる。',
        '詳細な停止指標・可変件数・時間・hash確認はsummary.json。', '']), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('trial','steps','measurements','stop_status','best_value',
        'oracle_grid_best','optimum_retained','saved_measurements')}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init','export','ingest','finalize'])
    parser.add_argument('--stage', required=True, choices=SOURCES)
    parser.add_argument('--step', type=int)
    args = parser.parse_args()
    if args.command=='init': init(args.stage)
    elif args.command=='finalize': finalize(args.stage)
    else:
        if args.step is None: parser.error('--step is required')
        module = relay()
        target=target_for(args.stage)
        getattr(module,args.command)(target,read(target/'simulator_contract.json'),args.step)


if __name__=='__main__':
    main()
