"""合成問題のCLI検証用ファイル接続。学習器・測定器を呼び出さない。"""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / 'results' / 'adaptive_smoke_20261010'
TRIAL = ROOT / 'trial_adaptive_smoke'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def init():
    if TRIAL.exists():
        raise ValueError('既存の検証結果は上書きしません')
    for folder in ['data', 'output', 'exchange', 'steps', 'logs']:
        (TRIAL / folder).mkdir(parents=True, exist_ok=True)
    (TRIAL / 'problem.csv').write_text(
        'column,display_name,unit,role,direction,lower,upper,step,target\n'
        'x,x,1,parameter,,-1,1,0.6666666666666666,\n'
        'y,y,1,parameter,,-1,1,0.6666666666666666,\n'
        'z,z,1,parameter,,-1,1,0.6666666666666666,\n'
        'cost,cost,s,objective,minimize,,,,\n'
        'quality,quality,1,constraint,less_equal,,,,1\n', encoding='utf-8')
    write_csv(TRIAL / 'data/experiments.csv', ['experiment_id', 'x', 'y', 'z', 'cost', 'quality'], [])
    write_json(TRIAL / 'stage_connection.json', {
        'schema_version': '1.0', 'stage_id': 'synthetic_adaptive_smoke',
        'controls': ['x', 'y', 'z'], 'incoming_context': [],
        'connector_outputs': ['cost', 'quality'], 'current_incoming_context': {}})
    write_json(TRIAL / 'focus_settings.json', {
        'initial_design_enabled': True, 'parameter_focus_enabled': True,
        'temporary_fix_enabled': True, 'max_total_experiments': 40,
        'initial_max_conditions': 20, 'stability_updates': 2,
        'effect_tolerances': {'cost': .5, 'quality': .05}, 'never_fix': ['x', 'y']})
    print(str(TRIAL))


def export(step):
    _, rows = read_csv(TRIAL / 'output/recommendations.csv')
    conditions = [{'experiment_id': f's{step:02d}_{i:02d}',
        **{name: float(row[name]) for name in ['x', 'y', 'z']}} for i, row in enumerate(rows)]
    write_json(TRIAL / 'exchange' / f'request_{step:02d}.json', {'conditions': conditions})
    print(json.dumps({'exported': len(conditions), 'step': step}))


def ingest(step):
    request = json.loads((TRIAL / 'exchange' / f'request_{step:02d}.json').read_text(encoding='utf-8'))
    result = json.loads((TRIAL / 'exchange' / f'measured_{step:02d}.json').read_text(encoding='utf-8'))
    header, rows = read_csv(TRIAL / 'data/experiments.csv')
    existing = {tuple(float(row[name]) for name in ['x', 'y', 'z']) for row in rows}
    assert len(request['conditions']) == len(result['measurements'])
    for condition, measurement in zip(request['conditions'], result['measurements']):
        assert all(condition[name] == measurement[name] for name in condition)
        key = tuple(measurement[name] for name in ['x', 'y', 'z'])
        assert key not in existing
        existing.add(key)
        rows.append(measurement)
    write_csv(TRIAL / 'data/experiments.csv', header, rows)
    snapshot = TRIAL / 'steps' / f'step_{step:03d}'
    snapshot.mkdir(exist_ok=True)
    write_csv(snapshot / 'experiments.csv', header, rows)
    ledger_path = TRIAL / 'ledger.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else []
    ledger.append({'step': step, 'total_measurements': len(rows)})
    write_json(ledger_path, ledger)
    print(json.dumps({'actual_conditions': len(rows), 'step': step}))


def report():
    _, rows = read_csv(TRIAL / 'data/experiments.csv')
    snapshots = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((TRIAL / 'output/parameter_focus').glob('*.json'))]
    states = [{'rows': state['data_rows'], 'phase': state['phase'], 'fixed_values': state['fixed_values'],
        'scope': state['candidate_space_scope']} for state in snapshots]
    assert all('x' not in state['fixed_values'] and 'y' not in state['fixed_values'] for state in snapshots)
    assert len({(row['x'], row['y'], row['z']) for row in rows}) == len(rows)
    status = json.loads((TRIAL / 'output/stopping_status.json').read_text(encoding='utf-8'))
    summary = {'scope': 'synthetic CLI smoke only, not a manufacturing accuracy comparison',
        'measurements': len(rows), 'best': min(float(row['cost']) for row in rows),
        'states': states, 'status': status['status'], 'metrics': status['metrics']}
    write_json(ROOT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['init', 'export', 'ingest', 'report'])
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()
    if args.action in ('export', 'ingest'):
        globals()[args.action](args.step)
    else:
        globals()[args.action]()
