"""Verify saved implementation evidence without training or importing physics."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'v005/benchmarks/results'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    trial = ROOT / 'v005/trials/trial_efficiency_press_20261010'
    contract = read(trial / 'simulator_contract.json')
    data = rows(trial / 'data/experiments.csv')
    assert len(data) == 34
    keys = [tuple(float(row[name]) for name in contract['parameters']) for row in data]
    assert len(set(keys)) == 34
    source = ROOT / 'v005/trials/trial_single_press_forming_20261010/steps/step_000/experiments.csv'
    assert (trial / 'steps/step_000/experiments.csv').read_bytes() == source.read_bytes()
    checks = []
    for step in (1, 2):
        request_path = trial / 'exchange' / f'request_{step:03d}.json'
        request = read(request_path)
        measured = read(trial / 'exchange' / f'measurement_{step:03d}.json')
        assert measured['request_sha256'] == hashlib.sha256(request_path.read_bytes()).hexdigest()
        assert len(request['conditions']) == len(measured['results']) == 5
        folder = trial / 'steps' / f'step_{step:03d}'
        predictions = rows(folder / 'recommendations.csv')
        budget = read(folder / 'search_budget.json')
        assert len(predictions) == 5 and budget['prediction_grid_passes'] == 1
        assert all(pool['threads'] == 1 for pool in budget['runtime']['libraries'])
        for prediction, asked, actual, row in zip(
            predictions, request['conditions'], measured['results'], data[24 + 5*(step-1):24 + 5*step]
        ):
            assert asked['experiment_id'] == actual['experiment_id'] == row['experiment_id']
            assert int(prediction['actual_recommendations']) == 5
            for name in contract['parameters']:
                assert float(prediction[name]) == asked[name] == actual['inputs'][name] == float(row[name])
            for name in contract['outputs']:
                assert actual['outputs'][name] == float(row[name])
        assert rows(folder / 'experiments.csv') == data[:24 + 5*step]
        checks.append({'step': step, 'measurements': 5,
            'passed': sum(row['feasible'] for row in measured['results']),
            'prediction_grid_passes': 1, 'effective_cpu_threads': 1,
            'request_measurement_csv_match': True})

    for entry in read(trial / 'ledger.json'):
        subset = data[:entry['total_measurements']]
        feasible = [row for row in subset if all(
            float(row[condition['name']]) <= condition['target']
            if condition['direction'] == 'less_equal'
            else float(row[condition['name']]) >= condition['target']
            for condition in contract['constraints'])]
        assert entry['feasible_measurements'] == len(feasible)
    status = read(trial / 'output/stopping_status.json')
    assert status['status'] == 'STOP_REQUIRED'
    assert not (trial / 'output/recommendations.csv').exists()
    assert read(trial / 'output/search_budget.json')['learning_skipped']

    replay = read(RESULTS / 'efficiency_final_replay/historical_replay.json')['cases']
    assert len(replay) == 7
    assert sum(case['previous_actual_failed_retained'] for case in replay
               if case['trial'] == 'trial_assembly_20261010') == 0
    laser = [case for case in replay if case['trial'] == 'trial_single_laser_welding_20261010']
    assert laser[0]['previous_actual_passed_retained'] == 1
    assert laser[0]['previous_actual_failed_retained'] == 0 and laser[1]['new_count'] == 1
    benchmark = read(RESULTS / 'efficiency_20261010/summary.json')['trials']
    assert len(benchmark) == 9 and sum(case['observations'] for case in benchmark) == 288
    reaches = {method: sum(case['first_optimum_observation'] is not None for case in benchmark
               if case['method'] == method) for method in ('hybrid', 'nn', 'random')}
    assert reaches == {'hybrid': 2, 'nn': 1, 'random': 0}
    # Native Windows output may contain non-UTF8 Japanese bytes; test status is ASCII.
    log = (RESULTS / 'implementation_tests.log').read_bytes().decode('ascii', errors='replace')
    assert re.search(r'Ran 61 tests', log) and re.search(r'^OK\r?$', log, re.MULTILINE)
    resource = read(RESULTS / 'resource_validation.json')
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value
               for path, value in resource['source_sha256'].items())
    review = {'review': 'Sol independent saved-data checks after Luna execution',
        'unit_tests': 61, 'unit_tests_passed': True, 'functional_trial': trial.name,
        'total_measured_conditions': 34, 'duplicate_conditions': 0, 'batches': checks,
        'stop_status': status['status'], 'actionable_recommendations_after_stop': False,
        'initial_snapshot_source_match': True, 'historical_replay_cases': 7,
        'random_nn_hybrid_optimum_reaches': reaches, 'comparison_total_conditions': 288,
        'resource_source_hashes_match_final': True,
        'limitations': '3 seeds in one noiseless low-dimensional problem; no generalization or earlier-stop guarantee',
        'source_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ROOT / 'v005/src').rglob('*.py')}}
    (RESULTS / 'implementation_validation.json').write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in review.items() if key != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
