"""File-only coordinator for separate optimizer and physical measurement CLIs.

No optimizer or physics import. Oracle audit is run only after exploration stops.
Fresh trial names required; existing trials are never overwritten.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name('interconnect_stage_worker.py')
SIMULATORS = ['cut_cu', 'cut_al', 'form_cu', 'form_al', 'assembly', 'welding']


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csvread(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def csvwrite(path, header, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, log):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.open('w', encoding='utf-8') as f:
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                                stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'CLI failed ({result.returncode}): {log}')


def feasible(row, constraints):
    return all(float(row[v['column']]) <= float(v['target']) if v['direction'] == 'less_equal'
               else float(row[v['column']]) >= float(v['target']) for v in constraints)


def run(simulator, seed, args, batch):
    name = f'trial_interconnect_stage_{simulator}_s{seed}_{args.tag}'
    trial = ROOT / 'v005/trials' / name
    reuse_empty = args.resume and trial.exists() and not (trial/'protocol.json').exists() and not (trial/'data/experiments.csv').exists()
    if trial.exists() and not reuse_empty:
        raise ValueError(f'Refusing existing trial: {trial}')
    started = time.time()
    if not reuse_empty:
        for attempt in range(3):
            try:
                command(['main.py', '--version', 'v005', '--new', name], batch / f'{name}_new_{attempt}.log')
                break
            except RuntimeError:
                if trial.exists() or attempt == 2:
                    raise
    source = batch / 'definitions' / simulator
    if not source.exists():
        source.parent.mkdir(exist_ok=True)
        command([WORKER, 'define', '--simulator', simulator, '--output', source], batch / f'{simulator}_define.log')
    shutil.copy2(source/'fixture.json', trial/'fixture.json')
    shutil.copy2(source / 'problem.csv', trial / 'problem.csv')
    _, definition = csvread(trial / 'problem.csv')
    params = [v for v in definition if v['role'] == 'parameter']
    constraints = [v for v in definition if v['role'] == 'constraint']
    objective = next(v for v in definition if v['role'] == 'objective')
    axes = {}
    for v in params:
        lo, hi, step = map(float, [v['lower'], v['upper'], v['step']])
        axis = [lo + step * i for i in range(int(math.floor((hi-lo)/step+1e-12))+1)]
        if hi-axis[-1] > max(1e-10, abs(hi)*1e-12):
            axis.append(hi)
        else:
            axis[-1] = hi
        axes[v['column']] = axis
    candidate_count = math.prod(map(len, axes.values()))
    settings = read(trial / 'search_settings.json')
    settings.update(seed=seed, candidate_mode='grid', max_candidates=max(2, candidate_count),
                    max_learning_seconds=120.0)
    dump(trial / 'search_settings.json', settings)
    focus = read(trial / 'focus_settings.json')
    focus.update(initial_design_enabled=True, parameter_focus_enabled=True,
                 temporary_fix_enabled=True, max_total_experiments=args.budget)
    dump(trial / 'focus_settings.json', focus)
    (trial / 'exchange').mkdir()
    (trial / 'steps').mkdir()
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'v005/src').rglob('*.py'))}
    protocol = {'simulator': simulator, 'seed': seed, 'budget_including_initial': args.budget,
                'batch_size': 1, 'initial_design': 'v005 automatic from empty experiments.csv',
                'focus_settings': focus, 'search_settings': settings,
                'oracle_policy': 'posthoc only after terminal state; never passed to optimizer',
                'arrival_definition': 'first measured feasible condition equal to constrained grid optimum',
                'noise': 0, 'fixed_context_fixture': read(trial/'fixture.json'), 'optimizer_source_sha256': hashes,
                'python': sys.version, 'python_executable': sys.executable,
                'worker_sha256': digest(WORKER), 'coordinator_sha256': digest(Path(__file__)),
                'started_unix': started}
    dump(trial / 'protocol.json', protocol)
    command(['main.py', '--version', 'v005', '--prepare', name], trial/'prepare.log')
    header, rows = csvread(trial/'data/experiments.csv')
    ledger = []
    seen = set()
    status = {'status': 'NOT_RUN', 'reasons': []}
    error = None
    try:
        for step in range(args.budget+1):
            snapshot = trial / 'steps' / f'step_{step:03d}'
            snapshot.mkdir()
            shutil.copy2(trial/'data/experiments.csv', snapshot/'experiments.csv')
            command(['main.py', '--version', 'v005', '--run', name, '--n', '1'], snapshot/'v005_run.log')
            for n in ['stopping_status.json', 'search_budget.json', 'recommendations.csv',
                      'parameter_focus.json', 'initial_design_status.json']:
                if (trial/'output'/n).exists():
                    shutil.copy2(trial/'output'/n, snapshot/n)
            status = read(trial/'output/stopping_status.json')
            if status['status'] != 'CONTINUE':
                break
            _, recs = csvread(trial/'output/recommendations.csv')
            assert len(recs) == 1
            row = recs[0]
            condition = {'experiment_id': f'measurement_{step+1:04d}'}
            for n, axis in axes.items():
                value = float(row[n])
                exact = min(axis, key=lambda x: abs(x-value))
                assert math.isclose(value, exact, rel_tol=1e-12, abs_tol=1e-12)
                condition[n] = exact
            key = tuple(condition[n] for n in axes)
            assert key not in seen, 'duplicate recommendation'
            request = trial/'exchange'/f'request_{step+1:03d}.json'
            measured_path = trial/'exchange'/f'measurement_{step+1:03d}.json'
            dump(request, {'conditions': [condition], 'selection_role': row.get('selection_role'),
                           'source_recommendation_sha256': digest(snapshot/'recommendations.csv')})
            command([WORKER, 'measure', '--simulator', simulator, '--request', request,
                     '--output', measured_path], snapshot/'physics.log')
            measurement = read(measured_path)
            assert measurement['request_sha256'] == digest(request)
            observed = measurement['measurements'][0]
            assert all(observed[n] == value for n, value in condition.items())
            rows.append(observed)
            seen.add(key)
            csvwrite(trial/'data/experiments.csv', header, rows)
            valid = [r for r in rows if feasible(r, constraints)]
            values = [float(r[objective['column']]) for r in valid]
            best = ((min if objective['direction']=='minimize' else max)(values) if values else None)
            ledger.append({'step': step+1, 'total_measurements': len(rows),
                           'feasible_measurements': len(valid), 'best_value': best,
                           'selection_role': row.get('selection_role'),
                           'measurement_sha256': digest(measured_path),
                           'request_sha256': digest(request)})
            dump(trial/'ledger.json', ledger)
            if len(rows)%10 == 0:
                print(json.dumps({'trial': name, 'measured': len(rows), 'best': best}), flush=True)
    except Exception as exc:
        error = str(exc)
        print(json.dumps({'trial': name, 'error': error}), flush=True)
    command([WORKER, 'audit', '--simulator', simulator, '--output', trial/'posthoc_grid_audit.json'], trial/'audit.log')
    audit = read(trial/'posthoc_grid_audit.json')
    arrived = [i for i, r in enumerate(rows, 1) if audit['best_value'] is not None and feasible(r, constraints) and
               math.isclose(float(r[objective['column']]), audit['best_value'], rel_tol=1e-9, abs_tol=1e-10)]
    valid = [r for r in rows if feasible(r, constraints)]
    bestrow = ((min if objective['direction']=='minimize' else max)(valid,
               key=lambda r: float(r[objective['column']])) if valid else None)
    best = float(bestrow[objective['column']]) if bestrow else None
    gap = None if best is None or audit['best_value'] is None else (best-audit['best_value'] if objective['direction']=='minimize' else audit['best_value']-best)
    for entry in ledger:
        val = entry['best_value']
        entry['objective_gap'] = None if val is None or audit['best_value'] is None else (val-audit['best_value'] if objective['direction']=='minimize' else audit['best_value']-val)
    csvwrite(trial/'progress.csv', ['step','total_measurements','feasible_measurements','best_value',
                                  'selection_role','measurement_sha256','request_sha256','objective_gap'], ledger)
    initial = sum(e['selection_role']=='INITIAL_DESIGN' for e in ledger)
    summary = {'trial': name, 'simulator': simulator, 'seed': seed, 'measurements': len(rows),
               'initial_design_measurements': initial, 'first_optimum_measurement': min(arrived) if arrived else None,
               'optimum_reached': bool(arrived), 'best_value': best, 'grid_optimum': audit['best_value'],
               'objective_gap': gap, 'normalized_regret': None if gap is None else gap/audit['objective_range'],
               'best_inputs': None if bestrow is None else {n: bestrow[n] for n in axes},
               'stop_status': 'CLI_ERROR' if error else status['status'], 'stop_reasons': status['reasons'],
               'error': error, 'candidate_count': candidate_count, 'unique_conditions': len(seen),
               'focus_flags_on': all(focus[n] for n in ['initial_design_enabled','parameter_focus_enabled','temporary_fix_enabled']),
               'wall_seconds': time.time()-started}
    assert len(seen) == len(rows)
    assert all(digest(ROOT/n) == h for n,h in hashes.items()), 'Optimizer sources changed during run'
    dump(trial/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--simulators', nargs='+', choices=SIMULATORS, default=SIMULATORS)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--budget', type=int, default=150)
    p.add_argument('--tag', required=True)
    p.add_argument('--resume', action='store_true', help='Skip completed trials; reuse only empty CLI templates')
    a = p.parse_args()
    batch = ROOT/'v005/trials'/f'_interconnect_stage_{a.tag}'
    batch.mkdir(exist_ok=a.resume)
    coordinator_copy = batch/('resumed_' + Path(__file__).name if a.resume else Path(__file__).name)
    shutil.copy2(Path(__file__), coordinator_copy)
    if not (batch/WORKER.name).exists():
        shutil.copy2(WORKER, batch/WORKER.name)
    summaries = []
    for seed in a.seeds:
        for sim in a.simulators:
            finished = ROOT/'v005/trials'/f'trial_interconnect_stage_{sim}_s{seed}_{a.tag}'/'summary.json'
            if a.resume and finished.exists():
                summaries.append(read(finished))
            else:
                summaries.append(run(sim, seed, a, batch))
            dump(batch/'summary.json', summaries)
    csvwrite(batch/'comparison.csv', list(summaries[0]), summaries)
    text = ['# v005 電気接続・独立単一工程の実試行到達検証', '',
            '各工程の流入状態はfixture.jsonの代表値に固定。工程間接続・全体検証は実施しない。',
            '完全情報の合成oracleによる局所制約評価であり、現場の観測可能性や最終電気品質は評価しない。',
            '初期データ0件からv005が推薦。1推薦→別プロセスの物理計算→CSV追記→再学習。',
            '初期設計・パラメータ集中・一時固定は全てON。許容作用量は既定の空設定を維持。',
            '最適値は探索終了後の全格子監査で照合。ノイズなし。到達回数は初期条件も含む。',
            '候補は元の全格子（最大65,536点）、学習上限256点・2,000 epoch・120秒、CPU 1スレッド。',
            '未到達は記録した停止状態または150条件の予算までの結果であり、到達回数は不明。', '',
            '|工程|seed|最適値初到達（条件数）|初期設計|総条件数|最良値|格子最適値|停止|',
            '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in summaries:
        text.append(f"|{r['simulator']}|{r['seed']}|{r['first_optimum_measurement'] or '未到達'}|{r['initial_design_measurements']}|{r['measurements']}|{r['best_value']}|{r['grid_optimum']}|{r['stop_status']}|")
    text.extend(['', '各trial: protocol.json / data/experiments.csv / exchange/ / steps/ / ledger.json / progress.csv / posthoc_grid_audit.json / summary.json。',
                 '入力・測定のSHA256照合、重複なし、optimizerソース不変を確認。',
                 'ONは機能の有効化を示す。effect_tolerances未指定のため、自動固定を強制する設定は追加していない。',
                 'この結果は固定seedの決定論的物理モデルと離散格子についての検証。連続空間の最適性や実機試行回数は評価していない。'])
    (batch/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
