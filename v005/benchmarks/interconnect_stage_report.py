"""Independent replay and posthoc report for six isolated electrical stages."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'v005'))
import numpy as np
from interconnect_stage_worker import StageSimulator
from src.hybrid.gp_component import constraint_probability

NAMES = {'cut_cu':'Cu切断','cut_al':'Al切断','form_cu':'Cu成形','form_al':'Al成形','assembly':'組立','welding':'溶接'}


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def csvread(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def dump(path, data): path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def verify(folder):
    summary = read(folder/'summary.json'); protocol = read(folder/'protocol.json')
    assert summary['error'] is None, summary
    sim = StageSimulator(summary['simulator'])
    audit = read(folder/'posthoc_grid_audit.json')
    assert audit['simulator_provenance'] == sim.provenance() == read(folder/'fixture.json')
    rows = csvread(folder/'data/experiments.csv')
    points = np.asarray([[float(r[n]) for n in sim.parameter_columns] for r in rows])
    assert len(np.unique(points,axis=0)) == len(rows) == summary['measurements']
    values = sim.evaluate_points(points)
    for n in sim.output_columns:
        assert np.allclose(values[n], [float(r[n]) for r in rows], rtol=1e-12, atol=1e-12), n
    for i, entry in enumerate(read(folder/'ledger.json'),1):
        request = folder/'exchange'/f'request_{i:03d}.json'
        response = folder/'exchange'/f'measurement_{i:03d}.json'
        assert digest(request)==entry['request_sha256']
        assert digest(response)==entry['measurement_sha256']
        measurement = read(response); requested = read(request)
        assert measurement['request_sha256']==digest(request)
        assert measurement['simulator_provenance']==sim.provenance()
        assert requested['source_recommendation_sha256']==digest(folder/'steps'/f'step_{i-1:03d}'/'recommendations.csv')
        assert all(str(measurement['measurements'][0][n])==rows[i-1][n] for n in rows[i-1])
        assert all(measurement['measurements'][0][n]==v for n,v in requested['conditions'][0].items())
    for n,h in protocol['optimizer_source_sha256'].items(): assert digest(ROOT/n)==h
    for n,h in sim.provenance()['sha256'].items(): assert digest(ROOT/n)==h
    archived=read(folder/'response_archive_manifest.json') if (folder/'response_archive_manifest.json').exists() else {}
    for item in archived.values():
        path=folder/'output/response_spaces'/item['archive']
        assert digest(path)==item['archive_sha256']
    feasible = sim.feasible(values)
    initial_count = summary['initial_design_measurements']
    initial_values = values[sim.objective.column][:initial_count][feasible[:initial_count]]
    initial_best = float(initial_values.min()) if len(initial_values) else None
    arrived = feasible & np.isclose(values[sim.objective.column],audit['best_value'],rtol=1e-9,atol=1e-10) if audit['best_value'] is not None else np.zeros(len(rows),bool)
    arrival = int(np.flatnonzero(arrived)[0]+1) if arrived.any() else None
    assert arrival == summary['first_optimum_measurement']
    flags = read(folder/'focus_settings.json')
    assert flags==protocol['focus_settings']
    assert read(folder/'search_settings.json')==protocol['search_settings']
    assert all(flags[n] for n in ['initial_design_enabled','parameter_focus_enabled','temporary_fix_enabled'])
    focus = [read(p) for p in (folder/'output/parameter_focus').glob('*.json')]
    summary.update(independent_replay_verified=True, provenance_and_exchange_hashes_verified=True,
                   initial_design_best_value=initial_best,
                   improvement_after_initial_design=None if initial_best is None or summary['best_value'] is None else initial_best-summary['best_value'],
                   optimum_reached_in_initial_design=arrival is not None and arrival<=initial_count,
                   feasible_measurements=int(feasible.sum()), grid_feasible_count=audit['feasible_count'],
                   equally_optimal_conditions=audit['optimal_count'],
                   runs_with_temporary_fixed_inputs=sum(bool(s.get('fixed_values')) for s in focus),
                   runs_with_reweighted_inputs=sum(any(abs(float(p.get('weight',1))-1)>1e-12 for p in s.get('parameters',[])) for s in focus),
                   recorded_focus_scopes=sorted({s.get('candidate_space_scope','unknown') for s in focus}),
                   model_runs=len(list((folder/'output/response_spaces').glob('*.csv*'))))
    summary['verified_lossless_archives']=len(archived)
    total,optimal_count=audit['candidate_count'],audit['optimal_count']
    budget=protocol['budget_including_initial']
    miss_probability=math.prod((total-optimal_count-i)/(total-i) for i in range(min(budget,total))) if optimal_count and budget<=total-optimal_count else 0.0
    summary['uniform_random_optimum_hit_probability_at_budget']=1-miss_probability if optimal_count else 0.0
    # Audit unmeasured saved candidates; no values from this audit return to the optimizer.
    paths = sorted((folder/'output/response_spaces').glob('hybrid_response_space_run_*.csv*'))
    if paths:
        prediction = csvread(paths[-1])
        grid = np.asarray([[float(r[n]) for n in sim.parameter_columns] for r in prediction])
        truth_values = sim.evaluate_points(grid)
        truth = sim.feasible(truth_values)
        probability = np.ones(len(grid))
        for c in sim.manifest['local_constraints']:
            n = c['name']
            probability *= constraint_probability(np.asarray([float(r[n+'_hybrid_mean']) for r in prediction]),
                np.asarray([float(r[n+'_hybrid_std']) for r in prediction]),c['direction'],c['target'])
        # Match recommendation inputs at saved CSV precision; exclude every actually measured condition.
        measured = {tuple(round(float(r[n]),10) for n in sim.parameter_columns) for r in rows}
        holdout = np.asarray([tuple(round(float(v),10) for v in p) not in measured for p in grid])
        accepted = holdout & (probability>=.95)
        actual_good = holdout & truth
        grid_optimal = truth & np.isclose(truth_values[sim.objective.column],audit['best_value'],rtol=1e-9,atol=1e-10) if audit['best_value'] is not None else np.zeros(len(grid),bool)
        if grid_optimal.any():
            predicted_objective=np.asarray([float(r[sim.objective.column+'_hybrid_mean']) for r in prediction])
            summary['optimal_candidate_prediction_audit']={
                'true_optimal_points':int(grid_optimal.sum()),
                'true_optimum':audit['best_value'],
                'predicted_objective_median_at_true_optimum':float(np.median(predicted_objective[grid_optimal])),
                'predicted_feasibility_min_at_true_optimum':float(probability[grid_optimal].min()),
                'predicted_feasibility_median_at_true_optimum':float(np.median(probability[grid_optimal])),
                'predicted_feasibility_max_at_true_optimum':float(probability[grid_optimal].max()),
                'true_optimal_points_predicted_95_percent':int((grid_optimal & (probability>=.95)).sum()),
                'interpretation':'posthoc diagnosis only; never supplied to optimizer'}
        summary['feasibility_map_audit'] = {
            'prediction_file':str(paths[-1].relative_to(folder)), 'training_rows':int(prediction[0]['data_rows']),
            'threshold':.95,'heldout_grid_points':int(holdout.sum()), 'predicted_accepted_points':int(accepted.sum()),
            'false_accepted_points':int((accepted & ~truth).sum()),
            'actual_feasibility_among_accepted':float(truth[accepted].mean()) if accepted.any() else None,
            'actual_feasible_points':int(actual_good.sum()),
            'recall_of_actual_feasible_points':float(accepted[actual_good].mean()) if actual_good.any() else None,
            'brier_score':float(np.mean((probability[holdout]-truth[holdout])**2)) if holdout.any() else None,
            'interpretation':'deterministic withheld grid audit; not calibrated manufacturing yield; product assumes independence'}
    return summary


def main():
    tag = sys.argv[1] if len(sys.argv)>1 else '20261010_focus_on'
    trials = ROOT/'v005/trials'; out = trials/f'_interconnect_stage_{tag}'
    folders = sorted(p for p in trials.glob(f'trial_interconnect_stage_*_{tag}') if (p/'summary.json').exists())
    assert len(folders)==18, f'Expected 18 finished trials; found {len(folders)}'
    results = [verify(p) for p in folders]
    dump(out/'all_results.json',results)
    with (out/'all_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        header=[n for n in results[0] if n not in ('feasibility_map_audit','optimal_candidate_prediction_audit')]
        w=csv.DictWriter(f,fieldnames=header,extrasaction='ignore');w.writeheader();w.writerows(results)
    text=['# v005：電気接続を構成する単一工程の検証','',
          '6工程×seed 42・0・1。通常v005 CLIから1条件ずつ推薦し、別プロセスの物理CLIで実測相当値を取得した。',
          '初期データ0件、初期設計・入力限定・一時固定ON、最大150条件。初期実験も条件数に含む。',
          '条件数はv005が新規推薦した条件の測定相当評価。後処理の全格子監査・独立再計算は別計算で、条件数に含めない。',
          '流入状態はfixture.jsonの固定代表値。組立・溶接はCu下材／Al上材。上流や工程全体を実行していない。',
          '局所目的は加工時間最小化、制約は既存STAGE_MANIFESTSのlocal_constraintsをそのまま使用。',
          '完全情報の未校正合成oracleであり、潜在状態も局所制約のラベルに使用する。現場の部分観測性能ではない。',
          '最終電気抵抗や強度の完成品制約は、今回の最適化や合否判定に追加していない。','',
          '## 格子最適値への初到達','',
          '|工程|操作数|seed 42|seed 0|seed 1|到達数|','|---|---:|---:|---:|---:|---:|']
    for stage,title in NAMES.items():
        group={r['seed']:r for r in results if r['simulator']==stage}
        cells=[str(group[s]['first_optimum_measurement']) if group[s]['optimum_reached'] else f"未到達（{group[s]['measurements']}条件）" for s in [42,0,1]]
        text.append(f"|{title}|{len(StageSimulator(stage).parameter_columns)}|{'|'.join(cells)}|{sum(r['optimum_reached'] for r in group.values())}/3|")
    text += ['', '## 個別試行','', '|工程|seed|初期設計|総条件|最良時間(s)|格子最適時間(s)|差(s)|停止|暫定固定run数|',
             '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for stage,title in NAMES.items():
        for r in sorted((r for r in results if r['simulator']==stage),key=lambda r:[42,0,1].index(r['seed'])):
            fmt=lambda x: 'なし' if x is None else f'{x:.10g}'
            text.append(f"|{title}|{r['seed']}|{r['initial_design_measurements']}|{r['measurements']}|{fmt(r['best_value'])}|{fmt(r['grid_optimum'])}|{fmt(r['objective_gap'])}|{r['stop_status']}|{r['runs_with_temporary_fixed_inputs']}|")
    text += ['', '最適性は各工程の操作4水準、7軸16,384点／8軸65,536点の全格子についてのみ判定。',
             '全格子監査は各探索終了後に実施し、値・最適条件をモデルや停止基準へ渡していない。',
             '最適値照合はrtol=1e-9、atol=1e-10。同率最適も到達。初期設計で到達した場合は学習探索による到達と区別する。',
             '未到達は停止時点／150条件の予算での結果であり、必要条件数を推定で補っていない。',
             'STOP_RECOMMENDEDは自動収束、REVIEW_REQUIREDは要確認、STOP_REQUIREDは予算等による停止。',
             'effect_tolerancesは前回同様に既定の空設定。一時固定0の場合、入力限定の効果を実証したとは扱わない。',
             f"一時固定run数の合計：{sum(r['runs_with_temporary_fixed_inputs'] for r in results)}。入力weight変更run数の合計：{sum(r['runs_with_reweighted_inputs'] for r in results)}。",
             '通常NN＋残差GPの効果を観測した試験であり、v004や限定OFFとの因果的比較試験ではない。','',
             '## 初期設計後の改善','',
             '|工程|seed|初期設計の最良時間(s)|最終最良時間(s)|改善量(s)|初期設計中に最適到達|',
             '|---|---:|---:|---:|---:|---|']
    for r in results:
        fmt=lambda x: 'なし' if x is None else f'{x:.10g}'
        text.append(f"|{NAMES[r['simulator']]}|{r['seed']}|{fmt(r['initial_design_best_value'])}|{fmt(r['best_value'])}|{fmt(r['improvement_after_initial_design'])}|{'はい' if r['optimum_reached_in_initial_design'] else 'いいえ'}|")
    text += ['', '## 一様ランダム選択の理論参考値','',
             '全格子から150条件を重複なしで一様抽出した場合の、少なくとも1つの制約付き同率最適条件に当たる確率。',
             '監査後の格子件数N・最適条件数Kを用い、1−∏(N−K−i)/(N−i)で算出する。',
             '追加の学習試験や限定ON/OFFの比較ではなく、最適条件の希少性を示す参考値。','',
             '|工程|格子点|実制約充足点|同率最適点|ランダム150条件の到達確率|','|---|---:|---:|---:|---:|']
    for stage,title in NAMES.items():
        r=next(r for r in results if r['simulator']==stage)
        text.append(f"|{title}|{r['candidate_count']}|{r['grid_feasible_count']}|{r['equally_optimal_conditions']}|{r['uniform_random_optimum_hit_probability_at_budget']:.2%}|")
    text += ['',
             '## 制約充足確率マップの独立監査','',
             '最終保存予測のうち、全実測条件を除いた格子点で検査。確率は制約ごとの正規近似の積。',
             '95%以上として表示した格子点が実際に局所制約を満たす比率を確認。製造良品率の校正保証ではない。','',
             '|工程|seed|未測定格子点|95%以上の点|実制約充足率|誤合格点|実合格条件を拾えた割合|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        a=r.get('feasibility_map_audit',{});rate=a.get('actual_feasibility_among_accepted')
        recall=a.get('recall_of_actual_feasible_points')
        text.append(f"|{NAMES[r['simulator']]}|{r['seed']}|{a.get('heldout_grid_points')}|{a.get('predicted_accepted_points')}|{'なし' if rate is None else format(rate,'.2%')}|{a.get('false_accepted_points')}|{'なし' if recall is None else format(recall,'.2%')}|")
    all_good=[NAMES[s] for s in NAMES if any(r['simulator']==s and r['grid_feasible_count']==r['candidate_count'] for r in results)]
    if all_good:
        text += ['', '・'.join(all_good)+'は、この固定代表流入状態では全格子が局所制約を満たす。時間最小化の検証であり、制約境界の識別性能は検証できていない。']
    failed=[r for r in results if not r['optimum_reached'] and 'optimal_candidate_prediction_audit' in r]
    if failed:
        text += ['', '## 未到達試行の最適候補に対する予測','',
                 '真の最適点を探索終了後に保存予測と照合。予測目的値・制約充足確率のどちらが探索を難しくしたかを検討するための記録。',
                 '|工程|seed|真の最適時間(s)|最適点での予測時間中央値(s)|最適点での制約充足確率中央値|同最大値|',
                 '|---|---:|---:|---:|---:|---:|']
        for r in failed:
            a=r['optimal_candidate_prediction_audit']
            text.append(f"|{NAMES[r['simulator']]}|{r['seed']}|{a['true_optimum']:.10g}|{a['predicted_objective_median_at_true_optimum']:.10g}|{a['predicted_feasibility_median_at_true_optimum']:.2%}|{a['predicted_feasibility_max_at_true_optimum']:.2%}|")
    text += ['', '## 再検証と保存物','',
             f"全{len(results)}試行、合計{sum(r['measurements'] for r in results)}条件の物理再計算一致、重複なし、要求・測定・推薦のhashを検証。",
             'optimizerおよび物理sourceの不変を検証。各trialのdata/、exchange/、steps/、ledger.json、progress.csvを保存。',
             'protocol.jsonは環境・設定・hash、fixture.jsonは固定流入状態、posthoc_grid_audit.jsonは真の格子最適値。',
             'all_results.jsonには制約充足確率の監査とfocus状態の要約を保存。source_snapshotには再現用ソースを保存。',
             '保存予測はv005/plot_prediction_gui.pyで時点別の予測面と制約充足確率マップを閲覧できる。']
    text += ['一部の完了trialの古い予測CSVは、内容をSHA256照合してgzipで可逆圧縮。未圧縮の履歴もそのまま保存。圧縮履歴もGUIで閲覧可能。',
             '各trialのresponse_archive_manifest.jsonに圧縮前の内容hash・サイズと圧縮ファイルのhash・サイズを保存。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    files={ROOT/n for n in read(folders[0]/'protocol.json')['optimizer_source_sha256']}
    files.update(ROOT/n for n in StageSimulator('cut_cu').provenance()['sha256'])
    files.update(Path(__file__).parent.glob('interconnect_stage_*.py'));files.add(ROOT/'main.py')
    files.add(ROOT/'v005/plot_prediction_gui.py')
    manifest={}
    for p in files:
        relative=p.relative_to(ROOT);target=out/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);manifest[relative.as_posix()]=digest(target)
    dump(out/'source_snapshot/sha256.json',manifest)
    dump(out/'runtime_dependencies.json',{'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,
        'cli_threads':{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        'benchmark_runtime':'bundled Python; GUI smoke used local Python312 with matplotlib'})
    print(json.dumps({'status':'OK','trials':len(results),'measurements':sum(r['measurements'] for r in results),'report':str(out/'REPORT.md')},ensure_ascii=False))


if __name__=='__main__': main()
