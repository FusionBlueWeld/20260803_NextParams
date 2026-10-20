"""Posthoc independent verification and combined human-readable report."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from validation.single.src.simulators import Simulator

NAMES = {'laser_welding': 'レーザー溶接', 'milling': 'フライス切削',
         'press_forming': 'プレス打抜き', 'thermal_curing': '樹脂・接着剤の熱硬化',
         'convection_drying': '塗膜の熱風乾燥', 'electroplating': '銅電解めっき'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    trials = ROOT/'v005/trials'
    folders = sorted(p for p in trials.glob('trial_single_arrival_*_20261010_focus_on*')
                     if (p/'summary.json').exists())
    assert len(folders) == 18, f'Expected 18 finished trials, found {len(folders)}'
    results = []
    for folder in folders:
        summary = read(folder/'summary.json')
        assert summary['error'] is None, summary
        protocol = read(folder/'protocol.json')
        sim = Simulator(ROOT/'validation/single'/summary['simulator'])
        audit = read(folder/'posthoc_grid_audit.json')
        assert sim.provenance() == audit['simulator_provenance']
        with (folder/'data/experiments.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        points = np.array([[float(r[n]) for n in sim.parameter_columns] for r in rows])
        assert len(np.unique(points, axis=0)) == len(rows) == summary['measurements']
        outputs = sim.evaluate_points(points)
        for n in sim.output_columns:
            assert np.allclose(outputs[n], [float(r[n]) for r in rows], rtol=1e-12, atol=1e-12), n
        ledger = read(folder/'ledger.json')
        for i, entry in enumerate(ledger, 1):
            request_path = folder/'exchange'/f'request_{i:03d}.json'
            measurement_path = folder/'exchange'/f'measurement_{i:03d}.json'
            assert entry['request_sha256'] == digest(request_path)
            assert entry['measurement_sha256'] == digest(measurement_path)
            measured = read(measurement_path)
            assert measured['simulator_provenance'] == audit['simulator_provenance']
            assert measured['request_sha256'] == digest(request_path)
            request = read(request_path)
            assert request['source_recommendation_sha256'] == digest(folder/'steps'/f'step_{i-1:03d}'/'recommendations.csv')
            assert all(measured['measurements'][0][n] == value for n,value in request['conditions'][0].items())
            assert all(str(measured['measurements'][0][n]) == rows[i-1][n] for n in rows[i-1])
        good = sim.feasible(outputs)
        optimal = good & np.isclose(outputs[sim.objective.column], audit['best_value'], rtol=1e-9, atol=1e-10)
        indices = np.flatnonzero(optimal)
        arrival = int(indices[0]+1) if len(indices) else None
        assert arrival == summary['first_optimum_measurement']
        for n,h in protocol['optimizer_source_sha256'].items():
            assert digest(ROOT/n) == h
        flags = ['initial_design_enabled','parameter_focus_enabled','temporary_fix_enabled']
        assert all(read(folder/'focus_settings.json')[n] is True for n in flags)
        snapshots = [read(p) for p in sorted((folder/'output/parameter_focus').glob('*.json'))]
        fixed = sum(bool(s.get('fixed_values')) for s in snapshots)
        scopes = sorted({s.get('candidate_space_scope', 'unknown') for s in snapshots})
        summary.update(independent_physics_replay_verified=True, request_measurement_hashes_verified=True,
                       recommendation_hashes_verified=True, optimizer_source_unchanged=True,
                       runs_with_temporary_fixed_inputs=fixed, recorded_focus_scopes=scopes,
                       equally_optimal_grid_conditions=audit['optimal_count'])
        results.append(summary)
    out = trials/'_single_arrival_20261010_focus_on'
    snapshot = out/'source_snapshot'
    source_files = {ROOT/n for n in read(folders[0]/'protocol.json')['optimizer_source_sha256']}
    source_files.update([ROOT/'main.py', ROOT/'validation/__init__.py',
                         ROOT/'validation/single/__init__.py'])
    source_files.update((ROOT/'validation/single/src').glob('*.py'))
    source_files.update(Path(__file__).parent.glob('single_arrival_*.py'))
    for name in NAMES:
        sim = Simulator(ROOT/'validation/single'/name)
        source_files.update(sim.folder/n for n in sim.provenance()['sha256_lf'])
    manifest = {}
    for source in sorted(source_files):
        relative = source.relative_to(ROOT)
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = digest(target)
    (snapshot/'sha256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out/'all_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out/'all_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    text = ['# v005：単一工程の実試行回数検証', '',
            '実行日：2026-10-10（日本時間）。validation/single/の6工程のみ。複合工程は使用していない。', '',
            '各工程をseed 42・0・1で実行した。初期データ0件から、v005通常CLIが1条件を提案し、',
            '別プロセスの物理シミュレータが評価、その結果だけを実験CSVに追記して再学習した。',
            '1回＝1入力条件の物理計算。初期設計も試行回数に含む。反復測定・測定ノイズはなし。',
            '各trialは最初の停止判定か最大150条件で終了。最適解到達と自動収束は別に記録した。', '',
            '## 到達回数', '',
            '|工程|seed 42|seed 0|seed 1|到達試行数 / 3|',
            '|---|---:|---:|---:|---:|']
    for sim, title in NAMES.items():
        group = {r['seed']: r for r in results if r['simulator']==sim}
        cells = [str(group[s]['first_optimum_measurement']) if group[s]['optimum_reached']
                 else f"未到達（{group[s]['measurements']}条件）" for s in [42,0,1]]
        text.append(f"|{title}|{'|'.join(cells)}|{sum(r['optimum_reached'] for r in group.values())}/3|")
    text.extend(['', 'フライス切削のseed 1は初期設計の最初の条件が同率最適に該当した。',
                 '同率最適は92 / 65,536条件ある。この1条件という結果を、学習後の探索効率や一般的な必要回数とは扱わない。'])
    text.extend(['', '## 全試行の最終結果', '',
                 '|工程|seed|初期設計|総条件数|最良値|格子最適値|最適値との差|停止状態|',
                 '|---|---:|---:|---:|---:|---:|---:|---|'])
    for sim, title in NAMES.items():
        for r in sorted((r for r in results if r['simulator']==sim), key=lambda r:[42,0,1].index(r['seed'])):
            text.append(f"|{title}|{r['seed']}|{r['initial_design_measurements']}|{r['measurements']}|{r['best_value']:.10g}|{r['grid_optimum']:.10g}|{r['objective_gap']:.10g}|{r['stop_status']}|")
    text.extend(['', '## 判定と設定', '',
                 '最適解はproblem.csvの全離散格子における、全制約を満たす最良目的値。同率最適も到達とする。',
                 '値の照合許容差はrelative=1e-9 / absolute=1e-10。全格子物理評価は各探索終了後のみ実施し、',
                 '監査値・最適条件を学習データ、停止目標、知見、探索設定に渡していない。',
                 '未到達は今回の予算・推薦停止までに到達しなかったことを示す。到達に必要な回数を推定で補っていない。',
                 'REVIEW_REQUIREDは推薦不足等による要確認、STOP_REQUIREDは予算等による必須停止、STOP_RECOMMENDEDが自動収束。', '',
                 'focus_settings.jsonのinitial_design_enabled / parameter_focus_enabled / temporary_fix_enabledは全てtrue。',
                 'effect_tolerancesは既定の空設定を維持。ONでも、根拠不足の入力を自動固定する許容幅を追加していない。',
                 f"一時固定が記録されたモデルrun：{sum(r['runs_with_temporary_fixed_inputs'] for r in results)}。集中探索の状態は各trialのoutput/parameter_focus/に保存。",
                 'search_settingsはcandidate_mode=grid・max_candidates=元格子件数、max_learning_seconds=120秒。',
                 'その他の候補選択・停止基準はv005の既定値。学習上限256点・2,000epoch、CPU 1スレッド。',
                 'seedは初期設計とNN等の乱数に使用する。3 seedから一般的な必要試行数の保証は行わない。', '',
                 '## 保存物と独立再検証', '',
                 '各trialのprotocol.jsonにモデルソースのSHA256、Python環境、設定、予算、再現条件を保存。',
                 'data/experiments.csvは全実測相当データ。exchange/は要求と物理計算応答。',
                 'steps/step_NNN/はその時点のデータ、CLIログ、推薦、停止判定、focus状態のスナップショット。',
                 'ledger.json / progress.csvに推移、posthoc_grid_audit.jsonに格子最適値・条件・物理モデルのSHA256、summary.jsonに到達回数。',
                 'output/response_spaces/の時点別予測はv005/plot_prediction_gui.pyから閲覧できる。',
                 '終了後、全測定点を物理関数で独立再計算して保存CSVとの一致を確認。全要求・応答・推薦のSHA256、',
                 '入力一致、測定条件の重複なし、optimizerソース不変、最適値初到達インデックス、機能ONも確認した。',
                 '集計：all_results.json / all_results.csv。再実行コード：v005/benchmarks/single_arrival_validation.py、single_arrival_worker.py。', '',
                 'source_snapshot/に検証時のoptimizer・物理モデル・共通定義・検証コードのソースとSHA256を保存。',
                 '今後のコード更新後も、今回の版と照合できる。', '',
                 '既存の物理モデル契約テスト7件は通過。既存test_simulators全17件では、一時フォルダの',
                 'Windowsアクセス制限により7件がエラーとなったため、検証器全体のテスト通過とは報告しない。',
                 '今回の18 trialの保存データは上記の独立物理再計算とファイル照合で別途検証済み。', '',
                 '再実行は新しいtagで行う。既存trialは上書きしない。', '',
                 '```powershell',
                 '& "C:/Users/tsuts/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" v005/benchmarks/single_arrival_validation.py --tag NEW_UNIQUE_TAG --budget 150 --seeds 42 0 1',
                 '```', '',
                 '本結果は決定論的シミュレータの離散条件についての検証であり、連続空間の最適解や実設備のばらつきは対象外。'])
    (out/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps({'verified_trials': len(results), 'measurements': sum(r['measurements'] for r in results),
                      'reached': sum(r['optimum_reached'] for r in results),
                      'report': str(out/'REPORT.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
