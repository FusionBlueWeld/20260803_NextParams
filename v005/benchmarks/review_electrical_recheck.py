"""Independent saved-data review and plots for the stopping recheck."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
TRIALS = ROOT / 'v005/trials'
BATCH = TRIALS / '_recheck_20261010'
STAGES = ['cut_al', 'form_cu', 'form_al', 'assembly', 'interconnect_welding']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def good(contract, row):
    return all(float(row[c['name']]) <= c['target']+1e-12 if c['direction']=='less_equal'
               else float(row[c['name']]) >= c['target']-1e-12 for c in contract['constraints'])


def close(left, right):
    return abs(left-right) <= max(1e-10, abs(right)*1e-9)


def verify_case(stage):
    result=read(BATCH/f'{stage}.json')
    trial=TRIALS/result['trial']
    source=TRIALS/result['source_trial']
    contract=read(trial/'simulator_contract.json')
    data=rows(trial/'data/experiments.csv')
    initial=rows(source/'steps/step_000/experiments.csv')
    assert data[:24]==initial and len(initial)==24
    assert len(data)==result['measurements']
    ids=[row['experiment_id'] for row in data]
    assert len(ids)==len(set(ids))
    keys=[tuple(float(row[p]) for p in contract['parameters']) for row in data]
    assert len(keys)==len(set(keys))
    for name in ['problem.csv','knowledge_constraints.csv','search_regions.csv',
                 'stop_settings.csv','stage_connection.json','simulator_contract.json']:
        old_path=source/name
        if old_path.exists(): assert old_path.read_bytes()==(trial/name).read_bytes()
    old_config=read(source/'search_settings.json')
    new_config=read(trial/'search_settings.json')
    assert all(new_config[name]==value for name,value in old_config.items())
    assert new_config['cpu_threads']==1
    position=24
    for entry in read(trial/'ledger.json')[1:]:
        step=entry['step']
        exchange=trial/'exchange'
        request_path=exchange/f'request_{step:03d}.json'
        requested=read(request_path)
        measured=read(exchange/f'measurement_{step:03d}.json')
        assert measured['request_sha256']==hashlib.sha256(request_path.read_bytes()).hexdigest()
        recs=rows(trial/'steps'/f'step_{step:03d}'/'recommendations.csv')
        count=len(requested['conditions'])
        assert 1<=count<=5 and len(recs)==len(measured['results'])==count
        for rec, asked, actual, row in zip(recs,requested['conditions'],measured['results'],data[position:position+count]):
            assert asked['experiment_id']==actual['experiment_id']==row['experiment_id']
            assert int(rec['actual_recommendations'])==count
            for name in contract['parameters']:
                assert asked[name]==actual['inputs'][name]==float(row[name])
                assert close(float(rec[name]),asked[name])
            for name in contract['outputs']:
                assert actual['outputs'][name]==float(row[name])
            assert actual['feasible']==good(contract,row)
        position+=count
        assert entry['total_measurements']==position
        assert rows(trial/'steps'/f'step_{step:03d}'/'experiments.csv')==data[:position]
    assert position==len(data)
    optimum=result['oracle_grid_best']
    passing=[row for row in data if good(contract,row)]
    objective=contract['objective']
    best=(min if contract['direction']=='minimize' else max)(float(row[objective]) for row in passing)
    assert close(best,result['best_value'])
    first=next((index for index,row in enumerate(data,1)
                if good(contract,row) and close(float(row[objective]),optimum)),None)
    result['first_optimum_measurement']=first
    result['measurements_after_first_optimum']=None if first is None else len(data)-first
    original=read(source/'summary.json')
    result['old_measurements_after_first_optimum']=original['measurements']-original['first_optimum_measurement']
    initial_feasible=sum(good(contract,row) for row in initial)
    original_data=rows(source/'data/experiments.csv')
    old_added=original_data[24:]
    new_added=data[24:]
    result['old_additional_feasible_rate']=sum(good(contract,row) for row in old_added)/len(old_added)
    result['new_additional_feasible_rate']=sum(good(contract,row) for row in new_added)/len(new_added)
    assert sum(good(contract,row) for row in passing)==result['feasible_measurements']
    result['initial_feasible_measurements']=initial_feasible
    first_step=None if first is None else (int(data[first-1]['experiment_id'].split('_')[1])
                    if data[first-1]['experiment_id'].startswith('step_') else 0)
    assert first_step==result['first_optimum_step']
    assert result['optimum_retained']==close(best,optimum)
    for run in result['run_timing']:
        pools=run['runtime'].get('libraries',[])
        assert pools and all(pool['threads']==1 for pool in pools)
        if not run['learning_skipped']: assert run['prediction_grid_passes']==1
    result['new_learning_seconds_sum']=sum(run['learning_seconds'] or 0 for run in result['run_timing'])
    result['new_pipeline_seconds_sum']=sum(run['wall_seconds'] or 0 for run in result['run_timing'])
    result['new_prediction_seconds_sum']=sum(run['prediction_seconds'] or 0 for run in result['run_timing'])
    old_budgets=[read(source/'steps'/f'step_{step:03d}'/'search_budget.json')
                 for step in range(1,result['old_steps']+1)]
    result['old_learning_seconds_sum']=sum(budget['learning']['seconds'] for budget in old_budgets)
    result['timing_comparison_limit']='Old pipeline wall/CPU environment is unavailable; learning sums are contextual, not a controlled speedup estimate.'
    diagnostics=[]
    for path in sorted((trial/'steps').glob('step_*/stopping_status.json')):
        record=read(path)
        if record['metrics'].get('promising_support_coverage') is not None:
            diagnostics.append(record)
    last=diagnostics[-1]
    result['last_model_stop_metrics']=last['metrics']
    result['last_model_training_rows']=last['runs'][-1]['data_rows']
    assert result['converged']==(result['engine_stop_status']=='STOP_RECOMMENDED')
    if result['engine_stop_status']!='CONTINUE':
        assert not (trial/'output/recommendations.csv').exists()
    gui_logs=list(trial.glob('*smoke*.log'))+list((trial/'recheck_logs').glob('*smoke*.log'))
    assert gui_logs,'Missing saved GUI verification'
    blob=gui_logs[-1].read_bytes()
    encoding='utf-16' if blob.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
    gui_log=blob.decode(encoding,errors='replace')
    assert re.search(r'"status"\s*:\s*"OK"',gui_log)
    frame_count=int(re.search(r'"frames"\s*:\s*(\d+)',gui_log).group(1))
    assert frame_count==result['steps']+1
    result['gui_verified_frames']=frame_count
    result['gui_smoke_status']='OK'
    audit_path=trial/'posthoc_grid_audit.json'
    assert read(audit_path)['not_supplied_to_optimizer']
    assert audit_path.stat().st_mtime >= (trial/'data/experiments.csv').stat().st_mtime
    result['posthoc_audit_after_last_ingest_verified']=True
    result['independent_saved_data_review_passed']=True
    return result


def plot(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={font.name for font in font_manager.fontManager.ttflist}
    for font in ['Meiryo','Yu Gothic','MS Gothic']:
        if font in fonts:
            plt.rcParams['font.family']=font
            break
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    x=list(range(len(cases)));width=.36
    axes[0].bar([i-width/2 for i in x],[case['old_measurements'] for case in cases],width,label='修正前')
    bars=axes[0].bar([i+width/2 for i in x],[case['measurements'] for case in cases],width,label='修正後')
    for bar,case in zip(bars,cases):
        axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,str(case['measurements']),ha='center')
    labels=['Al切出し','Cu成形','Al成形','組立','溶接']
    axes[0].set_xticks(x,labels)
    axes[0].set_ylabel('測定条件数（初期24条件を含む）')
    axes[0].set_title('修正前後の実験数')
    axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for label,color,case in zip(labels,colors,cases):
        for name,style in [(case['source_trial'],'--'),(case['trial'],'-')]:
            ledger=read(TRIALS/name/'ledger.json')
            records=[entry for entry in ledger if entry['best_value'] is not None]
            optimum=case['oracle_grid_best']
            gaps=[max(0,100*(entry['best_value']-optimum)/abs(optimum)) for entry in records]
            axes[1].plot([entry['total_measurements'] for entry in records],gaps,style,color=color,
                         label=label if style=='-' else None)
    axes[1].set_xlabel('累計測定条件数');axes[1].set_ylabel('最適値からの差 [%]')
    axes[1].set_title('実測最良値の推移：破線=修正前、実線=修正後')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(BATCH/'comparison.png',dpi=150)
    plt.close(fig)


def report(total):
    cases=total['cases']
    converged=total['automatically_converged_count']
    total_old,total_new=total['total_old_measurements'],total['total_new_measurements']
    saved=total['saved_measurements']
    lines=['# 電気接続5工程：v005改善後の実計算検証（2026-10-10）','',
        f"最適値維持 {total['optimum_retained_count']}/5工程、自動収束 {converged}/5工程。",
        f"測定数は合計 {total_old} → {total_new}、{saved}条件（{100*saved/total_old:.1f}%）削減した。",
        '']
    if converged==0:
        lines += ['実験数の削減は確認できた。一方、電気接続で自動収束しない問題は、この修正だけでは解消していない。', '']
    lines += ['## 比較条件','',
        'Solが設定・保存データの独立レビュー、Luna 2 workersが反復実行を担当した。',
        '前回と同じ初期実測24条件をコピーし、問題・制約・固定流入状態・格子・NN学習予算を一致させた。',
        '推薦上限5条件、最大20測定ステップ／追加100条件。改善後は推薦件数が可変。',
        'NN ensemble 5、最大2,000 epoch／120秒、学習上限256条件は前回と同じ。数値計算は1スレッド、同時探索は2工程まで。',
        '探索中にアルゴリズム・閾値を変更せず、終了後にのみ独立全格子監査を実行した。',
        '固定流入状態・ノイズなし・1 seed・離散格子での検証。実設備や連結ライン全体の保証ではない。',
        '', '## 実験数・終了理由','',
        '|工程|前回ステップ／測定数|今回ステップ／測定数|削減測定数|今回の終了|最適値維持|',
        '|---|---:|---:|---:|---|---|']
    for case in cases:
        title=case['title'].replace('電気接続｜','')
        label={'REVIEW_REQUIRED':'要確認','STOP_RECOMMENDED':'自動収束','STOP_REQUIRED':'必須終了',
               'EXTERNAL_STEP_LIMIT':'比較上限（自動収束なし）'}[case['stop_status']]
        if case['stop_status']=='STOP_REQUIRED' and any('実験回数' in r for r in case['stop_reasons']): label='実験予算到達'
        lines.append(f"|{title}|{case['old_steps']}／{case['old_measurements']}|{case['steps']}／{case['measurements']}|{case['saved_measurements']}|{label}|{'維持' if case['optimum_retained'] else '未到達'}|")
    lines += ['', 'ステップは実際に測定したバッチ数。停止確認だけの最後のCLI実行は測定ステップに含めない。',
        '前回5工程は全て追加100条件の実験予算到達で終了していた。今回の要確認は、有用候補0件を理由に推薦を止める状態で、自動収束・最適性の証明ではない。',
        '', '## 最適値への到達','',
        '|工程|前回の初到達ステップ|今回の初到達ステップ|今回の実測最良 [s]|独立格子最良との差|',
        '|---|---:|---:|---:|---:|']
    for case in cases:
        lines.append(f"|{case['title'].replace('電気接続｜','')}|{case['old_first_optimum_step']}|{case['first_optimum_step']}|{case['best_value']:.10g}|{case['objective_gap']:.3g}|")
    lines += ['', '最適値を発見するまでの速さは工程により前後しており、一律の改善ではない。格子最適値は探索後に真値を全列挙して確認し、学習データには追加していない。',
        '', '## 追加推薦の実際の合格率','',
        '|工程|前回|今回|', '|---|---:|---:|']
    for case in cases:
        lines.append(f"|{case['title'].replace('電気接続｜','')}|{100*case['old_additional_feasible_rate']:.1f}%|{100*case['new_additional_feasible_rate']:.1f}%|")
    lines += ['', '初期24条件を除いた実際の推薦・測定点の割合。測定点と測定件数が異なるため、共通heldoutでの予測精度比較ではない。制約違反率も一律に改善したわけではなく、溶接は84%から72%へ低下した。',
        '', '## 収束を妨げた条件','',
        '収束条件は有望領域の支持率80%以上、予測最適目的値の直近変動0.001秒以下などの組合せ。以下は最後にモデルを学習した時点の診断で、予算停止時の学習省略とは区別する。',
        '', '|工程|学習実測数|有望領域の支持率|予測最適値の変動 [s]|推薦スコアの小ささ・安定|',
        '|---|---:|---:|---:|---|']
    for case in cases:
        m=case['last_model_stop_metrics']
        lines.append(f"|{case['title'].replace('電気接続｜','')}|{case['last_model_training_rows']}|{100*m['promising_support_coverage']:.1f}%|{m['predicted_optimum_objective_spread']:.6g}|{'適合' if m['top_score_stable_and_small'] else '未適合'}|")
    lines += ['',
        '全域支持から有望領域支持への変更は支持率を上げたが、100追加条件の予算で80%支持を得る設計にはまだ隔たりがある。目的値が既に停滞していても、モデルの予測最適値の変動は別の条件として残る。',
        '要確認によって無理な実験継続を抑えられたことと、十分な根拠で自動収束できたことは分けて評価する。',
        '', '## 計算負荷と保存','',
        '|工程|前回の学習時間合計 [s]|今回の学習時間合計 [s]|今回の探索処理合計 [s]|',
        '|---|---:|---:|---:|']
    for case in cases:
        lines.append(f"|{case['title'].replace('電気接続｜','')}|{case['old_learning_seconds_sum']:.2f}|{case['new_learning_seconds_sum']:.2f}|{case['new_pipeline_seconds_sum']:.2f}|")
    lines += ['',
        '学習時間は各runの保存値の合計。今回の探索処理は入力検査・学習・全候補予測・診断・CSVとbundle保存を含む。Python起動、シミュレータ、Codexの受け渡し待ちは含まない。前回の探索処理全体時間とスレッド環境の厳密な記録はないため、表は統制された速度向上率ではない。',
        '各学習runの全候補予測は1回、実OpenBLASスレッド数は1と確認した。2 workersの同時探索で計算負荷を限定した。',
        '', '## 独立レビュー','',
        '全追加実測について、推薦CSV→条件JSON→測定JSON→実測CSVの一致、request SHA-256、各時点snapshotの範囲、重複なしをSolが別途検証した。',
        'v005／物理モデルのsource hashと前回の実測・初期snapshot・summary・ledgerのhashが実行前後で一致している。',
        '全trialのGUI smoke test結果は各trialのログへ保存。新trialはv005/plot_prediction_gui.pyのtrial選択で切替可能。',
        '', '## この結果から必要な追加改善','',
        '1. 有望領域の支持率と実験予算の整合を再設計する。根拠なく80%を下げず、残る改善可能性・制約リスク・有効な入力次元を併せて評価する。',
        '2. 予測最適値の変動許容幅を、工程の必要精度・目的値の尺度・モデルの予測誤差に対応させる。一律0.001秒で止まるかを判断する設計は再検討する。',
        '3. 要確認で止まった後の診断を具体化し、追加実験が必要な領域と、改善余地が十分小さい状態を区別する。',
        'この比較中には追加修正を加えていない。上記は今回の実計算で残った課題である。',
        '', '## 保存先','',
        '- comparison.json：独立レビュー済みの全数値と各runの時間・停止指標',
        '- comparison.png：測定数と実測最良値の推移',
        '- source_manifest.json：実行前のsource／旧データhash',
        '- v005/benchmarks/recheck_electrical.py：標準ライブラリのみの受け渡し・集計',
        '- v005/benchmarks/review_electrical_recheck.py：独立レビューと図の再生成', '']
    (BATCH/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    manifest=read(BATCH/'source_manifest.json')
    for group in ['source_sha256','historical_data_sha256']:
        for name,value in manifest[group].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name
    available=[stage for stage in STAGES if (BATCH/f'{stage}.json').exists()]
    if not args.partial: assert len(available)==5,'All five real searches must finish'
    cases=[verify_case(stage) for stage in available]
    total={'complete':len(cases)==5,'verified_cases':len(cases),
        'total_old_measurements':sum(case['old_measurements'] for case in cases),
        'total_new_measurements':sum(case['measurements'] for case in cases),
        'saved_measurements':sum(case['saved_measurements'] for case in cases),
        'optimum_retained_count':sum(case['optimum_retained'] for case in cases),
        'automatically_converged_count':sum(case['converged'] for case in cases),
        'review_required_count':sum(case['stop_status']=='REVIEW_REQUIRED' for case in cases),
        'source_and_old_data_unchanged':True,
        'scope':'one seed, same initial24 and fixed context, independent noiseless physical simulator, discrete grid',
        'cases':cases}
    (BATCH/'comparison.json').write_text(json.dumps(total,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.partial:
        plot(cases)
        report(total)
    print(json.dumps({key:value for key,value in total.items() if key!='cases'},indent=2))


if __name__=='__main__':
    main()
