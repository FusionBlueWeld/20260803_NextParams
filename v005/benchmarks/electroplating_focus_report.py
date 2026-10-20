"""Verify the paired physical runs and write the combined diagnosis report."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from validation.single.src.simulators import Simulator

OUT=ROOT/'v005/trials/_electroplating_focus_diagnosis_20261010'
LABELS=['電流密度','浴温度','時間','銅濃度','撹拌','電極間隔','デューティ比','pH']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    diagnosis=read(OUT/'diagnosis.json')
    sim=Simulator(ROOT/'validation/single/electroplating')
    pairs=[]
    for arm in ['on','off']:
        trial=ROOT/'v005/trials'/f'trial_plating_focus_{arm}_batch3_s42_20261010'
        summary=read(trial/'summary.json')
        protocol=read(trial/'protocol.json')
        with (trial/'data/experiments.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        points=np.array([[float(r[n]) for n in sim.parameter_columns] for r in rows])
        values=sim.evaluate_points(points)
        assert len(np.unique(points,axis=0))==150==len(rows)
        for n in sim.output_columns:
            assert np.allclose(values[n],[float(r[n]) for r in rows],rtol=1e-12,atol=1e-12)
        assert digest(trial/'steps/step_000/experiments.csv')==protocol['initial_snapshot_sha256']
        for n,h in protocol['optimizer_sha256'].items():assert digest(ROOT/n)==h
        ledger=read(trial/'ledger.json')
        for entry in ledger[1:]:
            step=entry['step']
            request=trial/'exchange'/f'request_{step:03d}.json'
            response=trial/'exchange'/f'measurement_{step:03d}.json'
            assert digest(request)==entry['request_sha256']
            assert digest(response)==entry['measurement_sha256']
            req,res=read(request),read(response)
            assert digest(request)==res['request_sha256']
            assert req['recommendation_sha256']==digest(trial/'steps'/f'step_{step-1:03d}'/'recommendations.csv')
            assert len(req['conditions'])==len(res['measurements'])
            for condition,measurement in zip(req['conditions'],res['measurements']):
                assert all(measurement[n]==value for n,value in condition.items())
        summary['focus_assessment_runs']=len(list((trial/'output/parameter_focus').glob('*.json')))
        summary['model_runs']=len(list((trial/'output/response_spaces').glob('*.csv')))
        summary['independent_physics_replay_verified']=True
        summary['request_response_recommendation_hashes_verified']=True
        summary['initial_snapshot_sha256']=protocol['initial_snapshot_sha256']
        summary['focus_settings']=protocol['focus_settings']
        (trial/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        pairs.append(summary)
    assert pairs[0]['initial_snapshot_sha256']==pairs[1]['initial_snapshot_sha256']
    settings0=dict(pairs[0]['focus_settings']);settings1=dict(pairs[1]['focus_settings'])
    for n in ['parameter_focus_enabled','temporary_fix_enabled']:
        assert settings0.pop(n) is True and settings1.pop(n) is False
    assert settings0==settings1
    (OUT/'pair_verified.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# 電解めっき：入力限定機能と未到達の原因調査','',
          '2026-10-10。v005の製品コード・物理関数・過去の18 trialは変更していない。',
          '保存済み3 trialの事後診断、全65,536格子の物理評価、観測データだけを使った回帰比較、',
          '同じ初期48条件からの限定ON/OFFの追加物理計算を行った。','',
          '電解めっきは8入力×4水準＝65,536条件。フライスも同じ8入力・同じ候補数だが、',
          '同率最適が92条件ある一方、電解めっきの格子最適は1条件だけ。入力数だけで難しさを説明できない。','',
          '## 1. 前回の限定機能は実質的に絞り込みを行っていなかった','',
          '3 seedそれぞれ、初期設計48条件＋改善推薦102条件。306回のモデル更新すべてで全8入力の重みは1。',
          '一時固定0、SCREENING / FROZEN_CHECK / GLOBAL_CHECKの実施0。','',
          '原因は以下の4段階にある。','',
          '- 許容作用量：effect_tolerancesが空。目的と全制約に許容幅が揃わないと固定判定に進まない。',
          '- 効果推定：補助回帰は切片＋各入力の一次・二次項の17列。入力間の交互作用項を含まない。',
          '- 保留条件：いずれかの出力の予測誤差が観測標準偏差の20%を超えると、全入力の配分重みを1に維持する。',
          '- 確認枠：--n 1では有用な改善候補がある限り改善枠1を保護するため、確認枠は0になる。',
          '  一時固定には全4水準を比較した複数背景と現在の最良条件の背景での確認を必要とするが、確認データを積極的に取得できない。','',
          '「重みを下げない・固定しない」は保守的な分岐として正常に実行されている。',
          'ただし今回の問題と運用では、入力限定の効果を生むまで進めていない。','',
          '## 2. 許容幅の設定だけでは固定できない','',
          '|seed|補助回帰の膜厚誤差の予測誤差目安 µm（149点時）|ノイズ推定値 µm（150点時）|ノイズだけの作用上限の床 µm|',
          '|---|---:|---:|---:|']
    for t in diagnosis['trials']:
        text.append(f"|{t['seed']}|{t['last_analysis']['prediction_errors']['thickness_error_um']:.3f}|{t['final_data_noise_proxy_um']:.3f}|{t['noise_only_lower_bound_on_upper_effect_um']:.3f}|")
    text.extend(['','固定用の作用上限は「推定作用＋2×予測誤差＋2×ノイズ目安」。これは統計的信頼区間ではない。',
                 'この物理シミュレーションに測定ノイズはないが、反復測定がないためv005は出力全体の標準偏差の5%をノイズ目安に使用している。',
                 '入力条件による出力変動が測定ノイズの目安にも入り、仮に回帰誤差と作用が0でも膜厚の固定用上限は約0.69 µmを下回れない。',
                 '膜厚の許容作用量0.01 / 0.1 / 1 µm、粗さ0.05 µm、電流効率0.02を設定して最終データの判定を再計算したが、',
                 'すべて固定0・全入力ACTIVEだった。これは事後の単一状態の再判定であり、新規逐次探索の性能証明ではない。',
                 '許容幅は診断用の仮定で、製品仕様として設定したものではない。','',
                 '## 3. 主要因と交互作用','',
                 '|入力|膜厚誤差の総効果指標|粗さの総効果指標|電流効率の総効果指標|',
                 '|---|---:|---:|---:|'])
    for j,label in enumerate(LABELS):
        text.append('|'+label+'|'+ '|'.join(f"{diagnosis['sensitivity'][n]['total_effect'][j]:.4f}" for n in ['thickness_error_um','roughness_ra_um','current_efficiency'])+'|')
    text.extend(['','総効果指標は、入力間の組合せ作用を含む離散格子上の分散指標。相互作用を重複して含むため合計1にはならない。',
                 '全格子を一様な重みで評価し、制約不適合の条件も含む。最適点近傍の因果効果や固定可否とは別の指標。',
                 f"膜厚誤差の全格子分散のうち、主効果の和で説明されない相互作用分は{diagnosis['sensitivity']['thickness_error_um']['interaction_variance_fraction']*100:.1f}%。",
                 '時間・電流密度・デューティ比が主軸。銅濃度は粗さ・電流効率の制約にも大きく影響する。',
                 '浴温度・撹拌・電極間隔・pHは膜厚の全域指標が小さいが、細かい目標合わせにも関与する。',
                 '物理モデルは膜厚 ∝ 電流効率×電流密度×時間×デューティ比という積の構造を持つ。',
                 '電流効率も他の入力の組合せで変わり、目的は|膜厚−20 µm|である。','',
                 '各trialの実測最良を基準に、1入力だけ、または任意の2入力だけを変えた全許可格子も事後評価した。',
                 '3 seedとも、その範囲に現在の最良値を改善する条件はなかった。単純な1～2変数ずつの改善探索では、この局所解から動けない。',
                 'seed 42で弱い変数に見える電極間隔を現在値1 cmに固定すると、元の一意な格子最適解を除外する。',
                 '固定した空間での最良誤差は0.0008815 µmで、元の格子最適0.00006046 µmとは異なる。',
                 '広域の影響が小さいことだけを理由に固定すると、厳密最適性を失う。実用許容差がある場合には評価が変わる。','',
                 '## 4. 探索用モデルは真の最適条件を高く評価していない','',
                 '|seed|真の最適条件への予測誤差 µm|予測合格確率|未測定候補内のEI×合格確率順位|',
                 '|---|---:|---:|---:|'])
    for t in diagnosis['trials']:
        p=t['prediction_metrics']
        text.append(f"|{t['seed']}|{p['true_optimal_point_predicted_error_um']:.3f}|{p['true_optimal_point_feasibility_probability']:.4f}|{p['true_optimal_point_EI_score_rank_among_unmeasured']} / {p['unmeasured_count']}|")
    text.extend(['','最後の149条件で学習した保存済み予測を診断した。真の最適値は0.00006046 µm。',
                 '制約適合の予測確率は高いが、目的誤差を大きく予測しているため、改善候補として優先されにくい。',
                 '順位は保存された予測から再計算したEI×合格確率の順位。全入力の配分重みが1であることも別途確認した。',
                 'これは最終時点の診断であり、探索全期間の未到達原因を一つに特定する因果実験ではない。','',
                 '## 5. 目的表現を変える改善案には数値的な手がかりがある','',
                 '同じ150観測点だけを使い、固定した5分割で補助回帰を比較した。候補全格子の正解を学習に使用していない。','',
                 '|seed|現行と同じ加法二次・絶対誤差のMAE µm|交互作用を追加した絶対誤差MAE µm|符号付き膜厚差＋交互作用のMAE µm|',
                 '|---|---:|---:|---:|'])
    for t in diagnosis['trials']:
        cv=t['observed_data_regression_CV']
        text.append(f"|{t['seed']}|{cv['absolute_error_additive']['heldout_mae_um']:.3f}|{cv['absolute_error_with_interactions']['heldout_mae_um']:.3f}|{cv['signed_thickness_with_interactions']['heldout_mae_um']:.3f}|")
    text.extend(['','交互作用項を足すだけでは改善しなかった。「膜厚−20 µm」という符号付き量を学習してから絶対値を取る組合せが良かった。',
                 'また、保存済みHybridの膜厚予測から誤差を計算すると、全格子MAEは0.88～0.98 µmで、誤差を直接学習した1.62～2.29 µmより小さかった。',
                 'ただし平均膜厚の絶対誤差は誤差の期待値と同じではなく、不確かさ・制約との関係を扱う獲得関数の設計も必要。',
                 'ここでは回帰・事後予測の比較までで、修正版v005が少ない試行で最適解へ到達することは未検証。','',
                 '## 6. 同じ初期48点・同じ150条件予算での追加ON/OFF実計算','',
                 'seed 42、--n 3、初期48点はSHA256まで同一。追加102条件をそれぞれ別プロセスの物理モデルで評価した。',
                 '両armの仮の許容作用量は膜厚0.01 µm・粗さ0.05 µm・電流効率0.02で共通。',
                 '違いはparameter_focus_enabledとtemporary_fix_enabledのみ。初期設計ON、候補・学習・停止・実験数の設定は共通。','',
                 '|設定|最良膜厚誤差 µm|GLOBAL_CHECK|SCREENING|固定run|配分重みを下げたrun|',
                 '|---|---:|---:|---:|---:|---:|'])
    for p in pairs:
        text.append(f"|限定{p['arm'].upper()}|{p['best_error_um']:.8f}|{p['roles'].get('GLOBAL_CHECK',0)}|{p['roles'].get('SCREENING',0)}|{p['runs_with_fixed_inputs']}|{p['runs_with_reweighted_inputs']}|")
    text.extend(['','3推薦にすると確認枠が働くことは実計算で確認できた。一時固定は発動しなかった。',
                 'この単一seedの比較では、ONが少ない試行で最適値に到達する改善は確認できなかった。',
                 '確認実験に枠を使うこと自体を性能向上と扱わない。一般的な優劣には複数seedと修正後の比較が必要。',
                 '両armは150条件の予算で停止。物理再計算、入力・要求・応答・推薦のハッシュ、初期データ一致、重複なし、製品コード不変を確認した。','',
                 '## 7. 実用許容差と次の改善の優先順','',
                 '前回3 trialの最良膜厚誤差は3.34～10.72 nm、目標20 µmに対し0.0167～0.0536%。',
                 '仮に±0.02 µmが目標なら3 trialは既に達成、±0.01 µmならseed 0・1は達成していた。これは仮の基準であり、実製品の許容差ではない。',
                 '一意な格子最適0.00006046 µmは約0.0605 nmで、近い良条件への到達と、その1条件への完全一致を区別する。','',
                 '改善は次の順で進めるのが妥当。','',
                 '1. 最終品質の許容差と、固定判断用の目的・各制約の許容作用量を明示する。単に固定を発動させるために緩めない。',
                 '2. 1推薦でも確認実験を定期実施する配分、複数背景での対比を意図的に作る実験計画を設計する。',
                 '3. 反復測定によるノイズと、入力変動・補助モデルの誤差を分ける。シミュレーション0ノイズの検証条件も明示的に扱う。',
                 '4. 符号付き膜厚差・相互作用を扱える補助モデルと、膜厚に基づく獲得関数を検証する。',
                 '5. 弱い入力を永久に切る方針ではなく、複数入力を同時に動かす確認と全域への復帰を保ちながら一時限定する。',
                 '6. 修正後に同一初期条件・同一総予算・複数seedでON/OFFを比較し、実用許容差への到達回数と厳密格子最適への到達回数を両方記録する。','',
                 '今回は原因調査まで。v005の製品コードや既存設定には変更を加えていない。','',
                 '## 保存物','',
                 'diagnosis.json：全格子感度、固定した場合の最良値、近傍監査、許容幅再判定、回帰CV、保存予測の診断。',
                 'pair_verified.json：追加2 trialの独立再計算・ハッシュ検証済み集計。',
                 'trial_plating_focus_on_batch3_s42_20261010 / trial_plating_focus_off_batch3_s42_20261010：実験CSV、時点別予測、推薦、物理計算要求・応答、全ログ。',
                 '検証コードはこのフォルダとv005/benchmarks/electroplating_focus_*.pyに保存。','',
                 '実装の確認箇所：parameter_focus.pyの_basis（51行）、作用上限（180行）、重み保留（186行）、',
                 '許容幅チェック（190行）、固定確認（210行）。hybrid/optimizer.pyの確認枠確保（288行）。',
                 'preprocessing.pyの反復なしノイズ推定（104行）。'])
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    for source in Path(__file__).parent.glob('electroplating_focus_*.py'):
        shutil.copy2(source,OUT/source.name)
    print(json.dumps({'pair_verified':pairs,'report':str(OUT/'REPORT.md')},ensure_ascii=False))


if __name__=='__main__':
    main()
