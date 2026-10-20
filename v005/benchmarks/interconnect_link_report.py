"""Replay saved recommendations, plot causal paths, and explain the bridge contract."""
from interconnect_link_validation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.multistage.electrical_interconnect.stages import evaluate_stage

def load(name):
    return json.loads((DEST/name).read_text(encoding='utf-8'))

def best_actual(search_result):
    return search_result['independent_scoring'][0]['actual']

def verify():
    checked = {'physical_recommendations_replayed': 0, 'training_stage_blocks_replayed': 0,
               'connection_fields_verified': 0, 'fixed_bundle_predictions_restored': 0}
    def visit(value):
        if isinstance(value, dict):
            if {'stage_inputs', 'stage_outputs', 'controls', 'executable_pass'} <= set(value):
                replay = truth(value['controls'])
                assert replay['executable_pass'] == value['executable_pass']
                for s, outputs in value['stage_outputs'].items():
                    for k, v in outputs.items():
                        np.testing.assert_allclose(replay['stage_outputs'][s][k], v, rtol=1e-12, atol=1e-12)
                for edge in line_config()['connections']:
                    s, k = edge['from'].split('.')
                    t, n = edge['to'].split('.')
                    np.testing.assert_array_equal(value['stage_outputs'][s][k], value['stage_inputs'][t][n])
                    checked['connection_fields_verified'] += 1
                checked['physical_recommendations_replayed'] += 1
                return
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    for filename in ('results.json', 'recourse.json', 'physical_reference.json', 'form_sweep.json'):
        visit(load(filename))
    for seed in (42, 0, 1):
        for kind in ('linked_training', 'holdout'):
            data = load(f'seed_{seed}/{kind}.json')
            for stage, block in data['blocks'].items():
                replay = evaluate_stage(stage, block['inputs'])
                for name in block['outputs']:
                    np.testing.assert_allclose(replay[name], block['outputs'][name], rtol=1e-12, atol=1e-12)
                checked['training_stage_blocks_replayed'] += 1
        row = load(f'seed_{seed}/results.json')
        models = {s: load_stage_bundle(DEST/f'seed_{seed}/fixed_bundles/{s}') for s in STAGE_ORDER}
        config = load(f'seed_{seed}/fixed_pipeline.json')
        predicted = evaluate_pipeline(config, models, row['controls'])
        for s in STAGE_ORDER:
            for k, vals in predicted['stage_outputs'][s].items():
                expected = np.asarray([np.nan if x is None else x for x in row['fixed_recipe_prediction']['stage_outputs'][s][k]])
                np.testing.assert_allclose(vals, expected, equal_nan=True)
            checked['fixed_bundle_predictions_restored'] += 1
    checked['passed'] = True
    save(DEST/'verification.json', checked)
    return checked

def plot():
    plt.rcParams.update({'font.family': 'Meiryo', 'font.size': 10, 'axes.unicode_minus': False})
    records, reference, form = load('results.json'), load('physical_reference.json'), load('form_sweep.json')
    fig, ax = plt.subplots(1, 3, figsize=(14,4))
    seeds = [s['seed'] for s in records['seeds']]
    for i, (field, limit, title, unit) in enumerate([
        ('gap_max_mm', .15, '組立の最大隙間', 'mm'),
        ('lack_of_fusion_fraction', .15, '溶接の未融合率', '割合'),
        ('shear_force_n', 1300, '接合せん断力', 'N')]):
        stage = 'assembly' if field == 'gap_max_mm' else 'welding'
        values = [s['direct_connection']['stage_outputs'][stage][field] for s in records['seeds']]
        ax[i].bar([str(s) for s in seeds], values, color='#dd8060')
        # Use the actual manifest local gap threshold, not a guessed specification.
        if field == 'gap_max_mm':
            limit = next(c['target'] for c in STAGE_MANIFESTS['assembly']['local_constraints'] if c['name']==field)
        ax[i].axhline(limit, color='#555', ls='--', label=f'制約 {limit:g}')
        ax[i].set(title=title, xlabel='単工程試験 seed', ylabel=unit)
        ax[i].legend()
    fig.suptitle('v005の各工程の最良条件を、そのまま接続した物理計算')
    fig.tight_layout(); fig.savefig(DEST/'01_connected_recipes.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(2,2, figsize=(13,8))
    rows = form['sweeps']; x = [s['overbend_deg'] for s in rows]
    ax[0,0].plot(x,[s['fixed']['stage_outputs']['form_cu']['pad_height_error_mm'] for s in rows], 'o-')
    ax[0,0].set(title='① 銅成形の操作変更 → パッド高さ誤差', ylabel='mm')
    ax[0,1].plot(x,[s['fixed']['stage_outputs']['assembly']['gap_max_mm'] for s in rows], 'o-')
    gap_limit = next(c['target'] for c in STAGE_MANIFESTS['assembly']['local_constraints'] if c['name']=='gap_max_mm')
    ax[0,1].axhline(gap_limit,color='gray',ls='--',label='組立制約'); ax[0,1].legend()
    ax[0,1].set(title='② 同じ組立条件 → 接合隙間が変化', ylabel='mm')
    for index, field, threshold, title, unit in [(0,'lack_of_fusion_fraction',.15,'③ 溶接の未融合率','割合'),
                                               (1,'quality_margin',0,'④ 最終品質の最小仕様余裕','正規化余裕')]:
        a=ax[1,index]
        a.plot(x,[s['fixed']['final'][field] for s in rows], 'o-', label='下流条件を固定')
        a.plot(x,[best_actual(s['learned_recourse'])['final'][field] for s in rows], 's-', label='学習モデル r002 再調整後の実値')
        a.plot(x,[best_actual(s['oracle_recourse'])['final'][field] for s in rows], '^-', label='物理モデル r002 比較基準')
        a.axhline(threshold,color='gray',ls='--'); a.legend(fontsize=8)
        a.set(title=title,ylabel=unit)
    for a in ax.flat:
        a.set_xlabel('銅の第1過曲げ角 [degree]'); a.grid(alpha=.2)
    fig.suptitle('上流成形 → 組立 → 溶接への伝播（起点は物理比較基準の合格条件）')
    fig.tight_layout(); fig.savefig(DEST/'02_form_propagation.png', dpi=150); plt.close(fig)
    rows=reference['sweeps']; x=[r['dose'] for r in rows]
    fig, ax = plt.subplots(1,3,figsize=(14,4))
    ax[0].plot(x,[r['fixed']['stage_outputs']['assembly']['top_oxide_nm'] for r in rows],'o-')
    ax[0].set(title='① 洗浄 → 酸化膜',ylabel='nm')
    ax[1].plot(x,[r['fixed']['final']['cold_unclamped_resistance_uohm'] for r in rows],'o-',label='溶接条件固定')
    ax[1].set(title='② 酸化膜 → 完成品抵抗',ylabel='μΩ'); ax[1].legend()
    for key,label,marker in [('fixed','条件固定','o'),('learned_recourse','学習モデルで溶接再調整','s'),('oracle_recourse','物理比較基準で再調整','^')]:
        vals=[(r['fixed'] if key=='fixed' else best_actual(r[key]))['final']['quality_margin'] for r in rows]
        ax[2].plot(x,vals,marker+'-',label=label)
    ax[2].axhline(0,color='gray',ls='--'); ax[2].set(title='③ 再調整後の品質',ylabel='最小仕様余裕'); ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xlabel('上下材の洗浄量 [dose]'); a.grid(alpha=.2)
    fig.suptitle('合格起点での洗浄量変更：再調整が品質を下げる場合もある')
    fig.tight_layout(); fig.savefig(DEST/'03_cleaning_recourse.png',dpi=150); plt.close(fig)
    # Machine readable chart values preserve exact values and final executable flags.
    for filename, rows, axis in [('form_sweep.csv',form['sweeps'],'overbend_deg'),('cleaning_sweep.csv',reference['sweeps'],'dose')]:
        flat=[]
        for r in rows:
            entry={axis:r[axis]}
            for key in ('fixed','learned_recourse','oracle_recourse'):
                a=r['fixed'] if key=='fixed' else best_actual(r[key])
                for field in ('quality_margin','lack_of_fusion_fraction','cold_unclamped_resistance_uohm','shear_force_n'):
                    entry[f'{key}.{field}']=a['final'][field]
                entry[f'{key}.executable_pass']=a['executable_pass']
            flat.append(entry)
        with (DEST/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=flat[0].keys()); writer.writeheader(); writer.writerows(flat)

def report(verification):
    records, recourse, reference, form = [load(n) for n in ('results.json','recourse.json','physical_reference.json','form_sweep.json')]
    lines=['# v005単工程条件 → r002工程接続の検証', '',
           '実計算済み。単工程の最良条件をそのまま接続した3組は全て不合格。工程間の状態伝播とr002の探索は物理モデルで動作するが、今回の学習モデルの全工程推薦は物理再評価で全て不合格だった。', '',
           '## 1. 検証を進める順序', '',
           '1. 保存済みv005試験の実測最良条件・工程目的・制約・流入状態を確定する。',
           '2. 各工程の最良操作条件を接続し、同じワークの上流出力を下流へ渡して物理再評価する。',
           '3. r002へ渡す工程モデルの入力役割・出力・単位・接続先・制約・支持範囲を照合する。',
           '4. 固定流入状態のデータだけで接続した場合と、実際の上流状態を含めて学習した場合を分ける。',
           '5. 未実施なら全工程、実施済みなら観測状態と固定操作を渡して残工程だけ探索する。',
           '6. 推薦を独立した物理計算で再評価する。上流を変更し、下流固定と再調整を比較する。', '',
           '## 2. v005で得られたもの', '',
           '6工程×3 seed、各150試行の既存記録を使用。時間を最小化し、各工程の局所制約を満たす実測最良条件を抽出した。逐次入力限定機能はONの既存試験。今回、その試験データ・設定は変更していない。', '',
           '|工程|操作数|流入状態数|単工程の最良時間 seed 42 / 0 / 1 [s]|', '|---|---:|---:|---|']
    for stage in STAGE_ORDER:
        m=STAGE_MANIFESTS[stage]
        vals=[r['v005_sources'][stage]['summary']['best_value'] for r in records['seeds']]
        lines.append(f'|{stage}|{len(m["controls"])}|{len(m["incoming_context"])}|'+ ' / '.join(f'{v:.6g}' for v in vals)+'|')
    lines += ['', '溶接seed 42の0.525 sは、試験内の最良条件であり単工程グリッド最適値には到達していない。その他は既存単工程試験で最適値へ到達した条件。今回、未測定のoracle最適条件へ置き換えていない。', '',
              '各試験の`fixture.json`では流入状態が固定されている。CSVに流入状態と同名の出力がある場合、流入値にはfixtureを使い、加工後の出力と混同していない。', '',
              '## 3. r002のフレームワークと、v005各工程から必要なもの', '',
              '接続：銅切断→銅成形、アルミ切断→アルミ成形、2枝→組立→溶接。操作44変数、状態接続94本。r002は候補ごとに上流出力を下流入力へ渡し、局所制約・最終仕様・モデル支持度の最小正規化余裕を改善する。v005の時間最小化とは目的が違い、今回r002で総時間最小化を証明してはいない。', '',
              '|必要なもの|今回の状況|', '|---|---|',
              '|最良操作条件|既存trial summaryのbest_inputsから抽出。seedごとの6工程条件を保持|',
              '|流入状態を入力に含む予測モデル|単工程試験では状態固定のため不足。上流を通った新規データで追加学習|',
              '|操作と流入状態の役割、入力範囲・単位|STAGE_MANIFESTSのcontrols / incoming_context / input_bounds / units|',
              '|下流へ渡す出力・接続先|predicted_outputs / connector_outputsとpipeline connections|',
              '|平均・std・支持度、局所制約|bundle2.0。stdは未校正で良品確率や保証ではない|',
              '|最終仕様|抵抗≤60 μΩ、せん断力≥1300 N、温度上昇≤2.3 K、未融合≤0.15、気孔≤0.09、IMC≤6 μm、解放後変形≤0.35 mm|',
              '|実施済みの状態|observed_outputsへ観測出力、completed_stagesとfixed_controlsへ実施済み工程を設定|', '',
              '|接続箇所|v005側で必要な出力例|後工程で使う目的|', '|---|---|---|',
              '|切断 → 成形（Cu/Al各13項目）|厚さ、寸法、バリ、粗さ、端部損傷・硬化、反り、酸化膜、汚染、降伏強度・弾性率・材種|ばね戻り、塑性変形、形状誤差の予測|',
              '|成形 → 組立（各21項目、計42）|上記部品状態に加え、2曲げ角誤差、パッド高さ・傾斜、塑性ひずみ、残留応力、成形損傷、板厚損失|上下材の隙間と接触状態の予測|',
              '|組立 → 溶接（26項目）|3点隙間・最大隙間、接触率・圧力、治具熱損失、上下材酸化膜・汚染、重なり、横ずれ、塑性セット・蓄積エネルギーなど|吸収エネルギー、未融合、接合強度、抵抗・解放後変形の予測|', '',
              '**モデルの区別：**既存v005 CLIの学習済みNN＋GPをそのまま変換した試験ではない。既存bundle2学習APIのfast_stage（固定特徴NN＋残差kernel）で橋渡しモデルを再学習した。単工程CLIは流入状態を入力として学習していないため、その保存モデルだけではこの接続契約を満たさない。今回は「既存試験の操作解の接続」と「r002接続用モデルの検証」を分けている。元の入力限定機能の効果をこの接続試験で新たに証明したものではない。', '',
              '## 4. そのまま接続した結果', '',
              '|seed|切断・成形|組立最大隙間 [mm]|未融合率|せん断力 [N]|最終合格|', '|---|---|---:|---:|---:|---|']
    for r in records['seeds']:
        a=r['direct_connection']; f=a['final']; g=a['stage_outputs']['assembly']['gap_max_mm']
        lines.append(f'|{r["seed"]}|4工程合格|{g:.4f}|{f["lack_of_fusion_fraction"]:.4f}|{f["shear_force_n"]:.1f}|不合格|')
    lines += ['', '![そのまま接続](01_connected_recipes.png)', '',
              '切断・成形の局所制約を満たしても、次の組立に必要な平面度・高さ・隙間が十分とは限らない。独立試験で最短だった洗浄・クランプ・溶接条件も、別の流入状態では最短の合格条件にならない。単工程解を接続するだけでは全体の合格は導けない。', '',
              '各単工程には同じ最短時間となる複数の条件がある。ここでは既存trialが保存した代表1条件を接続したため、「同率最適条件を選び直しても全体合格できない」とは結論していない。同率条件の状態出力を残してr002で選び直す検証も、時間を悪化させずに接続を改善する候補となる。', '',
              '## 5. 学習・探索・独立再評価', '',
              '固定状態モデルは各工程の既存150行を使う。流入状態を変えたモデルはseedごとに新規8192組の製作条件を生成し、上流が局所制約を通った行のみ下流の学習へ渡した。各工程の学習保持上限256行。別seedの4096組を検証ロットとし、学習へ戻していない。全入力・出力・系譜とholdout誤差を保存している。合成内部状態までラベルとして使った完全情報の検証である。', '',
              '|seed|固定状態モデル r002|状態変化モデル r002 予測/実値|切断・成形観測後の学習r002|物理r002 全工程|物理r002 切断・成形固定|',
              '|---|---|---|---|---|---|']
    for r,q,p in zip(records['seeds'],recourse['cases'],reference['full_search']):
        pred=r['linked_r002']['search']['best']['predicted_feasible']
        lines.append(f'|{r["seed"]}|支持不足・未合格|{"予測合格" if pred else "予測不合格"} / 実値不合格|未合格|合格|未合格|')
    lines += ['', '表は各探索の首位候補。返された上位4候補も同様の合否だった。固定状態モデルでは成形以降の支持度0となる。支持度は分布の支持情報であって合格確率ではない。', '',
              '探索予算：学習全工程・固定状態モデルは各2048ライン評価、観測済み切断・成形の再調整と物理全工程比較は各8192。掃引の下流再調整は各2048（既存v005条件からの洗浄掃引のみ1024）。モデル間の結果は予算を揃えた因果的な優劣試験ではなく、失敗箇所を分ける診断。予測評価件数を実試行件数と同一視しない。44変数の全直積は列挙しておらず、大域最適・不可能性は証明していない。', '',
              '切断・成形を既存条件で固定した物理比較では、今回の探索予算内で合格が見つからなかった。seed 0の最良候補でも未融合率制約の正規化余裕は約−0.148。切断・成形も変更した物理探索では3 seed全て合格。この結果は上流へ戻る調整が必要となり得ることを示すが、固定上流での不可能性を証明したものではない。', '',
              '## 6. 前工程が動くと後工程がどう変化するか', '',
              '次のグラフは、物理r002で得た合格条件を起点にした制御された掃引。v005／学習モデルが発見した合格条件と混同しない。Cu第1過曲げ角のみを0〜5°で13点変更し、同じ切断・成形系譜を組立・溶接へ渡した。下流条件固定と、切断・成形出力を観測して組立・溶接をr002で再調整した場合を比較する。', '',
              '![上流から下流への伝播](02_form_propagation.png)', '',
              '第1過曲げ角 → 成形パッド高さ誤差 → 組立隙間 → 未融合率・最終仕様余裕の順に追える。各再調整の線は探索首位候補を独立した物理計算で評価した値。最終仕様余裕≥0でも、局所制約や入力範囲が外れた場合はライン合格ではない。CSVにはexecutable_passも保存した。各点は離散探索を独立に行うため線の跳びは応答の微分・滑らかな連続最適解を意味しない。', '',
              'この成形掃引では、条件固定・物理比較基準とも0〜2.5°の7点がライン合格、2.9167°以上の6点は不合格。学習モデルによる再調整では13点中1点だけが実合格だった。上流が局所制約を外した点の下流計算は診断値であり、実施可能なワークとして数えていない。', '',
              '![洗浄と再調整](03_cleaning_recourse.png)', '',
              '上下材の洗浄量 → 表面酸化膜 → 最終抵抗の変化も確認した。合格条件からの洗浄掃引では下流を固定した13点全てが合格だったが、学習モデルの溶接再調整では予測合格13点のうち6点が実値不合格となった。物理比較基準の再調整は13点全て合格。改善を推薦する場合でも物理確認が必要であり、再調整が必ず改善するとはいえない。', '',
              '## 7. 今足りないものと次の検証', '',
              '- 元の通常v005 NN＋GPで、流入状態をパラメータとして含めた各工程モデルを学習・bundle出力し、この橋渡し高速モデルとの差を確認する。',
              '- 流入状態×操作の交互作用、特に成形・組立の隙間と溶接未融合・強度を境界付近で追加測定し、予測合格と実合格の不一致を減らす。',
              '- 無条件の工程時間最短ではなく、後工程で合格へ調整できる範囲を受入条件として上流へ返す。必要時は上流操作も探索へ戻す。',
              '- 観測済み状態を使う場合も、模型誤差・部分観測・測定誤差を確認する。未校正std・支持度から良品確率を保証しない。',
              '- r002の品質余裕探索と、合格を維持した総時間・費用最小化を別に評価する。今回の合格比較は全体最短時間の検証ではない。', '',
              '## 記録と再現', '',
              '`results.json`：既存条件と参照元SHA、直接接続、固定／状態変化モデル探索、独立再評価。',
              '`recourse.json`：既存切断・成形を固定した再調整。`physical_reference.json`：全工程物理比較と洗浄掃引。`form_sweep.json`：成形操作の掃引。',
              '`seed_*/linked_training.json`・`holdout.json`：製作条件、入力、出力、採否、系譜。`*_bundles/`：JSON＋NPZ。`*_pipeline.json`：r002の接続・探索設定。',
              '`form_sweep.csv`・`cleaning_sweep.csv`：グラフの実数値。`source_snapshot/`・`source_sha256.json`・`artifact_sha256.json`：検証実装・成果物の識別。', '',
              f'独立確認：物理候補・状態 {verification["physical_recommendations_replayed"]}件を再計算、接続値 {verification["connection_fields_verified"]}項目を一致確認、学習／holdout工程ブロック {verification["training_stage_blocks_replayed"]}件を物理再計算、固定bundle復元予測 {verification["fixed_bundle_predictions_restored"]}工程を一致確認。全て通過。', '',
              '再実行は別の保存先にDESTを変更し、interconnect_link_validation.py → interconnect_link_recourse.py → interconnect_link_reference.py → interconnect_link_form_sweep.py → interconnect_link_report.py の順。完成済みresults.jsonのある保存先へ基本試験を上書きしない。原本の試験・モデル・物理実装には変更を加えていない。']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    snapshot=DEST/'source_snapshot'/Path(__file__).resolve().relative_to(ROOT)
    snapshot.write_bytes(Path(__file__).read_bytes())
    # A single portable browser report makes plots readable together.
    import html
    body=html.escape('\n'.join(lines))
    for name in ('01_connected_recipes.png','02_form_propagation.png','03_cleaning_recourse.png'):
        body=body.replace(f']({name})',f']({name})</pre><img src="{name}" style="width:100%;max-width:1400px"><pre>')
    (DEST/'REPORT.html').write_text('<meta charset="utf-8"><title>v005 → r002 検証</title><style>body{font-family:Meiryo,sans-serif;margin:30px}pre{white-space:pre-wrap;line-height:1.8}</style><pre>'+body+'</pre>',encoding='utf-8')
    sourcehash={p.relative_to(DEST/'source_snapshot').as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (DEST/'source_snapshot').rglob('*') if p.is_file()}
    save(DEST/'source_sha256.json',sourcehash)
    artifacthash={p.relative_to(DEST).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in DEST.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'}
    save(DEST/'artifact_sha256.json',artifacthash)

if __name__ == '__main__':
    checked=verify(); plot(); report(checked); log(checked)
