# ParamOptimizer v005

v004のNumPy NN ensemble・残差GP・知見損失・探索領域・停止履歴を独立したsrc snapshotとして継承し、候補生成とbundleの契約を更新した版です。v004をimportしません。依存は標準ライブラリとNumPyのみです。

## 操作

プロジェクトrootから実行します。

```powershell
python main.py --version v005 --new trial_example
python main.py --version v005 --prepare trial_example
python main.py --version v005 --run trial_example --n 3
```

problem.csv、experiments.csv、knowledge_constraints.csv、search_regions.csv、stop_settings.csvの基本操作はv004と同じです。stage_connection.jsonもschema_version 1.0のまま、controlsとincoming_contextを全入力へ割り当て、current_incoming_contextへ今回の実測状態を設定します。

探索はCLIで実行し、保存した予測面・実測履歴は`plot_prediction_gui.py`で確認できます。GUIは学習を実行しません。

`--n`は推薦の**最大件数**です。既定では改善価値・予測合格確率を確認し、有用な候補だけを出します。`--n 5`でも1～4件になる場合があります。候補0件で収束の根拠が不足する場合は`REVIEW_REQUIRED`となり、推薦CSVを公開しません。

## PC負荷の制限

初期実験の自動設計、変数への配分集中、一時固定は、trialの`focus_settings.json`で個別に切り替えられます。既定はすべてONです。設定ファイルがない場合やフラグ省略時もONとなり、明示的なfalseは維持します。設定例・判断条件・停止上の制限は[ADAPTIVE_SEARCH.md](ADAPTIVE_SEARCH.md)を参照してください。

`--new`が生成するsearch_settings.jsonの既定値です。既存trialにファイルがない場合も同じ値を使用します。

```json
{
  "candidate_mode": "auto",
  "max_candidates": 4096,
  "max_training_samples": 256,
  "max_nn_epochs": 2000,
  "max_learning_seconds": 30.0,
  "seed": 42,
  "adaptive_batch": true,
  "min_recommendation_feasibility": 0.05,
  "min_information_feasibility": 0.01,
  "min_normalized_improvement": 0.001,
  "min_relative_score": 0.01,
  "max_information_candidates": 1,
  "ordinal_constraint_steps": {},
  "cpu_threads": 1,
  "distance_memory_mb": 16
}
```

- `auto`: 流入状態を直積の**前に固定**し、操作格子が予算内なら列挙、予算を超えたらLatin hypercubeで格子点を抽出します。
- `grid`: 操作格子の件数を生成前に計算し、予算超過を拒否します。
- `budget`: 格子を列挙せず、予算内の点群を生成します。
- `max_candidates`は2～100,000、`max_training_samples`は2～512。学習条件を間引く場合も最近の追加実験を保持します。元CSVは変えません。
- NNのepoch／時間予算を超えたら、そのrunの推薦を公開せず終了します。時間はNN epochとGP出力学習の境界で確認するため、進行中の一演算を強制中断するものではありません。
- 推薦・停止判定・GUI用CSVで全候補の予測を共有し、1 runの全候補予測は1回にします。GP予測と距離計算は分割し、候補×学習点×次元の巨大な距離配列や、三角行列の再分解を避けます。`distance_memory_mb`は距離計算の作業配列の目安であり、プロセス全体のメモリ上限ではありません。
- `cpu_threads`は数値計算のスレッド数で、既定1です。CLIではNumPy読込前にBLAS等の環境変数を設定します。`threadpoolctl`があれば実行中の数値ライブラリも制限・記録し、API呼出後は元に戻します。APIを既にNumPy読込済みのプロセスから使う場合は、同パッケージがないとスレッド制限の検証はできません。

生成件数、固定状態、列挙／抽出の別、学習件数と所要時間をoutput/search_budget.jsonへ保存します。抽出候補の枯渇を「全空間を探索済み」と判定しません。高次元の知見診断も最大2,048点と全実測の検査で、空間全域の保証ではありません。

## 推薦・停止判定の改善

合格実測がある場合、改善候補は既定で予測合格確率5%以上、正規化スコア0.001以上、最大スコアの1%以上を必要とします。確率1%以上5%未満では、予測の不確かさ・支持度・実測からの距離を確認し、最大1件の`INFORMATION`候補を許します。情報価値は簡易指標です。合格実測がまだない場合は、合格領域を探す`FEASIBILITY_SEARCH`を継続します。これらの予測確率は校正済みの製造良品率ではありません。

`ordinal_constraint_steps`は整数・順序尺度の制約に対する連続性補正です。例えば`{"spatter_level": 1}`では、実測合否の`<= 4`を維持し、モデルの予測確率は4.5を境界に計算します。`unit=level`で生の実測が全て整数なら刻み1を自動適用します。非整数の測定・平均された段階評価では自動適用せず、連続量は元の境界のままです。明示した刻みは自動判定より優先し、原点0を前提とします。未定義の制約列名は拒否します。

同一条件は格子軸の位置で識別し、旧CSVの10有効桁の値も一意に対応する場合だけ復元します。非常に近い異なる格子点や異なる流入状態は結合しません。新しい推薦・予測CSVの入力値は17有効桁で保存します。繰返し測定の集約と推薦済み条件の除外で同じ識別を使い、元の実測CSVは変更しません。

`stop_settings.csv`に次を追加しました。既存trialで未記載でも既定値を使います。

|設定|既定|意味|
|---|---|---|
|`coverage_scope`|`promising`|全許可格子を調べ、目的・制約の2σ予測境界から改善余地が残る領域の支持度を評価。`global`で従来の全域評価|
|`predicted_objective_tolerance`|`0.001`|予測最適目的値の変動許容幅。目的値と同じ単位で工程ごとに設定|

実測最良値の停滞、推薦スコアの低下・安定、支持度、予測最適**値**の安定を併用します。同じ目的値を持つ条件間の移動で停止が妨げられることを避けます。抽出候補だけでは自動収束を宣言せず、空の有望領域も収束の根拠にしません。モデル上の収束は離散格子の真の最適解の証明とは異なります。

|状態|扱い|
|---|---|
|`CONTINUE`|公開された推薦を実験し、実測を追記して再実行|
|`STOP_RECOMMENDED`|複数時点の履歴によりモデル上の収束を判定|
|`STOP_REQUIRED`|実験予算、全許可格子の測定完了、実測目標到達、必須知見違反等による終了|
|`REVIEW_REQUIRED`|有用な候補や候補予算が不足し、追加実験・設定の見直しが必要。収束とは扱わない|

停止時は前回の推薦を退避し、新しい実行用推薦を残しません。停止アルゴリズムの改訂・問題設定変更では収束履歴を再起算し、初期実験数を維持して実験予算を数えます。

`output/prediction_quality.json`は、前回推薦の予測と、その後追加された一致条件の実測だけを比較します。NN/Hybridの目的値MAE・予測合格確率のBrier scoreを記録し、再学習後の予測を過去の測定へ当てた評価と区別します。`adaptive_batch=false`で従来の固定件数選択に戻せます。

## 工程学習APIとbundle 2.0

連結モデルやdemoからは次の安定したAPIを使います。

```python
from pathlib import Path
from v005.src.stage_bundle import train_stage_bundle, load_stage_bundle

path = train_stage_bundle(
    Path("v005/output/stage_model"), manifest,
    inputs={"power": power, "thickness": thickness},
    outputs={"quality": quality}, seed=0, max_training_samples=256,
)
manifest, predictor = load_stage_bundle(path)
result = predictor.predict_arrays({"power": [1.0, 1.2], "thickness": 1.0})
```

manifestの必須項目は以下です。

```json
{
  "bundle_schema_version": "2.0",
  "stage_id": "stage_name",
  "controls": ["power"],
  "incoming_context": ["thickness"],
  "predicted_outputs": ["quality"],
  "input_bounds": {"power": [0.5, 2.0], "thickness": [0.8, 1.2]},
  "input_units": {"power": "kW", "thickness": "mm"},
  "output_units": {"quality": "1"},
  "local_objective": {"name": "quality", "direction": "maximize"},
  "local_constraints": [],
  "capabilities": {}
}
```

制約は`name/direction/target/margin_scale`です。無次元の単位は`"1"`とし、従来CSVの空単位は保存・読込時に`"1"`へ正規化します。入力と出力には同名の更新状態を許します。controlsとincoming_contextの間、および予測出力内では名前を重複できません。

APIの入力は名前の完全一致、有限値、broadcast可能なshapeが必要です。学習点は範囲内に限ります。予測範囲外は`valid=False/support=0/OUT_OF_BOUNDS`を返し、その予測値を推薦に使用しません。範囲内でも未経験の多変量状態ではsupportを下げます。

返り値は`outputs[name][mean/std]`、`support`、`support_class`、`valid`です。支持度は正解確率ではありません。

予測出力間の恒等関係は、任意の`derived_outputs`で保持できます。

```json
{
  "angle_error_max_deg": {"operation": "max_abs", "sources": ["first_angle_error_deg", "second_angle_error_deg"]},
  "gap_max_mm": {"operation": "max", "sources": ["gap_start_mm", "gap_center_mm", "gap_end_mm"]},
  "cold_resistance_uohm": {"operation": "sum", "sources": ["base_resistance_uohm", "interface_resistance_uohm"]}
}
```

対象とsourceは全て`predicted_outputs`に含め、単位を一致させます。宣言できる演算は`max_abs/max/sum`だけです。依存順序を検査し、循環・未知の出力・単位不一致は読込時に拒否します。対象の個別回帰値を予測後に置き換えるため、最大角度誤差が個々の角度誤差より小さくなる矛盾を避けられます。

これは予測点に適用する恒等式で、非線形変換後の厳密な期待値ではありません。派生出力の`std`はsourceの`std`の和とする未校正の代理量で、相関や変換後の分布を求めた値ではありません。返り値の`derived_output_policy`にもこの扱いを明示します。NPZ形式は変わりません。

| backend | 用途 | 継承する機能 |
|---|---|---|
| snapshot_hybrid | 通常CLIの学習・探索 | v004系の5-member学習NN、知見損失、残差GP・支持度・診断 |
| fast_stage | r002からの128／256／512点の工程再学習 | 3-memberの固定隠れ層NN＋学習readout、二次交互作用、共通RBF残差、多変量支持度 |

高速APIの隠れ層は乱数基底、readoutは正則化最小二乗で学習します。知見損失を学習する既存NNとは別のbackendで、`knowledge_loss=False`を明示します。出力間共通カーネルは使用しますが、相関を校正した同時サンプリングは実装していません。両backendとも工程ばらつき／測定ノイズ／モデル誤差の分離、校正済み良品確率を宣言しません。

高速APIのNN基底・残差RBFは、各入力の観測中心と観測標準偏差で正規化します。尺度の床は物理bounds正規化単位で0.02です。宣言範囲が広い入力でも、観測した狭い変動の感度が正則化で消えにくくなります。物理boundsによる`valid`判定と、多変量距離による支持度は別に保持し、未変動の流入状態へ外挿した点を支持したとは扱いません。学習入力`model.train_x`は物理boundsで正規化した元の座標のまま保存します。中心・尺度もNPZに保存し、これらを持たない旧fast_stage NPZは等価な座標変換で復元します。

保存はstage_manifest.jsonと数値NPZです。NPZはhashを確認し`allow_pickle=False`で読み込みます。通常CLIのNN／GPも数値重みで復元するため、v004コードやpickleの型名へ依存しません。schema 1.0はv004／r001、bundle 2.0はv005／r002の組合せで使用します。schema 1.0のpickle bundleはこのloaderでは読み込めません。

## 検証

### 予測空間を3Dで表示するGUI

`plot_prediction_gui.py`は、保存済みの予測空間CSVと実験CSVを読む単一ファイルのGUIです。
NumPy・matplotlib・Tkinterが必要です。matplotlibはCLI用の必須依存とは別に導入します。

```powershell
python -m pip install matplotlib
python v005/plot_prediction_gui.py
python v005/plot_prediction_gui.py --trial v005/trials/trial_cut_cu_20261009
```

trialまたはtrialsフォルダを選び、実験ステップ・入力X/Y・出力Z・NN/Hybridを切り替えます。
左欄には全入力を表示し、X/Y軸の変数は「X選択中」「Y選択中」と範囲を示して編集不可にします。
他の入力は左欄で固定し、カラーマップの予測断面に実測点を重ねます。
橙枠は固定入力が一致する実測、灰枠は他の入力が異なる実測です。
実測の表示を固定条件に一致する点だけへ切り替えることもできます。

`ledger.json`と`steps/`の実験スナップショットがあるtrialでは、各実験ステップに対応する履歴を使います。
通常のtrialではrunの`data_rows`に従い、現行実験CSVの先頭行を読みます（既存行が未変更の追記履歴を前提とします）。
予測更新のなかった最終ステップは、前回の予測を使用したことと学習データ数を明示します。
GUIは学習や物理計算を実行せず、trialのデータも変更しません。

```powershell
python v005/plot_prediction_gui.py --smoke-test --trial v005/trials/trial_cut_cu_20261009
```

### 工程APIとCLIの検証

```powershell
python -m unittest discover -s v005/tests -v
```

巨大な流入軸の生成前固定、24操作の有限候補予算、未経験状態での支持度低下、再学習による予測変化、hash破損の拒否、NN／GP重みと役割列順の保存一致、派生出力の恒等関係・不正定義の拒否、狭い入力変動のheldout予測・旧NPZの座標復元、実CLIのnew／prepare／runを検証します。これは合成関数と小規模CLIの検査で、製造物理や実ラインの精度検証はvalidationの別scenarioで行います。

改善実装・比較検証は`IMPROVEMENT_REPORT.md`、再実行コードは`benchmarks/efficiency_validation.py`と`benchmarks/resource_validation.py`、結果は`benchmarks/results/`に保存しています。

電気接続5工程の改善後の実探索比較は`trials/_recheck_20261010/REPORT.md`。全工程で格子最適値を維持し、総データ数620→557条件に減りましたが、自動収束は0工程で、4工程が要確認・溶接が予算到達でした。追加検証trialは`trial_recheck_*_20261010`としてGUIから選択できます。
