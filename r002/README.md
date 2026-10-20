# r002：予算付きの工程接続探索

次期rの構想は [R_SERIES_CONCEPT_20261011.md](R_SERIES_CONCEPT_20261011.md) に記録しています。量産の固定条件・センシング・ワーク状態による後工程適合の更新を扱う構想で、現行r002の実装仕様とは区別しています。

各工程の学習済みbundle2.0を分岐合流DAGで接続し、少数の条件組だけを評価します。
上流が変わった結果は下流入力へ毎回伝播します。全操作条件の直積・全頂点は生成しません。

```powershell
python main.py --version r002 --new trial_line
python main.py --version r002 --prepare trial_line
python main.py --version r002 --run trial_line --n 3
```

`pipeline.json`の`stages`・`connections`・`external_context`・`candidate_axes`・
`final_specifications`を記入します。操作のキーは`stage_id.control`です。
最終仕様は任意の工程出力を`stage`で指定できます。省略時は最後の工程です。
工程内の必須制約は常に判定し、単位不一致・循環・入力供給元の重複を拒否します。

`search`に`max_evaluations`・`population_size`・`seed`・`max_seconds`・
`support_threshold`・`batch_size`を指定します。既定値は4096評価、64条件の集団、
120秒、支持度0.5、64行のバッチです。時間制限は予測バッチの前に確認するため、
実行中の1バッチ分だけ上限を超える場合があります。キャッシュは既定4096点です。
NumPyによるLHS初期候補・差分進化・工程単位と全工程同時変更・局所提案を併用します。
`continuous_controls`へ指定した操作は候補軸の最小値と最大値の間を連続探索できます。

条件選択は全工程制約・最終仕様・支持度の最小正規化余裕です。
`search.conservative_std_multiplier`を正にすると、各局所／最終制約の平均余裕から
`係数×予測std／余裕の尺度`を引いて推薦・合否を比較します。既定値は0です。
平均の`process_margin`と`conservative_process_margin`を分けて保存し、適用時は
`uncertainty_policy=uncalibrated_heuristic`と表示します。stdは未校正なので、
この減算は信頼区間・良品確率・安全保証ではありません。実測された出力平均はstd0として
扱いますが、測定誤差が0であることを意味しません。
未合格候補も情報不足の診断用に返しますが、`predicted_feasible=false`で表示します。
探索の未発見は不可能の証明ではなく、大域最適も保証しません。

実施済み工程は`completed_stages`または`observed_outputs`で指定します。
実施済み工程とその祖先の全操作を`fixed_controls`へ指定してください。
観測出力のキーは`stage_id.output`です。全出力を観測した工程は予測で置き換えません。
部分観測の場合は未測定出力をモデルで推定します。測定平均を接続し、測定ノイズを
物理ばらつきへ混入しません。ただし本版は測定ノイズを用いた潜在状態推定器ではありません。

`inventory`は`workpiece_id`・`stage_id`・`outputs`・`lot_id`・`available`を持ちます。
合流には`assembly_pair_id`が同じで各供給元に1個ずつ存在する組だけを使い、
任意のワークの直積を作りません。原料在庫は`target_stage_id`・`incoming_context`を指定します。
`additional_experiments`で工程別`costs`と実験／提案／予測予算を指定できます。
追加実験は支持不足・出力std・制約境界への近さを費用で調整した代理指標で選びます。
厳密な情報価値ではありません。破壊する同一ワークを複数案へ重複割当しません。
提案は実行や在庫消費を行いません。

加工で状態が変わるワークは、非破壊検査であっても複数条件へ割り当てません。
`costs.<stage>.consumes_workpiece`は既定`true`で、検査の`destructive`と別です。
状態を変えない測定だけを繰り返せる工程に限り`false`にできます。

`process_variation`は候補数・標本数・予測予算を限定して入力ばらつきを再評価します。
`standard_deviations`で操作／外部流入状態、`shared_factors[].loadings`で共通ロット因子を
指定できます。範囲外の乱数点はクリップせず評価不能です。出力は
`UNCALIBRATED_INPUT_VARIATION_ESTIMATE`で、モデル誤差・同時出力分布・検査ノイズを
含みません。Wilson区間はMonte Carlo標本誤差だけです。

## 検証・統合API

```python
from r002.src.api import run_search, evaluate_acceptance, propose_additional_experiments
from r002.src.pipeline import evaluate_pipeline

# predictorsを省略するとv005.src.stage_bundle.load_stage_bundleで読みます。
predictors = {"cutting": (manifest, predictor)}
result = run_search(config, predictors=predictors)
evaluation = evaluate_pipeline(config, predictors, {"cutting.speed": values})
```

`result`は`best`・`candidates`・`history`・`evaluation_count`・`stop_reason`を持ちます。
候補は`controls`・`predicted_feasible`・`trusted_margin`・`process_margin`・
`minimum_stage_support`を持ちます。評価器は`stage_inputs`・`stage_outputs`・
`stage_support`・制約別余裕・接続の有効性を返します。

`evaluate_acceptance(config, state_overrides, predictors, budget=512)`は少数の上流状態で
残り工程の調整を再探索します。`state_overrides`は`outputs`・`fixed_controls`・
`reachable`・`workpiece_id`を持つ辞書の列です。固定操作は`reference_controls`でも供給できます。
全条件の`reference_controls`を渡した場合も、状態を指定した工程と祖先の操作だけを固定します。
探索器に渡す仮想状態を、実験に使える在庫と同一視しません。受入情報は局所・予算付きで、
全入力空間の厳密な合格領域ではありません。

```powershell
python -m unittest discover -s r002/tests -v
```

旧bundle1.0は専用pickle名前空間と異なるため直接読みません。v004/r001は変更せず、
新しい学習・出力契約はv005のbundle2.0を用います。
