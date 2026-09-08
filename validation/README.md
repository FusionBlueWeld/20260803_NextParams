# 多工程の物理シミュレータと共通検証

入力条件から加工・検査結果を返す6つの決定論的シミュレータです。
各工程の `physics_model.py` が正解値を返し、共通実行器が
「条件推薦 → 仮想検査 → 実験CSVへ追記 → 再学習」を繰り返します。
最適化側には検査結果だけを渡し、正解関数や未測定点の正解値は渡しません。

## v004・r001構想への適合評価と修正方針（2026-09-07）

**既存6工程はv004のv003互換経路を検証する正解関数として継続利用できます。**
一方、6工程をそのまま直列接続してr001の成立を検証する用途には適合しません。
工程接続の主試験は[複合工程側の評価・修正方針](../multistage_validation/README.md)を使います。
以下はREADME・コードに基づく設計評価と今後の実装要件で、対応済みを意味しません。

### 形式・仕様の差と対応

| 現状・根拠 | 合わない理由 | 修正方針 |
|---|---|---|
| `simulators.py`は全入力を`parameter`としてグリッド化する | 操作条件と与えられる流入状態を区別できない | 既存CSVと標準挙動を維持し、接続試験だけ別設定・アダプターで役割を分離する。流入状態を推薦対象にしない |
| 学習CSVは`problem.csv`掲載出力のみ | 物理モデルが返す診断値がすべて接続用の学習対象になるわけではない | 接続試験では必要出力をmonitor等で明示する。既存出力の意味・列順を変えない |
| `benchmark.versions()`は実装CLIを検出し、実行は共通`main.py`経由 | v004は未実装で、共通入口にも未登録 | v004実装時に入口を登録し、既存6工程・同じseed・同じ予算で互換経路を比較する |
| 既存suiteは既定の知識・領域・停止設定による固定予算比較 | 現場の非既定設定や推奨停止の操作互換まで保証しない | 知識、禁止・優先領域、必須／推奨停止、反復、monitorを含む専用互換ケースを追加する |
| 全候補はparameterの直積で上限200,000 | 流入状態をそのまま探索軸へ追加すると規模と役割が変わる | 操作候補と流入シナリオを分離し、一括予測を分割実行する。旧グリッドは回帰基準として固定する |
| `measure()`のノイズは出力ごとの独立加法正規ノイズ | 検査ノイズであり、工程ばらつき・共通ロット変動・モデル不確実性を表さない | 現行ノイズは測定互換試験に残す。確率接続用は別の生成層に要因・相関・seed・真値／測定値を記録する |

v004の互換経路では、入力・設定を変更せず、推薦条件・順位・停止状態・CSV契約をv003と
照合します。同じ環境とseedを使い、数値許容差を定義します。既存oracleの係数、閾値、
グリッドをv004に都合よく変更しません。拡張ケースは別の版・設定・結果として保存します。

### 工程別の物理指標の適否

以下の指標は現在の局所目的・制約の検証には使用できます。ただし、下流が必要とする
材料状態と同義ではありません。単位が一致しても材料、測定基準、工程履歴の意味を確認します。

| 工程 | 接続検証での注意点と修正方針 |
|---|---|
| laser_welding | 溶込み深さと0〜9のスパッタレベルは局所品質。下流状態への対応式は未定義。接続するなら形状・材料・熱履歴等の必要状態を別途定義する。スパッタの順序尺度を連続量や故障確率に読み替えない |
| milling | 除去率・粗さ・動力は局所評価に適するが、加工後形状や残留状態を網羅しない。`tool_wear_mm`は現在parameterだが、摩耗を外部状態とする接続試験では任意設定で固定・供給する。工具選択と摩耗を自由な連続操作として混同しない |
| press_forming | バリ高さ・荷重・平面度は使用可能。上流板厚・材質などの流入状態を受け取る契約はなく、同じ名前の品質量を受け渡すだけでは工程接続にならない。対象ラインが決まった段階で入力と応答式を拡張する |
| thermal_curing | 層厚はmm、複合工程側の乾燥膜厚はum。換算だけでは残留溶剤・応力・欠陥の影響を補えない。旧モデルは互換検証用に残し、複合工程のcuringを接続用に使用する。層厚を流入状態とする役割分離試験は別設定にする |
| convection_drying | 固定した水系膜の残留水分率（湿量基準）であり、複合工程の可変膜厚・残留溶剤率とは対象が異なる。列名変更で代用せず、複合工程のdryingを使用する |
| electroplating | `thickness_error_um`は目的用の膜厚誤差で、下流へ渡す絶対膜厚ではない。既存monitorの`deposit_thickness_um`を接続候補にする。粗さ・膜厚だけで下流に十分かは別途定義し、浴状態の操作可能性も案件ごとに指定する |

これらの修正候補は、6工程すべてをr001用に作り替える要求ではありません。まず既存6工程で
操作互換を確認し、接続の学習・分布検証は専用の塗工→乾燥→熱硬化で進めます。

### 評価指標と追加する検証

- 既存のregret、MAE/RMSE、制約適合率、Brier scoreは単工程比較に維持する。
- 現行Brier scoreは決定論oracleの合否との比較であり、工程ばらつきに対する製品良品率の
  校正試験ではない。確率接続では反復シナリオの合否と照合する評価を別に追加する。
- 流入状態を変える拡張試験では、操作条件と流入状態をともに変えた学習設計を用意する。
  固定流入状態での学習から、任意状態へ使えると判断しない。
- 製品・ロット・反復の識別情報を別途保持し、同じ製品やロットが学習・検証へ混入しない
  分割を追加する。oracle真値・全グリッド情報は採点側に限定する。

物理式を変更する場合はモデル版・ハッシュ・基準結果を更新し、旧回帰基準を保存します。
この評価は内部ベンチマークとしての適合性であり、採用指標の実機校正を行ったものではありません。

## 工程の選定

選定基準は、調整可能な加工条件、数値で扱える品質検査、
計算負荷の低さ、レーザー溶接と異なる応答・制約特性です。

|フォルダ|工程・主な入力|検査出力・探索目的|検証する特性|
|---|---|---|---|
|[laser_welding](laser_welding/README.md)|レーザー出力・径・走査速度|深さ最大化、スパッタ制約|モード遷移、飽和、離散制約|
|[milling](milling/README.md)|回転数・送り・切込み・工具／摩耗など8入力|除去率最大化、粗さ・動力制約|高次元、共振による非単調性、生産性と品質の競合|
|[press_forming](press_forming/README.md)|クリアランス・速度・保持力|バリ最小化、荷重・平面度制約|適正クリアランスの内部最適、最小化|
|[thermal_curing](thermal_curing/README.md)|温度・時間・層厚|強度最大化、硬化率・劣化率制約|熱遅れ、反応速度、過処理による悪化|
|[convection_drying](convection_drying/README.md)|温度・風速・時間|エネルギー最小化、水分・欠陥制約|内部／外部移動抵抗、狭い良品領域|
|[electroplating](electroplating/README.md)|電流・浴・濃度・撹拌・電極など8入力|目標膜厚誤差最小化、粗さ・効率制約|高次元、電気量の等価条件、目標値の谷、輸送飽和|

参照した旧プロジェクトは実際には
`C:/Users/tsuts/Documents/Codex/20260623_ml_dataset_generation` にありました。
[選定根拠と参照対応](DESIGN.md)に30種類からの選び方を記録しています。
過去の教材データの乱数生成をそのまま使わず、任意の入力条件を繰り返し評価できる
関数にしています。各モデルの係数は検証用の仮定で、実機に校正した値ではありません。

## 構成

```text
validation/
  laser_welding/       各工程の physics_model.py / problem.csv / manifest.json / README.md
  milling/
  press_forming/
  thermal_curing/
  convection_drying/
  electroplating/
  simulators.py         共通API・候補グリッド・検査ノイズ
  benchmark.py          optimizer接続・履歴・比較指標
  run.py               共通CLI
  tests/               工程間の契約・旧レーザー互換・実行器のテスト
  requirements.txt
  oracle/              旧レーザー入口の互換フォルダ
  results/             自動生成する結果（Git管理外）
```

## セットアップ

Python 3.10以上とNumPyが必要です。リポジトリを任意の場所に置けます。
以下はリポジトリ直下で実行します。ローカルの旧開発プロジェクトは必要ありません。

```powershell
python -m pip install -r validation/requirements.txt
python -m validation.run list
```

`python validation/run.py ...` という起動方法も使えます。
旧レーザーのグリッド描画機能だけは任意依存としてpandasとmatplotlibが必要です。

## 入力条件を1件評価する

```powershell
python -m validation.run evaluate laser_welding --set laser_power_w=5900 spot_diameter_um=300 scan_speed_mm_s=60
python -m validation.run evaluate electroplating --set current_density_a_dm2=7 bath_temperature_c=55 plating_time_min=15
```

JSONで入力、全出力、品質制約の合否を返します。CSV列名は `list` で確認できます。
Pythonからはスカラーとブロードキャスト配列の両方を渡せます。

```python
from validation.simulators import load

simulator = load("thermal_curing")
results = simulator.evaluate({
    "oven_temperature_c": [120, 140, 160],
    "hold_time_min": 35,
    "layer_thickness_mm": 0.25,
})
print(results["bond_strength_mpa"])
print(simulator.feasible(results))
```

入力名の誤り、範囲外、NaN・Inf、出力の形状不一致を検出します。
直接モデルを使う場合は各工程の `evaluate_model()` を参照してください。
共通APIはCSVの列名に対応するキーワード引数で関数を呼びます。
引数名が異なる既存レーザーでは `manifest.json` の `input_arguments` で対応を定義し、旧APIを維持しています。

## 仮想実験CSVの生成・正解グリッドの確認

```powershell
python -m validation.run export milling --design initial --output validation/results/milling_initial
python -m validation.run export milling --design grid --output validation/results/milling_grid
python -m validation.run audit --output validation/results/oracle_audit.json
```

exportは `problem.csv`、`experiments.csv`、出典ハッシュを出力します。
通常の3入力工程の `initial` は各入力軸の下端・中央・上端の27点です。
8入力のmilling/electroplatingでは、65,536候補から正規化距離が広がるように
決定論的maximin法で64点を選びます。
`grid` は問題定義の全候補です。既存の出力フォルダは上書きしません。
auditは各工程の候補数、適合数、初期最良値、グリッド内最良値、出力範囲を出します。

## 1工程で次条件探索を検証

```powershell
python -m validation.run run milling --version v000 --iterations 15 --recommendations 3
```

工程の問題定義でtrialを作り、初期設計（3入力は27点、高次元は64点）を学習させ、各回で指定件数を推薦して
仮想実験します。バッチ内の全条件を同じ学習状態で選び、結果をまとめて追記します。
最終追記後にも再計算し、最後の予測空間を評価します。
未知の工程ID、既存trial、範囲外／グリッド外条件、重複推薦はエラーになります。

`--trial trial_my_test`、`--output validation/results/my_test` で保存先を指定できます。
省略時は一意な名前を生成します。生成trialは `vNNN/trials/`、
レポートは `validation/results/multiphysics/` 配下に保存されます。

## 6工程・複数バージョンをまとめて比較

```powershell
# 接続確認：6工程 × 4バージョン × 各2反復
python -m validation.run suite --versions v000 v001 v002 v003 --iterations 2

# 通常の反復比較：1位推薦を15回、seedを変えて3回
python -m validation.run suite --versions v000 v002 v003 --iterations 15 --seeds 0 1 2

# 工程を絞った、検査ノイズ2%・3件ずつのバッチ検証
python -m validation.run suite --simulators milling electroplating --versions v000 v003 --iterations 10 --recommendations 3 --noise 0.02 --seeds 0 1 2
```

`--seeds 0` は決定論的な基準設計です。3入力工程では既存レーザーと同じ27点、
高次元工程では64点のmaximin設計です。正のseedでは、3入力工程は27点を無作為抽出し、
高次元工程はseed付きmaximin設計を使います。いずれも重複しません。
同じ工程・seed・ノイズ条件では、各バージョンの初期条件とノイズ系列が一致します。
v003では新規trialの知識・領域・停止設定の既定値を使用し、今回の比較では
工程別の技能者知識を自動投入しません。固定回数を実行する接続・探索比較です。
収束の推奨停止（STOP_RECOMMENDED）は固定予算のため継続しますが、
必須停止（STOP_REQUIRED）は尊重し、古い推薦を再利用せず終了します。
実行回数と終了理由はsummaryに記録し、最終観測を反映していない予測空間は採点しません。

物理モデルの正解値は常に決定論的です。検査ノイズは共通実行器で
「標準偏差 = noise × 当該出力の全グリッド最大値と最小値の差」として加えます。
出力ごとに独立な加法正規ノイズで、丸め・クリップしません。そのため測定値は
物理的範囲を外れることがあり、物理モデルの正解と測定値は別々に保存します。
診断値のうちproblem.csvに載せた出力だけが学習対象です。

## 保存結果と比較の読み方

suite全体に `comparison.md`、`comparison.csv`、`comparison.json` を出力します。
個々の工程・バージョン・seedのフォルダには次を保存します。

- `problem.csv`：その回の問題定義
- `run.json`：状態、seed、ノイズ幅、モデルとoptimizerのソースハッシュ、環境
- `observations.csv`：optimizerに渡した検査値
- `truth.csv`：同じ条件の物理モデル正解値
- `history.json`：推薦・予測・正解・測定値・制約合否・最良値の推移
- `summary.json`：正規化regret、制約適合率、予測誤差、選択条件の実際の品質

正規化regretは、制約を満たす観測済み条件の最良正解値とグリッド最良値との差を、
全グリッドの目的値の幅で割ります。最大化・最小化に共通で小さいほど良好、
最良と一致すれば0、適合点がまだなければnullです。連続空間の最適値ではありません。

推薦時予測のMAE・RMSE、制約達成確率のBrier scoreを記録します。
v001〜v003は最終の全候補予測も評価し、観測済みを含む誤差と
未観測候補だけの誤差を分けます。v001はNN、v002/v003はHybridの出力です。
v000は全候補予測を保存しないため、この項目を `not_available` と明記します。

ノイズ実験では、正解値から選んだ最良条件と、実際に測定値を見て選ぶ条件の
正解値・合否を分けています。単一工程・少ない反復の優劣だけで性能を断定せず、
工程別、複数seed、同じ実験予算で比較してください。
失敗時も理由と途中履歴を残し、suiteは他の組合せを続けて最後に非ゼロ終了します。

## テストと工程追加

[実装・検証レポート](IMPLEMENTATION_REPORT.md)に実行済みの検証と基準結果をまとめています。
[正解グリッド監査](reference_audit.json)には6工程の制約付き最良値とソースハッシュを保存しています。

```powershell
python -m unittest discover -s validation -t . -p "test*.py"
```

7番目以降は `validation/<工程ID>/` に `__init__.py`、
`physics_model.py`、`problem.csv`、`manifest.json` を追加すると自動検出します。
問題定義は既存CSV形式で、入力はparameter、目的は1つ、
制約はless_equal／greater_equal、任意の診断値はmonitorです。
共通APIは入力数を固定せず扱いますが、候補総数の上限は200,000点です。

## 8入力の高次元収束テスト

フライス切削と銅電解めっきは8入力×各4水準、各65,536候補です。
前者は切削量・動力・送りマーク・歯通過振動、後者はFaraday則・物質移動・
電解液導電性を骨格にしており、乱数を含まない同一入力同一出力のモデルです。

```powershell
# 全optimizerへの短い接続確認
python -m validation.run suite --simulators milling electroplating --versions v000 v001 v002 v003 --iterations 1

# 最新版で初期64点から36条件を追加する収束試験
python -m validation.run suite --simulators milling electroplating --versions v003 --iterations 12 --recommendations 3 --seeds 0
```

比較表には入力数、初期／最終の正規化regret、改善量を出力します。
今回の実行結果は[高次元収束レポート](results/high_dimension_convergence_20260905/comparison.md)です。
再現性は決定論・固定候補・seed固定を意味し、実機に対する予測精度を意味しません。

`evaluate_model(入力列名=値, ...)` はNumPy配列の辞書を返し、
少なくともproblem.csvの全出力列を含めます。乱数・副作用を持たせず、
各出力の形状を入力のブロードキャスト形状に揃えてください。
初期設計と制約に適合／不適合条件があること、単位・物理傾向・数値精度を
工程固有テストで確認します。今回の6工程を固定確認する契約テストも更新します。

## 旧レーザー検証との互換

`validation/oracle/physics_model.py` は `laser_welding/physics_model.py` への互換入口です。
従来のコマンドも引き続き使えます。

```powershell
python validation/laser_welding_pseudo_experiment.py --optimizer-root v000 --trial trial_laser_recheck --iterations 15
```

入力グリッド13,981点、初期27点、深さ最大化、スパッタ4以下を維持しました。
グリッド最良値は8.386370654563162 mm、5900 W / 300 µm / 60 mm/sです。
歴史的な `baselines/` と `results/` はコピー元のパスを記録している場合がありますが、
現行シミュレータはそれらを実行時に読みません。これらの旧結果はGit管理外です。

## v003 の検証（収束・知識制約・予測空間）

v003では、従来のレーザー溶接反復に加え、知識制約を含む小規模な合成問題と、
最後の予測空間を確認します。`v003_validation.py` は CLI の推薦列を候補名から
検出するため、v000〜v003の出力列差を吸収します。

```powershell
py validation\v003_validation.py `
  --mode all `
  --optimizer-root v003 `
  --trial trial_laser_welding_v003_validation `
  --synthetic-trial trial_synthetic_constraints_v003 `
  --iterations 15
```

保存先は `validation/results/v003/` です。

- `*.json`: 機械比較用の全履歴と知識制約違反率
- `*_trajectory.csv`: 反復ごとの実測値・推薦スコア
- `*.md`: 人間／LLM向けの短い要約
- `*.png`: 収束曲線、推薦スコア、最終予測断面、データ支持度

matplotlibがない環境では、数値検証は継続し、PNGだけ`SKIP`として扱います。
個別にグラフを再生成する場合は次を実行します。

```powershell
py validation\plot_v003_validation.py validation\results\v003\trial_laser_welding_v003_validation.json
```

合成問題の学習データは125候補すべてではなく、角点＋数点だけです。未測定候補を
残すことで、`--run` の推薦処理と response-space の制約検査を同時に確認できます。
