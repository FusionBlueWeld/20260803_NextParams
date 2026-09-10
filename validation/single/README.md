# 単工程の物理シミュレータと共通検証

入力条件から加工・検査結果を返す6つの決定論的シミュレータです。
各工程の `physics_model.py` が正解値を返し、共通実行器が
「条件推薦 → 仮想検査 → 実験CSVへ追記 → 再学習」を繰り返します。
最適化側には検査結果だけを渡し、正解関数や未測定点の正解値は渡しません。

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

## 構成と各ファイルの役割

v002と同じく、入口・設定・読込・計算・出力を分けています。
物理モデルは工程名のフォルダにまとめ、モデルの説明と定義を隣に置きます。

```text
validation/single/
├─ README.md / DESIGN.md / requirements.txt
├─ run.py                    起動入口
├─ src/
│  ├─ cli.py                 引数の解釈と処理の振り分け
│  ├─ settings.py            パス・候補数上限・Variable（問題定義の1行）
│  ├─ data_loader.py         CSV・JSONの読書き
│  ├─ simulators.py          工程読込・候補生成・正解評価・合否判定
│  ├─ benchmark.py           optimizerとの反復実験
│  ├─ metrics.py             予測誤差・regret・全候補の採点
│  ├─ reporting.py           比較表のCSV・JSON・Markdown出力
│  └─ checks/                用途が限定された追加検証
│     ├─ laser_welding_pseudo_experiment.py
│     ├─ v003_validation.py   レーザー検証とモード選択
│     ├─ v003_synthetic.py    合成データによる知識ルール検証
│     ├─ v003_runtime.py      追加検証共通のCSV契約・CLI呼出し
│     └─ plot_v003_validation.py
├─ laser_welding/            物理モデル・問題CSV・manifest・説明
├─ milling/                  同上。以下も工程ごとに独立
├─ press_forming/
├─ thermal_curing/
├─ convection_drying/
├─ electroplating/
├─ oracle/                   旧レーザー物理モデルへの互換入口
├─ tests/                    共通実行器と追加検証のテスト
└─ results/                  生成結果（Git管理外）
```

工程固有の `test_model.py` は、そのモデルと定義CSVの隣に残しています。
共通処理を直す場合は `tests/`、物理式を直す場合は工程側のテストを確認します。

| v002での役割 | 単工程検証で読む場所 |
|---|---|
| `src/cli.py` | [src/cli.py](src/cli.py)：引数から担当処理へ渡す |
| `src/settings.py` | [src/settings.py](src/settings.py)：パス・定数・変数定義 |
| `src/data_loader.py` | [src/data_loader.py](src/data_loader.py)：読書き |
| `src/validation.py`・`parameter_space.py` | [src/simulators.py](src/simulators.py)：定義検査と候補生成 |
| `hybrid/optimizer.py` | [src/benchmark.py](src/benchmark.py)：推薦と測定の反復を制御 |
| `hybrid/reporting.py` | [src/metrics.py](src/metrics.py)で採点し、[src/reporting.py](src/reporting.py)で比較表を保存 |

### 処理の読み順

1. `run.py` → `src/cli.py` で利用者が指定できる操作を確認します。
2. `src/settings.py` → `src/data_loader.py` → `src/simulators.py` で入力とモデルの契約を確認します。
3. 任意の工程の `problem.csv` → `manifest.json` → `physics_model.py` で具体例を追います。
4. `src/benchmark.py` → `src/metrics.py` → `src/reporting.py` で反復・採点・保存を追います。

```text
初期条件 → 物理モデルの真値 → 測定ノイズを加える → experiments.csv
                                                    ↓
                                    main.py → 指定版のoptimizer
                                                    ↓
                                推薦条件 → 仮想測定 → CSVへ追記
```

真値は採点用、ノイズ付きの値はoptimizerに渡す測定値です。
全候補の正解をoptimizerへ渡すことはありません。`regret` は候補グリッド上の最良値と
観測済み最良値の差を出力幅で割った値で、小さいほど良好です。

### 追加検証の位置付け

`src/checks/` は通常の6工程比較から分けて読む検証シナリオです。
レーザー専用の既存比較、v003の知識ルール検証、PNG作成を担当します。
共通処理の追加先は `src/`、特定用途の追加先は `src/checks/` とします。
過去の適合評価・未実装表記は [DESIGN.md](DESIGN.md) に履歴として残しています。

## セットアップ

Python 3.10以上とNumPyが必要です。リポジトリを任意の場所に置けます。
以下はリポジトリ直下で実行します。ローカルの旧開発プロジェクトは必要ありません。

```powershell
python -m pip install -r validation/single/requirements.txt
python -m validation.single.run list
```

`python validation/single/run.py ...` という起動方法も使えます。
旧レーザーのグリッド描画機能だけは任意依存としてpandasとmatplotlibが必要です。

## 入力条件を1件評価する

```powershell
python -m validation.single.run evaluate laser_welding --set laser_power_w=5900 spot_diameter_um=300 scan_speed_mm_s=60
python -m validation.single.run evaluate electroplating --set current_density_a_dm2=7 bath_temperature_c=55 plating_time_min=15
```

JSONで入力、全出力、品質制約の合否を返します。CSV列名は `list` で確認できます。
Pythonからはスカラーとブロードキャスト配列の両方を渡せます。

```python
from validation.single.src.simulators import load

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
python -m validation.single.run export milling --design initial --output validation/single/results/milling_initial
python -m validation.single.run export milling --design grid --output validation/single/results/milling_grid
python -m validation.single.run audit --output validation/single/results/oracle_audit.json
```

exportは `problem.csv`、`experiments.csv`、出典ハッシュを出力します。
通常の3入力工程の `initial` は各入力軸の下端・中央・上端の27点です。
8入力のmilling/electroplatingでは、65,536候補から正規化距離が広がるように
決定論的maximin法で64点を選びます。
`grid` は問題定義の全候補です。既存の出力フォルダは上書きしません。
auditは各工程の候補数、適合数、初期最良値、グリッド内最良値、出力範囲を出します。

## 1工程で次条件探索を検証

```powershell
python -m validation.single.run run milling --version v000 --iterations 15 --recommendations 3
```

工程の問題定義でtrialを作り、初期設計（3入力は27点、高次元は64点）を学習させ、各回で指定件数を推薦して
仮想実験します。バッチ内の全条件を同じ学習状態で選び、結果をまとめて追記します。
最終追記後にも再計算し、最後の予測空間を評価します。
未知の工程ID、既存trial、範囲外／グリッド外条件、重複推薦はエラーになります。

`--trial trial_my_test`、`--output validation/single/results/my_test` で保存先を指定できます。
省略時は一意な名前を生成します。生成trialは `vNNN/trials/`、
レポートは `validation/single/results/multiphysics/` 配下に保存されます。

## 6工程・複数バージョンをまとめて比較

```powershell
# 接続確認：6工程 × 4バージョン × 各2反復
python -m validation.single.run suite --versions v000 v001 v002 v003 --iterations 2

# 通常の反復比較：1位推薦を15回、seedを変えて3回
python -m validation.single.run suite --versions v000 v002 v003 --iterations 15 --seeds 0 1 2

# 工程を絞った、検査ノイズ2%・3件ずつのバッチ検証
python -m validation.single.run suite --simulators milling electroplating --versions v000 v003 --iterations 10 --recommendations 3 --noise 0.02 --seeds 0 1 2
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

7番目以降は `validation/single/<工程ID>/` に `__init__.py`、
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
python -m validation.single.run suite --simulators milling electroplating --versions v000 v001 v002 v003 --iterations 1

# 最新版で初期64点から36条件を追加する収束試験
python -m validation.single.run suite --simulators milling electroplating --versions v003 --iterations 12 --recommendations 3 --seeds 0
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

`validation/single/oracle/physics_model.py` は `laser_welding/physics_model.py` への互換入口です。
従来のコマンドも引き続き使えます。

```powershell
python -m validation.single.src.checks.laser_welding_pseudo_experiment --optimizer-root v000 --trial trial_laser_recheck --iterations 15
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
py -m validation.single.src.checks.v003_validation `
  --mode all `
  --optimizer-root v003 `
  --trial trial_laser_welding_v003_validation `
  --synthetic-trial trial_synthetic_constraints_v003 `
  --iterations 15
```

保存先は `validation/single/results/v003/` です。

- `*.json`: 機械比較用の全履歴と知識制約違反率
- `*_trajectory.csv`: 反復ごとの実測値・推薦スコア
- `*.md`: 人間／LLM向けの短い要約
- `*.png`: 収束曲線、推薦スコア、最終予測断面、データ支持度

matplotlibがない環境では、数値検証は継続し、PNGだけ`SKIP`として扱います。
個別にグラフを再生成する場合は次を実行します。

```powershell
py -m validation.single.src.checks.plot_v003_validation validation\single\results\v003\trial_laser_welding_v003_validation.json
```

合成問題の学習データは125候補すべてではなく、角点＋数点だけです。未測定候補を
残すことで、`--run` の推薦処理と response-space の制約検査を同時に確認できます。

## v003 GP残差アブレーション

GP残差の平均補正だけの効果を物理シミュレータで分離する検証です。v003の
`train_hybrid_model`と`run_optimization`を同一プロセスから呼び、GPありは
`nn_pred + support * gp_mean`、GPなしは同じ学習済みモデルの平均だけを`nn_pred`
へ置換します。GP標準偏差、support、NN、初期設計、無ノイズの物理真値は共通です。
両armに objective の物理的な`lower_bound=0, strength=2, enabled=true`を1ルールだけ
適用します（NN損失への知識priorを最小限にしたアブレーション）。

```powershell
python -m validation.single.src.checks.gp_residual_ablation `
  --simulators thermal_curing press_forming convection_drying `
  --seeds 0 1 2 --iterations 15 `
  --output validation/single/results/gp_residual_ablation_YYYYMMDD
```

既存の出力ディレクトリは上書きしません。`raw_results.json`はseed・armごとの
履歴とcheckpoint、`raw_results.csv`はarm・seed単位のcheckpoint集計、`trajectories.csv`は
stepごとの推薦履歴、`report.md`はpaired summaryを保存します。
最適グリッド点を初めて追加したstepでcheckpointを取り、未到達は予算最終時の
checkpointおよび`censored=true`とします。MAE/RMSE/NRMSE（出力別、全体／局所）、
macro NRMSE、出力幅正規化Gaussian NLPD、95% coverage、feasible分類精度を出力します。
local windowは最適点から正規化入力距離`<=0.20`です。

## v002 vs v003 製品版進化比較

v002の直接GP+NN support blendと、v003の残差GP（NN予測を基準にしたGP補正）および
objectiveの最小知識制約を、同じ工程・初期設計・seed・ノイズ0・1回1推薦・固定予算で
比較します。これはGP残差だけを切り出す要因分解ではなく、v002からv003への総合比較です。

既定では低次元3工程（`thermal_curing press_forming convection_drying`）と高次元2工程
（`electroplating milling`）、seed `0 1 2`、15反復を実行します。

```powershell
py -m validation.single.src.checks.v002_v003_evolution `
  --simulators thermal_curing press_forming convection_drying electroplating milling `
  --seeds 0 1 2 --iterations 15 `
  --output validation/single/results/v002_v003_evolution_YYYYMMDD
```

`--batch-size 3`を指定すると、各ラウンドで第1候補`BEST_SCORE`と第2候補以降の
`DIVERSITY`を同時取得し、3件をまとめて観測データへ追加してから再学習します。
`--iterations`はラウンド数ではなく、比較で共通化する総取得条件数です。

batch=1/3の結果を4セルにまとめるには、次を実行します。

```powershell
py -m validation.single.src.checks.batch_matrix `
  --batch1 path/to/batch1/raw_results.json `
  --batch3 path/to/batch3/raw_results.json `
  --output path/to/batch_matrix
```

`raw_results.json/csv`、`trajectories.csv`、`report.md`を出力します。最適点への到達step、
全体・local（最適点から正規化距離`<=0.20`）のMAE/RMSE/NRMSE、Gaussian NLPD、95% coverage、
feasible accuracyに加え、初期・最終の真値regretとその減少量を記録します。未到達は
`arrival=budget+1`のcensored値として集計します。到達判定は制約付き大域最適目的値を
持つ格子点（同率最適を含む）です。8入力の高次元工程ではradius=0.20の
local windowが到達した最適点1点になりやすいため、nearest-64 grid points（実効半径も保存）を
補助指標として出力し、reportでは高次元の判断にこちらを優先します。

`--seeds`は初期実験条件の選択を変え、探索経路の再現性を検査します。NNの初期化seedは
製品版v003と同じ固定値`42`です。最適点へ到達したarmは、その時点のモデルを採点して終了します。
