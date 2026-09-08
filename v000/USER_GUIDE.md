# ParamOptimizer v000 利用手順書

## この手順書で行うこと

この手順書では、実験条件の探索案件を作り、過去の実測データを入力し、次に実験する
候補を出力して、得られた実測結果を追加するところまでを説明します。

このツールが提案するのは「次に測定する価値が高い候補」です。確定した最適条件、
安全条件、量産条件ではありません。設備・品質・安全面の最終判断と実測確認は人間が
行ってください。

## 最初に理解する3ファイル

1つの探索案件を`trial`と呼びます。利用者が主に扱うのは次の3ファイルです。

```text
trials/trial_000/
├─ problem.csv
│  └─ 入力条件、探索範囲、目的、制約を設定する
├─ data/
│  └─ experiments.csv
│     └─ これまでに得られた実測結果を蓄積する
└─ output/
   └─ recommendations.csv
      └─ 次に実験する候補と予測根拠を確認する
```

## 全体フロー

```text
PowerShellでv000フォルダへ移動
  ↓
--newでtrialを作る
  ↓
problem.csvへ探索問題を入力する
  ↓
--prepareでexperiments.csvを作る
  ↓
experiments.csvへ過去の実測結果を入力する
  ↓
--run --nで次実験候補を出す
  ↓
recommendations.csvを確認して実験する
  ↓
得られた実測結果をexperiments.csvへ追記する
  ↓
同じ--runを繰り返す
```

---

## Step 0：PowerShellを開き、実行場所へ移動する

PowerShellを開き、`v000`フォルダへ移動します。

```powershell
cd C:\Users\tsuts\Desktop\PythonDev_std\20260803_NextParams\v000
```

現在位置を確認します。

```powershell
Get-Location
```

末尾が次のようになっていれば正しい場所です。

```text
20260803_NextParams\v000
```

初回だけ、必要なライブラリをインストールします。

```powershell
python -m pip install -r requirements.txt
```

以降の`python main.py ...`コマンドは、すべてこの`v000`フォルダで実行します。

---

## Step 1：新しいtrialを作る

例として`trial_000`を作ります。

```powershell
python main.py --new trial_000
```

実行すると、次のファイルとフォルダが作られます。

```text
v000/
└─ trials/
   └─ trial_000/
      ├─ problem.csv
      ├─ data/
      └─ output/
```

この段階では、`experiments.csv`と`recommendations.csv`はまだありません。

同じ名前のtrialがすでにある場合は上書きされません。別の名前を指定してください。

```powershell
python main.py --new trial_001
```

### trialを分ける目安

同じ工程、材料、設備、入力パラメータ、評価項目で探索を続ける場合は、同じtrialを
使い続けます。材料、設備、評価項目、最適化目的などが変わり、データを混ぜるべきで
ない場合は、新しいtrialを作ります。

---

## Step 2：problem.csvへ探索内容を入力する

次のファイルをExcelまたはテキストエディターで開きます。

```text
v000\trials\trial_000\problem.csv
```

作成直後は記入例が入っています。記入例を、実際の探索内容に合わせて置き換えます。

### problem.csvの列

| 列 | 入力内容 |
|---|---|
| `column` | `experiments.csv`で使う短い列名。重複不可 |
| `display_name` | 画面に表示する分かりやすい名称 |
| `unit` | W、mm、msなどの単位。なければ空欄 |
| `role` | `parameter`、`objective`、`constraint`、`monitor` |
| `direction` | 最小化・最大化・制約方向 |
| `lower` | 入力パラメータの探索下限 |
| `upper` | 入力パラメータの探索上限 |
| `step` | 候補条件を作る刻み幅 |
| `target` | 制約の基準値 |

### roleごとの記入方法

| role | 意味 | direction | lower / upper / step | target |
|---|---|---|---|---|
| `parameter` | 実験時に設定する入力条件 | 空欄 | 必須 | 空欄 |
| `objective` | 良くしたい評価結果 | `minimize`または`maximize` | 空欄 | 空欄 |
| `constraint` | 守りたい基準 | `greater_equal`または`less_equal` | 空欄 | 必須 |
| `monitor` | 最適化せず予測だけする結果 | 空欄 | 空欄 | 空欄 |

v000では、`objective`は1個だけ指定します。`parameter`、`constraint`、`monitor`は
複数指定できます。

### レーザー溶接の記入例

次の例は、溶込み深さを最大化し、スパッタレベルを4以下にする問題です。

```csv
column,display_name,unit,role,direction,lower,upper,step,target
laser_power_w,レーザー出力,W,parameter,,100,6000,200,
spot_diameter_um,スポット径,um,parameter,,50,300,25,
scan_speed_mm_s,走査速度,mm/s,parameter,,10,1000,25,
penetration_depth_mm,溶込み深さ,mm,objective,maximize,,,,
spatter_level_0_9,スパッタレベル,level,constraint,less_equal,,,,4
```

### problem.csvで注意すること

- ヘッダー名と列順を変更しない
- `column`に空白や単位を入れない
- `experiment_id`は予約名なので使用しない
- `lower`は`upper`より小さくする
- `step`は0より大きくする
- 探索候補の全組み合わせは最大200,000件
- 保存形式はCSVのままにする

---

## Step 3：experiments.csvを準備する

`problem.csv`を保存して閉じた後、次を実行します。

```powershell
python main.py --prepare trial_000
```

設定が正常なら、次のファイルが作られます。

```text
v000\trials\trial_000\data\experiments.csv
```

レーザー溶接例では、ヘッダーは次のようになります。

```csv
experiment_id,laser_power_w,spot_diameter_um,scan_speed_mm_s,penetration_depth_mm,spatter_level_0_9
```

設定に問題がある場合は、CLIに問題の行、列、修正方法が表示されます。
`problem.csv`を修正して、もう一度`--prepare`を実行してください。

すでに`experiments.csv`がある場合、`--prepare`は実測データを上書きしません。

---

## Step 4：experiments.csvへ過去の実測結果を入力する

次のファイルをExcelなどで開きます。

```text
v000\trials\trial_000\data\experiments.csv
```

1行を1回の実験として、実際に測定した結果を入力します。

```csv
experiment_id,laser_power_w,spot_diameter_um,scan_speed_mm_s,penetration_depth_mm,spatter_level_0_9
initial_001,100,50,10,0.05,0
initial_002,3000,175,500,1.42,2
initial_003,6000,300,1000,0.83,1
```

### experiments.csvのルール

- `experiment_id`はtrial内で重複しない名前にする
- 入力条件と測定結果をすべて数値で入力する
- 単位やコメントを数値セルへ書かない
- 欠損したセルを作らない
- 予測値を入力しない
- 実測できなかった条件は入力しない
- 明らかな入力ミスをシステムが勝手に削除・補正することはない
- 同じ条件の反復測定は、別の`experiment_id`で複数行入力できる

実験データの行数に固定上限はありません。過去データと新しいデータを同じ
`experiments.csv`へ蓄積して使います。ただし、固有条件数が非常に増えると計算時間も
増加します。

入力後、CSVを保存し、Excelで開いている場合は閉じてください。

---

## Step 5：次に実験する候補を出力する

次に9条件ほしい場合は、次を実行します。

```powershell
python main.py --run trial_000 --n 9
```

`--n`は「今回ほしい次実験候補の数」です。`experiments.csv`の総行数や、前回追加した
行数とは関係ありません。

```text
experiments.csvに100件ある
  + --n 9
  ↓
100件すべてでモデルを学習し、未測定候補を9件出力する
```

`--n`を省略した場合は3件出力します。

```powershell
python main.py --run trial_000
```

処理が成功すると、次のファイルが作成または更新されます。

```text
v000\trials\trial_000\output\recommendations.csv
```

### recommendations.csvで確認する主な列

| 列 | 意味 |
|---|---|
| `rank` | 推薦順位 |
| `selection_role` | 1位は`BEST_SCORE`、2位以降は`DIVERSITY` |
| 入力パラメータ列 | 次に設定する候補条件 |
| `<結果列>_mean` | モデルによる予測平均 |
| `<結果列>_std` | 予測の標準偏差 |
| `<制約列>_probability` | その制約を満たす予測確率 |
| `feasibility_probability` | すべての制約を満たす総合予測確率 |
| `expected_improvement` | 現在の実測最良値を改善する期待値 |
| `nearest_distance` | 最も近い実測条件までの正規化距離 |
| `recommendation_reason` | 推薦理由と注意 |

2位以降の分散具合は、モデルの不確かさと未観測領域の広さから自動調整されます。
利用者が分散率を設定する必要はありません。判断値は次の列に記録されます。

```text
auto_diversity_weight
auto_uncertainty_signal
auto_coverage_signal
feasibility_guard_active
```

`recommendations.csv`は`--run`に成功するたびに最新結果で上書きされます。履歴を残す
必要がある場合は、実験前に別名でコピーしてください。

---

## Step 6：推薦条件を実験し、実測結果を追記する

`recommendations.csv`を確認し、設備・品質・安全面を人間が判断して実験します。

実測できた条件だけを、`experiments.csv`の末尾へ追加します。

```csv
bo_001_01,5900,300,60,8.38,4
bo_001_02,5100,300,60,7.24,4
```

推薦が9件でも、実測に成功したのが7件なら7行だけ追加して構いません。次回も9件
ほしい場合は、再び同じコマンドを実行します。

```powershell
python main.py --run trial_000 --n 9
```

システムは、その時点で`experiments.csv`に入っているすべての実測データを使って
モデルを作り直します。

```text
過去の実測結果
  + 前回の推薦から得られた実測結果
  ↓
全データで再学習
  ↓
測定済み条件を除外
  ↓
次の候補を指定件数だけ推薦
```

この「実験結果を追記 → `--run`」を、目的を達成するまで同じtrialで繰り返します。

---

## 9スロットのバッチ実験での運用例

```text
1. experiments.csvへ現在までの全実測結果を保存
2. python main.py --run trial_000 --n 9
3. recommendations.csvの9条件を確認
4. 9スロットで同時に実験
5. 成功して実測値が得られた条件だけCSVへ追記
6. 次のバッチが必要なら、再び--n 9で実行
```

同じバッチ内の2～9位は、1位と既存実測点に集中しすぎないよう自動的に分散されます。
同じバッチの結果が得られる途中で再学習する必要はありません。結果が揃った段階で
まとめて追記します。

---

## よくあるエラー

### `python`が見つからない

Pythonがインストールされ、PowerShellから実行できることを確認してください。

```powershell
python --version
```

### NumPyがない

`v000`フォルダで次を実行します。

```powershell
python -m pip install -r requirements.txt
```

### CSVを開けない

ExcelでCSVを開いたままの場合は閉じてから、もう一度コマンドを実行してください。

### ヘッダーが一致しない

`problem.csv`を変更した後、既存の`experiments.csv`と列構成が合わなくなっています。
実測データを消さず、必要な列と列順を手作業で合わせてください。

### 推薦件数を出せない

`--n`で指定した件数より、未測定候補が少ない状態です。`--n`を小さくしてください。

---

## ファイルの役割まとめ

| ファイル | 人間が編集するか | 内容 |
|---|---|---|
| `trials/<trial>/problem.csv` | 最初に編集 | 探索問題の定義 |
| `trials/<trial>/data/experiments.csv` | 毎回追記 | 実測データだけを蓄積 |
| `trials/<trial>/output/recommendations.csv` | 編集しない | 最新の次実験候補 |
| `main.py` | 編集しない | `--new`、`--prepare`、`--run`の入口 |

## 最短コマンド一覧

```powershell
cd C:\Users\tsuts\Desktop\PythonDev_std\20260803_NextParams\v000

# 1. trial作成
python main.py --new trial_000

# 2. problem.csvを人間が編集した後、実験CSVを準備
python main.py --prepare trial_000

# 3. experiments.csvへ実測値を入力した後、次の9条件を推薦
python main.py --run trial_000 --n 9

# 4. 実験結果をexperiments.csvへ追記後、再推薦
python main.py --run trial_000 --n 9
```
