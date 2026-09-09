# v001：GP探索とNN予測空間の並走

> 初めて利用する場合は、このREADMEの「基本操作」から順に確認してください。
> v001はv000のGaussian Processによる次条件探索をそのまま維持し、同じ実験データから
> ニューラルネットの予測空間を同時に生成します。

## 1. v001の目的

v001は、実験を繰り返しながら次の2つを同時に確認するCLIシステムです。

```text
Gaussian Process（GP）
次にどの条件を実験する価値が高いかを推薦する

ニューラルネット（NN）
現在集まっているデータから、入力空間全体がどのように見えるかを出力する
```

次条件探索は引き続きGPが担当します。NNの予測値はGPの推薦スコアへ使用しません。

データが少ない段階では、NNが不自然または大きく変動する予測空間を作ることがあります。
v001ではそれも異常として隠さず、データ追加によって予測空間がどう変化するかを観察対象にします。

このツールの出力は確定した最適条件や量産条件ではありません。GP推薦もNN予測も、必ず実験と工程判断で確認してください。

## 2. v000との関係

v001はv000の累積版ですが、実行時にv000のコードを参照しません。
v001フォルダだけで完結する独立したスナップショットです。

v000から維持する機能は次のとおりです。

- trialの作成と分離
- problem.csvによる問題定義
- experiments.csvの生成と検証
- 同一入力条件の反復測定
- Gaussian Processによる予測
- 制約付きExpected Improvement
- 未測定条件の除外
- 複数推薦の分散選択
- recommendations.csvの安全な更新
- 日本語のCLIメッセージ

v001では次を追加します。

- 小規模な残差ニューラルネット
- 全入力グリッドに対するNN予測
- runごとのNN予測空間CSV
- 固定シードによるNN学習の再現性
- GPとNNが独立して並走するコード構成

## 3. 人間とシステムの役割

### 人間が行うこと

- trialを作る
- problem.csvへ探索範囲、目的、制約を書く
- experiments.csvへ実測結果だけを入力する
- GP推薦から実際に試す条件を選ぶ
- NN予測空間の変化をrun間で比較する
- 設備、安全、品質の観点から最終判断する

### システムが行うこと

- CSV設定と実験データを検査する
- 反復測定を入力条件ごとに集約する
- GPで未測定条件を評価し、次実験候補を推薦する
- 同じデータでNNを学習する
- problem.csvの全グリッド交点をNNで予測する
- GPの最新推薦とNNのrun別予測空間をCSV保存する
- 入力や計算に問題があれば日本語で原因を表示する

## 4. フォルダ構成

```text
v001/
├─ src/
│  ├─ data_loader.py
│  ├─ validation.py
│  ├─ preprocessing.py
│  ├─ parameter_space.py
│  ├─ settings.py
│  ├─ cli.py
│  ├─ gp/
│  │  ├─ model.py
│  │  ├─ acquisition.py
│  │  ├─ optimizer.py
│  │  └─ reporting.py
│  └─ nn/
│     ├─ model.py
│     ├─ trainer.py
│     ├─ predictor.py
│     └─ reporting.py
├─ trials/
│  └─ trial_###/
│     ├─ problem.csv
│     ├─ data/
│     │  └─ experiments.csv
│     └─ output/
│        ├─ recommendations.csv
│        ├─ nn_response_space_run_0001.csv
│        ├─ nn_response_space_run_0002.csv
│        └─ ...
├─ tests/
└─ requirements.txt
```

GPとNNの学習コードは分離しています。CSV読込、入力検証、反復集約、入力正規化、
パラメータグリッドは共通処理として共有します。

## 5. 基本操作

利用者はプロジェクト直下の共通main.pyだけを実行します。
v001を使う場合は、最初に`--version v001`を指定します。

### 新しいtrialを作る

```powershell
python main.py --version v001 --new trial_001
```

### problem.csvを検査してexperiments.csvを準備する

```powershell
python main.py --version v001 --prepare trial_001
```

### GP探索とNN予測空間生成を実行する

```powershell
python main.py --version v001 --run trial_001
```

GP推薦数を指定する場合は`--n`を使用します。

```powershell
python main.py --version v001 --run trial_001 --n 9
```

`--n`はGPの推薦件数です。NNは推薦数に関係なく、毎回パラメータ空間全体を予測します。

プロジェクト同梱Pythonを使う場合は次のように実行できます。

```powershell
.\.runtime\venv\Scripts\python.exe .\main.py --version v001 --run trial_001
```

## 6. problem.csv

problem.csvは探索内容とNN予測空間の範囲を定義します。

```csv
column,display_name,unit,role,direction,lower,upper,step,target
p1,コアパワー,W,parameter,,350,500,10,
p2,リングパワー,W,parameter,,1750,2395,5,
t,照射時間,ms,parameter,,34,70,1,
T,トルク強度,N,constraint,greater_equal,,,,70
S,セパレータ熱影響,mm,objective,minimize,,,,
```

### role

| role | 意味 |
|---|---|
| `parameter` | 実験時に設定する入力。lower、upper、stepが必要 |
| `objective` | 最小化または最大化する結果。v001では1個限定 |
| `constraint` | 指定値以上または以下にする結果。複数可 |
| `monitor` | 最適化には使わないがGP・NNで予測する結果。複数可 |

### direction

| role | 使用できるdirection |
|---|---|
| objective | `minimize`、`maximize` |
| constraint | `greater_equal`、`less_equal` |
| parameter、monitor | 空欄 |

入力パラメータ名、個数、結果変数名はコードへ固定していません。

## 7. experiments.csv

`--prepare`によってproblem.csvに対応するヘッダーが生成されます。

```csv
experiment_id,p1,p2,t,T,S
```

ルールはv000と同じです。

- 1行を1回の実験とする
- 実測値だけを書く
- 欠損値を作らない
- experiment_idを重複させない
- 同じ入力条件の反復測定は別行で記録できる
- 数値欄へ単位やコメントを書かない
- UTF-8、UTF-8 BOM、CP932のCSVを使用できる

同じ入力条件が複数行ある場合、GPとNNはいずれも条件ごとの平均値を学習します。
反復測定の元行は変更しません。

## 8. GPによる次条件探索

GP処理はv000と同じです。

```text
全グリッド候補を生成
  ↓
実測済み条件を除外
  ↓
目的・制約・monitorごとにGPを学習
  ↓
予測平均と標準偏差を計算
  ↓
制約達成確率を計算
  ↓
Expected Improvementを計算
  ↓
推薦スコアを計算
  ↓
1位は最高スコア、2位以降は分散性も考慮
```

制約を満たす実測条件がある場合の基本スコアは次です。

```text
推薦スコア = Expected Improvement × 全制約の達成確率
```

制約を満たす実測条件がまだない場合は、制約達成確率と目的予測の不確かさを使って探索候補を選びます。

## 9. NNによる予測空間

NNはGP推薦の完了後、同じ実験データを使って独立に学習します。

### 入力

- problem.csvでroleがparameterの全列
- 探索上下限に対して0～1へ正規化

### 出力

- objective、constraint、monitorの全列
- 学習時は出力ごとに平均0、標準偏差1へ変換
- CSV出力時は元の単位へ戻す

### ネットワーク

```text
正規化入力
  ├─ 線形スキップ ───────────┐
  └─ 全結合tanh層             │
       ↓                      │
     残差ブロック             │
       ↓                      │
     出力層 ──────────────────┤
                              ↓
                         標準化予測
```

初期設定は次のとおりです。

- 隠れユニット：32
- 最大エポック：2,000
- optimizer：Adam
- 学習率：0.01
- L2正則化：0.0001
- early stopping：200エポック改善なし
- 乱数シード：42
- CPU上のNumPy実装

NNは予測空間の観察用です。NN予測をGP推薦、制約達成確率、Expected Improvementへ混ぜません。

## 10. 共通パラメータグリッド

GPとNNはproblem.csvのlower、upper、stepから同じ規則でグリッドを作ります。

```text
parameter 1の候補
× parameter 2の候補
× ...
= 全グリッド交点
```

stepで上限へ正確に到達しない場合でも、upperを最後の候補として追加します。
候補数が200,000件を超えるproblem.csvは検証段階で停止します。

GPは全グリッドから実測済み条件を除外して推薦します。NNは実測済み点を含む全グリッドを出力します。

## 11. 出力

利用者が確認する成果物は2種類だけです。

```text
output/
├─ recommendations.csv
└─ nn_response_space_run_####.csv
```

### recommendations.csv

最新のGP推薦です。正常終了時に安全に上書きします。

主な列は次です。

- 推薦順位
- 入力パラメータ
- 各結果のGP予測平均・標準偏差
- 制約達成確率
- Expected Improvement
- 推薦スコア
- 最近傍実測点までの正規化距離
- 推薦理由

### nn_response_space_run_####.csv

そのrun時点のNN予測空間です。既存ファイルは上書きせず、連番で追加します。

```csv
run_id,data_rows,unique_conditions,nn_seed,p1,p2,t,T_nn_pred,S_nn_pred
run_0001,12,9,42,350,1750,34,51.2,1.14
```

列は次の順です。

1. `run_id`
2. `data_rows`
3. `unique_conditions`
4. `nn_seed`
5. 全parameter列
6. 全結果の`<column>_nn_pred`

別のモデルファイル、学習履歴JSON、評価CSVは作りません。

## 12. run履歴

NN予測空間は次のように連番保存されます。

```text
最初の--run    nn_response_space_run_0001.csv
次の--run      nn_response_space_run_0002.csv
その次         nn_response_space_run_0003.csv
```

既存ファイル名の最大番号に1を加えて採番します。過去のNN空間を残すことで、
実験データ追加前後のCSVを比較できます。

同じexperiments.csvで再実行した場合も新しいrunとして保存します。
固定シードと同じ設定であれば、予測値は同じになります。

## 13. CLI表示

実行時にはGP推薦の概要に続き、NNの状態を表示します。

```text
NNによる予測空間:
  学習データ: 42行
  固有条件: 42条件
  学習エポック: ...
  標準化MSE: ...
  予測グリッド: 13,981点
  run: run_0016
  出力: ...\nn_response_space_run_0016.csv
```

学習エポックとMSEは診断用にCLIへ表示しますが、別ファイルには保存しません。

## 14. GPとNNの独立性

GPとNNは同じデータを使いますが、モデルと出力は独立しています。

- NNの予測値はGP推薦へ影響しない
- GPの予測値をNNの教師データにしない
- experiments.csvの実測値だけを両モデルが学習する
- NN学習に失敗しても、正常に計算されたGP推薦は保持する
- NNファイルは最後まで完成した場合だけ正式名で保存する

NN学習が失敗した場合はCLIに`[NN注意]`を表示し、recommendations.csvの場所を案内します。

## 15. 少数データ時の扱い

v001は固有条件が2件以上あればNN学習を実行します。件数による採用判定や出力停止はしません。

少数データ時には次が起こり得ます。

- 実測点間で急な変化を作る
- 実測点から遠い領域で極端な値を出す
- データを1件追加しただけで空間形状が大きく変わる
- GPと全く異なる予測を出す

これらの変化をrun別CSVで観察することがv001の目的です。予測が正しいことを保証する機能ではありません。

## 16. エラーとファイル保護

- 不正なproblem.csvでは学習を開始しない
- 不正なexperiments.csvでは出力を更新しない
- recommendations.csvは一時ファイル完成後に置換する
- NN予測空間も一時ファイル完成後に正式名へ置換する
- 同名trialを上書きしない
- 既存のNN runファイルを上書きしない
- 未測定候補より多い`--n`は拒否する

CSVをExcelで開いたまま書き込みできない場合は、ファイルを閉じて再実行してください。

## 17. 依存ライブラリ

外部依存はNumPyだけです。

```powershell
python -m pip install -r requirements.txt
```

GPとNNはいずれもローカルで計算し、実験データを外部へ送信しません。

## 18. テスト

```powershell
python -m unittest discover -s tests -v
```

テストでは次を確認します。

- GP推薦1位が最高スコアであること
- 複数推薦が未観測領域へ分散すること
- 推薦条件が重複しないこと
- 自動分散比率と制約確率ガード
- NN予測の形状と有限値
- 固定シードによるNN予測の再現性
- NN run番号の採番

## 19. レーザー溶接モデルによる共通検証

プロジェクト直下の`validation/single/README.md`に、全バージョン共通の検証条件があります。

```powershell
python -m validation.single.src.checks.laser_welding_pseudo_experiment `
  --optimizer-root v001 `
  --trial trial_laser_welding_v001_001 `
  --iterations 15
```

検証では、初期27条件からGP推薦1位を論理モデルで評価し、実験CSVへ追加する処理を15回繰り返します。
各`--run`でGP推薦に加え、NN予測空間CSVが1つ生成されます。最後の再計算を含めて16個のNN runファイルが残ります。

## 20. v001で実装しないもの

- NNによる次実験条件推薦
- NN予測を使った制約付き最適化
- GPとNNの優劣判定
- GPとNNの比較CSV
- ニューラルネット重みの永続化
- 学習履歴JSON
- 交差検証
- 物理アンカー
- Web UIと3D表示
- UIからの再学習

## 21. 完成条件

- v000と同じGP推薦処理が動作する
- v000と同じ`--new`、`--prepare`、`--run`を使用できる
- GPとNNが独立したコードに分かれている
- 入力数と出力数をproblem.csvから決定できる
- GPとNNが同じグリッド規則を使用する
- NNが全グリッド点を予測できる
- NN予測空間をrunごとに上書きせず保存できる
- NN出力を予測空間CSVだけに限定する
- 少数データでもNN予測を生成する
- 同じデータ・設定・シードで同じNN予測を再生成できる
- NNが失敗してもGPの正常な推薦結果を保持する
- レーザー溶接共通検証を最後まで実行できる

## 22. 現在の状態

v001の初期実装は完了しています。

- v000互換のGP探索
- GPとNNのソース分離
- NumPyによる残差ニューラルネット
- 共通パラメータグリッド
- run別NN予測空間CSV
- GP・NNの独立した失敗処理
- 単体テスト

最初に新しいtrialを作成してください。

```powershell
python main.py --version v001 --new trial_001
```
