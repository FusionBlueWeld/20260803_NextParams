# v002：GP–NN統計Hybridによる探索と解空間

v002は、Gaussian ProcessとニューラルネットをHybrid内部の計算部品として融合し、
Hybrid予測による次実験条件とrun別の解空間を出力します。

利用者が扱うモデルはHybridだけです。独立したGP・NN用の操作や出力はありません。

## 1. v002の目的

```text
v001
GPとNNを独立して観察する

v002
GPとNNを1つの統計Hybridとして利用する
```

v002では物理アンカーを使用しません。物理仮説はv003で導入し、
統計モデル同士の融合効果と物理仮説の効果を別々に検証します。

## 2. フォルダ越境をしない方針

v002は実行時にv000・v001のコードを参照しません。

```python
# 使用しない
from v000...
from v001...
```

データ読込、検証、GP、NN、Hybrid、推薦、CSV出力までv002内で完結します。
バージョンフォルダを単位として配布できる構成です。

## 3. 利用者とシステムの役割

### 利用者

- trialを作成する
- problem.csvへ入力、目的、制約を書く
- experiments.csvへ実測値を追加する
- recommendations.csvから次に試す条件を選ぶ
- run別解空間からデータ追加による変化を確認する

### システム

- 入力CSVを検証する
- 同一条件の反復測定を集約する
- Hybrid内部のGPとNNを学習する
- 全候補のHybrid平均と不確かさを計算する
- 制約付きExpected Improvementで次条件を推薦する
- 全グリッドのNN予測、GP支持度、Hybrid予測を保存する

## 4. フォルダ構成

```text
v002/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ settings.py
│  ├─ data_loader.py
│  ├─ validation.py
│  ├─ preprocessing.py
│  ├─ parameter_space.py
│  └─ hybrid/
│     ├─ __init__.py
│     ├─ gp_component.py
│     ├─ nn_component.py
│     ├─ support.py
│     ├─ model.py
│     ├─ optimizer.py
│     └─ reporting.py
├─ tests/
└─ trials/
   └─ trial_###/
      ├─ problem.csv
      ├─ data/
      │  └─ experiments.csv
      └─ output/
         ├─ recommendations.csv
         └─ response_spaces/
            ├─ hybrid_response_space_run_0001.csv
            └─ ...
```

モデル固有のサブフォルダは`hybrid/`だけです。GPとNNはHybrid内部部品として配置します。

## 5. 基本操作

プロジェクト直下の共通main.pyから実行します。

```powershell
python main.py --version v002 --new trial_002
python main.py --version v002 --prepare trial_002
python main.py --version v002 --run trial_002
python main.py --version v002 --run trial_002 --n 9
```

同梱Pythonを使う場合：

```powershell
.\.runtime\venv\Scripts\python.exe main.py --version v002 --run trial_002
```

## 6. 入力データ

problem.csvとexperiments.csvの形式はv000・v001と同じです。

### problem.csv

```csv
column,display_name,unit,role,direction,lower,upper,step,target
p1,コアパワー,W,parameter,,350,500,10,
p2,リングパワー,W,parameter,,1750,2395,5,
t,照射時間,ms,parameter,,34,70,1,
T,トルク強度,N,constraint,greater_equal,,,,70
S,セパレータ熱影響,mm,objective,minimize,,,,
```

- parameter：1個以上
- objective：1個
- constraint：複数可
- monitor：複数可
- 入出力名と個数はコードへ固定しない
- 全グリッド上限：200,000点

### experiments.csv

```csv
experiment_id,p1,p2,t,T,S
```

- 実測値だけを入力する
- 欠損、非数、無限値を許可しない
- experiment_idを重複させない
- 同じ入力条件の反復測定を別行で記録できる
- UTF-8、UTF-8 BOM、CP932に対応する

## 7. Hybrid内部のGP

結果変数ごとにRBF Gaussian Processを学習します。

- 入力：探索上下限に対して0～1へ正規化
- 出力：平均・標準偏差で正規化
- length scale：0.35
- 測定ノイズ：反復測定から推定
- 反復がない場合：結果スケールの5%を使用
- 数値計算：Cholesky分解

GPは次を提供します。

- GP予測平均
- GP予測標準偏差
- Hybridの戻り先
- Hybrid推薦の不確かさ

独立したGP解空間は出力しません。

## 8. Hybrid内部のNN

v001と同じ小規模な残差ニューラルネットをv002内に実装しています。

- 隠れユニット：32
- 最大エポック：2,000
- optimizer：Adam
- 学習率：0.01
- L2正則化：0.0001
- early stopping：200エポック改善なし
- seed：42
- 実装：NumPy

NNはすべての結果変数を同時に直接予測します。独立したNN専用CSVは出力しません。

## 9. GPデータ支持度

入力位置が実測条件にどの程度支えられているかを、GP事後分散の減少量から計算します。

```text
support(x) = k(x,X) K^-1 k(X,x)
```

- 0以上1以下
- 全結果変数で共通
- 入力空間だけから計算
- RBF length scale：0.35
- support noise：0.025
- support multiplier：1.0

実装では逆行列を直接作らず、Cholesky分解による連立方程式を使用します。

`gp_support`は正解確率、制約達成確率、予測信頼区間ではありません。

## 10. Hybrid予測

Hybrid平均はGP平均とNN予測をGP支持度で融合します。

```text
Hybrid_mean(x)
= GP_mean(x)
  + support(x) × {NN(x) - GP_mean(x)}
```

または：

```text
Hybrid_mean(x)
= {1 - support(x)} × GP_mean(x)
  + support(x) × NN(x)
```

したがって：

```text
support = 0  → Hybrid = GP
support = 1  → Hybrid = NN
0 < support < 1 → GPとNNの間
```

初期v002ではHybridの標準偏差としてGP標準偏差を使用します。

```text
Hybrid_std = GP_std
```

これはNN由来の不確かさを含む厳密なHybrid事後分布ではありません。
入力位置と実測結果に基づくGP由来の不確かさとして表示します。

## 11. Hybridによる次条件推薦

v002では正式推薦もHybridを使用します。

```text
平均       Hybrid_mean
標準偏差   GP_std
```

目的変数についてExpected Improvementを計算し、制約変数について達成確率を計算します。

```text
推薦スコア
= Expected Improvement × 全制約達成確率
```

1位は最高スコア、2位以降は有望度と未観測領域の被覆を組み合わせて選びます。

## 12. 処理フロー

```text
CSV読込・検証
  ↓
反復条件集約・正規化
  ↓
Hybrid内部GP学習
  ↓
Hybrid内部NN学習
  ↓
GP支持度モデル作成
  ↓
未測定候補のHybrid予測
  ↓
Hybrid次条件推薦
  ↓
全グリッドのHybrid解空間生成
  ↓
CSV保存
```

## 13. 出力

### 最新推薦

```text
output/recommendations.csv
```

主な列：

- 入力パラメータ
- gp_support
- 結果ごとのNN予測
- 結果ごとのHybrid平均・標準偏差
- 制約達成確率
- Expected Improvement
- 推薦スコア
- 推薦理由

検証スクリプトとの互換性のため、Hybrid平均・標準偏差は`<result>_mean`、`<result>_std`列でも保存します。

### run別解空間

```text
output/response_spaces/hybrid_response_space_run_####.csv
```

基本列：

```csv
run_id,data_rows,unique_conditions,nn_seed,<parameter列>,gp_support,<result>_nn_pred,<result>_hybrid_mean,<result>_hybrid_std
```

1ファイルが1回の`--run`時点の全解空間、1行がグリッド上の1条件です。
過去runは上書きしません。

## 14. エラーとファイル保護

- problem.csv不正時は学習しない
- experiments.csv不正時は出力を更新しない
- GPまたはNN学習失敗時は正式出力を作らない
- recommendations.csvは一時ファイル完成後に置換する
- 解空間CSVも一時ファイル完成後に正式名へ置換する
- 既存runファイルを上書きしない
- 同名trialを上書きしない

## 15. テスト

```powershell
python -m unittest discover -s v002\tests -v
```

確認項目：

- supportが0～1
- 実測点付近のsupportが遠方より高い
- support=0でHybridがGPと一致
- support=1でHybridがNNと一致
- HybridがGPとNNの間にある
- 固定seedによる再現性
- Hybrid予測が有限値
- 複数推薦の分散性
- response_spaces内のrun採番

## 16. レーザー溶接モデル検証

```powershell
python validation\laser_welding_pseudo_experiment.py `
  --optimizer-root v002 `
  --trial trial_laser_welding_v002_validation `
  --iterations 15
```

固定条件：

- 初期実験：27点
- 追加実験：15回
- 候補グリッド：13,981点
- 目的：溶込み深さ最大化
- 制約：スパッタレベル4以下

検証結果：

| 項目 | v002 |
|---|---:|
| 最終最良値／真の最良値 | 100.00% |
| 真の最良条件 | 発見 |
| 真の最良条件へ到達 | 14回目 |
| 制約を満たした推薦 | 11 / 15 |
| 解空間run | 16 |
| 各runの行数 | 13,981 |
| 非数・無限値 | 0 |

今回の論理モデルでは真の最良条件へ到達しましたが、Hybridが常にGPまたはNNより優れることを保証する結果ではありません。

## 17. v002で実装しないもの

- 物理アンカー
- 物理残差NN
- 物理事前平均GP
- NN不確かさの専用推定
- GP・NNの独立出力
- モデル重みの永続化
- Web UI

## 18. 完成状態

v002の初期実装と検証は完了しています。

- 自己完結したHybrid内部実装
- Hybrid推薦
- NNグリッドとGP支持度の行別出力
- `output/response_spaces/`へのrun履歴
- 共通`--version v002`対応
- 単体テスト
- レーザー溶接15反復検証
