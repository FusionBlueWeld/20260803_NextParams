# ParamOptimizer v003

v003は、技能者の経験則を「説明文」ではなくニューラルネット（NN）の損失へ直接入れ、次の実験条件へ反映する版です。複雑な物理式より先に、現場で言語化しやすい単純な知識、実験可能範囲、終了判断を扱います。v000〜v002には依存せず、既存版も変更しません。

## 目視レビューの入口

部内で共有しているv002の粒度を基準に、v004と同じ名前・責務のファイルへ整理しています。
入口・入力設定・実行手順・計算・出力を分け、各ファイル冒頭に役割を記載しています。

```text
v003/
├─ README.md / VALIDATION_REPORT.md / requirements.txt
├─ src/
│  ├─ cli.py                 引数の解釈・終了コード・エラー表示
│  ├─ trials.py              trial作成・存在確認・実験CSV準備
│  ├─ trial_inputs.py        prepare/run共通の設定読込とTrialInputs
│  ├─ workflow.py            入力→停止→学習→診断→公開の実行順序
│  ├─ settings.py            定数・ProblemDefinition・VariableDefinition
│  ├─ data_loader.py         CSVの読書き
│  ├─ validation.py          問題CSV・実験CSVの検査
│  ├─ preprocessing.py       同一条件の集約と正規化
│  ├─ parameter_space.py     候補グリッド
│  ├─ diagnostics.py         NN・Hybrid・実測の知識診断とCSV保存
│  ├─ reporting.py           問題設定の確認表示
│  ├─ knowledge.py / knowledge_loss.py   知識ルールの読込・採点
│  ├─ hybrid/                NN・GP・支持度・推薦・解空間出力
│  ├─ policies/              禁止領域と優先領域
│  └─ stopping/              停止条件・収束判定・履歴
├─ tests/
└─ trials/                   利用者の入力と生成結果
```

| v002で共有している責務 | v003・v004で共通の配置 |
|---|---|
| `src/cli.py` の操作別処理 | [cli.py](src/cli.py) → [trials.py](src/trials.py) / [workflow.py](src/workflow.py) |
| 設定・入力読込・検査 | [trial_inputs.py](src/trial_inputs.py)が各読込器・検査器を同じ順序で呼ぶ |
| `hybrid/model.py`・`optimizer.py` | [hybrid/](src/hybrid/)に学習と推薦を維持 |
| 結果表示・CSV保存 | [reporting.py](src/reporting.py)、[diagnostics.py](src/diagnostics.py)、[hybrid/reporting.py](src/hybrid/reporting.py) |

### 処理の読み順と公開条件

1. `cli.py` で `--new`・`--prepare`・`--run` の入口を確認します。
2. `trials.py` → `trial_inputs.py` で入力ファイルと既定値の扱いを確認します。
3. `workflow.py` の番号付きコメントを追い、`hybrid/` の学習・推薦処理を読みます。
4. `diagnostics.py` と `stopping/` で、推薦を出さない条件を確認します。
5. `hybrid/reporting.py` で推薦CSVと全候補の解空間を確認します。

```text
CSV・任意設定 → 共通の入力検査 → 候補と実測の準備
                                      ↓
                             学習前の必須停止判定
                                      ↓
                          Hybrid学習 → 推薦条件の計算
                                      ↓
                   NN・Hybrid・実測の必須知識ルール診断
                                      ↓
                         解空間・停止履歴保存 → 推薦公開
```

`trial_inputs.py` はprepareとrunの共通処理です。実験CSVの内容はrun時に検査します。
必須知識ルールに違反・未検証があれば新しい推薦を公開せず、前回の推薦は履歴で確認できます。
共通ファイルはv003内に保持しており、v004のコードを実行時にimportする依存はありません。

### v004との機能差

| 項目 | v003 | v004で追加された機能 |
|---|---|---|
| 問題の入力変数 | 全parameterを操作候補として探索 | 操作条件と流入状態を分離 |
| 探索候補 | 問題CSVの全候補から既測定点・禁止領域を除外 | 現在の流入状態で候補を固定し、同じ状態内で既測定点を判定 |
| 改善基準・停止履歴 | 単一trialの実測と既存の設定契約 | 流入状態に対応した実測の選択と接続設定の記録 |
| 工程接続の設定・保存 | 対象外 | `connection.py`・`stage_bundle.py` |

ファイル名・関数名・コメントの粒度は共通部分でそろえ、上記の機能差がコード差分として
残るようにしています。学習係数・推薦方式・CSV列・既存の停止履歴形式は維持しています。

## バージョン上の位置付け

v003では、当初検討していた具体的な物理式によるアンカーを、技能者が入力できる
知識アンカーへ一般化しました。知識制約を学習したNNを基準とし、残差GPが実測データで
補正します。データ支持が弱い領域では残差補正が弱まり、知識付きNNへ戻ります。

したがってv003は「明示的な物理方程式を内蔵する版」ではありませんが、予測の戻り先を
技能者知見で形作るアンカーの枠組みは実装済みです。以前の「v004で物理式アンカーを
新設する」案は現在のロードマップから外し、必要性が明確になった場合に改めて検討します。

v003は単一工程・単一trial内で完結し、複数工程を接続する共通インターフェースは
持ちません。[v004](../v004/README.md)は流入状態を扱う個別探索とstage bundleの
公開を追加し、[r001](../r001/README.md)が複数工程を接続する分担です。両版は現行機能範囲で
実装・FIX済みです。今回のv003の整理では、これらの接続機能を含めず、共通部分の配置と説明をそろえています。

## 操作と入出力の索引

- 実行入口: リポジトリ直下の `main.py`。必ず `--version v003` を指定
- 問題定義: `trials/<trial>/problem.csv`
- 実験値: `trials/<trial>/data/experiments.csv`
- NN知識: `trials/<trial>/knowledge_constraints.csv`
- 禁止/優先領域: `trials/<trial>/search_regions.csv`
- 終了設定: `trials/<trial>/stop_settings.csv`
- 推薦: `trials/<trial>/output/recommendations.csv`
- 全候補予測: `trials/<trial>/output/response_spaces/*.csv`
- 知識違反診断: `trials/<trial>/output/knowledge_diagnostics.csv`
- 終了判定と履歴: `trials/<trial>/output/stopping_status.json`
- NN損失の実装: `src/knowledge_loss.py` → `src/hybrid/nn_component.py`
- 実行手順: `src/workflow.py`、共通入力: `src/trial_inputs.py`
- 全体統合: `src/hybrid/model.py`、推薦: `src/hybrid/optimizer.py`
- 禁止/優先領域: `src/policies/`、終了判定: `src/stopping/`

## v003で扱う知識

知識制約は次の4系統だけです。

| `type` | 意味 | 必須列 |
|---|---|---|
| `lower_bound` | 出力が指定値以上。`value=0`で「0以上」 | `target`, `value` |
| `monotonic_increasing` | 指定入力を増やすと対象出力は減らない | `target`, `wrt` |
| `monotonic_decreasing` | 指定入力を増やすと対象出力は増えない | `target`, `wrt` |
| `low_sensitivity` | 指定入力を1刻み変えた際の出力変化が許容値以内 | `target`, `wrt`, `tolerance` |

「変化しない」は `low_sensitivity` の `tolerance=0` で表せます。単調性と影響小は常に `target × wrt` の組で保持します。例えば `AA` が `P1` に対して単調増加でも、未指定の `P2` や `P3` へ同じ規則を推測適用しません。

### knowledge_constraints.csv

```csv
rule_id,type,target,wrt,value,tolerance,strength,enabled,note
K001,lower_bound,AA,,0,,3,true,AAは0以上
K002,monotonic_increasing,AA,P1,,,3,true,P1に対してAAは単調増加
K003,monotonic_decreasing,AA,P2,,,2,true,P2に対してAAは単調減少
K004,low_sensitivity,AA,P3,,0.5,2,true,P3を1刻み変えてもAAの変化は0.5以内
```

`target` は `problem.csv` の出力列、`wrt` は入力列です。空行や独自ヘッダーは使わず、不要な規則は `enabled=false` にします。

`lower_bound` の `wrt` は空欄にしてください。`tolerance` は0以上です。
単調性と影響小は、候補軸の隣接点を比較します。例えば範囲0～1・刻み0.6の
候補 `0, 0.6, 1` では、最後の短い区間 `0.6→1` も学習・診断の対象です。

## 知識強度

`strength` は1〜5です。出力の標準偏差で損失を無次元化してから、次の重みを掛けます。

| strength | 意味 | NN損失の重み | 動作 |
|---:|---|---:|---|
| 1 | 参考 | 0.1 | データを優先しやすい |
| 2 | 弱い知見 | 0.3 | 緩やかに誘導 |
| 3 | 標準 | 1.0 | 通常の知識制約 |
| 4 | 強い知見 | 3.0 | 知識を強く優先 |
| 5 | 必須 | 10.0 | 学習後も違反があれば推薦を公開しない |

実測値は知識と矛盾しても書き換えません。NN、最終Hybrid予測、実測値を別々に診断し、矛盾を `knowledge_diagnostics.csv` に残します。strength 5は最適化だけに頼らず、全候補グリッドのゲート検査を通過した場合だけ推薦を出力します。

必須ルールのモデル検査は、検査点が0件でも不合格になります。実測診断では、
比較できる隣接ペアが存在しない場合を `points=0` の未検証として記録します。
実測ペアがないこと自体では停止せず、NNとHybridの全候補検査を必須にします。

## 使用禁止範囲と好ましい範囲

範囲知識はNNの形を変える規則ではないため、`search_regions.csv` で推薦層へ適用します。

```csv
region_id,kind,parameter,lower,upper,lower_inclusive,upper_inclusive,strength,enabled,note
R001,forbidden,P1,2000,,false,true,5,true,P1>2000は実験しない
R002,preferred,P2,0,2,true,true,3,true,P2は0～2が好ましい
```

- `forbidden`: モデル学習には過去の実測値を残すが、新しい候補、実測最良値、改善期待値の比較基準、目標達成の判定からは除外
- `preferred`: 基本推薦スコアへ有界な倍率を掛ける。強度1〜5で `1.05, 1.10, 1.25, 1.50, 2.00` 倍
- 複数のpreferredが重なる場合は乗算し、最終倍率を2倍で上限化
- 同じ `region_id` の複数行はAND条件。異なるパラメータを組み合わせた領域も表現可能
- `lower_inclusive` / `upper_inclusive` で境界を含むか指定
- forbiddenは安全/実験可否の指定なのでstrengthにかかわらず絶対条件。通常はstrength 5を記録

処理順は「禁止候補の除外 → 基本獲得スコア計算 → preferred倍率 → 上位選択」です。preferredが低価値候補を無制限に押し上げたり、forbiddenを復活させたりはしません。

## モデルと推薦ロジック

```text
problem.csv + experiments.csv + knowledge_constraints.csv
        │
        ├─ 実測値を同一条件ごとに平均し、測定ノイズを推定
        ├─ 全候補から再現可能な知識評価点/隣接ペアを生成
        │
        ▼
NN損失 = データMSE + L2 + Σ(strength重み × 知識違反²)
        │
        ├─ d_loss/d_predictionをNNへ逆伝播
        ▼
残差GPを学習: 実測値 − NN予測
        │
        ▼
Hybrid平均 = NN予測 + GPデータ支持度 × 残差GP平均
Hybrid標準偏差 = 残差GP標準偏差
        │
        ├─ forbidden候補を除外
        ├─ 制約達成確率 × Expected Improvement
        ├─ preferred倍率
        └─ 重複を避ける分散選択
        ▼
recommendations.csv + response_spaces/*.csv + 各種診断
```

GPはNNの残差だけを学習します。測定点近傍ではデータでNNを補正し、未観測域では補正平均を弱めて知識付きNNへ戻します。一方、標準偏差は支持度で0へ潰さず、未観測域の探索可能性を残します。

現在のNNは追加依存を避けたNumPy実装です。知識損失は `calculate_rule_loss()` が損失と `d_loss/d_prediction` を返す境界に分離されています。将来PyTorch等へ移行し、物理式を追加する場合も、この境界へ新しい項を接続できます。

## 終了判定

終了は「即時停止」と「収束による停止推奨」を分離します。

### STOP_REQUIRED（いずれか1つ）

1. `max_additional_experiments` に到達
2. forbiddenを除く全候補を実測済み
3. 許可領域で制約を満たす実測目的値が `problem.csv` の `target` に到達

この3条件は学習前に判定し、該当時は新しいモデル学習と推薦を行いません。
残り追加実験予算が `--n` より少ない場合は、残予算に合わせて推薦件数を減らします。
追加実験数の起点は最初のrunの実験行数で、停止設定を変更しても保持します。

学習後の必須知識検査が不合格の場合も `STOP_REQUIRED` を記録し、推薦を公開しません。
この場合はCLIがエラー終了し、停止理由と知識診断に不合格ルールを残します。

### STOP_RECOMMENDED（次の4条件をすべて満たす）

1. 許可候補のGPデータ支持度カバー率が十分
2. 直近の制約適合済み実測ベストがほとんど改善していない
3. 正規化した最上位推薦スコアが小さく、変動も小さい
4. 全許可グリッド上の予測最適条件が大きく動かない

履歴不足なら必ず `CONTINUE` です。4は「未測定候補からの次推薦」ではなく、毎runの全グリッド予測最適点を比較します。測定済み点が候補から消えるだけで停止判定が揺れるのを避けるためです。`STOP_RECOMMENDED` は判断材料であり、自動的な安全保証ではありません。

収束判定では、同じ実験データの再実行を1つの状態として扱います。CSVの行順だけの
変更も新しい実験には数えません。問題・知識・探索領域・停止設定の変更、既存実験の
訂正や削除があれば、その時点から収束履歴を取り直します。過去のrun記録は保持します。
入力内容の識別情報がない旧形式の履歴も保存しつつ、更新後の収束根拠には使いません。

既定値は `stop_settings.csv` に説明付きで生成されます。主要値は `min_history=5`、`patience=3`、支持度閾値/カバー率ともに`0.8`、目的値改善許容`0.001`、正規化スコア上限`0.05`、条件移動距離`0.05`です。目的値の単位や候補刻みに応じてtrialごとに調整してください。

## 実行手順

リポジトリ直下で実行します。

```powershell
python main.py --version v003 --new trial_003
```

生成された `problem.csv`、`knowledge_constraints.csv`、`search_regions.csv`、`stop_settings.csv` を編集後、実験CSVを準備します。

```powershell
python main.py --version v003 --prepare trial_003
```

`v003/trials/trial_003/data/experiments.csv` に実測値を入力し、推薦を実行します。

```powershell
python main.py --version v003 --run trial_003 --n 3
```

推薦条件を実験し、同じ `experiments.csv` へ結果を追記して再実行します。各runの予測空間と停止履歴は上書きせず蓄積します。

## 出力

| 出力 | 内容 |
|---|---|
| `recommendations.csv` | 条件、NN/Hybrid予測、標準偏差、制約確率、基本/優先補正後スコア、根拠 |
| `recommendations_history/*.csv` | 再実行前に保存した過去の推薦 |
| `response_spaces/hybrid_response_space_run_*.csv` | 全候補のNN/Hybrid予測、標準偏差、GP支持度、実験可否、preferred倍率 |
| `knowledge_diagnostics.csv` | `nn` / `hybrid` / `observed` ごとの違反率、平均/最大違反、損失 |
| `stopping_status.json` | `CONTINUE` / `STOP_RECOMMENDED` / `STOP_REQUIRED`、判定指標、run履歴 |

予測は実測値ではありません。特に弱い知識ルールは「傾向を促す」もので、違反率が必ず0になるとは限りません。

`--run` の開始時に、前回の推薦CSVを `recommendations_history/` へ移します。
今回の入力検査・知識診断・解空間と停止状態の保存が完了してから、新しい
`recommendations.csv` を公開します。停止やエラーの場合は現行の推薦CSVを残しません。
前回の結果は履歴から確認できます。

## 検証

標準ライブラリのunittestで製品コードを検査します。各版が同じ `src` パッケージ名を
使うため、バージョンごとに別プロセスで実行してください。

過去の探索性能と今回の構造整理の確認結果は [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) に保存しています。

```powershell
python -m unittest discover -s v003/tests -p "test_*.py" -v
python -m unittest validation/single/tests/test_v003_validation.py -v
```

`validation/single/` のoracleを実験装置の代わりに呼び、15回の推薦・仮想実験とPNG作成を行えます。

```powershell
python -m validation.single.src.checks.v003_validation --mode laser --optimizer-root v003 `
  --trial trial_laser_welding_v003_validation --iterations 15 --recommendations 1

python -m validation.single.src.checks.v003_validation --mode synthetic --optimizer-root v003 `
  --synthetic-trial trial_synthetic_constraints_v003
```

レーザー検証は収束履歴、oracle上の真値との全候補誤差、適合判定、最終予測空間をJSON/Markdown/PNGへ保存します。合成検証は「非負」「P1に対する単調増加」「P2の影響小」を独立に検査します。

## 既知の限界

- 離散グリッド探索であり、候補数上限があります。
- ルールは指定範囲の全域に適用され、条件付き知識（例: P2が一定範囲のときだけ単調）は未実装です。
- strength 5で実測と知識が矛盾すると、安全側として推薦が止まります。入力ミスか、知識の適用範囲/強度を見直してください。
- 物理式、等式制約、論理式パーサーはv003の対象外です。
- 推薦は量産条件や設備安全を保証しません。禁止範囲と実測確認を併用してください。
