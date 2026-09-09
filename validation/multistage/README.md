# 複合工程物理シミュレータ

`functional_coating` は、塗工 → 熱風乾燥 → 熱硬化の3工程を個別モデルとして評価し、`pipeline.py` で工程間の状態を受け渡す検証用ラインです。既存の `validation/single/` 個別工程検証とは独立したパッケージです。

## 構成と各ファイルの役割

```text
validation/multistage/
├─ README.md / DESIGN.md / VALIDATION_REPORT.md / requirements.txt
├─ run.py / __main__.py      起動入口
├─ src/
│  ├─ cli.py                引数の解釈と物理評価の振り分け
│  ├─ settings.py           プロジェクト・結果フォルダの基準パス
│  ├─ stages.py             Stage（1工程の操作条件・流入状態・モデル）
│  ├─ common.py             物理モデル共通の入力・出力検査
│  └─ checks/               v004・r001を使う検証シナリオ
├─ functional_coating/
│  ├─ coating/              塗工の物理式とmanifest
│  ├─ drying/               乾燥の物理式とmanifest
│  ├─ curing/               硬化の物理式とmanifest
│  ├─ process_manifest.json 工程接続・最終仕様の定義
│  ├─ pipeline.py           同じ製品の状態を次工程へ渡す
│  ├─ benchmark.py          oracleの全候補監査と比較基準
│  ├─ robustness.py         条件ばらつきのサンプリング評価
│  └─ fault_scenarios.py     既知の故障を加える採点用シナリオ
├─ tests/                   物理モデル・接続・診断・窓合否のテスト
└─ results/                 保存済み検証結果と新しい出力
```

v002の `src/cli.py` と同様に、利用者の操作は `src/cli.py` から追えます。
物理モデル固有のまとまりは `functional_coating/`、学習済みモデルを評価する処理は
`src/checks/` に分けています。v002の実装コードを複製・変更する構成ではありません。

### 処理の読み順

1. `run.py` → [src/cli.py](src/cli.py) で操作を確認します。
2. [src/stages.py](src/stages.py) → 各工程の `manifest.json` と `physics_model.py` で操作条件と流入状態を確認します。
3. [pipeline.py](functional_coating/pipeline.py) → [process_manifest.json](functional_coating/process_manifest.json) で接続先と最終仕様を確認します。
4. [benchmark.py](functional_coating/benchmark.py) で物理oracleの比較基準を確認します。
5. 必要な検証だけ `src/checks/` から選び、対応する `results/` を読みます。

```text
操作条件・原料状態 → 塗工 → 湿潤膜の状態 → 乾燥 → 乾燥膜の状態 → 硬化 → 最終仕様
```

配列の同じ行は同じ仮想製品です。流入状態は上流の結果であり、下流が自由に選べる
操作条件とは区別します。範囲外の接続を黙ってクリップせず、評価不能として扱います。

### 検証シナリオの選び方

| `src/checks/` のファイル | 確認する内容・必要な準備 |
|---|---|
| `v004_validation.py` | v004の工程単体学習・bundle入出力。新規trialを作成 |
| `r001_validation.py` | oracle bundleをr001へ渡し、直接の物理接続と照合。新規trialを作成 |
| `learned_r001_validation.py` | 操作条件と流入状態を変えて学習し、学習済み工程を接続 |
| `stage_closed_loop_validation.py` | 工程ごとの反復探索と学習結果を比較 |
| `validated_chain_retest.py` | コード内で指定された学習済みbundleを再接続 |
| `probabilistic_r001_validation.py` | 指定trialについて入力ばらつきの確率評価をoracleと比較 |
| `r001_diagnostic_validation.py` | 保存済みの個別工程接続trialから予測・NG診断・回復を評価 |
| `calibrate_connected_window.py` | 同trialの残差から連結窓を校正し、校正結果を保存 |
| `connected_window_validation.py` | 同trialの一変数窓について誤許容・回収率を採点 |
| `window_center_joint_validation.py` | 同trialの窓中心と複数変数同時変更の成立を採点 |

最後の4シナリオは `r001/trials/trial_validated_individual_chain_r001/` を使います。
通常のCLI監査とは異なり、保存済みtrial・学習bundle・出力を必要とし、一部は既存結果を
更新します。実行前に各ファイル冒頭のパスと `run()` の入出力を確認してください。

```powershell
# 引数不足時には必要な引数が表示されます。以下は新規trial名と新規結果先の例です。
python -m validation.multistage.src.checks.r001_validation trial_review_oracle validation/multistage/results/review_oracle.json
```

合否の意味は「完全観測の合成物理モデルに対する検証」です。良品率の校正や実機での
保証とは区別します。過去の検討事項・未実装表記は [DESIGN.md](DESIGN.md) に残しています。

## CLI

リポジトリ直下から実行します。`evaluate-line` は原料状態を省略した場合、
粘度1.6 Pa·s、固形分率0.50、気泡率0.005の基準原料を使います。

```powershell
python -m validation.multistage.run list

python -m validation.multistage.run evaluate-stage coating --set `
  coating_gap_um=210 line_speed_m_min=18 web_tension_n=100 `
  incoming_viscosity_pa_s=1.6 incoming_solids_fraction=0.5 incoming_bubble_fraction=0.005

python -m validation.multistage.run evaluate-line --set `
  coating_gap_um=210 line_speed_m_min=18 web_tension_n=100 `
  air_temperature_c=90 air_speed_m_s=5.5 residence_time_min=13 `
  oven_temperature_c=140 hold_time_min=60 nip_pressure_mpa=0.325

python -m validation.multistage.run audit --samples 512 `
  --output validation/multistage/results/reference_audit.json
```

`evaluate-stage` は個別工程の操作条件と流入状態、`evaluate-line` は3工程を
通した全中間状態と最終合否をJSONで返します。`audit` は9操作条件×各3水準の
19,683候補を評価し、個別Best連結と全体調整候補、再現可能な良品確率を保存します。
既存の出力パスは誤上書きを避けるため拒否します。

## テスト

```powershell
python -m unittest discover -s validation/multistage/tests -t . -p "test*.py" -v
```

次を自動検査します。

- 各工程の単体テスト（物理的な傾向、境界、有限性、決定論性）
- scalar／vector／多次元配列のbroadcast契約
- 範囲外、NaN、Inf、形状不一致の入力拒否
- 塗工 → 乾燥 → 硬化の接続値が正しく伝播すること
- 個別Best連結が最終NGになり得ることと、全体調整条件の良品率改善
- ロバスト性乱数評価とoracle真値の分離・再現性
- 原因工程が既知のドリフト介入シナリオ
- 最終仕様、`quality_margin`、`feasible` の監査

既存 `validation/single/` のテスト・CLIとは別に実行できる構成を維持します。

確定した基準値と検証範囲は [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) を参照してください。

## 現在の対象範囲

このパッケージは物理oracleと比較用の粗い全体予測空間を提供します。学習済みv004 bundleの
合成、未学習4,096条件での最終予測、既知NGの工程順位付けと全工程回復は
[`r001_diagnostic_validation.py`](src/checks/r001_diagnostic_validation.py)で検証できます。詳細結果は
[`r001_diagnostic_validation.json`](results/r001_diagnostic_validation.json)です。v003への
自動接続、潜在原因の同定、追加実験の選択はまだ実装していません。

r001の連結ウィンドウ断面は[`connected_window_validation.py`](src/checks/connected_window_validation.py)で
物理oracle境界と照合します。実世界校正を合否条件に含めず、完全観測の決定論oracleに対して、
推定範囲が危険側へはみ出さないことと、真の許容範囲をどの程度残すかを別々に評価します。
独立した連結残差校正・方式選択データを各4,096点用いた片側残差方式では、危険側はみ出し0、
平均oracle窓回収率80.7%、走査点のFP 0でした。平均のみ／バッファ込み／支持度込みの縮小要因も
分離して保存します。[`window_center_joint_validation.py`](src/checks/window_center_joint_validation.py)は、
窓中心選択によるoracle対称余裕の改善と、9変数同時変更4,096点・512頂点・二変数断面を独立oracleで
照合します。`rho`箱では予測側のFP 0、FN 88であり、有限標本検証であって連続領域の保証ではありません。
`incoming_state` と操作可能な `controls` をmanifestで分離しているため、将来の
個別最適化・ロバスト最適化・後戻り実験選択で役割を混同せず利用できます。
