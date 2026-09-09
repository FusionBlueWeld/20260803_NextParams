# 検証コードの案内

人間による目視レビューを前提に、可読性と責務の分かりやすさを重視して整理しています。

| フォルダ | 検証対象 | 詳細 |
|---|---|---|
| `single/` | レーザー溶接など6工程を個別に評価し、各バージョンの探索を比較 | [単工程の説明](single/README.md) |
| `multistage/` | 塗工→乾燥→硬化の状態接続、v004・r001の連結探索 | [複合工程の説明](multistage/README.md) |

各フォルダに実行器、物理モデル、テスト、検証結果をまとめています。
物理式と設計根拠は各 `DESIGN.md` を参照してください。

## v002とそろえた読み方

部内で共有しているv002の「入口・設定・読込・計算・出力」の粒度を基準にしています。
両領域の `run.py` は起動だけを担当し、引数の解釈は `src/cli.py`、共通設定は
`src/settings.py`、用途別の追加検証は `src/checks/` に置きます。
単工程では読書きを `data_loader.py`、採点を `metrics.py`、比較表出力を `reporting.py` に分けています。

まず各READMEの「構成と各ファイルの役割」→「処理の読み順」を読み、その後にコードを追ってください。
関数のコメントでは、入力・出力と、その検査が必要な理由を説明しています。
物理式の係数や合否閾値、乱数seedの意味は、工程別の説明と設計資料で確認できます。

| 変更したいもの | 主な変更先 |
|---|---|
| 利用者が指定するコマンド | 各 `src/cli.py` |
| フォルダの基準パス・共通定数 | 各 `src/settings.py` |
| 単工程の比較手順・採点・保存 | `single/src/benchmark.py`・`metrics.py`・`reporting.py` |
| 物理モデル・接続 | 工程フォルダ、複合工程の `functional_coating/` |
| 特定機能の検証シナリオ | 各 `src/checks/` |

通常のCLI名は今回の `src/` 整理でも維持しています。追加検証は、例えば
`python -m validation.single.src.checks.v003_validation --help` のようにモジュールで起動します。
Pythonから共通APIを使う場合の入口は `validation.single.src.simulators` です。

## 実行方法

以下は作業フォルダ直下で実行します。

```powershell
python -m validation.single.run list
python -m validation.multistage.run list
python -m unittest discover -s validation -t . -v
```

共通テストは各 `tests/`、単工程の物理式テストは工程内の `test_model.py`、
パッケージ配置・起動の回帰テストは `validation/tests/` にあります。

旧 `validation.run` は `validation.single.run`、旧 `multistage_validation` は
`validation.multistage` に変更しました。保存先もそれぞれの配下へ移しています。
外部スクリプトで旧パスやモジュール名を指定している場合は更新してください。

保存済みの結果JSONとソースハッシュは実行当時の記録として保持しています。
記録内の旧パスは、単工程なら `validation/single/`、複合工程なら
`validation/multistage/` に読み替えてください。ハッシュは移動前のコードのものです。

## 今回の整理で確認した範囲（2026-09-09）

- 自動テスト88件：既存84件と、パス・起動・追加検証のimportに関する4件が合格。
- 変更前後の監査比較：単工程6モデルの全候補・初期点・出力範囲・最良値、および複合工程の監査結果が数値一致。
- v002実CLI：熱硬化・seed 0・推薦1件・反復1回の比較処理が完了。最終予測と比較表出力まで確認。
- 資料の相対リンクにリンク切れなし。v002の実装・テストコードには変更なし。

実CLIの結果は [比較表](single/results/refactor_review_20260909/comparison.md) にあります
（ローカル生成物のためGit管理外）。全モデル再学習や長時間の本番規模検証は今回の確認範囲に含みません。

分割後の実行器ハッシュには `src/` 配下の設定・読書き・採点・シナリオも含めます。
保存済みの結果・ハッシュは書き換えず、新規実行分から新しい構成を記録します。
