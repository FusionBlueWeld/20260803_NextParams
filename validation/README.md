# 検証コードの案内

人間による目視レビューを前提に、可読性と責務の分かりやすさを重視して整理しています。

| フォルダ | 検証対象 | 詳細 |
|---|---|---|
| `single/` | レーザー溶接など6工程を個別に評価し、各バージョンの探索を比較 | [単工程の説明](single/README.md) |
| `multistage/` | 塗工3工程、Cu/Al電気接続6ノード、v004/r001・v005/r002の連結探索 | [複合工程の説明](multistage/README.md) |

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

追加検証は、例えば
`python -m validation.single.src.checks.v003_validation --help` のようにモジュールで起動します。
Pythonから共通APIを使う場合の入口は `validation.single.src.simulators` です。

## 実行方法

以下は作業フォルダ直下で実行します。

```powershell
python -m validation.single.run list
python -m validation.multistage.run list
python -m validation.multistage interconnect audit --samples 1024
python -m unittest discover -s validation -t . -v
```

共通テストは各 `tests/`、単工程の物理式テストは工程内の `test_model.py`、
パッケージ配置・起動の回帰テストは `validation/tests/` にあります。

2026-10-08までの単工程生成結果は [`_old/validation/single/results/README.md`](../_old/validation/single/results/README.md) に整理しました。
新しい検証結果は従来どおり `validation/single/results/` に保存します。検証コードは過去の保存結果を自動で学習データとして読み込みません。

## 手入力GUI

`python validation/konogui.py` で起動します。従来の左入力・中央計算・右出力の構成に、
電気接続の独立6工程と銅／アルミ・銅／銅・アルミ／アルミの3連結ラインを追加しています。
連結ラインは44操作＋14原材条件を工程別に入力し、途中工程と最終品質を右欄で確認できます。
材種は選択したラインに固定されます。プルダウンの先頭には
「複合工程全体｜電気接続 銅／アルミ」「銅／銅」「アルミ／アルミ」の3ライン、
続いて電気接続の6工程と従来の10シミュレータを並べ、19項目の一覧を表示します。
起動時は銅／アルミの連結ラインが選択されます。

このGUIは物理式を直接計算します。v005・r002や保存済みbundleの読み込みは不要です。
左欄を変更して「計算」を押すと、連結ラインでは前工程の加工状態が後工程へ伝播します。
「結果JSON保存」は入力と物理計算・検査結果を保存します。
予測モデルの学習や連結最適化は、各バージョンのAPI・CLIで実行します。

新しい物理モデルの「検査ノイズあり」は非破壊検査項目だけを表示し、
内部状態・破壊検査値は未測定表示になります。このとき合否・制約余裕は真値基準の参考値です。
物理モデルと検査ノイズは未校正の合成モデルです。
`python validation/konogui.py --smoke-test` で画面を開かず19モデルを確認できます。
