# 銅・アルミ電気接続部品の連成検証モデル

このケースは **文献の物理構造を採用した未校正の合成oracle** です。
実設備の加工レシピや品質の定量予測を保証するモデルではありません。
操作数を増やしただけの独立な関数ではなく、切断端・成形履歴・接触状態が
下流の融込みと最終電気／機械品質に影響する検証を目的とします。

## ワークと工程

主ケースは無めっきCu-OF下材とAl99.5上材、各公称1 mm、幅40 mm、
長さ80 mm、重ね12 mmの段差端子です。CuとAlをそれぞれ切り出し、2箇所を曲げ、
組立で合流させて1080 nmのCWレーザーで接合します。

```text
Cu原材 → cut_cu → form_cu ┐
                           ├→ assembly → welding → 検査
Al原材 → cut_al → form_al ┘
```

最終検査は独立した加工操作ではなく、`inspection.py`が担当します。
`cu_cu`／`al_al`は同じ幾何・同じ光源の対照ケースです。ノード名は枝の識別子なので、
`cu_cu`の`cut_al`枝にもCuが入ります。材種フラグはvariantで固定し、最適化で変更しません。
Cu-OFの降伏90 MPaとAl99.5の降伏85 MPaは**成形履歴を持たせるための代表仮定**です。
特定の調質・供給ロットを論文から特定した値ではありません。Al1050-H14板の情報を
参考にしていますが、溶接論文試片の調質と同一であるとは扱いません。

## 文献から採用した構造と、採用しない数値

| 対象 | 一次資料 | 採用内容と境界 |
|---|---|---|
| 主溶接仕様 | [Jarwitz et al., Metals 2018](https://www.mdpi.com/2075-4701/8/7/510) | Al99.5上／Cu-OF下各1 mm、1080 nm、焦点径280 µm、Rayleigh長5 mm。接合幅・浅いCu融込み・混合と抵抗の関係を採用 |
| 機械切断端 | [Bohdal et al., Materials 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7412372/) | 端部のバリ・損傷・変形と後加工への影響。AA6111-T4せん断の係数を切削Cu／Al1050へ移植しない |
| 曲げ成形 | [Copper Development Association](https://live.copper.org/publications/pub_list/pdf/a1360.pdf) | R/t、強度、圧延方向と曲げ適性、弾性回復によるspringback、overbend／restrikeの意味 |
| Al圧延板 | [Hulamin 1050 datasheet](https://www.hulamin.com/sites/default/files/downloads/1050%20Data%20Sheet.pdf) | 材種・調質・厚さに依存する物性と曲げ半径。押出材H112の値を圧延板へ混用しない |
| 組立接触 | [Cu–Al green-laser overlap study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9921288/) | 治具でtechnical zero gapを確保する必要性。Niめっき薄板・515 nmの実験係数を主仕様へ移植しない |
| 電気経路 | [Copper for Busbars](https://www.kupfer.de/wp-content/uploads/2019/11/Copper-for-Busbars.pdf) | 基材経路、表面、接触と電流集中の区別。ボルト接触の圧力改善を完成溶接の抵抗と混同しない |

主溶接論文では送り6 m/minが一定です。直線照射は1.5–3.25 kW、
振動照射は3.25 kW、振幅0.25–1 mm、周波数200–1000 Hz、長さ35 mmです。
oracleはこれを構造の基準にします。送り4–8 m/min、焦点±2 mm、長さ25–35 mm、
位置補正、rampの同時変更は**合成検証で追加した探索領域**で、実験で裏付けた応答面ではありません。
緑色光、Niめっき、0.2／0.3 mmタブ、Al0.4 mm／Cu1 mmケースの係数は混用していません。

## 44操作と探索規模

| ノード | 操作数 | 操作 |
|---|---:|---|
| cut_cu / cut_al | 各7 | 切断速度、一刃送り、保持力、仕上げ取り代、仕上げ送り、バリ取り深さ、長さ補正 |
| form_cu / form_al | 各7 | 2曲げのoverbend、punch半径、die開口、保持力、ram速度、接合面coining圧力 |
| assembly | 8 | x/y/回転補正、上下clamp力、clamp span、上下清掃dose |
| welding | 8 | 出力、送り、焦点、振幅、周波数、長さ、照射位置、出力ramp長 |

各操作の境界・単位・代表値は`STAGE_MANIFESTS`が宣言します。標準は各4水準で、
全直積は`4**44 = 309485009821345068724781056`条件です。`axes()`は44個の軸だけを返し、
この直積を生成しません。比較試験や学習ワークの抽出も予算付きサンプリングで行います。
工具半径、die開口、治具spanは設備段取りを伴う操作で、現場の即時再調整では固定する必要があります。
各ノードの局所目的は加工時間最小化です。速さと仕上げ・寸法・損傷のtradeoffを通じて、
局所条件だけの選択が最終品質と一致しない可能性を持たせています。勝者は埋め込んでいません。

## 物理式の責務と単位

1. 切断では送り・速度・原材強度による負荷を計算し、仕上げとバリ取りが
   バリ／端部損傷を減らします。保持剛性が寸法ずれ・camberへ影響します。
2. 成形ではひずみを`t/(2R+t)`、回復の尺度を`σ/E × R/t`として求め、
   2曲げ角・薄肉化・加工硬化・残留応力・接合面高さを関連させます。
   完全な3D塑性／破断FEAではなく、経験係数を持つ簡約モデルです。
3. 組立では材料の`E t³/span³`に基づく剛性尺度、clamp力、姿勢とバリから
   3点のgap fieldを導出します。隙間を自由操作として与えません。
   線形補間した3点fieldであり、実接触面全体の最大隙間を保証するモデルではありません。
   過大荷重による永久変形、残存弾性エネルギー、接触による治具放熱も区別します。
4. 溶接は`P/v`だけでなく、焦点と強度閾値、吸収、振動の掃引幅、接触放熱を用います。
   上材厚さを貫いた部分が下材融込みへ進み、隙間が接合を阻害します。
   密度を掛けた下材の混合質量比から異材のIMC proxyを計算します。
   吸収／融込み／欠陥／IMCの定量係数は合成値です。気液流動・相図・反応拡散の解ではありません。
5. 冷却・治具解除後の抵抗は基材2経路の`ρ L/A`と界面項の和です。
   `ρ[nΩ m] × L[mm] / A[mm²]`の数値は`µΩ`になります。
   押し付けただけの未接合面を低抵抗の完成品質として扱いません。
   せん断力は欠陥・IMCを含む接合経路と、損傷した母材net sectionの小さい方です。
   温度上昇は固定試験電流200 Aと熱抵抗0.9 K/Wで`I²R Rth`を求めます。
   動的な電熱連成・長期疲労・腐食・熱サイクル寿命はこの初版の検証対象外です。

最終仕様は抵抗≤60 µΩ、力≥1300 N、温度上昇≤2.3 K、未接合率≤0.15、
気孔率≤0.09、IMC proxy≤6 µm、解除変形≤0.35 mmです。
これらはベンチマークの合成仕様で、法規・製品規格の認証閾値ではありません。
`final.feasible`は最終仕様、`final.all_local_feasible`は工程制約、
`final.line_feasible`は両方を満たす完成品です。

## 観測可能性とワークの消費

oracleは加工状態の真値を返し、測定noiseを含めません。
`observe_stage`／`inspect_final`はmanifest whitelistの測定値だけを返します。
残留応力、損傷、塑性履歴、膜厚nmでの酸化物、汚染率、界面混合、
接触／放熱などを生産ワークの完全な直接観測として渡しません。
材料定数はロット情報や別試片の代表値であり、加工後の正確な局所真値を意味しません。

非破壊検査は冷却後の四端子抵抗・温度・寸法など、破壊検査は別試片のせん断力・断面・
IMC proxy・気孔です。`destructive=True`の試片は`consumed=True`で後工程へ渡せません。
潜在oracle値は破壊検査でも自動的にすべて開放しません。

`incoming_proxy(stage_id, inputs, regime='proxy')`は不可視な流入状態を固定代表値へ置換します。
これは**低情報の対照方式**で、潜在状態を推定できたことを意味しません。
`regime='complete'`はoracle真値を含む上限比較です。root検証driverは内部真値を使う模型検証と、部分観測時の未確定扱いを分けて報告し、
上流制約に合格した実現可能ワーク由来の流入状態だけを学習へ使います。
追加の真値状態を知る必要がある場合は、別試片・追加センサー・実験の費用と消費を扱います。

## APIと検証

```python
from validation.multistage.electrical_interconnect.pipeline import evaluate_line, line_config, axes
from validation.multistage.electrical_interconnect.stages import STAGE_MANIFESTS, evaluate_stage
from validation.multistage.electrical_interconnect.inspection import inspect_final

result = evaluate_line({'welding.laser_power_kw': [2.0, 2.5, 3.0]})
observed = inspect_final(result['final'], destructive=False)
```

`evaluate_stage`は必須名・有限性・inclusive範囲・材種0/1・broadcastを検査します。
`evaluate_line`は全操作と原材contextを同じshapeへそろえてDAGを計算し、
`stage_inputs`、`stage_outputs`、`local_feasible`、`final`を返します。
部分操作指定では残りを`DEFAULT_CONTROLS`で補います。
`source_hashes()`は物理・接続・観測・READMEのSHA-256を実行結果へ記録できます。

物理回帰テストは`validation/multistage/tests/test_electrical_interconnect.py`です。
各工程の操作端点、3variantの良否混在、上流変更の伝播、力／抵抗の物理上限、
noise分離、検査whitelist、潜在情報漏洩防止を確認します。
このテストの成功は、文献試片や実設備の定量応答を再現したという意味ではありません。

学習bundleには最大角度＝符号付き2角度の絶対値の最大、最大隙間＝3点隙間の最大、
抵抗＝基材経路＋界面の`derived_outputs`定義を渡します。v005はこれらの点予測を
数値関係で整合させます。変換後の期待値・標準偏差の厳密な同時分布は扱いません。
