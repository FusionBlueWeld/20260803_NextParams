# フライス切削・高次元収束ベンチマーク

## 目的と位置付け

このフォルダは、8入力・4水準ずつの高次元パラメータ空間で、制約付き最適化器の収束挙動を比較するための決定論的な合成オラクルである。全候補数は `4^8 = 65,536`。`physics_model.py` は NumPy だけで計算し、乱数、ファイル書き込み、外部プロジェクトの実行時依存、隠れた状態を持たない。

実機の加工条件や工具寿命を保証する校正済みモデルではない。係数は、物理的に追跡しやすい入力依存性、共振による非単調性、目的と制約の連成を検証できるように選んだ合成値である。

## 入力と固定条件

今回の8条件は固定値ではなく、すべてモデル入力である。被削材は代表的な鋼材とし、基準比切削抵抗 `kc0=1850 N/mm²`、主軸効率、冷却、材料ロット差、工具姿勢は固定・省略する。

| column | 意味 | 単位 | 範囲 | step | 水準数 |
|---|---|---:|---:|---:|---:|
| `spindle_speed_rpm` | 主軸回転数 | rpm | 1800–7200 | 1800 | 4 |
| `feed_per_tooth_mm` | 一刃当たり送り `fz` | mm/tooth | 0.040–0.160 | 0.040 | 4 |
| `axial_depth_mm` | 軸方向切込み `ap` | mm | 0.50–3.50 | 1.00 | 4 |
| `radial_engagement_mm` | 半径方向切削幅 `ae` | mm | 5–20 | 5 | 4 |
| `cutter_diameter_mm` | カッタ径 `D` | mm | 30–60 | 10 | 4 |
| `tooth_count` | 刃数 `z` | tooth | 2–8 | 2 | 4 |
| `nose_radius_mm` | 有効ノーズ半径 `rε` | mm | 0.40–1.60 | 0.40 | 4 |
| `tool_wear_mm` | 摩耗量サロゲート | mm | 0–0.30 | 0.10 | 4 |

`problem.csv` のstepは各軸で上下限を含む4水準を厳密に作る。したがって直積は `4 × 4 × 4 × 4 × 4 × 4 × 4 × 4 = 65,536` 点である。

## 式と出力

材料除去率は、`ae`、`ap` を mm、`fz` を mm/tooth、`n` を rpm として、指定のメートル法式を使う。

`Q = ae × ap × fz × n × z / 1000`

`Q` は cm³/min。切削速度は `vc = π × D × n / 1000`（m/min）で計算する。基準比切削抵抗から、切削速度、送り、切削幅/径比、摩耗を明示的な滑らかな合成補正として加え、

`kc = kc0 × speed_factor × chip_factor × engagement_factor × wear_factor`

とする。主軸動力は、`kc` を N/mm²、`Q` を cm³/min として、

`Pc = Q × kc / 60000`

（kW）で計算する。このため、MRRに直接現れないカッタ径や摩耗も、切削速度・比切削抵抗を介して動力へ効く。

粗さ `Ra`（µm）は、まず送りマークの幾何学的寄与を

`Ra_feed = 1000 × fz² / (8 × rε)`

とし、摩耗倍率、切込み、切削幅/径比のプラウイング寄与を加える。さらに歯通過周波数

`ft = n × z / 60`（Hz）

に対する3つの減衰モードを、

`H_i = 1 / sqrt((1 - (ft/fi)²)² + (2ζ_i ft/fi)²)`

で計算する。モードは160、300、440 Hz、減衰比と重みはモデル内に明示している。MRR、比切削抵抗、切削幅/径比、摩耗で負荷をスケールし、動的粗さ寄与とする。これは `dataset_08()` の減衰振動の構造を参考にした経験的強制応答サロゲートであり、再生型びびりの安定ローブや実機の加工可否を主張するものではない。

出力とベンチマーク規則は次のとおり。

- 目的: `material_removal_rate_cm3_min` 最大化
- 制約: `roughness_ra_um <= 7.50 µm`
- 制約: `spindle_power_kw <= 14.00 kW`

閾値は全候補を一律に通す/落とすことがないように設定している。低負荷の良品領域、高送り・小ノーズ半径や高負荷の不適合領域、共振近傍を含む非単調な速度依存を同時に残している。

## 参照データと注意

参照した合成データ生成器は `C:/Users/tsuts/Documents/Codex/20260623_ml_dataset_generation/src/dataset_generators.py` の `dataset_18()`（切込み、送り、回転数、工具摩耗に対する主軸電流波形）と `dataset_08()`（質量・ばね・ダンパの減衰振動）である。データセットはコピーせず、関連する変数の組み立てと減衰応答の考え方だけを参照している。

本モデルは、絶対値や制約閾値を実機へ外挿してはならない。材料、工具、機械剛性、工具突出し、冷却、切削方向、摩耗機構、再生効果を含む実機モデルには追加の計測・同定・校正が必要である。

## 参考資料

- Sandvik Coromant, *Formulas and definitions for milling (metric)*: https://cdn.sandvik.coromant.com/files/sitecollectiondocuments/services/metal-cutting-e-learning/formulas-and-definitions/formulas-and-deinitions-for-milling-metric-enu.pdf
- プロジェクト内の参照コード: `C:/Users/tsuts/Documents/Codex/20260623_ml_dataset_generation/src/dataset_generators.py`（`dataset_18()`、`dataset_08()`）
