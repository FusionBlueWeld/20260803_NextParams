# 銅電解めっき・高次元収束ベンチマーク

## 目的と範囲

`physics_model.py` は、銅電解めっきの最適化・高次元収束を検証するための、NumPy-only、完全決定論の合成真値関数である。実験データや外部プロジェクトのランタイムを読み込まず、乱数、ファイル書き込み、隠れた状態を持たない。実機の設計値・品質保証・生産条件を置き換えるものではなく、実機校正前のベンチマーク専用である。

## 入力（8軸、各4水準）

| 入力 | 単位 | 範囲 | 水準 |
| --- | --- | --- | --- |
| `current_density_a_dm2` | A/dm² | 1--7 | 1, 3, 5, 7 |
| `bath_temperature_c` | °C | 20--50 | 20, 30, 40, 50 |
| `plating_time_min` | min | 5--35 | 5, 15, 25, 35 |
| `copper_concentration_mol_l` | mol/L | 0.4--1.6 | 0.4, 0.8, 1.2, 1.6 |
| `agitation_speed_m_s` | m/s | 0.05--0.35 | 0.05, 0.15, 0.25, 0.35 |
| `electrode_gap_cm` | cm | 0.5--2.0 | 0.5, 1.0, 1.5, 2.0 |
| `duty_cycle` | 1 | 0.4--1.0 | 0.4, 0.6, 0.8, 1.0 |
| `bath_ph` | 1 | 1.5--3.0 | 1.5, 2.0, 2.5, 3.0 |

各軸の直積は `4^8 = 65,536` 候補である。`problem.csv` の `step` は上下限を含めて各軸を厳密に4水準にする。入力はスカラー、配列、相互にbroadcast可能な多次元配列を受け付け、非有限値と範囲外値は `ValueError` とする。

## 出力、目的、制約

* `thickness_error_um`: 目標膜厚 20 µm との差の絶対値。objective、minimize。
* `roughness_ra_um`: 算術平均粗さの合成指標。constraint、`<= 0.65 µm`。
* `current_efficiency`: 電流効率。constraint、`>= 0.80`。
* `deposit_thickness_um`: 析出膜厚。monitor。
* `cell_voltage_v`: セル電圧。monitor。

## 固定条件

酸性銅浴、銅カソード、固定電極面積、直流電源の平均条件、電極材料、初期表面状態、電極配置、工具・治具、液量、外部冷却、前処理を固定する。物性定数は銅モル質量 `M=0.063546 kg/mol`、銅イオン価数 `z=2`、銅密度 `ρ=8960 kg/m³`、ファラデー定数 `F=96485 C/mol` とする。浴組成の変化、蒸発、補給、異物、気泡付着、電極消耗、工具摩耗、浴の経時劣化は扱わない。

## 物理式と経験補正

電流密度を `j` [A/dm²]、時間を `t` [min]、duty を `D` [1] とする。内部単位を `j_m2=100j` [A/m²]、`t_s=60t` [s] に変換し、Faraday 則による析出膜厚を次で計算する。

`h [µm] = η j_m2 [A/m²] t_s [s] D M [kg/mol] / (z F [C/mol] ρ [kg/m³]) × 10⁶`

ここで `η` は電流効率である。したがって、時間と平均通電量（`j×D`）が膜厚を増やし、高電流では効率と輸送利用率が低下する。

固定撹拌の物質移動限界電流は、合成式

`j_lim = 7.2 C^0.65 (1+0.010(T-35)) [1+0.80(1-exp(-u/0.12))] / sqrt(1+0.10g)`

（`j_lim` [A/dm²]、`C` [mol/L]、`T` [°C]、`u` [m/s]、`g` [cm]）である。`r=j/j_lim` に対して `tanh(r)/r` を輸送利用率とし、濃度・温度・撹拌の上昇で限界電流が上がり、輸送が滑らかに飽和する。

相対導電率は、濃度・温度・pHの合成閉包

`κ_rel = C^0.70 (1+0.018(T-25)) [1+0.50(2.2-pH)]`

とし、セル電圧を

`V_cell = 1.55 + 0.018j + 0.025 jg/κ_rel + 0.060(pH-2.2)^2 + 0.003(35-T) + 0.015(1-D)`

（`V_cell` [V]）で表す。これにより濃度・pH・温度・電極間隔・電流密度・duty が導電/セル電圧に作用する。

電流効率は、温度・濃度・pH・duty の基礎効率、輸送利用率、セル電圧による電気的補正、高電流燃焼ゲートの積である。粗さは電流密度、輸送混雑、燃焼ゲート、duty、pH、セル電圧の経験補正式である。Faraday 則、電荷と時間の関係、単位変換、限界電流という構造と、これらの係数・ロジスティック遷移・粗さ式は明確に分離している。後者は第一原理から直接導いたものではなく、未校正の合成経験補正である。

## データセット参照

参照元は `C:/Users/tsuts/Documents/Codex/20260623_ml_dataset_generation/src/dataset_generators.py` の `dataset_01()` と `dataset_12()` である。`dataset_01()` の電圧・抵抗・電流の電気関係を電気量の背景に、`dataset_12()` の差圧・管径・粘性・密度による流量構造を、濃度・撹拌・gap による物質移動の類推に利用した。データセットは実行時に依存せず、コピーもしない。

## テスト

`test_model.py` は、8入力のCSV契約と各4水準、`4^8=65,536` 全候補の有限性、制約の可行/不可行混在、Faraday 時間比例、duty比例傾向、濃度・撹拌・gap・pH・温度の物理傾向、broadcast、決定性、全入力の NaN/±Inf/範囲外拒否を確認する。

## 参考資料

* COMSOL Electrochemistry Module, Faraday mass flux and electrochemical deposition: https://doc.comsol.com/6.4/doc/com.comsol.help.fce/fce_ug_electrochem.07.120.html
* COMSOL Electrochemistry Module introduction: https://doc.comsol.com/6.4/doc/com.comsol.help.edecm/edecm_introduction.02.01.html
* IUPAC Gold Book, Faraday’s laws of electrolysis: https://goldbook.iupac.org/terms/view/09075
* IUPAC Gold Book, direct coulometry at controlled current: https://goldbook.iupac.org/terms/view/09160
* NIST CODATA, Faraday constant: https://physics.nist.gov/cuu/pdf/RevModPhys.93.025010.pdf

上記資料はFaraday則・電荷・析出質量/厚さの物理的骨格の参照である。輸送係数、電圧係数、効率低下、粗さ係数は本ベンチマーク固有であり、実機校正値ではない。
