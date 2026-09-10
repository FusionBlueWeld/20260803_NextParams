# プレス打抜き（press_forming）

## 目的

これはパラメータ最適化の検証用に作った、NumPyだけで動く決定論的な合成モデルです。実機の設計値や品質保証値を再現するものではなく、物理的な傾向を持つ滑らかな疑似実験の正解関数です。`evaluate_model()` はファイルを書き込まず、乱数・ノイズ・隠れた状態を使用しません。

## 固定条件と仮定

- 材料は代表的な軟鋼とし、板厚は **2.0 mm**、せん断強さは **320 MPa** に固定します。
- 切断対象のせん断周長は **20 mm**、パンチ・ダイは剛体で刃先は一定の鋭さとします。パンチ先端半径、潤滑、温度、材料ロットは入力にしません。
- 工具摩耗は中程度の固定値 **0.20（無次元の正規化摩耗率）** とし、最適化ノブにはしません。摩耗を変動させる場合は別モデルを作る必要があります。
- クリアランスは板厚に対する百分率です。入力範囲は3–15 %、ストローク速度は80–320 mm/s、ブランクホルダー力は20–100 kNです。
- 速度は荷重波形全体ではなく、ピーク荷重と品質指標へ影響する代表速度として扱います。出力はスカラー指標です。

## 入出力と最適化問題

`problem.csv` に定義した入力は次の3つです。

| 入力 | 単位 | 範囲 |
|---|---:|---:|
| `clearance_pct` | %（板厚比） | 3–15 |
| `stroke_speed_mm_s` | mm/s | 80–320 |
| `blank_holder_force_kn` | kN | 20–100 |

目的は `burr_height_mm` の **最小化**です。制約は `peak_force_kn <= 19.5 kN` と `flatness_error_mm <= 0.20 mm` です。出力はこの3つだけで、モニタ出力は追加していません。

## 式

まず、せん断荷重の基準値を次で置きます。

```text
F_shear [kN] = P [m] × t [m] × τ [Pa] / 1000
             = 0.020 × 0.002 × 320,000,000 / 1000
             = 12.8 kN
```

速度とホルダー力を無次元化した変数を

```text
r = (stroke_speed_mm_s - 80) / 240
h = (blank_holder_force_kn - 20) / 80
```

とします。最大荷重は、基準せん断荷重にクリアランス、速度、ホルダー摩擦の補正を掛け、ホルダー接触荷重を加えます。

```text
L_c = 1 - 0.005 (clearance_pct - 8)
        + 0.18 exp(-(clearance_pct - 3)/2.5)
L_r = 1 + 0.12 r + 0.03 r²
L_h = 1 + 0.04 h
F_peak = 12.8 L_c L_r L_h + 0.014 BHF + 0.10 h²  [kN]
```

`L_c`は、低クリアランスで拘束荷重が増え、クリアランス増加に伴って
ピーク荷重が緩やかに低下する実験傾向を表します。

平面度誤差は、破断不整合と速度による変形をホルダーの拘束で割り戻し、過大なホルダー力の二次的な接触変形を加えます。

```text
D_c = 0.020 + 0.085 d² + 0.012 max(-d, 0)       [mm]
D_r = 0.030 r^1.3                              [mm]
E_flat = 0.018 + (D_c + D_r)/(1 + 0.90 h) + 0.006 h² [mm]
```

バリ高さは、工具摩耗の固定オフセットと、4 %以上でクリアランスとともに
増加する実験傾向を中心に構成します。4 %未満にだけ小さな二次せん断・
破断不整合ペナルティを置き、材料を問わない8 %最適点は仮定しません。

```text
B = 0.018 + 0.010 × wear
    + 0.0040 (clearance_pct - 4)
    + 0.00065 (clearance_pct - 4)²
    + 0.0040 max(4 - clearance_pct, 0)²
    + 0.004 ((speed - 200)/120)²
    + 0.003 ((BHF - 60)/40)²                         [mm]
```

## 物理部分と合成経験部分

物理的に構造化した部分は、周長×板厚×せん断強さというせん断荷重の次元整合した基礎式、クリアランス増加に伴う荷重低下とバリ増加、速度が荷重を増やすこと、ホルダー拘束が平面度を改善することです。`L_c`、速度・摩擦係数、平面度の残差、バリ高さの係数は、滑らかな検証面を作るための経験的な合成補正です。特に絶対値は材料・刃先・潤滑・測定法に依存するため、実機への外挿はできません。

## データセットとの関係

参照したソースは `C:/Users/tsuts/Documents/Codex/20260623_ml_dataset_generation/src/dataset_generators.py` です。`press_wave(thickness, hardness, clearance, wear=0.0, n=220)` の荷重波形生成、`dataset_17()` のプレス荷重波形、`dataset_29()` の摩耗付き `burr_height_mm` 生成が関連します。これらを実行時に import したり、データをコピーしたりはしません。dataset_29 の摩耗は、本モデルでは固定条件に置き換えています。

クリアランス、破断面、バリの連成を説明する一次資料・技術解説の入口として、次を参照できます（本モデルの係数を同定したものではありません）。係数は検証用に合成したものです。

- MISUMI Tech Central, “Problems in Punching and Their Countermeasures (1): Punching Burrs”: https://www.misumi-techcentral.com/tt/en/press/2013/06/162-problems-in-punching-and-their-countermeasures-1-punching-burrs.html
- MISUMI, “Understanding Punch and Die Clearance”: https://us.misumi-ec.com/blog/understanding-punch-and-die-clearance/
- 実験研究（アルミ板のクリアランスとバリ・打抜き荷重）: https://doi.org/10.1016/j.matdes.2005.03.013

## テスト方針

`test_model.py` は、(1) 4 %以上でのバリ増加とクリアランス増加に伴う荷重低下、速度・ホルダー力の物理傾向、(2) NaN・±Inf・範囲外値の拒否、(3) スカラー／ベクトル／多次元ブロードキャスト、(4) 全1,521点グリッドの有限性・境界、実行可能・不可能領域、(5) 決定性を確認します。

リポジトリ直下で次を実行してください。

```powershell
C:/Users/tsuts/Desktop/PythonDev_std/2026031_ParamOptimizer2/.runtime/venv/Scripts/python.exe -m unittest validation.single.press_forming.test_model -v
```
