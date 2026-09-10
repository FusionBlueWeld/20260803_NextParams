# validation/内の物理シミュレータ：文献照合・精度評価レポート

調査日：2026-09-11  
対象：validation/ 配下の physics_model.py  
目的：公開文献・公開データ・標準的な工学式と照合して、各関数の「物理的な向き」「定量値の妥当性」「未検証部分」を切り分ける。初回調査はコードを変更せず実施し、2026-09-11に根拠が十分な2工程を修正した（16節）。

## 1. 結論

validation/ には、独立した6モデルと、coating → drying → curing の3段階モデルがある。single/oracle/physics_model.py はレーザー溶接モデルへの互換 import であり、独立した第10モデルではない。したがって、物理的な評価対象は合計9ステージである。

結論を先に述べると、9モデルのいずれも、現時点で「実機に対して絶対値まで検証済みの生産用シミュレータ」とは判定できない。各モデル自身のドキュメントにも、決定論的な reduced-order synthetic oracle、すなわち最適化・検証用の透明な合成真値関数であり、特定設備に校正した digital twin ではない、と明記されている。根拠は [validation/multistage/DESIGN.md](multistage/DESIGN.md) と各 physics_model.py のモジュール説明である。

一方、構造として妥当な部分は多い。特に、切削の MRR・動力の次元式、電気めっきの Faraday 則、レーザーのガウス強度、硬化の Arrhenius 型温度依存、塗工・乾燥における温度・速度・拡散・外部移動抵抗の向きは、文献と整合している。

### 1.1 最も重要な定量比較

| 対象 | 文献または標準式 | このコードの再計算値 | 差分の意味 |
|---|---|---:|---|
| レーザー溶接の最大溶込み | 6 kW、130 µm、0.6 m/min で Type 304 の約11 mm | 6,000 W、130 µm、10 mm/s で 11.0003 mm | 約 +0.0003 mm、+0.003%。ただし材質・係数が異なる一点一致で、精度の証明ではない |
| 電気めっきの電流効率 | 酸性銅浴の3 A/dm²、24 ℃、10 minで95%以上 | 92.8645% | 文献の下限に対し少なくとも −2.14ポイント |
| 電気めっきの膜厚 | 同条件の理論 Faraday 厚さ 6.6155 µm、効率95%なら6.2847 µm以上 | 6.1435 µm | 理論値に −7.13%、95%下限に −2.25% |
| 切削の理想粗さ | 標準的な幾何式は Ra = 1000 f²/(32 r)、Rt = 1000 f²/(8 r) | 修正後はRa式を使用 | 幾何項の定義不整合を解消。ploughing・dynamic項は未校正 |
| 塗工の wet thickness | 二層 slot-die の例：20 m/min、gap 127 µmで下限約90 µm | gap 120 µm、20 m/min、粘度1.6 Pa·sで91.1400 µm | +1.27%だが、別材料・別ダイ・別表面張力のため偶然の近さ |
| エポキシ硬化の活性化エネルギー | DGEBA/DDSで約67.2–69 kJ/mol、商用 epoxy-amine で53–61 kJ/mol | single thermal_curing 65、multistage curing 62 kJ/mol | オーダーは整合。ただし反応次数・転化率依存・ガラス化は未再現 |

### 1.2 判定の要約

ここでいう「高・中・低」は、実機精度の百分率ではなく、公開文献からどの程度まで支持できるかを示す証拠強度である。文献の材質、形状、装置、測定定義が一致しない場合、誤差率を作らずに「比較不能」とした。

| ステージ | 構造・傾向 | 絶対値 | 特に信頼できない出力 |
|---|---|---|---|
| laser_welding | 中 | 低〜中 | spatter_level_0_9、広い条件での溶込み |
| milling | MRRとRa幾何項は高、その他は中以下 | MRR高、総粗さ中、動力中 | roughnessの動的項 |
| press_forming | clearanceのバリ・荷重方向は中 | 荷重低〜中 | flatness_error_mm、速度効果 |
| thermal_curing | Arrhenius の向きは中 | 低 | degradation_fraction、bond_strength_mpa |
| convection_drying | 熱・物質移動の向きは中 | 低〜中 | defect_index、energy_kj_m2 |
| electroplating | Faraday厚さは高、輸送は中 | 膜厚中、効率低〜中 | roughness_ra_um、cell_voltage_v |
| multistage coating | coating window のU字傾向は中 | 低〜中 | coating_defect_index、絶対膜厚 |
| multistage drying | 状態受け渡しと乾燥方向は中 | 低〜中 | stress、skinning、energy |
| multistage curing | 硬化の方向は中 | 低 | bond strength、blister、最終欠陥 |

## 2. 評価方法

### 2.1 比較のランク

文献照合は次の4段階に分けた。

1. **A：直接定量比較** — 同じ入力またはほぼ同じ入力と、同じ出力定義がある。
2. **B：間接定量比較** — 同じ支配式・近い寸法・近い材料だが、装置や材料が異なる。
3. **C：挙動比較** — 文献が示す増減、遷移、失敗モードと、コードの偏微分・極値・遷移が一致するかを見る。
4. **D：比較対象なし** — 指標が合成スコア、文献で一般的な測定量でない、または条件が不足している。

差分を数値化できる場合は、相対差を

|計算値 − 文献値| / |文献値| × 100

で計算した。ただし「約11 mm」「95%以上」のような範囲・下限しかない場合は、差分を下限または近似値として記載した。

### 2.2 文献の選び方

一次論文、出版社・学協会の公開本文、大学リポジトリ、標準式を公開している工具メーカー資料、再利用可能なデータセットを優先した。各文献は本文末の番号付き参考文献から原典へ直接追跡できるようにした。

文献値をこのコードの合成出力へ無理に当てはめることは避けた。例えば、spatter_level_0_9、coating_defect_index、drying_defect_index、blister_index は文献で標準化された物理量ではないため、現在の公開情報だけでは「何％正確」とは言えない。

### 2.3 コード側の対象

| 区分 | 対象ファイル | 主な出力 |
|---|---|---|
| single | [laser_welding/physics_model.py](single/laser_welding/physics_model.py) | penetration depth、spatter class |
| single | [milling/physics_model.py](single/milling/physics_model.py) | MRR、roughness、spindle power |
| single | [press_forming/physics_model.py](single/press_forming/physics_model.py) | burr、peak force、flatness |
| single | [thermal_curing/physics_model.py](single/thermal_curing/physics_model.py) | cure、degradation、bond strength |
| single | [convection_drying/physics_model.py](single/convection_drying/physics_model.py) | residual moisture、energy、defect |
| single | [electroplating/physics_model.py](single/electroplating/physics_model.py) | thickness、efficiency、roughness、voltage |
| multistage | [functional_coating/coating/physics_model.py](multistage/functional_coating/coating/physics_model.py) | wet thickness、Ca、coating defect |
| multistage | [functional_coating/drying/physics_model.py](multistage/functional_coating/drying/physics_model.py) | dry thickness、residual solvent、stress |
| multistage | [functional_coating/curing/physics_model.py](multistage/functional_coating/curing/physics_model.py) | cure、strength、blister、final defect |

## 3. レーザー溶接：laser_welding

### 3.1 モデルの構造

コードは CW Gaussian beam を前提にし、spot_um を1/e²強度直径として

I_peak = 8P / (πd²)

を計算する。吸収率を conduction から keyhole へ固定点反復で遷移させ、溶込みは、低エネルギー側の伝導型項と、keyhole 側の P/(d v) 項を滑らかに接続している。spatter は、keyhole 遷移、低速での崩壊、recoil、低速・高速側の不安定化を足し合わせ、最終的に0〜9の整数へ丸める。

### 3.2 文献との整合

Kawahito、Mizutani、Katayama は、6 kW fiber laser による Type 304 bead-on-plate 溶接で、130、200、360、560 µmの4スポットを比較し、130 µm・0.6 m/minで最大約11 mmの溶込みを報告している。また、高速側では power density が溶込みに強く影響し、360/560 µmでは4.5〜10 m/minに欠陥の少ない広い条件窓が得られ、6 m/min・360 µmでは長い溶融池がスパッタを吸収して安定化したと報告している。[^2]

同じ入力をこのモデルへ入れると、P=6000 W、d=130 µm、v=10 mm/sで penetration_depth_mm=11.0003となる。文献の約11 mmとの差は約0.0003 mm、相対約0.003%である。しかし、コードは厚板 mild steel 固定、文献は Type 304、コード内の吸収率・drilling coefficient・表面張力等も文献から同定されていない。これは精度検証ではなく、合成係数がこの一点に近い値を生成することを示すだけである。

TWIの超高強度鋼の整理では、4 kW CWで0.8〜1.5 mm鋼板を完全溶込みさせる速度は、0.6 mmスポットで2.5〜6.5 m/min、0.2 mmスポットでは20 m/min超の例が示され、スポット径・power density・速度の結合が重要である。[^1] これはコードの「小スポット・高出力密度・低速ほど深くなりやすい」という方向を支持する。

Fabbroの keyhole/melt-pool 解析は、深溶込みの高速側で速度に対するおおむね1/v型のスケーリングを示す一方、低速では溶融池、プルーム、keyholeの不安定性によって単純な逆速度則から外れることを扱っている。[^3] コードの P/(d v) 項と低速 collapse 項はこの挙動を低次元で表そうとしているが、keyhole内の圧力、蒸発反力、液面形状を解いてはいない。

KimとKiの scaling-law 研究は、強度だけでなく beam diameter と interaction time を含めた無次元量、さらに多重反射の寄与を必要とし、「単一のしきい power density では整理できない」と示す。[^4] この点は重要である。現コードは P/(d v) と peak intensity を持つものの、beam quality、焦点位置、吸収の空間分布、シールドガス、板厚・熱伝導率の材質依存、多重反射を十分に持っていない。したがって、溶込みの絶対値を広い範囲で外挿する根拠は弱い。

10 kW fiber laserによる厚板鋼の研究では、0.3 m/minでガス噴流なし約18.2 mm、あり約24.5 mmの溶込みが報告されている。低速域で出力を増やしても溶込みが単純に比例しないことも示されている。[^8] これはコードの smooth minimum と深さ飽和の発想には整合するが、同一条件ではない。

spatter については、低速・高出力密度のkeyhole不安定、front/behind keyholeからの飛散、速度増加時の humping や spatter 発生が複数研究で観察されている。[^9][^10][^11] したがって、コードの spatter_propensity が単調な一方向ではなく、遷移・低速・高速の複数項を持つことは妥当である。ただし、文献の測定量は、spatter個数、サイズ、質量、画像分類、溶接欠陥などであり、コードの spatter_level_0_9 との変換規則は存在しない。

### 3.3 判定

* peak intensity、line energy、P/(d v) の向き：**C〜B相当**。
* 伝導型からkeyhole型への遷移を持つこと：**C相当**。ただし遷移点の数値は未校正。
* 11 mmの一点：**Aに近いが、材料不一致のため検証には使えない**。
* penetration_depth_mm の広域絶対精度：**低**。文献が必要とする相似則の変数を取り切れていない。
* spatter_level_0_9：**低／未定義**。物理量ではなく合成 ordinal label。

なお、公開データセットには1〜16 kW、4〜60 m/min、100/200/600 µmスポット、溶込み0.1〜6 mmの測定領域がある。[^12] 将来このモデルを校正するなら、まずこのような同一材質・同一板厚・同一光学条件の bead-on-plate データへ合わせる必要がある。

## 4. 切削：milling

### 4.1 モデルの構造

MRRは

Q = ae ap fz n z / 1000

で cm³/minを出し、切削速度は π D n / 1000、動力は Q kc / 60000 である。kc は代表値1850 N/mm²へ、速度、送り、切込み比、摩耗の合成補正を掛ける。

粗さは、feed-mark geometry、ploughing、160/300/440 Hzの3モード強制応答を足している。ファイル自身にも、後者は regenerative chatter の stability-lobe 計算ではなく、forced-response surrogate だと明記されている。

### 4.2 MRR・動力の定量評価

Sandvik Coromantの公式式は、vf=fz n zc、Q=ap ae vf/1000、Pc=ae ap vf kc/(60×10⁶)である。[^14] コードは MRRについてこの式をそのまま実装し、cm³/minから動力式へ単位変換しているため、**次元式としては高い整合**がある。

代表的な再計算は次の通りである。

| n | fz | ap | ae | z | コード MRR |
|---:|---:|---:|---:|---:|---:|
| 4000 rpm | 0.08 mm | 2 mm | 10 mm | 4 | 25.6 cm³/min |
| 4000 rpm | 0.12 mm | 2 mm | 10 mm | 4 | 38.4 cm³/min |
| 4000 rpm | 0.16 mm | 2 mm | 10 mm | 4 | 51.2 cm³/min |

この部分には「モデル予測誤差」はなく、入力の単位・有効切削幅・切込みが実際の加工と一致する場合に限り、幾何学的な除去体積を表す。ただし、実際の有効歯数、切削弧、入口・出口、工具の振れ、切込み形状、chip thinning は別途必要である。

### 4.3 粗さの幾何項（調査後に修正済み）

文献で一般的な理想幾何式は、送り f とノーズ半径 re に対して、Ra=1000 f²/(32re)、Rt=1000 f²/(8re)である。[^17] 初回調査時のコードは後者を `geometric_ra` としていたが、修正後は前者のRa式を使用する。

| fz | nose radius | 修正後 geometric_ra | 理想 Ra | 理想 Rt | 修正後 roughness_ra |
|---:|---:|---:|---:|---:|---:|
| 0.08 mm | 0.8 mm | 0.250 µm | 0.250 µm | 1.000 µm | 0.7015 µm |
| 0.12 mm | 1.6 mm | 0.281 µm | 0.281 µm | 1.125 µm | 0.7817 µm |
| 0.16 mm | 0.4 mm | 2.000 µm | 2.000 µm | 8.000 µm | 2.5416 µm |

これにより幾何項の名称と標準定義は一致した。ただし、最終出力にはploughingとdynamic項が追加され、これらは実測同定されていないため、最終Ra全体が校正済みになったわけではない。定義変更後の全候補レンジは0.173–4.192 µmで、旧7.50 µm制約では全点通過したため、検証問題の閾値を2.50 µmへ更新した。

Munoz EscalonaとMaropoulosの face milling 研究は、ノーズ半径、工具径、歯数、送り、振れを含む幾何モデルをAl7075で36条件検証し、実験Raと予測Raの平均相対誤差2.4%を報告している。[^15] これは、**新しい工具・特定材料・特定工具形状の幾何成分**に対する良いベンチマークである。一方、現コードは代表鋼のkc、摩耗、ploughing、動的項を追加しているため、この2.4%を現コード全体へ移植することはできない。

切削速度、送り、chip cross-sectionが切削力と粗さへ与える影響は、AISI 1020/1040の実験でも単純な一方向関係ではなく、速度域・built-up edge・工具条件に依存する。[^16] 現コードの速度依存kcは方向を持つが、材料の構成式やBUEの領域を含まない。

### 4.4 動力学・摩耗

文献の regenerative chatter 解析は、切削係数だけでなく、機械・工具・工作物系の周波数応答、剛性、減衰、切削深さを使って stability lobe を予測する。安定／不安定境界は spindle speed と axial depth の図として現れる。[^18] 現コードの160/300/440 Hzは代表的な非単調応答を作るための仮定であり、実測FRFや工具突出し長さを持たない。したがって、共振ピークの位置、チャタリング限界、振幅は未検証である。

公開データセットには、切削力、振動、音、電流、摩耗、工具寿命を含むものがあり、CK45・Al7030の115サンプル、968サイクルの工具摩耗データ、さらに2026年のVP100実験データなどが利用できる。[^19][^20][^21] これらは、摩耗を単一の線形補正ではなく、力・振動・粗さ・工具寿命の連成として検証するための候補である。

### 4.5 判定

* MRRの幾何式：**高**。
* spindle power：**中**。式の骨格は正しいが、kc=1850と補正係数は未校正。
* geometric roughness：**高**。標準Ra式へ修正済み。
* 最終roughness：**中**。ploughing・摩耗・dynamic項の数値根拠はまだ不足。
* chatterの非単調性：**方向として中**、絶対的な安定限界は**低**。

## 5. プレス成形／ブランキング：press_forming

### 5.1 モデルの構造

実装は一般的な深絞りモデルではなく、固定周長、2 mm板厚、せん断強さ320 MPaの blanking / shearing oracle である。基本荷重は

F₀ = perimeter × thickness × shear strength

で、コードの固定値では12.8 kNとなる。修正後はclearance増加に伴うバリ増加と荷重低下、低clearanceの追加ペナルティ、stroke speed と blank-holder force の補正をburr・force・flatnessの合成式へ入れている。

### 5.2 バリとクリアランス

AA5754を対象としたÇavuşoğluとGürünの研究は、板厚1/2 mm、clearance 8〜18%で、バリ高さはclearanceの影響が大きく、ANOVAの寄与率はclearance 76.15%、板厚23.24%だったと報告している。一方、blanking forceは板厚の寄与99.72%、clearanceは0.24%で、バリと荷重を同じ単純な関数で扱えないことが示される。[^22]

KüçüktürkのAA5754実験は、20 mmパンチ、1/1.5/2 mm板、clearance 0.07〜0.18 mmで105回の切断を行い、低clearanceで荷重が増え、clearanceを大きくするとバリが増える傾向を示している。clearance変化による荷重差はおおむね5〜10%、バリの変化は60〜80%に達する。[^23]

コードを速度160 mm/s、holder 60 kNで再計算すると次のようになる。

| clearance | burr_height | peak_force | flatness_error |
|---:|---:|---:|---:|
| 3% | 0.02109 mm | 17.279 kN | 0.10515 mm |
| 5% | 0.02509 mm | 15.793 kN | 0.06432 mm |
| 8% | 0.04684 mm | 14.819 kN | 0.03825 mm |
| 10% | 0.06784 mm | 14.500 kN | 0.04763 mm |
| 12% | 0.09404 mm | 14.281 kN | 0.07577 mm |
| 15% | 0.14309 mm | 14.030 kN | 0.15315 mm |

修正後は、実験範囲と重なる4〜15%でバリが単調増加し、4%未満だけ二次せん断・破断不整合を表す小さなペナルティを置く。材料に依存しない8%最適という旧仮定は除去した。

荷重もclearance増加に伴い単調低下する式へ修正した。8〜15%の低下幅は約5.3%で、文献の5〜10%程度という範囲に入る。低clearanceでは指数的な拘束ペナルティを残す。高速度の影響は工具形状・材料・速度域に依存するため、速度項は依然として合成仮定である。厚板鋼の研究では工具摩耗がバリを増やし、特殊パンチ形状では高速度がバリを低減する例もある。[^24]

### 5.3 絶対荷重の比較不能性

コードの周長は20 mm、板厚2 mm、せん断強さ320 MPaである。一方、20 mm径パンチの円周は約62.8 mmであり、文献側も板厚・材質・工具クリアランス・パンチ形状が異なる。したがって、コードの12.8 kNまたは14〜19 kNを文献の数kNと直接比べることは不適切である。基本式は次元的には正しいが、せん断面積、実効せん断強さ、摩擦、fracture propagation、holderの拘束を同定していない。

flatness_error_mm は、引用したブランキング文献の主要出力であるバリ、荷重、smooth-sheared ratioとは別指標であり、現在は直接比較できない。Fraunhoferの数値研究のように、force-displacement曲線、破断形状、clearance、摩耗、holder条件を同時に測定するデータが必要である。[^27]

### 5.4 判定

* 基本せん断荷重：**式の骨格は中**、絶対値は**低**。
* clearance増加で高側バリが増える：**中**。
* 4〜15%でのバリ増加と荷重低下：**中**。文献方向へ修正済み。
* speed / holder の効果：**低〜中**、工具・材料依存。
* flatness_error_mm：**低／比較対象なし**。

## 6. 熱硬化：thermal_curing

### 6.1 モデルの構造

温度は、周囲温度25 ℃から oven温度へ向かう一階遅れ

T(t)=Tamb+(Toven−Tamb)(1−exp(−t/τ))

で与えられる。τは厚さの二乗に比例する形で、硬化と劣化をそれぞれ一段のArrhenius速度で積分し、転化率を1−exp(−exposure)とする。singleモデルの主な係数は、硬化Ea=65 kJ/mol、120 ℃での基準速度0.018 /min、劣化Ea=80 kJ/mol、170 ℃での基準速度0.003 /minである。

### 6.2 硬化活性化エネルギー

公開研究の値は樹脂・硬化剤・触媒・転化率によって広い。

| 文献・系 | 報告された代表値 | singleコード65 kJ/molとの差 |
|---|---:|---:|
| DGEBA/DDSの熱硬化 | 約69 kJ/mol | −5.8% |
| DGEBA/Jeffamine D230の主反応 | 67.2 kJ/mol | −3.3% |
| Kaimon・Saitoの各種polyamine | 約40.2〜56.1 kJ/mol | コードが約16〜62%高い |
| 商用高固形分 epoxy-amine | apparent Ea 53〜61 kJ/mol | コードが約7〜23%高い |

HillらはDGEBA/DDSについて約69 kJ/molを報告しており、コードの65 kJ/molは同じオーダーである。[^34] MacanらのDGEBA/Jeffamine D230系は、Kamal型の主反応にEa=67.2 kJ/molを用いている。[^33] KamonとSaitoは硬化剤によっておおむね40〜56 kJ/molの幅を示している。[^32] Bashirの商用epoxy-amine研究では、三つの高固形分製品について conversion-dependent なapparent Eaが53〜61 kJ/molであり、湿度による触媒効果のため非等温DSCから予測した等温転化率が実測値を下回ることも示されている。[^36]

したがって、コードのEa=65は「代表的な一例」としては妥当な範囲にあるが、モデルの絶対精度を保証しない。

### 6.3 一段一次反応の限界

エポキシ硬化では、Kamal型

dα/dt = (k′ + k αᵐ)(1−α)ⁿ

のようなautocatalytic項が広く使われる。Kamal式は個々の等温曲線に合うことがある一方、mとnの一意性や温度をまたぐ予測には限界がある。[^31] さらに、isoconversional研究はapparent Eaが転化率によって変化することを示し、後半はgelation、vitrification、拡散制限を受ける。[^29][^30]

現コードは、化学反応速度を最後まで温度だけの一次速度として扱う。そのため、初期のautocatalysisを過小評価したり、後半のvitrificationによる速度低下を過大評価したりし得る。文献で「一つの平均Ea」を得ても、全転化率域へ同じ値を適用できるとは限らない。

コードの代表再計算は次の通りである。

| oven | hold | thickness | cure_fraction | degradation_fraction | bond_strength |
|---:|---:|---:|---:|---:|---:|
| 120 ℃ | 60 min | 0.24 mm | 0.6230 | 0.01013 | 46.780 MPa |
| 140 ℃ | 60 min | 0.24 mm | 0.9214 | 0.03258 | 57.006 MPa |
| 170 ℃ | 60 min | 0.24 mm | 0.9999 | 0.14745 | 43.833 MPa |
| 180 ℃ | 60 min | 0.24 mm | 1.0000 | 0.22686 | 35.392 MPa |

この非単調な強度は、硬化による上昇と劣化による低下を表す合成形状としては分かりやすい。しかし、特定の樹脂・硬化剤・接着試験に基づくMPa曲線ではない。

### 6.4 劣化と接着強度

Andersonの高温エポキシ接着剤研究では、TGA劣化と接着強度劣化を別々に扱い、第一の系で劣化および接着強度低下の活性化エネルギーはおよそ142〜144 kJ/molであった。劣化速度は単純な一次反応ではなく、autocatalyticな速度式との比較が必要であり、TGA重量減少と接着強度低下も同一の状態量ではない。[^28]

したがって、コードの劣化Ea=80 kJ/molはこの高温劣化研究の値より約44%低い。ただし、同じ意味のEaではない。Andersonの対象は完全硬化接着剤の長時間高温劣化、コードは加熱中の未硬化／硬化／劣化の合成であるため、これを直接の予測誤差とはしない。正しくは、「現コードの劣化項には、少なくとも材料・劣化モード・測定量の同定がない」と判定する。

bond_strength_mpaについても、文献は転化率だけでなく、ネットワーク密度、化学量論、硬化剤、Tg、相分離、残留応力、被着体、試験速度に依存する。接着剤の凝集機械特性を硬化レベルだけから普遍的に予測できないことは、専用研究でも強調されている。[^37]

### 6.5 判定

* 温度上昇で硬化が進み、高温・長時間で劣化が増える：**中**。
* Ea=65 kJ/molのオーダー：**中**。
* cure_fractionの絶対値：**低**。autocatalysis、vitrification、湿度、材料組成がない。
* degradation_fraction：**低**。文献の高温劣化Eaと大きく異なり、状態量も別。
* bond_strength_mpa：**低／合成出力**。

## 7. 単独の対流乾燥：convection_drying

### 7.1 モデルの構造

固定膜厚0.35 mm、水系膜、乾燥固形分0.05 kg/m²、初期水分wet basis 28%を前提に、熱容量によるheat-up、Antoine式の飽和蒸気圧、Reynolds・Schmidt・Sherwoodを用いた外部移動、内部拡散、直列抵抗を組み合わせている。水分の残量は全体の一次指数減衰で計算し、defect_indexは表面とbulkの乾燥差から作る。

### 7.2 文献が支持する部分

Naseriらは、polymer strip filmの強制対流乾燥について、熱伝達、moisture diffusion、moving boundary、free-volume依存拡散を含む詳細モデルを構築し、複数温度・速度・膜厚で検証した。水分プロファイルについてR²>0.95を報告している。乾燥は外部移動が支配するconstant-rate期から、内部拡散が支配するfalling-rate期へ移る。[^38]

同研究の乾燥膜厚は、wet 262 µmに対し実測50〜55 µm、モデル63 µm（40 ℃、1 m/s）、実測値に対して約+14〜+26%。70 ℃、1 m/sではモデル58 µmで、実測50〜55 µmに対して約+5〜+16%である。これは現コードの誤差ではないが、詳細な校正モデルでも材料収縮、free-volume、境界条件によりこの程度の膜厚差が残ることを示す。

1987年の coating drying model は、表面蒸発と膜内拡散を結合し、拡散係数を溶媒濃度依存、温度依存にしている。[^39] WaggonerとBlumも、溶媒揮発性と膜内拡散を分けたモデルで、volatility-controlled と diffusion-controlled の二つの領域を実験曲線と比較している。[^40]

これに対して現コードは、内部拡散係数を温度だけの指数関数とし、全体の残水分を単一のexp(−kt)で処理する。したがって、以下を表せない。

* 乾燥初期のconstant-rate periodと臨界含水率。
* 表面にできるpolymer-rich skinによる後半の拡散障壁。
* 溶媒濃度によるDの低下。
* 複数溶媒、水分活量、湿度、膜の移動境界。

一方、コードの低温・短時間で残水分が大きく、高温・高速・長時間で低くなる向きは、基本的な熱・物質移動と整合する。Powersの実験でも、高温ガスで表面skinが形成されると溶媒除去速度が低下する可能性が報告されている。[^43]

### 7.3 コードの代表値

| air temperature | air speed | residence | residual moisture | energy | defect |
|---:|---:|---:|---:|---:|---:|
| 45 ℃ | 1 m/s | 1 min | 27.622% | 4.177 kJ/m² | 0.0266 |
| 70 ℃ | 1 m/s | 6 min | 21.868% | 23.031 kJ/m² | 0.2948 |
| 80 ℃ | 2 m/s | 6 min | 19.032% | 31.383 kJ/m² | 0.5021 |
| 105 ℃ | 6 m/s | 12 min | 2.111% | 79.597 kJ/m² | 0.1509 |

energy_kj_m2は sensible、latent、fan work を足した合成エネルギーである。外気温、熱回収、排気湿度、実測ファン効率、基材の熱容量が固定されているため、文献との絶対比較はできない。defect_indexも、実際のひび、pin-hole、orange peel、skin、adhesion failureの確率や面積率ではない。

### 7.4 判定

* heat-up、蒸気圧、外部／内部移動抵抗の向き：**中**。
* residual_moisture_pct：**低〜中**。構造は妥当だが、単一指数で乾燥段階を潰している。
* energy_kj_m2：**低**。熱収支と設備条件が不足。
* defect_index：**低／合成出力**。

## 8. 電気めっき：electroplating

### 8.1 モデルの構造

コードはlimiting currentを銅濃度、温度、攪拌、電極間gapから作り、current / limiting_currentに対して tanh(ratio)/ratio の輸送利用率を掛ける。cell voltageは導電率の相対閉包から計算し、電流効率を輸送、電気、burning gateで補正する。膜厚はFaraday則

h = η j t M / (z F ρ)

を単位変換している。

### 8.2 Faraday膜厚の直接比較

Rotating disk electrodeの銅析出研究では、limiting currentは銅濃度にほぼ比例し、回転数の平方根にも依存する。銅イオンの拡散係数を5.3×10⁻⁶ cm²/sとして、Levich型の輸送限界を確認している。[^54] HsuehとNewmanは、銅析出の limiting current と、濃度依存物性・ohmic dropを含む計算を比較している。[^55]

コードの j=3 A/dm²、t=10 minで、η=1の理論厚さは6.6155 µmである。公開されている酸性銅浴の例では、240 g/L CuSO₄、硫酸、3 A/dm²、24±1 ℃、10 minでcurrent efficiencyが95%以上と報告されている。[^58]

同条件に近いコード計算は次の通りである。

| 量 | 文献・理論 | コード |
|---|---:|---:|
| current efficiency | 95%以上 | 92.8645% |
| ideal Faraday thickness | 6.6155 µm | — |
| 95% efficiency thickness | 6.2847 µm以上 | — |
| deposit thickness | — | 6.1435 µm |
| cell voltage | — | 1.7134 V |
| roughness | — | 0.2273 µm |

したがって、コード効率は文献の95%下限より少なくとも2.14ポイント低く、膜厚は理論値より7.13%、95%下限より2.25%低い。ただし、文献の浴組成、電極配置、攪拌、基板、表面前処理が一致していないため、これは「同じ浴での誤差」ではなく、近い運転点における要注意差である。

### 8.3 電流効率と粗さ

Palliのcitrate銅浴のmodified Hull cell研究は、2、4.1、8.2 A/dm²に相当する条件で、current efficiencyをそれぞれ188.7%、約105%、84.5%と報告している。100%超は有機物の取り込み、低い効率は高電流密度での水素発生と関係する。[^57] 現コードはefficiencyを0.05〜0.995にclipするため、100%超という実測質量基準や異常な共析を再現できない。

銅膜の接着評価研究では、6 A/dm²、22 ℃、10 µm付近の膜について、添加剤なし・単一添加剤でRa約0.153〜0.194 µm、複数添加剤で約0.046 µmという差が示されている。[^59] コードの6 A/dm²、35 ℃条件ではroughness_ra_um=0.2778 µmで、添加剤なしの例より約43〜81%高く、複数添加剤条件の約6倍である。ただし、浴組成と粗さ定義が同じではないため、ここでも直接誤差ではなく、添加剤・表面化学を欠くモデルの限界を示す。

2024年の公開データ・解析は、銅めっき表面粗さが電流だけでなく温度、電極間距離、ガス／流動条件の影響を強く受けることを示している。[^60] 現コードは温度、gap、攪拌、pH、濃度を入力に持つ点では方向がよいが、brightener、chloride、PEG、MPSA、基材、電極の局所電流分布を持っていない。

### 8.4 輸送・電圧

コードのlimiting currentは、濃度・攪拌・温度が増えると増え、gapが増えると下がるため、Levich的な向きはある。しかし、tanh型の利用率、gapの平方根補正、relative conductivity、burning gateの係数は合成閉包である。cuprous cyanide浴の数値研究のように、拡散・移流・移動（migration）・電流分布・ohmic dropを同時に扱うモデルとの比較では、現コードは大幅に低次元である。[^56]

cell_voltage_vは導電率の絶対単位を用いず、濃度・pH・gapから1.55 V近辺を作る。よって、文献のセル電圧や電流効率の校正なしには絶対値の主張はできない。

### 8.5 判定

* Faraday厚さの式・単位変換：**高**。
* limiting-currentの向き：**中**。
* current_efficiency：**低〜中**。実測は84.5〜100%超まで幅があり、コードのclipでは異常析出を表せない。
* roughness_ra_um：**低**。添加剤・表面化学がない。
* cell_voltage_v：**低**。相対導電率の合成値。

## 9. 多段階塗工：functional_coating / coating

### 9.1 モデルの構造

capillary number

Ca = μU/γ

を使い、surface tensionは0.034 N/m固定である。wet thicknessはcoating gapにtransfer ratioを掛け、transfer ratioをCa・粘度・web tensionで補正する。low Ca ribbing/dewetting、高Ca air entrainment、tension deviation、incoming bubblesから、thickness CVとcoating defect indexを作る。

モデルの入力範囲から、Caはおよそ1.9608〜44.1176である。内部のsigmoidが50%になる目安は、low instabilityがCa≈4.61、high instabilityがCa≈30.8である。この「中間が安定で両端が悪い」という構造は、slot-die coatingの文献と整合する。

### 9.2 文献との比較

slot coatingの最小wet thickness研究では、低Ca領域で最小膜厚が速度・粘度・Caに依存し、一定の領域ではgapに支配され、coating windowが形成される。[^45][^46] die geometry、flow rate、surface tension、contact angle、web tensionで窓が変わる。[^47][^49]

高粘度液のair entrainment実験では、slot gapやcoating gap、粘度、動的接触角が気泡発生に影響し、高Ca側で気泡／air entrainmentが重要になる。[^48] web tensionについても、tensioned-web-over-slot-dieの圧力・meniscus状態が膜厚限界へ影響する。[^52]

Diehmらの二層slot-die coating研究では、gap 127 µmに対し、0.5 m/minのair-entrainment-free最小wet thicknessが87 µm、20 m/minで約90 µmだった。上限側は0.5 m/minの147 µmから20 m/minの133 µmへ低下した。[^50] これと近い数値を比較するため、コードのgap120 µm、speed20 m/min、viscosity1.6 Pa·s、solids0.5、bubble0ではwet thickness=91.1400 µmとなる。90 µmに対して+1.27%だが、文献は二層の水系電池スラリー、コードはpolymer solutionの合成モデルで、surface tension、rheology、die geometry、flow rateが異なる。この差は校正精度ではなく、同じスケールに落ちた偶然の一致として扱う。

high-speed primer coatingの研究は、最大550 m/minまでの欠陥-free process limitを調べ、vacuum boxによりminimum wet thicknessが半分にできることを示している。[^51] これは、web speedだけでなく圧力境界、vacuum、die geometryが膜厚窓を左右することを示す。現コードにはpump flow rate、die lip geometry、vacuum、contact angle、rheology curveがないため、wet thicknessをgapの固定比率で決める以上の定量性は持たない。

### 9.3 コードの代表値

| gap | speed | viscosity | Ca | wet thickness | coating defect |
|---:|---:|---:|---:|---:|---:|
| 120 µm | 5 m/min | 0.8 Pa·s | 1.9608 | 76.7565 µm | 0.3938 |
| 120 µm | 20 m/min | 1.6 Pa·s | 15.6863 | 91.1400 µm | 0.0906 |
| 300 µm | 30 m/min | 3.0 Pa·s | 44.1176 | 241.0174 µm | 0.2751 |

低Ca側と高Ca側の欠陥増加は、低Caのribbing／dewetting、高Caのair entrainmentという文献の失敗モードと整合する。ただし、文献の境界は装置ごとに異なるため、コード内のCa≈4.6/30.8を普遍的な物理しきい値とは解釈できない。さらに、現モデルの下限Caは約1.96であり、低Caの遷移全体を覆っていない。

### 9.4 判定

* Caを使った低／高側のcoating window：**中**。
* gap・速度・粘度の方向：**中**。
* wet thicknessの絶対値：**低〜中**。二層文献との数値は近いが、条件不一致。
* tension／bubble係数：**低**。
* coating_defect_index：**低／合成出力**。

## 10. 多段階乾燥：functional_coating / drying

### 10.1 モデルの構造

coatingから受け取ったwet thickness、solids、thickness CV、defectを用い、lumped heat-up、温度依存内部拡散、速度依存外部移動を直列抵抗として計算する。残留溶媒は一次指数、dry thicknessは固形分＋残留溶媒の体積和、skinningはsurface removedとbulk removedの差から作り、stressとdefectへ渡す。

この状態接続は、単独の乾燥モデルより一歩進んでいる。実際の連続塗工でも、wet thickness、固形分、溶媒組成、膜内濃度、表面skinが後段の収縮・残留溶媒・機械特性へ連鎖する。

### 10.2 文献との定量・構造比較

FegerとCorsoの連続epoxy coating乾燥研究は、10 m/minの布速度、30 mのhot-air ovenを対象に、free-volume理論から得た拡散係数を使った一様濃度モデルで、実際の残留溶媒濃度を2%以内で記述したと報告している。[^42] 30 mを10 m/minで通過する滞留時間は約3 minであり、現モデルの1〜15 minの範囲に入る。ただし、対象はepoxy/MEK varnishとglass clothであり、現モデルのsolids・density・solventとは異なる。

1987年のcoating drying model、WaggonerとBlumのモデル、Soft Matterのpolymer coating乾燥モデルはいずれも、表面蒸発、膜内拡散、濃度依存拡散、skin形成、stressの相互作用を重視している。[^39][^40][^41] 現モデルのskinning_indexは、この現象の向きを表す surrogate としては妥当だが、skinの透過係数、Tg、粘弾性、表面張力勾配を持たないため、stressの絶対値や欠陥率を予測できる段階ではない。

コードの代表値は次の通りである。

| air temperature | air speed | residence | wet thickness | dry thickness | residual solvent | stress | defect |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 ℃ | 1 m/s | 6 min | 100 µm | 48.1235 µm | 13.298% | 0.5875 MPa | 0.0992 |
| 90 ℃ | 2 m/s | 6 min | 100 µm | 39.7448 µm | 1.636% | 0.5340 MPa | 0.0937 |
| 120 ℃ | 6 m/s | 15 min | 200 µm | 77.4512 µm | 0.000159% | 0.8443 MPa | 0.1451 |

温度・速度・時間で残留溶媒が下がる方向は整合するが、実際には高温・高速度が常に乾燥を改善するとは限らず、surface skin、溶媒の種類、粘度増加、湿度、膜厚、乾燥ゾーン配置によって後半乾燥が遅くなり得る。多段階モデルのresidual solventは、文献のような濃度プロファイルを持たないため、絶対値は低信頼とした。

### 10.3 判定

* coating→dryingの状態接続：**中**。
* heat/mass-transferの向き：**中**。
* residual_solvent_pct：**低〜中**。単一指数のため、最終域の予測は未校正。
* dry thickness：**低〜中**。体積和は透明だが、収縮・移動境界が不足。
* stress、skinning、drying_defect：**低／合成出力**。

## 11. 多段階硬化：functional_coating / curing

### 11.1 モデルの構造

dryingからdry thickness、residual solvent、internal stress、drying defect、thickness CVを受け取る。温度は厚さ依存の一階遅れで立ち上げ、硬化Ea=62 kJ/mol、劣化Ea=82 kJ/mol、残留溶媒escape Ea=36 kJ/molを用いて96区間のmidpoint積分を行う。

gel_before_escape、residual solvent、温度、厚さからblister_indexを作り、圧力・収縮・stress・defectを介してbond_strength_mpaを計算する。

### 11.2 硬化速度の妥当性

multistage curingのEa=62 kJ/molは、商用epoxy-amineの53〜61 kJ/mol、DGEBA/DDSの約69 kJ/mol、DGEBA/Jeffamineの67.2 kJ/molと同じオーダーである。[^33][^34][^36] これは初期の化学反応を代表する値としては支持される。

しかし、前節と同様に、epoxy-amine cureではautocatalysis、gelation、vitrification、湿度、硬化剤比、添加剤が重要である。文献のisoconversional解析ではEaが転化率依存となり、窒素下の非等温DSCから湿度下の等温硬化をそのまま予測できない。[^29][^30][^36] 現モデルはこのため、特に高転化率域で硬化を過大評価し得る。

### 11.3 残留溶媒・blister・強度

コードの代表値は以下である。

| oven | hold | dry thickness | incoming solvent | cure | degradation | final solvent | bond strength | blister | final defect |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 ℃ | 30 min | 70 µm | 2% | 0.37445 | 0.00293 | 0.0985% | 28.692 MPa | 0.0073 | 0.0648 |
| 150 ℃ | 30 min | 70 µm | 2% | 0.82959 | 0.01695 | 0.0031% | 42.428 MPa | 0.0203 | 0.0814 |
| 180 ℃ | 60 min | 100 µm | 10% | 0.99999 | 0.15283 | 0.0000019% | 22.462 MPa | 0.2987 | 0.3525 |

高温で残留溶媒がほぼゼロになる一方、blisterが増えるという形は、「早いgel化が内部揮発分を閉じ込める」という仮説を表す合成形状としては理解できる。しかし、検索した文献群から、このblister_indexの係数、またはresidual solvent・gel conversion・nip pressureからblister率へ変換する普遍式は確認できなかった。したがって、blister_indexは**校正前のリスク指標**であって、欠陥面積率や発生確率ではない。

同様に、bond_strength_mpaは5+60×各種penaltyの合成式で、接着剤の種類、被着体、試験モード、硬化後Tg、凝集破壊／界面破壊を指定していない。文献の接着力は、例えば二層電極でbinder gradientにより23 N/mから44 N/mへ増えるなど、組成・界面・乾燥履歴に大きく依存する。[^50] この出力をMPaの文献値と直接比較するのは不適切である。

### 11.4 判定

* cure_fractionの温度・時間方向：**中**。
* cure Ea=62 kJ/mol：**中**。
* residual solvent escape：**低〜中**。単一Arrhenius＋指数減衰で、乾燥段階とskinを省略。
* degradation：**低**。
* bond_strength、blister、final_defect、stress：**低／合成出力**。

## 12. 多段階モデル全体の整合性

### 12.1 良い点

多段階モデルは、単純に3個の独立モデルを並べたものではなく、次の状態を引き継ぐ。

coating → wet thickness、solids、thickness CV、coating defect  
drying → dry thickness、residual solvent、internal stress、drying defect  
curing → cure、degradation、final thickness、blister、bond strength、final defect

この因果方向は、文献で観察される「塗工のwet filmとrheologyが乾燥後膜へ影響し、乾燥速度・残留溶媒・binder migrationが硬化後の接着・応力・欠陥へ影響する」という構造と整合する。二層slot-dieの研究でも、乾燥速度が接着力、層間状態、coating windowと結び付いている。[^50]

### 12.2 限界

ただし、各状態量の誤差が後段へ渡る。coatingのwet thicknessが gap の合成比でずれると、dryingの拡散時間は厚さの二乗でずれ、curingでは温度遅れ・溶媒escape・強度式へ連続的に影響する。上流のdefectやCVも下流で係数倍される。

このため、個々のstageに中程度の傾向整合があっても、end-to-endのabsolute accuracyは低くなりやすい。現時点では、最終bond_strengthやfinal_defectを文献から検証済みとする根拠はない。

## 13. 今回の調査で確認できたこと／できないこと

### 確認できたこと

* 9ステージのモデル構成と入出力を特定できた。
* 切削のMRR・動力、電気めっきのFaraday膜厚など、次元式が正しい領域を特定できた。
* レーザーのpower density、interaction time、keyhole／spatter遷移の向きは文献と整合した。
* 硬化のEa=62〜65 kJ/molは、複数のepoxy-amine系の文献値と同じオーダーにある。
* 塗工の低Ca／高Caの欠陥窓、乾燥の外部移動→内部拡散の構造は文献と整合した。
* レーザー11 mm、電気めっき3 A/dm²の効率・膜厚、塗工91 µmという、限定付きの比較値を再計算できた。

### まだ確認できないこと

* 材料・設備・工具・電極配置を揃えたglobal MAPE/RMSE。
* spatter_level、defect_index、blister_indexの実測確率／質量／面積率との対応。
* bond_strength_mpaのMPa絶対値。
* press_formingのflatness_errorと、実際のforce-displacement・破断進展。
* millingの実FRF、stability lobe、摩耗寿命、Ra定義の統一。
* drying／curingの膜内濃度分布、skin透過性、Tg、湿度、収縮・応力の実測。
* electroplatingの添加剤、局所電流分布、migration、conductivityを含む電圧・粗さ。

## 14. 変更を行わない段階での推奨検証計画

これはコード修正の提案ではなく、将来の実測校正に必要な最低限のデータ設計である。

| ステージ | 最低限そろえる実測 | まず同定する量 |
|---|---|---|
| laser_welding | P・spot・speed・板厚・材質・shieldingを固定したbead-on-plate、断面溶込み、幅、spatter質量／個数、X線／高速画像 | 吸収率遷移、P/(dv)係数、低速不安定、spatterの物理定義 |
| milling | dynamometer、加速度計、実FRF、Ra/Rt/Rz、工具摩耗、切削条件 | kc、Ra/Rtの幾何項、modal frequency/damping、wear |
| press_forming | 同じパンチ・ダイ・板厚・材質でのforce-displacement、burr、smooth-sheared ratio、flatness | clearanceと速度の符号、破断進展、摩耗 |
| thermal_curing | DSC/DEA、等温・非等温、転化率、Tg、TGA、硬化後接着強度 | autocatalytic反応、Ea(α)、vitrification、劣化モード |
| convection_drying | mass-vs-time、膜厚-vs-time、表面／bulk温度、湿度、残水分、欠陥画像 | constant/falling-rate、D(c,T)、外部移動係数、energy |
| electroplating | mass gain、断面膜厚、電流効率、bath conductivity、cell voltage、AFM/粗さ、添加剤濃度 | limiting current、ohmic drop、efficiency、粗さ・添加剤 |
| coating | pump flow、gap、速度、粘度曲線、surface tension、contact angle、wet thickness、air entrainment／ribbing画像 | coating window、transfer ratio、Ca境界、tension効果 |
| multistage drying | 同一試料をstage境界で採取し、残留溶媒、膜厚、stress、Tg、skinを測定 | state handoff、移動境界、skin透過性 |
| multistage curing | 乾燥後状態を入力として、温度履歴、DSC、残留溶媒、blister面積、接着試験 | cure・escape・gel化の競合、strength／defectの実測対応 |

文献値を新しい合成係数へ直ちに埋め込むのではなく、まず材質・装置・出力定義が同一の校正用データと、外部検証用データを分けるべきである。特に、現在の合成指標は、測定された物理量へ名前を合わせてから精度指標を計算する必要がある。

## 15. 追加文献監査（2026-09-11追補）

### 15.1 既存文献セットは妥当か

結論は「モデルの構造・増減傾向を評価する資料群としては概ね妥当だが、絶対値の精度を主張するには直接比較可能な一次実験が不足する」である。既存62件のうち、査読一次研究、大学・公的機関のリポジトリ、公開データセットは引き続き採用できる。一方、レビュー、メーカーの工学式、数値解析だけのデータ、材料や出力定義が異なる研究は、真値ではなく構造確認用に限定すべきである。

今回の監査では、題名・DOI・掲載誌を原典側で再照合した。主な書誌修正は、[^26]、[^32]、[^34]、[^35]、[^45]、[^49]、[^52]、[^53]、[^58]、[^59]である。旧[^33]に付けていたScribdコピーは原典性が弱いためリンクを外した。これらは論旨を反転させる誤りではなかったが、後から再現可能な調査記録にするため修正した。

| ステージ | 既存参照の妥当性 | 今回追加した強い参照 | 監査後の扱い |
|---|---|---|---|
| laser_welding | 支配則と高出力一点比較は妥当。ただし材料不一致 | AA6082の反復実験、316L実験ベンチマーク、316L CFD公開データ[^63][^64][^65] | 構造B、絶対値C。新データで面比較可能だが材料別に分ける |
| milling | 標準式と実測Ra表があり良好 | C45 face millingのDoE・回帰[^66] | MRRはA、Raは式名の不整合によりC、力・動力はB |
| press_forming | 1–2 mm材の一次実験があり良好 | 2 mm CuZn30、2 mm SUS304の実験[^67][^68] | クリアランス依存に反証があり、現式の絶対値はC |
| thermal_curing | DSC/isoconversional研究の選択は妥当 | DGEBA/DDSの転化率、DGEBA/TETA、BF3-amine、接着剤の逆解析[^69][^70][^71][^72] | EaのオーダーはB、全転化率曲線と強度はC |
| convection_drying | 輸送モデル中心で、直接測定が不足 | FTIR乾燥曲線、HADES、強制対流インク膜実験[^73][^74][^75] | 温度・膜厚傾向はB、残留量・欠陥はC |
| electroplating | Faraday則と同一運転点があり最も強い部類 | AFMによる電流密度・温度・添加剤の粗さ研究[^81] | 膜厚A〜B、効率B、粗さ・電圧C |
| multistage coating | coating window文献は妥当だが絶対膜厚の比較が弱い | tensioned-web実験2件、二層解析、電池電極・PEO実験[^76][^77][^78][^79][^80] | 安定窓B、gap固定比による膜厚はC |
| multistage drying | 薄膜乾燥文献を共有できる | FTIR、HADES、強制対流実験[^73][^74][^75] | 状態受渡しB、残留溶媒・stressはC |
| multistage curing | 硬化文献を共有できる | 等温転化率と接着物性の追加資料[^69][^71][^72] | 初期速度B、高転化率・bond strength・blisterはC |

### 15.2 追加調査で得た定量的な更新

#### レーザー溶接

SchmoellerらのAA6082実験は、焦点径100 µm、出力600–1200 W、速度4–10 m/minをfull factorialで調べ、各条件10本、合計30断面の平均溶込みを使っている。試験点には400 Wと1400 Wも含む。[^63] 現コードの入力範囲と重なるため、従来の6 kW一点より外部検証に向く。ただし材料がmild steelではなくAA6082であり、論文図からの値抽出も必要なので、現段階では誤差率を付けず「最優先のデジタイズ候補」とした。

316Lについては、2.5 kW、600・1600・2600 mm/minで溶融池寸法、温度、残留応力を測った実験ベンチマークがある。[^65] また、4–8 kW、0.2–0.4 m/s、複数ビーム形状の73ケースを含む高忠実度CFDデータが公開された。[^64] 後者は実験真値ではないが、現モデルの `P/(d v)` 型スケーリングが広い領域で過度に単純化されていないかを自動照合する数値ベンチマークとして有用である。

#### 切削

既存[^17]は、nose radius 0.397 mm、feed 0.0254–0.635 mm/toothに対する反復Ra表を持ち、例えば0.0254で0.170 µm、0.127で0.740 µm、0.203で4.210 µmを報告する。このため、粗さの追加校正では標準幾何式より優先すべき一次表である。追加したC45のface-milling DoEも、feedが力・粗さの主因、切込みが中程度、速度が比較的小さい因子という順位を示す。[^66] 現モデルはfeedと切込みの方向が合い、追加監査後に`geometric_ra`もRt相当から標準Ra式へ修正した。総Raの経験項は引き続き未校正である。

#### プレス・ブランキング

2 mm CuZn30の実験ではclearance 0.08–0.18 mm、すなわち板厚比4–9%で、clearance増加に伴いバリが増え、blanking forceとsmooth-sheared ratioが低下した。[^67] 初回モデルは8%を境に荷重を再増加させていたため、追加監査後に全入力範囲で荷重が低下する式へ修正した。

さらに2 mm SUS304の実験では、clearance約15%tで寸法精度・断面品質・総合品質が最良と報告される。[^68] これは最小バリだけを目的とするモデルと同一指標ではないが、最適clearanceが材料・品質定義に依存する証拠である。そこで固定8%最適を除去し、速度200 mm/s、holder 60 kNでは8%から15%でバリ0.0464→0.14265 mm、peak force 15.1418→14.3351 kNとなるよう修正した。

#### 熱硬化・多段階硬化

DGEBA/DDSの一次研究はNIRとDSCで転化率を測り、130 ℃と205 ℃を含む転化率–時間曲線を提示している。[^69] DGEBA/TETAの非等温DSCではautocatalyticなŠesták–Berggren型が選ばれ、Ea=69.5 kJ/molが得られた。[^70] DGEBA/BF3-amineの等温DSCは125–175 ℃を直接カバーし、初期反応用 `k1` と自己触媒重合用 `k2` の二つが必要である。[^71] したがって、コードの65/62 kJ/molは代表Eaとして妥当な範囲だが、単一一次反応で全温度・全転化率を表す根拠にはならない。

接着剤の実測と逆解析を組み合わせた研究も、硬化速度の同定には複数温度の転化率曲線が必要であることを示す。[^72] また、硬化レベルと機械物性の関係は単調一対一ではなく、既存[^37]では弾性率、延性、Tg、lap shear、T-peelが硬化条件により異なる変化をする。よって `bond_strength_mpa = f(cure_fraction, degradation)` の一式を普遍的な強度式とみなすことはできない。

#### 対流乾燥・多段階乾燥

PVAc–methanol/benzene膜をFTIRで追跡した実験では、25/40 ℃、平行風速0.05–0.70 m/sで、乾燥時間は膜厚の二乗に強く依存し、後半は表面のdry skinと濃度依存拡散が律速となった。[^73] HADES実験では風速0.072–0.72 m/sに対し熱伝達係数7–26 W m⁻² K⁻¹を測定し、PMMA–acetoneでは残留溶媒が風量に対して単調でなく、中間風量で最小になる例も得ている。[^74] 強制対流インク膜の博士研究も、定率期では風速・温度・膜厚・初期溶媒率、減率期では濃度依存拡散が支配することを示す。[^75]

したがって、現コードの温度上昇・滞留時間増加で乾燥が進む方向は支持されるが、風速を常に単調改善因子とすること、一定の有効拡散式、単一skinning penaltyは材料によって外れる。残留溶媒の絶対値を評価するには、対象樹脂–溶媒系を固定した重量または分光時系列が必要である。

#### 塗工

tensioned-web slot coatingのpilot実験では、最小wet thicknessが0.5–1.5 µmまで到達し、web tensionを含む無次元群に比例した。[^76] 可視化実験は、bead breakup、weeping、feed-slot vortexの限界を別々に測定している。[^77] 二層モデルでも速度・粘度・tension・各層流量でcoating windowが変わる。[^78] これらは、現コードのtensionを膜厚への小さな線形補正と二次defect penaltyだけで表す構造が不足していることを示す。

電池電極の一次実験では、10 m/minでgap 180→wet 156 µm、gap 300→wet 260 µmが報告されている。[^79] 現コードを同じgap・速度、粘度1.6 Pa·s、tension 100 N、solids 0.5、bubble 0で計算すると129.37 µmと215.61 µmで、どちらも文献値より17.1%低い。ただし、slot dieは本来flow rateで膜厚をpremeterするため、この17.1%をモデル誤差と断定するのではなく、flow rate入力を持たずgap比だけで厚さを決める構造欠落の検出値と扱う。

PEO電解質の2025年実験は、15–25 wt%のshear-thinning液についてgap、速度、吐出量を変え、約Ca=1付近を含む安定窓を測定した。[^80] 現コードのCa窓は固定表面張力と単一粘度で決まるため、この公開本文は非Newton流体での次の検証候補になる。

#### 電気めっき

既存文献のうち、最も直接的なのは同一運転点のcurrent efficiencyを持つ[^58]と、60条件の公開データを持つ[^60]であり、両者は維持する。追加したAFM研究は1.83–2.53 A/dm²で温度、電流密度、leveling agent、時間に対する粗さ発達を測り、短時間像から工業時間スケールの粗さを予測している。[^81] これは現コードの粗さが電流・温度だけでほぼ決まり、添加剤と析出時間に依存する成長則を持たない点を直接検証できる。

### 15.3 追加後の資料数と優先順位

今回19件を追加し、追跡可能な資料は重複を除いて81件となった。ここには査読論文だけでなく、公的データセット、博士論文、標準式資料を含む。ステージ別の参照プールは共有文献を重複計上して、laser 16、milling 9、press 8、thermal curing 14、single drying 10、electroplating 10、multistage coating 14、multistage drying 10、multistage curing 15件である。

次に実データを抽出して一括誤差表を作る優先順位は、(1) NIMS銅めっき60条件、(2) AA6082レーザー溶込み反復実験、(3) millingのRa数表、(4) 2 mmブランキング、(5) FTIR/HADES乾燥曲線、(6) DGEBA等温転化率、(7) slot-die安定窓である。これらは入力と出力の対応が比較的明確で、グラフのデジタイズまたは公開表の転記によりRMSE、MAE、bias、順位相関を計算できる。

## 16. 文献監査に基づく実装修正

2026-09-11、式の定義または増減方向について複数の資料から十分な根拠が得られた2工程を修正した。入出力列は変更していない。

| 工程 | 修正 | 根拠 | バージョン |
|---|---|---|---|
| milling | 幾何粗さをRt相当の `1000 fz²/(8r)` からRaの `1000 fz²/(32r)` へ変更。Ra制約を7.50から2.50 µmへ更新 | 標準幾何式と実測Ra研究[^15][^17] | 2.0.0 → 2.1.0 |
| press_forming | clearance–peak forceのU字を単調低下へ変更。burrの固定8%最適を除去し、4〜15%で単調増加、3〜4%のみ低clearance penaltyとした | 1–2 mm AA5754、CuZn30、SUS304実験[^22][^23][^67][^68] | 1.0.0 → 1.1.0 |

修正後の全候補監査では、millingは65,536点中61,278点がRa≤2.50 µmかつpower≤14 kWを満たす。press_formingは1,521点中1,500点が二制約を満たし、peak force範囲12.651–19.947 kN、flatness範囲0.0331–0.2346 mmとなった。両工程とも適合・不適合点が残り、最適化ベンチマークとしての非自明性を維持している。

`python -m unittest discover -s validation -p 'test*.py' -v` に相当する全100テストを実行し、すべて成功した。Ra標準式の数値テストと、pressのclearanceに対するburr増加・force低下の回帰テストも追加した。

レーザー、熱硬化、単独乾燥、電気めっき、多段階coating/drying/curingは変更しなかった。これらは不足変数や未校正係数が課題だが、現在のインターフェースのまま係数だけを置換すると別材料の値を恣意的に移植することになる。修正には、材料・装置を固定した校正データ、またはflow rate、添加剤、反応系、濃度依存拡散などの新しい入力契約が必要である。

真値関数とmillingの制約閾値が変わったため、旧モデル版で生成された最適化結果を新モデル版の結果と同一系列として直接比較してはならない。manifestのmodel versionとprovenance hashで区別する。

## 17. 参考文献（原典追跡用）

以下は本文で参照した原典・公開データ・標準式である。出版社側で本文が読めない場合も、DOI、大学リポジトリ、公開データセットのいずれかへリンクしている。

### レーザー溶接

[^1]: TWI, “Laser Welding of Ultra-High Strength Steels for Automotive Applications,” 2008. [公開ページ](https://www.twi-global.com/technical-knowledge/published-papers/laser-welding-of-ultra-high-strength-steels-for-automotive-applications-april-2008).
[^2]: Y. Kawahito, M. Mizutani, and S. Katayama, “Investigation of High-Power Fiber Laser Welding Phenomena of Stainless Steel,” Transactions of JWRI 36(2), 11–15, 2007. [大阪大学リポジトリ](https://ir.library.osaka-u.ac.jp/repo/ouka/all/7955/) / [DOI](https://doi.org/10.18910/7955).
[^3]: M. Fabbro, “Melt Pool and Keyhole Behaviour Analysis for Deep Penetration Laser Welding,” Journal of Physics D: Applied Physics 43, 445501, 2010. [DOI](https://doi.org/10.1088/0022-3727/43/44/445501) / [HAL](https://hal.archives-ouvertes.fr/hal-00569740).
[^4]: J. Kim and H. Ki, “Scaling Law for Penetration Depth in Laser Welding,” Journal of Materials Processing Technology 214, 2908–2914, 2014. [DOI](https://doi.org/10.1016/j.jmatprotec.2014.06.025) / [出版社ページ](https://www.sciencedirect.com/science/article/pii/S0924013614002489).
[^5]: P. Briand, M. Fabbro, and A. Coste, “Study of Keyhole Geometry for Full Penetration Nd:YAG CW Laser Welding,” Journal of Physics D: Applied Physics 38, 2005. [DOI](https://doi.org/10.1088/0022-3727/38/12/005).
[^6]: “Asymptotic Analysis for Penetration Depth During Laser Welding,” Procedia Engineering 15, 5212–5216, 2011. [DOI](https://doi.org/10.1016/j.proeng.2011.08.966).
[^7]: “Penetration-Depth Control in a Remote Laser-Welding System,” Optics and Lasers in Engineering, 2020. [DOI](https://doi.org/10.1016/j.optlaseng.2020.106464).
[^8]: “Deep Penetration Welding of Thick Section Steels with 10 kW Fiber Laser,” Journal of the Japan Welding Society 27(2). [J-STAGE本文](https://www.jstage.jst.go.jp/article/qjjws/27/2/27_2_64s/_article) / [PDF](https://www.jstage.jst.go.jp/article/qjjws/27/2/27_2_64s/_pdf).
[^9]: “Elucidation of Laser Welding Phenomena by High-Speed Video and X-Ray Transmission Imaging,” Physics Procedia 5, 9–17, 2010. [DOI](https://doi.org/10.1016/j.phpro.2010.08.024).
[^10]: “Correlation between Spatters and Evaporation Vapor in High-Power Laser Welding,” Journal of Materials Research and Technology 9, 15143–15152, 2020. [DOI](https://doi.org/10.1016/j.jmrt.2020.10.103).
[^11]: “The Full Penetration Hole as a Stochastic Process in Deep Penetration Laser Welding,” Applied Physics B 108, 97–107, 2012. [DOI](https://doi.org/10.1007/s00340-012-5104-8).
[^12]: “Laser Welding Process Data,” University of Stuttgart / DaRUS dataset. [データセット](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-2111).
[^13]: “Frontal Pyrometric Snapshot of the Keyhole in Laser Welding,” 2023. [DOI](https://doi.org/10.1007/s40194-023-01636-x).

### 切削

[^14]: Sandvik Coromant, “Formulas and Definitions for Milling – Metric.” [公式PDF](https://cdn.sandvik.coromant.com/files/sitecollectiondocuments/services/metal-cutting-e-learning/formulas-and-definitions/formulas-and-deinitions-for-milling-metric-enu.pdf).
[^15]: P. Munoz Escalona and P. Maropoulos, “A Geometrical Model for Surface Roughness Prediction when Face Milling Al 7075-T7351 with Square Insert Tools,” 2014. [大学リポジトリ](https://strathprints.strath.ac.uk/48985/) / [PDF](https://strathprints.strath.ac.uk/48985/1/Munoz_Escalona_P_Pure_A_geometrical_model_for_surface_roughness_prediction_when_face_milling_Jun_2014.pdf).
[^16]: “The Influence of Feed Rate and Cutting Speed on the Cutting Forces and Surface Roughness in High-Speed Face Milling of AISI 1020 and AISI 1040 Steels,” Materials & Design 28, 2007. [DOI](https://doi.org/10.1016/j.matdes.2005.06.002).
[^17]: “Models for Prediction of Surface Roughness in a Face Milling Process Using Triangular Inserts,” Lubricants 7, 9, 2019. [MDPI本文](https://www.mdpi.com/2075-4442/7/1/9).
[^18]: “Prediction of Regenerative Chatter by Modelling and Analysis of High-Speed Milling,” International Journal of Machine Tools & Manufacture, 2003. [出版社ページ](https://www.sciencedirect.com/science/article/pii/S0890695503001718).
[^19]: “Multi-Sensor Monitoring Dataset for Milling Process,” Data in Brief 55, 110703, 2024. [DOI](https://doi.org/10.1016/j.dib.2024.110703) / [Zenodo](https://zenodo.org/records/10613521).
[^20]: “A New Open Dataset from a Milling Process,” Scientific Data, 2025. [DOI](https://doi.org/10.1038/s41597-025-04923-y) / [本文](https://www.nature.com/articles/s41597-025-04923-y.pdf).
[^21]: “VP100 Milling Dataset / Tool Wear and Process-Parameter Study,” 2026. [DOI](https://doi.org/10.1007/s00170-026-18658-6) / [Zenodo](https://zenodo.org/records/20147269).

### プレス・ブランキング

[^22]: B. Çavuşoğlu and H. Gürün, “The Relationship of Burr Height and Blanking Force with Clearance in Blanking Process,” Transactions of FAMENA 41(1), 55–62, 2017. [本文](https://hrcak.srce.hr/179259) / [DOI](https://doi.org/10.21278/TOF.41105).
[^23]: M. Küçüktürk, “Experimental Investigation of Blanking Force, Burr Height and Smooth-Sheared Ratio in Blanking of AA5754,” Gazi University Journal of Science 31(2), 285–294, 2016. [PDF](https://dergipark.org.tr/en/download/article-file/225433).
[^24]: H. Nagai and K. Kaneko, “Experimental Studies on Burrs of Thick Metal Plate in Blanking,” Journal of the Japan Society for Technology of Plasticity 52(604), 564–568, 2011. [J-STAGE](https://www.jstage.jst.go.jp/article/sosei/52/604/52_604_564/_article/-char/en) / [DOI](https://doi.org/10.9773/sosei.52.564).
[^25]: E. M. Gaudillière, “High Speed Blanking,” doctoral thesis, Arts et Métiers. [本文](https://sam.ensam.eu/handle/10985/8094) / [PDF](https://sam.ensam.eu/bitstream/handle/10985/8094/PIMM-GAUDILLIERE-EM-2013.pdf).
[^26]: A. Tekiner et al., “An Experimental Study for the Effect of Different Clearances on Burr, Smooth-Sheared and Blanking Force on Aluminium Sheet Metal,” Materials & Design 27(10), 1134–1138, 2006. [DOI](https://doi.org/10.1016/j.matdes.2005.03.013) / [出版社ページ](https://www.sciencedirect.com/science/article/pii/S0261306905000828).
[^27]: Fraunhofer, “Numerical Investigation of the Blanking Process,” ABAQUS/Explicit and experimental comparison. [研究データベース](https://publica.fraunhofer.de/entities/publication/b5cb5906-9064-4455-88f8-67a2fa875f44).

### エポキシ硬化・接着

[^28]: D. G. Anderson, “Thermal Stability of High Temperature Epoxy Adhesives by Thermogravimetric and Adhesive Strength Measurements,” Polymer Degradation and Stability 96, 1874–1881, 2011. [DOI](https://doi.org/10.1016/j.polymdegradstab.2011.07.010) / [公開PDF](https://dataset-dl.liris.cnrs.fr/db_amethyst/PDFs/10.1016/j.polymdegradstab.2011.07.010.pdf).
[^29]: J. Perrin et al., “Kinetic Analysis of Isothermal and Nonisothermal Epoxy-Amine Cures by Model-Free Isoconversional Methods,” Macromolecular Chemistry and Physics 208, 718–729, 2007. [DOI](https://doi.org/10.1002/macp.200600614).
[^30]: “Learning about Epoxy Cure Mechanisms from Isoconversional Analysis of DSC Data,” Thermochimica Acta 388, 289–298, 2002. [DOI](https://doi.org/10.1016/S0040-6031(02)00053-9).
[^31]: “Is the Kamal Model Appropriate for Modelling Cure Kinetics of Epoxy Resins?,” Thermochimica Acta 505, 47–52, 2010. [DOI](https://doi.org/10.1016/j.tca.2010.03.024).
[^32]: T. Kamon and K. Saito, “Isothermal Cure Kinetics of Epoxy Resin with Various Polyamines by DSC / 種々のアミンによるエポキシ樹脂の硬化反応,” 高分子論文集 41(5), 293–299, 1984. [DOI](https://doi.org/10.1295/koron.41.293) / [国立国会図書館](https://ndlsearch.ndl.go.jp/books/R000000004-I2984986).
[^33]: B. Macan et al., “DSC Study of Cure Kinetics of DGEBA-Based Epoxy Resin with Poly(Oxypropylene) Diamine,” Journal of Thermal Analysis and Calorimetry 81, 369–373, 2005. [J-GLOBAL書誌](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=200902271492779763).
[^34]: D. J. T. Hill, G. A. George, and D. G. Rogers, “A Systematic Study of the Microwave and Thermal Cure Kinetics of the DGEBA/DDS and DGEBA/DDM Epoxy-Amine Resin Systems,” Polymers for Advanced Technologies 13(5), 353–362, 2002. [DOI](https://doi.org/10.1002/pat.198) / [CiNii書誌・要旨](https://cir.nii.ac.jp/crid/1360869859561617280).
[^35]: N. Sbirrazzuoli et al., “Isoconversional Kinetic Analysis of Stoichiometric and Off-Stoichiometric Epoxy-Amine Cures,” Thermochimica Acta 447(2), 167–177, 2006. [DOI](https://doi.org/10.1016/j.tca.2006.06.005) / [出版社ページ](https://www.sciencedirect.com/science/article/pii/S0040603106003297).
[^36]: M. A. Bashir, “Cure Kinetics of Commercial Epoxy-Amine Products with Iso-Conversional Methods,” Coatings 13(3), 592, 2023. [MDPI本文](https://www.mdpi.com/2079-6412/13/3/592) / [DOI](https://doi.org/10.3390/coatings13030592).
[^37]: I. Stewart, A. Chambers, and T. Gordon, “The Cohesive Mechanical Properties of a Toughened Epoxy Adhesive as a Function of Cure Level,” International Journal of Adhesion and Adhesives 27(4), 277–287, 2007. [DOI](https://doi.org/10.1016/j.ijadhadh.2006.05.003) / [出版社ページ](https://www.sciencedirect.com/science/article/pii/S0143749606000698).

### 乾燥・溶媒移動

[^38]: S. Naseri et al., “A Predictive Transport Model for Convective Drying of Polymer Strip Films Loaded with a BCS Class II Drug,” European Journal of Pharmaceutics and Biopharmaceutics 137, 164–174, 2019. [PMC本文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6449172/) / [DOI](https://doi.org/10.1016/j.ejpb.2019.02.023).
[^39]: “Modelling of Drying of Coatings: Effect of the Thickness, Temperature and Concentration of Solvent,” Progress in Organic Coatings 15, 163–172, 1987. [DOI](https://doi.org/10.1016/0033-0655(87)80005-5).
[^40]: R. A. Waggoner and F. D. Blum, “Solvent Diffusion and Drying of Coatings,” Journal of Coatings Technology, 1989. [大学リポジトリ](https://scholarsmine.mst.edu/chem_facwork/1379/).
[^41]: “Modeling the Drying of Polymer Coatings,” Soft Matter, 2022. [DOI](https://doi.org/10.1039/D1SM01343B) / [RSC本文](https://pubs.rsc.org/en/content/articlehtml/2022/sm/d1sm01343b).
[^42]: C. Feger and W. Corso, “Modeling of the Continuous Thermal Drying Process of Epoxy Coatings,” SPE ANTEC, 1997. [IBM Research record](https://research.ibm.com/publications/modeling-of-the-continuous-thermal-drying-process-of-epoxy-coatings).
[^43]: J. Powers et al., “Experimental Modeling of Solvent-Casting Thin Polymer Films,” Polymer Engineering & Science, 1990. [DOI](https://doi.org/10.1002/pen.760300208).
[^44]: M. Yamamura, Y. Mawatari, and H. Kage, “Numerical Modeling of Drying Thin Film Coating with a Surface-Wiping Process,” KAGAKU KOGAKU RONBUNSHU 35(5), 436–441, 2009. [J-STAGE本文](https://www.jstage.jst.go.jp/article/kakoronbunshu/35/5/35_5_436/_article/-char/en).

### 塗工・slot-die

[^45]: M. S. Carvalho and H. S. Kheshgi, “Minimum Wet Thickness in Extrusion Slot Coating,” Chemical Engineering Science 47(7), 1703–1713, 1992. [DOI](https://doi.org/10.1016/0009-2509(92)85018-7).
[^46]: “Three Minimum Wet Thickness Regions of Slot Die Coating,” Journal of Colloid and Interface Science 308, 222–230, 2007. [DOI](https://doi.org/10.1016/j.jcis.2006.11.054).
[^47]: “Improved Coating Window for Slot Coating,” Industrial & Engineering Chemistry Research, 2010. [DOI](https://doi.org/10.1021/ie801900t).
[^48]: “Experimental Study on Air Entrainment in Slot-Die Coating,” Chemical Engineering Science 80, 195–204, 2012. [DOI](https://doi.org/10.1016/j.ces.2012.06.033).
[^49]: X. Ding, J. Liu, and T. A. L. Harris, “A Review of the Operating Limits in Slot Die Coating Processes,” AIChE Journal 62(7), 2508–2524, 2016. [DOI](https://doi.org/10.1002/aic.15268) / [書誌・要旨](https://oamonitor.ireland.openaire.eu/national/search/publication?pid=10.1002%2Faic.15268).
[^50]: R. Diehm et al., “In Situ Investigations of Simultaneous Two-Layer Slot Die Coating of Component-Graded Anodes for Improved High-Energy Li-Ion Batteries,” Energy Technology 8, 1901251, 2020. [Wiley本文](https://onlinelibrary.wiley.com/doi/full/10.1002/ente.201901251).
[^51]: R. Diehm et al., “High-Speed Coating of Primer Layer for Li-Ion Battery Electrodes by Using Slot-Die Coating,” Energy Technology 8, 2000259, 2020. [Wiley本文](https://onlinelibrary.wiley.com/doi/full/10.1002/ente.202000259).
[^52]: “Two-Layer Tensioned-Web-over-Slot Die Coating: Effect of Operating Conditions on Coating Window,” Chemical Engineering Science 65, 4065–4079, 2010. [DOI](https://doi.org/10.1016/j.ces.2010.03.038).
[^53]: S. Spiegel et al., “High-Speed Slot-Die Coating of Primer Layers for Li-Ion Battery Electrodes: Model Calculations and Experimental Validation of the Extended Coating Window Depending on Coating Speed, Coating Gap and Viscosity,” Journal of Coatings Technology and Research 21, 2024. [DOI](https://doi.org/10.1007/s11998-023-00877-1) / [KIT公開本文](https://publikationen.bibliothek.kit.edu/1000167973/152151672).

### 電気めっき

[^54]: “Current versus Potential Studies for Copper Electrodeposition at a Rotating Disc Electrode,” Transactions of the Institute of Metal Finishing 74(1), 1996. [DOI](https://doi.org/10.1080/00202967.1996.11871089).
[^55]: C.-C. Hsueh and J. Newman, “Mass Transfer and Polarization at a Rotating Disk Electrode,” Electrochimica Acta 12, 429–432, 1967. [DOI](https://doi.org/10.1016/0013-4686(67)80087-2).
[^56]: “Electrodeposition of Copper from Cuprous Cyanide Electrolyte: Mass Transfer and Current Distribution,” Journal of Electroanalytical Chemistry. [DOI](https://doi.org/10.1016/S0022-0728(99)00300-9).
[^57]: R. Palli, “Theoretical and Experimental Study of Copper Electrodeposition in a Modified Hull Cell,” 2016. [DOI](https://doi.org/10.1155/2016/3482406) / [Wiley本文](https://onlinelibrary.wiley.com/doi/10.1155/2016/3482406).
[^58]: “Antipathogenic Copper Coatings: Electrodeposition Process and Microstructure Analysis,” Archives of Civil and Mechanical Engineering, 2023. [Springer本文](https://link.springer.com/article/10.1007/s43452-023-00772-x).
[^59]: “Evaluation of Adhesion Properties of Electrodeposited Copper Thin Films: Theoretical and Experimental Approach,” 2025. [PMC本文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12156451/).
[^60]: T. Tamura et al., “Predicting Surface Roughness of Electrodeposited Copper by Machine Learning,” STAM Methods 4, 2416889, 2024. [DOI](https://doi.org/10.1080/27660400.2024.2416889) / [NIMSデータリポジトリ](https://mdr.nims.go.jp/datasets/5bc7ba84-83f9-43e6-9de5-e6f1aa5bc0e4?locale=en).
[^61]: “High-Rate Copper Electrodeposition from Copper Sulfate–Sulfuric Acid Electrolytes,” Eindhoven University of Technology. [公開PDF](https://pure.tue.nl/ws/portalfiles/portal/2271764/620039.pdf).
[^62]: “Effects of Organic Additives on Copper Electrodeposition and Surface Morphology,” 2023. [公開論文](https://www.sciencedirect.com/science/article/pii/S1452398123028912).

### 追加監査で採用した文献・データ

[^63]: M. Schmoeller et al., “Investigation of the Influences of the Process Parameters on the Weld Depth in Laser Beam Welding of AA6082 Using Machine Learning Methods,” Procedia CIRP 94, 702–707, 2020. [DOI](https://doi.org/10.1016/j.procir.2020.09.121) / [公開PDF](https://researchmgt.monash.edu/ws/portalfiles/portal/727562552/472042352_oa.pdf).
[^64]: J. D. Neuhauser et al., “Weld Pool and Keyhole Geometry Dataset from High-Fidelity CFD Simulations of Laser Beam Welding,” 73 simulations, 316L, 2026. [Zenodoデータセット](https://zenodo.org/records/20413437).
[^65]: “Experimental Study of Thermomechanical Processes: Laser Welding and Melting of a Powder Bed,” Crystals 10(4), 246, 2020. [本文・実験表](https://www.mdpi.com/2073-4352/10/4/246).
[^66]: “Experimental Investigation and Multi-Objective Optimization of High-Feed Face Milling of C45 Steel,” Lubricants 14(2), 71, 2026. [本文](https://www.mdpi.com/2075-4442/14/2/71).
[^67]: O. Çavuşoğlu and H. Gürün, “Investigation and Fuzzy Logic Prediction of the Effects of Clearance on the Blanking Process of CuZn30 Sheet Metal,” Kovove Materialy 54(2), 125–131, 2016. [DOI](https://doi.org/10.4149/km_2016_2_125) / [書誌・要旨](https://avesis.gazi.edu.tr/yayin/77bcedcc-f042-432e-8714-2050f8aa3aa9/investigation-and-fuzzy-logic-prediction-of-the-effects-of-clearance-on-the-blanking-process-of-cuzn30-sheet-metal).
[^68]: P. Li et al., “Investigation on Cold-Blanking and Blanked Edge Quality of 304 Stainless Steel Sheet,” Materials Science and Technology 26(2), 34–40, 2018. [DOI・本文](https://doi.org/10.11951/j.issn.1005-0299.20170121) / [公開ページ](https://hit.alljournals.cn/mst_cn/article/html/20180206).
[^69]: “Cure Kinetics of Elementary Reactions of a DGEBA/DDS Epoxy Resin: 1. Glass Transition Temperature versus Conversion,” Polymer 34(23), 4908–4912, 1993. [DOI](https://doi.org/10.1016/0032-3861(93)90017-5) / [公開PDF](https://cpsm.kpi.ua/polymer/1993/23/4908-4912.pdf).
[^70]: “Cure Kinetics of Epoxy Resins Studied by Non-Isothermal DSC Data,” Thermochimica Acta 383, 119–127, 2002. [DOI](https://doi.org/10.1016/S0040-6031(01)00672-4).
[^71]: M. Ghaemy and H. Khandani, “Kinetics of Curing Reaction of DGEBA with BF3-Amine Complexes Using Isothermal DSC Technique,” European Polymer Journal 34(3–4), 477–486, 1998. [DOI](https://doi.org/10.1016/S0014-3057(97)00122-5) / [本文コピー](https://www.researchgate.net/publication/244062557_Kinetics_of_curing_reaction_of_DGEBA_with_BF-amine_complexes_using_isothermal_DSC_technique).
[^72]: “Cure Kinetics and Inverse Analysis of Epoxy-Amine Based Adhesive Used for Fastening Systems,” Materials 14(14), 3853, 2021. [本文](https://www.mdpi.com/1996-1944/14/14/3853).
[^73]: “Drying of Solvent-Borne Polymeric Coatings: II. Experimental Results Using FTIR Spectroscopy,” Surface and Coatings Technology 99(3), 257–265, 1998. [DOI](https://doi.org/10.1016/S0257-8972(97)00565-3) / [出版社要旨](https://www.sciencedirect.com/science/article/pii/S0257897297005653).
[^74]: M. Vinjamur and R. A. Cairncross, “A High Airflow Drying Experimental Set-up to Study Drying Behavior of Polymer Solvent Coatings,” Drying Technology 19(8), 1591–1612, 2001. [DOI](https://doi.org/10.1081/DRT-100107261) / [書誌・要旨](https://openalex.org/W2086801055).
[^75]: H. Hardisty, “An Investigation into the Drying of Thin Films of Ink, Using Infra-Red Dryness Measurement,” doctoral thesis, University of Bath, 1980. [大学リポジトリ・本文](https://researchportal.bath.ac.uk/en/studentTheses/an-investigation-into-the-drying-of-thin-films-of-ink-using-infra/).
[^76]: F. Lin et al., “Experimental Study on Tensioned-Web Slot Coating,” Polymer Engineering & Science 47, 841–851, 2007. [DOI](https://doi.org/10.1002/pen.20764).
[^77]: “Flow Visualization and Operating Limits of Tensioned-Web-over-Slot Die Coating Process,” Chemical Engineering and Processing 50, 471–477, 2011. [DOI](https://doi.org/10.1016/j.cep.2010.09.005).
[^78]: “Two-Layer Tensioned-Web-over-Slot Die Coating: Effect of Operating Conditions on Coating Window,” Chemical Engineering Science 65, 4065–4079, 2010. [DOI](https://doi.org/10.1016/j.ces.2010.03.038).
[^79]: S. Spiegel et al., “Optimization of Edge Quality in the Slot-Die Coating Process of High-Capacity Lithium-Ion Battery Electrodes,” Energy Technology 11(5), 2200684, 2023. [DOI](https://doi.org/10.1002/ente.202200684) / [KIT公開本文](https://publikationen.bibliothek.kit.edu/1000151313/149420846).
[^80]: A. Wiegandt et al., “Process Window Evaluation for Slot Die Coating of PEO-Based Electrolytes in All-Solid-State Batteries,” Energy Technology, 2500457, 2025. [DOI](https://doi.org/10.1002/ente.202500457) / [Fraunhofer公開記録](https://publica.fraunhofer.de/entities/publication/622aaad2-4f57-4c2f-b1d4-9ee6e8f9887f).
[^81]: T. Zhao et al., “Application of Atomic Force Microscopy and Scaling Analysis of Images to Predict the Effect of Current Density, Temperature and Leveling Agent on the Morphology of Electrolytically Produced Copper,” Electrochimica Acta 51(11), 2255–2260, 2006. [DOI](https://doi.org/10.1016/j.electacta.2005.06.042) / [出版社ページ](https://www.sciencedirect.com/science/article/pii/S0013468605008856).
