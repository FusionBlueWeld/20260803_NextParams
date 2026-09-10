# v003 NN ensemble・不確実性校正 実装前後比較

## 結論

5-member NN ensembleと標準偏差2倍校正は、v003で問題だった**不確実性の過小評価を大幅に改善した**。全体95% coverageは約67%から約94%、高次元では約57～60%から約95%となり、Gaussian NLPDも正の大値から負値へ正常化した。点予測NRMSEと最終regretも改善しており、実装を残す価値は高い。

ただし、最適解への到達は一様には改善していない。batch=1では到達率が73.3%から80.0%へ上がったが、到達までの消費条件中央値は3から4へ増えた。batch=3の到達率と中央値は不変で、thermal curingでは悪化seedがある。したがって本改修は「校正と平均面の明確な改善」であり、「常に探索を高速化する改修」とはまだ言えない。

## 実装

- NNをseed 42～46の5モデルで学習
- `nn_pred`は5予測の平均、`nn_std`はmember間標準偏差
- GPは増やさず、NN ensemble平均に対する残差を出力ごとに1個だけ学習
- `raw_hybrid_std = sqrt(gp_std^2 + nn_std^2)`
- `hybrid_std = 2.0 * raw_hybrid_std`
- optimizerのExpected Improvement、制約達成確率、diversityは校正後`hybrid_std`を利用
- 通常CSVの行数とファイル数は従来どおり。member別列は出さず、`nn_std`、校正前std、倍率を追加
- `ensemble_size=1, calibration_scale=1`で旧v003の不確実性を再現可能

変更対象はv003と、その物理validationだけであり、v002など他バージョンには変更を加えていない。

## 検証条件

- 物理系: thermal curing、press forming、convection drying、electroplating、milling
- seed: 0、1、2
- 総追加取得条件数: 15
- batch size: 1、3
- baseline: 実装前v003、NN 1モデル、校正倍率1
- enhanced: NN 5-member ensemble、校正倍率2
- 同じ物理真値、初期設計、候補格子、知識ルール、ノイズ0
- 同率最適を含む制約付き大域最適目的値で到達判定
- 予測評価は到達バッチ反映直後、未到達は予算末

## 全体結果

| batch | variant | 到達率 | 消費条件中央値 | final regret | global NRMSE | global NLPD | coverage95 | k64 NRMSE | k64 NLPD | k64 coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 73.3% | 3 | 0.003105 | 0.046410 | 6.347 | 66.6% | 0.034470 | 11.812 | 70.9% |
| 1 | enhanced | **80.0%** | 4 | **0.000583** | **0.042472** | **-1.866** | **93.6%** | **0.030085** | **-2.021** | **94.4%** |
| 3 | baseline | 66.7% | 6 | 0.003908 | 0.045877 | 6.446 | 67.6% | 0.035229 | 13.681 | 71.3% |
| 3 | enhanced | 66.7% | 6 | **0.001717** | **0.039929** | **-1.937** | **93.9%** | **0.026391** | **-2.145** | **96.0%** |

改善率は、batch=1でglobal NRMSE 8.5%、k64 NRMSE 12.7%、final regret 81.2%。batch=3ではそれぞれ13.0%、25.1%、56.1%。

## 単純系

| batch | variant | 到達率 | regret | NRMSE | NLPD | coverage | k64 NRMSE | k64 coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 88.9% | 0.001781 | 0.043470 | 3.145 | 73.1% | 0.013210 | 90.4% |
| 1 | enhanced | **100.0%** | **0.000000** | **0.040615** | **-1.934** | **92.9%** | **0.010401** | 99.5% |
| 3 | baseline | 77.8% | 0.002930 | 0.043270 | 3.164 | 72.3% | 0.013873 | 89.5% |
| 3 | enhanced | 77.8% | **0.002298** | **0.039013** | **-1.994** | **92.9%** | **0.009868** | 99.4% |

press formingではbatch=1の到達が2/3から3/3となり、旧版で未到達だったseed 0を2条件目で救済した。convection dryingは概ね維持。thermal curingは不確実性拡大による過探索が見られ、batch=1で到達が遅れ、batch=3では2/3から1/3へ低下した。

## 高次元・複雑系

| batch | variant | 到達率 | regret | NRMSE | NLPD | coverage | k64 NRMSE | k64 coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 50.0% | 0.005092 | 0.050820 | 11.150 | 56.9% | 0.066360 | 41.6% |
| 1 | enhanced | 50.0% | **0.001457** | **0.045258** | **-1.764** | **94.8%** | **0.059609** | **86.7%** |
| 3 | baseline | 50.0% | 0.005376 | 0.049787 | 11.368 | 60.4% | 0.067263 | 43.9% |
| 3 | enhanced | 50.0% | **0.000846** | **0.041304** | **-1.850** | **95.5%** | **0.051174** | **91.0%** |

高次元の到達率は変わらないが、regretはbatch=1で71.4%、batch=3で84.3%低下した。

electroplatingは前後とも全seed未到達だったが、問題だったseed 1・2のregretはbatch=1でそれぞれ0.00570→0.000108、0.02443→0.000587へ改善した。batch=3では3 seedすべてregretが改善した。一方、batch=1 seed 0だけは0.000427→0.00805へ悪化しており、完全なseed頑健性は得られていない。

millingは前後とも全seed到達したが、新実装は探索を広げるため一部seedの到達が遅くなった。点予測と校正は改善している。

## 改善をどう捉えるか

1. **校正は成功**: 目標95%に対し全体約94%、高次元約95%。NLPDも全ケース群で大幅改善。
2. **ensemble平均も有効**: 校正倍率は平均値を変えないため、NRMSE改善はensemble平均と、それを基準にした残差GPによるもの。
3. **探索の破綻は減少**: 最終regretは30ペア中、batch=1で3改善・11同等・1悪化、batch=3で4改善・10同等・1悪化。
4. **到達速度は混在**: 両batchとも到達条件のペア比較は3改善・7同等・5悪化。平均到達率だけで成功判定しない方がよい。
5. **単一倍率の限界**: 単純系k64 coverageは約99.5%でやや保守的なのに対し、高次元k64は86.7～91.0%でまだ過信。空間・工程依存の校正余地がある。

## 採用判断

本実装はv003へ残すことを推奨する。理由は、平均精度を損なわず、主要課題だったcoverage/NLPDと最終regretを大幅に改善したため。

ただし倍率2.0は、同じvalidation群で観測した旧v003のcoverage不足を基に選んだ値であり、完全な外部holdout校正ではない。別の物理系または未使用seedで再確認する必要がある。また探索速度を優先する場合、校正後stdをそのまま2倍にするのではなく、予測区間用stdと獲得関数用stdに別係数を持たせる検討価値がある。

## 最終FIX追記

上記検証後、予測区間用stdと探索用stdを分離した。最終v003は、95%区間と制約達成確率に`predictive_std = 2.0 * raw_std`、Expected Improvementとdiversityに`acquisition_std = 1.0 * raw_std`を使用する。上表のfull validationは分離前（探索にも2.0を使用）の結果であり、最終分離実装については全35回帰テストとthermal curing 1更新のsmoke testを実施した。モデル改善はこの構成でFIXする。

## 再現物

- v003モデル: `v003/src/hybrid/model.py`
- NN seed対応: `v003/src/hybrid/nn_component.py`
- 設定: `v003/src/settings.py`
- optimizer連携: `v003/src/hybrid/optimizer.py`
- CSV出力: `v003/src/hybrid/reporting.py`
- validation: `validation/single/src/checks/v003_uncertainty_ablation.py`
- 生結果: `validation/single/results/v003_uncertainty_ablation_20260909/raw_results.json`
- 自動集計: `validation/single/results/v003_uncertainty_ablation_20260909/report.md`
