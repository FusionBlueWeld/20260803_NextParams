# 複合工程の過去実行結果

現行の合否に使わない試行・ログを、2026-10-08に `validation/multistage/results/` から移動しました。

| 項目 | 移動理由 |
|---|---|
| `stage_loop_smoke3_coating_s0.json` | 単発smoke。9 runの最終集約結果に置き換え済み |
| `stage_closed_loop_production2_validation.json` | 硬化128点の初期失敗。512点で再実行した最終集約結果に含まれる |
| `pre_fix_20260909/` | FIX前の校正・出力 |
| `calibration_fix_run.json` | FIX時の校正方式比較ログ |
| `fix_broad_domain_limitation.json` | FIX時点で記録した広域の既知制限 |
| `v004_r001_fix_manifest.json` | 2026-09-09のソース固定ハッシュ |

現行の個別閉ループ結果は `validation/multistage/results/stage_closed_loop_final_validation.json`、窓検証の結果は同じ現行 `results/` 内の `connected_window_validation.json` と `window_center_joint_validation.json` を参照してください。
