"""pipeline.jsonとstage bundleの接続契約を検査します。

工程順・単位・入力供給元・候補・校正条件を確認し、学習器を読み込む前に
設定の不整合を検出します。prepareとrunは同じ検査を使います。
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .window_calibration import validate_calibration
from .settings import CONFIG_NAME


class UserInputError(Exception):
    """利用者が設定や入力を修正する必要がある場合の例外です。"""

    pass


def load_pipeline_config(path: Path) -> dict[str, object]:
    """設定JSONの形式・版・工程IDを確認し、接続設定を返します。"""

    try:
        config = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserInputError(f"{CONFIG_NAME}を読み取れません: {error}") from error
    if config.get("schema_version") != "1.0":
        raise UserInputError("pipeline.jsonのschema_versionは1.0にしてください。")
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise UserInputError("stagesを1件以上指定してください。")
    ids = [item.get("id") for item in stages]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise UserInputError("stage idは空欄・重複なしで指定してください。")
    return config


def resolve_bundle_path(trial: Path, value: str) -> Path:
    """相対bundleパスはtrialを基準に、絶対パスはそのまま解決します。"""

    path = Path(value)
    return (path if path.is_absolute() else trial / path).resolve()


def load_bundle_manifest(path: Path) -> dict[str, object]:
    """予測器を実行せず、bundleの版と平均予測能力を確認します。"""

    try:
        value = json.loads((path / "stage_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserInputError(f"stage bundleを読み取れません: {path}: {error}") from error
    if value.get("bundle_schema_version") != "1.0" or not value.get("capabilities", {}).get("mean_prediction"):
        raise UserInputError(f"平均予測に対応しないstage bundleです: {path}")
    return value


def validate_pipeline(trial: Path, config: dict[str, object]) -> list[tuple[dict, Path, dict]]:
    """全工程の接続契約と任意評価設定を検査し、工程定義・パス・manifestを返します。"""

    # 工程順・公開出力・入力供給元を確認して、逆向き接続や単位違いを拒否します。
    stages = []
    for item in config["stages"]:
        path = resolve_bundle_path(trial, str(item.get("bundle", "")))
        stages.append((item, path, load_bundle_manifest(path)))
    stage_ids = {str(item[0]["id"]) for item in stages}
    stage_order = {str(item[0]["id"]): index for index, item in enumerate(stages)}
    produced = {(str(item[0]["id"]), name) for item in stages for name in item[2]["connector_outputs"]}
    targets: set[tuple[str, str]] = set()
    for connection in config.get("connections", []):
        try:
            source_stage, source_name = str(connection["from"]).split(".", 1)
            target_stage, target_name = str(connection["to"]).split(".", 1)
        except (KeyError, ValueError) as error:
            raise UserInputError("connectionsのfrom/toはstage.variable形式で指定してください。") from error
        if (source_stage, source_name) not in produced:
            raise UserInputError(f"接続元がbundleのconnector_outputsにありません: {connection['from']}")
        target = next((item for item in stages if str(item[0]["id"]) == target_stage), None)
        if target is None or target_name not in target[2]["incoming_context"]:
            raise UserInputError(f"接続先がbundleのincoming_contextにありません: {connection['to']}")
        if stage_order[source_stage] >= stage_order[target_stage]:
            raise UserInputError(f"初版は順方向の直列接続だけを受け付けます: {connection['from']} -> {connection['to']}")
        key = (target_stage, target_name)
        if key in targets:
            raise UserInputError(f"接続先が重複しています: {connection['to']}")
        targets.add(key)
        source = next(item for item in stages if str(item[0]["id"]) == source_stage)
        source_unit = source[2].get("output_units", {}).get(source_name)
        target_unit = target[2].get("input_units", {}).get(target_name)
        if source_unit is None or target_unit is None or source_unit != target_unit:
            raise UserInputError(f"接続単位が一致しません: {connection['from']}[{source_unit}] -> {connection['to']}[{target_unit}]")
    # 操作候補と、上流が供給しない外部状態を区別して検査します。
    axes = config.get("candidate_axes", {})
    all_controls = {name for _, _, manifest in stages for name in manifest["controls"]}
    if set(axes) != all_controls:
        raise UserInputError(f"candidate_axesは全controlを指定してください: required={sorted(all_controls)}")
    for name, values in axes.items():
        if not isinstance(values, list) or not values:
            raise UserInputError(f"candidate_axes.{name}は1件以上の配列にしてください。")
    external = set(config.get("external_context", {}))
    required_external = {
        f"{item[0]['id']}.{name}"
        for item in stages for name in item[2]["incoming_context"]
        if (str(item[0]["id"]), name) not in targets
    }
    if external != required_external:
        raise UserInputError(f"external_contextが一致しません: required={sorted(required_external)}")
    final_outputs = set(stages[-1][2]["predicted_outputs"])
    for spec in config.get("final_specifications", []):
        if spec.get("name") not in final_outputs:
            raise UserInputError(f"最終仕様の出力が最終stageにありません: {spec.get('name')}")
        if spec.get("direction") not in {"greater_equal", "less_equal", "between"}:
            raise UserInputError("最終仕様のdirectionが不正です。")
        if "margin_scale" in spec and (not np.isfinite(float(spec["margin_scale"])) or float(spec["margin_scale"]) <= 0):
            raise UserInputError("最終仕様のmargin_scaleは0より大きい有限値にしてください。")
    # 任意機能は有効時だけ検査し、欠けた確率設定を推測で補いません。
    variation = config.get("process_variation")
    if variation is not None:
        if not isinstance(variation, dict):
            raise UserInputError("process_variationはobjectで指定してください。")
        if variation.get("distribution") != "independent_clipped_normal":
            raise UserInputError("process_variation.distributionはindependent_clipped_normalにしてください。")
        samples = int(variation.get("samples", 0))
        candidate_limit = int(variation.get("candidate_limit", 0))
        seed = int(variation.get("seed", -1))
        if samples < 100 or candidate_limit < 1 or seed < 0:
            raise UserInputError("process_variationはsamples>=100、candidate_limit>=1、seed>=0にしてください。")
        deviations = variation.get("standard_deviations")
        varied_names = set(axes) | required_external
        if not isinstance(deviations, dict) or set(deviations) != varied_names:
            raise UserInputError(f"process_variation.standard_deviationsが一致しません: required={sorted(varied_names)}")
        if any(not np.isfinite(float(value)) or float(value) < 0 for value in deviations.values()):
            raise UserInputError("process_variationの標準偏差は0以上の有限値にしてください。")
    window = config.get("connected_window")
    if window is not None:
        if not isinstance(window, dict):
            raise UserInputError("connected_windowはobjectで指定してください。")
        threshold = float(window.get("support_threshold", 0.5))
        interval_method = str(window.get("interval_method", "std_multiplier"))
        confidence_z = float(window.get("confidence_z", 0.0))
        scan_points = int(window.get("scan_points", 121))
        if not np.isfinite(threshold) or not 0 <= threshold < 1:
            raise UserInputError("connected_window.support_thresholdは0以上1未満にしてください。")
        if interval_method not in {"std_multiplier", "residual_quantile", "standardized_residual_quantile"}:
            raise UserInputError("connected_window.interval_methodが不正です。")
        if not np.isfinite(confidence_z) or not 0 <= confidence_z <= 5:
            raise UserInputError("connected_window.confidence_zは0以上5以下にしてください。")
        if scan_points < 11 or scan_points > 2001:
            raise UserInputError("connected_window.scan_pointsは11以上2001以下にしてください。")
        for key in ("enforce_local_constraints", "use_for_selection"):
            if key in window and not isinstance(window[key], bool):
                raise UserInputError(f"connected_window.{key}はtrue/falseにしてください。")
        if interval_method in {"residual_quantile", "standardized_residual_quantile"}:
            raw_path = window.get("calibration_file")
            if not isinstance(raw_path, str) or not raw_path:
                raise UserInputError("residual_quantileにはconnected_window.calibration_fileが必要です。")
            calibration_path = Path(raw_path)
            calibration_path = (calibration_path if calibration_path.is_absolute() else trial / calibration_path).resolve()
            try:
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise UserInputError(f"連結窓の残差調整情報を読み取れません: {error}") from error
            try:
                validate_calibration(calibration, config, stages)
            except (ValueError, TypeError, KeyError) as error:
                raise UserInputError(str(error)) from error
            config["_connected_window_calibration"] = calibration
    center_selection = config.get("window_center_selection")
    if center_selection is not None:
        if not isinstance(center_selection, dict) or not isinstance(center_selection.get("enabled", False), bool):
            raise UserInputError("window_center_selectionの形式が不正です。")
        limit = int(center_selection.get("candidate_limit", 0))
        coarse = int(center_selection.get("coarse_scan_points", 0))
        if limit < 1 or coarse < 11 or coarse > 501:
            raise UserInputError("window_center_selectionのcandidate_limit>=1、coarse_scan_points=11..501が必要です。")
        variations = center_selection.get("required_variations", {})
        if center_selection.get("enabled") and set(variations) != set(axes):
            raise UserInputError("window_center_selection.required_variationsは全controlを指定してください。")
        if any(not np.isfinite(float(value)) or float(value) <= 0 for value in variations.values()):
            raise UserInputError("required_variationsは0より大きい有限値にしてください。")
    simultaneous = config.get("simultaneous_window")
    if simultaneous is not None:
        if not isinstance(simultaneous, dict) or not isinstance(simultaneous.get("enabled", False), bool):
            raise UserInputError("simultaneous_windowの形式が不正です。")
        if int(simultaneous.get("samples", 0)) < 100 or int(simultaneous.get("seed", -1)) < 0:
            raise UserInputError("simultaneous_windowはsamples>=100、seed>=0にしてください。")
        scales = simultaneous.get("radius_scales", [])
        if not isinstance(scales, list) or not scales or any(not np.isfinite(float(v)) or float(v) <= 0 for v in scales):
            raise UserInputError("simultaneous_window.radius_scalesは正の数値配列にしてください。")
        if int(simultaneous.get("pair_grid_points", 0)) < 5 or int(simultaneous.get("pair_count", 0)) < 1:
            raise UserInputError("simultaneous_windowのpair_grid_points>=5、pair_count>=1が必要です。")
    return stages

