"""Binding and validation of connected residual calibration artifacts."""

import hashlib
import json
import math


def calibration_contract(config, stages):
    """Bind empirical tuning to the exact chain, manifests and decision policy."""
    window = config.get("connected_window") or {}
    return {
        "stage_order": [str(item["id"]) for item, _, _ in stages],
        "bundle_signatures": {
            str(item["id"]): hashlib.sha256((path / "predictor.pkl").read_bytes()).hexdigest()
            for item, path, _ in stages
        },
        "manifest_signatures": {
            str(item["id"]): hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                                       allow_nan=False).encode("utf-8")).hexdigest()
            for item, _, manifest in stages
        },
        "connections": sorted(config.get("connections", []), key=lambda row: (row["to"], row["from"])),
        "external_context": config["external_context"],
        "final_specifications": config.get("final_specifications", []),
        "enforce_local_constraints": bool(window.get("enforce_local_constraints", False)),
        "support_threshold": float(window.get("support_threshold", 0.0)),
    }


def validate_calibration(calibration, config, stages):
    if not isinstance(calibration, dict):
        raise ValueError("残差調整情報はobjectで指定してください。")
    if calibration.get("schema_version") != "1.1":
        raise ValueError("残差調整情報をschema 1.1で再生成してください。")
    if calibration.get("contract") != calibration_contract(config, stages):
        raise ValueError("残差調整情報の適用条件が不一致です（原料・接続・bundle・制約・支持度）。再校正してください。")
    offsets = calibration.get("offsets")
    if not isinstance(offsets, dict):
        raise ValueError("残差調整情報のoffsetsが不正です。")
    method = config["connected_window"]["interval_method"]
    keys = ("lower", "upper") if method == "residual_quantile" else ("lower_z", "upper_z", "std_floor")
    for item, _, manifest in stages:
        stage_offsets = offsets.get(str(item["id"]))
        if not isinstance(stage_offsets, dict):
            raise ValueError(f"工程の残差調整情報が不足しています: {item['id']}")
        for output in manifest["predicted_outputs"]:
            offset = stage_offsets.get(output)
            if not isinstance(offset, dict) or any(k not in offset for k in keys):
                raise ValueError(f"残差調整情報が不足しています: {item['id']}.{output}")
            if any(not isinstance(offset[k], (int, float)) or not math.isfinite(offset[k]) for k in keys):
                raise ValueError("残差調整値は有限数値で指定してください。")
            if method == "residual_quantile":
                if offset["lower"] > offset["upper"]:
                    raise ValueError("残差区間の上下限が逆転しています。")
            elif offset["lower_z"] < 0 or offset["upper_z"] < 0 or offset["std_floor"] <= 0:
                raise ValueError("標準化残差の係数は非負、std_floorは正にしてください。")


def interval_bounds(values, std, method, offset, confidence_z):
    import numpy as np
    if method == "std_multiplier":
        return values - confidence_z * std, values + confidence_z * std
    if offset is None:
        raise ValueError("残差調整情報がありません。従来方式へ暗黙には切り替えません。")
    if method == "residual_quantile":
        return values + offset["lower"], values + offset["upper"]
    if method == "standardized_residual_quantile":
        effective = np.maximum(std, offset["std_floor"])
        return values - offset["lower_z"] * effective, values + offset["upper_z"] * effective
    raise ValueError(f"不明な区間推定方式: {method}")
