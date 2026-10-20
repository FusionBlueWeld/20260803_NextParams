"""工程の恒等関係を、検査済みの数値演算だけで保ちます。"""

from __future__ import annotations

import numpy as np


def validate_derived_outputs(source, output_names, units):
    """同じ単位の出力だけを接続し、循環のない評価順序を返します。"""
    if not isinstance(source, dict):
        raise ValueError("derived_outputsはobjectにしてください。")
    normalized = {}
    for target, spec in source.items():
        if target not in output_names or not isinstance(spec, dict) or set(spec) != {"operation", "sources"}:
            raise ValueError("derived_outputsの対象または必須項目が不正です。")
        operation, sources = spec["operation"], spec["sources"]
        if operation not in ("max_abs", "max", "sum"):
            raise ValueError("derived_outputsのoperationはmax_abs/max/sumです。")
        if not isinstance(sources, list) or not sources or any(not isinstance(name, str) or name not in output_names for name in sources):
            raise ValueError("derived_outputsのsourcesは定義済み出力の非空listです。")
        if len(set(sources)) != len(sources):
            raise ValueError("derived_outputsのsourcesを重複させないでください。")
        if any(units[name] != units[target] for name in sources):
            raise ValueError("derived_outputsの出力単位が一致しません。")
        normalized[target] = {"operation": operation, "sources": list(sources)}
    pending, ordered = dict(normalized), {}
    while pending:
        ready = [name for name, spec in pending.items() if not set(spec["sources"]) & set(pending)]
        if not ready:
            raise ValueError("derived_outputsの依存に循環があります。")
        for name in ready:
            ordered[name] = pending.pop(name)
    return ordered


def apply_derived_outputs(mean, std, output_names, definitions):
    """予測点へ恒等式を適用。変換後の期待値・分散を厳密計算しません。"""
    if not definitions:
        return mean, std
    mean, std = mean.copy(), std.copy()
    indices = {name: index for index, name in enumerate(output_names)}
    for target, spec in definitions.items():
        columns = [indices[name] for name in spec["sources"]]
        source_mean = mean[:, columns]
        if spec["operation"] == "max_abs":
            value = np.max(np.abs(source_mean), axis=1)
        elif spec["operation"] == "max":
            value = np.max(source_mean, axis=1)
        else:
            value = np.sum(source_mean, axis=1)
        mean[:, indices[target]] = value
        # 相関・非線形変換の未校正proxy。良品確率や厳密な変換stdではない。
        std[:, indices[target]] = np.sum(std[:, columns], axis=1)
    return mean, std
