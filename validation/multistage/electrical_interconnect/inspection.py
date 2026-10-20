"""真の加工状態と検査値を分離し、破壊検査ワークの消費を明示します。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .stages import STAGE_MANIFESTS


# 校正前の検査装置を模した合成の1σ。oracle式には加えません。
NOISE_BY_UNIT = {"mm": 0.01, "um": 0.15, "degree": 0.10, "s": 0.01,
                 "fraction": 0.005, "strain": 0.005, "MPa": 2.0,
                 "N": 25.0, "uohm": 0.2, "K": 0.05,
                 "J/mm": 0.03, "kW/mm2": 0.1, "mm2": 0.2,
                 "mJ": 0.5, "category": 0.0, "GPa": 0.0, "nm": 0.3}


def observe_stage(stage_id: str, outputs: Mapping, *, rng: np.random.Generator | None = None,
                  destructive: bool = False, noise: bool = True) -> dict:
    """許可された検査項目だけを返し、潜在状態のoracle値は常に除外します。

    destructive=Trueは切断断面または強度試験を行った別ワークを意味します。
    そのワークは後工程へ流せず、同じ完成ワークの非破壊検査と区別します。
    """
    if stage_id not in STAGE_MANIFESTS:
        raise ValueError(f"Unknown stage: {stage_id}")
    manifest = STAGE_MANIFESTS[stage_id]
    policy = manifest["observation_policy"]
    allowed = [*policy["nondestructive"], *(policy["destructive"] if destructive else [])]
    generator = rng if rng is not None else np.random.default_rng(0)
    measurements, errors = {}, {}
    for name in allowed:
        if name not in outputs:
            raise ValueError(f"Missing inspection output: {name}")
        values = np.asarray(outputs[name], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Inspection input must be finite: {name}")
        sigma = NOISE_BY_UNIT.get(manifest["output_units"][name], 0.0) if noise else 0.0
        measurements[name] = values + generator.normal(0.0, sigma, values.shape)
        errors[name] = sigma
    consumed = bool(destructive and policy["destructive"])
    return {"measurements": measurements, "measurement_std": errors, "consumed": consumed,
            "available_for_next_stage": not consumed, "stage_id": stage_id,
            "latent_unavailable": list(policy["latent"]),
            "inspection_status": "SYNTHETIC_UNCALIBRATED_OBSERVATION_MODEL"}


def inspect_final(outputs: Mapping, *, rng: np.random.Generator | None = None,
                  destructive: bool = False, noise: bool = True) -> dict:
    """冷却・治具解除後の電気／寸法検査と別試片の破壊検査を返します。"""
    return observe_stage("welding", outputs, rng=rng, destructive=destructive, noise=noise)
