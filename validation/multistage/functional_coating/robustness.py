"""Reproducible process-variation scenarios around a nominal line setting.

Randomness lives here, outside the deterministic physical oracle.  A fixed
seed produces an identical scenario matrix, enabling exact regression tests.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .pipeline import LINE_INPUT_BOUNDS, evaluate_line


DEFAULT_STANDARD_DEVIATIONS = {
    "coating_gap_um": 4.0,
    "line_speed_m_min": 0.35,
    "web_tension_n": 3.0,
    "air_temperature_c": 2.0,
    "air_speed_m_s": 0.12,
    "residence_time_min": 0.25,
    "oven_temperature_c": 2.0,
    "hold_time_min": 0.50,
    "nip_pressure_mpa": 0.015,
    "material_viscosity_pa_s": 0.08,
    "material_solids_fraction": 0.010,
    "material_bubble_fraction": 0.0015,
}

STAGE_VARIABLES = {
    "material_and_coating": {
        "coating_gap_um",
        "line_speed_m_min",
        "web_tension_n",
        "material_viscosity_pa_s",
        "material_solids_fraction",
        "material_bubble_fraction",
    },
    "drying": {
        "air_temperature_c",
        "air_speed_m_s",
        "residence_time_min",
    },
    "curing": {
        "oven_temperature_c",
        "hold_time_min",
        "nip_pressure_mpa",
    },
}


def sample_conditions(
    nominal: Mapping[str, float],
    samples: int = 1024,
    seed: int = 0,
    standard_deviations: Mapping[str, float] = DEFAULT_STANDARD_DEVIATIONS,
    frozen_stages: frozenset[str] = frozenset(),
) -> dict[str, np.ndarray]:
    """Draw clipped Gaussian setting/material scenarios around ``nominal``."""

    if set(nominal) != set(LINE_INPUT_BOUNDS):
        raise ValueError("nominal must contain the exact line input contract")
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    if set(standard_deviations) != set(LINE_INPUT_BOUNDS):
        raise ValueError("standard_deviations must cover every line input")
    if not frozen_stages.issubset(STAGE_VARIABLES):
        raise ValueError(f"Unknown frozen stage: {sorted(frozen_stages - STAGE_VARIABLES.keys())}")

    frozen_variables = set().union(
        *(STAGE_VARIABLES[name] for name in frozen_stages)
    ) if frozen_stages else set()
    rng = np.random.default_rng(seed)
    scenarios: dict[str, np.ndarray] = {}
    for name, bounds in LINE_INPUT_BOUNDS.items():
        centre = float(nominal[name])
        sigma = float(standard_deviations[name])
        if not np.isfinite(centre) or not np.isfinite(sigma) or sigma < 0:
            raise ValueError(f"Invalid nominal or standard deviation: {name}")
        if centre < bounds[0] or centre > bounds[1]:
            raise ValueError(f"Nominal {name} is outside {bounds}")
        # Consume the same random stream whether or not a stage is frozen.
        # This common-random-number design makes freeze comparisons attributable
        # to that stage rather than to a different draw in downstream stages.
        standard_normal = rng.normal(0.0, 1.0, size=samples)
        if name in frozen_variables or sigma == 0:
            values = np.full(samples, centre)
        else:
            values = centre + sigma * standard_normal
        scenarios[name] = np.clip(values, bounds[0], bounds[1])
    return scenarios


def estimate_robustness(
    nominal: Mapping[str, float],
    samples: int = 1024,
    seed: int = 0,
    standard_deviations: Mapping[str, float] = DEFAULT_STANDARD_DEVIATIONS,
) -> dict[str, object]:
    """Estimate final yield and stage risk reductions for a nominal setting."""

    scenarios = sample_conditions(
        nominal,
        samples=samples,
        seed=seed,
        standard_deviations=standard_deviations,
    )
    result = evaluate_line(scenarios)
    feasible = np.asarray(result["final"]["feasible"])
    baseline_yield = float(np.mean(feasible))

    risk_reduction: dict[str, float] = {}
    for stage in STAGE_VARIABLES:
        frozen = sample_conditions(
            nominal,
            samples=samples,
            seed=seed,
            standard_deviations=standard_deviations,
            frozen_stages=frozenset({stage}),
        )
        frozen_yield = float(np.mean(evaluate_line(frozen)["final"]["feasible"]))
        risk_reduction[stage] = frozen_yield - baseline_yield

    return {
        "samples": samples,
        "seed": seed,
        "yield_probability": baseline_yield,
        "mean_quality_margin": float(np.mean(result["final"]["quality_margin"])),
        "p05_quality_margin": float(np.quantile(result["final"]["quality_margin"], 0.05)),
        "mean_bond_strength_mpa": float(np.mean(result["final"]["bond_strength_mpa"])),
        "stage_freeze_yield_improvement": risk_reduction,
    }
