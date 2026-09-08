"""Known in-domain drift interventions for future root-cause benchmarks."""

from __future__ import annotations

import numpy as np

from .pipeline import DEFAULT_MATERIAL_STATE, evaluate_line


REFERENCE_NOMINAL = {
    **DEFAULT_MATERIAL_STATE,
    "coating_gap_um": 210.0,
    "line_speed_m_min": 18.0,
    "web_tension_n": 100.0,
    "air_temperature_c": 90.0,
    "air_speed_m_s": 5.5,
    "residence_time_min": 8.0,
    "oven_temperature_c": 140.0,
    "hold_time_min": 60.0,
    "nip_pressure_mpa": 0.325,
}

FAULT_SCENARIOS = {
    "coating_gap_positive_bias": {
        "source_stages": ["coating"],
        "overrides": {"coating_gap_um": 280.0},
        "description": "Gap bias produces an over-thick film that remains outside final thickness specification.",
    },
    "dryer_residence_loss": {
        "source_stages": ["drying"],
        "overrides": {"residence_time_min": 3.0},
        "description": "Lost dryer residence raises skinning/defect and leaves an over-thick solvent-bearing film.",
    },
    "curing_temperature_loss": {
        "source_stages": ["curing"],
        "overrides": {"oven_temperature_c": 105.0},
        "description": "Low cure temperature leaves conversion and bond strength below specification.",
    },
    "distributed_small_drifts": {
        "source_stages": ["coating", "drying", "curing"],
        "overrides": {
            "coating_gap_um": 230.0,
            "air_temperature_c": 80.0,
            "oven_temperature_c": 130.0,
            "hold_time_min": 50.0
        },
        "description": "Individually modest shifts accumulate into an under-cured final product.",
    },
}


def evaluate_fault_scenarios() -> dict[str, object]:
    """Return reference and known-source drift outcomes."""

    cases = {
        "reference": {
            "source_stages": [],
            "description": "Coordinated in-spec reference condition.",
            "conditions": REFERENCE_NOMINAL,
        },
        **{
            name: {
                "source_stages": definition["source_stages"],
                "description": definition["description"],
                "conditions": {**REFERENCE_NOMINAL, **definition["overrides"]},
            }
            for name, definition in FAULT_SCENARIOS.items()
        },
    }
    evaluated: dict[str, object] = {}
    for name, case in cases.items():
        result = evaluate_line(case["conditions"])
        evaluated[name] = {
            **case,
            "final": {
                key: (
                    bool(np.asarray(value))
                    if np.asarray(value).dtype == bool
                    else float(np.asarray(value))
                )
                for key, value in result["final"].items()
            },
        }
    return evaluated
