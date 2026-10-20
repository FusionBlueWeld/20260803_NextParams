"""単一BLASスレッドで距離計算の数値・時間・追加メモリを独立測定します。

旧3次元broadcastの大きな参考計算は2,048候補で一度だけ実行します。
既存trialは読み取りのみ。tracemallocはPython/NumPyの追加確保量であり、
OSのプロセス全体RSSとは異なります。
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import tracemalloc
from types import SimpleNamespace

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v005.src.candidate_identity import CandidateIdentity
from v005.src.connection import StageConnection, current_context_observation_mask
from v005.src.hybrid.optimizer import constraint_probability_target, nearest_measurement_distance
from v005.src.hybrid.reporting import csv_value
from v005.src.search_config import SearchConfig, effective_search_config
from v005.src.settings import ProblemDefinition, VariableDefinition

RESULT_PATH = Path(__file__).resolve().parent / "results" / "resource_validation.json"


def measure(call):
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    result = call()
    seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {"seconds": seconds, "peak_additional_mib": peak / 2**20}


def broadcast_reference(candidates, measured):
    difference = candidates[:, None, :] - measured[None, :, :]
    return np.min(np.linalg.norm(difference, axis=2), axis=1)


def variable(name, role="parameter", unit="", target=None, direction=""):
    return VariableDefinition(name, name, unit, role, direction,
                              0 if role == "parameter" else None,
                              1 if role == "parameter" else None,
                              1 / 3 if role == "parameter" else None, target)


def design_checks():
    default = SearchConfig()
    cases = [
        (variable("integer", "constraint", "level", 4.0, "less_equal"), default, 4.5),
        (variable("integer", "constraint", "level", 4.0, "greater_equal"), default, 3.5),
        (variable("integer", "constraint", "level", 3.4, "less_equal"), default, 3.5),
        (variable("integer", "constraint", "level", 3.4, "greater_equal"), default, 3.5),
        (variable("fraction", "constraint", "fraction", 0.34, "less_equal"),
         SearchConfig(ordinal_constraint_steps={"fraction": 0.25}), 0.375),
        (variable("fraction", "constraint", "fraction", 0.34, "greater_equal"),
         SearchConfig(ordinal_constraint_steps={"fraction": 0.25}), 0.375),
        (variable("continuous", "constraint", "mm", 4.0, "less_equal"), default, 4.0),
    ]
    for constraint, config, expected in cases:
        config=effective_search_config(config,SimpleNamespace(constraints=[constraint]),
            SimpleNamespace(result_rows={constraint.column:[3.,4.]}))
        assert constraint_probability_target(constraint, config) == expected

    parameters = [variable("control"), variable("incoming")]
    objective = variable("y", "objective", direction="minimize")
    problem = ProblemDefinition(parameters, objective, [], [], [objective], 16)
    context = {"incoming": 1 / 3}
    connection = StageConnection("test", ("control",), ("incoming",), (), context)
    rows = np.asarray([[1 / 3, 1 / 3], [1 / 3, 0.3333333333], [1 / 3, np.nextafter(1 / 3, 1.0)]])
    np.testing.assert_array_equal(current_context_observation_mask(rows, problem, connection), [True, False, False])
    assert len(set(CandidateIdentity(problem, context).keys(rows))) == 3

    rng = np.random.default_rng(77)
    values = np.concatenate((rng.uniform(-1e5, 1e5, 2000), [1 / 3, 1e-200, 1e200, np.nextafter(1.0, 2.0)]))
    for value in values:
        assert float(csv_value(float(value))) == value
    return {"ordinal_boundary_cases_passed": len(cases), "strict_context_cases_passed": len(rows),
            "csv_roundtrip_values_passed": len(values),
            "design_assumptions": [
                "unit=level with integer raw observations means an integer ordinal with origin 0",
                "explicit fractional ordinal steps have origin 0",
                "continuity correction applies to predicted probability; actual feasibility retains the original limit",
                "old rounded incoming contexts are not silently normalized to a different true incoming state",
            ],
            "review_notes": [
                "An ordinal origin other than 0 requires additional metadata before enabling this correction.",
                "Noninteger raw level observations keep the continuous boundary unless an explicit ordinal step is configured.",
                "Unknown ordinal constraint column names are rejected by trial input validation.",
                "17 significant decimal digits preserve finite binary64 input values when parsed back to float.",
            ]}


def main():
    rng = np.random.default_rng(20261010)
    measured = rng.random((124, 50))
    candidates = rng.random((65536, 50))
    small = candidates[:2048]
    reference, old_metrics = measure(lambda: broadcast_reference(small, measured))
    improved, new_metrics = measure(lambda: nearest_measurement_distance(small, measured, memory_mb=2))
    full, full_metrics = measure(lambda: nearest_measurement_distance(candidates, measured, memory_mb=2))
    np.testing.assert_allclose(improved, reference, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(full))
    np.testing.assert_array_equal(full[:2048], improved)

    # Exact and almost-identical rows expose cancellation in a dot-product distance.
    edge = measured[:4].copy()
    edge[2:] = np.nextafter(edge[2:], 1.0)
    edge_reference = np.min(np.linalg.norm(edge[:, None, :] - measured[None, :, :], axis=2), axis=1)
    edge_actual = nearest_measurement_distance(edge, measured, memory_mb=2)

    result = {"schema_version": "1.0", "python": platform.python_version(), "numpy": np.__version__,
              "thread_environment": {name: os.environ[name] for name in
                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")},
              "seed": 20261010,
              "memory_measurement": "tracemalloc additional Python/NumPy allocations, excluding pre-created inputs and OS RSS",
              "distance_small": {"candidate_shape": list(small.shape), "measured_shape": list(measured.shape),
                                 "old_broadcast_once": old_metrics, "new_memory_mb_2": new_metrics,
                                 "max_absolute_error": float(np.max(np.abs(improved - reference))),
                                 "peak_memory_reduction_ratio": old_metrics["peak_additional_mib"] / new_metrics["peak_additional_mib"],
                                 "speedup_ratio": old_metrics["seconds"] / new_metrics["seconds"]},
              "distance_full": {"candidate_shape": list(candidates.shape), "measured_shape": list(measured.shape),
                                "memory_mb": 2, **full_metrics, "finite": True},
              "near_identical_diagnostic": {"reference": edge_reference.tolist(), "new": edge_actual.tolist(),
                                             "max_absolute_error": float(np.max(np.abs(edge_actual - edge_reference)))},
              "design_checks": design_checks(),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                ("v005/src/hybrid/optimizer.py", "v005/src/connection.py",
                                 "v005/src/hybrid/reporting.py", "v005/src/candidate_identity.py",
                                 "v005/src/search_config.py", "v005/src/trial_inputs.py")}}
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"result_path": str(RESULT_PATH), "distance_small": result["distance_small"],
                      "distance_full": result["distance_full"],
                      "near_identical_diagnostic": result["near_identical_diagnostic"],
                      "design_checks": result["design_checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
