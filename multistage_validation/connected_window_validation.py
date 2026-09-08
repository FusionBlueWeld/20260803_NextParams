"""Compare r001 connected-window boundaries with the deterministic physics oracle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from multistage_validation.functional_coating.pipeline import evaluate_line
from r001.src.cli import _contiguous_true_interval


ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "r001/trials/trial_validated_individual_chain_r001"
REPORT = ROOT / "multistage_validation/results/connected_window_validation.json"


def _scalar(value: object) -> float:
    return float(np.asarray(value))


def _oracle_process_feasible(conditions: dict[str, float]) -> tuple[bool, dict[str, float | bool | list[str]]]:
    try:
        result = evaluate_line(conditions)
    except ValueError:
        return False, {"valid_connection": False, "failed": ["connection_range"]}
    coating, drying, curing, final = result["coating"], result["drying"], result["curing"], result["final"]
    checks = {
        "local.coating.thickness_cv_fraction": _scalar(coating["thickness_cv_fraction"]) <= 0.08,
        "local.coating.coating_defect_index": _scalar(coating["coating_defect_index"]) <= 0.25,
        "local.drying.residual_solvent_pct": _scalar(drying["residual_solvent_pct"]) <= 5.0,
        "local.drying.drying_defect_index": _scalar(drying["drying_defect_index"]) <= 0.30,
        "local.curing.cure_fraction": _scalar(curing["cure_fraction"]) >= 0.80,
        "local.curing.degradation_fraction": _scalar(curing["degradation_fraction"]) <= 0.20,
        "local.curing.final_residual_solvent_pct": _scalar(curing["final_residual_solvent_pct"]) <= 2.0,
        "local.curing.final_defect_index": _scalar(curing["final_defect_index"]) <= 0.25,
        "final": bool(final["feasible"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, {
        "valid_connection": True, "failed": failed,
        "quality_margin": _scalar(final["quality_margin"]),
    }


def _acceptance(center_ok, rows, confusion, empty_windows, unsafe_overshoots, scan_points, expected_axes,
                center_predicted_feasible=True):
    """Benchmark gate: reject false allowances in any disconnected interval."""
    recoveries = [row["true_window_recovery_grid_fraction"] for row in rows.values()]
    mean_recovery = float(np.mean([0.0 if value is None else value for value in recoveries])) if recoveries else 0.0
    checks = {
        "center_oracle_feasible": bool(center_ok),
        "center_predicted_feasible": bool(center_predicted_feasible),
        "all_axes_evaluated": set(rows) == set(expected_axes),
        "no_false_allowances": confusion["false_positive"] == 0,
        "no_empty_windows": empty_windows == 0,
        "mean_recovery_at_least_0_60": mean_recovery >= 0.60,
        "boundary_overshoot_within_grid_step": bool(unsafe_overshoots) and max(unsafe_overshoots) <= 1 / (scan_points - 1),
    }
    return {"checks": checks, "mean_true_window_recovery_grid_fraction": mean_recovery,
            "status": "pass" if all(checks.values()) else "fail"}


def run() -> dict[str, object]:
    learned = json.loads((TRIAL / "output/connected_window.json").read_text(encoding="utf-8"))
    pipeline = json.loads((TRIAL / "pipeline.json").read_text(encoding="utf-8"))
    center = {**{
        "material_viscosity_pa_s": pipeline["external_context"]["coating.incoming_viscosity_pa_s"],
        "material_solids_fraction": pipeline["external_context"]["coating.incoming_solids_fraction"],
        "material_bubble_fraction": pipeline["external_context"]["coating.incoming_bubble_fraction"],
    }, **learned["center"]}
    center_ok, center_detail = _oracle_process_feasible(center)
    rows = {}
    errors = []
    unsafe_overshoots = []
    coverage_ratios = []
    total_confusion = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    empty_windows = 0
    unavailable = 0
    shrinkage_rows = []
    scan_points = int(pipeline["connected_window"]["scan_points"])
    for name, profile in learned["controls"].items():
        lower, upper = profile["model_input_bounds"]
        axis = np.unique(np.append(np.linspace(lower, upper, scan_points), center[name]))
        oracle_mask = np.zeros(len(axis), dtype=bool)
        for index, value in enumerate(axis):
            oracle_mask[index] = _oracle_process_feasible({**center, name: float(value)})[0]
        oracle_low, oracle_high = _contiguous_true_interval(axis, oracle_mask, center[name])
        learned_low, learned_high = profile["process_window"]
        learned_intervals = profile.get("buffered_all_intervals", [profile["process_window"]])
        trusted_intervals = profile.get("trusted_all_intervals", [profile["trusted_window"]])
        mean_intervals = profile.get("mean_only_all_intervals", [profile.get("mean_only_window")])
        def interval_mask(intervals):
            mask = np.zeros(len(axis), dtype=bool)
            for interval in intervals:
                if interval and interval[0] is not None:
                    mask |= (axis >= interval[0]) & (axis <= interval[1])
            return mask
        learned_mask = interval_mask(learned_intervals)
        trusted_mask = interval_mask(trusted_intervals)
        mean_mask = interval_mask(mean_intervals)
        confusion = {
            "true_positive": int(np.sum(trusted_mask & oracle_mask)),
            "false_positive": int(np.sum(trusted_mask & ~oracle_mask)),
            "false_negative": int(np.sum(~trusted_mask & oracle_mask)),
            "true_negative": int(np.sum(~trusted_mask & ~oracle_mask)),
        }
        for key in total_confusion:
            total_confusion[key] += confusion[key]
        empty_windows += int(oracle_low is not None and profile["trusted_window"][0] is None)
        unavailable += int(profile.get("evaluation_unavailable_count", 0))
        oracle_count = int(oracle_mask.sum())
        learned_count = int(trusted_mask.sum())
        recovery = confusion["true_positive"] / oracle_count if oracle_count else None
        false_allowance = confusion["false_positive"] / learned_count if learned_count else 0.0
        shrinkage_rows.append({
            "control": name, "mean_only_grid_fraction": float(mean_mask.mean()),
            "buffered_grid_fraction": float(learned_mask.mean()), "trusted_grid_fraction": float(trusted_mask.mean()),
            "buffer_loss_fraction": float((mean_mask.sum() - learned_mask.sum()) / max(mean_mask.sum(), 1)),
            "support_loss_fraction": float((learned_mask.sum() - trusted_mask.sum()) / max(learned_mask.sum(), 1)),
        })
        span = upper - lower
        low_error = None if learned_low is None or oracle_low is None else (learned_low - oracle_low) / span
        high_error = None if learned_high is None or oracle_high is None else (learned_high - oracle_high) / span
        if low_error is not None:
            errors.extend([abs(low_error), abs(high_error)])
            unsafe_overshoots.extend([max(0.0, -low_error), max(0.0, high_error)])
            oracle_width = oracle_high - oracle_low
            coverage_ratios.append((learned_high - learned_low) / oracle_width if oracle_width > 0 else 1.0)
        rows[name] = {
            "oracle_process_window": [oracle_low, oracle_high],
            "learned_process_window": [learned_low, learned_high],
            "learned_trusted_window": profile["trusted_window"],
            "normalized_boundary_error": {"lower": low_error, "upper": high_error},
            "unsafe_normalized_overshoot": {
                "lower": None if low_error is None else max(0.0, -low_error),
                "upper": None if high_error is None else max(0.0, high_error),
            },
            "oracle_window_coverage_fraction": coverage_ratios[-1] if low_error is not None else None,
            "true_window_recovery_grid_fraction": recovery,
            "false_allowance_grid_fraction": false_allowance,
            "classification": confusion,
            "oracle_feasible_grid_fraction": float(oracle_mask.mean()),
        }
    acceptance = _acceptance(center_ok, rows, total_confusion, empty_windows,
                             unsafe_overshoots, scan_points, pipeline["candidate_axes"],
                             np.isfinite(learned["center_trusted_margin"]) and learned["center_trusted_margin"] >= 0)
    report = {
        "status": acceptance["status"],
        "acceptance": acceptance,
        "scope": "Deterministic fully observed oracle; no real-world probability calibration is used.",
        "center": learned["center"], "center_oracle_feasible": center_ok,
        "center_oracle_detail": center_detail,
        "learned_center": {
            "process_margin": learned["center_process_margin"],
            "trusted_margin": learned["center_trusted_margin"],
            "minimum_support": learned["center_minimum_support"],
            "bottleneck": learned["center_bottleneck"],
        },
        "maximum_absolute_normalized_boundary_error": max(errors) if errors else None,
        "mean_absolute_normalized_boundary_error": float(np.mean(errors)) if errors else None,
        "maximum_unsafe_normalized_overshoot": max(unsafe_overshoots) if unsafe_overshoots else None,
        "mean_oracle_window_coverage_fraction": float(np.mean(coverage_ratios)) if coverage_ratios else None,
        "aggregate_grid_classification": total_confusion,
        "empty_window_count": empty_windows,
        "evaluation_unavailable_count": unavailable,
        "shrinkage_decomposition": shrinkage_rows,
        "axis_profiles": rows,
        "limitation": "Intervals vary one control at a time. They are cross-sections of the 9-D window, not an axis-independent box.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=True, indent=2))
