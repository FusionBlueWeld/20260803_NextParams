"""全直積を作らず、差分進化と工程別・共同提案を評価予算内で比較します。"""

from __future__ import annotations

from collections import OrderedDict
import math
import time

import numpy as np

from .config import qualified, validate_config
from .pipeline import evaluate_pipeline


def latin_hypercube(count: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """NumPyだけで層化された初期条件を作ります。"""
    result = np.empty((count, dimensions))
    for index in range(dimensions):
        result[:, index] = (rng.permutation(count) + rng.random(count)) / count
    return result


class CandidateSpace:
    """修飾名、離散水準、連続操作、実施済み固定操作を対応付けます。"""

    def __init__(self, config: dict):
        self.names = list(config["candidate_axes"])
        self.fixed = {name: float(value) for name, value in config.get("fixed_controls", {}).items()}
        self.axes = {name: np.unique(np.asarray(axis, dtype=float)) for name, axis in config["candidate_axes"].items()}
        self.continuous = set(config.get("continuous_controls", []))
        for name, axis in self.axes.items():
            if len(axis) == 1 and name not in self.fixed:
                self.fixed[name] = float(axis[0])
        self.free = [name for name in self.names if name not in self.fixed]
        self.stage_dimensions = {}
        for index, name in enumerate(self.free):
            self.stage_dimensions.setdefault(qualified(name)[0], []).append(index)
        self.candidate_count = None if self.continuous & set(self.free) else math.prod(len(self.axes[name]) for name in self.free)

    def decode(self, normalized: np.ndarray) -> dict:
        """探索座標を実操作へ戻します。流入状態は探索軸に含めません。"""
        count = len(normalized)
        values = {name: np.full(count, value) for name, value in self.fixed.items()}
        for index, name in enumerate(self.free):
            axis = self.axes[name]
            coordinate = np.clip(normalized[:, index], 0.0, 1.0)
            if name in self.continuous:
                values[name] = axis[0] + coordinate * (axis[-1] - axis[0])
            else:
                level = np.minimum((coordinate * len(axis)).astype(int), len(axis) - 1)
                values[name] = axis[level]
        return {name: values[name] for name in self.names}

    def encode(self, controls: dict) -> np.ndarray:
        """保存済み候補を局所・共同提案の起点へ戻します。"""
        coordinates = []
        for name in self.free:
            axis, value = self.axes[name], float(controls[name])
            if name in self.continuous:
                coordinates.append((value - axis[0]) / max(axis[-1] - axis[0], 1e-12))
            else:
                index = int(np.argmin(np.abs(axis - value)))
                coordinates.append((index + 0.5) / len(axis))
        return np.asarray(coordinates)


def run_budgeted_search(config: dict, predictors: dict, *, recommendation_count: int = 8) -> dict:
    """限られた予測回数・時間で全工程条件を選び、未発見を不可能証明と区別します。"""
    plan = validate_config(config, predictors)
    settings = plan["search"]
    maximum = int(settings["max_evaluations"])
    population_size = min(int(settings["population_size"]), maximum)
    batch_size = int(settings["batch_size"])
    rng = np.random.default_rng(int(settings["seed"]))
    space = CandidateSpace(config)
    started = time.monotonic()
    evaluations, cache_hits, model_prediction_rows = 0, 0, 0
    cache = OrderedDict()
    cache_limit = max(1, min(maximum, int(settings.get("cache_size", 4096))))
    archive = {}
    history = []
    stop_reason = "EVALUATION_BUDGET"

    def expired() -> bool:
        return time.monotonic() - started >= float(settings["max_seconds"])

    def assess(normalized: np.ndarray) -> tuple[np.ndarray, list]:
        """未評価点だけを小さいバッチで予測し、計算予算を厳密に数えます。"""
        nonlocal evaluations, cache_hits, model_prediction_rows
        values = space.decode(normalized)
        keys = [tuple(float(values[name][row]) for name in space.names) for row in range(len(normalized))]
        scores = np.full(len(keys), -np.inf)
        records = [None] * len(keys)
        waiting = {}
        for row, key in enumerate(keys):
            if key in cache:
                records[row] = cache[key]
                scores[row] = float(cache[key]["trusted_margin"])
                cache_hits += 1
            else:
                waiting.setdefault(key, []).append(row)
        unique = list(waiting)
        for start in range(0, len(unique), batch_size):
            if expired() or evaluations >= maximum:
                break
            selected = unique[start:start + min(batch_size, maximum - evaluations)]
            batch = {name: np.asarray([key[column] for key in selected]) for column, name in enumerate(space.names)}
            result = evaluate_pipeline(config, predictors, batch, validated=plan)
            evaluations += len(selected)
            model_prediction_rows += result["model_prediction_rows"]
            for row, key in enumerate(selected):
                feasible = bool(result["trusted_feasible"][row])
                margin = float(result["trusted_margin"][row])
                record = {
                    "controls": dict(zip(space.names, key)), "predicted_feasible": feasible,
                    "trusted_margin": margin,
                    "process_margin": float(np.nan_to_num(result["process_margin"][row], nan=-1e6)),
                    "conservative_process_margin": float(np.nan_to_num(result["conservative_process_margin"][row], nan=-1e6, neginf=-1e6, posinf=1e6)),
                    "minimum_stage_support": float(np.nan_to_num(result["minimum_support"][row], nan=0.0, posinf=0.0, neginf=0.0)),
                    "bottleneck": str(result["bottleneck"][row]),
                    "valid_connection": bool(result["valid_connection"][row]),
                }
                cache[key] = record
                cache.move_to_end(key)
                if len(cache) > cache_limit:
                    cache.popitem(last=False)
                archive[key] = record
                for destination in waiting[key]:
                    records[destination], scores[destination] = record, margin
            if len(archive) > max(population_size * 2, recommendation_count * 4, 128):
                kept = sorted(archive, key=lambda key: archive[key]["trusted_margin"], reverse=True)[:max(population_size, recommendation_count * 4, 64)]
                archive_keys = set(kept)
                for key in list(archive):
                    if key not in archive_keys:
                        del archive[key]
            best = max(archive.values(), key=lambda item: item["trusted_margin"])
            history.append({"evaluations": evaluations, "best_trusted_margin": best["trusted_margin"], "feasible_found": any(item["predicted_feasible"] for item in archive.values()), "elapsed_seconds": time.monotonic() - started})
        return scores, records

    dimensions = len(space.free)
    population = latin_hypercube(population_size, dimensions, rng)
    if dimensions:
        population[0] = 0.5
        if population_size > 1:
            population[1] = 0.0
        if population_size > 2:
            population[2] = 1.0
    for index, warm in enumerate(config.get("warm_start", [])):
        if index >= population_size:
            break
        if set(warm) != set(space.names):
            raise ValueError("warm_startは全操作値を指定してください。")
        population[index] = space.encode(warm)
    scores, _ = assess(population)
    stagnant = 0
    generations = 0
    while dimensions and evaluations < maximum and not expired():
        if space.candidate_count is not None and space.candidate_count <= cache_limit and len(cache) == space.candidate_count:
            stop_reason = "FINITE_SPACE_EXHAUSTED"
            break
        generations += 1
        old_evaluations = evaluations
        proposals = np.array(population, copy=True)
        best_position = int(np.argmax(scores))
        for row in range(population_size):
            if population_size >= 4:
                pool = np.asarray([index for index in range(population_size) if index != row])
                a, b, c = rng.choice(pool, 3, replace=False)
                donor = population[a] + rng.uniform(0.4, 0.9) * (population[b] - population[c])
                mask = rng.random(dimensions) < 0.7
                mask[rng.integers(dimensions)] = True
                proposals[row] = np.where(mask, donor, population[row])
            else:
                proposals[row] = rng.random(dimensions)
            if row % 5 == 0:
                proposals[row] = rng.random(dimensions)
            elif row % 5 == 1:
                # 工程別提案でも時折全工程を同時に動かし、補償関係を探索します。
                proposals[row] = population[best_position]
                groups = list(space.stage_dimensions.values())
                group = groups[int(rng.integers(len(groups)))] if rng.random() < 0.7 else list(range(dimensions))
                proposals[row, group] += rng.normal(0.0, 0.18, len(group))
            elif row % 5 == 2:
                proposals[row] = population[best_position]
                dimension = int(rng.integers(dimensions))
                name = space.free[dimension]
                step = 0.06 if name in space.continuous else 1.0 / len(space.axes[name])
                proposals[row, dimension] += rng.choice([-1.0, 1.0]) * step
        proposals = np.clip(proposals, 0.0, 1.0)
        proposal_scores, _ = assess(proposals)
        improve = proposal_scores >= scores
        population[improve], scores[improve] = proposals[improve], proposal_scores[improve]
        stagnant = stagnant + 1 if evaluations == old_evaluations else 0
        if stagnant >= 20:
            stop_reason = "NO_NEW_CANDIDATES"
            break
    if expired():
        stop_reason = "TIME_BUDGET"
    if not dimensions and evaluations:
        stop_reason = "FIXED_CONTROLS_ONLY"
    candidates = sorted(archive.values(), key=lambda item: item["trusted_margin"], reverse=True)[:max(1, recommendation_count)]
    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank
    best = candidates[0] if candidates else None
    status = "FEASIBLE_CANDIDATE_FOUND" if best and best["predicted_feasible"] else "NO_TRUSTED_CANDIDATE_FOUND_WITHIN_BUDGET"
    return {
        "schema_version": "2.0", "status": status, "best": best, "selected": best,
        "candidates": candidates, "history": history, "evaluation_count": evaluations,
        "cache_hits": cache_hits, "generations": generations, "stop_reason": stop_reason,
        "model_prediction_rows": model_prediction_rows,
        "elapsed_seconds": time.monotonic() - started,
        "full_grid_candidate_count": space.candidate_count,
        "free_control_count": dimensions, "completed_stages": sorted(plan["completed_stages"]),
        "search_budget": settings, "probability_status": "NOT_EVALUATED",
        "uncertainty_policy": "uncalibrated_heuristic" if settings.get("conservative_std_multiplier", 0.0) else "mean_prediction_only",
        "claim": "budgeted_search_not_global_optimum_or_infeasibility_proof",
        "time_limit_kind": "checked_between_prediction_batches",
    }
