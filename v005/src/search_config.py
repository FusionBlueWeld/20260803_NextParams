"""単工程探索の費用設定。全候補の列挙前に読んで適用する。"""
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class SearchConfig:
    candidate_mode: str = "auto"
    max_candidates: int = 4096
    max_training_samples: int = 256
    max_nn_epochs: int = 2000
    max_learning_seconds: float = 30.0
    seed: int = 42
    adaptive_batch: bool = True
    min_recommendation_feasibility: float = 0.05
    min_information_feasibility: float = 0.01
    min_normalized_improvement: float = 0.001
    min_relative_score: float = 0.01
    max_information_candidates: int = 1
    ordinal_constraint_steps: dict[str, float] = field(default_factory=dict)
    cpu_threads: int = 1
    distance_memory_mb: int = 16


def load_search_config(path: Path) -> SearchConfig:
    value = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    if not isinstance(value, dict) or set(value) - set(SearchConfig.__dataclass_fields__):
        raise ValueError("search_settings.jsonに未定義の設定があります。")
    result = SearchConfig(**value)
    if result.candidate_mode not in ("auto", "grid", "budget"):
        raise ValueError("candidate_modeはauto/grid/budgetにしてください。")
    for name, lower, upper in (("max_candidates", 2, 100_000), ("max_training_samples", 2, 512),
                               ("max_nn_epochs", 1, 2000), ("seed", 0, 2**32 - 1),
                               ("max_information_candidates", 0, 100), ("cpu_threads", 1, 64),
                               ("distance_memory_mb", 1, 256)):
        number = getattr(result, name)
        if type(number) is not int or not lower <= number <= upper:
            raise ValueError(f"{name}は{lower}～{upper}の整数にしてください。")
    if isinstance(result.max_learning_seconds,bool) or not isinstance(result.max_learning_seconds,(int,float)) or not math.isfinite(result.max_learning_seconds) or not 0.1 <= result.max_learning_seconds <= 600:
        raise ValueError("max_learning_secondsは0.1～600にしてください。")
    if type(result.adaptive_batch) is not bool:
        raise ValueError("adaptive_batchはtrue/falseにしてください。")
    for name in ("min_recommendation_feasibility", "min_information_feasibility", "min_relative_score"):
        number = getattr(result, name)
        if isinstance(number, bool) or not isinstance(number, (int,float)) or not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"{name}は0～1の有限値にしてください。")
    if not 0 <= result.min_information_feasibility <= result.min_recommendation_feasibility:
        raise ValueError("min_information_feasibilityはmin_recommendation_feasibility以下にしてください。")
    number=result.min_normalized_improvement
    if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("min_normalized_improvementは0～1の有限値にしてください。")
    if not isinstance(result.ordinal_constraint_steps,dict) or any(
        not isinstance(name,str) or isinstance(step,bool) or not isinstance(step,(int,float))
        or not math.isfinite(step) or step<=0 for name,step in result.ordinal_constraint_steps.items()):
        raise ValueError("ordinal_constraint_stepsは制約列名と正の有限刻みの辞書にしてください。")
    return result


def effective_search_config(config,problem,experiments):
    """A declared 'level' unit is ordinal only when raw observations are integers.

    Averaged or continuous scores keep their original continuous threshold.
    Explicit steps take precedence; no discreteness is inferred from numbers alone.
    """
    steps=dict(config.ordinal_constraint_steps)
    for constraint in problem.constraints:
        values=experiments.result_rows[constraint.column]
        if constraint.column not in steps and constraint.unit=='level' and values and all(
            float(v).is_integer() for v in values):steps[constraint.column]=1.0
    return replace(config,ordinal_constraint_steps=steps)
