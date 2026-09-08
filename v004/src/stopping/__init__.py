"""実験ループの停止判定API。"""

from .criteria import (
    StopConfig,
    StopDecision,
    StopStatus,
    evaluate_stop,
    normalized_condition_distance,
    support_coverage,
)
from .config import load_stop_config, write_stop_settings_template
from .reporting import (
    build_stop_context, convergence_history, load_stop_history, write_stop_status,
)
from .metrics import objective_target_reached, predicted_optimum_index

__all__ = [
    "StopConfig", "StopDecision", "StopStatus", "evaluate_stop",
    "normalized_condition_distance", "support_coverage",
    "load_stop_config", "write_stop_settings_template",
    "load_stop_history", "write_stop_status",
    "build_stop_context", "convergence_history",
    "objective_target_reached", "predicted_optimum_index",
]
