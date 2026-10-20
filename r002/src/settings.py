"""r002固有のパスと計算予算の既定値。"""

from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parent
TRIALS_ROOT = VERSION_ROOT / "trials"
DEFAULT_SEARCH = {
    "max_evaluations": 4096, "population_size": 64, "seed": 0,
    "max_seconds": 120.0, "support_threshold": 0.5, "batch_size": 64,
    "conservative_std_multiplier": 0.0,
}
