"""v003全体で共有する設定値とデータ構造。

このファイルには、計算処理そのものではなく、複数の機能から参照する
定数と「データの入れ物」をまとめています。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# フォルダとファイルに関する共通設定
# ---------------------------------------------------------------------------

VERSION_ROOT = Path(__file__).resolve().parents[1]
TRIALS_ROOT = VERSION_ROOT / "trials"

PROBLEM_FILE_NAME = "problem.csv"
EXPERIMENT_FILE_NAME = "experiments.csv"
RECOMMENDATION_FILE_NAME = "recommendations.csv"
RESPONSE_SPACE_DIRECTORY_NAME = "response_spaces"
HYBRID_RESPONSE_FILE_PREFIX = "hybrid_response_space_run_"


# ---------------------------------------------------------------------------
# 探索処理に関する共通設定
# ---------------------------------------------------------------------------

MAX_CANDIDATES = 200_000
# --nを省略した場合の推薦件数です。従来動作との互換性を保つため3件にします。
DEFAULT_RECOMMENDATION_COUNT = 3

# RBFカーネルの長さ尺度です。
# 入力を0～1へそろえた後、この距離を基準に点の近さを判断します。
GP_LENGTH_SCALE = 0.35

# 反復測定からノイズを推定できない場合に使う既定値です。
DEFAULT_NOISE_RATIO = 0.05

# 推薦条件が同じ周辺に集中しすぎないようにする最小距離です。
MIN_RECOMMENDATION_DISTANCE = 0.08

# 2位以降の分散比率は、モデルの不確かさと未被覆距離から毎回自動決定します。
# 退化局面を除き、この範囲内で連続的に変化します。
MIN_DIVERSITY_WEIGHT = 0.15
MAX_DIVERSITY_WEIGHT = 0.50
DIVERSITY_TOP_FRACTION = 0.05
DIVERSITY_COVERAGE_REFERENCE = 0.35

# 1位候補の制約達成確率がこの値を下回った退化局面では、2位以降について
# 達成確率が極端に低い候補を優先対象から外します。
MIN_RECOMMENDATION_FEASIBILITY = 0.01

# 最近傍の実測点からこの距離以上離れている場合、推薦理由に注意を付けます。
FAR_FROM_MEASUREMENTS_DISTANCE = 0.35

# Hybrid内部のNN設定です。少数データ時も学習を止めません。
NN_SEED = 42
NN_HIDDEN_SIZE = 32
NN_MAX_EPOCHS = 2_000
NN_LEARNING_RATE = 0.01
NN_L2_WEIGHT = 1e-4
NN_EARLY_STOPPING_PATIENCE = 200
NN_MIN_IMPROVEMENT = 1e-9

# 学習中は決定論的に間引き、学習後の診断では全候補を検査します。
KNOWLEDGE_TRAINING_MAX_POINTS = 2_048

# GP事後分散の減少量から、入力位置のデータ支持度を計算します。
# 出力変数ごとの測定ノイズとは分離し、全出力で1つの支持度を共有します。
HYBRID_SUPPORT_NOISE = 0.025
HYBRID_SUPPORT_MULTIPLIER = 1.0


# ---------------------------------------------------------------------------
# problem.csvを読み込んだ後のデータ構造
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariableDefinition:
    """problem.csvの1行を表します。"""

    column: str
    display_name: str
    unit: str
    role: str
    direction: str
    lower: float | None
    upper: float | None
    step: float | None
    target: float | None


@dataclass(frozen=True)
class ProblemDefinition:
    """1つのtrialで解く最適化問題を表します。"""

    parameters: list[VariableDefinition]
    objective: VariableDefinition
    constraints: list[VariableDefinition]
    monitors: list[VariableDefinition]
    result_order: list[VariableDefinition]
    candidate_count: int

    @property
    def result_variables(self) -> list[VariableDefinition]:
        """実験後に測定する列を、problem.csvへ書いた順番で返します。"""

        return self.result_order

    @property
    def experiment_header(self) -> list[str]:
        """experiments.csvに必要な列名を返します。"""

        return [
            "experiment_id",
            *[item.column for item in self.parameters],
            *[item.column for item in self.result_variables],
        ]


def trial_path(trial_name: str) -> Path:
    """trial名から、v003内のtrialフォルダを組み立てます。"""

    return TRIALS_ROOT / trial_name
