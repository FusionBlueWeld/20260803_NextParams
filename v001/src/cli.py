

"""ParamOptimizer v001の内部CLI実装。

このファイルには、処理の詳しい計算式を書かず、利用者の操作ごとに
「何をどの順番で実行するか」をまとめています。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from src.data_loader import read_csv, write_csv_atomic, write_problem_template
from src.settings import (
    DEFAULT_RECOMMENDATION_COUNT,
    EXPERIMENT_FILE_NAME,
    PROBLEM_FILE_NAME,
    RECOMMENDATION_FILE_NAME,
    TRIALS_ROOT,
    ProblemDefinition,
    trial_path,
)
from src.validation import (
    UserInputError,
    load_and_validate_experiments,
    load_and_validate_problem,
    validate_trial_name,
)


# ---------------------------------------------------------------------------
# コマンドラインオプションを読み取る機能
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """--new、--prepare、--runのいずれか1つを受け取ります。"""

    parser = argparse.ArgumentParser(
        description="実験データから、次に試すパラメータ条件を推薦します。"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--new", metavar="TRIAL", help="新しいtrialを作成します")
    actions.add_argument(
        "--prepare",
        metavar="TRIAL",
        help="problem.csvを確認し、experiments.csvを準備します",
    )
    actions.add_argument("--run", metavar="TRIAL", help="指定trialの探索を実行します")
    parser.add_argument(
        "--n",
        metavar="COUNT",
        type=positive_integer,
        help=(
            "--runで出力する次実験候補の件数を指定します"
            f"（省略時: {DEFAULT_RECOMMENDATION_COUNT}）"
        ),
    )
    arguments = parser.parse_args()
    if arguments.n is not None and arguments.run is None:
        parser.error("--nは--runと一緒に指定してください")
    return arguments


def positive_integer(value: str) -> int:
    """CLIで1以上の整数だけを受け付けます。"""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return parsed


# ---------------------------------------------------------------------------
# --new：新しいtrialを作る処理
# ---------------------------------------------------------------------------


def create_trial(trial_name: str) -> None:
    """trialフォルダ、problem.csv、data/outputフォルダを作ります。"""

    validate_trial_name(trial_name)
    TRIALS_ROOT.mkdir(parents=True, exist_ok=True)
    destination = trial_path(trial_name)

    if destination.exists():
        raise UserInputError(
            f"{trial_name}は既に存在します。\n"
            "既存ファイルは変更していません。別のtrial名を指定してください。"
        )

    # 完成途中のtrialを残さないため、一時フォルダで作ってから移動します。
    temporary = Path(tempfile.mkdtemp(prefix=f".{trial_name}_", dir=TRIALS_ROOT))
    try:
        (temporary / "data").mkdir()
        (temporary / "output").mkdir()
        write_problem_template(temporary / PROBLEM_FILE_NAME)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print(f"{trial_name}を作成しました。")
    print("\n次のファイルを編集してください。")
    print(destination / PROBLEM_FILE_NAME)
    print("\n編集後、次のコマンドを実行してください。")
    print(f"python main.py --prepare {trial_name}")


# ---------------------------------------------------------------------------
# --prepare：problem.csvから実験CSVの列を準備する処理
# ---------------------------------------------------------------------------


def prepare_trial(trial_name: str) -> None:
    """problem.csvを検査し、experiments.csvのヘッダーを作ります。"""

    validate_trial_name(trial_name)
    current_trial = require_trial(trial_name)
    problem = load_and_validate_problem(current_trial / PROBLEM_FILE_NAME)

    experiment_path = current_trial / "data" / EXPERIMENT_FILE_NAME
    expected_header = problem.experiment_header

    if experiment_path.exists():
        current_header, _ = read_csv(experiment_path)
        if current_header != expected_header:
            raise UserInputError(
                "既存のexperiments.csvとproblem.csvの列が一致しません。\n"
                f"必要: {','.join(expected_header)}\n"
                f"現在: {','.join(current_header)}\n"
                "実験データを確認し、列を手動で合わせてください。"
            )
        print("experiments.csvは既に準備されています。既存ファイルは変更していません。")
    else:
        write_csv_atomic(experiment_path, expected_header, [])
        print("problem.csvを確認しました。")
        print(f"\n実験結果の入力ファイルを作成しました。\n{experiment_path}")

    print_problem_summary(problem)
    print("\n実験結果を入力後、次のコマンドを実行してください。")
    print(f"python main.py --run {trial_name}")


# ---------------------------------------------------------------------------
# --run：実験データから推薦条件を計算する処理
# ---------------------------------------------------------------------------


def run_trial(trial_name: str, recommendation_count: int) -> None:
    """入力検査、学習、探索、CSV出力を順番に実行します。"""

    validate_trial_name(trial_name)
    current_trial = require_trial(trial_name)
    problem = load_and_validate_problem(current_trial / PROBLEM_FILE_NAME)
    experiments = load_and_validate_experiments(
        current_trial / "data" / EXPERIMENT_FILE_NAME,
        problem,
    )

    # 数値計算ライブラリは探索実行時にだけ必要です。
    try:
        from src.gp.model import ModelTrainingError
        from src.gp.optimizer import run_optimization
        from src.gp.reporting import print_run_result, write_recommendations
        from src.nn import NeuralTrainingError, train_network
        from src.nn.predictor import predict_response_space
        from src.nn.reporting import (
            next_run_id,
            response_space_path,
            write_response_space,
        )
        from src.parameter_space import make_parameter_grid
        from src.preprocessing import prepare_experiments
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise UserInputError(
                "NumPyがインストールされていません。\n"
                "次を実行してから、もう一度--runを実行してください。\n"
                "python -m pip install -r requirements.txt"
            ) from error
        raise

    prepared = prepare_experiments(experiments, problem)
    try:
        result = run_optimization(
            experiments,
            prepared,
            problem,
            recommendation_count=recommendation_count,
        )
    except ModelTrainingError as error:
        raise UserInputError(str(error)) from error

    output_path = current_trial / "output" / RECOMMENDATION_FILE_NAME
    write_recommendations(output_path, experiments, problem, result)
    print_run_result(trial_name, experiments, problem, result, output_path)

    # NNはGPとは独立に、同じデータと同じグリッドから予測空間を作ります。
    # 少数データ時の不自然な予測も観察対象なので、件数による停止はしません。
    try:
        raw_grid = make_parameter_grid(problem)
        training = train_network(prepared, problem)
        predictions = predict_response_space(training.model, raw_grid, problem)
        output_directory = current_trial / "output"
        run_id = next_run_id(output_directory)
        nn_output_path = response_space_path(output_directory, run_id)
        write_response_space(
            nn_output_path,
            run_id,
            experiments,
            prepared,
            problem,
            raw_grid,
            predictions,
        )
    except NeuralTrainingError as error:
        print(f"\n[NN注意] NN予測空間を生成できませんでした。\n{error}")
        print("GPのrecommendations.csvは正常に更新されています。")
        return

    print("\nNNによる予測空間:")
    print(f"  学習データ: {experiments.row_count}行")
    print(f"  固有条件: {prepared.unique_condition_count}条件")
    print(f"  学習エポック: {training.epochs}")
    print(f"  標準化MSE: {training.normalized_mse:.6g}")
    print(f"  予測グリッド: {len(raw_grid):,}点")
    print(f"  run: {run_id}")
    print(f"  出力: {nn_output_path}")
    print(
        "\n[注意] NN予測は現在のデータから形成された予測空間です。"
        "少数データ時には不自然または不安定な形状になることがあります。"
    )


# ---------------------------------------------------------------------------
# 複数コマンドから使う補助処理
# ---------------------------------------------------------------------------


def require_trial(trial_name: str) -> Path:
    """指定trialがなければ、作成方法を案内します。"""

    current_trial = trial_path(trial_name)
    if not current_trial.is_dir():
        raise UserInputError(
            f"{trial_name}が見つかりません。\n"
            f"新しく作る場合: python main.py --new {trial_name}"
        )
    return current_trial


def print_problem_summary(problem: ProblemDefinition) -> None:
    """prepare完了時に、読み取った問題設定を確認表示します。"""

    parameters = problem.parameters
    objective = problem.objective
    constraints = problem.constraints

    print("\n入力パラメータ:")
    for parameter in parameters:
        print(f"- {parameter.display_name} ({parameter.column})")

    objective_action = "最小化" if objective.direction == "minimize" else "最大化"
    print(f"\n目的:\n- {objective.display_name}を{objective_action}")

    print("\n制約:")
    if not constraints:
        print("- なし")
    for constraint in constraints:
        symbol = ">=" if constraint.direction == "greater_equal" else "<="
        print(f"- {constraint.display_name} {symbol} {constraint.target:g}")

    print(f"\n探索候補数: {problem.candidate_count:,}件")


# ---------------------------------------------------------------------------
# プログラム全体の入口
# ---------------------------------------------------------------------------


def main() -> int:
    """選択されたオプションに対応する処理を1つだけ実行します。"""

    arguments = parse_arguments()
    action_name = ""

    try:
        if arguments.new:
            action_name = "--new"
            create_trial(arguments.new)
        elif arguments.prepare:
            action_name = "--prepare"
            prepare_trial(arguments.prepare)
        else:
            action_name = "--run"
            recommendation_count = arguments.n or DEFAULT_RECOMMENDATION_COUNT
            run_trial(arguments.run, recommendation_count)
        return 0
    except UserInputError as error:
        print(f"[エラー]\n{error}", file=sys.stderr)
        if action_name == "--run":
            print(
                "\nrecommendations.csvは更新されていません。\n"
                "既存ファイルがある場合は、以前のデータに基づく結果です。",
                file=sys.stderr,
            )
        return 1
    except PermissionError as error:
        print(
            "[エラー]\nファイルを開けませんでした。Excelで開いている場合は閉じてください。\n"
            f"対象: {error.filename or '不明'}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



