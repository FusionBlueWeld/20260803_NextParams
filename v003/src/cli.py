


"""ParamOptimizer v003の内部CLI実装。

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
from src.knowledge import (
    build_knowledge_points,
    evaluate_rules_on_hybrid,
    evaluate_rules_on_model,
    evaluate_rules_on_observations,
    load_knowledge_constraints,
    write_knowledge_template,
)
from src.policies import load_search_regions, write_search_regions_template
from src.settings import (
    DEFAULT_RECOMMENDATION_COUNT,
    EXPERIMENT_FILE_NAME,
    PROBLEM_FILE_NAME,
    RECOMMENDATION_FILE_NAME,
    RESPONSE_SPACE_DIRECTORY_NAME,
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
from src.stopping import (
    StopDecision,
    StopStatus,
    build_stop_context,
    convergence_history,
    evaluate_stop,
    load_stop_config,
    load_stop_history,
    objective_target_reached,
    predicted_optimum_index,
    write_stop_settings_template,
    write_stop_status,
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


def _knowledge_diagnostic_rows(
    model_type: str,
    report: dict[str, object],
) -> list[dict[str, object]]:
    """知識診断の共通形式へ、モデル種別を付けて変換します。"""

    return [
        {
            "model_type": model_type,
            "rule_id": item.rule_id,
            "loss": item.loss,
            "violation_rate": item.violation_rate,
            "mean_violation": item.mean_violation,
            "max_violation": item.max_violation,
            "points": item.points,
        }
        for item in report["rules"]
    ]


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
        write_knowledge_template(temporary / "knowledge_constraints.csv")
        write_search_regions_template(temporary / "search_regions.csv")
        write_stop_settings_template(temporary / "stop_settings.csv")
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
    # prepare時にも補助CSVを検査し、学習前に入力ミスを知らせる。
    load_knowledge_constraints(current_trial / "knowledge_constraints.csv", problem)
    parameter_names = [item.column for item in problem.parameters]
    try:
        load_search_regions(current_trial / "search_regions.csv", parameter_names)
        load_stop_config(
            current_trial / "stop_settings.csv",
            objective_direction=problem.objective.direction,
        )
    except ValueError as error:
        raise UserInputError(str(error)) from error

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
    from src.hybrid.reporting import archive_recommendations

    output_path = current_trial / "output" / RECOMMENDATION_FILE_NAME
    archived = archive_recommendations(output_path)
    if archived is not None:
        print(f"前回の推薦を履歴へ保存しました: {archived}")
    problem = load_and_validate_problem(current_trial / PROBLEM_FILE_NAME)
    knowledge_rules = load_knowledge_constraints(
        current_trial / "knowledge_constraints.csv", problem
    )
    parameter_names = [item.column for item in problem.parameters]
    try:
        region_policy = load_search_regions(
            current_trial / "search_regions.csv", parameter_names
        )
        stop_config = load_stop_config(
            current_trial / "stop_settings.csv",
            objective_direction=problem.objective.direction,
        )
    except ValueError as error:
        raise UserInputError(str(error)) from error
    experiments = load_and_validate_experiments(
        current_trial / "data" / EXPERIMENT_FILE_NAME,
        problem,
    )

    # 数値計算ライブラリは探索実行時にだけ必要です。
    try:
        import numpy as np

        from src.hybrid.gp_component import ModelTrainingError
        from src.hybrid.model import train_hybrid_model
        from src.hybrid.nn_component import NeuralTrainingError
        from src.hybrid.optimizer import (
            find_observed_best,
            observed_feasible_mask,
            run_optimization,
        )
        from src.hybrid.reporting import (
            next_run_id,
            print_run_result,
            response_space_path,
            write_recommendations,
            write_response_space,
        )
        from src.parameter_space import make_parameter_grid
        from src.preprocessing import normalize_parameters, prepare_experiments
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise UserInputError(
                "NumPyがインストールされていません。\n"
                "次を実行してから、もう一度--runを実行してください。\n"
                "python -m pip install -r requirements.txt"
            ) from error
        raise

    prepared = prepare_experiments(experiments, problem)
    raw_grid = make_parameter_grid(problem)
    grid_policy = region_policy.evaluate(raw_grid, parameter_names)
    stop_path = current_trial / "output" / "stopping_status.json"
    input_context = build_stop_context(
        experiments,
        problem=problem,
        knowledge_rules=knowledge_rules,
        region_policy=region_policy,
        stop_config=stop_config,
    )
    stop_history = load_stop_history(
        stop_path, experiments.row_count, input_context=input_context,
    )
    additional_experiments = max(
        0, experiments.row_count - int(stop_history["initial_data_rows"])
    )
    observed_best = find_observed_best(
        prepared,
        problem,
        observed_feasible_mask(prepared, problem, region_policy),
    )
    required_stop = evaluate_stop(
        config=stop_config,
        additional_experiments=additional_experiments,
        allowed_candidates=raw_grid[grid_policy.allowed_mask],
        explored_candidates=prepared.raw_parameters,
        target_reached=objective_target_reached(observed_best, problem),
    )
    if required_stop.should_stop:
        write_stop_status(
            stop_path,
            decision=required_stop,
            initial_data_rows=int(stop_history["initial_data_rows"]),
            runs=stop_history["runs"],
            input_context=input_context,
            convergence_start_index=stop_history["convergence_start_index"],
        )
        print(f"終了判定: {required_stop.status.value}")
        for reason in required_stop.reasons:
            print(f"- {reason}")
        print(f"判定詳細: {stop_path}")
        return

    if stop_config.max_additional_experiments is not None:
        remaining = stop_config.max_additional_experiments - additional_experiments
        if recommendation_count > remaining:
            print(f"残り実験予算に合わせ、推薦件数を{remaining}件に調整します。")
            recommendation_count = remaining

    try:
        training = train_hybrid_model(prepared, problem, knowledge_rules)
        result = run_optimization(
            experiments,
            prepared,
            problem,
            recommendation_count=recommendation_count,
            model=training.model,
            region_policy=region_policy,
        )
    except (ModelTrainingError, NeuralTrainingError) as error:
        raise UserInputError(str(error)) from error

    # 知見損失へ直接入ったNNと、最終Hybrid平均を分けて診断する。
    diagnostic_points = build_knowledge_points(
        problem,
        knowledge_rules,
        max_points=problem.candidate_count,
    )
    diagnostics = evaluate_rules_on_model(
        training.model.nn_model,
        problem,
        knowledge_rules,
        points=diagnostic_points,
    )
    hybrid_diagnostics = evaluate_rules_on_hybrid(
        training.model,
        problem,
        knowledge_rules,
        points=diagnostic_points,
    )
    observed_diagnostics = evaluate_rules_on_observations(
        prepared,
        problem,
        knowledge_rules,
    )
    diagnostic_path = current_trial / "output" / "knowledge_diagnostics.csv"
    diagnostic_rows = [
        *_knowledge_diagnostic_rows("nn", diagnostics),
        *_knowledge_diagnostic_rows("hybrid", hybrid_diagnostics),
        *_knowledge_diagnostic_rows("observed", observed_diagnostics),
    ]
    diagnostic_header = [
        "model_type",
        "rule_id",
        "loss",
        "violation_rate",
        "mean_violation",
        "max_violation",
        "points",
    ]
    write_csv_atomic(diagnostic_path, diagnostic_header, diagnostic_rows)
    print(f"  知見診断: {diagnostic_path}")
    failed_gates = sorted(
        set(diagnostics["gate"]["failed_rule_ids"])
        | set(hybrid_diagnostics["gate"]["failed_rule_ids"])
        | set(observed_diagnostics["gate"]["failed_rule_ids"])
    )
    if failed_gates:
        write_stop_status(
            stop_path,
            decision=StopDecision(
                StopStatus.STOP_REQUIRED,
                ("必須知見ルールの検査に不合格: " + ", ".join(failed_gates),),
            ),
            initial_data_rows=int(stop_history["initial_data_rows"]),
            runs=stop_history["runs"],
            input_context=input_context,
            convergence_start_index=len(stop_history["runs"]),
        )
        raise UserInputError(
            "strength=5の必須知見ルールに違反または未検証が残ったため、"
            "新しい推薦を公開しません。\n"
            "違反ルール: " + ", ".join(failed_gates)
        )

    normalized_grid = normalize_parameters(raw_grid, problem)
    prediction = training.model.predict(normalized_grid)
    response_directory = current_trial / "output" / RESPONSE_SPACE_DIRECTORY_NAME
    run_id = next_run_id(response_directory)
    hybrid_output_path = response_space_path(response_directory, run_id)
    write_response_space(
        hybrid_output_path,
        run_id,
        experiments,
        prepared,
        problem,
        raw_grid,
        prediction,
        experiment_allowed=grid_policy.allowed_mask,
        preferred_multiplier=grid_policy.preferred_multiplier,
    )

    optimum_index = predicted_optimum_index(
        prediction,
        problem,
        grid_policy.allowed_mask,
    )
    objective_scale = max(
        float(np.std(prepared.result_means[problem.objective.column])),
        1e-12,
    )
    previous_runs = list(stop_history["runs"])
    run_entry = {
        "run_id": run_id,
        "data_rows": experiments.row_count,
        "data_fingerprint": input_context["data_fingerprint"],
        "best_feasible_objective": (
            None if observed_best is None else observed_best[problem.objective.column]
        ),
        "normalized_top_score": (
            float(result.recommendations[0]["recommendation_score"])
            / objective_scale
        ),
        "predicted_optimum": (
            None
            if optimum_index is None
            else normalized_grid[optimum_index].tolist()
        ),
    }
    previous_runs.append(run_entry)
    valid_runs = convergence_history(
        previous_runs, stop_history["convergence_start_index"],
    )
    convergence = evaluate_stop(
        config=stop_config,
        additional_experiments=additional_experiments,
        allowed_candidates=raw_grid[grid_policy.allowed_mask],
        explored_candidates=prepared.raw_parameters,
        target_reached=objective_target_reached(observed_best, problem),
        best_feasible_objective_history=[
            float(item["best_feasible_objective"]) for item in valid_runs
        ],
        support=prediction.support[grid_policy.allowed_mask],
        top_scores=[float(item["normalized_top_score"]) for item in valid_runs],
        predicted_optimum_conditions=[
            item["predicted_optimum"]
            for item in valid_runs
            if item.get("predicted_optimum") is not None
        ],
    )
    write_stop_status(
        stop_path,
        decision=convergence,
        initial_data_rows=int(stop_history["initial_data_rows"]),
        runs=previous_runs,
        input_context=input_context,
        convergence_start_index=stop_history["convergence_start_index"],
    )

    # 診断・解空間・停止状態がそろってから、今回の推薦を公開する。
    write_recommendations(output_path, experiments, problem, result)
    print_run_result(trial_name, experiments, problem, result, output_path)

    print("\nHybrid解空間:")
    print(f"  学習データ: {experiments.row_count}行")
    print(f"  固有条件: {prepared.unique_condition_count}条件")
    print(f"  NN学習エポック: {training.nn_epochs}")
    print(f"  NN標準化MSE: {training.nn_normalized_mse:.6g}")
    print(f"  予測グリッド: {len(raw_grid):,}点")
    print(
        f"  GPデータ支持度: 最小={float(np.min(prediction.support)):.3f}, "
        f"中央値={float(np.median(prediction.support)):.3f}, "
        f"最大={float(np.max(prediction.support)):.3f}"
    )
    print(f"  run: {run_id}")
    print(f"  出力: {hybrid_output_path}")
    print(f"  終了判定: {convergence.status.value} ({stop_path})")
    print(
        "\n[注意] GPデータ支持度は正解確率ではありません。"
        "Hybrid予測は統計モデル同士の融合であり、物理的妥当性を保証しません。"
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
                "\n今回の推薦は公開されていません。\n"
                "前回の推薦はoutput/recommendations_history/で確認できます。",
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
