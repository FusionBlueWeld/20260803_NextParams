"""v004の探索手順。入力検査→停止確認→学習→推薦計算→診断→保存の順で進めます。

学習器と推薦ロジックはhybrid/に置き、必須知識の診断を通過してから推薦を公開します。
"""

from __future__ import annotations

from .trial_inputs import load_trial_inputs

from .settings import (
    EXPERIMENT_FILE_NAME,
    RECOMMENDATION_FILE_NAME,
    RESPONSE_SPACE_DIRECTORY_NAME,
)
from .validation import UserInputError, load_and_validate_experiments, validate_trial_name
from .connection import current_context_observation_mask, fixed_context_grid
from .stopping import (
    StopDecision,
    StopStatus,
    build_stop_context,
    convergence_history,
    evaluate_stop,
    load_stop_history,
    objective_target_reached,
    predicted_optimum_index,
    write_stop_status,
)
from .trials import require_trial
from .diagnostics import write_knowledge_diagnostics


def run_trial(trial_name: str, recommendation_count: int) -> None:
    """入力検査、学習、探索、CSV出力を順番に実行します。"""

    # 1. 入力を検査し、前回の推薦を履歴へ退避します。
    validate_trial_name(trial_name)
    current_trial = require_trial(trial_name)
    from .hybrid.reporting import archive_recommendations

    output_path = current_trial / "output" / RECOMMENDATION_FILE_NAME
    archived = archive_recommendations(output_path)
    if archived is not None:
        print(f"前回の推薦を履歴へ保存しました: {archived}")
    inputs = load_trial_inputs(current_trial)
    problem = inputs.problem
    connection = inputs.connection
    knowledge_rules = inputs.knowledge_rules
    region_policy = inputs.region_policy
    stop_config = inputs.stop_config
    parameter_names = [item.column for item in problem.parameters]

    experiments = load_and_validate_experiments(
        current_trial / "data" / EXPERIMENT_FILE_NAME,
        problem,
    )

    # 数値計算ライブラリは探索実行時にだけ必要です。
    try:
        import numpy as np

        from .hybrid.gp_component import ModelTrainingError
        from .hybrid.model import train_hybrid_model
        from .hybrid.nn_component import NeuralTrainingError
        from .hybrid.optimizer import (
            find_observed_best,
            observed_feasible_mask,
            run_optimization,
        )
        from .hybrid.reporting import (
            next_run_id,
            print_run_result,
            response_space_path,
            write_recommendations,
            write_response_space,
        )
        from .parameter_space import make_parameter_grid
        from .preprocessing import normalize_parameters, prepare_experiments
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise UserInputError(
                "NumPyがインストールされていません。\n"
                "次を実行してから、もう一度--runを実行してください。\n"
                "python -m pip install -r requirements.txt"
            ) from error
        raise

    # 2. 流入状態を固定した候補と実測を準備し、学習前の必須停止を判定します。
    prepared = prepare_experiments(experiments, problem)
    raw_grid = fixed_context_grid(make_parameter_grid(problem), problem, connection)
    grid_policy = region_policy.evaluate(raw_grid, parameter_names)
    stop_path = current_trial / "output" / "stopping_status.json"
    input_context = build_stop_context(
        experiments,
        problem=problem,
        knowledge_rules=knowledge_rules,
        region_policy=region_policy,
        stop_config=stop_config,
        connection=connection,
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
        observed_feasible_mask(prepared, problem, region_policy)
        & current_context_observation_mask(prepared.raw_parameters, problem, connection),
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
        # 3. 予測器を学習し、現在の流入状態に対して次の実験条件を選びます。
        training = train_hybrid_model(prepared, problem, knowledge_rules)
        result = run_optimization(
            experiments,
            prepared,
            problem,
            recommendation_count=recommendation_count,
            model=training.model,
            region_policy=region_policy,
            candidate_grid=raw_grid,
            observed_context_mask=current_context_observation_mask(
                prepared.raw_parameters, problem, connection
            ),
            candidate_identity_columns=(
                None if connection is None else np.asarray(
                    [parameter_names.index(name) for name in connection.controls], dtype=int
                )
            ),
        )
    except (ModelTrainingError, NeuralTrainingError) as error:
        raise UserInputError(str(error)) from error

    # NN・Hybrid・実測のどれかに必須違反があれば、推薦を公開しません。
    failed_gates = write_knowledge_diagnostics(
        current_trial, training.model, prepared, problem, knowledge_rules,
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

    # 4. 知識診断を通過したモデルの解空間と収束履歴を保存します。
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
    if connection is not None:
        from .stage_bundle import export_stage_bundle
        bundle_path = export_stage_bundle(current_trial, run_id, problem, training.model, connection)
        print(f"  stage bundle: {bundle_path}")
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

