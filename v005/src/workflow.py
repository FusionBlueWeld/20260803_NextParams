"""v005の探索手順。入力検査→停止確認→学習→推薦計算→診断→保存の順で進めます。

学習器と推薦ロジックはhybrid/に置き、必須知識の診断を通過してから推薦を公開します。
"""

from __future__ import annotations
from dataclasses import asdict, replace
import json
import time
from .runtime import numerical_limits

from .trial_inputs import load_trial_inputs

from .settings import (
    EXPERIMENT_FILE_NAME,
    RECOMMENDATION_FILE_NAME,
    RESPONSE_SPACE_DIRECTORY_NAME,
)
from .validation import UserInputError, load_and_validate_experiments, validate_trial_name
from .connection import current_context_observation_mask
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
    validate_trial_name(trial_name)
    trial=require_trial(trial_name)
    from .hybrid.reporting import archive_recommendations
    archived=archive_recommendations(trial/'output'/RECOMMENDATION_FILE_NAME)
    if archived is not None:print(f"前回の推薦を履歴へ保存しました: {archived}")
    inputs=load_trial_inputs(trial)
    config=inputs.search_config
    with numerical_limits(config.cpu_threads) as runtime:
        _run_trial(trial_name,recommendation_count,runtime,inputs,archived)


def _run_trial(trial_name: str, recommendation_count: int, runtime: dict, inputs, archived) -> None:
    """入力検査、学習、探索、CSV出力を順番に実行します。"""

    # 1. 入力を検査し、前回の推薦を履歴へ退避します。
    started=time.perf_counter()
    validate_trial_name(trial_name)
    current_trial = require_trial(trial_name)
    output_path = current_trial / "output" / RECOMMENDATION_FILE_NAME
    problem = inputs.problem
    connection = inputs.connection
    knowledge_rules = inputs.knowledge_rules
    region_policy = inputs.region_policy
    stop_config = inputs.stop_config
    search_config = inputs.search_config
    focus_config = inputs.focus_config
    parameter_names = [item.column for item in problem.parameters]

    experiments = load_and_validate_experiments(
        current_trial / "data" / EXPERIMENT_FILE_NAME,
        problem,
        allow_empty=focus_config.initial_design_enabled,
    )
    from .search_config import effective_search_config
    search_config=effective_search_config(search_config,problem,experiments)

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
        from .parameter_space import make_parameter_grid, candidate_metadata
        from .preprocessing import normalize_parameters, prepare_experiments
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise UserInputError(
                "NumPyがインストールされていません。\n"
                "次を実行してから、もう一度--runを実行してください。\n"
                "python -m pip install -r requirements.txt"
            ) from error
        raise

    # 2. 初期条件にも同じ実験予算を適用。結果を作らず入力条件だけを公開します。
    focus_active = focus_config.initial_design_enabled or focus_config.parameter_focus_enabled or focus_config.max_total_experiments is not None
    if _run_initial_design(current_trial, experiments, inputs, recommendation_count, runtime, started):
        return

    # 流入状態を固定した候補と実測を準備し、学習前の必須停止を判定します。
    fixed_context={} if connection is None else connection.current_incoming_context
    prepared = prepare_experiments(experiments, problem, fixed_context=fixed_context)
    focus_state = None
    focus_plan = None
    if focus_config.parameter_focus_enabled:
        from .parameter_focus import analyze_focus, focus_candidate_plan, save_focus
        focus_path = current_trial / 'output' / 'parameter_focus.json'
        previous_focus = json.loads(focus_path.read_text(encoding='utf-8')) if focus_path.exists() else None
        focus_state = analyze_focus(prepared, problem, connection, focus_config, previous_focus, region_policy)
    try:
        # 一時固定した軸は直積の前に固定する。全空間の完全性とは別に記録する。
        raw_grid = np.empty((0, len(problem.parameters))) if focus_state and focus_state['fixed_values'] else make_parameter_grid(problem, connection,
                                       max_candidates=search_config.max_candidates,
                                       seed=search_config.seed, mode=search_config.candidate_mode)
    except ValueError as error:
        raise UserInputError(str(error)) from error
    search_metadata = candidate_metadata(problem, connection,
                                         max_candidates=search_config.max_candidates,
                                         mode=search_config.candidate_mode)
    if focus_state is not None:
        raw_grid, focus_plan = focus_candidate_plan(raw_grid, prepared, problem, connection,
            focus_state, focus_config, search_config, region_policy)
        if focus_plan['pool_changed']:
            search_metadata['exhaustive'] = False
        search_metadata['candidate_space_scope'] = focus_plan['candidate_space_scope']
        focus_state['candidate_space_scope'] = focus_plan['candidate_space_scope']
        focus_state['candidate_space_complete'] = search_metadata['exhaustive']
        focus_state['observed_data_rows'] = experiments.row_count
        save_focus(current_trial / 'output' / 'parameter_focus.json', focus_state)
    search_metadata["evaluated_candidates"] = len(raw_grid)
    search_metadata["settings"] = asdict(search_config)
    search_metadata['runtime']=runtime
    grid_policy = region_policy.evaluate(raw_grid, parameter_names)
    from .candidate_identity import CandidateIdentity
    grid_known=CandidateIdentity(problem,fixed_context).measured_mask(raw_grid,prepared.raw_parameters)
    all_allowed_explored=bool(np.all(grid_known[grid_policy.allowed_mask])) if search_metadata['exhaustive'] else None
    stop_path = current_trial / "output" / "stopping_status.json"
    input_context = build_stop_context(
        experiments,
        problem=problem,
        knowledge_rules=knowledge_rules,
        region_policy=region_policy,
        stop_config=stop_config,
        connection=connection,
        search_config=search_config,
        focus_config=focus_config if focus_active else None,
        focus_signature=None if focus_state is None else focus_state['fixed_values'],
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
    from .workflow_metrics import promising_region, write_prediction_quality
    quality=write_prediction_quality(current_trial/'output'/'prediction_quality.json',
                                    archived,experiments,problem,fixed_context)
    required_stop = evaluate_stop(
        config=stop_config,
        additional_experiments=additional_experiments,
        all_allowed_explored=all_allowed_explored,
        target_reached=objective_target_reached(observed_best, problem),
    )
    if not required_stop.should_stop and not np.any(grid_policy.allowed_mask & ~grid_known):
        required_stop=StopDecision(StopStatus.REVIEW_REQUIRED,
            ('抽出候補に未測定の許可条件がありません。全空間の探索完了ではなく、候補予算・seed・領域設定の見直しが必要',),
            {**required_stop.metrics,'sampled_candidates_exhausted':True,'converged':False})
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
        search_metadata['wall_seconds']=time.perf_counter()-started
        search_metadata['learning_skipped']=True
        (current_trial/'output'/'search_budget.json').write_text(json.dumps(search_metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        return

    if stop_config.max_additional_experiments is not None:
        remaining = stop_config.max_additional_experiments - additional_experiments
        if recommendation_count > remaining:
            print(f"残り実験予算に合わせ、推薦件数を{remaining}件に調整します。")
            recommendation_count = remaining
    if focus_config.max_total_experiments is not None:
        recommendation_count = min(recommendation_count, focus_config.max_total_experiments - experiments.row_count)

    try:
        # 3. 予測器を学習し、現在の流入状態に対して次の実験条件を選びます。
        training = train_hybrid_model(prepared, problem, knowledge_rules,
                                      max_training_samples=search_config.max_training_samples,
                                      max_nn_epochs=search_config.max_nn_epochs,
                                      max_learning_seconds=search_config.max_learning_seconds)
        # One full prediction feeds selection, stopping and GUI response CSV.
        prediction_started=time.perf_counter()
        normalized_grid=normalize_parameters(raw_grid,problem)
        prediction=training.model.predict(normalized_grid)
        prediction_seconds=time.perf_counter()-prediction_started
        if focus_state is not None:
            from .parameter_focus import check_hybrid_effects, save_focus
            check_hybrid_effects(focus_state, raw_grid, prediction, focus_plan, problem, focus_config)
            save_focus(current_trial / 'output' / 'parameter_focus.json', focus_state)
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
            search_config=search_config,candidate_prediction=prediction,fixed_context=fixed_context,
            candidate_focus=focus_plan,
            confirmation_slots=focus_config.confirmation_slots if focus_plan is not None else 0,
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
    response_directory = current_trial / "output" / RESPONSE_SPACE_DIRECTORY_NAME
    run_id = next_run_id(response_directory)
    if focus_state is not None:
        from .parameter_focus import save_focus
        save_focus(current_trial / 'output' / 'parameter_focus' / f'{run_id}.json', focus_state)
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
        ensemble_size=len(training.model.nn_model.members),
        uncertainty_calibration_scale=training.model.uncertainty_calibration_scale,
        acquisition_uncertainty_scale=training.model.acquisition_uncertainty_scale,
    )

    optimum_index = predicted_optimum_index(
        prediction,
        problem,
        grid_policy.allowed_mask,
        search_config=search_config,
    )
    objective_scale = max(
        float(np.std(prepared.result_means[problem.objective.column])),
        1e-12,
    )
    if objective_scale<=1e-12:
        objective_scale=max(abs(float(np.mean(prepared.result_means[problem.objective.column])))*0.05,1.0)
    promising_mask=promising_region(prediction,problem,grid_policy.allowed_mask,
                                   observed_best,search_config,objective_scale)
    previous_runs = list(stop_history["runs"])
    run_entry = {
        "run_id": run_id,
        "data_rows": experiments.row_count,
        "data_fingerprint": input_context["data_fingerprint"],
        "best_feasible_objective": (
            None if observed_best is None else observed_best[problem.objective.column]
        ),
        "normalized_top_score": (
            float(result.top_score)
            / objective_scale
        ),
        "predicted_optimum": (
            None
            if optimum_index is None
            else normalized_grid[optimum_index].tolist()
        ),
        'predicted_optimum_objective':None if optimum_index is None else
            float(prediction.results[problem.objective.column]['hybrid_mean'][optimum_index]),
    }
    previous_runs.append(run_entry)
    valid_runs = convergence_history(
        previous_runs, stop_history["convergence_start_index"],
    )
    convergence = evaluate_stop(
        config=stop_config,
        additional_experiments=additional_experiments,
        all_allowed_explored=all_allowed_explored,
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
        promising_support=prediction.support[promising_mask],
        promising_region_complete=search_metadata['exhaustive'],
        predicted_optimum_objective_history=[item['predicted_optimum_objective'] for item in valid_runs
            if item.get('predicted_optimum_objective') is not None],
    )
    # 限定した断面の支持度や枯渇を、元の全空間の収束には読み替えない。
    if focus_state is not None:
        convergence = replace(convergence, metrics={**convergence.metrics,
            'focus_phase': focus_state['phase'],
            'temporarily_fixed_controls': len(focus_state['fixed_values']),
            'convergence_scope': focus_plan['candidate_space_scope'],
            'converged': convergence.status == StopStatus.STOP_RECOMMENDED})
    if not result.recommendations and not convergence.should_stop:
        convergence=StopDecision(StopStatus.REVIEW_REQUIRED,
            ('有用な未測定候補が不足。収束の根拠は未完了のため、追加実験または探索設定の見直しが必要',),
            {**convergence.metrics,'converged':False,'useful_candidate_count':result.useful_candidate_count})
    write_stop_status(
        stop_path,
        decision=convergence,
        initial_data_rows=int(stop_history["initial_data_rows"]),
        runs=previous_runs,
        input_context=input_context,
        convergence_start_index=stop_history["convergence_start_index"],
    )

    # 診断・解空間・停止状態がそろってから、今回の推薦を公開する。
    # A stopped run must never publish actionable recommendations.
    if not convergence.should_stop:
        write_recommendations(output_path, experiments, problem, result)
    search_metadata["learning"] = {"source_samples": training.source_samples,
                                    "retained_samples": training.retained_samples,
                                    "seconds": training.learning_seconds}
    search_metadata["knowledge_diagnostic_scope"] = "budgeted_points_and_all_observations"
    search_metadata['prediction_seconds']=prediction_seconds
    search_metadata['prediction_grid_passes']=1
    search_metadata['wall_seconds']=time.perf_counter()-started
    search_metadata['recommendations']={'requested':recommendation_count,'selected':len(result.recommendations),
        'published':len(result.recommendations) if not convergence.should_stop else 0,
        'useful_candidate_count':result.useful_candidate_count,'reason':result.batch_reduction_reason}
    search_metadata['prediction_quality']=quality
    if focus_state is not None:
        search_metadata['parameter_focus'] = {
            'phase': focus_state['phase'], 'fixed_values': focus_state['fixed_values'],
            'confirmation_slots': focus_config.confirmation_slots,
            'full_model_input_columns_retained': True}
    if connection is not None:
        from .stage_bundle import export_stage_bundle
        bundle_path = export_stage_bundle(current_trial, run_id, problem, training.model, connection)
        print(f"  stage bundle: {bundle_path}")
    search_metadata['wall_seconds']=time.perf_counter()-started
    (current_trial / "output" / "search_budget.json").write_text(
        json.dumps(search_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    display_result=replace(result,recommendations=[]) if convergence.should_stop else result
    print_run_result(trial_name, experiments, problem, display_result, output_path)

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


def _run_initial_design(trial, experiments, inputs, count, runtime, started):
    """空データから始め、配置と予測誤差に応じ初期確認を延長する。"""
    config = inputs.focus_config
    enabled = config.initial_design_enabled or config.parameter_focus_enabled or config.max_total_experiments is not None
    if not enabled:
        return False
    from .experimental_design import design_target, design_diagnostics, recommend_design
    from .parameter_focus import save_focus
    context = build_stop_context(experiments, problem=inputs.problem,
        knowledge_rules=inputs.knowledge_rules, region_policy=inputs.region_policy,
        stop_config=inputs.stop_config, connection=inputs.connection,
        search_config=inputs.search_config, focus_config=config)
    stop_path = trial / 'output' / 'stopping_status.json'
    history = load_stop_history(stop_path, experiments.row_count, input_context=context)
    remaining = count
    if config.max_total_experiments is not None:
        remaining = min(remaining, config.max_total_experiments - experiments.row_count)
    if inputs.stop_config.max_additional_experiments is not None:
        additional = max(0, experiments.row_count - history['initial_data_rows'])
        remaining = min(remaining, inputs.stop_config.max_additional_experiments - additional)
    if remaining <= 0:
        decision = StopDecision(StopStatus.STOP_REQUIRED, ('実験予算に到達（初期・改善・確認の実測を含む）',),
                                {'converged': False, 'actual_experiments': experiments.row_count})
    elif not config.initial_design_enabled:
        return False
    else:
        initial_status_path = trial / 'output' / 'initial_design_status.json'
        initial_status = json.loads(initial_status_path.read_text(encoding='utf-8')) if initial_status_path.exists() else {}
        if initial_status.get('phase') == 'SEARCH' and initial_status.get('settings_fingerprint') == context['settings_fingerprint']:
            return False  # 初期段階は同じ問題・流入状態で一度だけ。確認は探索側が担当する。
        diagnostics = design_diagnostics(experiments, inputs.problem, inputs.connection)
        unique = diagnostics['current_context_unique_conditions']
        target = design_target(inputs.problem, inputs.connection, config)
        from .parameter_space import candidate_metadata, make_parameter_grid
        from .candidate_identity import CandidateIdentity
        import numpy as np
        metadata = candidate_metadata(inputs.problem, inputs.connection,
            max_candidates=inputs.search_config.max_candidates, mode=inputs.search_config.candidate_mode)
        finite_space_measured = False
        if metadata['exhaustive']:
            finite_grid = make_parameter_grid(inputs.problem, inputs.connection,
                max_candidates=inputs.search_config.max_candidates, mode=inputs.search_config.candidate_mode)
            allowed = inputs.region_policy.evaluate(finite_grid, [p.column for p in inputs.problem.parameters]).allowed_mask
            incoming = {} if inputs.connection is None else inputs.connection.current_incoming_context
            known = CandidateIdentity(inputs.problem, incoming).measured_mask(finite_grid, experiments.parameter_rows)
            finite_space_measured = bool(np.any(allowed) and np.all(known[allowed]))
        ready = unique >= target and diagnostics['design_ready']
        # 単純な配置だけでは見えない曲がり・組合せ作用は、未使用データの誤差で保留。
        if ready and unique < config.initial_max_conditions:
            import numpy as np
            from .preprocessing import prepare_experiments
            from .parameter_focus import analyze_focus
            incoming = {} if inputs.connection is None else inputs.connection.current_incoming_context
            prepared = prepare_experiments(experiments, inputs.problem, incoming)
            assessment = analyze_focus(prepared, inputs.problem, inputs.connection, config, region_policy=inputs.region_policy)
            ready = assessment['analysis']['identifiable']
            for output, error in assessment['analysis']['prediction_errors'].items():
                scale = max(float(np.std(prepared.result_means[output])), 1e-9)
                ready &= error <= .25 * scale
        if ready or unique >= config.initial_max_conditions or finite_space_measured:
            save_focus(trial / 'output' / 'initial_design_status.json', {
                'phase': 'SEARCH', 'target_conditions': target, 'actual_experiments': experiments.row_count,
                'design_diagnostics': diagnostics, 'converged': False,
                'settings_fingerprint': context['settings_fingerprint'],
                'initial_limit_reached': unique >= config.initial_max_conditions,
                'finite_space_measured': finite_space_measured})
            return False
        remaining = min(remaining, config.initial_max_conditions - unique)
        conditions = recommend_design(inputs.problem, inputs.connection, inputs.region_policy,
            experiments, remaining, config, inputs.search_config, seed_offset=unique)
        from .hybrid.optimizer import OptimizationResult
        from .hybrid.reporting import write_recommendations
        recommendations = [{'rank': i + 1, 'selection_role': 'INITIAL_DESIGN',
            'recommendation_reason': '水準の不足と入力配置を確認する初期条件。予測値は未取得',
            **{p.column: float(value) for p, value in zip(inputs.problem.parameters, row)}}
            for i, row in enumerate(conditions)]
        decision = StopDecision(StopStatus.CONTINUE if recommendations else StopStatus.REVIEW_REQUIRED,
            ('初期実験の実測待ち' if recommendations else '抽出した許可候補が不足。初期設計は未完了',),
            {'converged': False, 'phase': 'INITIAL_DESIGN', 'target_conditions': target})
        save_focus(trial / 'output' / 'initial_design_status.json', {
            'phase': 'INITIAL_DESIGN', 'target_conditions': target, 'design_diagnostics': diagnostics,
            'settings_fingerprint': context['settings_fingerprint'],
            'actual_experiments': experiments.row_count, 'recommended_conditions': len(recommendations),
            'converged': False, 'predictions_available': False})
        if recommendations:
            result = OptimizationResult(recommendations, count, 0., 0., 0., False, len(conditions),
                                        len(conditions), 0, None, False, [])
            write_recommendations(trial / 'output' / RECOMMENDATION_FILE_NAME, experiments, inputs.problem, result)
    write_stop_status(stop_path, decision=decision, initial_data_rows=history['initial_data_rows'],
        runs=history['runs'], input_context=context, convergence_start_index=history['convergence_start_index'])
    save_focus(trial / 'output' / 'search_budget.json', {
        'learning_skipped': True, 'prediction_grid_passes': 0, 'runtime': runtime,
        'actual_experiments': experiments.row_count, 'wall_seconds': time.perf_counter() - started,
        'recommendations': {'published': 0 if decision.should_stop else len(recommendations)}})
    print(f"探索段階: {decision.metrics.get('phase', 'BUDGET_STOP')} / {decision.status.value}")
    for reason in decision.reasons:
        print(f"- {reason}")
    return True

