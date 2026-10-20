"""Reuse the current prediction for convergence and prospective quality checks."""
import csv
import json
import numpy as np
from .candidate_identity import CandidateIdentity
from .hybrid.optimizer import constraint_probability_target


def promising_region(prediction,problem,allowed,observed_best,config,objective_scale):
    # Use conservative, calibrated two-sigma bounds. Do not remove unsupported
    # candidates merely because their current predicted mean looks unfavorable.
    mask=np.asarray(allowed,dtype=bool).copy()
    for constraint in problem.constraints:
        values=prediction.results[constraint.column]
        target=constraint_probability_target(constraint,config)
        if constraint.direction=='less_equal':
            mask &= values['hybrid_mean']-2*values['hybrid_std']<=target
        else:mask &= values['hybrid_mean']+2*values['hybrid_std']>=target
    if observed_best is not None:
        values=prediction.results[problem.objective.column]
        tolerance=objective_scale*config.min_normalized_improvement
        best=observed_best[problem.objective.column]
        if problem.objective.direction=='minimize':
            mask &= values['hybrid_mean']-2*values['hybrid_std']<=best+tolerance
        else:mask &= values['hybrid_mean']+2*values['hybrid_std']>=best-tolerance
    return mask


def write_prediction_quality(path,archived,experiments,problem,fixed_context):
    payload={'scope':'new_observations_matching_previous_recommendations_only',
             'probability_calibration':'diagnostic_only_not_calibrated_guarantee','matched_measurements':0}
    if archived is not None:
        with archived.open(encoding='utf-8-sig',newline='') as f:recs=list(csv.DictReader(f))
        # 初期設計のCSVには予測がない。その実測を予測精度の評価へ混ぜない。
        recs = [r for r in recs if r.get(problem.objective.column + '_mean', '') != ''
                and r.get(problem.objective.column + '_nn_pred', '') != '']
        if recs:
            prior_rows=int(recs[0]['data_rows'])
            identity=CandidateIdentity(problem,fixed_context)
            expected=identity.keys([[float(r[p.column]) for p in problem.parameters] for r in recs])
            measured=identity.keys(experiments.parameter_rows[prior_rows:])
            lookup={key:r for key,r in zip(expected,recs)}
            matches=[(i+prior_rows,lookup[key]) for i,key in enumerate(measured) if key in lookup]
            payload['matched_measurements']=len(matches)
            if matches:
                objective=problem.objective.column
                truth=np.asarray([experiments.result_rows[objective][i] for i,_ in matches])
                for source,suffix in [('nn','_nn_pred'),('hybrid','_mean')]:
                    estimate=np.asarray([float(r[objective+suffix]) for _,r in matches])
                    payload[source+'_objective_mae']=float(np.mean(np.abs(truth-estimate)))
                passed=np.asarray([all(experiments.result_rows[c.column][i]<=c.target if c.direction=='less_equal'
                    else experiments.result_rows[c.column][i]>=c.target for c in problem.constraints) for i,_ in matches])
                probs=np.asarray([float(r['feasibility_probability']) for _,r in matches])
                payload['actual_feasible_rate']=float(np.mean(passed))
                payload['feasibility_brier']=float(np.mean((probs-passed)**2))
                payload['mean_predicted_feasibility']=float(np.mean(probs))
                payload['previous_prediction_data_rows']=prior_rows
                payload['output_mae'] = {}
                for output in [problem.objective, *problem.constraints]:
                    measured_values = np.array([experiments.result_rows[output.column][i] for i, _ in matches])
                    predicted_values = np.array([float(r[output.column + '_mean']) for _, r in matches])
                    payload['output_mae'][output.column] = float(np.mean(np.abs(measured_values - predicted_values)))
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return payload
