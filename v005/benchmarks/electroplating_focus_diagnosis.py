"""Posthoc diagnosis of saved trials. Oracle is never used for optimization.

Reads immutable historical trials; writes a new diagnosis directory only.
"""
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from validation.single.src.simulators import Simulator
from v005.src.trial_inputs import load_trial_inputs
from v005.src.validation import load_and_validate_experiments
from v005.src.preprocessing import prepare_experiments
from v005.src.parameter_focus import analyze_focus
from v005.src.hybrid.gp_component import expected_improvement, constraint_probability

OUT = ROOT/'v005/trials/_electroplating_focus_diagnosis_20261010'
NAMES = {'current_density_a_dm2':'電流密度', 'bath_temperature_c':'浴温度',
         'plating_time_min':'時間', 'copper_concentration_mol_l':'銅濃度',
         'agitation_speed_m_s':'撹拌', 'electrode_gap_cm':'電極間隔',
         'duty_cycle':'デューティ比', 'bath_ph':'pH'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variance_indices(values, shape):
    tensor = values.reshape(shape)
    var = np.var(tensor)
    first, total = [], []
    for j in range(len(shape)):
        others = tuple(i for i in range(len(shape)) if i != j)
        first.append(float(np.var(np.mean(tensor, axis=others))/var))
        total.append(float(1-np.var(np.mean(tensor, axis=j))/var))
    return {'first_order': first, 'total_effect': total,
            'interaction_variance_fraction': float(1-sum(first)),
            'domain': 'uniform full discrete grid, including constraint failures'}


def basis(z, interactions=False):
    columns = [np.ones(len(z))]
    for j in range(z.shape[1]):
        columns.extend([z[:,j], z[:,j]**2])
    if interactions:
        columns.extend(z[:,i]*z[:,j] for i in range(z.shape[1]) for j in range(i+1,z.shape[1]))
    return np.column_stack(columns)


def regression_cv(z, error, thickness):
    folds = np.array_split(np.random.default_rng(20261010).permutation(len(z)), 5)
    result = {}
    for interactions in [False, True]:
        features = basis(z, interactions)
        for signed in [False, True]:
            target = thickness-20 if signed else error
            prediction = np.empty(len(z))
            for held in folds:
                keep = np.setdiff1d(np.arange(len(z)), held)
                penalty = np.sqrt(1e-3)*np.eye(features.shape[1]); penalty[0,0] = 0
                fit = np.linalg.lstsq(np.vstack([features[keep],penalty]),
                    np.r_[target[keep],np.zeros(len(penalty))], rcond=None)[0]
                p = features[held]@fit
                prediction[held] = np.abs(p) if signed else p
            key = ('signed_thickness' if signed else 'absolute_error') + ('_with_interactions' if interactions else '_additive')
            result[key] = {'features': features.shape[1],
                           'heldout_mae_um': float(np.mean(np.abs(prediction-error))),
                           'heldout_rmse_um': float(np.sqrt(np.mean((prediction-error)**2)))}
    return result


def main():
    OUT.mkdir(exist_ok=False)
    sim = Simulator(ROOT/'validation/single/electroplating')
    grid = sim.grid(); truth = sim.evaluate_points(grid); feasible = sim.feasible(truth)
    objective = truth[sim.objective.column]
    optimal_index = sim.best_index(truth)
    optimum = float(objective[optimal_index])
    optimal_point = grid[optimal_index]
    counts = {str(t): int(np.sum(feasible & (objective<=t))) for t in [.0001,.001,.01,.02,.1,1]}
    sensitivity = {n: variance_indices(truth[n], [len(a) for a in sim.axes])
                   for n in ['thickness_error_um','deposit_thickness_um','roughness_ra_um','current_efficiency']}
    result = {'scope': 'posthoc diagnostic only; no oracle-informed optimizer execution',
              'parameters': sim.parameter_columns, 'grid_count':len(grid),
              'feasible_count':int(feasible.sum()), 'grid_optimum_um':optimum,
              'optimal_point': dict(zip(sim.parameter_columns, optimal_point.tolist())),
              'feasible_conditions_below_illustrative_tolerance_um':counts,
              'illustrative_tolerances_are_not_product_specifications':True,
              'sensitivity':sensitivity, 'trials':[]}
    for seed in [42,0,1]:
        tag = '20261010_focus_on' if seed==42 else '20261010_focus_on_repeats'
        trial = ROOT/'v005/trials'/f'trial_single_arrival_electroplating_s{seed}_{tag}'
        before = {str(p.relative_to(trial)):digest(p) for p in trial.rglob('*')
                  if p.is_file() and ('response_spaces' not in p.parts)}
        rows = csvrows(trial/'data/experiments.csv')
        settings = load_trial_inputs(trial)
        experiments = load_and_validate_experiments(trial/'data/experiments.csv', settings.problem)
        prepared = prepare_experiments(experiments,settings.problem)
        focus = read(trial/'output/parameter_focus.json')
        snapshots = [read(p) for p in sorted((trial/'output/parameter_focus').glob('*.json'))]
        ledger = read(trial/'ledger.json')
        pmat = np.array([[float(r[n]) for n in sim.parameter_columns] for r in rows])
        y = np.array([float(r['thickness_error_um']) for r in rows])
        measured_outputs = {n:np.array([float(r[n]) for r in rows]) for n in sim.output_columns}
        best_index = sim.best_index(measured_outputs); anchor = pmat[best_index]
        local_distance = np.sum(~np.isclose(grid,anchor,atol=1e-12,rtol=1e-12),axis=1)
        neighborhoods = {}
        for k in [1,2,3]:
            mask = feasible & (local_distance<=k)
            indices = np.flatnonzero(mask)
            best = indices[np.argmin(objective[indices])]
            neighborhoods[str(k)] = {'best_error_um':float(objective[best]),
                                    'conditions':int(mask.sum()),
                                    'best_point':grid[best].tolist()}
        freeze = []
        for j,n in enumerate(sim.parameter_columns):
            mask = feasible & np.isclose(grid[:,j],anchor[j],atol=1e-12,rtol=1e-12)
            indices = np.flatnonzero(mask)
            index = indices[np.argmin(objective[indices])]
            line = np.tile(anchor,(len(sim.axes[j]),1)); line[:,j]=sim.axes[j]
            line_outputs = sim.evaluate_points(line)
            ranges = {n:float(np.ptp(line_outputs[n])) for n in
                      ['thickness_error_um','roughness_ra_um','current_efficiency']}
            background_groups = {}
            for row in pmat:
                key = tuple(np.delete(row,j))
                background_groups.setdefault(key,set()).add(float(row[j]))
            freeze.append({'parameter':n,'anchor_level':float(anchor[j]),
                           'optimal_level':float(optimal_point[j]),
                           'global_optimum_kept_if_fixed_here':bool(np.isclose(anchor[j],optimal_point[j])),
                           'best_error_um_after_fixing_at_anchor':float(objective[index]),
                           'true_one_variable_ranges_at_anchor':ranges,
                           'backgrounds_with_two_or_more_levels':sum(len(v)>=2 for v in background_groups.values()),
                           'backgrounds_with_all_four_levels':sum(len(v)==4 for v in background_groups.values())})
        replay = {}
        for tolerance in [.01,.1,1.]:
            config = replace(settings.focus_config, effect_tolerances={
                'thickness_error_um':tolerance,'roughness_ra_um':.05,'current_efficiency':.02})
            state = analyze_focus(prepared,settings.problem,settings.connection,config,
                                  region_policy=settings.region_policy)
            replay[str(tolerance)] = {'scope':'single final-data diagnostic; not sequential stability proof',
                                     'fixed_values':state['fixed_values'],
                                     'states': {p['column']:p['state'] for p in state['parameters']},
                                     'checks': {p['column']:p['background_checks'] for p in state['parameters']}}
        last_response = sorted((trial/'output/response_spaces').glob('*.csv'))[-1]
        columns = sim.parameter_columns + ['thickness_error_um_hybrid_mean','thickness_error_um_acquisition_std',
                  'deposit_thickness_um_hybrid_mean','roughness_ra_um_hybrid_mean','roughness_ra_um_hybrid_std',
                  'current_efficiency_hybrid_mean','current_efficiency_hybrid_std']
        prediction_rows = []
        with last_response.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                prediction_rows.append([float(r[n]) for n in columns])
        a = np.array(prediction_rows); del prediction_rows
        assert np.allclose(a[:,:8],grid,rtol=1e-12,atol=1e-12)
        pred = dict(zip(columns,a.T))
        mean = pred['thickness_error_um_hybrid_mean']
        probability = constraint_probability(pred['roughness_ra_um_hybrid_mean'],pred['roughness_ra_um_hybrid_std'],
                                              'less_equal',.65)*constraint_probability(
                                              pred['current_efficiency_hybrid_mean'],pred['current_efficiency_hybrid_std'],
                                              'greater_equal',.8)
        # Last model used 149 measurements; the 150th measurement is held out.
        modeled_rows = int(last_response.name.split('_')[-1].split('.')[0])+47
        assert modeled_rows == 149
        observed = np.array([float(r['thickness_error_um']) for r in rows[:149] if
                            float(r['roughness_ra_um'])<=.65 and float(r['current_efficiency'])>=.8])
        ei = expected_improvement(mean,pred['thickness_error_um_acquisition_std'],float(observed.min()),'minimize')
        scores = ei*probability
        measured_indices = np.ravel_multi_index(tuple(np.argmin(np.abs(pmat[:149,j,None]-sim.axes[j]),axis=1)
                                               for j in range(8)),[4]*8)
        unmeasured = np.ones(len(grid),dtype=bool); unmeasured[measured_indices]=False
        rank = int(np.sum(unmeasured & (scores>scores[optimal_index]))+1)
        pred_metrics = {'source':str(last_response.relative_to(ROOT)), 'modeled_measurements':149,
                        'objective_mae_all_um':float(np.mean(np.abs(mean-objective))),
                        'objective_mae_near_target_um':float(np.mean(np.abs(mean[feasible & (objective<=.1)]-objective[feasible & (objective<=.1)]))),
                        'error_derived_from_predicted_thickness_mae_um':float(np.mean(np.abs(np.abs(pred['deposit_thickness_um_hybrid_mean']-20)-objective))),
                        'negative_error_predictions':int(np.sum(mean<0)),
                        'true_optimal_point_predicted_error_um':float(mean[optimal_index]),
                        'true_optimal_point_acquisition_std_um':float(pred['thickness_error_um_acquisition_std'][optimal_index]),
                        'true_optimal_point_feasibility_probability':float(probability[optimal_index]),
                        'true_optimal_point_EI_score_rank_among_unmeasured':rank,
                        'unmeasured_count':int(unmeasured.sum()),
                        'scope':'rank uses unweighted EI*feasibility; focus weights are inspected separately'}
        scales = {n:float(np.std(prepared.result_means[n])) for n in ['thickness_error_um','roughness_ra_um','current_efficiency']}
        t = {'seed':seed,'trial':trial.name,'best_error_um':float(y[best_index]),
             'best_error_nm':float(y[best_index]*1000),
             'relative_thickness_error_percent':float(y[best_index]/20*100),
             'roles':dict(Counter(e['selection_role'] for e in ledger)),
             'model_runs':len(snapshots),
             'runs_with_all_weights_one':sum(all(p['weight']==1 for p in s['parameters']) for s in snapshots),
             'runs_with_fixed_inputs':sum(bool(s['fixed_values']) for s in snapshots),
             'last_analysis':focus['analysis'],
             'last_parameter_effects':focus['parameters'],
             'final_data_noise_proxy_um':float(prepared.result_noise_std['thickness_error_um']),
             'noise_only_lower_bound_on_upper_effect_um':float(2*prepared.result_noise_std['thickness_error_um']),
             'last_errors_divided_by_observed_std':{n:focus['analysis']['prediction_errors'][n]/scales[n] for n in scales},
             'neighborhood_search_posthoc':neighborhoods,'freeze_audit_posthoc':freeze,
             'tolerance_only_replay':replay,
             'observed_data_regression_CV':regression_cv(prepared.normalized_parameters*2-1,
                 prepared.result_means['thickness_error_um'], prepared.result_means['deposit_thickness_um']),
             'prediction_metrics':pred_metrics}
        assert all(digest(trial/n)==h for n,h in before.items()), 'Historical files changed'
        result['trials'].append(t)
        print(json.dumps({'seed':seed,'roles':t['roles'],'prediction':pred_metrics,
                          'noise_floor':t['noise_only_lower_bound_on_upper_effect_um']},ensure_ascii=False),flush=True)
    dump(OUT/'diagnosis.json',result)
    shutil_path = OUT/Path(__file__).name
    shutil_path.write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':
    main()
