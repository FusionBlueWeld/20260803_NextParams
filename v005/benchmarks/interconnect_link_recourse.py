"""Separate state-observed recourse and physical-reference diagnostics."""
import copy
from interconnect_link_validation import *

def run():
    result = {'cases': [], 'reference_anchor_sweeps': []}
    for seed in (42, 0, 1):
        controls, _ = recipe(seed)
        actual = truth(controls)
        conf = json.loads((DEST/f'seed_{seed}/linked_pipeline.json').read_text(encoding='utf-8'))
        models = {s: load_stage_bundle(DEST/f'seed_{seed}/linked_bundles/{s}') for s in STAGE_ORDER}
        conf['completed_stages'] = list(STAGE_ORDER[:4])
        conf['fixed_controls'] = {k:v for k,v in controls.items() if k.split('.')[0] in STAGE_ORDER[:4]}
        conf['observed_outputs'] = {f'{s}.{k}': float(v) for s in STAGE_ORDER[:4]
                                    for k,v in actual['stage_outputs'][s].items()}
        learned_conf, learned = search(conf, models, controls, seed, 8192)
        reference_conf, reference = search(conf, oracle_predictors(), controls, seed, 8192)
        save(DEST/f'seed_{seed}/observed_prefix_pipeline.json', learned_conf)
        case = {'seed': seed, 'learned_observed_prefix': learned, 'oracle_observed_prefix_reference': reference}
        result['cases'].append(case)
        log(f'observed cut/form seed {seed}: learned={[x["actual"]["executable_pass"] for x in learned["independent_scoring"]]}, oracle={[x["actual"]["executable_pass"] for x in reference["independent_scoring"]]}')
        if seed == 0:
            anchor = reference['search']['best']['controls']
            result['reference_anchor_controls'] = anchor
            result['reference_anchor_source'] = 'PHYSICS_ORACLE_R002_NOT_V005_RECOMMENDATION'
            for dose in np.linspace(0, 1, 13):
                changed = {**anchor, 'assembly.bottom_cleaning_dose': float(dose),
                           'assembly.top_cleaning_dose': float(dose)}
                state = truth(changed)
                local_conf = copy.deepcopy(conf)
                local_conf['completed_stages'] = list(STAGE_ORDER[:-1])
                local_conf['fixed_controls'] = {k:v for k,v in changed.items() if not k.startswith('welding.')}
                local_conf['observed_outputs'] = {f'{s}.{k}': float(v) for s in STAGE_ORDER[:-1]
                                                 for k,v in state['stage_outputs'][s].items()}
                _, adjusted = search(local_conf, models, changed, 800+len(result['reference_anchor_sweeps']), 2048)
                _, oracle = search(local_conf, oracle_predictors(), changed, 800+len(result['reference_anchor_sweeps']), 2048)
                result['reference_anchor_sweeps'].append({'dose': float(dose), 'fixed_welding': state,
                    'learned_recourse': adjusted, 'oracle_recourse': oracle})
    save(DEST/'recourse.json', result)
    snapshot = DEST/'source_snapshot'/Path(__file__).resolve().relative_to(ROOT)
    snapshot.write_bytes(Path(__file__).read_bytes())
    log('recourse complete')

if __name__ == '__main__':
    run()
