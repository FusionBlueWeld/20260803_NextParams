"""Reachable form-control perturbations and downstream recourse."""
from interconnect_link_validation import *

def run():
    anchor = json.loads((DEST/'physical_reference.json').read_text(encoding='utf-8'))['anchor_controls']
    models = {s: load_stage_bundle(DEST/f'seed_0/linked_bundles/{s}') for s in STAGE_ORDER}
    result = {'anchor_source': 'PHYSICS_ORACLE_FULL_SEARCH_SEED0', 'sweeps': []}
    for overbend in np.linspace(0, 5, 13):
        controls = {**anchor, 'form_cu.first_overbend_deg': float(overbend)}
        state = truth(controls)
        conf = line_config()
        conf['completed_stages'] = list(STAGE_ORDER[:4])
        conf['fixed_controls'] = {k:v for k,v in controls.items() if k.split('.')[0] in STAGE_ORDER[:4]}
        conf['observed_outputs'] = {f'{s}.{k}': float(v) for s in STAGE_ORDER[:4]
                                    for k,v in state['stage_outputs'][s].items()}
        _, learned = search(conf, models, controls, 1200+len(result['sweeps']), 2048)
        _, oracle = search(conf, oracle_predictors(), controls, 1200+len(result['sweeps']), 2048)
        result['sweeps'].append({'overbend_deg': float(overbend), 'fixed': state,
                                'learned_recourse': learned, 'oracle_recourse': oracle})
    save(DEST/'form_sweep.json', result)
    snapshot = DEST/'source_snapshot'/Path(__file__).resolve().relative_to(ROOT)
    snapshot.write_bytes(Path(__file__).read_bytes())
    log('form sweep complete')

if __name__ == '__main__':
    run()
