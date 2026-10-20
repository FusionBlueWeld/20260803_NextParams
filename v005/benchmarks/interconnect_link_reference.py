"""Physical r002 reference: distinguishes search mechanics from learned models."""
from interconnect_link_validation import *

def run():
    result = {'full_search': [], 'sweeps': [], 'anchor_kind': 'PHYSICS_ORACLE_NOT_LEARNED_MODEL'}
    for seed in (42, 0, 1):
        controls, _ = recipe(seed)
        _, reference = search(line_config(), oracle_predictors(), controls, seed, 8192)
        result['full_search'].append({'seed': seed, 'result': reference})
        log(f'full physical reference {seed}: {[x["actual"]["executable_pass"] for x in reference["independent_scoring"]]}')
        if seed == 0:
            anchor = reference['search']['best']['controls']
    result['anchor_controls'] = anchor
    result['anchor_actual'] = truth(anchor)
    models = {s: load_stage_bundle(DEST/f'seed_0/linked_bundles/{s}') for s in STAGE_ORDER}
    for dose in np.linspace(0, 1, 13):
        controls = {**anchor, 'assembly.bottom_cleaning_dose': float(dose),
                    'assembly.top_cleaning_dose': float(dose)}
        state = truth(controls)
        conf = line_config()
        conf['completed_stages'] = list(STAGE_ORDER[:-1])
        conf['fixed_controls'] = {k:v for k,v in controls.items() if not k.startswith('welding.')}
        conf['observed_outputs'] = {f'{s}.{k}': float(v) for s in STAGE_ORDER[:-1]
                                    for k,v in state['stage_outputs'][s].items()}
        _, learned = search(conf, models, controls, 900+len(result['sweeps']), 2048)
        _, oracle = search(conf, oracle_predictors(), controls, 900+len(result['sweeps']), 2048)
        result['sweeps'].append({'dose': float(dose), 'fixed': state,
                                'learned_recourse': learned, 'oracle_recourse': oracle})
    save(DEST/'physical_reference.json', result)
    snapshot = DEST/'source_snapshot'/Path(__file__).resolve().relative_to(ROOT)
    snapshot.write_bytes(Path(__file__).read_bytes())
    log('physical reference complete')

if __name__ == '__main__':
    run()
