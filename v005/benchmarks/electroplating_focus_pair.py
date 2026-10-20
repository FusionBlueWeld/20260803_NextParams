"""Same-initial-data, same-batch comparison of focus ON/OFF; separate physics CLI."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from v005.benchmarks.single_arrival_validation import command, csvread, csvwrite, dump, read, digest

SOURCE = ROOT/'v005/trials/trial_single_arrival_electroplating_s42_20261010_focus_on'
OUT = ROOT/'v005/trials/_electroplating_focus_diagnosis_20261010'
WORKER = ROOT/'v005/benchmarks/single_arrival_worker.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['on','off'],required=True)
    args = parser.parse_args()
    name = f'trial_plating_focus_{args.arm}_batch3_s42_20261010'
    trial = ROOT/'v005/trials'/name
    assert not trial.exists()
    for attempt in range(3):
        try:
            command(['main.py','--version','v005','--new',name],OUT/f'{name}_new_{attempt}.log')
            break
        except RuntimeError:
            if trial.exists() or attempt==2:
                raise
    for n in ['problem.csv','knowledge_constraints.csv','search_regions.csv','stop_settings.csv','search_settings.json','focus_settings.json']:
        shutil.copy2(SOURCE/n,trial/n)
    focus = read(trial/'focus_settings.json')
    focus.update(parameter_focus_enabled=args.arm=='on',temporary_fix_enabled=args.arm=='on',
                 effect_tolerances={'thickness_error_um':.01,'roughness_ra_um':.05,'current_efficiency':.02})
    dump(trial/'focus_settings.json',focus)
    command(['main.py','--version','v005','--prepare',name],trial/'prepare.log')
    initial = SOURCE/'steps/step_048/experiments.csv'
    shutil.copy2(initial,trial/'data/experiments.csv')
    header,rows = csvread(trial/'data/experiments.csv')
    (trial/'steps').mkdir(); (trial/'exchange').mkdir()
    _, definition = csvread(trial/'problem.csv')
    params = [r['column'] for r in definition if r['role']=='parameter']
    seen = {tuple(float(r[n]) for n in params) for r in rows}
    protocol = {'arm':args.arm,'seed':42,'batch_maximum':3,'max_total_experiments':150,
                'initial_measurements':len(rows),'initial_snapshot_sha256':digest(initial),
                'focus_settings':focus,'hypothetical_effect_tolerances_not_product_specifications':True,
                'optimizer_sha256':read(SOURCE/'protocol.json')['optimizer_source_sha256'],
                'oracle_policy':'audit after termination, never passed to optimizer',
                'comparison_scope':'one seed, ON/OFF differ only in parameter_focus_enabled and temporary_fix_enabled'}
    dump(trial/'protocol.json',protocol)
    for n,h in protocol['optimizer_sha256'].items():
        assert digest(ROOT/n)==h
    good = lambda r: float(r['roughness_ra_um'])<=.65 and float(r['current_efficiency'])>=.8
    def ledger_entry(step):
        valid = [float(r['thickness_error_um']) for r in rows if good(r)]
        return {'step':step,'total_measurements':len(rows),'best_value':min(valid) if valid else None}
    ledger=[ledger_entry(0)];roles=Counter()
    for step in range(103):
        snapshot=trial/'steps'/f'step_{step:03d}'; snapshot.mkdir()
        shutil.copy2(trial/'data/experiments.csv',snapshot/'experiments.csv')
        command(['main.py','--version','v005','--run',name,'--n','3'],snapshot/'v005_run.log')
        status=read(trial/'output/stopping_status.json')
        for n in ['recommendations.csv','stopping_status.json','search_budget.json','parameter_focus.json','initial_design_status.json']:
            if (trial/'output'/n).exists():shutil.copy2(trial/'output'/n,snapshot/n)
        if status['status']!='CONTINUE':break
        _,recs=csvread(trial/'output/recommendations.csv')
        assert 1<=len(recs)<=3
        conditions=[]
        for rank,r in enumerate(recs,1):
            condition={'experiment_id':f'step_{step+1:03d}_rank_{rank}',**{n:float(r[n]) for n in params}}
            key=tuple(condition[n] for n in params)
            assert key not in seen;seen.add(key)
            conditions.append(condition);roles[r['selection_role']]+=1
        request=trial/'exchange'/f'request_{step+1:03d}.json'
        response=trial/'exchange'/f'measurement_{step+1:03d}.json'
        dump(request,{'conditions':conditions,'recommendation_sha256':digest(snapshot/'recommendations.csv')})
        command([WORKER,'measure','--simulator','electroplating','--request',request,'--output',response],snapshot/'physics.log')
        measured=read(response)
        assert measured['request_sha256']==digest(request)
        assert len(measured['measurements'])==len(conditions)
        for condition,actual in zip(conditions,measured['measurements']):
            assert all(actual[n]==v for n,v in condition.items())
            rows.append(actual)
        csvwrite(trial/'data/experiments.csv',header,rows)
        entry=ledger_entry(step+1);entry.update(request_sha256=digest(request),measurement_sha256=digest(response))
        ledger.append(entry);dump(trial/'ledger.json',ledger)
        print(json.dumps({'arm':args.arm,'measurements':len(rows),'best':entry['best_value'],'roles':dict(roles)}),flush=True)
    assert len(rows)<=150 and len(seen)==len(rows)
    command([WORKER,'audit','--simulator','electroplating','--output',trial/'posthoc_grid_audit.json'],trial/'audit.log')
    audit=read(trial/'posthoc_grid_audit.json')
    snapshots=[read(p) for p in (trial/'output/parameter_focus').glob('*.json')]
    summary={'trial':name,'arm':args.arm,'initial':48,'measurements':len(rows),
             'best_error_um':ledger[-1]['best_value'],'roles':dict(roles),
             'model_runs':len(list((trial/'output/response_spaces').glob('*.csv'))),
             'focus_assessment_runs':len(snapshots),'runs_with_fixed_inputs':sum(bool(s['fixed_values']) for s in snapshots),
             'runs_with_reweighted_inputs':sum(any(p['weight']<1 for p in s['parameters']) for s in snapshots),
             'stop_status':status['status'],'stop_reasons':status['reasons'],
             'grid_optimum_um':audit['best_value'],
             'source_initial_preserved':digest(initial)==protocol['initial_snapshot_sha256']}
    dump(trial/'summary.json',summary)
    for n,h in protocol['optimizer_sha256'].items():assert digest(ROOT/n)==h
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
