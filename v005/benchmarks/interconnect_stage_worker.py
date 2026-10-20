"""Isolated electrical stage physics CLI; never runs the line or imports v005."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from validation.multistage.electrical_interconnect.stages import STAGE_MANIFESTS, default_inputs, evaluate_stage


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class StageSimulator:
    def __init__(self, stage):
        self.stage = stage
        self.manifest = STAGE_MANIFESTS[stage]
        self.parameter_columns = self.manifest['controls']
        self.output_columns = self.manifest['predicted_outputs']
        self.objective = SimpleNamespace(column=self.manifest['local_objective']['name'], direction='minimize')
        self.context = {n: v for n, v in default_inputs(stage).items() if n not in self.parameter_columns}
        if stage == 'assembly':
            self.context.update(top_material_is_cu=0., top_yield_strength_mpa=85., top_elastic_modulus_gpa=69.)

    def evaluate_points(self, points):
        inputs = {**self.context, **{n: points[:, i] for i, n in enumerate(self.parameter_columns)}}
        return evaluate_stage(self.stage, inputs)

    def grid(self):
        return np.asarray(list(itertools.product(*(self.manifest['candidate_axes'][n] for n in self.parameter_columns))))

    def feasible(self, values):
        mask = np.ones(len(values[self.objective.column]), dtype=bool)
        for c in self.manifest['local_constraints']:
            mask &= values[c['name']] <= c['target'] if c['direction'] == 'less_equal' else values[c['name']] >= c['target']
        return mask

    def provenance(self):
        files = [ROOT/'validation/multistage/electrical_interconnect/stages.py', ROOT/'validation/multistage/src/common.py', Path(__file__)]
        return {'stage': self.stage, 'fixed_incoming_context': self.context, 'material_case': 'Cu/Al',
                'context_policy': 'independent nominal fixture; no upstream or line simulation',
                'observation_regime': 'complete synthetic oracle outputs, including latent local constraints',
                'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}

    def experiment_rows(self, points, values):
        return [{'experiment_id': f'measurement_{i+1:04d}',
                 **{n: float(points[i, j]) for j, n in enumerate(self.parameter_columns)},
                 **{n: float(values[n][i]) for n in self.output_columns}} for i in range(len(points))]

    def define(self, folder):
        folder.mkdir(parents=True, exist_ok=False)
        header = ['column','display_name','unit','role','direction','lower','upper','step','target']
        rows = []
        for n in self.parameter_columns:
            axis = self.manifest['candidate_axes'][n]
            rows.append(dict(column=n, display_name=n, unit=self.manifest['input_units'][n], role='parameter',
                             lower=axis[0], upper=axis[-1], step=(axis[-1]-axis[0])/3))
        constraints = {c['name']: c for c in self.manifest['local_constraints']}
        for n in self.output_columns:
            role = 'objective' if n == self.objective.column else 'constraint' if n in constraints else 'monitor'
            row = dict(column=n, display_name=n, unit=self.manifest['output_units'][n], role=role)
            if role == 'objective': row['direction'] = 'minimize'
            elif role == 'constraint': row.update(direction=constraints[n]['direction'], target=constraints[n]['target'])
            rows.append(row)
        with (folder/'problem.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
        dump(folder/'fixture.json', self.provenance())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['define','measure','audit'])
    p.add_argument('--simulator', choices=list(STAGE_MANIFESTS), required=True)
    p.add_argument('--request', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); sim = StageSimulator(a.simulator)
    if a.action == 'define':
        sim.define(a.output); return
    if a.action == 'measure':
        request = json.loads(a.request.read_text(encoding='utf-8'))
        points = np.asarray([[r[n] for n in sim.parameter_columns] for r in request['conditions']])
        rows = sim.experiment_rows(points, sim.evaluate_points(points))
        for row, condition in zip(rows, request['conditions']): row['experiment_id'] = condition['experiment_id']
        dump(a.output, {'request_sha256': hashlib.sha256(a.request.read_bytes()).hexdigest(),
                        'measurements': rows, 'simulator_provenance': sim.provenance()})
    else:
        grid = sim.grid(); values = sim.evaluate_points(grid); mask = sim.feasible(values)
        objective = values[sim.objective.column]
        best = float(objective[mask].min()) if mask.any() else None
        optimal = mask & np.isclose(objective, best, rtol=1e-9, atol=1e-10) if best is not None else np.zeros(len(grid), dtype=bool)
        dump(a.output, {'candidate_count': len(grid), 'feasible_count': int(mask.sum()), 'best_value': best,
                        'objective': sim.objective.column, 'direction': 'minimize', 'objective_range': float(np.ptp(objective)),
                        'optimal_count': int(optimal.sum()), 'optimal_points': grid[optimal].tolist(),
                        'parameters': sim.parameter_columns, 'simulator_provenance': sim.provenance()})


if __name__ == '__main__': main()
