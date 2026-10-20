"""Independent measurement/audit CLI. Never imports the optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from validation.single.src.simulators import Simulator


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['measure', 'audit'])
    p.add_argument('--simulator', required=True)
    p.add_argument('--request', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sim = Simulator(ROOT / 'validation/single' / a.simulator)
    if a.action == 'measure':
        request = json.loads(a.request.read_text(encoding='utf-8'))
        conditions = request['conditions']
        points = np.array([[r[n] for n in sim.parameter_columns] for r in conditions])
        values = sim.evaluate_points(points)
        rows = sim.experiment_rows(points, values, 'measurement')
        for row, condition in zip(rows, conditions):
            row['experiment_id'] = condition['experiment_id']
        dump(a.output, {'request_sha256': hashlib.sha256(a.request.read_bytes()).hexdigest(),
                        'measurements': rows, 'simulator_provenance': sim.provenance()})
    else:
        grid = sim.grid()
        values = sim.evaluate_points(grid)
        best = sim.best_index(values)
        objective = values[sim.objective.column]
        mask = sim.feasible(values)
        optimal = mask & np.isclose(objective, objective[best], rtol=1e-9, atol=1e-10)
        dump(a.output, {'candidate_count': len(grid), 'feasible_count': int(mask.sum()),
                        'best_value': float(objective[best]), 'objective': sim.objective.column,
                        'direction': sim.objective.direction,
                        'objective_range': float(np.ptp(objective)),
                        'optimal_count': int(optimal.sum()),
                        'optimal_points': grid[optimal].tolist(),
                        'parameters': sim.parameter_columns, 'simulator_provenance': sim.provenance()})


if __name__ == '__main__':
    main()
