"""独立した合成測定器。v005をimportせず、JSON条件の測定結果だけ返す。"""
import argparse
import json
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    conditions = json.loads(args.request.read_text(encoding='utf-8'))['conditions']
    measurements = []
    for row in conditions:
        x, y = row['x'], row['y']
        measurements.append({**row, 'cost': 10 + (x - .3) ** 2 + 2 * (y + .2) ** 2,
                             'quality': .2 * x + .1 * y})
    args.output.write_text(json.dumps({'measurements': measurements}, indent=2), encoding='utf-8')
    print(json.dumps({'measured_conditions': len(measurements)}))
