"""128/256/512点と追加実験後の学習・予測費用を記録する。精度保証とは別。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import time
import tracemalloc
import numpy as np
from v005.src.stage_bundle import train_stage_bundle, load_stage_bundle


def run():
    root = Path(__file__).resolve().parents[1] / "output"
    root.mkdir(exist_ok=True)
    names = [f"u{i}" for i in range(8)] + [f"c{i}" for i in range(8)]
    output_names = [f"y{i}" for i in range(6)]
    manifest = {"bundle_schema_version": "2.0", "stage_id": "cost_benchmark", "controls": names[:8],
                "incoming_context": names[8:], "predicted_outputs": output_names,
                "input_bounds": {name: [0, 1] for name in names}, "input_units": {name: "" for name in names},
                "output_units": {name: "" for name in output_names},
                "local_objective": {"name": "y0", "direction": "maximize"}, "local_constraints": []}
    rng = np.random.default_rng(14)
    x = rng.random((528, 16))
    y = np.column_stack([x[:, i] + x[:, i + 8] ** 2 + 0.3 * np.sin(x[:, i + 2] * 5)
                         + x[:, i + 3] * x[:, i + 9] for i in range(6)])
    query = rng.random((4096, 16))
    results = []
    with tempfile.TemporaryDirectory(prefix=".stage_cost_", dir=root) as temporary:
        path = Path(temporary)
        if path.resolve().parent != root.resolve():
            raise RuntimeError("benchmarkの一時directoryがworkspace外です。")
        for rows, cap in ((128, 128), (256, 256), (512, 512), (528, 512)):
            tracemalloc.start()
            started = time.perf_counter()
            train_stage_bundle(path, manifest, {name: x[:rows, i] for i, name in enumerate(names)},
                               {name: y[:rows, i] for i, name in enumerate(output_names)},
                               seed=0, max_training_samples=cap)
            saved = time.perf_counter()
            _, predictor = load_stage_bundle(path)
            loaded = time.perf_counter()
            predictor.predict_arrays({name: query[:, i] for i, name in enumerate(names)})
            ended = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results.append({"source_samples": rows, "cap": cap, "added_experiments": max(rows - 512, 0),
                            "fit_and_save_seconds": saved - started, "load_seconds": loaded - saved,
                            "predict_4096_seconds": ended - loaded, "traced_peak_bytes": peak})
    evidence = {"python": sys.version, "numpy": np.__version__, "input_axes": 16, "output_axes": 6,
                "predictions": 4096, "seed": 14, "memory_kind": "tracemalloc_peak_not_process_RSS",
                "accuracy_or_physics_validation": False, "results": results}
    destination = root / "stage_model_cost.json"
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return evidence


if __name__ == "__main__":
    run()
