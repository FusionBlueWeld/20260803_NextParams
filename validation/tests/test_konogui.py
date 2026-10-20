"""GUI表示と計算APIの接続、検査マスク、非同期処理を検証します。"""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from validation import konogui as gui
from validation.gui_interconnect import electrical_specs


class GuiTests(unittest.TestCase):
    def test_existing_and_new_choices(self):
        specs = gui.load_specs()
        self.assertEqual(len(specs), 19)
        self.assertEqual([item.id for item in specs[:3]],
                         ["interconnect_cu_al", "interconnect_cu_cu", "interconnect_al_al"])
        gui.smoke_test(specs)

    def test_physics_runs_without_learning_or_linked_models(self):
        # 新しいプロセスで最適化パッケージのimportを禁止し、依存の混入を検出します。
        program = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'v001', 'v002', 'v003', 'v004', 'v005', 'r001', 'r002'}:
        raise AssertionError('物理GUIが最適化模型へ依存しています: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from validation.konogui import load_specs, smoke_test
smoke_test(load_specs())
"""
        result = subprocess.run([sys.executable, "-c", program],
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inspection_hides_latent_and_destructive_outputs(self):
        from validation.multistage.electrical_interconnect.stages import STAGE_MANIFESTS
        for spec in electrical_specs()[:6]:
            values = {field.name: field.initial_value() for field in spec.inputs}
            result = gui.evaluate_display(spec, values, True, np.random.default_rng(2))
            stage = spec.id.removeprefix("interconnect_")
            policy = STAGE_MANIFESTS[stage]["observation_policy"]
            for name in [*policy["latent"], *policy["destructive"]]:
                self.assertIsNone(result[name])
            for name in policy["nondestructive"]:
                self.assertTrue(np.isfinite(result[name]))
            self.assertEqual(set(result), {field.name for field in spec.outputs})
            self.assertEqual(np.asarray(result["local_feasible"]).dtype.kind, "b")

    def test_upstream_change_reaches_downstream(self):
        spec = next(item for item in electrical_specs() if item.id == "interconnect_cu_al")
        values = {field.name: field.initial_value() for field in spec.inputs}
        first = spec.evaluate(values)
        values["cut_cu.stock_thickness_mm"] *= 1.05
        second = spec.evaluate(values)
        self.assertNotEqual(float(first["welding.base_path_resistance_uohm"]),
                            float(second["welding.base_path_resistance_uohm"]))

    def test_hidden_tk_dropdown_and_async_physics(self):
        # 実Tkウィジェットを非表示で構築し、計算中もイベントが処理されることを確認。
        spec = electrical_specs()[-3]
        from dataclasses import replace
        original_evaluate = spec.evaluate
        def delayed_evaluate(values):
            time.sleep(0.08)
            return original_evaluate(values)
        spec = replace(spec, evaluate=delayed_evaluate)
        errors = []
        dialog = patch.object(gui.messagebox, "showerror", side_effect=lambda *args, **kwargs: errors.append(args))
        dialog.start()
        self.addCleanup(dialog.stop)
        specs = tuple([spec, *(item for item in gui.load_specs() if item.id != spec.id)])
        app = gui.ValidationGui(specs)
        app.withdraw()
        self.addCleanup(app.destroy)
        app.update()
        self.assertEqual(int(app.selector["height"]), 19)
        self.assertEqual(tuple(app.selector["values"]), tuple(item.label for item in specs))
        self.assertFalse(hasattr(app, "bundle_button"))
        self.assertFalse(hasattr(app, "pipeline_button"))
        self.assertFalse(hasattr(app, "search_button"))
        for index, item in enumerate(app.specs):
            app.selector.current(index)
            app.selector.event_generate("<<ComboboxSelected>>")
            app.update()
            self.assertEqual(set(app.input_vars), {field.name for field in item.inputs})
            self.assertEqual(set(app.output_vars), {field.name for field in item.outputs})
        app._select(spec.label)
        initial_inputs = app._read_inputs()
        pulse = []
        app.after(0, lambda: pulse.append(True))
        with patch.object(gui, "PROCESS_DURATION_MS", 0):
            app._run()
            deadline = time.monotonic() + 5
            while app._pending_values is not None and time.monotonic() < deadline:
                app.update()
                time.sleep(0.01)
        self.assertIsNone(app._pending_values)
        self.assertEqual(errors, [])
        self.assertTrue(pulse)
        self.assertEqual(app._last_result["mode"], "physical_simulation")
        self.assertEqual(app._read_inputs(), initial_inputs)
        expected = original_evaluate(initial_inputs)
        self.assertEqual(app.output_vars["welding.shear_force_n"].get(),
                         app._format_output(expected["welding.shear_force_n"]))
        self.assertEqual(str(app.save_button["state"]), "normal")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            destination = Path(directory) / "result.json"
            with patch.object(gui.filedialog, "asksaveasfilename", return_value=str(destination)):
                app._save_result()
            saved = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(saved["simulator"], "interconnect_cu_al")
            self.assertEqual(saved["values"], initial_inputs)
            self.assertNotIn("raw_result", saved)
            self.assertIn("requested_inputs", saved)
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
