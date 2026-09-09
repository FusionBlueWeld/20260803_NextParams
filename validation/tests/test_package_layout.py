"""フォルダ移動で壊れやすい実行場所・import・モデル探索の回帰テスト。"""

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOT = PROJECT_ROOT / "validation"


class PackageLayoutTests(unittest.TestCase):
    def test_entry_points_work_outside_project_directory(self):
        """絶対パスで起動すれば、カレントディレクトリに依存しません。"""
        with tempfile.TemporaryDirectory() as directory:
            for kind, expected in (("single", "laser_welding"), ("multistage", "coating")):
                with self.subTest(kind=kind):
                    result = subprocess.run(
                        [sys.executable, str(VALIDATION_ROOT / kind / "run.py"), "list"],
                        cwd=directory, capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(expected, result.stdout)

    def test_settings_resolve_models_results_and_optimizer_root(self):
        for kind in ("single", "multistage"):
            settings = importlib.import_module(f"validation.{kind}.src.settings")
            self.assertEqual(settings.PROJECT_ROOT, PROJECT_ROOT)
            self.assertEqual(settings.VALIDATION_ROOT, VALIDATION_ROOT / kind)
            self.assertEqual(settings.RESULTS_ROOT, VALIDATION_ROOT / kind / "results")
        from validation.single.src.benchmark import versions
        self.assertIn("v002", versions())

    def test_all_check_modules_import_without_running_trials(self):
        """相対importを使う追加検証は、読み込むだけでは実行を開始しません。"""
        for kind in ("single", "multistage"):
            for path in sorted((VALIDATION_ROOT / kind / "src" / "checks").glob("*.py")):
                with self.subTest(module=path.stem, kind=kind):
                    importlib.import_module(f"validation.{kind}.src.checks.{path.stem}")

    def test_specialized_cli_exposes_help_as_a_module(self):
        result = subprocess.run(
            [sys.executable, "-m", "validation.single.src.checks.v003_validation", "--help"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("synthetic", result.stdout)


if __name__ == "__main__":
    unittest.main()
