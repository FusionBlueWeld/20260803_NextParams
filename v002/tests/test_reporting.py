"""Hybrid解空間のサブフォルダ採番を確認します。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from src.hybrid.reporting import next_run_id, response_space_path  # noqa: E402


class ReportingTest(unittest.TestCase):
    def test_run_files_are_numbered_inside_response_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "response_spaces"
            root.mkdir()
            (root / "hybrid_response_space_run_0001.csv").touch()
            (root / "hybrid_response_space_run_0003.csv").touch()
            self.assertEqual(next_run_id(root), "run_0004")
            self.assertEqual(
                response_space_path(root, "run_0004").name,
                "hybrid_response_space_run_0004.csv",
            )


if __name__ == "__main__":
    unittest.main()
