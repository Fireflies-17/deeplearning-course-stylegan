from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import preflight


class DependencyTests(unittest.TestCase):
    def test_official_backend_import_dependencies_are_required(self) -> None:
        self.assertIn("scipy", preflight.REQUIRED_PACKAGES)


if __name__ == "__main__":
    unittest.main()
