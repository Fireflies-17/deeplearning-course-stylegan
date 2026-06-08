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
        self.assertIn("setuptools", preflight.REQUIRED_PACKAGES)
        self.assertIn("wheel", preflight.REQUIRED_PACKAGES)

    def test_python_build_package_fix_mentions_setuptools_and_wheel(self) -> None:
        self.assertIn("setuptools", preflight.BUILD_PACKAGE_FIX)
        self.assertIn("wheel", preflight.BUILD_PACKAGE_FIX)

    def test_analysis_and_video_packages_are_recommended(self) -> None:
        self.assertIn("imageio", preflight.RECOMMENDED_PACKAGES)
        self.assertIn("matplotlib", preflight.RECOMMENDED_PACKAGES)


if __name__ == "__main__":
    unittest.main()
