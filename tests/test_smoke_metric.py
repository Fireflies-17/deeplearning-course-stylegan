from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.smoke_metric import calculate


class SmokeMetricTests(unittest.TestCase):
    def test_identical_sets_have_zero_distance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Image.new("RGB", (16, 16), (0, 0, 0)).save(path / "a.png")
            Image.new("RGB", (16, 16), (255, 255, 255)).save(path / "b.png")
            result = calculate(path, path, max_images=2)
            self.assertAlmostEqual(float(result["value"]), 0.0, places=8)

    def test_different_sets_have_positive_distance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            for index in range(2):
                Image.new("RGB", (16, 16), (0, 0, 0)).save(first / "{}.png".format(index))
                Image.new("RGB", (16, 16), (255, 0, 0)).save(second / "{}.png".format(index))
            result = calculate(first, second, max_images=2)
            self.assertGreater(float(result["value"]), 0.0)


if __name__ == "__main__":
    unittest.main()
