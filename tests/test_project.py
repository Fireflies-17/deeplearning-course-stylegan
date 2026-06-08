from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.project import find_latest_snapshot


class ProjectTests(unittest.TestCase):
    def test_latest_snapshot_uses_kimg_not_copy_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            high = outdir / "network-snapshot-002000.pkl"
            low = outdir / "network-snapshot-001500.pkl"
            high.touch()
            time.sleep(0.01)
            low.touch()
            now = time.time()
            os.utime(high, (now - 60, now - 60))
            os.utime(low, (now, now))

            chosen = find_latest_snapshot({"outdir": str(outdir)})
            self.assertEqual(chosen, high)


if __name__ == "__main__":
    unittest.main()
