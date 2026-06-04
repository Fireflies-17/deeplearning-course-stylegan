from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.commands import build_train_command
from stylegan_course.project import load_config


class CommandTests(unittest.TestCase):
    def test_p0_train_command_contains_controlled_settings(self) -> None:
        config = load_config("configs/baseline/p0_smoke.json")
        command = build_train_command(config, backend_dry_run=True)
        joined = " ".join(command)
        self.assertIn("--gpus=1", joined)
        self.assertIn("--kimg=1", joined)
        self.assertIn("--batch=32", joined)
        self.assertIn("--metrics=none", joined)
        self.assertIn("--augpipe=blit", joined)
        self.assertNotIn("--desc=", joined)
        self.assertIn("--dry-run", command)

    def test_unknown_train_option_is_rejected(self) -> None:
        config = load_config("configs/baseline/p0_smoke.json")
        config["train"]["not_an_official_option"] = True
        with self.assertRaises(ValueError):
            build_train_command(config)


if __name__ == "__main__":
    unittest.main()
