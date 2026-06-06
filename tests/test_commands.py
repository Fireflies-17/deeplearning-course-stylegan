from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stylegan_course.commands import (
    build_evaluate_command,
    build_generate_command,
    build_train_command,
)
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

    def test_p1_short_command_uses_lsun_baseline_controls(self) -> None:
        config = load_config("configs/baseline/p1_lsun_church256_short.json")
        command = build_train_command(config, backend_dry_run=True)
        joined = " ".join(command)
        self.assertIn("lsun-church-256-100k.zip", joined)
        self.assertIn("--cfg=paper256", joined)
        self.assertIn("--cond=false", joined)
        self.assertIn("--mirror=true", joined)
        self.assertIn("--aug=ada", joined)
        self.assertIn("--augpipe=bgc", joined)
        self.assertIn("--target=0.6", joined)
        self.assertIn("--kimg=100", joined)
        self.assertIn("--metrics=none", joined)

    def test_p1_baseline_command_keeps_report_grade_metric(self) -> None:
        config = load_config("configs/baseline/p1_lsun_church256_baseline.json")
        command = build_train_command(config)
        joined = " ".join(command)
        self.assertIn("--cfg=paper256", joined)
        self.assertIn("--cond=false", joined)
        self.assertIn("--gpus=2", joined)
        self.assertIn("--kimg=2000", joined)
        self.assertIn("--snap=25", joined)
        self.assertIn("--metrics=fid50k_full", joined)

    def test_p1_baseline_evaluate_command_requests_metric_suite(self) -> None:
        config = load_config("configs/baseline/p1_lsun_church256_baseline.json")
        command = build_evaluate_command(config, Path("network-snapshot-005000.pkl"))
        joined = " ".join(command)
        self.assertIn("--metrics=fid50k_full,kid50k_full,pr50k3_full", joined)
        self.assertIn("lsun-church-256-100k.zip", joined)
        # Offline evaluation must disable mirroring so the FID reference set matches
        # the unmirrored full distribution, regardless of the mirror used in training.
        self.assertIn("--mirror=false", joined)

    def test_p2_configs_use_fixed_budget_and_unmirrored_evaluation(self) -> None:
        p2_configs = [
            "configs/baseline/p2_lsun_church256_noada_1500.json",
            "configs/baseline/p2_lsun_church256_fixedp02_1500.json",
            "configs/baseline/p2_lsun_church256_subset50k_ada_1500.json",
            "configs/baseline/p2_lsun_church256_target04_1500.json",
            "configs/baseline/p2_lsun_church256_target08_1500.json",
        ]
        for path in p2_configs:
            config = load_config(path)
            train = " ".join(build_train_command(config))
            evaluate = " ".join(
                build_evaluate_command(config, Path("network-snapshot-001500.pkl"))
            )
            with self.subTest(config=path):
                # 1500 kimg is the fair comparison budget shared by every P2 group.
                self.assertIn("--kimg=1500", train)
                # Training metrics stay off to save budget; FID is computed offline.
                self.assertIn("--metrics=none", train)
                # Every group evaluates against the unmirrored full distribution.
                self.assertIn("--mirror=false", evaluate)

    def test_p1_generate_command_is_unconditional(self) -> None:
        config = load_config("configs/baseline/p1_lsun_church256_short.json")
        command = build_generate_command(config, Path("network-snapshot-000100.pkl"))
        joined = " ".join(command)
        self.assertNotIn("--class=", joined)
        self.assertIn("--seeds=0-63", joined)

    def test_generate_class_override_is_supported(self) -> None:
        config = load_config("configs/baseline/p1_lsun_church256_short.json")
        command = build_generate_command(
            config,
            Path("network-snapshot-000100.pkl"),
            class_idx=7,
        )
        joined = " ".join(command)
        self.assertIn("--class=7", joined)

    def test_unknown_train_option_is_rejected(self) -> None:
        config = load_config("configs/baseline/p0_smoke.json")
        config["train"]["not_an_official_option"] = True
        with self.assertRaises(ValueError):
            build_train_command(config)


if __name__ == "__main__":
    unittest.main()
