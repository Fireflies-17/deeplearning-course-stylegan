from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_dataset_provenance import parse_history, redact_command


class DatasetProvenanceTests(unittest.TestCase):
    def test_history_timestamp_is_attached_to_next_command(self) -> None:
        rows = parse_history(
            [
                "#1780790400\n",
                "wget https://example.com/church.zip?token=secret\n",
                "echo ignored\n",
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["timestamp_utc"])
        self.assertIn("<redacted-query>", rows[0]["command"])
        self.assertNotIn("secret", rows[0]["command"])

    def test_secret_options_and_authorization_are_redacted(self) -> None:
        command = (
            "curl -H 'Authorization: Bearer abc' "
            "--token xyz https://example.com/file.zip?a=b"
        )
        redacted = redact_command(command)
        self.assertNotIn("abc", redacted)
        self.assertNotIn("xyz", redacted)
        self.assertNotIn("a=b", redacted)

    def test_lsun_training_command_is_not_download_provenance(self) -> None:
        rows = parse_history(
            [
                "python scripts/run_experiment.py train "
                "--config configs/baseline/p1_lsun_church256_baseline.json\n"
            ]
        )
        self.assertEqual(rows, [])

    def test_unrelated_curl_command_is_not_download_provenance(self) -> None:
        rows = parse_history(["curl https://example.com/metrics.json\n"])
        self.assertEqual(rows, [])

    def test_zip_fixture_can_be_created_for_target_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("dataset.json", '{"labels": null}')
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
