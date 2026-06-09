from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_evidence import RUNS, build_claims


class BuildEvidenceTests(unittest.TestCase):
    def test_manifest_declares_all_completed_formal_conditions(self) -> None:
        self.assertEqual(
            list(RUNS),
            ["E1", "E2", "E3", "E4", "E5", "E1b", "E2b", "E4b"],
        )

    def test_claims_keep_required_limitations(self) -> None:
        claims = {row["claim_id"]: row for row in build_claims()}
        self.assertIn("Do not claim less data is better", claims["C2"]["limitation"])
        self.assertIn("cannot establish", claims["C6"]["limitation"])
        self.assertIn("two GPUs", claims["C7"]["allowed_wording"])


if __name__ == "__main__":
    unittest.main()
