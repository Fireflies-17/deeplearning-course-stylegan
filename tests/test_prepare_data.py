from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import validate_dataset_zip


def image_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class PrepareDataTests(unittest.TestCase):
    def test_existing_dataset_zip_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("dataset.json", json.dumps({"labels": None}))
                archive.writestr("00000/img00000000.png", image_bytes((16, 16)))
                archive.writestr("00000/img00000001.png", image_bytes((16, 16)))

            validate_dataset_zip(path, expected_count=2, expected_resolution=(16, 16))

    def test_invalid_existing_dataset_zip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("00000/img00000000.png", image_bytes((16, 16)))

            with self.assertRaisesRegex(RuntimeError, "dataset.json is missing"):
                validate_dataset_zip(path)


if __name__ == "__main__":
    unittest.main()
