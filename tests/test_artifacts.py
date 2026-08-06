import tempfile
import unittest
from pathlib import Path

from artifacts import read_model_name


class DummyModel:
    pass


class ReadModelNameTest(unittest.TestCase):
    def test_reads_model_name_from_file_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best_model_name.txt"
            path.write_text("XGBoost\n", encoding="utf-8")

            self.assertEqual(read_model_name(DummyModel(), path), "XGBoost")

    def test_falls_back_to_model_class_when_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "best_model_name.txt"

            self.assertEqual(read_model_name(DummyModel(), missing_path), "DummyModel")


if __name__ == "__main__":
    unittest.main()
