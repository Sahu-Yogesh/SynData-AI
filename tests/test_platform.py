import unittest
import numpy as np
import pandas as pd
from io import BytesIO

from src.data_io import read_uploaded_csv
from src.profiler import profile_dataset
from src.generation import generate_synthetic
from src.evaluation import evaluate_datasets


class ProfilerTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"age": [10, 20, 20, 100], "kind": ["a", "a", "b", None], "flag": [True, False, True, False], "constant": [1] * 4})

    def test_profile_facts_and_heuristics(self):
        result = profile_dataset(self.data, rare_threshold=.3)
        self.assertEqual(result["overview"]["rows"], 4)
        self.assertEqual(result["overview"]["duplicate_rows"], 0)
        self.assertTrue(any(x["column"] == "constant" for x in result["warnings"]))
        self.assertTrue(result["rare_values"])
        self.assertEqual({x["kind"] for x in result["columns"]}, {"numerical", "categorical", "boolean"})

    def test_pandas_categorical_dtype_is_profiled_as_categorical(self):
        data = pd.DataFrame({"kind": pd.Series(["a", "b"], dtype="category")})

        result = profile_dataset(data)

        self.assertEqual(result["columns"][0]["kind"], "categorical")

    def test_boolean_columns_with_missing_values_are_profiled_as_boolean(self):
        data = pd.DataFrame({"flag": pd.Series([True, False, pd.NA, True], dtype="boolean")})

        result = profile_dataset(data)

        self.assertEqual(result["columns"][0]["kind"], "boolean")

    def test_empty_and_invalid_input(self):
        result = profile_dataset(pd.DataFrame())
        self.assertTrue(result["overview"]["empty"])
        with self.assertRaises(TypeError): profile_dataset([1, 2])


class GenerationTests(unittest.TestCase):
    def test_custom_generation_schema_and_rows(self):
        data = pd.DataFrame({"n": [1, 2, 3], "label": ["x", "y", "x"]})
        generated, metadata = generate_synthetic(data, "Custom Generator", 20, seed=7)
        self.assertEqual(generated.shape, (20, 2))
        self.assertEqual(list(generated.columns), list(data.columns))
        self.assertEqual(metadata["actual_rows"], 20)

    def test_invalid_generation_parameters(self):
        with self.assertRaises(ValueError): generate_synthetic(pd.DataFrame({"x": [1]}), "Custom", 0)
        with self.assertRaises(ValueError): generate_synthetic(pd.DataFrame({"x": [1]}), "Unknown", 2)


class EvaluationTests(unittest.TestCase):
    def test_identical_data_has_zero_fidelity_differences(self):
        data = pd.DataFrame({"x": [1., 2., 3.], "label": ["a", "b", "a"]})
        result = evaluate_datasets(data, data.copy())
        self.assertEqual(result["schema"]["columns_match"], True)
        self.assertAlmostEqual(result["numerical_fidelity"][0]["wasserstein_distance"], 0.)
        self.assertAlmostEqual(result["categorical_fidelity"][0]["total_variation_distance"], 0.)

    def test_mismatched_schema_is_reported(self):
        real = pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})
        synthetic = pd.DataFrame({"x": [9, 10], "extra": ["q", "q"]})
        result = evaluate_datasets(real, synthetic)
        self.assertFalse(result["schema"]["columns_match"])
        self.assertIn("label", result["schema"]["missing_in_synthetic"])
        self.assertTrue(result["warnings"])


class DataIOTests(unittest.TestCase):
    def test_reads_utf8_bom_and_legacy_encoded_csv(self):
        utf8 = BytesIO("name,score\nJos\u00e9,7\n".encode("utf-8-sig"))
        legacy = BytesIO("name,score\nAndr\u00e9,8\n".encode("latin-1"))

        self.assertEqual(read_uploaded_csv(utf8).loc[0, "name"], "Jos\u00e9")
        self.assertEqual(read_uploaded_csv(legacy).loc[0, "name"], "Andr\u00e9")

    def test_rejects_empty_upload(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            read_uploaded_csv(BytesIO(b""))

    def test_rejects_missing_file_object(self):
        with self.assertRaisesRegex(TypeError, "file-like"):
            read_uploaded_csv(None)


if __name__ == "__main__": unittest.main()
