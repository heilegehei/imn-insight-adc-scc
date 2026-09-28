"""Run with: python -m unittest discover -s tests -v"""
from __future__ import annotations

import hashlib
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import model_runtime as rt


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = rt.load_package()
        cls.example = rt.example_frame()

    def test_frozen_files_unchanged(self):
        manifest = rt.metadata()
        files = {"model_sha256": rt.MODEL_PATH, "frozen_runtime_sha256": rt.ROOT / "runtime/frozen_runtime.py",
                 "formal_factory_sha256": rt.ROOT / "runtime/runtime_support/formal_model_factory.py",
                 "integrated_factory_sha256": rt.ROOT / "runtime/runtime_support/integrated_model_factory.py"}
        for key, path in files.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest[key])

    def test_saved_fixture_parity_when_available(self):
        # These local source fixtures are intentionally not copied into this app.
        folder = rt.ROOT.parent
        if not (folder / "known_input_fixture.csv").exists():
            self.skipTest("Source-project fixtures are not part of the deployment bundle")
        source = pd.read_csv(folder / "known_input_fixture.csv")
        expected = pd.read_csv(folder / "known_expected_predictions.csv")
        actual = rt.score_frame(source)["results"].set_index("analysis_row_id")
        max_error = 0.0
        for row in expected.itertuples(index=False):
            layer = str(row.layer).replace("+", "_plus_")
            value = actual.loc[row.analysis_row_id, f"{layer}__adenocarcinoma_probability"]
            error = abs(float(value) - row.probability)
            max_error = max(max_error, error)
            self.assertLessEqual(error, 1e-12)
        print(f"\nSource fixture parity: {len(expected)} predictions; max absolute error {max_error:.3g}")

    def test_original_path_matches_complete_and_missing(self):
        frame = pd.concat([self.example] * 2, ignore_index=True)
        frame.loc[1, ["CRP", "MONO", "ALB", "PA"]] = np.nan
        reference = rt.frozen_runtime.predict_frame(self.package, frame)
        result = rt.score_frame(frame)["results"]
        for layer in rt.LAYERS:
            np.testing.assert_allclose(reference[f"{layer}__Stacking"], result[f"{layer}__adenocarcinoma_probability"], atol=1e-12, rtol=0)
        self.assertEqual(list(result.missing_input_count), [0, 4])

    def test_composites_and_fixed_thresholds(self):
        result = rt.score_frame(self.example)["results"].iloc[0]
        x = self.example.iloc[0]
        values = {"NLR": x.NEUT/x.LYMPH, "SIRI": x.NEUT*x.MONO/x.LYMPH, "LMR": x.LYMPH/x.MONO,
                  "PNI": x.ALB+5*x.LYMPH, "AGR": x.ALB/(x.TP-x.ALB), "GAR": x.GLU/x.ALB}
        for name, expected in values.items():
            self.assertAlmostEqual(result[name], expected, places=12)
        for layer in rt.LAYERS:
            self.assertEqual(result[f"{layer}__threshold"], rt.THRESHOLDS[layer])
            expected = "Adenocarcinoma" if result[f"{layer}__adenocarcinoma_probability"] >= rt.THRESHOLDS[layer] else "Squamous cell carcinoma"
            self.assertEqual(result[f"{layer}__class"], expected)

    def test_aliases_and_crp_boundaries(self):
        frame = self.example.rename(columns={"TC": "CHOL", "RDW_CV": "RDW-CV", "NEUT_PCT": "NEUT%"}).astype(object)
        for text, expected in (("<5", 5/np.sqrt(2)), (">120", 120), ("<1e1", 10/np.sqrt(2))):
            frame["CRP"] = text
            out, _, notice = rt.normalize_inputs(frame)
            self.assertAlmostEqual(out.CRP.iloc[0], expected)
            self.assertEqual(out.TC.iloc[0], self.example.TC.iloc[0])
            self.assertEqual(notice["crp_converted"], 1)

    def test_invalid_values(self):
        for field, value in (("MONO", 0), ("LYMPH", 0), ("ALB", 0), ("CRP", "bad"), ("UA", np.inf), ("WBC", -1), ("NEUT_PCT", 101), ("TP", 20)):
            with self.subTest(field=field, value=value):
                frame = self.example.astype(object).copy()
                frame.loc[0, field] = value
                with self.assertRaises(rt.InputError):
                    rt.score_frame(frame)
        with self.assertRaises(rt.InputError):
            rt.score_frame(pd.DataFrame([{c: None for c in rt.DIRECT_COLUMNS}]))

    def test_csv_validation_and_empty(self):
        for raw in (b"", b"CRP,CRP\n2,3\n", b"TC,CHOL\n3,3\n", b"\xff\xfe", b"WBC\n", b"WBC,NEUT\n1,2,3\n", b"WBC,NEUT\n1\n"):
            with self.subTest(raw=raw):
                with self.assertRaises(rt.InputError):
                    rt.normalize_inputs(rt.read_csv_bytes(raw))
        with self.assertRaises(rt.InputError):
            rt.read_csv_bytes(b"x" * (rt.MAX_UPLOAD_BYTES+1))
        with self.assertRaises(rt.InputError):
            rt.normalize_inputs(pd.concat([self.example]*(rt.MAX_ROWS+1), ignore_index=True))

    def test_missing_columns_batch_order_and_duplicate_ids(self):
        frame = pd.concat([self.example, self.example], ignore_index=True).drop(columns="PA")
        frame["analysis_row_id"] = ["duplicate", "duplicate"]
        frame.loc[1, "MONO"] = .65
        batch = rt.score_frame(frame)["results"]
        self.assertEqual(list(batch.analysis_row_id), ["duplicate", "duplicate"])
        self.assertEqual(list(batch.missing_input_count), [1, 1])
        for i in range(2):
            one = rt.score_frame(frame.iloc[[i]])["results"]
            for layer in rt.LAYERS:
                col = f"{layer}__adenocarcinoma_probability"
                self.assertAlmostEqual(batch.loc[i, col], one.loc[0, col], places=12)

    def test_repeated_and_concurrent_are_deterministic(self):
        frame = self.example.copy()
        frame["CRP"] = np.nan
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(lambda _: rt.score_frame(frame)["results"], range(3)))
        for value in values[1:]:
            pd.testing.assert_frame_equal(values[0], value)

    def test_export_protects_formula_ids(self):
        content = rt.export_csv(pd.DataFrame({"analysis_row_id": ["=1+1", " @SUM(A1)", "ordinary"], "p": [.2, .3, .4]})).decode("utf-8-sig")
        self.assertIn("'=1+1", content)
        self.assertIn("' @SUM(A1)", content)
        self.assertIn("ordinary", content)


if __name__ == "__main__":
    unittest.main()
