"""Exercise actual Streamlit form callbacks without a running browser."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_prediction_error_reset_and_session_persistence(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.number_input), 26)
        self.assertEqual(len(app.text_input), 1)
        submit_key = "FormSubmitter:prediction_form-Generate prediction"
        app.button(submit_key).click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("All laboratory values", app.error[0].value)
        app.button("example").click().run()
        self.assertAlmostEqual(app.number_input("input_PA").value, 19.6)
        app.button(submit_key).click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn("single_result", app.session_state)
        result_before = app.session_state["single_result"]["results"].copy()
        app.run()
        self.assertTrue(result_before.equals(app.session_state["single_result"]["results"]))
        app.number_input("input_TP").set_value(10.0)
        app.button(submit_key).click().run()
        self.assertIn("Total protein must exceed albumin", app.error[0].value)
        self.assertNotIn("single_result", app.session_state)
        app.button("clear").click().run()
        self.assertIsNone(app.number_input("input_PA").value)
        self.assertEqual(app.text_input("input_CRP").value, "")
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
