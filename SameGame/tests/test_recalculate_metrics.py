"""Saved model responses can be rescored without changing evidence or progression."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.recalculate_metrics import recalculate
from src.evaluate import EvaluationSession


class RecalculationTests(unittest.TestCase):
    def saved_run(self, directory):
        items = [{"id": "demo", "type": "same_letter", "rounds": [
            {"round": 1, "input": "AABB", "target": 0, "forbidden": "BA"},
            {"round": 2, "input": "AABBAABBAA", "target": 0, "forbidden": "AA"},
        ]}]
        settings = {"max_attempts": 10, "required_streak": 2}
        session = EvaluationSession(items, **settings)
        responses = [json.dumps({"moves": move, "answer": ""})
                     for move in ("AB", "AB", "AAAB", "BBA", "BBA")]
        for raw in responses:
            session.submit(raw)
        old_report = deepcopy(session.report())
        old_report["summary"]["mpe"] = 1 / 3
        old_report["summary"]["persistent_error_basis"] = "current_rule_violation"
        directory.mkdir()
        for name, value in (("protocol.json", {"items": items, "settings": settings}),
                            ("checkpoint.json", {"responses": responses}), ("report.json", old_report)):
            (directory / name).write_text(json.dumps(value))
        return {p.name: sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}

    def test_recalculation_preserves_original_and_exposes_denominator_change(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            before = self.saved_run(directory)
            output = directory / "analysis"
            comparison = recalculate(directory, output)
            self.assertEqual(comparison["model_calls"], 0)
            self.assertEqual(comparison["persistent_errors"], 1)
            self.assertEqual(comparison["eligible_submissions"], 3)
            self.assertEqual(comparison["recalculated_summary"]["mpe"], .1)
            self.assertEqual(comparison["recalculated_summary"]["observed_mpe"], 1 / 3)
            self.assertEqual(comparison["original_summary"]["mpe"], 1 / 3)
            self.assertTrue(comparison["current_correctness_feedback_and_progress_identical"])
            self.assertEqual(before, {name: sha256((directory / name).read_bytes()).hexdigest() for name in before})
            self.assertTrue((output / "report.json").exists())
            with self.assertRaises(ValueError):
                recalculate(directory, output)

    def test_source_output_and_changed_feedback_are_rejected(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            self.saved_run(directory)
            with self.assertRaises(ValueError):
                recalculate(directory, directory)
            original = json.loads((directory / "report.json").read_text())
            original["submissions"][0]["feedback"] = "Different historical feedback"
            (directory / "report.json").write_text(json.dumps(original))
            output = directory / "analysis"
            with self.assertRaises(ValueError):
                recalculate(directory, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
