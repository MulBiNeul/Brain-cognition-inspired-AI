"""Check selecting and inspecting a condition through the grading CLI."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.grade import grade_case


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return {"version": 1, "items": [{"id": "SG-TEST", "type": "different_letter", "rounds": [
        {"round": 1, "forbidden": "CD", "cases": [
            {"id": "c01", "input": "B" + "A"*6 + "B"*6 + "A"*6 + "C"*3 + "D"*3 + "E"*3, "target": 0},
            {"id": "c02", "input": "C" + "A"*6 + "C"*6 + "A"*6 + "B"*3 + "D"*3 + "E"*3, "target": 1},
        ]},
    ]}]}


def run_cli(*args):
    return subprocess.run([sys.executable, "-B", "-m", "src.grade", *args], cwd=ROOT,
                          capture_output=True, text=True)


class GradeCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public = json.loads((ROOT / "data/problems.json").read_text())
        report = json.loads((ROOT / "data/design_report.json").read_text())
        cls.item = cls.public["items"][0]
        cls.record = cls.item["rounds"][1]
        cls.case = cls.record["cases"][0]
        cls.reference = next(row["reference"] for row in report["case_certificates"]
                             if row["id"] == cls.item["id"] and row["round"] == cls.record["round"]
                             and row["case_id"] == cls.case["id"])

    def test_case_selection_and_three_physical_count_measures(self):
        public = fixture()
        first = grade_case(public, "SG-TEST", 1, "c01", group_counts=True)
        second = grade_case(public, "SG-TEST", 1, "c02", group_counts=True)
        self.assertEqual((first["case_id"], first["input"], first["target"]),
                         ("c01", public["items"][0]["rounds"][0]["cases"][0]["input"], 0))
        self.assertEqual(first["groupings_by_final"], {"": 1, "B": 2})
        self.assertEqual((first["total_groupings"], first["target_length_groupings"],
                          first["final_answer_types"]), (3, 1, 2))
        self.assertEqual(second["groupings_by_final"], {"": 1, "C": 2})
        self.assertEqual((second["total_groupings"], second["target_length_groupings"],
                          second["final_answer_types"]), (3, 2, 2))
        self.assertEqual(second["target"], 1)
        changed_rule = deepcopy(public)
        changed_rule["items"][0]["rounds"][0]["forbidden"] = "AB"
        changed = grade_case(changed_rule, "SG-TEST", 1, "c01", group_counts=True)
        for field in ("total_groupings", "target_length_groupings", "final_answer_types", "groupings_by_final"):
            self.assertEqual(changed[field], first[field])

    def test_actual_reference_uses_selected_case_and_previous_round_rule(self):
        score = grade_case(self.public, self.item["id"], self.record["round"], self.case["id"],
                           self.reference["moves"], self.reference["answer"])
        self.assertTrue(score["valid"])
        self.assertEqual(score["mode"], "actions_and_final")
        self.assertEqual(score["case_id"], self.case["id"])
        self.assertEqual(score["input"], self.case["input"])
        self.assertEqual(score["target"], self.case["target"])
        self.assertEqual(score["forbidden"], self.record["forbidden"])
        self.assertEqual(score["previous_forbidden"], self.item["rounds"][0]["forbidden"])
        self.assertEqual(score["final_state"], self.reference["answer"])

    def test_cli_accepts_actions_after_case_option(self):
        result = run_cli(self.item["id"], str(self.record["round"]), "--case", self.case["id"],
                         self.reference["moves"], "--answer", self.reference["answer"])
        self.assertEqual(result.returncode, 0, result.stderr)
        score = json.loads(result.stdout)
        self.assertTrue(score["valid"])
        self.assertEqual(score["mode"], "actions_and_final")
        self.assertEqual(score["case_id"], self.case["id"])

    def test_cli_supports_final_answer_only(self):
        result = run_cli(self.item["id"], str(self.record["round"]), "--case", self.case["id"],
                         "--answer", self.reference["answer"])
        self.assertEqual(result.returncode, 0, result.stderr)
        score = json.loads(result.stdout)
        self.assertTrue(score["valid"])
        self.assertEqual(score["mode"], "final_only")

    def test_cli_group_counts_reads_requested_catalog_and_case(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "problems.json").write_text(json.dumps(fixture()))
            result = run_cli("SG-TEST", "1", "--case", "c02", "--group-counts", "--data-dir", str(folder))
            self.assertEqual(result.returncode, 0, result.stderr)
            score = json.loads(result.stdout)
            self.assertEqual(score["case_id"], "c02")
            self.assertEqual(score["mode"], "physical_group_counts")
            self.assertEqual((score["total_groupings"], score["target_length_groupings"],
                              score["final_answer_types"]), (3, 2, 2))

    def test_cli_requires_case_id(self):
        result = run_cli(self.item["id"], "1", "--group-counts")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--case CASE", result.stderr)

    def test_unknown_case_is_rejected_without_selecting_a_fallback(self):
        with self.assertRaisesRegex(ValueError, "unknown case"):
            grade_case(fixture(), "SG-TEST", 1, "missing", group_counts=True)
        result = run_cli(self.item["id"], "1", "--case", "missing", "--group-counts")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown case 'missing'", result.stderr)

    def test_cli_rejects_conflicting_modes_and_validation_case(self):
        for args in ((self.item["id"], "1", "--case", "c01", "--group-counts", "--answer", ""),
                     ("--validate", "--case", "c01")):
            with self.subTest(args=args):
                result = run_cli(*args)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
