"""Check generated case pools and replay their evidence with the shared scorer."""
from collections import Counter
import json
from pathlib import Path
import random
import unittest

from src.dataset_checks import mechanical_shape
from src.evaluate import grade_response
from src.grade import read_round
from src.session import ExperimentSession
from src.dataset import _digest, _make_case, _seed_structures, design, rule_schedule, validate

DATA = Path(__file__).resolve().parents[1] / "data"


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public = json.loads((DATA / "problems.json").read_text())
        cls.key = json.loads((DATA / "answer_key.json").read_text())
        cls.report = json.loads((DATA / "design_report.json").read_text())
        cls.certificates = {(row["id"], row["round"], row["case_id"]): row
                            for row in cls.report["case_certificates"]}

    def test_saved_report_matches_exhaustive_validation(self):
        actual = validate(self.public, self.key)
        self.assertEqual(json.loads(json.dumps(actual)), self.report)
        self.assertEqual(actual["transition_cases_with_persistent_and_ordinary_errors"], 2400)
        self.assertEqual(actual["same_letter_cases_without_alternating_goal_paths"], 1500)
        for certificate in actual["case_certificates"]:
            self.assertGreaterEqual(certificate["minimum_merged_deletions"], 2)
            self.assertGreaterEqual(certificate["minimum_singleton_merge_deletions"], 1)
            if certificate["round"] > 1:
                baseline = certificate["uniform_physical_group_baseline"]
                self.assertGreaterEqual(baseline["persistent_error_given_incorrect"], .05 - 1e-12)
                self.assertGreaterEqual(baseline["ordinary_error_given_incorrect"], .05 - 1e-12)

    def test_fifty_balanced_items_have_three_fresh_cases_per_goal(self):
        self.assertEqual(self.public["version"], 1)
        self.assertEqual(len(self.public["items"]), 50)
        self.assertEqual(Counter(i["type"] for i in self.public["items"]),
                         {"different_letter": 25, "same_letter": 25})
        shapes = Counter()
        _, seeds = _seed_structures()
        for item in self.public["items"]:
            previous = None
            self.assertEqual(len(item["rounds"]), 5)
            for number, row in enumerate(item["rounds"], 1):
                self.assertNotEqual(row["forbidden"], previous)
                self.assertEqual(len(row["cases"]), 12)
                self.assertEqual(Counter(c["target"] for c in row["cases"]), {t: 3 for t in range(4)})
                self.assertEqual({c["id"] for c in row["cases"]}, {f"c{n:02d}" for n in range(1, 13)})
                local_shapes = set()
                for case in row["cases"]:
                    read_round({"round": number, "input": case["input"], "target": case["target"],
                                "forbidden": row["forbidden"]}, number, item["type"])
                    shape = mechanical_shape(case["input"])
                    self.assertNotIn(shape, local_shapes)
                    local_shapes.add(shape)
                    shapes[shape] += 1
                    if item["type"] == "same_letter":
                        self.assertNotIn(shape, seeds)
                previous = row["forbidden"]
        self.assertEqual(sum(shapes.values()), 3000)
        self.assertLessEqual(max(shapes.values()), 2)
        self.assertEqual(self.report["unique_mechanical_structures"], len(shapes))
        self.assertEqual(self.report["dataset_sha256"], _digest({"problems": self.public, "answer_key": self.key}))

    def test_all_reference_and_error_witnesses_use_the_shared_definition(self):
        records = {(item["id"], row["round"], c["id"]):
                   {"round": row["round"], "forbidden": row["forbidden"], "input": c["input"], "target": c["target"]}
                   for item in self.public["items"] for row in item["rounds"] for c in row["cases"]}
        self.assertEqual(len(self.report["case_certificates"]), 3000)
        for cert in self.report["case_certificates"]:
            location = cert["id"], cert["round"], cert["case_id"]
            for name in ("reference", "alternative_goal", "current_rule_violation",
                         "persistent_error", "ordinary_error"):
                witness = cert[name]
                if witness is None:
                    self.assertEqual(cert["round"], 1)
                    self.assertIn(name, ("persistent_error", "ordinary_error"))
                    continue
                with self.subTest(location=location, kind=name):
                    score = grade_response(json.dumps({k: witness[k] for k in ("moves", "answer")}),
                                           records[location], cert["previous_forbidden"])
                    self.assertTrue(score["base_valid"] and score["board_exhausted"] and score["final_matches"])
                    self.assertEqual(score["correct"], name == "reference")
                    if name in ("reference", "persistent_error", "ordinary_error"):
                        self.assertEqual(bool(score["persistent_error"]), name == "persistent_error")
                    elif name == "alternative_goal":
                        self.assertFalse(score["goal_reached"])
                        self.assertFalse(score["violation_steps"])
                    elif name == "current_rule_violation":
                        self.assertTrue(score["goal_reached"])
                        self.assertTrue(score["violation_steps"])

    def test_correct_witnesses_finish_all_problems_with_zero_mpe(self):
        session = ExperimentSession(self.public["items"], seed=self.public["seed"],
                                    max_attempts=10, required_streak=2)
        while not session.complete:
            record = session.current_record
            location = session.current_item["id"], record["round"], record["case_id"]
            row = session.submit(json.dumps(self.certificates[location]["reference"]))
            self.assertTrue(row["correct"], location)
            self.assertFalse(row["persistent_error"], location)
        report = session.report()
        self.assertEqual(len(report["submissions"]), 500)
        self.assertEqual(report["summary"]["observed_transition_rounds"], 200)
        self.assertEqual(report["summary"]["mpe_eligible_problems"], 50)
        self.assertEqual(report["summary"]["mpe"], 0)

    def test_one_persistent_error_per_transition_uses_fixed_attempt_budget(self):
        session = ExperimentSession(self.public["items"][:1], seed=self.public["seed"],
                                    max_attempts=10, required_streak=2)
        errored_rounds = set()
        while not session.complete:
            record = session.current_record
            location = session.current_item["id"], record["round"], record["case_id"]
            persistent = record["round"] > 1 and record["round"] not in errored_rounds
            name = "persistent_error" if persistent else "reference"
            witness = self.certificates[location][name]
            raw = json.dumps({field: witness[field] for field in ("moves", "answer")})
            row = session.submit(raw)
            self.assertEqual(bool(row["persistent_error"]), persistent)
            if persistent:
                errored_rounds.add(location[1])
            else:
                self.assertTrue(row["correct"])
        summary = session.report()["summary"]
        self.assertEqual(summary["mpe"], .1)
        self.assertAlmostEqual(summary["observed_mpe"], 1 / 3)

    def test_nonadjacent_repeats_and_two_endpoint_changes_are_supported(self):
        schedules = {kind: [rule_schedule(random.Random(seed), kind) for seed in range(100)]
                     for kind in ("different_letter", "same_letter")}
        for rows in schedules.values():
            self.assertTrue(all(all(a != b for a, b in zip(row, row[1:])) for row in rows))
            self.assertTrue(any(len(set(row)) < 5 for row in rows))
        self.assertTrue(any(all(x != y for x, y in zip(a, b))
                            for row in schedules["different_letter"] for a, b in zip(row, row[1:])))
        self.assertGreater(self.report["nonadjacent_rule_repeat_items"], 0)
        self.assertGreater(self.report["different_letter_transitions_changing_both_endpoints"], 0)

    def test_seed_changes_structure_and_same_seed_is_reproducible(self):
        job = (12345, "SG-001", "same_letter", 2, "CC", "BB", 1, 1, 0)
        first = _make_case(job)
        self.assertEqual(first, _make_case(job))
        other = _make_case((12346, *job[1:]))
        self.assertNotEqual(mechanical_shape(first["case"]["input"]), mechanical_shape(other["case"]["input"]))

    def test_three_hundred_item_design_preserves_the_initial_fifty(self):
        small, large = design(count=50), design(count=300)
        self.assertEqual(small["items"], large["items"][:50])
        self.assertEqual(Counter(i["type"] for i in large["items"]),
                         {"different_letter": 150, "same_letter": 150})


if __name__ == "__main__":
    unittest.main()
