"""Regression cases for eliminating the all-letter alternation shortcut."""

import json
from pathlib import Path
import unittest

from src.evaluate import EvaluationSession, grade_response
from src.grade import Game
from src.same_letter_dataset import repeat_certificate
from scripts.build_review import transition_metadata


class RepeatCertificateTests(unittest.TestCase):
    def check_certificate(self, tape, target, forbidden, expected):
        result = repeat_certificate(tape, target, forbidden)
        self.assertEqual(result["minimum_same_letter_transitions"], expected)
        self.assertEqual(result["alternating_solution_exists"], expected == 0)
        if expected is None:
            self.assertIsNone(result["witness"])
        else:
            witness = result["witness"]
            grade = Game(tape, target).grade(witness, forbidden)
            self.assertTrue(grade["valid"])
            letters = [move[0] for move in witness.split()]
            self.assertEqual(sum(a == b for a, b in zip(letters, letters[1:])), expected)
        return result

    def test_dependency_requires_repeat_even_with_balanced_letters(self):
        # Removing B before both A groups leaves a single B and misses target zero.
        self.check_certificate("BAABAAB", 0, "BB", 1)
        self.check_certificate("EDCBAABAABCDE", 0, "BB", 1)

    def test_spaces_keep_groups_separate_and_allow_same_letter_final(self):
        self.check_certificate("ABB ABB", 2, "AA", 1)
        self.check_certificate("ABBABB", 2, "AA", None)

    def test_weak_board_is_rejected_even_when_some_solutions_repeat(self):
        # A1 A1 B1 repeats, but B1 A1 clears it without any repeated letter.
        self.check_certificate("AABBAA", 0, "CC", 0)
        self.check_certificate("ABBBA", 0, "AA", 0)

    def test_no_current_rule_solution_is_distinct_from_forced_repeat(self):
        self.check_certificate("BAABAAB", 0, "AA", None)


class CommonReferenceReviewTests(unittest.TestCase):
    def test_revision_allows_common_correct_answer_but_requires_old_only_probe(self):
        record = {"round": 2, "input": "EDAADCDBAABAABDCDAADEDAAD", "target": 0, "forbidden": "DD"}
        metadata = transition_metadata(record, "", "BB", minimum_recoverable=1,
                                       allow_common_reference=True)
        self.assertIsNone(metadata["examples"]["currentOnly"])
        current = metadata["examples"]["correct"]
        probe = metadata["examples"]["previousOnly"]
        game = Game(record["input"], 0)
        self.assertTrue(game.grade(current["moves"], "DD", "BB", final_answer=current["answer"])["valid"])
        self.assertTrue(game.grade(probe["moves"], "DD", "BB", final_answer=probe["answer"])["previous_only"])
        self.assertGreaterEqual(len(metadata["recoverableSteps"]), 1)
        # Original variant callers retain their stronger reference condition.
        with self.assertRaises(ValueError):
            transition_metadata(record, "", "BB")


class RevisedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = Path(__file__).resolve().parents[1] / "data"
        cls.original = json.loads((data / "easy_all" / "problems.json").read_text())
        cls.revised = json.loads((data / "easy_same_letter_v2" / "problems.json").read_text())
        cls.subset = json.loads((data / "easy_same_letter_v2" / "same_letter" / "problems.json").read_text())

    def test_subset_and_unchanged_conditions(self):
        old, new = self.original["items"], self.revised["items"]
        self.assertEqual(len(new), 50)
        self.assertEqual(self.subset["items"], [item for item in new if item["type"] == "same_letter"])
        self.assertEqual(len(self.subset["items"]), 25)
        for before, after in zip(old, new):
            self.assertEqual((before["id"], before["type"]), (after["id"], after["type"]))
            if after["type"] == "different_letter":
                self.assertEqual(before, after)
                continue
            for left, right in zip(before["rounds"], after["rounds"]):
                self.assertEqual({k: v for k, v in left.items() if k != "input"},
                                 {k: v for k, v in right.items() if k != "input"})
                self.assertNotEqual(left["input"], right["input"])
                self.assertEqual(len(left["input"].replace(" ", "")), len(right["input"].replace(" ", "")))
                self.assertEqual(left["input"].count(" "), right["input"].count(" "))

    def test_every_round_requires_repeat_and_passes_shared_evaluation(self):
        session = EvaluationSession(self.subset["items"], max_attempts=10, required_streak=2)
        count = 0
        for item in self.subset["items"]:
            for index, record in enumerate(item["rounds"]):
                certificate = repeat_certificate(record["input"], record["target"], record["forbidden"])
                self.assertIsNotNone(certificate["minimum_same_letter_transitions"])
                self.assertGreaterEqual(certificate["minimum_same_letter_transitions"], 1)
                self.assertFalse(certificate["alternating_solution_exists"])
                game = Game(record["input"], record["target"])
                graded = game.grade(certificate["witness"], record["forbidden"])
                self.assertTrue(graded["valid"])
                raw = json.dumps({"moves": certificate["witness"], "answer": graded["final_state"]})
                previous = item["rounds"][index - 1]["forbidden"] if index else None
                letters = [token[0] for token in certificate["witness"].split()]
                old_steps = (None if previous is None else [
                    step for step, (left, right) in enumerate(zip(letters, letters[1:]), 2)
                    if left + right == previous])
                for _ in range(2):
                    row = session.submit(raw)
                    self.assertEqual(row["feedback"], "Correct")
                    self.assertEqual(row["previous_violation_steps"], old_steps)
                    # Correct responses are never persistent errors, even if
                    # the diagnostic previous rule happens to be violated.
                    self.assertEqual(row["persistent_error"], None if old_steps is None else False)
                    self.assertEqual(row["previous_rule_compliant"],
                                     None if old_steps is None else not old_steps)
                count += 1
        self.assertEqual(count, 125)
        self.assertTrue(session.complete)
        self.assertEqual(session.report()["summary"]["mpe"], 0)

    def test_every_transition_has_a_previous_rule_compliant_current_error(self):
        path = Path(__file__).resolve().parents[1] / "data" / "easy_same_letter_v2" / "design_report.json"
        report = json.loads(path.read_text())
        records = {(item["id"], record["round"]): record
                   for item in self.subset["items"] for record in item["rounds"]}
        checked = 0
        for row in report["items"]:
            if row["type"] != "same_letter" or row["round"] == 1:
                continue
            probe = row["easy"]["previous_rule_probe"]
            response = json.dumps({"moves": probe["moves"], "answer": probe["answer"]})
            score = grade_response(response, records[row["id"], row["round"]], row["previous_forbidden"])
            self.assertFalse(score["correct"])
            self.assertTrue(score["previous_rule_compliant"])
            self.assertTrue(score["persistent_error"])
            self.assertEqual(score["previous_violation_steps"], [])
            checked += 1
        self.assertEqual(checked, 100)


if __name__ == "__main__":
    unittest.main()
