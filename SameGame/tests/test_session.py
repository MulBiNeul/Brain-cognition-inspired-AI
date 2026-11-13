"""Changing cases must retain the shared scoring, progression, and metric contract."""

from copy import deepcopy
import json
import unittest

from src.evaluate import EvaluationSession, INSTRUCTIONS, grade_response
from src.session import ExperimentSession


def catalog(*, items=1, rules=("AA", "BB", "CC", "DD", "EE"), core="single"):
    records = []
    for item_number in range(1, items + 1):
        rounds = []
        for number, forbidden in enumerate(rules, 1):
            cases = []
            for target in range(4):
                for variant in range(3):
                    if core == "single":
                        tape = "A" * (2 + variant) + "BCDE"[:target]
                    elif core == "two":
                        tape = "A" * (2 + variant) + "BB" + "CDE"[:target]
                    elif core == "branch":
                        tape = "A" * (2 + variant) + "BBA" + "CDE"[:target]
                    else:
                        raise AssertionError(core)
                    cases.append({"id": f"c{len(cases)+1:02}", "input": tape, "target": target})
            rounds.append({"round": number, "forbidden": forbidden, "cases": cases})
        records.append({"id": f"PRIVATE-{item_number:03}",
                        "type": "same_letter" if rules[0][0] == rules[0][1] else "different_letter",
                        "rounds": rounds})
    return records


def answer(moves, text=""):
    return json.dumps({"moves": moves, "answer": text})


def single_solution(record):
    return answer("A1", record["input"].lstrip("A"))


class SessionTests(unittest.TestCase):
    def test_pending_case_is_stable_private_and_does_not_mutate_source(self):
        items = catalog()
        frozen = deepcopy(items)
        session = ExperimentSession(items, seed=19)
        current = session.current_record
        pending = deepcopy(current)
        expected_messages = [{"role": "system", "content": INSTRUCTIONS},
                             {"role": "user", "content": "Initial string: " + json.dumps(current["input"])}]
        for _ in range(3):
            self.assertEqual(session.messages(), expected_messages)
            self.assertEqual(session.report()["problems"][0]["rounds"][0]["submissions"], 0)
            self.assertEqual(current, pending)
        self.assertIs(current, session.items[0]["rounds"][0])
        row = session.submit("not JSON")
        self.assertIs(session.current_record, current)
        self.assertEqual(session.report()["problems"][0]["rounds"][0]["submissions"], 1)
        self.assertEqual({k: row[k] for k in ("case_id", "input", "target", "forbidden")},
                         {k: pending[k] for k in ("case_id", "input", "target", "forbidden")})
        self.assertNotIn("PRIVATE-", json.dumps(session.messages()))
        self.assertEqual(items, frozen)

    def test_input_and_goal_change_through_round_and_item_boundaries(self):
        items = catalog(items=2)
        session = ExperimentSession(items, seed=501)
        observed = []
        while not session.complete:
            record = deepcopy(session.current_record)
            item_index, round_index = session.item_index, session.round_index
            observed.append((item_index, round_index, record["input"], record["target"]))
            row = session.submit(single_solution(record))
            self.assertTrue(row["correct"])
            self.assertEqual(row["forbidden"], items[item_index]["rounds"][round_index]["forbidden"])
            if not session.complete and session.item_index != item_index:
                self.assertEqual(len(session.messages()), 2)
        self.assertEqual(len(observed), 20)
        for previous, current in zip(observed, observed[1:]):
            self.assertNotEqual(previous[2], current[2])
            self.assertNotEqual(previous[3], current[3])
        self.assertTrue(all(r["status"] == "success" for p in session.report()["problems"] for r in p["rounds"]))

    def test_unused_eligible_cases_are_preferred_then_nonconsecutive_reuse_is_allowed(self):
        session = ExperimentSession(catalog(rules=("AA",)), seed=7, max_attempts=20)
        used, previous, selected = set(), None, []
        for _ in range(20):
            record = deepcopy(session.current_record)
            eligible_unused = [c for c in record["cases"] if c["id"] not in used and
                               (previous is None or c["input"] != previous["input"] and c["target"] != previous["target"])]
            if eligible_unused:
                self.assertNotIn(record["case_id"], used)
            if previous is not None:
                self.assertNotEqual(record["input"], previous["input"])
                self.assertNotEqual(record["target"], previous["target"])
            used.add(record["case_id"])
            selected.append(record["case_id"])
            previous = record
            session.submit("not JSON")
        self.assertLess(len(set(selected)), len(selected))
        self.assertTrue(session.complete)

    def test_raw_response_replay_restores_cases_history_and_reports(self):
        items = catalog(items=2)
        session = ExperimentSession(items, seed=23, max_attempts=4)
        responses = []
        for wrong in (False, False, True, True, True, True, False, True, False):
            raw = "not JSON" if wrong else single_solution(session.current_record)
            responses.append(raw)
            session.submit(raw)
        restored = ExperimentSession(items, seed=23, max_attempts=4)
        for raw in responses:
            restored.submit(raw)
        self.assertEqual(restored.responses, responses)
        self.assertEqual(restored.items, session.items)
        self.assertEqual(restored.current_record, session.current_record)
        self.assertEqual(restored.messages(), session.messages())
        self.assertEqual(restored.report(), session.report())
        self.assertEqual((restored.item_index, restored.round_index, restored.attempt),
                         (session.item_index, session.round_index, session.attempt))

    def test_shared_grading_and_fixed_T_metrics_use_each_attempt_goal(self):
        items = catalog(rules=("BA", "AB"), core="two")
        session = ExperimentSession(items, seed=14)
        scored = []
        for moves in ("A1 B1", "A1 B1", "A1 B1", "B1 A1", "B1 A1"):
            record = deepcopy(session.current_record)
            previous = items[0]["rounds"][session.round_index-1]["forbidden"] if session.round_index else None
            raw = answer(moves, "CDE"[:record["target"]])
            expected = grade_response(raw, record, previous)
            row = session.submit(raw)
            self.assertEqual({key: row[key] for key in expected}, expected)
            self.assertEqual(row["target"], record["target"])
            scored.append(row)
        self.assertIsNone(scored[0]["persistent_error"])
        self.assertTrue(scored[2]["persistent_error"])
        self.assertEqual(scored[2]["violation_steps"], [2])
        self.assertEqual(scored[2]["previous_violation_steps"], [])
        self.assertTrue(scored[3]["correct"])
        self.assertEqual(scored[3]["previous_violation_steps"], [2])
        self.assertFalse(scored[3]["persistent_error"])
        report = session.report()
        second = report["problems"][0]["rounds"][1]
        self.assertEqual((second["persistent_errors"], second["submissions"], second["per_denominator"]), (1, 3, 10))
        self.assertEqual((second["per"], second["observed_per"]), (0.1, 1 / 3))
        self.assertEqual(report["summary"]["mpe"], 0.1)
        self.assertEqual(report["summary"]["submission_classes"],
                         {"current_correct": 4, "persistent_error": 1, "ordinary_error": 0, "invalid": 0})
        self.assertIs(ExperimentSession.report, EvaluationSession.report)
        self.assertIs(ExperimentSession._finish_round, EvaluationSession._finish_round)

    def test_current_goal_failure_counts_without_current_or_previous_rule_violation(self):
        session = ExperimentSession(catalog(rules=("CC", "DD"), core="branch"), seed=3)
        for _ in range(2):
            session.submit(answer("B1 A1", "CDE"[:session.current_record["target"]]))
        current = deepcopy(session.current_record)
        self.assertNotEqual(current["target"], session.records[-1]["target"])
        row = session.submit(answer("A1 B1", "A" + "CDE"[:current["target"]]))
        self.assertTrue(row["base_valid"] and row["board_exhausted"] and row["final_matches"])
        self.assertFalse(row["goal_reached"])
        self.assertEqual(row["violation_steps"], [])
        self.assertEqual(row["previous_violation_steps"], [])
        self.assertTrue(row["persistent_error"])
        self.assertEqual(session.report()["summary"]["mpe"], 0.1)

    def test_exhaustion_and_generation_failure_preserve_boundary_exclusions(self):
        for failure in ("not JSON", ""):
            with self.subTest(failure=failure):
                session = ExperimentSession(catalog(items=2), seed=6, max_attempts=2)
                submitted = None
                while session.item_index == 0:
                    submitted = deepcopy(session.current_record)
                    session.submit(failure)
                self.assertNotEqual(session.current_record["input"], submitted["input"])
                self.assertNotEqual(session.current_record["target"], submitted["target"])
                self.assertEqual(len(session.messages()), 2)
                first = session.report()["problems"][0]
                status = "exhausted" if failure else "generation_failure"
                self.assertEqual([r["status"] for r in first["rounds"]], [status] + ["excluded_after_"+status] * 4)
                self.assertIsNone(first["mpe"])

    def test_invalid_python_response_does_not_consume_a_case(self):
        session = ExperimentSession(catalog(), seed=4)
        before = deepcopy(session.current_record)
        with self.assertRaisesRegex(ValueError, "a submission must be a string"):
            session.submit(None)
        self.assertEqual(session.current_record, before)
        self.assertEqual(session.responses, [])


    def test_jsonl_replay_uses_the_same_case_selection_and_metrics(self):
        from src.evaluate import run
        puzzles = catalog(items=2)
        session = ExperimentSession(puzzles, seed=19)
        expected_feedback, raw_responses = [], []
        while not session.complete:
            raw = single_solution(session.current_record)
            raw_responses.append(raw)
            expected_feedback.append(session.submit(raw)["feedback"])
        iterator = iter(raw_responses)
        feedback = []
        report = run(puzzles, lambda _: next(iterator), feedback.append, seed=19)
        self.assertEqual(report, session.report())
        self.assertEqual(feedback, expected_feedback)


if __name__ == "__main__":
    unittest.main()
