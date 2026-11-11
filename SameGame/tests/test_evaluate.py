"""Regression tests for participant feedback and submission-level scoring."""

import json
import unittest

from src.evaluate import INSTRUCTIONS, grade_response, run
from src.grade import Game, terminal_group_counts


class TerminalGroupTests(unittest.TestCase):
    def test_swapping_independent_deletions_counts_once_but_a_merged_group_is_distinct(self):
        # Removing BB first joins the two AA groups into a single four-letter action.
        self.assertEqual(terminal_group_counts(Game("AABBAA", 0)), {"": 2})
        self.assertEqual(terminal_group_counts(Game("AABB", 0)), {"": 1})

    def test_all_terminal_outcomes_count_even_if_the_goal_is_missed(self):
        # Either AA can go first. Two groupings leave B; a third clears the board.
        for target in (0, 1, 3):
            self.assertEqual(terminal_group_counts(Game("BAABBAA", target)), {"B": 2, "": 1})
        self.assertEqual(terminal_group_counts(Game("ABB ACCCCC", 0)), {"A A": 1})
        self.assertEqual(terminal_group_counts(Game("ABCDE", 0)), {"ABCDE": 1})


class FeedbackTests(unittest.TestCase):
    def test_public_requests_keep_feedback_but_hide_evaluator_metadata(self):
        item = {"id": "PRIVATE_PROBLEM_204", "type": "PRIVATE_RULE_TYPE", "rounds": [
            {"round": 1, "input": "AABB", "target": 0, "forbidden": "BA"},
            {"round": 2, "input": "CCDDE", "target": 1, "forbidden": "DC"},
        ]}
        submissions = [
            '{"moves":"A1","answer":"BB"}',
            '{"moves":"A1 B1","answer":""}',
            '{"moves":"A1 B1","answer":""}',
            '{"moves":"C1 D1","answer":"E"}',
            '{"moves":"C1 D1","answer":"E"}',
        ]
        delivered, histories = [], []
        hidden_terms = r"\b(?:goals?|targets?|forbidden|rules?|rounds?|restrictions?)\b"
        self.assertNotRegex(INSTRUCTIONS.lower(), hidden_terms)
        self.assertNotRegex(INSTRUCTIONS.lower(), r"zero,\s*one,\s*two|\b0\s*[-–]\s*3\b")

        def respond(history):
            index = len(histories)
            histories.append(history)
            initial = item["rounds"][0 if index < 3 else 1]["input"]
            self.assertEqual(history[-1], {"role": "user", "content":
                                           "Initial string: " + json.dumps(initial)})
            self.assertEqual(sum(m["content"] == INSTRUCTIONS for m in history), 1)
            self.assertEqual([m for m in history if m["role"] == "system"],
                             [{"role": "system", "content": INSTRUCTIONS}])
            for message in history:
                self.assertEqual(set(message), {"role", "content"})
                self.assertNotIn(item["id"], message["content"])
                self.assertNotIn(item["type"], message["content"])
                if message["role"] != "assistant" and message["content"] not in delivered:
                    self.assertNotRegex(message["content"].lower(), hidden_terms)
            self.assertEqual([m["content"] for m in history if m["role"] == "assistant"], submissions[:index])
            self.assertEqual([m["content"] for m in history if m["role"] == "user"
                              and m["content"] in delivered], delivered)
            return submissions[index]

        report = run([item], respond, delivered.append)
        self.assertTrue(report["complete"])
        self.assertEqual(delivered, ["Incorrect, but no rule violation", "Correct", "Correct", "Correct", "Correct"])
        self.assertEqual(len(histories), 5)
        self.assertEqual(histories[-1][0], {"role": "system", "content": INSTRUCTIONS})

    def test_occurrence_selection_changes_the_final_string(self):
        record = {"input": "BAABBAA", "target": 1, "forbidden": "CC"}
        for moves, expected in [("A1 B1 A1", ""), ("A2 B1 A1", "B")]:
            with self.subTest(moves=moves):
                result = grade_response(json.dumps({"moves": moves, "answer": expected}), record)
                self.assertEqual(result["final_state"], expected)
                self.assertEqual(result["correct"], expected == "B")

    def test_numbers_span_segments_but_deletions_preserve_the_boundary(self):
        record = {"input": "BAA AABBB", "target": 1, "forbidden": "BA"}
        first = grade_response('{"moves":"A2","answer":"BAA BBB"}', record)
        self.assertEqual(first["final_state"], "BAA BBB")
        result = grade_response('{"moves":"A2 B1 A1","answer":"B"}', record)
        self.assertEqual(result["executed_steps"], 3)
        self.assertEqual(result["violation_steps"], [3])
        self.assertEqual(result["feedback"], "Correct, but rule violation. Rule violation: move 3.")

    def test_live_ordinals_change_after_removing_a_group(self):
        record = {"input": "AAACAADAAA", "target": 2, "forbidden": "BE"}
        result = grade_response('{"moves":"A2 A2 A1","answer":"CD"}', record)
        self.assertTrue(result["correct"])
        invalid = grade_response('{"moves":"A2 A3 A1","answer":"CD"}', record)
        self.assertTrue(invalid["parse_valid"])
        self.assertEqual((invalid["first_basic_error_step"], invalid["executed_steps"]), (2, 1))
        self.assertEqual(invalid["final_state"], "AAACDAAA")
        self.assertEqual(invalid["violation_steps"], [])
        self.assertEqual(invalid["feedback"], "Incorrect, but no rule violation")

    def test_missing_ordinal_stops_at_the_requested_group_and_counts_its_letter(self):
        record = {"input": "BBAABBAA", "target": 0, "forbidden": "BA"}
        failed = grade_response('{"moves":"B2 A2","answer":""}', record)
        self.assertEqual((failed["executed_steps"], failed["first_basic_error_step"]), (1, 2))
        self.assertEqual((failed["final_state"], failed["violation_steps"]), ("BBAAAA", [2]))
        self.assertEqual(failed["feedback"], "Incorrect, also rule violation. Rule violation: move 2.")

    def test_format_limit_counts_tokens_and_rejects_malformed_ordinals(self):
        record = {"input": "AABB", "target": 0, "forbidden": "AB"}
        for length, valid in [(30, True), (31, False)]:
            reply = json.dumps({"moves": " ".join(["A1"] * length), "answer": ""})
            self.assertEqual(grade_response(reply, record)["parse_valid"], valid)
        for moves in ("A0", "A01", "A2 B-1", "A1B1", ""):
            with self.subTest(moves=moves):
                self.assertFalse(grade_response(json.dumps({"moves": moves, "answer": ""}), record)["parse_valid"])

    def test_copies_match_canonical_tokens_including_the_group_number(self):
        item = {"id": "test", "type": "different_letter", "rounds": [
            {"round": 1, "input": "AABBAA", "target": 0, "forbidden": "BA"}
        ]}
        plans = iter(["AAB", "A1 A1 B1", "A2 A1 B1"])
        report = run([item], lambda _: json.dumps({"moves": next(plans), "answer": ""}),
                     lambda _: None, required_streak=3)
        first, second, third = report["submissions"]
        self.assertEqual(first["moves"], "A1 A1 B1")
        self.assertEqual(second["copy"]["previous_submission"], {"moves": True, "response": True})
        self.assertEqual(third["copy"]["same_round_success"], {"moves": False, "response": False})
        self.assertEqual([row["correct"] for row in (first, second, third)], [True, True, True])

    def test_same_endpoint_can_produce_different_feedback(self):
        record = {"input": "AABABBABB", "target": 0, "forbidden": "AA"}
        cases = [
            ("ABAB", "", "Correct"),
            ("BAAB", "", "Correct, but rule violation. Rule violation: move 3."),
            ("ABBA", "B", "Incorrect, but no rule violation"),
            ("BBAA", "B", "Incorrect, also rule violation. Rule violation: move 4."),
        ]
        for moves, answer, expected in cases:
            with self.subTest(moves=moves):
                submitted = json.dumps({"moves": moves, "answer": answer})
                result = grade_response(submitted, record, previous="BA")
                self.assertEqual(result["feedback"], expected)
                self.assertEqual(result["executed_steps"], len(moves))

    def test_replay_beyond_hidden_target_and_stop_at_first_impossible_move(self):
        result = grade_response('{"moves":"ABC","answer":"B"}',
                                {"input": "AABBCCB", "target": 3, "forbidden": "BC"})
        self.assertEqual(result["feedback"], "Incorrect, also rule violation. Rule violation: move 3.")
        self.assertEqual(result["violation_steps"], [3])
        self.assertEqual(result["final_state"], "B")
        result = grade_response('{"moves":"AAA","answer":""}',
                                {"input": "AABB", "target": 0, "forbidden": "AA"})
        self.assertEqual(result["executed_steps"], 1)
        self.assertEqual(result["first_basic_error_step"], 2)
        self.assertEqual(result["violation_steps"], [2])

    def test_final_answer_must_preserve_the_boundary_and_match_the_plan(self):
        record = {"input": "ABB A", "target": 2, "forbidden": "CD"}
        for answer, expected in [("A A", "Correct"), ("AA", "Incorrect, but no rule violation")]:
            with self.subTest(answer=answer):
                result = grade_response(json.dumps({"moves": "B", "answer": answer}), record)
                self.assertEqual(result["feedback"], expected)
        self.assertEqual(grade_response('{"answer":"A A"}', record)["feedback"], "Incorrect, but no rule violation")

    def test_repeated_violations_count_once_per_submission(self):
        item = {"id": "test", "type": "same_letter", "rounds": [
            {"round": 1, "input": "AABB", "target": 0, "forbidden": "BB"},
            {"round": 2, "input": "AABBAABBAA", "target": 0, "forbidden": "AA"},
        ]}
        submissions = iter(["AB", "AB", "AAAB", "BBA", "BBA"])
        delivered = []
        report = run([item], lambda _: json.dumps({"moves": next(submissions), "answer": ""}),
                     delivered.append)
        first, second = report["problems"][0]["rounds"]
        self.assertEqual(delivered, ["Correct", "Correct",
                                    "Correct, but rule violation. Rule violation: move 2.",
                                    "Correct", "Correct"])
        self.assertIsNone(first["per"])
        self.assertEqual(report["submissions"][2]["violation_steps"], [2, 3])
        self.assertEqual(second["persistent_errors"], 1)
        self.assertEqual(second["per"], 1 / 3)
        self.assertEqual(report["summary"]["mpe"], 1 / 3)

    def test_multiple_violations_reveal_only_the_first_submitted_move_number(self):
        record = {"input": "AABBAABBAA", "target": 0, "forbidden": "AA"}
        for answer, label in [("", "Correct, but rule violation"),
                              ("B", "Incorrect, also rule violation")]:
            with self.subTest(answer=answer):
                result = grade_response(json.dumps({"moves": "A2 A1 A1 B1", "answer": answer}), record)
                self.assertEqual(result["violation_steps"], [2, 3])
                self.assertEqual(result["executed_steps"], 4)
                self.assertEqual(result["feedback"], label + ". Rule violation: move 2.")
                self.assertFalse(result["correct"])

    def test_the_same_final_can_be_valid_under_one_or_both_rules(self):
        record = {"input": "AABBCC", "target": 0, "forbidden": "BC"}
        for moves, outcome in [("CAB", "current_only"), ("BCA", "previous_only"),
                               ("ABC", "neither"), ("BAC", "both")]:
            with self.subTest(moves=moves):
                result = grade_response(json.dumps({"moves": moves, "answer": ""}), record, previous="AB")
                self.assertEqual(result["rule_outcome"], outcome)
                self.assertEqual(result["correct"], outcome in {"current_only", "both"})

    def test_stopping_at_target_with_a_group_is_incomplete(self):
        record = {"input": "AABBCC", "target": 2, "forbidden": "BA"}
        for moves, feedback in [("AB", "Incorrect, but no rule violation"),
                                ("BA", "Incorrect, also rule violation. Rule violation: move 2.")]:
            with self.subTest(moves=moves):
                result = grade_response(json.dumps({"moves": moves, "answer": "CC"}), record)
                self.assertEqual(result["feedback"], feedback)
                self.assertFalse(result["board_exhausted"])
                self.assertTrue(result["final_matches"])
                self.assertFalse(result["physical_goal_reached"])
                self.assertEqual(result["executed_steps"], 2)
                self.assertEqual(result["final_state"], "CC")

    def test_replay_after_a_violation_requires_the_remaining_actions(self):
        record = {"input": "AABBCC", "target": 0, "forbidden": "AB"}
        short = grade_response('{"moves":"AB","answer":""}', record, previous="BA")
        full = grade_response('{"moves":"ABC","answer":""}', record, previous="BA")
        self.assertEqual(short["feedback"], "Incorrect, also rule violation. Rule violation: move 2.")
        self.assertEqual((short["final_state"], short["board_exhausted"]), ("CC", False))
        self.assertEqual(full["feedback"], "Correct, but rule violation. Rule violation: move 2.")
        self.assertEqual((full["final_state"], full["board_exhausted"]), ("", True))
        self.assertEqual(short["violation_steps"], full["violation_steps"])
        self.assertEqual(full["rule_outcome"], "previous_only")

    def test_exhausted_board_can_miss_the_target(self):
        record = {"input": "ABCCDA", "target": 2, "forbidden": "BB"}
        result = grade_response('{"moves":"C","answer":"ABDA"}', record)
        self.assertTrue(result["board_exhausted"])
        self.assertTrue(result["final_matches"])
        self.assertFalse(result["physical_goal_reached"])
        self.assertEqual(result["feedback"], "Incorrect, but no rule violation")

    def test_only_remaining_move_counts_even_if_it_will_violate_the_rule(self):
        record = {"input": "AABBCC", "target": 0, "forbidden": "BC"}
        waiting = grade_response('{"moves":"AB","answer":"CC"}', record)
        finished = grade_response('{"moves":"ABC","answer":""}', record)
        self.assertEqual((waiting["feedback"], waiting["board_exhausted"]), ("Incorrect, but no rule violation", False))
        self.assertEqual((finished["feedback"], finished["board_exhausted"]),
                         ("Correct, but rule violation. Rule violation: move 3.", True))
        self.assertEqual(finished["violation_steps"], [3])

    def test_two_correct_submissions_can_repeat_on_the_same_input(self):
        item = {"id": "test", "type": "different_letter", "rounds": [
            {"round": 1, "input": "ABB", "target": 1, "forbidden": "DB"}
        ]}
        prompts = []

        def respond(history):
            prompts.append(history[-1]["content"])
            return '{"moves":"B","answer":"A"}'

        report = run([item], respond, lambda _: None)
        rows = report["submissions"]
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual([r["input"] for r in rows], ["ABB", "ABB"])
        self.assertEqual([r["success_streak"] for r in rows], [1, 2])
        self.assertTrue(rows[1]["copy"]["previous_submission"]["moves"])
        self.assertTrue(rows[1]["copy"]["same_round_success"]["response"])
        self.assertEqual(report["problems"][0]["rounds"][0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
