"""Exercise shared progression, resumable human sessions, and public data boundaries."""

import copy
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlencode

from src.evaluate import EvaluationSession, INSTRUCTIONS, run


def fixtures():
    return [{
        "id": "PRIVATE_ITEM_001",
        "type": "same_letter",
        "rounds": [
            {"round": 1, "input": "AABB", "target": 0, "forbidden": "BA"},
            {"round": 2, "input": "AABBAABBAA", "target": 0, "forbidden": "AA"},
        ],
    }]


def response(moves, answer=""):
    return json.dumps({"moves": moves, "answer": answer})


class EvaluationSessionTests(unittest.TestCase):
    def test_round_progression_keeps_history_and_submission_level_per(self):
        session = EvaluationSession(fixtures())
        initial = session.messages()
        self.assertEqual(session.messages(), initial)
        self.assertEqual(session.attempt, 1)
        self.assertEqual(initial, [{"role": "system", "content": INSTRUCTIONS},
                                   {"role": "user", "content": 'Initial string: "AABB"'}])
        expected = [
            "Correct", "Correct", "Correct, but rule violation. Rule violation: moves 2, 3.",
            "Correct", "Correct",
        ]
        replies = [response(moves) for moves in ("AB", "AB", "AAAB", "BBA", "BBA")]
        for index, raw in enumerate(replies):
            row = session.submit(raw)
            self.assertEqual(row["feedback"], expected[index])
            if index == 1:
                self.assertEqual((session.round_index, session.attempt), (1, 1))
                history = session.messages()
                self.assertEqual(sum(m["content"] == INSTRUCTIONS for m in history), 1)
                self.assertEqual([m["content"] for m in history if m["role"] == "assistant"], replies[:2])
            if index == 2:
                self.assertEqual(row["violation_steps"], [2, 3])
                self.assertEqual(row["previous_violation_steps"], [])
                self.assertTrue(row["persistent_error"])
                self.assertTrue(row["previous_rule_compliant"])
                self.assertEqual(row["success_streak"], 0)
                self.assertEqual(session.messages()[-2], {"role": "user", "content": expected[index]})
            if index in (3, 4):
                self.assertTrue(row["correct"])
                self.assertFalse(row["persistent_error"])
                self.assertEqual(row["previous_violation_steps"], [3])
                self.assertEqual(row["success_streak"], index - 2)
        report = session.report()
        self.assertTrue(session.complete)
        self.assertTrue(report["complete"])
        self.assertEqual(session.responses, replies)
        first, second = report["problems"][0]["rounds"]
        self.assertEqual((first["submissions"], first["per"]), (2, None))
        self.assertIsNone(first["per_denominator"])
        self.assertEqual((second["submissions"], second["persistent_errors"]), (3, 1))
        self.assertEqual(second["per"], 1 / 10)
        self.assertEqual(second["per_denominator"], 10)
        self.assertEqual(second["observed_per"], 1 / 3)
        self.assertEqual(report["summary"]["mpe"], 1 / 10)
        self.assertEqual(report["summary"]["persistent_error_basis"], "previous_rule_compliant_current_incorrect")
        self.assertEqual(report["summary"]["per_denominator"], "max_attempts")
        self.assertEqual(report["summary"]["observed_mpe"], 1 / 3)

    def test_per_uses_configured_attempt_budget_after_early_success(self):
        for budget in (4, 10):
            with self.subTest(budget=budget):
                session = EvaluationSession(fixtures(), max_attempts=budget)
                for moves in ("AB", "AB", "AAAB", "BBA", "BBA"):
                    session.submit(response(moves))
                report = session.report()
                second = report["problems"][0]["rounds"][1]
                self.assertEqual(second["submissions"], 3)
                self.assertEqual(second["per"], 1 / budget)
                self.assertEqual(second["observed_per"], 1 / 3)
                self.assertEqual(report["summary"]["mpe"], 1 / budget)

    def test_replaying_saved_responses_restores_exact_progress_and_public_history(self):
        session = EvaluationSession(fixtures())
        for moves in ("AB", "AB", "AAAB"):
            session.submit(response(moves))
        restored = EvaluationSession(fixtures())
        for raw in session.responses:
            restored.submit(raw)
        self.assertEqual(restored.messages(), session.messages())
        self.assertEqual(restored.report(), session.report())
        self.assertEqual(restored.messages()[-2], {"role": "user", "content":
                         "Correct, but rule violation. Rule violation: moves 2, 3."})
        self.assertEqual((restored.item_index, restored.round_index, restored.attempt), (0, 1, 2))
        restored.submit(response("BBA"))
        self.assertFalse(restored.complete)
        restored.submit(response("BBA"))
        self.assertTrue(restored.complete)

    def test_exhaustion_skips_later_rounds_and_resets_history_for_next_item(self):
        items = fixtures()
        second = copy.deepcopy(items[0])
        second["id"] = "PRIVATE_ITEM_002"
        items.append(second)
        session = EvaluationSession(items, max_attempts=2, required_streak=2)
        for _ in range(2):
            session.submit(response("A", "BB"))
        self.assertEqual((session.item_index, session.round_index, session.attempt), (1, 0, 1))
        self.assertFalse(any(m["role"] == "assistant" for m in session.messages()))
        first = session.report()["problems"][0]
        self.assertEqual([r["status"] for r in first["rounds"]], ["exhausted", "excluded_after_exhausted"])
        self.assertEqual([r["submissions"] for r in first["rounds"]], [2, 0])

    def test_empty_generation_is_not_a_scored_attempt_and_eof_is_resumable(self):
        session = EvaluationSession(fixtures())
        self.assertIsNone(session.submit(""))
        report = session.report()
        self.assertTrue(session.complete)
        self.assertEqual(session.responses, [""])
        self.assertEqual(report["submissions"], [])
        self.assertEqual(
            [r["status"] for r in report["problems"][0]["rounds"]],
            ["generation_failure", "excluded_after_generation_failure"],
        )
        replies = iter([response("AB"), None])
        interrupted = run(fixtures(), lambda _: next(replies), lambda _: None)
        self.assertFalse(interrupted["complete"])
        self.assertEqual(len(interrupted["submissions"]), 1)
        self.assertEqual(interrupted["problems"][0]["rounds"][0]["status"], "incomplete")


class HumanSessionHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name)
        self.start_server()

    def start_server(self):
        from src.human_server import create_server

        self.server = create_server(fixtures(), self.output, max_attempts=3, required_streak=2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive())

    def tearDown(self):
        self.stop_server()
        self.directory.cleanup()

    def request(self, method, path, data=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"} if data is not None else {}
        connection.request(method, path, json.dumps(data) if data is not None else None, headers)
        reply = connection.getresponse()
        body = reply.read().decode()
        connection.close()
        try:
            value = json.loads(body)
        except json.JSONDecodeError:
            value = body
        return reply.status, value

    def begin(self):
        status, snapshot = self.request("POST", "/api/session", {"participant_id": "test_subject"})
        self.assertEqual(status, 200)
        self.assert_public(snapshot)
        return snapshot

    def assert_public(self, snapshot):
        self.assertEqual(snapshot["instructions"], INSTRUCTIONS)
        for key in ("target", "forbidden", "round", "type", "success_streak", "per", "mpe",
                    "violation_steps", "previous_violation_steps", "previous_rule_compliant",
                    "persistent_error", "persistent_error_basis", "observed_per", "per_denominator",
                    "rule_outcome", "copy"):
            self.assertNotIn(key, snapshot)
        self.assertNotIn("PRIVATE_ITEM_", json.dumps(snapshot))
        for entry in snapshot["history"]:
            self.assertEqual(set(entry), {"attempt", "moves", "answer", "feedback"})

    def submission(self, snapshot, request_id, moves="A1 B1", answer=""):
        return {"token": snapshot["token"], "turn_id": snapshot["turn_id"],
                "request_id": request_id, "moves": moves, "answer": answer}

    def test_server_serves_public_html_and_never_exposes_private_files(self):
        status, info = self.request("GET", "/api/info")
        self.assertEqual(status, 200)
        self.assertEqual(info["instructions"], INSTRUCTIONS)
        for path in ("/problems.json", "/answer_key.json", "/review.html", "/evaluate.py",
                     "/data/problems.json", "/data/answer_key.json", "/web/review.html",
                     "/src/evaluate.py", "/runs", "/../data/problems.json"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("GET", "/")[0], 200)

    def test_submission_retries_are_idempotent_and_stale_turns_do_not_score(self):
        initial = self.begin()
        payload = self.submission(initial, "first")
        status, first = self.request("POST", "/api/submit", payload)
        self.assertEqual(status, 200)
        self.assert_public(first)
        self.assertNotEqual(first["turn_id"], initial["turn_id"])
        self.assertEqual(first["history"], [{
            "attempt": 1, "moves": "A1 B1", "answer": "", "feedback": "Correct",
        }])
        status, retried = self.request("POST", "/api/submit", payload)
        self.assertEqual(status, 200)
        self.assertEqual(retried, first)
        status, stale = self.request("POST", "/api/submit", self.submission(initial, "stale"))
        self.assertEqual(status, 409)
        self.assertEqual(stale["session"], first)
        status, next_round = self.request("POST", "/api/submit", self.submission(first, "second"))
        self.assertEqual(status, 200)
        self.assertEqual(next_round["input"], "AABBAABBAA")
        self.assertEqual(next_round["history"], [])
        self.assertEqual(next_round["last_feedback"], "Correct")
        self.assert_public(next_round)

    def test_persistence_survives_server_restart_and_repeated_delivery(self):
        initial = self.begin()
        payload = self.submission(initial, "persist")
        status, first = self.request("POST", "/api/submit", payload)
        self.assertEqual(status, 200)
        self.stop_server()
        self.start_server()
        status, restored = self.request("GET", "/api/session?" + urlencode({"token": first["token"]}))
        self.assertEqual(status, 200)
        self.assertEqual(restored, first)
        status, repeated = self.request("POST", "/api/submit", payload)
        self.assertEqual(status, 200)
        self.assertEqual(repeated, first)

    def test_concurrent_submissions_of_same_turn_score_only_once(self):
        initial = self.begin()
        barrier = threading.Barrier(2)

        def submit(request_id):
            barrier.wait()
            return self.request("POST", "/api/submit", self.submission(initial, request_id))

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit, ["race-a", "race-b"]))
        self.assertEqual(sorted(status for status, _ in results), [200, 409])
        status, state = self.request("GET", "/api/session?" + urlencode({"token": initial["token"]}))
        self.assertEqual(status, 200)
        self.assertEqual(len(state["history"]), 1)

    def test_all_violation_positions_survive_human_history_and_restart(self):
        current = self.begin()
        for index, moves in enumerate(["A1 B1", "A1 B1", "A1 A1 A1 B1"]):
            status, current = self.request("POST", "/api/submit", self.submission(current, str(index), moves))
            self.assertEqual(status, 200)
        feedback = "Correct, but rule violation. Rule violation: moves 2, 3."
        self.assertEqual(current["last_feedback"], feedback)
        self.assertEqual(current["history"], [{
            "attempt": 1, "moves": "A1 A1 A1 B1", "answer": "", "feedback": feedback,
        }])
        self.assert_public(current)
        self.stop_server()
        self.start_server()
        status, restored = self.request("GET", "/api/session?" + urlencode({"token": current["token"]}))
        self.assertEqual(status, 200)
        self.assertEqual(restored, current)
        self.assert_public(restored)

    def test_completion_keeps_feedback_without_exposing_hidden_scores(self):
        current = self.begin()
        for index, moves in enumerate(["A1 B1", "A1 B1", "B1 B1 A1", "B1 B1 A1"]):
            status, current = self.request("POST", "/api/submit", self.submission(current, str(index), moves))
            self.assertEqual(status, 200)
            self.assert_public(current)
        self.assertTrue(current["complete"])
        self.assertIsNone(current["input"])
        self.assertEqual(current["last_feedback"], "Correct")


if __name__ == "__main__":
    unittest.main()
