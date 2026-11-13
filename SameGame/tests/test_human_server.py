"""Changing human cases retain private goals and replay each attempt's own input."""

from copy import deepcopy
from http.client import HTTPConnection
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from urllib.parse import urlencode

from src.evaluate import INSTRUCTIONS
from src.human_server import HumanApp, RequestError, create_server


def case_items(*, paired=False):
    rounds = []
    for number, forbidden in enumerate(("BB", "AA") if paired else ("AA", "BB"), 1):
        core = ("AABB" if number == 1 else "AABBAABBAA") if paired else "AA"
        suffix = "CDE" if paired else "BCDE"
        cases = [{"id": f"c{target*3+variant+1:02}", "input": "A"*variant + core + suffix[:target],
                  "target": target} for target in range(4) for variant in range(3)]
        rounds.append({"round": number, "forbidden": forbidden, "cases": cases})
    return [{"id": "PRIVATE_ITEM_001", "type": "same_letter", "rounds": rounds}]


ROOT = Path(__file__).resolve().parents[1]
JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")


class HumanAppTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name)
        self.items = case_items()
        self.app = HumanApp(self.items, self.output, 3, 2, seed=11)

    def tearDown(self):
        self.directory.cleanup()

    def payload(self, state, request_id, *, wrong=False):
        return {"token": state["token"], "turn_id": state["turn_id"], "request_id": request_id,
                "moves": "A1", "answer": "wrong" if wrong else state["input"].lstrip("A")}

    def assert_public(self, state):
        for key in ("target", "forbidden", "case_id", "selection_seed", "round", "type", "mpe", "per",
                    "persistent_error", "previous_rule_compliant", "violation_steps", "success_streak"):
            self.assertNotIn(key, state)
        self.assertNotIn("PRIVATE_ITEM_", json.dumps(state))
        for entry in state["history"]:
            self.assertEqual(set(entry), {"attempt", "input", "moves", "answer", "feedback"})

    def test_case_changes_preserve_round_history_and_round_tokens_change_only_at_transition(self):
        original = deepcopy(self.items)
        state = self.app.create({"participant_id": "test"})
        initial = deepcopy(state)
        state = self.app.submit(self.payload(state, "wrong", wrong=True))
        self.assert_public(state)
        self.assertNotEqual(initial["input"], state["input"])
        self.assertEqual(initial["round_token"], state["round_token"])
        self.assertEqual(state["history"][0]["input"], initial["input"])
        second_input = state["input"]
        state = self.app.submit(self.payload(state, "correct-1"))
        self.assertEqual([row["input"] for row in state["history"]], [initial["input"], second_input])
        self.assertEqual(initial["round_token"], state["round_token"])
        state = self.app.submit(self.payload(state, "correct-2"))
        self.assertNotEqual(initial["round_token"], state["round_token"])
        self.assertEqual(state["history"], [])
        self.assert_public(state)
        self.assertEqual(self.items, original)
        data = json.loads(self.app.path(state["token"]).read_text())
        self.assertEqual(data["items"], original)
        self.assertEqual(data["selection_seed"], 11)
        self.assertTrue(all({"case_id", "target", "forbidden"} <= set(row)
                            for row in data["report"]["submissions"]))

    def test_restart_and_duplicate_http_submission_recover_the_same_pending_case(self):
        initial = self.app.create({})
        payload = self.payload(initial, "one", wrong=True)
        once = self.app.submit(payload)
        self.assertEqual(self.app.submit(payload), once)
        restarted = HumanApp(self.items, self.output, 3, 2, seed=999)
        self.assertEqual(restarted.get(initial["token"]), once)
        later = restarted.submit(self.payload(once, "two"))
        self.assertEqual(restarted.submit(payload), once)
        self.assertEqual(restarted.get(initial["token"]), later)
        stale = dict(payload, request_id="stale")
        with self.assertRaises(RequestError) as failure:
            restarted.submit(stale)
        self.assertEqual(failure.exception.status, 409)
        self.assertEqual(restarted.get(initial["token"]), later)

    def test_server_serves_participant_page_without_private_data(self):
        server = create_server(self.items, self.output, seed=11, max_attempts=3)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for path, expected in (("/", 200), ("/human.html", 200), ("/problems.json", 404),
                                   ("/review.html", 404), ("/data/problems.json", 404)):
                connection = HTTPConnection(*server.server_address, timeout=5)
                connection.request("GET", path)
                response = connection.getresponse()
                body = response.read().decode()
                connection.close()
                self.assertEqual(response.status, expected)
                if expected == 200:
                    self.assertIn("samegame.participant", body)
                    self.assertIn("historyStates(entry)", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    @unittest.skipUnless(JSC.exists(), "JavaScriptCore is needed for participant replay checks")
    def test_browser_history_replays_distinct_inputs_and_retains_the_completed_round(self):
        html = (ROOT / "web/human.html").read_text()
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        functions = script[script.index("    function groups("):script.index("    function showString(")]
        first = {"attempt": 1, "input": "AAB", "moves": "A1", "answer": "B", "feedback": "Correct"}
        second = {"attempt": 2, "input": "CCCDE", "moves": "C1", "answer": "DE", "feedback": "Correct"}
        data = {"first": first, "second": second,
                "draft": {"input": second["input"], "round_token": "old"},
                "sent": {"moves": second["moves"], "answer": second["answer"]},
                "same": {"round_token": "old", "history": [first, second], "last_feedback": "Correct"},
                "next": {"round_token": "new", "history": [], "last_feedback": "Correct"}}
        source = ("new Function(" + json.dumps(script) + ");\n" + functions + "\nconst v=" + json.dumps(data) + ";\n"
                  "print(JSON.stringify({states:[historyStates(v.first),historyStates(v.second)],"
                  "same:historyAfterSubmission([v.first],v.draft,v.sent,v.same),"
                  "next:historyAfterSubmission([v.first],v.draft,v.sent,v.next)}));")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.js"
            path.write_text(source)
            result = subprocess.run([str(JSC), str(path)], text=True, capture_output=True, check=True)
        actual = json.loads(result.stdout)
        self.assertEqual([states[0]["text"] for states in actual["states"]], ["AAB", "CCCDE"])
        self.assertEqual([states[-1]["text"] for states in actual["states"]], ["B", "DE"])
        self.assertEqual(actual["same"], [first, second])
        self.assertEqual(actual["next"], [first, second])



class HumanSessionHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name)
        self.start_server()

    def start_server(self):
        from src.human_server import create_server

        self.server = create_server(case_items(paired=True), self.output, seed=11, max_attempts=3, required_streak=2)
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
                    "submission_class", "submission_classes", "copy", "case_id", "selection_seed"):
            self.assertNotIn(key, snapshot)
        self.assertNotIn("PRIVATE_ITEM_", json.dumps(snapshot))
        for entry in snapshot["history"]:
            self.assertEqual(set(entry), {"attempt", "input", "moves", "answer", "feedback"})

    def submission(self, snapshot, request_id, moves="A1 B1", answer=None):
        if answer is None:
            answer = snapshot["input"].lstrip("AB")
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
            "attempt": 1, "input": initial["input"], "moves": "A1 B1",
            "answer": initial["input"].lstrip("AB"), "feedback": "Correct",
        }])
        status, retried = self.request("POST", "/api/submit", payload)
        self.assertEqual(status, 200)
        self.assertEqual(retried, first)
        status, stale = self.request("POST", "/api/submit", self.submission(initial, "stale"))
        self.assertEqual(status, 409)
        self.assertEqual(stale["session"], first)
        status, next_round = self.request("POST", "/api/submit", self.submission(first, "second"))
        self.assertEqual(status, 200)
        self.assertNotEqual(next_round["input"], first["input"])
        self.assertNotEqual(next_round["round_token"], first["round_token"])
        self.assertEqual(next_round["input"].count("BB"), 2)
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

    def test_one_violation_position_survives_human_history_and_restart(self):
        current = self.begin()
        for index, moves in enumerate(["A1 B1", "A1 B1", "A1 A1 A1 B1"]):
            submitted_input = current["input"]
            status, current = self.request("POST", "/api/submit", self.submission(current, str(index), moves))
            self.assertEqual(status, 200)
        feedback = "Correct, but rule violation. Rule violation: move 2."
        self.assertEqual(current["last_feedback"], feedback)
        self.assertEqual(current["history"], [{
            "attempt": 1, "input": submitted_input, "moves": "A1 A1 A1 B1",
            "answer": submitted_input.lstrip("AB"), "feedback": feedback,
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
