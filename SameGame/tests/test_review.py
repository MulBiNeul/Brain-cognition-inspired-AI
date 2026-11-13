"""Check review navigation, outcome witnesses and submission scoring."""

import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from scripts.build_review import TEMPLATE, build, case_metadata, diagnostic_witness, scoring_javascript
from src.evaluate import grade_response
from src.grade import Game


JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")


def fixture():
    return {
        "version": 1, "seed": 7, "rounds_per_item": 2, "cases_per_target": 1,
        "items": [{"id": "SG-TEST", "type": "different_letter", "rounds": [
            {"round": 1, "forbidden": "BA", "cases": [
                {"id": "C0-1", "input": "AABBAACC", "target": 0},
                {"id": "C1-1", "input": "AABBA", "target": 1},
            ]},
            {"round": 2, "forbidden": "AB", "cases": [
                {"id": "C0-1", "input": "AABBAACC", "target": 0},
                {"id": "C1-1", "input": "BAABB", "target": 1},
            ]},
        ]}],
    }


class ReviewTests(unittest.TestCase):
    def test_distinct_correct_persistent_and_general_error_examples(self):
        case = {"id": "C0-1", "input": "AABBAACC", "target": 0}
        metadata = case_metadata(case, 2, "AB", "BA")
        game = Game(case["input"], case["target"])
        results = {
            name: game.grade(example["moves"], "AB", "BA", final_answer=example["answer"])
            for name, example in metadata["examples"].items() if example is not None
        }
        self.assertTrue(results["correct"]["valid"])
        persistent = results["previousOnly"]
        self.assertTrue(persistent["board_exhausted"])
        self.assertEqual(persistent["remaining_length"], 0)
        self.assertTrue(persistent["violation_steps"])
        self.assertFalse(persistent["previous_violation_steps"])
        general = results["generalError"]
        self.assertFalse(general["valid"])
        self.assertTrue(general["previous_violation_steps"])
        self.assertTrue(results["both"]["valid"])
        self.assertFalse(results["both"]["previous_violation_steps"])

    def test_reachability_includes_incorrect_finals_and_groups_not_orders(self):
        case = {"id": "C0-1", "input": "AABBA", "target": 0}
        outcomes = {item["text"]: item for item in case_metadata(case, 1, "CC", None)["outcomes"]}
        self.assertEqual(set(outcomes), {"", "A"})
        self.assertTrue(outcomes[""]["correct"])
        self.assertFalse(outcomes["A"]["correct"])
        self.assertEqual(outcomes[""]["groups"], 1)
        self.assertEqual(outcomes["A"]["groups"], 1)
        independent = {"id": "C0-2", "input": "AABBCC", "target": 0}
        self.assertEqual(case_metadata(independent, 1, "DD", None)["outcomes"][0]["groups"], 1)

    def test_general_error_can_fail_goal_without_current_rule_violation(self):
        game = Game("AABBA", 0)
        example = diagnostic_witness(game, "CC", "AB", "generalError")
        self.assertIsNotNone(example)
        result = game.grade(example["moves"], "CC", "AB", final_answer=example["answer"])
        self.assertTrue(result["board_exhausted"])
        self.assertFalse(result["violation_steps"])
        self.assertTrue(result["previous_violation_steps"])
        self.assertNotEqual(result["remaining_length"], game.target)

    def test_build_embeds_cases_without_flattening_round_rules(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "problems.json").write_text(json.dumps(fixture()))
            (folder / "design_report.json").write_text(json.dumps({
                "valid": True, "case_certificates": [{"id": "SG-TEST", "round": 1, "case_id": "C0-1",
                    "target": 0, "forbidden": "BA", "shortest_success_moves": 4,
                    "minimum_merged_deletions": 0}],
            }))
            output = folder / "review.html"
            result = build(folder, output)
            html = output.read_text()
            embedded = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html).group(1))
            self.assertEqual(embedded, result)
            self.assertEqual(result["summary"]["rounds"], 2)
            self.assertEqual(result["summary"]["cases"], 4)
            self.assertEqual(result["summary"]["transitionRounds"], 1)
            self.assertEqual(result["summary"]["transitions"], 2)
            rounds = result["items"][0]["rounds"]
            self.assertEqual([r["forbidden"] for r in rounds], ["BA", "AB"])
            self.assertTrue(all(c["forbidden"] == "AB" for c in rounds[1]["cases"]))
            self.assertEqual(rounds[0]["cases"][0]["certificate"]["shortest_success_moves"], 4)
            self.assertEqual(result["reportHref"], "design_report.json")
            self.assertIn(scoring_javascript(), html)
            self.assertNotIn("__SCORING_JAVASCRIPT__", html)
            self.assertNotIn("__DATA__", html)
            self.assertIn('id="case-select"', html)
            self.assertIn('id="hand-trace"', html)
            self.assertIn("R1 · 지속 오류 판정 제외", html)

    def test_rejects_round_without_a_changed_input_and_target(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            document = fixture()
            document["items"][0]["rounds"][0]["cases"] = [document["items"][0]["rounds"][0]["cases"][0]]
            (folder / "problems.json").write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "different input and target"):
                build(folder, folder / "review.html")

    def test_rejects_stale_generator_report(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "problems.json").write_text(json.dumps(fixture()))
            (folder / "answer_key.json").write_text('{"items": []}')
            (folder / "design_report.json").write_text('{"dataset_sha256": "stale-report"}')
            with self.assertRaisesRegex(ValueError, "does not match"):
                build(folder, folder / "review.html")

    @unittest.skipUnless(JSC.exists(), "JavaScriptCore CLI is unavailable")
    def test_next_case_changes_only_adjacent_input_and_target(self):
        template = TEMPLATE.read_text()
        start = template.index("function eligibleCaseIndices(")
        end = template.index("function chooseCase(", start)
        cases = [
            {"id": "a", "input": "AABB", "target": 0},
            {"id": "b", "input": "AABB", "target": 1},
            {"id": "c", "input": "AABBA", "target": 0},
            {"id": "d", "input": "BAABB", "target": 1},
            {"id": "e", "input": "CCAA", "target": 2},
        ]
        script = template[start:end] + "\nconst cases=" + json.dumps(cases) + ";"
        script += "const next=nextCaseIndex(cases,0,()=>0);"
        script += "print(JSON.stringify({eligible:eligibleCaseIndices(cases,0),next,again:nextCaseIndex(cases,next,()=>0)}));"
        result = subprocess.run([str(JSC), "-e", script], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), {"eligible": [3, 4], "next": 3, "again": 0})

    @unittest.skipUnless(JSC.exists(), "JavaScriptCore CLI is unavailable")
    def test_case_replay_matches_python_scorer(self):
        requests, expected = [], []
        document = fixture()
        previous = None
        for record in document["items"][0]["rounds"]:
            for case in record["cases"]:
                meta = case_metadata(case, record["round"], record["forbidden"], previous)
                plans = [ex for ex in meta["examples"].values() if ex is not None]
                plans += [{"moves": o["moves"], "answer": o["text"]} for o in meta["outcomes"]]
                plans += [{"moves": "A99", "answer": ""}, {"moves": "A1", "answer": ""},
                          {"moves": "bad format", "answer": ""}]
                for plan in plans:
                    requests.append([case["input"], case["target"], record["forbidden"], previous,
                                     plan["moves"], plan["answer"]])
                    expected.append(grade_response(json.dumps(plan), meta, previous))
            previous = record["forbidden"]
        for target, answer in ((0, ""), (1, ""), (0, "B")):
            record = {"input": "AABBAABBAA", "target": target, "forbidden": "AA"}
            plan = {"moves": "A1 A1 A1 B1", "answer": answer}
            requests.append([record["input"], target, "AA", "BA", plan["moves"], answer])
            expected.append(grade_response(json.dumps(plan), record, "BA"))
        script = scoring_javascript() + "\nconst requests=" + json.dumps(requests) + ";"
        script += "print(JSON.stringify(requests.map(x=>inspectorReplay(...x))));"
        completed = subprocess.run([str(JSC), "-e", script], capture_output=True, text=True, check=True)
        actual = json.loads(completed.stdout)
        for index, (js, py) in enumerate(zip(actual, expected)):
            with self.subTest(request=requests[index]):
                self.assertEqual(js["feedback"], py["feedback"])
                self.assertEqual(js["violationSteps"], py["violation_steps"])
                self.assertEqual(js["persistentError"], bool(py["persistent_error"]))
                self.assertEqual(js["feedback"] == "Correct", py["correct"])
                self.assertEqual(js["finalState"], py["final_state"])
        self.assertEqual(actual[-3]["violationSteps"], [2, 3])
        self.assertEqual(actual[-3]["feedback"], "Correct, but rule violation. Rule violation: move 2.")


if __name__ == "__main__":
    unittest.main()
