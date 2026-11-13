"""Exercise the model evaluation protocol without model API calls."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import model_runner, providers
from src.catalog import load_catalog
from src.evaluate import INSTRUCTIONS


def items():
    return [{"id": f"SG-{number:03d}", "type": "same_letter", "rounds": [
        {"round": r, "forbidden": forbidden, "cases": [
            {"id": f"c{goal}", "input": "A"*20 + "BBCCDDEE" + "CDE"[:goal], "target": goal}
            for goal in range(4)]}
        for r, forbidden in enumerate(("CC", "DD"), 1)]}
        for number in (1, 2)]


def answer(request):
    history = request.get("input", request.get("messages"))
    text = json.loads(history[-1]["content"].removeprefix("Initial string: "))
    return json.dumps({"moves": "A1 B1 C1 D1 E1", "answer": text[28:]})


def response(text, index=0):
    return {"status": "completed", "output": [
        {"type": "reasoning", "encrypted_content": f"opaque-{index}", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 12, "output_tokens": 8}}


class ModelExecutionTests(unittest.TestCase):
    def setUp(self):
        self.settings = model_runner.load_settings(profile="sol")
        self.settings.update(selection_seed=20261112, max_attempts=5, required_streak=2)

    def execute(self, directory, receive, *, resume=False):
        with redirect_stderr(StringIO()):
            return model_runner.execute(items(), self.settings, directory, resume=resume,
                                           client=object(), receive=receive, retry_delay=0)

    def test_new_cases_native_history_and_hidden_metadata(self):
        requests = []

        def receive(client, provider, request, path):
            requests.append(deepcopy(request))
            return response(answer(request), len(requests))

        with TemporaryDirectory() as folder:
            report = self.execute(Path(folder), receive)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["submissions"]), 8)
            self.assertEqual(report["summary"]["mpe"], 0)
            for before, after in zip(report["submissions"], report["submissions"][1:]):
                self.assertNotEqual(before["input"], after["input"])
                self.assertNotEqual(before["target"], after["target"])
            for request in requests:
                self.assertEqual(request["instructions"], INSTRUCTIONS)
                visible = json.dumps(request["input"])
                for field in ("case_id", "target", "forbidden", "persistent_error"):
                    self.assertNotIn(field, visible)
            self.assertTrue(any(x.get("encrypted_content") == "opaque-1" for x in requests[2]["input"]))
            self.assertEqual(len(requests[4]["input"]), 1)
            frozen = model_runner.read_json(Path(folder)/"protocol.json")
            self.assertEqual(frozen["items"], items())
            self.assertEqual(frozen["version"], 1)

    def test_interrupted_resume_reuses_the_exact_pending_case(self):
        requests = []

        def receive(client, provider, request, path):
            requests.append(deepcopy(request))
            if len(requests) == 2:
                raise KeyboardInterrupt
            return response(answer(request), len(requests))

        with TemporaryDirectory() as folder:
            directory = Path(folder)
            with self.assertRaises(KeyboardInterrupt):
                self.execute(directory, receive)
            self.assertEqual(len(model_runner.read_json(directory/"checkpoint.json")["responses"]), 1)
            pending = requests[-1]
            recovered_requests = []

            def resumed(client, provider, request, path):
                recovered_requests.append(deepcopy(request))
                return response(answer(request), len(recovered_requests)+1)

            report = self.execute(directory, resumed, resume=True)
            self.assertEqual(recovered_requests[0], pending)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["submissions"]), 8)
            self.assertEqual(report["summary"]["mpe"], 0)

    def test_transport_retry_does_not_select_another_case(self):
        requests = []

        def receive(client, provider, request, path):
            requests.append(deepcopy(request))
            if len(requests) == 1:
                raise providers.ProviderError("temporary", retryable=True)
            return response(answer(request), len(requests))

        with TemporaryDirectory() as folder, patch.object(model_runner.time, "sleep"):
            report = self.execute(Path(folder), receive)
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(len(report["submissions"]), 8)

    def test_completed_resume_makes_no_more_requests(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            first = self.execute(directory, lambda c, p, request, path: response(answer(request)))
            second = self.execute(directory, lambda *args: self.fail("unexpected request"), resume=True)
            self.assertEqual(first, second)

    def test_incomplete_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            model_runner.require_protocol({"version": 1, "settings": {}})

    def test_dry_run_needs_no_api_key_and_creates_no_results(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root/"problems.json"
            dataset.write_text(json.dumps({"version": 1, "seed": 72, "rounds_per_item": 2,
                                           "cases_per_target": 1, "items": items()}))
            template = Path(model_runner.DEFAULT_CONFIGS["sol"]).read_text()
            config = root/"config.toml"
            config.write_text(template.replace("../data/problems.json", "problems.json"))
            output = root/"results"
            with redirect_stdout(StringIO()) as printed, patch.object(providers, "make_client", side_effect=AssertionError("API client created")):
                code = model_runner.main(["--config", str(config), "--dry-run", "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            request = json.loads(printed.getvalue())["request"]
            self.assertEqual(request["instructions"], INSTRUCTIONS)
            self.assertEqual(len(request["input"]), 1)

    def test_declared_counts_and_dataset_order_are_preserved(self):
        document = {"version": 1, "seed": 72, "rounds_per_item": 2,
                    "cases_per_target": 1, "items": list(reversed(items()))}
        with TemporaryDirectory() as folder:
            path = Path(folder)/"problems.json"
            path.write_text(json.dumps(document))
            loaded, seed = load_catalog(path)
            self.assertEqual(loaded, document["items"])
            for field, value in (("rounds_per_item", 3), ("cases_per_target", 2)):
                with self.subTest(field=field):
                    altered = {**document, field: value}
                    path.write_text(json.dumps(altered))
                    with self.assertRaises(ValueError):
                        load_catalog(path)


    def test_provider_profiles_and_top_level_instructions(self):
        expected = {"sol": "gpt-6.1-sol", "opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}
        for profile, name in expected.items():
            settings = model_runner.load_settings(profile=profile)
            self.assertEqual(settings["model"], name)
            initial = [{"role": "user", "content": 'Initial string: "AABB"'}]
            request = providers.make_request(settings, INSTRUCTIONS, initial)
            self.assertNotIn("temperature", request)
            self.assertEqual(request["tools"], [])
            if profile == "sol":
                self.assertEqual(request["instructions"], INSTRUCTIONS)
                self.assertEqual(request["input"], initial)
                self.assertFalse(request["store"])
                self.assertEqual(request["reasoning"], {"effort": "medium", "context": "all_turns"})
                self.assertEqual(request["include"], ["reasoning.encrypted_content"])
            else:
                self.assertEqual(request["system"], INSTRUCTIONS)
                self.assertEqual(request["messages"], initial)
                self.assertEqual(request["thinking"], {"type": "adaptive", "display": "omitted"})
                self.assertEqual(request["output_config"], {"effort": "medium"})

    def test_one_violation_position_reaches_both_providers_on_changing_inputs(self):
        puzzle = [{"id": "private-item", "type": "same_letter", "rounds": [
            {"round": 1, "forbidden": "AA", "cases": [
                {"id": f"c{goal}", "input": "AABBAABBAA" + "CDE"[:goal], "target": goal}
                for goal in range(4)]}]}]
        feedback = "Correct, but rule violation. Rule violation: move 2."
        for profile, key in (("sol", "input"), ("opus", "messages")):
            settings = model_runner.load_settings(profile=profile)
            settings.update(selection_seed=72, max_attempts=2, required_streak=1)
            requests, outputs = [], []
            def receive(client, provider, request, path):
                requests.append(deepcopy(request))
                tape = json.loads(request[key][-1]["content"].removeprefix("Initial string: "))
                moves = "A1 A1 A1 B1" if len(requests) == 1 else "B1 B1 A1"
                raw_text = json.dumps({"moves": moves, "answer": tape[10:]})
                raw = response(raw_text)
                if provider == "anthropic":
                    raw = {"type": "message", "role": "assistant", "stop_reason": "end_turn",
                           "content": [{"type": "thinking", "thinking": "", "signature": "opaque"},
                                       {"type": "text", "text": raw_text}]}
                outputs.append(raw)
                return raw
            with self.subTest(profile=profile), TemporaryDirectory() as folder, redirect_stderr(StringIO()):
                report = model_runner.execute(puzzle, settings, Path(folder), client=object(), receive=receive)
            self.assertTrue(report["complete"])
            self.assertEqual([r["feedback"] for r in report["submissions"]], [feedback, "Correct"])
            self.assertEqual(report["submissions"][0]["violation_steps"], [2, 3])
            self.assertNotEqual(requests[0][key][-1], requests[1][key][-1])
            self.assertEqual(requests[1][key], requests[0][key] +
                providers.continuation_items(outputs[0], settings["provider"]) +
                [{"role": "user", "content": feedback}, requests[1][key][-1]])
            self.assertNotIn("forbidden", json.dumps(requests))

    def test_persistent_error_is_saved_privately_with_its_actual_case(self):
        puzzle = [{"id": "private-item", "type": "same_letter", "rounds": [
            {"round": number, "forbidden": forbidden, "cases": [
                {"id": f"c{goal}", "input": "AABBAABB" + "CDE"[:goal], "target": goal}
                for goal in range(4)]}
            for number, forbidden in enumerate(("AA", "BB", "CC"), 1)]}]
        settings = {**self.settings, "max_attempts": 10, "required_streak": 1, "reasoning_summary": "auto"}
        requests = []
        def receive(client, provider, request, path):
            requests.append(deepcopy(request))
            tape = json.loads(request["input"][-1]["content"].removeprefix("Initial string: "))
            moves = "A1 A1 B1" if len(requests) == 3 else "B1 B1 A1"
            return response(json.dumps({"moves": moves, "answer": tape[8:]}))
        with TemporaryDirectory() as folder, redirect_stderr(StringIO()):
            directory = Path(folder)
            report = model_runner.execute(puzzle, settings, directory, client=object(), receive=receive)
            row = report["submissions"][1]
            self.assertTrue(row["persistent_error"])
            self.assertEqual(row["previous_violation_steps"], [])
            self.assertEqual(row["violation_steps"], [2])
            self.assertEqual(report["summary"]["mpe"], .05)
            self.assertEqual(model_runner.read_json(directory/"submissions/00002_record.json")["score"], row)
            detail = model_runner.read_json(directory/"submissions/00002_reasoning_summary.json")
            for field in ("input", "target", "case_id", "persistent_error"):
                self.assertEqual(detail[field], row[field])
            self.assertTrue(report["submissions"][2]["correct"])
            self.assertFalse(report["submissions"][2]["persistent_error"])
            self.assertEqual(report["submissions"][2]["previous_violation_steps"], [2])
            self.assertEqual(requests[2]["input"][-2]["content"], row["feedback"])
            for private in ("persistent_error", "previous_violation_steps", "case_id", "target", "forbidden"):
                self.assertNotIn(private, json.dumps(requests))


if __name__ == "__main__":
    unittest.main()
