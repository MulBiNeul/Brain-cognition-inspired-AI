"""Regression coverage for native context and interruption-safe API evaluation."""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import model_runner, providers
from src.evaluate import INSTRUCTIONS


def response(text='{"moves":"A1 B1","answer":""}', *, marker="opaque", status="completed"):
    return {
        "status": status,
        "output": [
            {"type": "reasoning", "encrypted_content": marker, "summary": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
        ],
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }


def problem(identifier, rounds=1):
    return {"id": identifier, "type": "same_letter", "rounds": [
        {"round": number, "input": "AABB", "target": 0, "forbidden": "AA"}
        for number in range(1, rounds+1)
    ]}


class ModelExecutionTests(unittest.TestCase):
    def setUp(self):
        self.settings = model_runner.load_settings(profile="sol")
        self.settings.update(max_attempts=2, required_streak=1)

    def execute(self, items, directory, receive, *, resume=False):
        with redirect_stderr(StringIO()):
            return model_runner.execute(items, self.settings, directory, resume=resume,
                                        client=object(), receive=receive)

    def test_profiles_match_hanoi_models_and_wire_parameters(self):
        for profile, name in (("sol", "gpt-6.1-sol"), ("opus", "claude-opus-5-5"), ("sonnet", "claude-sonnet-5-5")):
            settings = model_runner.load_settings(profile=profile)
            self.assertEqual(settings["model"], name)
            request = providers.make_request(settings, "system", [{"role": "user", "content": "input"}])
            self.assertNotIn("temperature", request)
            self.assertEqual(request["tools"], [])
            if profile == "sol":
                self.assertFalse(request["store"])
                self.assertEqual(request["reasoning"], {"effort": "medium", "context": "all_turns"})
                self.assertEqual(request["include"], ["reasoning.encrypted_content"])
            else:
                self.assertEqual(request["thinking"], {"type": "adaptive", "display": "omitted"})
                self.assertEqual(request["output_config"], {"effort": "medium"})

    def test_native_context_crosses_rounds_and_resets_at_new_problem(self):
        requests = []
        def receive(client, provider, request, path):
            requests.append(request)
            return response(marker=f"opaque-{len(requests)}")
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            items = [problem("one", 2), problem("two")]
            report = self.execute(items, directory, receive)
            self.assertEqual(len(report["submissions"]), 3)
            self.assertTrue(report["complete"])
            self.assertTrue(all(row["feedback"] == "Correct" for row in report["submissions"]))
            self.assertEqual(len(requests[0]["input"]), 1)
            self.assertTrue(any(item.get("encrypted_content") == "opaque-1" for item in requests[1]["input"]))
            self.assertEqual(len(requests[2]["input"]), 1)
            self.assertEqual(requests[1]["input"][-2], {"role": "user", "content": "Correct"})
            self.assertEqual(model_runner.read_json(directory/"checkpoint.json")["responses"],
                             ['{"moves":"A1 B1","answer":""}']*3)
            resumed = self.execute(items, directory, lambda *args: self.fail("completed run regenerated"),
                                   resume=True)
            self.assertEqual(resumed, report)

    def test_unified_instructions_stay_at_top_level_for_both_providers(self):
        initial = {"role": "user", "content": 'Initial string: "AABB"'}
        for profile, history_key, instruction_key in (
                ("sol", "input", "instructions"), ("opus", "messages", "system")):
            with self.subTest(profile=profile), TemporaryDirectory() as temporary:
                self.settings = model_runner.load_settings(profile=profile)
                self.settings.update(max_attempts=2, required_streak=1)
                requests = []
                raw = response()
                if profile == "opus":
                    raw = {"type": "message", "role": "assistant", "stop_reason": "end_turn",
                           "content": [{"type": "thinking", "thinking": "", "signature": "original-signature"},
                                       {"type": "text", "text": '{"moves":"A1 B1","answer":""}'}]}

                def receive(client, provider, request, path):
                    requests.append(request)
                    return raw

                report = self.execute([problem("one", 2), problem("two")], Path(temporary), receive)
                self.assertTrue(report["complete"])
                for request in requests:
                    self.assertEqual(request[instruction_key], INSTRUCTIONS)
                    self.assertFalse(any(item.get("role") == "system" or item.get("content") == INSTRUCTIONS
                                         for item in request[history_key]))
                    self.assertEqual(request[history_key][-1], initial)
                self.assertEqual(requests[0][history_key], [initial])
                self.assertEqual(requests[1][history_key],
                                 [initial] + providers.continuation_items(raw, self.settings["provider"])
                                 + [{"role": "user", "content": "Correct"}, initial])
                self.assertEqual(requests[2][history_key], [initial])

    def test_previous_rule_compliant_error_is_private_and_persisted(self):
        item = {"id": "one", "type": "same_letter", "rounds": [
            {"round": 1, "input": "AABB", "target": 0, "forbidden": "AA"},
            {"round": 2, "input": "AABBAABB", "target": 0, "forbidden": "BB"},
            {"round": 3, "input": "AABB", "target": 0, "forbidden": "CC"},
        ]}
        self.settings["reasoning_summary"] = "auto"
        self.settings["max_attempts"] = 10
        requests = []

        def receive(client, provider, request, path):
            requests.append(request)
            moves = {2: "B1 B1 A1", 3: "A1 A1 B1"}.get(len(requests), "A1 B1")
            return response(json.dumps({"moves": moves, "answer": ""}))

        with TemporaryDirectory() as temporary, redirect_stderr(StringIO()) as terminal:
            directory = Path(temporary)
            report = model_runner.execute([item], self.settings, directory, client=object(), receive=receive)
            second = report["submissions"][1]
            self.assertTrue(report["complete"])
            self.assertEqual(second["feedback"], "Correct, but rule violation. Rule violation: move 2.")
            self.assertFalse(second["correct"])
            self.assertEqual(second["success_streak"], 0)
            self.assertTrue(second["persistent_error"])
            self.assertTrue(second["previous_rule_compliant"])
            self.assertEqual(second["previous_violation_steps"], [])
            self.assertEqual(second["violation_steps"], [2])
            third = report["submissions"][2]
            self.assertTrue(third["correct"])
            self.assertFalse(third["persistent_error"])
            self.assertEqual(third["previous_violation_steps"], [2])
            self.assertEqual(report["summary"]["mpe"], .05)
            self.assertEqual(report["summary"]["observed_mpe"], .25)
            self.assertEqual(report["summary"]["persistent_error_basis"], "previous_rule_compliant_current_incorrect")
            self.assertEqual(report["summary"]["per_denominator"], "max_attempts")
            protocol = model_runner.read_json(directory / "protocol.json")
            self.assertEqual(protocol["persistent_error_basis"], "previous_rule_compliant_current_incorrect")
            self.assertEqual(protocol["per_denominator"], "max_attempts")
            saved = model_runner.read_json(directory / "submissions" / "00002_record.json")["score"]
            self.assertEqual(saved, second)
            diagnostic = model_runner.read_json(directory / "submissions" / "00002_reasoning_summary.json")
            self.assertTrue(diagnostic["persistent_error"])
            self.assertEqual(diagnostic["previous_violation_steps"], [])
            self.assertIn("이전 규칙 위반 행동: 2", (directory / "reasoning_log.txt").read_text())
            for label in ("이전 규칙 위반", "2번째 행동", "지속 오류", "발생", "PE=1"):
                self.assertIn(label, terminal.getvalue())
            self.assertEqual(requests[2]["input"][-2], {"role": "user", "content": second["feedback"]})
            self.assertEqual(requests[3]["input"][-2], {"role": "user", "content": "Correct"})
            self.assertNotIn("previous_violation_steps", json.dumps(requests))
            self.assertNotIn("previous_rule_violation", json.dumps(requests))
            self.assertNotIn("previous_rule_compliant_current_incorrect", json.dumps(requests))

    def test_all_violation_positions_reach_both_providers_with_native_history(self):
        item = {"id": "one", "type": "same_letter", "rounds": [
            {"round": 1, "input": "AABBAABBAA", "target": 0, "forbidden": "AA"},
        ]}
        feedback = "Correct, but rule violation. Rule violation: moves 2, 3."
        initial = {"role": "user", "content": 'Initial string: "AABBAABBAA"'}
        for profile, history_key in (("sol", "input"), ("opus", "messages")):
            with self.subTest(profile=profile), TemporaryDirectory() as temporary:
                self.settings = model_runner.load_settings(profile=profile)
                self.settings.update(max_attempts=2, required_streak=1)
                requests, outputs = [], []

                def receive(client, provider, request, path):
                    requests.append(request)
                    moves = "A1 A1 A1 B1" if len(requests) == 1 else "B1 B1 A1"
                    text = json.dumps({"moves": moves, "answer": ""})
                    raw = response(text)
                    if provider == "anthropic":
                        raw = {"type": "message", "role": "assistant", "stop_reason": "end_turn",
                               "content": [{"type": "thinking", "thinking": "", "signature": "opaque-signature"},
                                           {"type": "text", "text": text}]}
                    outputs.append(raw)
                    return raw

                report = self.execute([item], Path(temporary), receive)
                self.assertTrue(report["complete"])
                self.assertEqual(report["submissions"][0]["violation_steps"], [2, 3])
                self.assertEqual([row["feedback"] for row in report["submissions"]], [feedback, "Correct"])
                self.assertEqual(requests[1][history_key],
                                 [initial] + providers.continuation_items(outputs[0], self.settings["provider"])
                                 + [{"role": "user", "content": feedback}, initial])

    def test_saved_response_is_reused_after_checkpoint_interruption(self):
        calls = []
        def receive(*args):
            calls.append(1)
            return response()
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            real_save = model_runner.save_progress
            def interrupt_after_response(directory, session, item, history):
                if session.responses:
                    raise OSError("simulated interruption")
                real_save(directory, session, item, history)
            with patch.object(model_runner, "save_progress", side_effect=interrupt_after_response):
                with self.assertRaises(OSError):
                    self.execute([problem("one")], directory, receive)
            self.assertEqual(model_runner.read_json(directory/"checkpoint.json")["responses"], [])
            self.assertTrue((directory/"submissions"/"00001_response.json").exists())
            report = self.execute([problem("one")], directory, receive, resume=True)
            self.assertTrue(report["complete"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(report["submissions"]), 1)

    def test_transport_failure_is_ungraded_and_resume_recovers(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(providers.ProviderError):
                self.execute([problem("one")], directory,
                             lambda *args: (_ for _ in ()).throw(providers.ProviderError("connection stopped")))
            saved = model_runner.read_json(directory/"checkpoint.json")
            self.assertEqual(saved["responses"], [])
            report = self.execute([problem("one")], directory, lambda *args: response(), resume=True)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["submissions"]), 1)
            self.assertEqual(len(list((directory/"submissions").glob("*_started.json"))), 2)

    def test_terminal_stream_is_recovered_before_another_request(self):
        for provider in ("openai", "anthropic"):
            for finished in (True, False):
                with self.subTest(provider=provider, finished=finished), TemporaryDirectory() as temporary:
                    self.settings["provider"] = provider
                    directory = Path(temporary)
                    raw = response(status="completed" if finished else "incomplete")
                    event = {"type": f"response.{raw['status']}", "response": raw}
                    if provider == "anthropic":
                        raw = {"type": "message", "role": "assistant",
                               "stop_reason": "end_turn" if finished else "max_tokens",
                               "content": [{"type": "text", "text": '{"moves":"A1 B1","answer":""}'}]}
                        event = {"type": "message_stop", "message": raw}
                    def interrupt_after_event(client, name, request, stream_path):
                        stream_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
                        raise KeyboardInterrupt
                    with self.assertRaises(KeyboardInterrupt):
                        self.execute([problem("one")], directory, interrupt_after_event)
                    self.assertFalse((directory/"submissions"/"00001_response.json").exists())
                    report = self.execute([problem("one")], directory,
                                          lambda *args: self.fail("completed stream regenerated"), resume=True)
                    self.assertTrue(report["complete"])
                    self.assertEqual(len(report["submissions"]), int(finished))
                    envelope = model_runner.read_json(directory/"submissions"/"00001_response.json")
                    self.assertEqual(envelope["response"], raw)
                    self.assertEqual(envelope["recovered_from_stream"], "00001_transport_001_stream.jsonl")
                    self.assertEqual(len(list((directory/"submissions").glob("*_started.json"))), 1)

    def test_partial_stream_never_becomes_a_response(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"partial.jsonl"
            path.write_text(json.dumps({"type": "response.output_text.delta", "delta": "partial"}) +
                            '\n{"type":"response.completed","response":', encoding="utf-8")
            self.assertIsNone(providers.recover_response(path, "openai"))
            path.write_text(json.dumps({"type": "content_block_stop", "index": 0}) + "\n" +
                            json.dumps({"type": "message_stop"}) + "\n", encoding="utf-8")
            self.assertIsNone(providers.recover_response(path, "anthropic"))

    def test_incomplete_generation_skips_problem_without_grading(self):
        calls = []
        def receive(*args):
            calls.append(1)
            return response(status="incomplete") if len(calls) == 1 else response()
        with TemporaryDirectory() as temporary:
            report = self.execute([problem("one", 2), problem("two")], Path(temporary), receive)
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["submissions"]), 1)
        self.assertEqual(report["problems"][0]["rounds"][0]["status"], "generation_failure")
        self.assertEqual(report["problems"][0]["rounds"][1]["status"], "excluded_after_generation_failure")

    def test_anthropic_thinking_signatures_are_kept_and_not_graded(self):
        raw = {"type": "message", "role": "assistant", "stop_reason": "end_turn", "content": [
            {"type": "thinking", "thinking": "", "signature": "original-signature"},
            {"type": "text", "text": '{"moves":"A1 B1","answer":""}'},
        ]}
        self.assertEqual(providers.visible_text(raw, "anthropic"), ('{"moves":"A1 B1","answer":""}', None))
        native = providers.continuation_items(raw, "anthropic")
        self.assertEqual(native, [{"role": "assistant", "content": raw["content"]}])
        native[0]["content"][0]["signature"] = "modified-copy"
        self.assertEqual(raw["content"][0]["signature"], "original-signature")


if __name__ == "__main__":
    unittest.main()
