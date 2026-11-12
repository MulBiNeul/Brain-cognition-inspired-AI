"""Transient transport recovery must not change evaluation attempts or context."""

from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from src import model_runner, providers


def complete_response(marker="original-state"):
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "encrypted_content": marker, "summary": []},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": '{"moves":"A1 B1","answer":""}'},
            ]},
        ],
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }


def item(rounds=1):
    return {"id": "transport-test", "type": "same_letter", "rounds": [
        {"round": number, "input": "AABB", "target": 0, "forbidden": "AA"}
        for number in range(1, rounds + 1)
    ]}


class TransportRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.settings = model_runner.load_settings(profile="sol")
        self.settings.update(max_attempts=2, required_streak=1, timeout=180.0)

    def execute(self, directory, receive, *, rounds=1, client=None, **options):
        options.setdefault("retry_delay", 0)
        with redirect_stderr(StringIO()), patch.object(model_runner.random, "uniform", return_value=0):
            return model_runner.execute(
                [item(rounds)], self.settings, directory,
                client=object() if client is None else client, receive=receive, **options,
            )

    def test_partial_stream_retry_keeps_request_history_and_counts_once(self):
        requests = []

        def receive(client, provider, request, path):
            requests.append(deepcopy(request))
            if len(requests) == 2:
                path.write_text(json.dumps({
                    "type": "response.output_text.delta", "delta": "PARTIAL_SHOULD_NOT_ENTER_CONTEXT",
                }) + "\n", encoding="utf-8")
                raise providers.ProviderError("stream stopped", retryable=True, kind="incomplete_stream")
            return complete_response(f"state-{len(requests)}")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = self.execute(directory, receive, rounds=2)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["submissions"]), 2)
            self.assertEqual([row["attempt"] for row in report["submissions"]], [1, 1])
            self.assertEqual(requests[1], requests[2])
            self.assertTrue(any(block.get("encrypted_content") == "state-1"
                                for block in requests[2]["input"]))
            self.assertNotIn("PARTIAL_SHOULD_NOT_ENTER_CONTEXT", json.dumps(requests[2]))
            self.assertEqual(len(model_runner.read_json(directory / "checkpoint.json")["responses"]), 2)
            trials = directory / "submissions"
            self.assertEqual(len(list(trials.glob("00002_transport_*_started.json"))), 2)
            errors = list(trials.glob("00002_transport_*_error.json"))
            self.assertEqual(len(errors), 1)
            error = model_runner.read_json(errors[0])
            self.assertEqual(error["kind"], "incomplete_stream")
            self.assertTrue(error["retryable"])
            self.assertTrue(error["will_retry"])
            self.assertEqual(len(list(trials.glob("*_record.json"))), 2)

    def test_exhausted_transport_retries_preserve_attempt_for_resume(self):
        requests = []

        def unavailable(client, provider, request, path):
            requests.append(deepcopy(request))
            raise providers.ProviderError("timeout", retryable=True, kind="timeout")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(providers.ProviderError):
                self.execute(directory, unavailable, transport_retries=3)
            self.assertEqual(len(requests), 4)
            self.assertTrue(all(request == requests[0] for request in requests))
            self.assertEqual(model_runner.read_json(directory / "checkpoint.json")["responses"], [])
            saved_report = model_runner.read_json(directory / "report.json")
            self.assertFalse(saved_report["complete"])
            self.assertEqual(saved_report["submissions"], [])
            trials = directory / "submissions"
            errors = [model_runner.read_json(path) for path in sorted(trials.glob("*_error.json"))]
            self.assertEqual(len(errors), 4)
            self.assertEqual([error["will_retry"] for error in errors], [True, True, True, False])
            self.assertFalse(list(trials.glob("*_response.json")))

            resumed_requests = []

            def restored(client, provider, request, path):
                resumed_requests.append(deepcopy(request))
                return complete_response()

            report = self.execute(directory, restored, resume=True)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["submissions"]), 1)
            self.assertEqual(report["submissions"][0]["attempt"], 1)
            self.assertEqual(resumed_requests, [requests[0]])
            self.assertEqual(len(list(trials.glob("*_started.json"))), 5)

    def test_authentication_failure_is_not_retried(self):
        calls = []

        def denied(*args):
            calls.append(1)
            raise providers.ProviderError("authentication failed", status_code=401, kind="authentication")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(providers.ProviderError):
                self.execute(directory, denied, transport_retries=3)
            self.assertEqual(len(calls), 1)
            self.assertEqual(model_runner.read_json(directory / "checkpoint.json")["responses"], [])
            errors = list((directory / "submissions").glob("*_error.json"))
            self.assertEqual(len(errors), 1)
            self.assertFalse(model_runner.read_json(errors[0])["will_retry"])

    def test_keyboard_interrupt_stops_without_retry_or_attempt_consumption(self):
        calls = []

        def interrupted(*args):
            calls.append(1)
            raise KeyboardInterrupt

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(KeyboardInterrupt):
                self.execute(directory, interrupted, transport_retries=3)
            self.assertEqual(len(calls), 1)
            self.assertEqual(model_runner.read_json(directory / "checkpoint.json")["responses"], [])
            self.assertEqual(model_runner.read_json(directory / "report.json")["submissions"], [])

    def test_completed_event_is_recovered_before_retrying_request(self):
        calls = []
        raw = complete_response()

        def completed_then_disconnected(client, provider, request, path):
            calls.append(1)
            path.write_text(json.dumps({"type": "response.completed", "response": raw}) + "\n",
                            encoding="utf-8")
            raise providers.ProviderError("connection closed", retryable=True, kind="connection")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = self.execute(directory, completed_then_disconnected)
            self.assertTrue(report["complete"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(report["submissions"]), 1)
            envelope = model_runner.read_json(directory / "submissions" / "00001_response.json")
            self.assertEqual(envelope["response"], raw)
            self.assertIn("recovered_from_stream", envelope)

    def test_timeout_override_only_changes_transport_not_frozen_protocol(self):
        original_settings = deepcopy(self.settings)
        client = MagicMock()
        requests = []

        def receive(client, provider, request, path):
            requests.append(deepcopy(request))
            return complete_response()

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with redirect_stderr(StringIO()), patch.object(providers, "make_client", return_value=client) as factory:
                report = model_runner.execute([item()], self.settings, directory, receive=receive,
                                              timeout=600.0, retry_delay=0)
            self.assertTrue(report["complete"])
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.call_args.args[0]["timeout"], 600.0)
            self.assertEqual(self.settings, original_settings)
            self.assertEqual(model_runner.read_json(directory / "protocol.json")["settings"], original_settings)
            self.assertNotIn("timeout", requests[0])
            client.close.assert_called_once()

    def test_server_retry_after_is_honored(self):
        calls = []

        def throttled_once(*args):
            calls.append(1)
            if len(calls) == 1:
                raise providers.ProviderError("rate limit", retryable=True, kind="rate_limit",
                                              status_code=429, retry_after=17.0)
            return complete_response()

        with TemporaryDirectory() as temporary, patch.object(model_runner.time, "sleep") as sleep:
            report = self.execute(Path(temporary), throttled_once)
            self.assertTrue(report["complete"])
            self.assertEqual(len(calls), 2)
            self.assertGreaterEqual(sum(call.args[0] for call in sleep.call_args_list), 17.0)

    def test_programming_error_does_not_regenerate_a_response(self):
        calls = []

        def broken(*args):
            calls.append(1)
            raise ValueError("local bug")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(ValueError):
                self.execute(directory, broken, transport_retries=3)
            self.assertEqual(len(calls), 1)
            self.assertEqual(model_runner.read_json(directory / "checkpoint.json")["responses"], [])

    def test_resume_compatibility_requires_exact_reviewed_from_and_to(self):
        with TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "configs").mkdir()
            manifest = project / "configs" / "resume_compatibility.json"
            with patch.object(model_runner, "PROJECT_DIR", project), \
                    patch.object(model_runner, "implementation_fingerprint", return_value="current"):
                self.assertTrue(model_runner.compatible_fingerprint("current", self.settings))
                self.assertFalse(model_runner.compatible_fingerprint("previous", self.settings))
                manifest.write_text(json.dumps({"transport_only_upgrades": [
                    {"from": "previous", "to": "different-code"},
                    {"from": "unrelated-run", "to": "current"},
                ]}), encoding="utf-8")
                self.assertFalse(model_runner.compatible_fingerprint("previous", self.settings))
                manifest.write_text(json.dumps({"transport_only_upgrades": [
                    {"from": "previous", "to": "current"},
                ]}), encoding="utf-8")
                self.assertTrue(model_runner.compatible_fingerprint("previous", self.settings))
                self.assertFalse(model_runner.compatible_fingerprint("old-feedback-version", self.settings))
            with patch.object(model_runner, "PROJECT_DIR", project), \
                    patch.object(model_runner, "implementation_fingerprint", return_value="unreviewed-new-code"):
                self.assertFalse(model_runner.compatible_fingerprint("previous", self.settings))

    def test_unreviewed_resume_fails_before_any_provider_request(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.execute(directory, lambda *args: complete_response())
            protocol_path = directory / "protocol.json"
            protocol = model_runner.read_json(protocol_path)
            protocol["fingerprint"] = "a-different-feedback-implementation"
            model_runner.write_json(protocol_path, protocol)
            with self.assertRaises(ValueError):
                self.execute(directory, lambda *args: self.fail("unreviewed resume called the API"), resume=True)


if __name__ == "__main__":
    unittest.main()
