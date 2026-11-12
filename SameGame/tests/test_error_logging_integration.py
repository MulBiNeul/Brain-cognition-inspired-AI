"""Errors retain evaluation context and safe evidence without changing scoring."""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import model_runner, providers
from tests.test_transport_recovery import complete_response, item


def events(directory):
    return [json.loads(line) for line in (directory / "run_events.jsonl").read_text().splitlines()]


class ErrorLoggingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.settings = model_runner.load_settings(profile="sol")
        self.settings.update(max_attempts=2, required_streak=1)

    def execute(self, directory, receive, *, items=None):
        with redirect_stderr(StringIO()), patch.object(model_runner.time, "sleep"):
            return model_runner.execute(items or [item()], self.settings, directory,
                                        client=object(), receive=receive)

    def test_transport_details_and_context_survive_retry_and_recovery(self):
        calls = []

        def receive(*args):
            calls.append(1)
            if len(calls) == 1:
                raise providers.ProviderError(
                    "API request interrupted", retryable=True, kind="timeout", status_code=408,
                    exception_type="APITimeoutError", cause_types=["ReadTimeout"],
                    request_id="req_diagnostic", error_codes=["request_timeout"],
                    stream={"event_count": 12, "last_event_type": "response.output_text.delta",
                            "seconds_since_last_event": 180.0, "response_id": "resp_diagnostic"},
                )
            return complete_response()

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = self.execute(directory, receive)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["submissions"]), 1)
            self.assertEqual(len(calls), 2)
            error_file = directory / "submissions" / "00001_transport_001_error.json"
            saved = model_runner.read_json(error_file)
            for key, value in {"id": "transport-test", "round": 1, "attempt": 1,
                               "submission": 1, "transport_attempt": 1, "retry_limit": 10,
                               "exception_type": "APITimeoutError", "request_id": "req_diagnostic",
                               "status_code": 408, "error_codes": ["request_timeout"]}.items():
                self.assertEqual(saved[key], value)
            self.assertIn("ReadTimeout", saved["cause_types"])
            self.assertEqual(saved["stream"]["event_count"], 12)
            self.assertEqual(saved["stream"]["last_event_type"], "response.output_text.delta")
            self.assertTrue(saved["traceback_frames"])
            recorded = events(directory)
            error = next(event for event in recorded if event["event"] == "transport_error")
            self.assertEqual(error["request_id"], saved["request_id"])
            successful = [event for event in recorded if event["event"] == "transport_success"]
            self.assertEqual(len(successful), 1)
            self.assertEqual(successful[0]["transport_attempt"], 2)
            self.assertEqual(recorded[-1]["event"], "complete")
            readable = (directory / "errors.log").read_text()
            for text in ("transport-test", "R1", "APITimeoutError", "ReadTimeout", "req_diagnostic",
                         "response.output_text.delta"):
                self.assertIn(text, readable)

    def test_default_ten_retries_log_each_failure_without_grading(self):
        calls = []

        def unavailable(*args):
            calls.append(1)
            raise providers.ProviderError("stream ended", retryable=True, kind="incomplete_stream",
                                          exception_type="IncompleteStream")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(providers.ProviderError):
                self.execute(directory, unavailable)
            self.assertEqual(len(calls), 11)
            failures = [event for event in events(directory) if event["event"] == "transport_error"]
            self.assertEqual(len(failures), 11)
            self.assertEqual([event["will_retry"] for event in failures], [True] * 10 + [False])
            self.assertTrue(all(event["attempt"] == 1 for event in failures))
            self.assertEqual(model_runner.read_json(directory / "checkpoint.json")["responses"], [])
            self.assertEqual(events(directory)[-1]["event"], "stopped")

    def test_authentication_failure_records_the_current_evaluation_location(self):
        def denied(*args):
            raise providers.ProviderError("authentication failed", kind="authentication",
                                          status_code=401, exception_type="AuthenticationError")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(providers.ProviderError):
                self.execute(directory, denied)
            failures = [event for event in events(directory) if event["event"] == "transport_error"]
            self.assertEqual(len(failures), 1)
            self.assertFalse(failures[0]["will_retry"])
            stopped = events(directory)[-1]
            self.assertEqual(stopped["kind"], "authentication")
            self.assertEqual(stopped["exception_type"], "AuthenticationError")
            self.assertEqual((stopped["id"], stopped["round"], stopped["attempt"]),
                             ("transport-test", 1, 1))

    def test_keyboard_interrupt_has_a_distinct_record(self):
        def interrupted(*args):
            raise KeyboardInterrupt

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(KeyboardInterrupt):
                self.execute(directory, interrupted)
            event = events(directory)[-1]
            self.assertEqual(event["event"], "interrupted")
            self.assertEqual(event["exception_type"], "KeyboardInterrupt")
            self.assertEqual(event["phase"], "receive")
            self.assertEqual(event["attempt"], 1)
            self.assertIn("KeyboardInterrupt", (directory / "errors.log").read_text())

    def test_unexpected_local_error_keeps_stack_locations_without_exception_value(self):
        secret = "PRIVATE_EXCEPTION_VALUE_DO_NOT_LOG"

        def broken(*args):
            raise RuntimeError(secret)

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(RuntimeError):
                self.execute(directory, broken)
            event = events(directory)[-1]
            self.assertEqual(event["kind"], "local_error")
            self.assertEqual(event["exception_type"], "RuntimeError")
            self.assertTrue(any(frame["function"] == "broken" for frame in event["traceback_frames"]))
            self.assertTrue(all(set(frame) == {"file", "line", "function"}
                                for frame in event["traceback_frames"]))
            self.assertNotIn(secret, (directory / "run_events.jsonl").read_text())
            self.assertNotIn(secret, (directory / "errors.log").read_text())

    def test_generation_failure_is_logged_before_advancing_to_next_problem(self):
        first, second = item(rounds=2), item()
        first["id"], second["id"] = "SG-first", "SG-next"
        calls = []

        def receive(*args):
            calls.append(1)
            raw = complete_response()
            if len(calls) == 1:
                raw.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
            return raw

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = self.execute(directory, receive, items=[first, second])
            event = next(event for event in events(directory) if event["event"] == "generation_failure")
            self.assertEqual((event["id"], event["round"], event["attempt"], event["submission"]),
                             ("SG-first", 1, 1, 1))
            self.assertEqual(event["failure"], "openai_incomplete")
            self.assertEqual(report["problems"][0]["rounds"][0]["status"], "generation_failure")
            self.assertEqual(len(report["submissions"]), 1)
            self.assertEqual(report["submissions"][0]["id"], "SG-next")
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
