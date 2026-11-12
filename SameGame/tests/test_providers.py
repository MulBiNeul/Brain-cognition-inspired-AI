"""Transport failures are retryable only when safe, without losing final events."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anthropic
import httpx
import httpx2
import openai

from src import providers


class Event:
    def __init__(self, value):
        self.value = value

    def model_dump(self, **kwargs):
        return self.value


class FakeStream:
    def __init__(self, events=(), *, error=None, final=None):
        self.events = events
        self.error = error
        self.final = final
        self.final_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        for event in self.events:
            yield Event(event)
        if self.error is not None:
            raise self.error

    def get_final_message(self):
        self.final_calls += 1
        return Event(self.final)


class ProviderTransportTests(unittest.TestCase):
    def receive(self, stream, provider="openai", path=None):
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: stream),
                                 messages=SimpleNamespace(stream=lambda **kwargs: stream))
        if path is not None:
            return providers.receive(client, provider, {}, path)
        with TemporaryDirectory() as directory:
            return providers.receive(client, provider, {}, Path(directory)/"events.jsonl")

    def test_eof_and_partial_text_are_retryable_without_a_final(self):
        events = [{"type": "response.output_text.delta", "delta": '{"moves":"A1"'}]
        for stream in (FakeStream(), FakeStream(events)):
            with self.subTest(events=stream.events), self.assertRaises(providers.ProviderError) as caught:
                self.receive(stream)
            self.assertTrue(caught.exception.retryable)
            self.assertEqual(caught.exception.kind, "incomplete_stream")

    def test_anthropic_requires_message_stop_before_using_snapshot(self):
        partial = {"stop_reason": None, "content": [{"type": "text", "text": "partial"}]}
        stream = FakeStream([{"type": "content_block_stop"}], final=partial)
        with self.assertRaises(providers.ProviderError) as caught:
            self.receive(stream, "anthropic")
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(stream.final_calls, 0)

    def test_complete_response_survives_transport_error_after_final_event(self):
        for provider in ("openai", "anthropic"):
            raw = {"status": "completed", "output": []} if provider == "openai" else {
                "stop_reason": "end_turn", "content": []}
            event = {"type": "response.completed", "response": raw} if provider == "openai" else {
                "type": "message_stop", "message": raw}
            stream = FakeStream([event], error=httpx.RemoteProtocolError("private details"))
            with self.subTest(provider=provider):
                self.assertEqual(self.receive(stream, provider), raw)

    def test_keyboard_interrupt_and_local_errors_are_not_retried(self):
        final = {"type": "response.completed", "response": {"status": "completed"}}
        with self.assertRaises(KeyboardInterrupt):
            self.receive(FakeStream([final], error=KeyboardInterrupt()))
        for error in (ValueError("private details"), OSError("disk full")):
            with self.subTest(error=type(error).__name__), self.assertRaises(providers.ProviderError) as caught:
                self.receive(FakeStream([final], error=error))
            self.assertFalse(caught.exception.retryable)
            self.assertNotIn(str(error), str(caught.exception))

    def test_sdk_and_stream_network_timeouts_are_retryable(self):
        request = httpx.Request("POST", "https://example.invalid")
        request2 = httpx2.Request("POST", "https://example.invalid")
        cases = [
            (openai.APITimeoutError(request), "timeout"),
            (openai.APIConnectionError(request=request), "connection"),
            (anthropic.APITimeoutError(request2), "timeout"),
            (anthropic.APIConnectionError(request=request2), "connection"),
            (httpx.ReadTimeout("private details"), "timeout"),
            (httpx.RemoteProtocolError("private details"), "connection"),
            (httpx2.ReadTimeout("private details"), "timeout"),
            (httpx2.RemoteProtocolError("private details"), "connection"),
        ]
        for error, kind in cases:
            with self.subTest(error=type(error)), self.assertRaises(providers.ProviderError) as caught:
                self.receive(FakeStream(error=error))
            self.assertTrue(caught.exception.retryable)
            self.assertEqual(caught.exception.kind, kind)
            self.assertNotIn("private details", str(caught.exception))

    def test_http_retry_policy_excludes_authentication_and_quota_errors(self):
        for status in (400, 401, 403, 404, 408, 409, 422, 429, 500, 503):
            response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"),
                                      headers={"Retry-After": "12", "Authorization": "secret"})
            error = openai.APIStatusError("private details", response=response, body=None)
            with self.subTest(status=status), self.assertRaises(providers.ProviderError) as caught:
                self.receive(FakeStream(error=error))
            retryable = status in (408, 409, 429, 500, 503)
            self.assertEqual(caught.exception.retryable, retryable)
            self.assertEqual(caught.exception.status_code, status)
            self.assertEqual(caught.exception.retry_after, 12 if retryable else None)
            self.assertNotIn("secret", str(caught.exception))
        for code in ("insufficient_quota", "billing_hard_limit_reached", "authentication_error"):
            response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
            error = openai.RateLimitError("private details", response=response,
                                         body={"error": {"code": code}})
            with self.subTest(code=code), self.assertRaises(providers.ProviderError) as caught:
                self.receive(FakeStream(error=error))
            self.assertFalse(caught.exception.retryable)

    def test_retry_after_formats_and_no_retry_header(self):
        cases = [({"retry-after-ms": "1250", "retry-after": "99"}, 1.25),
                 ({"retry-after": "7.5"}, 7.5), ({"retry-after": "nan"}, None),
                 ({"retry-after": "inf"}, None), ({"retry-after": "-1"}, None)]
        for headers, expected in cases:
            with self.subTest(headers=headers):
                self.assertEqual(providers._retry_after(headers), expected)
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
        delay = providers._retry_after({"retry-after": future})
        self.assertGreater(delay, 28)
        self.assertLessEqual(delay, 30)
        response = httpx.Response(503, request=httpx.Request("POST", "https://example.invalid"),
                                  headers={"x-should-retry": "false"})
        error = openai.APIStatusError("private details", response=response, body=None)
        self.assertFalse(providers._classify_exception(error).retryable)

    def test_failed_transient_events_are_retried_and_never_recovered_as_answers(self):
        for code in ("server_error", "rate_limit_exceeded", "overloaded_error"):
            raw = {"status": "failed", "error": {"code": code, "message": "private details"}}
            event = {"type": "response.failed", "response": raw}
            with self.subTest(code=code), TemporaryDirectory() as directory:
                path = Path(directory)/"events.jsonl"
                with self.assertRaises(providers.ProviderError) as caught:
                    self.receive(FakeStream([event]), path=path)
                self.assertTrue(caught.exception.retryable)
                self.assertIsNone(providers.recover_response(path, "openai"))
        event = {"type": "error", "error": {"type": "overloaded_error", "message": "private details"}}
        with self.assertRaises(providers.ProviderError) as caught:
            self.receive(FakeStream([event]))
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn("private details", str(caught.exception))

    def test_nontransient_generation_failures_keep_existing_policy(self):
        for raw in ({"status": "incomplete"}, {"status": "failed", "error": {"code": "invalid_prompt"}}):
            event = {"type": "response." + raw["status"], "response": raw}
            with self.subTest(raw=raw), TemporaryDirectory() as directory:
                path = Path(directory)/"events.jsonl"
                self.assertEqual(self.receive(FakeStream([event]), path=path), raw)
                self.assertEqual(providers.recover_response(path, "openai"), raw)

    def test_original_exception_and_causes_survive_wrapping_without_private_details(self):
        request = httpx.Request("POST", "https://example.invalid")
        root_cause = OSError("private network address and credentials")
        timeout = httpx.ReadTimeout("private transport text", request=request)
        timeout.__cause__ = root_cause
        error = openai.APITimeoutError(request)
        error.__cause__ = timeout
        with self.assertRaises(providers.ProviderError) as caught:
            self.receive(FakeStream(error=error))
        diagnostic = caught.exception
        self.assertEqual(diagnostic.exception_type, "APITimeoutError")
        self.assertEqual(diagnostic.cause_types, ["ReadTimeout", "OSError"])
        self.assertTrue(diagnostic.__suppress_context__)
        self.assertNotIn("private", json.dumps(vars(diagnostic)))
        self.assertNotIn("private", str(diagnostic))

    def test_http_error_keeps_only_allowed_request_id_and_codes(self):
        reply = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"), headers={
            "x-request-id": "req_diagnostic_001", "Authorization": "Bearer private-token",
        })
        error = openai.RateLimitError("private body text", response=reply,
                                     body={"error": {"code": "insufficient_quota", "message": "private prompt"}})
        with self.assertRaises(providers.ProviderError) as caught:
            self.receive(FakeStream(error=error))
        diagnostic = caught.exception
        self.assertEqual(diagnostic.request_id, "req_diagnostic_001")
        self.assertEqual(diagnostic.error_codes, ["insufficient_quota"])
        self.assertEqual(diagnostic.kind, "quota")
        self.assertFalse(diagnostic.retryable)
        self.assertNotIn("private", json.dumps(vars(diagnostic)))

    def test_diagnostic_fields_reject_headers_payload_and_credentials(self):
        with patch.dict(providers.os.environ, {"OPENAI_API_KEY": "secret-for-test"}):
            error = providers.ProviderError("safe", exception_type="ValueError\nprivate text",
                cause_types=["ReadTimeout", "private body content"], request_id="req_secret-for-test",
                error_codes=["rate_limit_error", "private prompt", "sk_api_key"],
                stream={"event_count": 2, "last_event_type": "response.output_text.delta",
                        "seconds_since_last_event": float("inf"), "response_id": "resp_secret-for-test",
                        "delta": "private prompt", "Authorization": "private header"})
        self.assertIsNone(error.exception_type)
        self.assertEqual(error.cause_types, ["ReadTimeout"])
        self.assertIsNone(error.request_id)
        self.assertEqual(error.error_codes, ["rate_limit_error"])
        self.assertIsNone(error.stream["response_id"])
        self.assertIsNone(error.stream["seconds_since_last_event"])
        self.assertNotIn("private", json.dumps(vars(error)))
        self.assertNotIn("secret-for-test", json.dumps(vars(error)))

    def test_incomplete_stream_has_event_timing_and_response_identifier(self):
        events = [{"type": "response.created", "response": {"id": "resp_diagnostic_001"}},
                  {"type": "response.output_text.delta", "delta": "private partial answer"}]
        with patch.object(providers.time, "monotonic", side_effect=[5.0, 6.0, 9.5]):
            with self.assertRaises(providers.ProviderError) as caught:
                self.receive(FakeStream(events))
        error = caught.exception
        self.assertEqual(error.exception_type, "IncompleteStream")
        self.assertEqual(error.stream, {"event_count": 2, "last_event_type": "response.output_text.delta",
                                       "seconds_since_last_event": 3.5, "response_id": "resp_diagnostic_001"})
        self.assertNotIn("private", json.dumps(vars(error)))

    def test_stream_error_kind_preserves_retry_decision(self):
        for code, kind, retryable in (("authentication_error", "authentication", False),
                                      ("permission_error", "permission", False),
                                      ("insufficient_quota", "quota", False),
                                      ("invalid_request_error", "invalid_request", False),
                                      ("rate_limit_exceeded", "rate_limit", True),
                                      ("overloaded_error", "server_error", True)):
            with self.subTest(code=code):
                event = {"type": "error", "error": {"type": code, "message": "private prompt"}}
                with self.assertRaises(providers.ProviderError) as caught:
                    self.receive(FakeStream([event]))
                self.assertEqual(caught.exception.kind, kind)
                self.assertEqual(caught.exception.retryable, retryable)
                self.assertIn(code, caught.exception.error_codes)


if __name__ == "__main__":
    unittest.main()
