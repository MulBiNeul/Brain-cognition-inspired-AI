"""Error reporting keeps diagnoses useful without exporting provider payloads."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.error_log import append_error, exception_details, format_event, inspect_run


class ErrorLogTests(unittest.TestCase):
    def test_exception_details_keep_classes_and_locations_without_values(self):
        secret = "sk-private-secret"
        try:
            try:
                raise ConnectionError(secret)
            except ConnectionError:
                raise RuntimeError("Authorization: Bearer sensitive") from None
        except RuntimeError as error:
            error.request_id = "req_test_123"
            error.error_codes = ["server_error"]
            error.stream = {"event_count": 3, "last_event_type": "response.output_text.delta",
                            "delta": secret, "encrypted_content": secret}
            details = exception_details(error)
        self.assertEqual(details["exception_type"], "RuntimeError")
        self.assertEqual(details["cause_types"], ["ConnectionError"])
        self.assertEqual(details["request_id"], "req_test_123")
        self.assertTrue(details["traceback_frames"])
        self.assertNotIn(secret, json.dumps(details))
        self.assertNotIn("sensitive", json.dumps(details))
        self.assertNotIn("delta", details["stream"])

    def test_error_log_contains_context_and_retry_decision(self):
        event = {"at": "2026-11-11T12:00:00Z", "event": "transport_error", "kind": "timeout",
                 "id": "SG-001", "round": 2, "attempt": 5, "submission": 9, "transport_attempt": 2,
                 "exception_type": "ReadTimeout", "will_retry": True, "retry_delay_seconds": 10,
                 "status_code": 503, "request_id": "req_123", "error_codes": ["server_error"],
                 "stream": {"event_count": 12, "last_event_type": "response.output_text.delta",
                            "seconds_since_last_event": 600}, "message": "safe explanation"}
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            append_error(directory, event)
            text = (directory / "errors.log").read_text()
        for expected in ("통신 시간 초과", "SG-001", "R2", "풀이 시도 5", "통신 #2",
                         "ReadTimeout", "HTTP: 503", "req_123", "10초 후 자동 재시도", "600초"):
            self.assertIn(expected, text)
        self.assertIn("[REDACTED]", format_event(dict(event, message="Bearer token-secret sk-another-secret")))

    def test_resumed_run_does_not_report_historical_stop_as_current(self):
        events = [{"event": "start"}, {"event": "stopped", "kind": "timeout"},
                  {"event": "resume"}, {"event": "transport_success", "submission": 9, "at": "now"}]
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "run_events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            result = inspect_run(directory)
        self.assertIn("시작·재개 이후 종료 기록 없음", result)
        self.assertNotIn("상태: 마지막 실행 중단", result)
        self.assertIn("마지막 통신 성공", result)
        self.assertIn("과거 오류 이력", result)

    def test_malformed_event_is_skipped_without_losing_prior_error(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "run_events.jsonl").write_text(
                json.dumps({"event": "stopped", "kind": "timeout"}) + '\n{"event":')
            result = inspect_run(directory)
        self.assertIn("중단 기록 있음", result)
        self.assertIn("읽을 수 없는 이벤트 1줄", result)

    def test_legacy_partial_stream_reports_observation_not_timeout(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trials = directory / "submissions"
            trials.mkdir()
            (trials / "00009_transport_001_stream.jsonl").write_text(json.dumps({
                "type": "response.output_text.delta", "delta": "sk-secret-value",
                "encrypted_content": "private-state",
            }) + "\n")
            result = inspect_run(directory)
        self.assertIn("정확한 종료 예외를 복원할 수 없습니다", result)
        self.assertIn("종료 이벤트 없음", result)
        self.assertIn("response.output_text.delta", result)
        for absent in ("timeout", "시간 초과", "sk-secret-value", "private-state"):
            self.assertNotIn(absent, result)

    def test_old_individual_error_file_is_supported(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trials = directory / "submissions"
            trials.mkdir()
            (trials / "00009_transport_002_error.json").write_text(json.dumps({
                "kind": "incomplete_stream", "will_retry": True, "retry_delay_seconds": 5,
            }))
            result = inspect_run(directory)
        self.assertIn("최종 응답 없는 스트림 종료", result)
        self.assertIn("제출 #9", result)
        self.assertIn("통신 #2", result)

    def test_complete_marker_and_limit(self):
        events = [{"event": "transport_error", "kind": "timeout", "id": "first"},
                  {"event": "transport_error", "kind": "connection", "id": "last"},
                  {"event": "complete"}]
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "run_events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            result = inspect_run(directory, limit=1)
        self.assertIn("정상 완료 기록 있음", result)
        self.assertNotIn("위치: first", result)
        self.assertIn("위치: last", result)

    def test_execution_phase_retry_policy_and_generation_reason_are_shown(self):
        event = {"event": "generation_failure", "kind": "permission", "phase": "grade",
                 "retry_number": 3, "retry_limit": 10, "timeout_seconds": 600,
                 "failure": "openai_incomplete", "generation_reason": "max_output_tokens"}
        text = format_event(event)
        for expected in ("권한 오류", "실행 단계: grade", "3/10회", "timeout 설정: 600초",
                         "openai_incomplete", "max_output_tokens"):
            self.assertIn(expected, text)
        success = format_event({"event": "transport_success", "retry_number": 2, "retry_limit": 10})
        self.assertIn("통신 성공 / 복구", success)
        self.assertIn("재시도 후 응답 수신 성공", success)

    def test_clean_completion_does_not_imply_an_unknown_exception(self):
        for marker in (True, False):
            with self.subTest(marker=marker), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                if marker:
                    (directory / "run_events.jsonl").write_text(json.dumps({"event": "complete"}))
                else:
                    (directory / "report.json").write_text(json.dumps({"complete": True, "submissions": []}))
                result = inspect_run(directory)
            self.assertIn("오류 기록 없음", result)
            self.assertNotIn("종료 예외를 복원할 수 없습니다", result)

    def test_historical_error_marks_only_subsequent_same_submission_recovery(self):
        events = [{"event": "transport_success", "submission": 8},
                  {"event": "transport_error", "kind": "timeout", "id": "unrecovered", "submission": 8},
                  {"event": "transport_error", "kind": "connection", "id": "recovered", "submission": 9},
                  {"event": "transport_success", "submission": 9, "at": "recovery-time"},
                  {"event": "submission_graded", "submission": 9, "at": "graded-time"}]
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "run_events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            result = inspect_run(directory)
        self.assertEqual(result.count("후속 결과:"), 1)
        self.assertIn("같은 제출의 응답 수신 성공 확인 · recovery-time", result)


if __name__ == "__main__":
    unittest.main()
