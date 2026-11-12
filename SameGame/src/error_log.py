"""Safe, human-readable error records and read-only inspection of saved runs."""

import json
import math
import os
from pathlib import Path
import re
import traceback


ERROR_EVENTS = {"transport_error", "stopped", "interrupted", "generation_failure"}
LABELS = {
    "timeout": "통신 시간 초과", "connection": "연결 오류",
    "incomplete_stream": "최종 응답 없는 스트림 종료", "rate_limit": "요청 한도 초과",
    "server_error": "서버 오류", "authentication": "인증 오류", "quota": "사용량·결제 한도 오류",
    "permission": "권한 오류", "transport_success": "통신 성공 / 복구",
    "invalid_request": "잘못된 요청", "stream_error": "스트림 오류",
    "provider_error": "API 또는 실행 오류", "KeyboardInterrupt": "사용자 인터럽트",
    "generation_failure": "모델 응답 생성 미완료", "local_error": "로컬 실행 오류",
}


def clean(value, limit=500):
    text = "".join(char for char in str(value) if char.isprintable())
    text = re.sub(r"(?i)bearer\s+\S+|\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:limit]


def _identifier(value):
    return clean(value, 160) if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value) else None


def exception_details(error):
    """Keep exception classes and stack locations, never values, source lines or locals."""
    causes = []
    seen = {id(error)}
    cause = error.__cause__ or error.__context__
    while cause is not None and id(cause) not in seen and len(causes) < 10:
        seen.add(id(cause))
        causes.append(type(cause).__name__)
        cause = cause.__cause__ or cause.__context__
    for name in getattr(error, "cause_types", ()) or ():
        name = _identifier(name)
        if name and name not in causes:
            causes.append(name)
    details = {
        "exception_type": _identifier(getattr(error, "exception_type", None)) or type(error).__name__,
        "cause_types": causes,
        "request_id": _identifier(getattr(error, "request_id", None)),
        "error_codes": [code for value in (getattr(error, "error_codes", ()) or ())
                        if (code := _identifier(value))],
        "traceback_frames": [{"file": Path(frame.filename).name, "line": frame.lineno,
                              "function": clean(frame.name, 100)}
                             for frame in traceback.extract_tb(error.__traceback__)],
    }
    stream = getattr(error, "stream", None)
    if isinstance(stream, dict):
        details["stream"] = {}
        for field in ("last_event_type", "response_id"):
            details["stream"][field] = _identifier(stream.get(field))
        for field in ("event_count", "seconds_since_last_event"):
            value = stream.get(field)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                details["stream"][field] = value
    return details


def format_event(event):
    kind = clean(event.get("kind", event.get("event", "unknown")))
    title = LABELS.get(kind, kind)
    location = " · ".join(str(value) for value in (
        clean(event.get("id", "문제 미상")),
        f"R{event['round']}" if event.get("round") is not None else None,
        f"풀이 시도 {event['attempt']}" if event.get("attempt") is not None else None,
        f"제출 #{event['submission']}" if event.get("submission") is not None else None,
        f"통신 #{event['transport_attempt']}" if event.get("transport_attempt") is not None else None,
    ) if value is not None)
    lines = [f"[{clean(event.get('at', '시각 미상'))}] {title}", f"  위치: {clean(location)}"]
    if event.get("message"):
        lines.append(f"  설명: {clean(event['message'])}")
    metadata = [("예외", event.get("exception_type")), ("HTTP", event.get("status_code")),
                ("API 요청 ID", event.get("request_id")), ("실행 단계", event.get("phase")),
                ("생성 실패 유형", event.get("failure")), ("생성 미완료 사유", event.get("generation_reason"))]
    for label, value in metadata:
        if value is not None:
            lines.append(f"  {label}: {clean(value)}")
    for label, key in (("원인 예외", "cause_types"), ("API 오류 코드", "error_codes")):
        if event.get(key):
            lines.append(f"  {label}: {', '.join(clean(value) for value in event[key])}")
    if event.get("elapsed_seconds") is not None:
        lines.append(f"  경과 시간: {clean(event['elapsed_seconds'])}초")
    if event.get("timeout_seconds") is not None:
        lines.append(f"  통신 timeout 설정: {clean(event['timeout_seconds'])}초")
    if event.get("retry_number") is not None:
        lines.append(f"  통신 재시도: {clean(event['retry_number'])}/{clean(event.get('retry_limit', '?'))}회")
    stream = event.get("stream") or {}
    if stream:
        lines.append(f"  스트림: 이벤트 {clean(stream.get('event_count', '?'))}개 · "
                     f"마지막 {clean(stream.get('last_event_type') or '없음')}")
        if stream.get("seconds_since_last_event") is not None:
            lines.append(f"  마지막 이벤트 이후: {clean(stream['seconds_since_last_event'])}초")
        if stream.get("response_id"):
            lines.append(f"  응답 ID: {clean(stream['response_id'])}")
    if event.get("recovered_from_stream"):
        lines.append("  처리: 저장된 완료 응답 복구 · 추가 요청 없음")
    elif event.get("event") == "transport_success":
        lines.append("  처리: 재시도 후 응답 수신 성공" if event.get("retry_number", 0) > 0
                     else "  처리: 응답 수신 성공")
    elif event.get("will_retry"):
        lines.append(f"  처리: {clean(event.get('retry_delay_seconds', '?'))}초 후 자동 재시도 예정 · 풀이 시도 미차감")
    elif event.get("event") == "transport_error":
        lines.append("  처리: 자동 재시도하지 않음 · 풀이 시도 미차감")
    recovered = event.get("subsequent_recovery")
    if recovered:
        result = "채점 완료" if recovered.get("event") == "submission_graded" else "응답 수신 성공"
        lines.append(f"  후속 결과: 같은 제출의 {result} 확인 · {clean(recovered.get('at', '시각 미상'))}")
    for frame in event.get("traceback_frames", []):
        lines.append(f"  위치 추적: {clean(frame.get('file'))}:{clean(frame.get('line'))} "
                     f"{clean(frame.get('function'))}")
    return "\n".join(lines)


def append_error(directory, event):
    with (Path(directory) / "errors.log").open("a", encoding="utf-8") as stream:
        stream.write(format_event(event) + "\n\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_object(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_events(path):
    events, malformed = [], 0
    if not path.exists():
        return events, malformed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError
                events.append(value)
            except ValueError:
                malformed += 1
    return events, malformed


def _latest_control(events):
    return next((event for event in reversed(events)
                 if event.get("event") in {"start", "resume", "complete", "stopped", "interrupted"}), None)


def _status(events, report):
    control = _latest_control(events)
    if control:
        state = control["event"]
        if state == "complete":
            return "정상 완료 기록 있음"
        if state in {"stopped", "interrupted"}:
            return "마지막 실행 중단 기록 있음: " + LABELS.get(control.get("kind"), clean(control.get("kind", state)))
        return "시작·재개 이후 종료 기록 없음 — 실행 중인지, 기록 없이 종료됐는지는 확인할 수 없음"
    return "평가 완료 보고서 있음" if report.get("complete") else "종료 원인 기록 없음 — 현재 실행 상태는 확인할 수 없음"


def inspect_run(directory, limit=10):
    """Inspect saved observations only; old missing logs cannot establish a timeout."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"실행 폴더가 없습니다: {directory}")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit은 1 이상이어야 합니다.")
    events, malformed = _read_events(directory / "run_events.jsonl")
    report = _read_object(directory / "report.json")
    errors, subsequent = [], {}
    for event in reversed(events):
        kind, submission = event.get("event"), event.get("submission")
        if kind in {"transport_success", "submission_graded"} and submission is not None:
            subsequent[submission] = event
        if kind in ERROR_EVENTS:
            record = dict(event)
            if kind == "transport_error" and submission in subsequent:
                record["subsequent_recovery"] = subsequent[submission]
            errors.append(record)
    errors.reverse()
    if not errors:
        for path in sorted((directory / "submissions").glob("*_transport_*_error.json")):
            event = _read_object(path)
            if event:
                match = re.fullmatch(r"(\d+)_transport_(\d+)_error.json", path.name)
                if match:
                    event.setdefault("submission", int(match[1]))
                    event.setdefault("transport_attempt", int(match[2]))
                event.setdefault("event", "transport_error")
                errors.append(event)
    lines = [f"실행: {directory}", f"상태: {_status(events, report)}",
             f"저장된 채점: {len(report.get('submissions', []))}건 · 오류·중단 기록: {len(errors)}건"]
    if malformed:
        lines.append(f"주의: 읽을 수 없는 이벤트 {malformed}줄은 건너뛰었습니다.")
    if errors:
        lines.append("아래는 과거 오류 이력입니다. 재개 전 오류가 현재 종료 원인이라는 뜻은 아닙니다.")
        lines.extend(format_event(event) for event in errors[-limit:])
    elif ((_latest_control(events) or {}).get("event") == "complete"
          or (_latest_control(events) is None and report.get("complete"))):
        lines.append("오류 기록 없음")
    else:
        lines.append("상세 오류 기록이 없습니다. 기존 파일만으로 정확한 종료 예외를 복원할 수 없습니다.")
        streams = sorted((directory / "submissions").glob("*_transport_*_stream.jsonl"))
        if streams:
            stream_events, broken = _read_events(streams[-1])
            if stream_events:
                last = stream_events[-1].get("type", "미상")
                terminal = any(event.get("type") in {"response.completed", "response.incomplete",
                                                      "response.failed", "message_stop"} for event in stream_events)
                lines.append(f"관찰값: {streams[-1].name} · 이벤트 {len(stream_events)}개 · 마지막 {clean(last)} · "
                             f"종료 이벤트 {'있음' if terminal else '없음'}")
            if broken:
                lines.append(f"마지막 스트림에서 읽을 수 없는 이벤트 {broken}줄이 있습니다.")
    successes = [event for event in events if event.get("event") == "transport_success"]
    if successes:
        last = successes[-1]
        lines.append(f"마지막 통신 성공: {clean(last.get('at', '시각 미상'))} · 제출 #{clean(last.get('submission', '?'))}")
    return "\n\n".join(lines) + "\n"
