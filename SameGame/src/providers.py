"""Provider requests and native conversation state shared by the model runner."""

from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
import re
import time


KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _safe_token(value, pattern):
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        return None
    if re.search(r"(?:^|_)sk[-_]", value) or any(
            secret and secret in value for key in KEY_NAMES.values() if (secret := os.environ.get(key))):
        return None
    return value


def _safe_class(value):
    return _safe_token(value, r"[A-Za-z_][A-Za-z0-9_]{0,95}")


def _safe_request_id(value):
    return _safe_token(value, r"(?:req_[A-Za-z0-9_-]{1,120}|[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12})")


def _safe_stream(value):
    value = value if isinstance(value, dict) else {}
    count = value.get("event_count")
    seconds = value.get("seconds_since_last_event")
    return {
        "event_count": count if type(count) is int and count >= 0 else 0,
        "last_event_type": _safe_token(value.get("last_event_type"), r"[a-z][a-z0-9_.]{0,95}"),
        "seconds_since_last_event": float(seconds) if type(seconds) in (int, float)
            and math.isfinite(seconds) and seconds >= 0 else None,
        "response_id": _safe_token(value.get("response_id"), r"(?:resp|msg)_[A-Za-z0-9_-]{1,120}"),
    }


class ProviderError(RuntimeError):
    """A diagnostic safe to print without response headers or credentials."""

    def __init__(self, message, *, retryable=False, kind="provider_error", status_code=None, retry_after=None,
                 exception_type=None, cause_types=None, request_id=None, error_codes=None, stream=None):
        super().__init__(message)
        self.retryable = bool(retryable)
        self.kind = kind
        self.status_code = status_code if type(status_code) is int else None
        self.retry_after = (float(retry_after) if type(retry_after) in (int, float)
                            and math.isfinite(retry_after) and retry_after >= 0 else None)
        self.exception_type = _safe_class(exception_type)
        self.cause_types = [name for item in (cause_types or []) if (name := _safe_class(item))]
        self.request_id = _safe_request_id(request_id)
        self.error_codes = sorted({code for item in (error_codes or [])
                                   if (code := _safe_token(item, r"[a-z][a-z0-9_]{0,95}"))})
        self.stream = _safe_stream(stream)


TRANSIENT_CODES = {
    "server_error", "internal_server_error", "api_error", "overloaded_error",
    "rate_limit_exceeded", "rate_limit_error", "request_timeout", "timeout",
}
PERMANENT_CODES = {
    "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
    "billing_error", "authentication_error", "invalid_api_key", "permission_error",
}


def _error_kind(codes, status=None, *, fallback="provider_error"):
    if codes & {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "billing_error"}:
        return "quota"
    if status == 401 or codes & {"authentication_error", "invalid_api_key"}:
        return "authentication"
    if status == 403 or "permission_error" in codes:
        return "permission"
    if status in (400, 404, 422) or codes & {"invalid_request_error", "invalid_prompt", "not_found_error"}:
        return "invalid_request"
    if status == 429 or codes & {"rate_limit_exceeded", "rate_limit_error"}:
        return "rate_limit"
    return fallback


def _cause_types(exc):
    result, seen = [], {id(exc)}
    current = exc.__cause__ or exc.__context__
    while current is not None and id(current) not in seen and len(result) < 8:
        seen.add(id(current))
        result.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return result


def _error_codes(value):
    if not isinstance(value, dict):
        return set()
    nested = value.get("error")
    fields = [value.get("code"), value.get("type")]
    if isinstance(nested, dict):
        fields.extend((nested.get("code"), nested.get("type")))
    return {field for field in fields if isinstance(field, str)}


def _retry_after(headers):
    """Read only the server's delay; do not retain or print headers."""
    if headers is None:
        return None
    for name, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            delay = float(value) / divisor
            if math.isfinite(delay) and delay >= 0:
                return delay
        except (TypeError, ValueError, OverflowError):
            pass
        if name == "retry-after":
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return None


def _classify_exception(exc):
    """Separate transient transport failures from authentication and local bugs."""
    if isinstance(exc, ProviderError):
        return exc
    status = getattr(exc, "status_code", None)
    status = status if type(status) is int else None
    codes = _error_codes(getattr(exc, "body", None))
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        codes.add(code)
    classes = {cls.__name__ for cls in type(exc).__mro__
               if cls.__module__.split(".")[0] in {"openai", "anthropic", "httpx", "httpx2", "httpcore", "httpcore2"}}
    timeout = bool(classes & {"APITimeoutError", "TimeoutException", "TimeoutError"})
    connection = bool(classes & {"APIConnectionError", "NetworkError", "RemoteProtocolError"})
    permanent = status in (400, 401, 403, 404, 422) or bool(codes & PERMANENT_CODES)
    retryable = not permanent and (timeout or connection or bool(codes & TRANSIENT_CODES)
                                  or status in (408, 409, 429) or (status is not None and 500 <= status <= 599))
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None and headers.get("x-should-retry") == "false":
        retryable = False
    kind = _error_kind(codes, status, fallback=("timeout" if timeout else "connection" if connection else
                                               "server_error" if retryable else "provider_error"))
    request_id = _safe_request_id(getattr(exc, "request_id", None))
    if request_id is None and headers is not None:
        request_id = _safe_request_id(headers.get("x-request-id")) or _safe_request_id(headers.get("request-id"))
    suffix = f" (HTTP {status})" if status is not None else ""
    return ProviderError(f"API 요청이 중단되었습니다: {type(exc).__name__}{suffix}",
                         retryable=retryable, kind=kind, status_code=status,
                         retry_after=_retry_after(headers) if retryable else None,
                         exception_type=type(exc).__name__, cause_types=_cause_types(exc),
                         request_id=request_id, error_codes=codes)


def _stream_error(value):
    codes = _error_codes(value)
    retryable = bool(codes & TRANSIENT_CODES) and not bool(codes & PERMANENT_CODES)
    return ProviderError("API가 스트림 오류를 반환했습니다. 저장된 이벤트를 확인하세요.",
                         retryable=retryable,
                         kind=_error_kind(codes, fallback="server_error" if retryable else "stream_error"),
                         exception_type="StreamError", error_codes=codes,
                         request_id=value.get("request_id") if isinstance(value, dict) else None)


def _transient_failed_response(raw):
    return (isinstance(raw, dict) and raw.get("status") == "failed"
            and _stream_error(raw).retryable)


def make_client(settings):
    provider = settings["provider"]
    key_name = KEY_NAMES[provider]
    key = os.environ.get(key_name, "").strip()
    if not key:
        raise ProviderError(f"{key_name}가 없습니다. 실행할 터미널에 API 키를 설정하세요.",
                            kind="authentication", exception_type="MissingAPIKey")
    options = {"api_key": key, "timeout": settings["timeout"], "max_retries": 0}
    try:
        if provider == "openai":
            from openai import OpenAI
            return OpenAI(base_url="https://api.openai.com/v1", **options)
        from anthropic import Anthropic
        custom_headers = os.environ.pop("ANTHROPIC_CUSTOM_HEADERS", None)
        try:
            return Anthropic(base_url="https://api.anthropic.com", auth_token="",
                             open_telemetry=False, **options)
        finally:
            if custom_headers is not None:
                os.environ["ANTHROPIC_CUSTOM_HEADERS"] = custom_headers
    except ImportError:
        raise ProviderError("SDK를 설치하세요: python -m pip install -r requirements.txt") from None


def make_request(settings, system, history):
    """Only public messages and original provider output enter the request."""
    if settings["provider"] == "anthropic":
        return {
            "model": settings["model"],
            "system": system,
            "messages": deepcopy(history),
            "max_tokens": settings["max_output_tokens"],
            "output_config": {"effort": settings["effort"]},
            "thinking": {"type": "adaptive", "display": "omitted"},
            "tools": [],
        }
    return {
        "model": settings["model"],
        "instructions": system,
        "input": deepcopy(history),
        "reasoning": {"effort": settings["effort"], "context": "all_turns"},
        "include": ["reasoning.encrypted_content"],
        "text": {"format": {"type": "text"}},
        "max_output_tokens": settings["max_output_tokens"],
        "tools": [],
        "store": False,
    }


def receive(client, provider, request, stream_path):
    """Save events as they arrive and return the complete provider response."""
    final = None
    message_stopped = False
    event_count = 0
    last_event_type = response_id = last_event_at = None

    def annotate(error):
        error.stream = _safe_stream({
            "event_count": event_count, "last_event_type": last_event_type, "response_id": response_id,
            "seconds_since_last_event": None if last_event_at is None else time.monotonic() - last_event_at,
        })
        return error

    try:
        with stream_path.open("w", encoding="utf-8") as log:
            resource = (client.messages.stream(**request) if provider == "anthropic"
                        else client.responses.create(**request, stream=True))
            with resource as stream:
                for event in stream:
                    data = event.model_dump(mode="json")
                    event_count += 1
                    last_event_type, last_event_at = data.get("type"), time.monotonic()
                    snapshot = data.get("response") or data.get("message")
                    if isinstance(snapshot, dict):
                        response_id = _safe_token(snapshot.get("id"), r"(?:resp|msg)_[A-Za-z0-9_-]{1,120}") or response_id
                    log.write(json.dumps(data, ensure_ascii=False) + "\n")
                    log.flush()
                    if data.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                        final = data.get("response")
                    elif provider == "anthropic" and data.get("type") == "message_stop":
                        message_stopped = True
                        if isinstance(data.get("message"), dict):
                            final = data["message"]
                    elif data.get("type") == "error":
                        raise _stream_error(data)
                if provider == "anthropic" and message_stopped:
                    final = stream.get_final_message().model_dump(mode="json")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        error = _classify_exception(exc)
        # A connection can close badly after delivering the entire terminal
        # response. Keep that response; partial deltas never qualify.
        if not isinstance(final, dict) or not error.retryable:
            raise annotate(error) from None
    if not isinstance(final, dict):
        raise annotate(ProviderError("최종 응답 없이 스트림이 끝났습니다. 저장된 이벤트를 확인하세요.",
                                     retryable=True, kind="incomplete_stream", exception_type="IncompleteStream"))
    if _transient_failed_response(final):
        raise annotate(_stream_error(final))
    return final


def recover_response(stream_path, provider):
    """Recover only a terminal event containing the complete native response."""
    final = None
    with stream_path.open(encoding="utf-8") as log:
        for line in log:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted write may leave the last event truncated.
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if provider == "openai" and kind in (
                    "response.completed", "response.incomplete", "response.failed"):
                candidate = event.get("response")
            elif provider == "anthropic" and kind == "message_stop":
                # The SDK's public message stream attaches its final snapshot.
                candidate = event.get("message")
            else:
                continue
            if isinstance(candidate, dict):
                final = candidate
    return None if _transient_failed_response(final) else final


def visible_text(raw, provider):
    """Return grading text and an optional generation failure reason."""
    if provider == "anthropic":
        stop = raw.get("stop_reason")
        if stop not in ("end_turn", "refusal"):
            return "", f"anthropic_{stop or 'unfinished'}"
        blocks = raw.get("content", [])
        if any(block.get("type") not in ("text", "thinking", "redacted_thinking") for block in blocks):
            return "", "unexpected_output"
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    else:
        if raw.get("status") != "completed":
            return "", f"openai_{raw.get('status', 'unfinished')}"
        parts = []
        for item in raw.get("output", []):
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message" or item.get("role") != "assistant":
                return "", "unexpected_output"
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "refusal":
                    parts.append(block.get("refusal", ""))
                else:
                    return "", "unexpected_output"
        text = "".join(parts)
    if not text.strip():
        return "", "empty_visible_response"
    return text, None


def continuation_items(raw, provider):
    """Keep native reasoning/signatures without converting them to visible text."""
    if provider == "anthropic":
        blocks = deepcopy(raw.get("content", []))
        for block in blocks:
            if block.get("type") == "thinking" and not block.get("signature"):
                raise ProviderError("응답의 thinking 서명이 없어 대화 상태를 이어갈 수 없습니다.")
            if block.get("type") == "redacted_thinking" and not block.get("data"):
                raise ProviderError("응답의 redacted_thinking 상태가 없습니다.")
        return [{"role": "assistant", "content": blocks}]

    def input_fields(value):
        if isinstance(value, dict):
            return {key: input_fields(child) for key, child in value.items() if child is not None}
        if isinstance(value, list):
            return [input_fields(child) for child in value]
        return value

    items = []
    for block in raw.get("output", []):
        if block.get("type") == "reasoning" and not block.get("encrypted_content"):
            raise ProviderError("응답의 암호화된 reasoning 상태가 없어 대화를 이어갈 수 없습니다.")
        item = input_fields(block)
        item.pop("created_by", None)
        items.append(item)
    return items


def token_usage(raw, provider):
    usage = raw.get("usage") or {}
    inputs = usage.get("input_tokens", 0) or 0
    if provider == "anthropic":
        inputs += (usage.get("cache_creation_input_tokens", 0) or 0)
        inputs += (usage.get("cache_read_input_tokens", 0) or 0)
    return {"input_tokens": inputs, "output_tokens": usage.get("output_tokens", 0) or 0}
