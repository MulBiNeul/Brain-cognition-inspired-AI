"""Provider requests and native conversation state shared by the model runner."""

from copy import deepcopy
import json
import os


KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class ProviderError(RuntimeError):
    """A diagnostic safe to print without response headers or credentials."""


def make_client(settings):
    provider = settings["provider"]
    key_name = KEY_NAMES[provider]
    key = os.environ.get(key_name, "").strip()
    if not key:
        raise ProviderError(f"{key_name}가 없습니다. 실행할 터미널에 API 키를 설정하세요.")
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
    try:
        with stream_path.open("w", encoding="utf-8") as log:
            resource = (client.messages.stream(**request) if provider == "anthropic"
                        else client.responses.create(**request, stream=True))
            with resource as stream:
                for event in stream:
                    data = event.model_dump(mode="json")
                    log.write(json.dumps(data, ensure_ascii=False) + "\n")
                    log.flush()
                    if data.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                        final = data.get("response")
                if provider == "anthropic":
                    final = stream.get_final_message().model_dump(mode="json")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        suffix = f" (HTTP {status})" if type(status) is int else ""
        raise ProviderError(f"API 요청이 중단되었습니다: {type(exc).__name__}{suffix}") from None
    if not isinstance(final, dict):
        raise ProviderError("최종 응답 없이 스트림이 끝났습니다. 저장된 이벤트를 확인하세요.")
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
    return final


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
