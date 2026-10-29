"""Explicit environment credentials and one non-generative model access check.

No dotenv, key files, credential fallback, generation requests, or disk writes.
Official references (checked 2026-10-28):
https://developers.openai.com/api/reference/python/resources/models/methods/retrieve
https://platform.claude.com/docs/en/api/python/models/retrieve
https://platform.claude.com/docs/en/api/errors
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import sys


KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
OFFICIAL_URLS = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com"}
QUOTA_CODES = {
    "insufficient_quota", "insufficient_credits", "credit_balance_too_low",
    "billing_error", "billing_hard_limit_reached", "usage_limit_reached",
    "workspace_spend_limit", "organization_spend_limit", "spend_limit_exceeded",
}


class AuthenticationCheckError(RuntimeError):
    """Contains only a fixed, safe diagnostic; never the SDK exception text."""

    def __init__(self, category: str, message: str, *, status: int | None = None):
        super().__init__(message)
        self.category = category
        self.status = status
        self.exit_code = 78 if category == "authentication" else 1


def key_from_environment(provider: str) -> str:
    if provider not in KEY_NAMES:
        raise AuthenticationCheckError("configuration", "지원하지 않는 API 공급자입니다.")
    name = KEY_NAMES[provider]
    key = os.environ.get(name, "")
    if not key or key in {"your-openai-api-key-here", "your-anthropic-api-key-here"}:
        raise AuthenticationCheckError(
            "missing_key", f"{name}가 없습니다. 같은 터미널에서 source ./keys.sh {provider} 를 실행하세요.")
    if any(character.isspace() or ord(character) < 33 or ord(character) > 126 for character in key):
        raise AuthenticationCheckError(
            "invalid_key_format", f"{name}에 공백 또는 잘못된 문자가 있습니다. source ./keys.sh {provider} --replace 로 다시 입력하세요.")
    return key


@contextmanager
def _without_anthropic_overrides():
    """Prevent token/custom-header auth, then restore the caller's environment."""
    names = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = ""
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def make_client(timeout: float, provider: str = "openai"):
    """Construct a client only. No requests or retries occur here."""
    key = key_from_environment(provider)
    common = {"api_key": key, "base_url": OFFICIAL_URLS[provider],
              "max_retries": 0, "timeout": timeout, "default_headers": {}}
    try:
        if provider == "openai":
            from openai import OpenAI
            return OpenAI(**common)
        from anthropic import Anthropic
        with _without_anthropic_overrides():
            return Anthropic(**common, auth_token="", open_telemetry=False)
    except ImportError:
        raise AuthenticationCheckError(
            "dependency", "SDK를 불러올 수 없습니다. python -m pip install -r requirements.txt 를 실행하세요.") from None
    except Exception:
        raise AuthenticationCheckError("client_setup", "API 클라이언트를 만들지 못했습니다. 설치된 SDK와 실행 환경을 확인하세요.") from None


def _quota_error(error: Exception) -> bool:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    nested = body.get("error", body)
    if not isinstance(nested, dict):
        return False
    details = nested.get("details", {})
    values = [nested.get("type"), nested.get("code")]
    if isinstance(details, dict):
        values.append(details.get("error_code"))
    return any(isinstance(value, str) and value in QUOTA_CODES for value in values)


def safe_error(error: Exception, provider: str) -> AuthenticationCheckError:
    """Classify HTTP status/allowlisted codes without emitting bodies or keys."""
    status = getattr(error, "status_code", None)
    status = status if type(status) is int else None
    if status == 401:
        return AuthenticationCheckError("authentication",
            f"인증 실패 (HTTP 401). source ./keys.sh {provider} --replace 로 키를 교체한 뒤 다시 실행하세요.", status=status)
    if status == 403:
        return AuthenticationCheckError("permission",
            "접근 거부 (HTTP 403). 키의 권한과 조직·프로젝트 접근 설정을 확인하세요.", status=status)
    if status == 404:
        return AuthenticationCheckError("model_access",
            "모델 조회 실패 (HTTP 404). 모델 ID가 없거나 이 키에 모델 접근 권한이 없습니다.", status=status)
    if status == 402 or _quota_error(error):
        return AuthenticationCheckError("billing",
            "결제·크레딧 또는 사용 한도 오류입니다. 공급자 콘솔에서 잔액과 지출 한도를 확인하세요.", status=status)
    if status == 429:
        return AuthenticationCheckError("rate_or_usage_limit",
            "요청 또는 사용 한도 제한 (HTTP 429). 공급자 콘솔의 요청률·지출 한도를 확인한 뒤 다시 실행하세요.", status=status)
    if status is not None:
        return AuthenticationCheckError("http",
            f"모델 조회가 거절되었습니다 (HTTP {status}). 자동 재시도하지 않았습니다.", status=status)
    if type(error).__name__ in {"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutException", "NetworkError"} or isinstance(error, (ConnectionError, TimeoutError)):
        return AuthenticationCheckError("network", "API 연결 또는 시간 초과 오류입니다. 네트워크를 확인한 뒤 다시 실행하세요.")
    return AuthenticationCheckError("check_failed", "모델 조회를 완료하지 못했습니다. SDK와 연결 상태를 확인하세요.")


def check_model_access(client, provider: str, model: str) -> None:
    """Exactly one GET model request; no inference or experimental conversation."""
    try:
        if provider == "anthropic":
            info = client.models.retrieve(model_id=model)
        else:
            info = client.models.retrieve(model)
    except Exception as error:
        raise safe_error(error, provider) from None
    identifier = info.get("id") if isinstance(info, dict) else getattr(info, "id", None)
    # Aliases may resolve to a dated ID. Do not replace the configured model.
    if not isinstance(identifier, str) or not identifier:
        raise AuthenticationCheckError("invalid_response", "모델 조회 응답을 확인할 수 없습니다. 실험을 시작하지 않았습니다.")


def checked_client(config: dict):
    """Return the same authenticated client that will perform the experiment."""
    client = make_client(config["timeout"], config["provider"])
    try:
        check_model_access(client, config["provider"], config["model"])
    except BaseException:
        client.close()
        raise
    print("인증·모델 조회 확인 완료. 생성 요청의 잔액·권한·한도를 보장하는 검사는 아닙니다.", file=sys.stderr, flush=True)
    return client
