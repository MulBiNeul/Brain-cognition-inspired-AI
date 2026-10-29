#!/usr/bin/env python3
"""Convert TOML settings into Hanoi runner arguments without making API calls.

Run ``python run.py --profile sol|opus|sonnet`` from the terminal.
This module neither reads nor stores keys and does not change experiment or grading logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tomllib

from .prompts import FEEDBACK_POLICIES


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    "sol": ROOT / "configs" / "sol.toml",
    "opus": ROOT / "configs" / "opus.toml",
    "sonnet": ROOT / "configs" / "sonnet.toml",
}
DEFAULT_CONFIG = CONFIG_PATHS["sol"]
TOP_KEYS = {"config_version", "default_profile", "experiment", "profiles"}
EXPERIMENT_KEYS = {
    "dataset", "problems", "start_problem", "T", "k", "r1_apply_k",
    "max_output_tokens", "timeout",
}
OPTIONAL_EXPERIMENT_KEYS = {"feedback_policy"}
PROFILE_KEYS = {"label", "provider", "model", "effort"}
PROVIDERS = {"openai", "anthropic"}
EFFORTS = {"low", "medium", "high", "xhigh", "max"}


class ConfigError(ValueError):
    """Configuration error that does not expose file contents or secret values."""


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name}: 비어 있지 않은 문자열이어야 합니다.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfigError(f"{name}: 줄바꿈·탭·제어 문자는 사용할 수 없습니다.")
    return value


def _keys(value: object, allowed: set[str], name: str, *, optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: TOML 테이블이어야 합니다.")
    if set(value) - allowed - optional:
        # Names and values may contain secrets; never echo the supplied contents.
        raise ConfigError(f"{name}: 지원하지 않는 설정 항목이 있습니다. API 키는 설정 파일에 넣지 마세요.")
    if allowed - set(value):
        missing = ", ".join(sorted(allowed - set(value)))
        raise ConfigError(f"{name}: 필수 항목이 없습니다: {missing}")
    return value


def _absolute_path(value: str | Path, base: Path, name: str) -> Path:
    text = _string(str(value), name)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    _string(str(resolved), name)
    return resolved


def _provider_model(provider: object, model: object) -> None:
    _string(provider, "provider")
    _string(model, "model")
    if provider not in PROVIDERS:
        raise ConfigError("provider는 openai 또는 anthropic이어야 합니다.")
    if any(character.isspace() for character in model) or model.startswith("-"):
        raise ConfigError("model에는 공백 또는 옵션 형태의 값을 사용할 수 없습니다.")
    if (model.startswith("claude-") != (provider == "anthropic")):
        raise ConfigError("provider와 model이 일치하지 않습니다. Claude 모델은 anthropic을 사용하세요.")


def load_config(config_path: str | Path | None = None) -> tuple[Path, dict]:
    """Validate the complete TOML and return (absolute path, settings object)."""
    path = _absolute_path(config_path or DEFAULT_CONFIG, Path.cwd(), "config 경로")
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    except OSError as exc:
        raise ConfigError("설정 파일을 읽을 수 없습니다. --config 경로를 확인하세요.") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError("설정 파일의 TOML 문법을 확인하세요. 문자열은 따옴표로 감싸야 합니다.") from exc
    except UnicodeError as exc:
        raise ConfigError("설정 파일은 UTF-8 형식이어야 합니다.") from exc
    _keys(config, TOP_KEYS, "설정 최상위")
    if type(config["config_version"]) is not int or config["config_version"] != 1:
        raise ConfigError("config_version은 정수 1이어야 합니다.")
    experiment = _keys(config["experiment"], EXPERIMENT_KEYS, "experiment", optional=OPTIONAL_EXPERIMENT_KEYS)
    experiment.setdefault("feedback_policy", "binary_v1")
    if not isinstance(experiment["feedback_policy"], str) or experiment["feedback_policy"] not in FEEDBACK_POLICIES:
        raise ConfigError("experiment.feedback_policy는 binary_v1 또는 goal_rule_4way_v1이어야 합니다.")
    _string(experiment["dataset"], "experiment.dataset")
    for name in ("problems", "start_problem", "T", "k", "max_output_tokens"):
        if type(experiment[name]) is not int or experiment[name] < 1:
            raise ConfigError(f"experiment.{name}: 1 이상의 정수여야 합니다.")
    if experiment["k"] > experiment["T"]:
        raise ConfigError("T와 k는 1 <= k <= T를 만족해야 합니다.")
    if type(experiment["r1_apply_k"]) is not bool:
        raise ConfigError("experiment.r1_apply_k는 true 또는 false여야 합니다.")
    timeout = experiment["timeout"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ConfigError("experiment.timeout은 유한한 양수여야 합니다.")
    profiles = config["profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise ConfigError("profiles에는 하나 이상의 모델 프로필이 필요합니다.")
    for identifier, profile in profiles.items():
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", identifier):
            raise ConfigError("프로필 이름에는 영문·숫자·밑줄·하이픈만 사용할 수 있습니다.")
        _keys(profile, PROFILE_KEYS, f"profiles.{identifier}")
        _string(profile["label"], f"profiles.{identifier}.label")
        _provider_model(profile["provider"], profile["model"])
        if not isinstance(profile["effort"], str) or profile["effort"] not in EFFORTS:
            raise ConfigError(f"profiles.{identifier}.effort는 low/medium/high/xhigh/max 중 하나여야 합니다.")
    default = _string(config["default_profile"], "default_profile")
    if default not in profiles:
        raise ConfigError("default_profile에 등록된 프로필 이름을 지정하세요.")
    for identifier, standard_path in CONFIG_PATHS.items():
        if path == standard_path.resolve() and (set(profiles) != {identifier} or default != identifier):
            raise ConfigError(f"configs/{identifier}.toml에는 {identifier} 프로필만 등록하세요.")
    return path, config


def _selected_config(config_path: str | Path | None, profile: str | None) -> str | Path | None:
    """Select the model-specific TOML for a built-in profile."""
    if config_path is not None or profile is None:
        return config_path
    if not isinstance(profile, str) or profile not in CONFIG_PATHS:
        raise ConfigError("등록되지 않은 프로필입니다. python -m hanoi.config --list-profiles로 목록을 확인하세요.")
    return CONFIG_PATHS[profile]


def list_profiles(config_path: str | Path | None = None) -> dict[str, dict]:
    """Return profiles from the default model files or an explicit configuration."""
    if config_path is not None:
        return load_config(config_path)[1]["profiles"]
    profiles = {}
    for identifier, path in CONFIG_PATHS.items():
        config = load_config(path)[1]
        profiles[identifier] = config["profiles"][identifier]
    return profiles


def load_settings(config_path: str | Path | None = None, profile: str | None = None) -> dict:
    """Merge the selected profile with shared experiment settings.

    The dataset path is absolute. Returned keys include the experiment fields
    plus profile, label, provider, model, and effort. No API keys or environment
    variables are read.
    """
    path, config = load_config(_selected_config(config_path, profile))
    identifier = profile if profile is not None else config["default_profile"]
    if not isinstance(identifier, str) or identifier not in config["profiles"]:
        raise ConfigError("등록되지 않은 프로필입니다. python -m hanoi.config --list-profiles로 목록을 확인하세요.")
    settings = {**config["experiment"], **config["profiles"][identifier], "profile": identifier}
    dataset = _absolute_path(settings["dataset"], path.parent, "dataset 경로")
    if not dataset.is_file():
        raise ConfigError("dataset 파일을 찾지 못했습니다. 경로는 TOML 파일이 있는 폴더 기준입니다.")
    try:
        content = json.loads(dataset.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ConfigError("dataset 파일을 읽을 수 없거나 JSON 형식이 올바르지 않습니다.") from exc
    problems = content.get("problems") if isinstance(content, dict) else None
    if not isinstance(problems, list) or not problems:
        raise ConfigError("dataset에는 비어 있지 않은 problems 배열이 필요합니다.")
    if settings["start_problem"] + settings["problems"] - 1 > len(problems):
        raise ConfigError(
            f"실행 범위가 dataset의 전체 {len(problems)}문제를 초과합니다. "
            "start_problem과 problems를 확인하세요."
        )
    settings["dataset"] = str(dataset)
    return settings


def _resume_settings(resume: str | Path) -> tuple[Path, dict]:
    directory = _absolute_path(resume, Path.cwd(), "resume 경로")
    try:
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ConfigError("재개 폴더의 protocol.json을 읽을 수 없습니다.") from exc
    if not isinstance(protocol, dict) or not isinstance(protocol.get("frozen"), dict):
        raise ConfigError("재개 protocol.json의 frozen 설정 형식이 올바르지 않습니다.")
    frozen = protocol["frozen"]
    canonical = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if checksum != protocol.get("frozen_sha256"):
        raise ConfigError("재개 protocol.json의 설정 해시가 일치하지 않습니다.")
    saved = frozen.get("config")
    if not isinstance(saved, dict):
        raise ConfigError("재개 protocol.json의 config 형식이 올바르지 않습니다.")
    _provider_model(saved.get("provider"), saved.get("model"))
    if not isinstance(saved.get("effort"), str) or saved["effort"] not in EFFORTS:
        raise ConfigError("재개 protocol.json의 effort 설정이 올바르지 않습니다.")
    return directory, saved


def resolve_arguments(
    config_path: str | Path | None = None,
    profile: str | None = None,
    resume: str | Path | None = None,
) -> list[str]:
    """Return arguments for the experiment runner.

    Resuming ignores experiment settings in the TOML. If config or profile is
    explicit, only verify that provider, model, and effort match the saved run.
    """
    if resume is not None:
        directory, saved = _resume_settings(resume)
        if profile is not None or config_path is not None:
            _, config = load_config(_selected_config(config_path, profile))
            selected = config["profiles"].get(profile or config["default_profile"])
            if selected is None:
                raise ConfigError("재개 시 지정한 프로필이 설정 파일에 없습니다.")
            if any(selected[key] != saved[key] for key in ("provider", "model", "effort")):
                raise ConfigError("선택한 프로필과 재개 실행의 공급자·모델·effort가 다릅니다. 다른 모델로 재개할 수 없습니다.")
            if config["experiment"]["feedback_policy"] != saved.get("feedback_policy", "binary_v1"):
                raise ConfigError("선택한 설정과 재개 실행의 피드백 정책이 다릅니다. 피드백을 바꿔 재개할 수 없습니다.")
        return ["--provider", saved["provider"], "--resume", str(directory)]
    settings = load_settings(config_path, profile)
    arguments = []
    for option, key in (
        ("--provider", "provider"), ("--model", "model"), ("--effort", "effort"),
        ("--dataset", "dataset"), ("--problems", "problems"),
        ("--start-problem", "start_problem"), ("--T", "T"), ("--k", "k"),
        ("--max-output-tokens", "max_output_tokens"), ("--timeout", "timeout"),
        ("--feedback-policy", "feedback_policy"),
    ):
        arguments.extend((option, str(settings[key])))
    arguments.append("--r1-apply-k" if settings["r1_apply_k"] else "--no-r1-apply-k")
    arguments.extend(("--context-format", "messages", "--output-format", "plain"))
    return arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="TOML 파일 (기본: configs/sol.toml)")
    parser.add_argument("--profile", help="모델 프로필: sol, opus, sonnet (기본: sol)")
    parser.add_argument("--resume", type=Path, help="기존 실행 폴더. 저장된 실험 설정을 그대로 사용")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-profiles", action="store_true", help="프로필 ID와 이름을 탭으로 구분해 출력")
    mode.add_argument("--default-profile", action="store_true", help="설정 파일의 기본 프로필 ID 출력")
    mode.add_argument("--resolve", action="store_true", help="실행 인자를 한 줄에 하나씩 출력 (API 호출 없음)")
    args = parser.parse_args(argv)
    if not args.resolve and (args.profile is not None or args.resume is not None):
        parser.error("--profile과 --resume은 --resolve와 함께 사용하세요.")
    try:
        if args.resolve:
            print("\n".join(resolve_arguments(args.config, args.profile, args.resume)))
        else:
            if args.default_profile:
                print(load_config(args.config)[1]["default_profile"])
            else:
                for identifier, profile in list_profiles(args.config).items():
                    print(f"{identifier}\t{profile['label']}")
    except (ConfigError, OSError) as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
