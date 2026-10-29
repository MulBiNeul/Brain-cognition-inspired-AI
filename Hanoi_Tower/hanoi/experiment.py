#!/usr/bin/env python3
"""Five-round Tower of Hanoi experiment using OpenAI or Anthropic and configured feedback.

Requires Python 3.11+. See README.md for usage. --dry-run makes no API calls.
Hidden rule details and grading diagnostics are available only to the local evaluator.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import count
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from .prompts import FEEDBACK_POLICIES, feedback_values, task_prompt


ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = 10  # T: Maximum submissions per round, configurable through runner options.
SUCCESS_STREAK = 2  # k: Required consecutive correct submissions; an incorrect answer resets the streak.
MAX_MOVES = 30
PEGS = ("A", "B", "C", "D")
DISK_COUNT = 3
PROTOCOL_VERSION = 26
FOURWAY_PROTOCOL_VERSION = 27
WHITESPACE_LIMIT = 256
DATASET_VERSION = 7
COMMANDS = tuple(source + target for source in PEGS for target in PEGS if source != target)
DEFAULTS = {
    "provider": "openai", "model": "gpt-6.1-sol", "effort": "medium", "episodes": 300,
    "max_attempts": 10, "success_streak": 2, "r1_apply_k": True,
    "context_format": "messages", "output_format": "plain", "max_output_tokens": 25000, "timeout": 180.0,
    "dataset": str(ROOT / "data/benchmark_300.json"), "start_problem": 1,
    "feedback_policy": "goal_rule_4way_v1",
}
PROVIDER_MODELS = {"openai": "gpt-6.1-sol", "anthropic": "claude-opus-5-5"}
CONDITION_LABELS = {"peg": "기둥 조건", "occupancy": "기둥 상태 조건",
                    "source_sequence": "연속 출발 조건", "visit": "선행 방문 조건"}


@dataclass(frozen=True)

class Rule:
    name: str
    goal_peg: str = "C"
    candidate_rule: dict | None = None
    initial_positions: tuple[str, ...] = ("A", "A", "A")
    initial_family: str | None = None


@dataclass(frozen=True)
class Problem:
    id: str
    rules: tuple[Rule, ...]
    condition: str


def round_policies(config):
    limit = config.get("max_attempts", ATTEMPTS)
    streak = config.get("success_streak", SUCCESS_STREAK)
    apply_r1 = config.get("r1_apply_k", True)
    if type(limit) is not int or limit < 1 or type(streak) is not int or not 1 <= streak <= limit:
        raise ExperimentError("T와 k는 1 <= k <= T를 만족하는 정수여야 합니다.")
    if type(apply_r1) is not bool:
        raise ExperimentError("r1_apply_k는 참/거짓이어야 합니다.")
    return tuple({"round": number, "max_attempts": limit,
                  "success_streak_required": streak if number > 1 or apply_r1 else 1}
                 for number in range(1, 6))


def get_problem(problem_number, problems):
    pool = problems
    if type(problem_number) is not int or not 1 <= problem_number <= len(pool):
        raise ValueError(f"실행 순번은 1~{len(pool)}이어야 합니다.")
    return pool[problem_number - 1]


def problem_snapshot(problem):
    return {"id": problem.id, "condition": problem.condition,
            "rounds": [rule_snapshot(r) for r in problem.rules]}


def rule_snapshot(rule):
    return {**asdict(rule), "initial_positions": list(rule.initial_positions)}


_CANDIDATE_CACHE = {}


def load_candidate_problems(path):
    """Validate every candidate round independently before allowing API execution."""
    from .validation import (validate_round, canonical_problem_under_peg_relabeling,
                                            RULE_TYPES, FAMILIES, ValidationError)
    path = Path(path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
        sha = hashlib.sha256(raw).hexdigest()
        if sha in _CANDIDATE_CACHE:
            return _CANDIDATE_CACHE[sha], data, sha
        if not isinstance(data, dict):
            raise ValueError("데이터셋은 JSON 객체여야 합니다.")
        for field, expected in (("schema_version", 1), ("dataset_version", DATASET_VERSION), ("pegs", list(PEGS)),
                                ("disk_count", DISK_COUNT), ("max_moves", MAX_MOVES)):
            if data.get(field) != expected or type(data.get(field)) is not type(expected):
                raise ValueError(f"잘못된 {field}: {data.get(field)!r}")
        if not isinstance(data.get("dataset_id"), str) or type(data.get("dataset_revision")) is not int:
            raise ValueError("데이터셋 ID와 revision이 필요합니다.")
        rows = data.get("problems")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 300:
            raise ValueError("후보 데이터셋은 1~300문제여야 합니다.")
        problems, ids, signatures = [], set(), set()
        for entry in rows:
            pid = entry.get("id")
            kind = entry.get("rule_type")
            if (not isinstance(pid, str) or len(pid) != 4 or pid[0] != "P" or
                    not pid[1:].isdigit() or not 1 <= int(pid[1:]) <= 999 or pid in ids or kind not in RULE_TYPES):
                raise ValueError("중복 또는 잘못된 문제 ID/규칙 유형입니다.")
            rounds = entry.get("rounds")
            if not isinstance(rounds, list) or len(rounds) != 5:
                raise ValueError(f"{pid}: 5라운드가 필요합니다.")
            compiled, infos, previous_rule = [], [], None
            for index, row in enumerate(rounds):
                info = validate_round(row, kind, previous_rule, index, f"{pid}/R{index + 1}", MAX_MOVES)
                infos.append(info)
                compiled.append(Rule(name=row["name"], goal_peg=row["goal_peg"],
                                     candidate_rule=deepcopy(row["rule"]),
                                     initial_positions=tuple(row["initial_positions"]),
                                     initial_family=row["initial_family"]))
                previous_rule = row["rule"]
            if (len({i["rule"] for i in infos}) != 5 or
                    {i["initial_family"] for i in infos} != set(FAMILIES) or
                    len({r.initial_positions for r in compiled}) != 5 or
                    len({r.goal_peg for r in compiled}) != 4 or
                    any(a.goal_peg == b.goal_peg for a, b in zip(compiled, compiled[1:]))):
                raise ValueError(f"{pid}: 규칙·초기 상태·목표 전환 조건이 맞지 않습니다.")
            signature = canonical_problem_under_peg_relabeling(entry)
            if signature in signatures:
                raise ValueError(f"{pid}: 기둥 이름 치환까지 고려하면 중복 문제입니다.")
            ids.add(pid)
            signatures.add(signature)
            problems.append(Problem(pid, tuple(compiled), kind))
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata는 객체여야 합니다.")
        selection = metadata.get("selection")
        if selection is not None and not isinstance(selection, dict):
            raise ValueError("selection metadata는 객체여야 합니다.")
        if selection and selection.get("selected_problem_ids") is not None:
            if selection["selected_problem_ids"] != [p.id for p in problems]:
                raise ValueError("선정 ID 기록과 실제 문제 순서가 다릅니다.")
        by_type = {kind: [p.id for p in problems if p.condition == kind] for kind in RULE_TYPES}
        actual_types = {kind for kind, problem_ids in by_type.items() if problem_ids}
        if selection and ("type_order" in selection or selection.get("problem_ids_by_type") is not None):
            order = selection.get("type_order")
            if (not isinstance(order, list) or
                    any(not isinstance(kind, str) or kind not in RULE_TYPES for kind in order) or
                    len(set(order)) != len(order) or not actual_types.issubset(order)):
                raise ValueError("선정 유형 순서는 중복 없는 합법 유형이며 실제 유형을 모두 포함해야 합니다.")
            if [pid for kind in order for pid in by_type[kind]] != [p.id for p in problems]:
                raise ValueError("선정 문제의 실행 순서가 기록과 다릅니다.")
        if selection and selection.get("problem_ids_by_type") is not None:
            if selection["problem_ids_by_type"] != {kind: by_type[kind] for kind in order}:
                raise ValueError("유형별 선정 ID 기록과 실제 문제 순서가 다릅니다.")
        for field, actual in (("problem_count", len(problems)), ("round_count", 5 * len(problems))):
            if field in metadata and metadata[field] != actual:
                raise ValueError(f"metadata.{field}가 실제 자료와 다릅니다.")
        from .rules import rule_key
        actual_counts = {kind: Counter(rule_key(r.candidate_rule) for p in problems if p.condition == kind for r in p.rules)
                         for kind in RULE_TYPES}
        if "rule_usage" in metadata:
            reported_usage = metadata["rule_usage"]
            if (not isinstance(reported_usage, dict) or not set(reported_usage).issubset(RULE_TYPES) or
                    not actual_types.issubset(reported_usage)):
                raise ValueError("규칙 빈도 metadata는 합법 유형만 사용하고 실제 유형을 모두 포함해야 합니다.")
            for kind, usage in reported_usage.items():
                if not isinstance(usage, list):
                    raise ValueError(f"{kind}: 규칙 빈도 metadata는 배열이어야 합니다.")
                recorded = {rule_key(item["rule"]): item["round_count"] for item in usage}
                if len(recorded) != len(usage) or recorded != dict(actual_counts[kind]):
                    raise ValueError(f"{kind}: 규칙 빈도 metadata가 실제 라운드와 다릅니다.")
        _CANDIDATE_CACHE[sha] = tuple(problems)
        return tuple(problems), data, sha
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ValidationError) as exc:
        raise ExperimentError(f"후보 데이터셋 검증 실패: {path}: {exc}") from exc


def configured_problems(config):
    pool = load_candidate_problems(config["dataset"])[0]
    start, size = config.get("start_problem", 1), config["episodes"]
    if (type(start) is not int or type(size) is not int or start < 1 or size < 1
            or start + size - 1 > len(pool)):
        raise ExperimentError(f"실행 순번 범위가 데이터셋의 1~{len(pool)}을 벗어납니다.")
    return pool[start - 1:start - 1 + size]


def initial_board_text(rule):
    return ", ".join(f"{peg}=" + json.dumps([d for d in (3, 2, 1) if rule.initial_positions[d - 1] == peg],
                                          separators=(",", ":")) for peg in PEGS)


def submission_message(rule, number, attempt):
    return {"role": "user", "content": "Initial state: " + initial_board_text(rule) +
            '\nSubmit your next complete move plan as {"moves":["AB", ...]}.'}
class ExperimentError(RuntimeError):
    pass


class GenerationFailure(ExperimentError):
    """An unscored generation failure; abandon this problem without resampling."""


class ExperimentStopped(ExperimentError):
    """The user requested a stop after the current response was saved."""


def launch_settings(arguments):
    """Select the provider and model before key entry, without key or network access."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--provider", choices=tuple(PROVIDER_MODELS))
    parser.add_argument("--model")
    parser.add_argument("--resume", type=Path)
    args, _ = parser.parse_known_args(arguments)
    saved = read_json(args.resume / "protocol.json").get("frozen", {}).get("config", {}) if args.resume else {}
    provider = args.provider or saved.get("provider") or ("anthropic" if (args.model or "").startswith("claude-") else "openai")
    if provider not in PROVIDER_MODELS:
        raise ExperimentError("지원하지 않는 API 공급자입니다.")
    model = args.model or saved.get("model") or PROVIDER_MODELS[provider]
    if not isinstance(model, str) or not model or (model.startswith("claude-") != (provider == "anthropic")):
        raise ExperimentError("모델과 API 공급자가 일치하지 않습니다. Claude는 --provider anthropic을 사용하세요.")
    return provider, model


def parse_plan(raw):
    """Parse a plan without repairing its format or requesting another answer."""
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"moves"}:
        return None
    moves = value["moves"]
    if not isinstance(moves, list) or not 1 <= len(moves) <= MAX_MOVES:
        return None
    return moves if all(isinstance(m, str) and m in COMMANDS for m in moves) else None


def evaluate_plan(moves, rule):
    """Evaluate movement restrictions along the physically executable attempted path."""
    from .rules import _signature, _forbids
    restriction = _signature(rule.candidate_rule)
    positions = [PEGS.index(peg) for peg in rule.initial_positions]
    pegs = {peg: [d for d in (3, 2, 1) if rule.initial_positions[d - 1] == peg] for peg in PEGS}
    previous_source = 4
    visited = bool(restriction[0] == "visit" and positions[restriction[1]] == restriction[2])
    violations, base_error = [], None
    if not isinstance(moves, (list, tuple)) or not 1 <= len(moves) <= MAX_MOVES:
        base_error = {"step": None, "move": None, "disk": None, "kind": "invalid_plan"}
    for step, move in enumerate(moves if base_error is None else (), 1):
        if not isinstance(move, str) or move not in COMMANDS:
            base_error = {"step": step, "move": move, "disk": None, "kind": "invalid_command"}
            break
        source, target = move
        disk = pegs[source][-1] if pegs[source] else None
        source_id, target_id = PEGS.index(source), PEGS.index(target)
        # A nonexistent disk cannot trigger a disk-identity restriction. Peg, occupancy,
        # and source-sequence restrictions are identifiable even on an empty source.
        if _forbids(restriction, tuple(positions), disk - 1 if disk is not None else -1,
                    source_id, target_id, previous_source, visited):
            violations.append({"step": step, "move": move, "disk": disk, "kind": restriction[0]})
        if disk is None:
            base_error = {"step": step, "move": move, "disk": None, "kind": "empty_source"}
            break
        if pegs[target] and disk > pegs[target][-1]:
            base_error = {"step": step, "move": move, "disk": disk, "kind": "larger_on_smaller"}
            break
        pegs[target].append(pegs[source].pop())
        positions[disk - 1] = target_id
        previous_source = source_id
        if restriction[0] == "visit" and disk - 1 == restriction[1] and target_id == restriction[2]:
            visited = True
    base_valid = base_error is None
    goal_reached = pegs[rule.goal_peg] == [3, 2, 1] if base_valid else None
    return {"current_success": bool(base_valid and not violations and goal_reached),
            "base_valid": base_valid, "base_error": base_error,
            "movement_rule_violations": violations, "movement_rule_error": bool(violations),
            "goal_reached": goal_reached, "goal_error": not goal_reached if base_valid else None}


def is_success(moves, rule):
    """Require basic legality, additional movement restrictions, and the final goal."""
    return evaluate_plan(moves, rule)["current_success"]


def grade(raw, round_number, problem_number=1, problem=None, *, feedback_policy="binary_v1"):
    feedback_values(feedback_policy)
    moves = parse_plan(raw)
    if problem is None:
        raise ExperimentError("채점에는 명시적인 문제 객체가 필요합니다.")
    rules = problem.rules
    current = evaluate_plan(moves, rules[round_number - 1])
    previous_rule = rules[round_number - 2] if round_number > 1 else None
    if previous_rule is not None:
        previous_rule = replace(previous_rule, initial_positions=rules[round_number - 1].initial_positions,
                                goal_peg=rules[round_number - 1].goal_peg)
    previous = is_success(moves, previous_rule) if previous_rule is not None else None
    result = {
        "moves": moves, "parse_valid": moves is not None,
        **current, "previous_success": previous,
        "persistent_error": current["movement_rule_error"] if round_number > 1 else None,
        "feedback": "Correct" if current["current_success"] else "Incorrect",
    }
    if feedback_policy == "goal_rule_4way_v1":
        if current["current_success"]:
            code, feedback = "correct", "Correct"
        elif current["movement_rule_error"]:
            if current["base_valid"] and current["goal_reached"]:
                code, feedback = "goal_reached_rule_violation", "Correct, but rule violation"
            else:
                code, feedback = "rule_violation", "Rule violation"
        else:
            code, feedback = "incorrect", "Incorrect"
        result.update(correctness_feedback=result["feedback"], feedback=feedback,
                      feedback_code=code, feedback_policy=feedback_policy)
    return result


def make_request(history, config, continuation_history=None):
    """Only participant instructions/history enter requests; no evaluator fields."""
    if config["context_format"] != "messages" or config["output_format"] != "plain":
        raise ExperimentError("현재 실험은 messages/plain 입력·출력만 지원합니다.")
    inputs = deepcopy(continuation_history if continuation_history is not None else history)
    prompt = task_prompt(config.get("feedback_policy", "binary_v1"))
    if config["provider"] == "anthropic":
        return {
            "model": config["model"], "system": prompt["system"], "messages": inputs,
            "max_tokens": config["max_output_tokens"], "output_config": {"effort": config["effort"]},
            "thinking": {"type": "adaptive", "display": "omitted"}, "tools": [],
        }
    return {
        "model": config["model"], "instructions": prompt["system"], "input": inputs,
        "reasoning": {"effort": config["effort"], "context": "all_turns", "summary": "auto"},
        "include": ["reasoning.encrypted_content"], "text": {"format": {"type": "text"}},
        "max_output_tokens": config["max_output_tokens"], "tools": [], "store": False,
    }


def continuation_items(raw):
    """Provider-native state only. Never reconstruct or summarize private reasoning."""
    if raw.get("type") == "message":
        blocks = raw.get("content", [])
        for block in blocks:
            if block.get("type") == "thinking" and not isinstance(block.get("signature"), str):
                raise ExperimentError("서명 없는 thinking 블록은 다음 호출로 전달할 수 없습니다.")
            if block.get("type") == "thinking" and not block["signature"]:
                raise ExperimentError("빈 thinking 서명은 다음 호출로 전달할 수 없습니다.")
            if block.get("type") == "redacted_thinking" and not block.get("data"):
                raise ExperimentError("암호화 상태 없는 redacted_thinking 블록입니다.")
        return [{"role": "assistant", "content": deepcopy(blocks)}]

    def omit_none(value):
        # Same normalization as the existing puzzle adapter: output-only nullable
        # defaults (e.g. reasoning.content/status) are not valid input fields.
        if isinstance(value, dict):
            return {k: omit_none(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [omit_none(v) for v in value]
        return deepcopy(value)
    items = []
    for block in raw.get("output", []):
        if block.get("type") == "reasoning" and not block.get("encrypted_content"):
            raise ExperimentError("암호화 상태가 없는 reasoning 항목입니다. 추론을 버린 채 진행하지 않습니다.")
        item = omit_none(block)
        item.pop("created_by", None)  # Output provenance is not a Responses input field.
        items.append(item)
    return items


def continuation_counts(items):
    blocks = []
    for item in items:
        if item.get("type") == "reasoning":
            blocks.append(item)
        if item.get("role") == "assistant" and isinstance(item.get("content"), list):
            blocks.extend(b for b in item["content"] if b.get("type") in ("thinking", "redacted_thinking"))
    return {"items": len(items), "reasoning_items": len(blocks),
            "opaque_reasoning_items": sum(bool(b.get("encrypted_content") or b.get("signature") or b.get("data")) for b in blocks)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def save_text(path, text):
    """Replace the checkpoint atomically so an interruption preserves the previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def save_json(path, data):
    save_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperimentError(f"기록을 읽을 수 없습니다: {path}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"JSON 객체 형식의 기록이 아닙니다: {path}")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def publish_state(directory, phase, **fields):
    path = directory / "state.json"
    state = read_json(path) if path.exists() else {}
    state.update(phase=phase, updated_at=utc_now(), pid=os.getpid(), **fields)
    save_json(path, state)


def append_event(directory, event, **fields):
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(canonical({"event": event, "at": utc_now(), **fields}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def refresh_reports(directory):
    # Display or report failures must not discard or regenerate completed responses.
    try:
        from .reporting import write_reports
        write_reports(directory)
    except Exception as exc:
        append_event(directory, "report_warning", error_type=type(exc).__name__)
        print("보고서 갱신을 완료하지 못했습니다. 원본 응답과 채점은 저장되어 있습니다.", flush=True)


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@contextmanager
def run_lock(directory):
    """Prevent two processes from writing to the same result directory concurrently."""
    with (directory / ".run.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ExperimentError("이 결과 폴더에서 다른 실행이 진행 중입니다.") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ExperimentError("이 결과 폴더에서 다른 실행이 진행 중입니다.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_protocol(directory, config, resume):
    problems = configured_problems(config)
    feedback_policy = config.get("feedback_policy", "binary_v1")
    prompt = task_prompt(feedback_policy)
    if config.get("context_format") != "messages" or config.get("output_format") != "plain":
        raise ExperimentError("현재 실험은 messages/plain만 지원합니다.")
    path = directory / "protocol.json"
    frozen = {
        "version": FOURWAY_PROTOCOL_VERSION if feedback_policy == "goal_rule_4way_v1" else PROTOCOL_VERSION,
        "config": config, "feedback_policy": feedback_policy,
        "dataset_version": DATASET_VERSION,
        "problems": [problem_snapshot(p) for p in problems],
        "round_policies": list(round_policies(config)),
        "baseline_until_success": False,
        "stop_problem_on_exhaustion": True,
        "r1_apply_k": config["r1_apply_k"],
        "pegs": list(PEGS), "disk_count": DISK_COUNT,
        "goal_policy": "Hidden destination changes between every adjacent round; fixed within a round",
        "metric_policy": "round_per_then_problem_mpe_then_total_mpe_v1",
        "metric_formula": "round PER = persistent errors / actual graded submissions; problem MPE = arithmetic mean of attempted R2-R5 PERs; Total MPE = arithmetic mean of eligible problem MPEs; R1 excluded",
        "public_objective": "A complete plan ends with all three disks stacked on one peg; the accepted destination is undisclosed",
        "disclosure_policy": "The basic single-stack task objective is public. Participant is not told the existence, candidates, types, or changes of hidden destinations and additional restrictions; only basic Hanoi mechanics, submission procedure, progress, and binary feedback are supplied",
        "rounds": [rule_snapshot(r) for r in problems[0].rules],
        "attempts_per_round": config["max_attempts"], "max_attempts_per_round": config["max_attempts"], "max_moves": MAX_MOVES,
        "success_streak_required": config["success_streak"],
        "prompt": prompt,
        "output_validation": "Local JSON parser: exactly moves; 1..30 legal peg-pair commands",
        "prompt_language": "en", "feedback_values": list(feedback_values(feedback_policy)),
        "api_provider": config.get("provider", "openai"),
        "provider_output_policy": ("Both providers receive the same English JSON instructions without server-side JSON/schema/enum/array-length constrained decoding. OpenAI text.format=text; Anthropic output_config.format is omitted. The same local parser validates the original JSON, legal commands and at most 30 moves; completed invalid outputs are scored once, never repaired or regenerated. Native responses and partial stream captures are preserved"),
        "source_sha256": source_hash(), "early_stopping": True,
        "round_end_policy": f"Every round: at most {config['max_attempts']} submissions. R1 criterion: {round_policies(config)[0]['success_streak_required']} consecutive correct answer(s); R2-5 criterion: {config['success_streak']}. Incorrect resets streak. A round exhausting T without meeting its criterion ends the problem; remaining rounds are excluded and the next problem starts with fresh history",
        "context": "Complete public history and original provider-native continuation items retained across all rounds of the same problem; reset only at the next problem. No previous_response_id or server-stored conversation",
        "reasoning_replay_policy": {"scope": "same_problem_across_all_rounds", "reset": "on_problem_change_only", "openai": "Replay complete original output items including encrypted_content; reasoning.context=all_turns; include=reasoning.encrypted_content; store=false. Only output-only null defaults and created_by provenance are omitted for input compatibility", "anthropic": "Replay complete assistant content with thinking signatures and redacted_thinking data, byte-for-byte field values and original block order; thinking=adaptive, display=omitted", "visible_history": "Public episode history remains separate; private blocks are not converted to text messages or shown as feedback", "integrity": "Persist per-trial continuation items and hashes; reconstruct solely from saved native responses on resume, with no new sampling"},
        "reasoning_summary_policy": {
            "version": 1, "openai": "reasoning.summary=auto",
            "anthropic": "thinking.display=omitted remains unchanged; copy readable thinking only if returned",
            "text_file": "trials/<trial>/reasoning_summary.txt",
            "index_file": "reasoning_summaries.jsonl",
            "source": "Copy only readable provider-returned summary text from each saved native response, including unscored responses. Never decode encrypted state, reconstruct reasoning, or make another API request",
            "missing": "Mark not_returned; missing summary does not imply no reasoning. Partial stream events remain in the original stream log",
            "isolation": "Derived text and its metadata are not participant messages or grading input. Original native reasoning items, including summaries and encrypted state, retain the existing replay behavior",
            "resume": "Regenerate derived text and index from saved native responses without resampling",
        },
        "primary_metric": "Round PER = persistent errors / actual submissions; problem MPE = arithmetic mean of attempted R2-R5 PERs; Total MPE = arithmetic mean of eligible problem MPEs. R1 excluded; destination error and previous success do not determine persistent error",
        "persistent_error_policy": "any_current_movement_restriction_violation_v1",
        "diagnostic_path_policy": "Follow the intended plan using basic Hanoi mechanics while ignoring additional restrictions for state updates; record all identifiable restriction violations up to and including the first basic-mechanics-invalid attempted move. Do not assess later unreachable moves. Count at most one persistent error per submission",
        "destination_error_policy": "Separate completion criterion: compare final destination only when the intended plan is valid under basic Hanoi mechanics; otherwise goal_error is null. Destination and movement errors can co-occur",
        "invalid_or_refusal_policy": "Completed visible output is scored once as invalid; binary failure feedback only",
        "incomplete_response_policy": "Save raw response, leave submission unscored, abandon remaining rounds of this problem, and continue with fresh history on next problem. Never regenerate this submission, including on resume",
        "stream_policy": {"enabled": True, "consecutive_visible_whitespace_limit": WHITESPACE_LIMIT, "automatic_retries": 0, "partial_usage": "Observed usage only; early closure may omit final billable usage"},
        "transport_policy": "SDK retries disabled. The public run.py --resume command authorizes one new transport attempt for each unresolved request; another transport error stops the run. Saved responses are reused, never resampled",
    }
    if config.get("dataset"):
        _, dataset, dataset_sha = load_candidate_problems(config["dataset"])
        frozen.update(
            dataset_version=dataset["dataset_version"],
            dataset_identity={"path": str(Path(config["dataset"]).resolve()), "sha256": dataset_sha,
                              "dataset_id": dataset["dataset_id"], "dataset_revision": dataset["dataset_revision"]},
            dataset_snapshot=dataset,
            problem_ids=[p.id for p in problems],
            problem_selection_policy="start_problem is a 1-based index in the selected dataset order; episodes is the number to run. The saved problem_ids list gives the exact execution order. Problem IDs and complete round contents are preserved from the selected source dataset; dataset_snapshot.metadata.selection records subset selection when present",
            dependency_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("hanoi/rules.py", "hanoi/validation.py", "hanoi/prompts.py", "hanoi/reasoning_logs.py")},
            condition_policy="One rule type per problem; one additional restriction in every round R1-R5. Five distinct rule variants and five initial-state families per problem",
            restriction_revision="Four-type schema version 1, dataset version 7; all selected round plans and witnesses independently validated before execution",
            initial_state_policy="Initial positions are ordered by disk 1,2,3. They change across rounds. Each attempt resets the board, previous-source memory, and visit memory to that round's initial state",
            goal_policy="Hidden accepted destination changes between adjacent rounds and may be any of A/B/C/D",
            public_objective="A complete plan ends with all three disks stacked on one peg; the selected accepted peg remains undisclosed",
            disclosure_policy="Only English basic Hanoi mechanics, JSON instructions, the current initial board, and binary Correct/Incorrect feedback are supplied. No round numbers, submission numbers, T, k, hidden targets, additional rule types, rules or changes are disclosed",
            previous_success_policy="Diagnostic only: evaluate the previous additional restriction using the CURRENT initial positions and CURRENT goal. Does not determine persistent error",
        )
    if feedback_policy == "goal_rule_4way_v1":
        frozen.update(
            disclosure_policy="Only English basic Hanoi mechanics, JSON instructions, the current initial board, and four feedback labels are supplied. Feedback distinguishes detected movement-restriction violations and whether the accepted goal was reached. No round numbers, transitions, submission numbers, T, k, accepted target peg, restriction types or identities, violating steps, or persistent-error flags are disclosed",
            feedback_classification="Correct: current_success. Otherwise: detected movement_rule_error with base_valid and goal_reached -> Correct, but rule violation; detected movement_rule_error otherwise -> Rule violation; no detected movement_rule_error -> Incorrect. Goal attainment is unknown for basic-mechanics-invalid plans. Only Correct counts toward k; all other labels remain wrong and reset the streak",
            invalid_or_refusal_policy="Completed invalid JSON or refusal is scored once and receives Incorrect. For parseable plans, detectable additional movement violations through the first basic-mechanics error receive Rule violation; no extra failure location or diagnostic is disclosed",
            round_end_policy=f"Every round: at most {config['max_attempts']} submissions. R1 criterion: {round_policies(config)[0]['success_streak_required']} consecutive Correct answer(s); R2-5 criterion: {config['success_streak']}. Any feedback other than Correct resets the streak, including Correct, but rule violation. A round exhausting T without meeting its criterion ends the problem; remaining rounds are excluded and the next problem starts with fresh history",
        )
    if resume:
        saved = read_json(path)
        if saved.get("frozen_sha256") != digest(saved.get("frozen")) or saved.get("frozen") != frozen:
            raise ExperimentError("코드·설정·규칙이 원래 실행과 다릅니다. 별도의 --output 폴더로 실행하세요.")
    else:
        save_json(path, {"created_at": utc_now(), "frozen_sha256": digest(frozen), "frozen": frozen})


def receive_stream(resource, request, prefix):
    """Persist stream events immediately; inspect only participant-visible text."""
    # Offline clients may expose create only. Production clients receive SSE streams.
    if not hasattr(resource, "stream"):
        response = resource.create(**request)
        return response.model_dump(mode="json"), getattr(response, "_request_id", None), None
    anthropic = "messages" in request
    consecutive = 0
    chunks = []
    observed_usage = None
    response_id = None
    native_final = None
    event_file = Path(str(prefix) + "_stream.jsonl")
    with event_file.open("w", encoding="utf-8") as log:
        with (resource.stream(**request) if anthropic else resource.create(**request, stream=True)) as stream:
            http_response = getattr(stream, "response", None) or getattr(stream, "_response", None)
            headers = http_response.headers
            request_id = headers.get("request-id" if anthropic else "x-request-id")
            for event in stream:
                data = event.model_dump(mode="json")
                log.write(canonical(data) + "\n")
                log.flush()
                kind = data.get("type")
                delta = ""
                if anthropic:
                    if kind == "message_start":
                        message = data.get("message", {})
                        response_id = message.get("id")
                        observed_usage = message.get("usage")
                    elif kind == "message_delta" and data.get("usage"):
                        observed_usage = {**(observed_usage or {}), **data["usage"]}
                    elif kind == "content_block_delta" and data.get("delta", {}).get("type") == "text_delta":
                        delta = data["delta"].get("text", "")
                    elif kind == "content_block_start" and data.get("content_block", {}).get("type") == "text":
                        delta = data["content_block"].get("text", "")
                else:
                    if kind in ("response.completed", "response.incomplete", "response.failed"):
                        native_final = data.get("response")
                    if isinstance(data.get("response"), dict):
                        response_id = data["response"].get("id", response_id)
                        observed_usage = data["response"].get("usage") or observed_usage
                    if kind == "response.output_text.delta":
                        delta = data.get("delta", "")
                if delta:
                    chunks.append(delta)
                    for char in delta:
                        consecutive = consecutive + 1 if char.isspace() else 0
                        if consecutive >= WHITESPACE_LIMIT:
                            # This is an explicitly labeled local partial capture, not a native completed response.
                            return {"id": response_id, "type": "message" if anthropic else "response", "status": "locally_aborted",
                                    "partial_visible_text": "".join(chunks), "usage": observed_usage,
                                    "usage_is_partial": True}, request_id, "consecutive_whitespace_limit"
            if not anthropic:
                if native_final is None:
                    raise ExperimentError("최종 상태 이벤트 없이 API 스트림이 끝났습니다. 원본 이벤트를 보존했습니다.")
                return native_final, request_id, None
            response = stream.get_final_message()
            return response.model_dump(mode="json"), request_id, None


def obtain_response(directory, request, client, retry_pending, on_request=None):
    """Reuse completed responses; do not automatically repeat requests with unknown outcomes."""
    directory.mkdir(parents=True, exist_ok=True)
    request_file = directory / "request.json"
    starts = sorted(directory.glob("transport_*_started.json"))
    saved_responses = sorted(directory.glob("transport_*_response.json"))
    if request_file.exists():
        if read_json(request_file) != request:
            raise ExperimentError(f"저장된 요청과 재구성한 대화가 다릅니다: {directory}")
    elif starts or saved_responses or (directory / "record.json").exists():
        raise ExperimentError(f"요청 원본이 빠진 checkpoint입니다: {directory}")
    else:
        save_json(request_file, request)
    for index, start in enumerate(starts, 1):
        if start.name != f"transport_{index:03d}_started.json" or read_json(start).get("request_sha256") != digest(request):
            raise ExperimentError(f"전송 시작 기록의 순서 또는 요청 해시가 다릅니다: {directory}")
    if len(saved_responses) > 1:
        raise ExperimentError(f"한 제출에 응답이 여러 개 있습니다: {directory}")
    if saved_responses:
        matching_start = saved_responses[0].with_name(saved_responses[0].name.replace("_response.json", "_started.json"))
        if matching_start not in starts:
            raise ExperimentError(f"응답에 대응하는 전송 시작 기록이 없습니다: {directory}")
        envelope = read_json(saved_responses[0])
        if envelope.get("request_sha256") != digest(request):
            raise ExperimentError(f"응답의 요청 해시가 일치하지 않습니다: {directory}")
        if envelope.get("response_sha256") != digest(envelope.get("response")):
            raise ExperimentError(f"응답 원본 해시가 일치하지 않습니다: {directory}")
        return envelope
    if (directory / "record.json").exists():
        raise ExperimentError(f"채점 기록은 있지만 응답 원본이 없습니다: {directory}")
    if starts and not retry_pending:
        raise ExperimentError(
            f"응답이 저장되지 않은 요청이 있습니다: {directory}\n"
            "서버 실행 여부가 불명확합니다. 재호출하려면 --resume 경로 --retry-pending을 명시하세요."
        )
    attempt = len(starts) + 1
    prefix = directory / f"transport_{attempt:03d}"
    save_json(Path(str(prefix) + "_started.json"), {
        "started_at": utc_now(), "request_sha256": digest(request),
        "explicit_retry_of_unresolved_request": bool(starts),
    })
    if on_request:
        on_request(attempt)
    begin = time.monotonic()
    try:
        resource = client.messages if "messages" in request and "max_tokens" in request else client.responses
        raw, request_id, generation_failure = receive_stream(resource, request, prefix)
    except Exception as exc:
        # Never print or save exception bodies that may contain keys or headers.
        details = {"error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None),
                   "request_id": getattr(exc, "request_id", None), "at": utc_now()}
        save_json(Path(str(prefix) + "_error.json"), details)
        raise ExperimentError(
            f"API 요청 중단: {details['error_type']} (HTTP {details['status_code']}). "
            "이 제출은 채점하지 않았습니다. API 키·모델 접근 권한·할당량·네트워크를 확인하세요."
        ) from None
    envelope = {
        "received_at": utc_now(), "elapsed_seconds": time.monotonic() - begin,
        "request_sha256": digest(request), "response_sha256": digest(raw), "response": raw,
        "request_id": request_id, "transport_attempt": attempt,
        "generation_failure": generation_failure, "stream": True,
    }
    save_json(Path(str(prefix) + "_response.json"), envelope)
    return envelope


def visible_response(raw):
    """Extract only visible grading text; preserve reasoning state in continuation_items."""
    if raw.get("status") == "locally_aborted":
        raise GenerationFailure("consecutive_whitespace_limit")
    if raw.get("type") == "message" and "stop_reason" in raw:
        if raw.get("stop_reason") in ("max_tokens", "pause_turn", "model_context_window_exceeded"):
            raise GenerationFailure("anthropic_" + raw["stop_reason"])
        if raw.get("role") != "assistant" or raw.get("stop_reason") not in ("end_turn", "refusal"):
            raise ExperimentError(
                f"완료되지 않았거나 예상하지 않은 Anthropic 응답(stop_reason={raw.get('stop_reason')})을 저장하고 중단했습니다. "
                "해당 제출은 미채점이며 재생성하지 않습니다."
            )
        parts = []
        for block in raw.get("content", []):
            if block.get("type") in ("thinking", "redacted_thinking"):
                continue
            if block.get("type") != "text":
                raise ExperimentError(f"예상하지 않은 Anthropic 출력/도구 항목입니다: {block.get('type')}")
            parts.append(block.get("text", ""))
        text = "".join(parts).strip()
        if not text:
            raise GenerationFailure("empty_visible_response")
        return text, raw["stop_reason"] == "refusal"
    if raw.get("status") == "incomplete":
        raise GenerationFailure("incomplete_response")
    if raw.get("status") != "completed":
        raise ExperimentError(
            f"완료되지 않은 API 응답(status={raw.get('status')})을 저장하고 중단했습니다. "
            "해당 제출은 미채점이며 재생성하지 않습니다."
        )
    parts, refused = [], False
    for item in raw.get("output", []):
        kind = item.get("type")
        if kind == "reasoning":
            continue
        if kind != "message" or item.get("role") != "assistant":
            raise ExperimentError(f"예상하지 않은 출력/도구 항목입니다: {kind}")
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content["text"])
            elif content.get("type") == "refusal":
                parts.append(content["refusal"])
                refused = True
            else:
                raise ExperimentError("예상하지 않은 메시지 내용 형식입니다.")
    text = "".join(parts).strip()
    if not text:
        raise GenerationFailure("empty_visible_response")
    return text, refused


def response_usage(raw):
    """Normalize logged input/output token totals while preserving native response usage."""
    usage = raw.get("usage")
    if raw.get("type") != "message" or not isinstance(usage, dict):
        return usage
    normalized = dict(usage)
    uncached = usage.get("input_tokens", 0) or 0
    normalized["input_tokens_uncached"] = uncached
    normalized["input_tokens"] = uncached + (usage.get("cache_creation_input_tokens", 0) or 0) + (usage.get("cache_read_input_tokens", 0) or 0)
    normalized["total_tokens"] = normalized["input_tokens"] + (usage.get("output_tokens", 0) or 0)
    return normalized


def summarize(records, episodes, required_successes=SUCCESS_STREAK, policies=None, problems=None):
    policies = policies or round_policies(dict(DEFAULTS, success_streak=required_successes))
    if type(required_successes) is not int or not 1 <= required_successes <= policies[1]["max_attempts"]:
        raise ValueError("연속 정답 기준은 1부터 최대 시도 수 사이의 정수여야 합니다.")
    criterion = required_successes
    def stats(rows):
        pe = sum(bool(x["persistent_error"]) for x in rows)
        failures = sum(not x["current_success"] for x in rows)
        return {"attempts": len(rows), "successes": len(rows) - failures,
                "failures": failures, "persistent_errors": pe, "general_errors": failures - pe,
                "persistent_error_rate": pe / len(rows) if rows else None,
                "failure_rate": failures / len(rows) if rows else None,
                "movement_rule_errors": sum(x.get("movement_rule_error") is True for x in rows),
                "goal_errors": sum(x.get("goal_error") is True for x in rows),
                "goal_evaluated_attempts": sum(x.get("goal_error") is not None for x in rows),
                "base_errors": sum(x.get("parse_valid") is True and x.get("base_valid") is False for x in rows)}

    episode_results = []
    for episode in range(1, episodes + 1):
        batch = [x for x in records if x["episode"] == episode]
        rounds = []
        exhausted_at = None
        for number in range(1, 6):
            required = policies[number - 1]["success_streak_required"]
            limit = policies[number - 1]["max_attempts"]
            rows = [x for x in batch if x["round"] == number]
            entry = {"round": number, **stats(rows)}
            entry["first_success_attempt"] = next((x["attempt"] for x in rows if x["current_success"]), None)
            streak, peak, met_at = 0, 0, None
            for row in rows:
                streak = streak + 1 if row["current_success"] else 0
                peak = max(peak, streak)
                if streak >= required and met_at is None:
                    met_at = row["attempt"]
            entry.update(success_streak=streak, max_success_streak=peak,
                         criterion_met_at=met_at, success_streak_required=required)
            valid_sequence = ([x["attempt"] for x in rows] == list(range(1, len(rows) + 1))
                              and (limit is None or len(rows) <= limit))
            valid_sequence = valid_sequence and (met_at is None or met_at == len(rows))
            terminal = bool(rows) and (met_at is not None or len(rows) == limit)
            entry["complete"] = bool(valid_sequence and terminal)
            entry["max_attempts"] = limit
            entry["status"] = ("success" if met_at is not None else "exhausted") if entry["complete"] else ("in_progress" if rows else "pending")
            failed, repeats = set(), 0
            for row in rows:
                key = canonical(row["moves"]) if row["moves"] is not None else row["raw_output"]
                if not row["current_success"]:
                    repeats += key in failed
                    failed.add(key)
            entry["repeat_of_identical_failed_plan"] = repeats
            if number == 1:
                entry["persistent_errors"] = entry["persistent_error_rate"] = None
            if exhausted_at is not None and not rows:
                entry.update(status="excluded", excluded_reason="prior_round_exhausted", excluded_after_round=exhausted_at)
            if entry["status"] == "exhausted" and exhausted_at is None:
                exhausted_at = number
            rounds.append(entry)
        scored = [x for x in batch if x["round"] > 1]
        scored_rounds = [row for row in rounds if row["round"] > 1 and row["attempts"] > 0]
        error_rounds = [row for row in rounds if row["attempts"] > 0]
        round_rates = [row["persistent_error_rate"] for row in scored_rounds]
        round_error_rates = [row["failure_rate"] for row in error_rounds]
        mpe = sum(round_rates) / len(round_rates) if round_rates else None
        mean_error_rate = sum(round_error_rates) / len(round_error_rates) if round_error_rates else None
        problem = get_problem(episode, problems) if problems is not None else None
        condition = problem.condition if problem is not None else None
        episode_results.append({"episode": episode, "problem_id": problem.id if problem is not None else f"P{episode:02d}",
                                "condition": condition,
                                "complete": all(r["complete"] or r["status"] == "excluded" for r in rounds),
                                "exhausted_at_round": exhausted_at,
                                "rounds": rounds, "scored": stats(scored),
                                "per_by_round": {str(row["round"]): row["persistent_error_rate"] for row in rounds if row["round"] > 1},
                                "error_rate_by_round": {str(row["round"]): row["failure_rate"] for row in rounds},
                                "eligible_scored_rounds": len(scored_rounds), "mpe": mpe,
                                "mean_error_rate": mean_error_rate,
                                "eligible_error_rounds": len(error_rounds),
                                "error_metric_eligible": bool(error_rounds),
                                "metric_eligible": bool(scored_rounds)})
    scored = [x for x in records if x["round"] > 1]
    problem_mpes = [row["mpe"] for row in episode_results if row["metric_eligible"]]
    problem_mean_error_rates = [row["mean_error_rate"] for row in episode_results if row["error_metric_eligible"]]
    rates = [x["scored"]["persistent_error_rate"] for x in episode_results if x["complete"] and x["scored"]["persistent_error_rate"] is not None]
    conditions = {}
    for condition, label in CONDITION_LABELS.items():
        members = [row for row in episode_results if row["condition"] == condition]
        if not members:
            continue
        numbers = {row["episode"] for row in members}
        completed = [row for row in members if row["complete"]]
        group_rates = [row["scored"]["persistent_error_rate"] for row in completed if row["scored"]["persistent_error_rate"] is not None]
        group_mpes = [row["mpe"] for row in members if row["metric_eligible"]]
        group_mean_error_rates = [row["mean_error_rate"] for row in members if row["error_metric_eligible"]]
        conditions[condition] = {"label": label, "problem_count": len(members),
                                 "completed_problems": len(completed),
                                 **stats([row for row in scored if row["episode"] in numbers]),
                                 "metric_eligible_problems": len(group_mpes),
                                 "error_metric_eligible_problems": len(group_mean_error_rates),
                                 "mpe": sum(group_mpes) / len(group_mpes) if group_mpes else None,
                                 "mean_error_rate": sum(group_mean_error_rates) / len(group_mean_error_rates) if group_mean_error_rates else None,
                                 "complete_problem_rate_mean": sum(group_rates) / len(group_rates) if group_rates else None}
    minimum = 0
    for policy in reversed(policies):
        minimum = min(policy["max_attempts"], policy["success_streak_required"] + minimum)
    return {
        "source": "Model API", "complete": all(e["complete"] for e in episode_results),
        "early_stopping": True,
        "success_streak_required": criterion,
        "baseline_until_success": False,
        "planned_submissions": sum(p["max_attempts"] for p in policies) * episodes,
        "min_submissions": minimum * episodes,
        "max_submissions": sum(p["max_attempts"] for p in policies) * episodes,
        "scored_max_submissions": sum(p["max_attempts"] for p in policies[1:]) * episodes,
        "baseline_submissions": sum(x["round"] == 1 for x in records),
        "recorded_submissions": len(records),
        "completed_rounds": sum(r["complete"] for e in episode_results for r in e["rounds"]),
        "total_rounds": 5 * episodes,
        "successful_rounds": sum(r["status"] == "success" for e in episode_results for r in e["rounds"]),
        "exhausted_rounds": sum(r["status"] == "exhausted" for e in episode_results for r in e["rounds"]),
        "scored_rounds": [2, 3, 4, 5], "pooled": stats(scored),
        "metric_policy": "round_per_then_problem_mpe_then_total_mpe_v1",
        "error_metric_policy": "round_error_rate_r1_r5_then_problem_mean_then_total_mean_v2",
        "metric_definition": {
            "round_per": "persistent_errors / actual graded submissions in that round",
            "problem_mpe": "unweighted mean of available R2-R5 PERs with at least one graded submission",
            "total_mpe": "unweighted mean of problem MPEs with at least one graded R2-R5 submission",
            "condition_mpe": "unweighted mean of eligible problem MPEs within each condition",
            "round_error_rate": "incorrect answers / actual graded submissions in that round",
            "problem_mean_error_rate": "unweighted mean of available R1-R5 error rates with at least one graded submission",
            "total_mean_error_rate": "unweighted mean of problem mean error rates with at least one graded R1-R5 submission, including R1-only problems",
            "condition_mean_error_rate": "unweighted mean of eligible R1-R5 problem mean error rates within each condition",
            "r1": "excluded from PER/MPE; included in mean error rate even for R1-only problems",
            "missing": "unattempted rounds are excluded, never filled with zero; MPE requires a graded R2-R5 submission, mean error requires a graded R1-R5 submission",
            "observed_failures": "graded incorrect submissions and exhausted rounds are included; ungraded generation failures are not submissions",
            "scale": "fraction in JSON; percentage for display",
            "diagnostics": "pooled, scored, condition failure_rate, excluding_first_attempt and complete_*_rate_mean retain submission-weighted legacy diagnostics; they are not the equal-round then equal-problem primary metrics",
        },
        "total_mpe": sum(problem_mpes) / len(problem_mpes) if problem_mpes else None,
        "total_mean_error_rate": sum(problem_mean_error_rates) / len(problem_mean_error_rates) if problem_mean_error_rates else None,
        "metric_eligible_problems": len(problem_mpes),
        "error_metric_eligible_problems": len(problem_mean_error_rates),
        "problem_mpes": problem_mpes,
        "problem_mean_error_rates": problem_mean_error_rates,
        "problems_in_mpe_target_band": sum(0.30 <= value <= 0.50 for value in problem_mpes),
        "excluding_first_attempt": stats([x for x in scored if x["attempt"] > 1]),
        "episodes": episode_results,
        "conditions": conditions,
        "complete_episode_rate_mean": sum(rates) / len(rates) if rates else None,
        "complete_episode_rates": rates,
        "episodes_in_target_band": sum(0.30 <= r <= 0.50 for r in rates),
        "completed_episodes": sum(x["complete"] for x in episode_results),
        "metric_eligible_completed_episodes": len(rates),
        "excluded_rounds": sum(r["status"] == "excluded" for e in episode_results for r in e["rounds"]),
        "exhausted_problems": sum(x["exhausted_at_round"] is not None for x in episode_results),
        "stop_problem_on_exhaustion": True,
        "format_failures": sum(not x["parse_valid"] for x in records),
        "refusals": sum(x["refusal"] for x in records),
        "usage": {key: sum((x.get("usage") or {}).get(key, 0) or 0 for x in records)
                  for key in ("input_tokens", "output_tokens", "total_tokens")},
        "note": ("Each problem starts with fresh history. R1 has a finite submission budget and is excluded from PER/MPE. ") + (" A round exhausting its budget without meeting its criterion ends the problem; remaining rounds are excluded, never scored as failures. ") + " R2-5 persistent error means any current additional movement-restriction violation, counted once per submission and independent of destinations or previous success. Goal errors are separate and may overlap with persistent errors. Round PER uses actual graded submissions without padding. Problem MPE equally averages attempted scored-round PERs; Total MPE equally averages eligible problem MPEs, including partially observed problems. Mean error rate includes R1: incorrect submissions / actual graded submissions per observed R1-5 round, equal round means per problem, then equal problem means per condition or overall for every problem with at least one graded submission, including R1-only problems. Pooled R2-5 rates are diagnostic only. Partial runs are not final estimates.",
    }


def execute(directory, config, client, retry_pending=False):
    """Reconstruct the conversation by reading response checkpoints in order."""
    from .terminal import TerminalReporter
    from .reasoning_logs import summary_artifact
    policies = round_policies(config)
    feedback_policy = config.get("feedback_policy", "binary_v1")
    prompt = task_prompt(feedback_policy)
    problems = configured_problems(config)
    dataset_identity = read_json(directory / "protocol.json")["frozen"]["dataset_identity"] if config.get("dataset") else None
    reporter = TerminalReporter(directory, config, problems[0].rules, config["max_attempts"], config["success_streak"],
                                problems=problems, round_policies=policies)
    reporter.start()
    records, generation_failures, reasoning_summaries = [], [], []
    call = 0
    def snapshot_result(finished=False):
        result = summarize(records, config["episodes"], required_successes=config["success_streak"], policies=policies, problems=problems)
        result.update(execution_finished=finished, generation_failures=list(generation_failures),
                      generation_failure_count=len(generation_failures),
                      api_submissions=len(records) + len(generation_failures),
                      usage_incomplete=any(x["usage_incomplete"] for x in generation_failures))
        if dataset_identity is not None:
            result.update(dataset_identity=dataset_identity, problem_ids=[p.id for p in problems],
                          start_problem=config.get("start_problem", 1))
        audits = [record.get("reasoning_replay", {}) for record in records]
        result["reasoning_replay"] = {"policy": "provider_native_same_problem_all_rounds",
                                      "submissions_with_replayed_reasoning": sum(a.get("input", {}).get("reasoning_items", 0) > 0 for a in audits),
                                      "returned_reasoning_items": sum(a.get("output", {}).get("reasoning_items", 0) for a in audits),
                                      "returned_opaque_reasoning_items": sum(a.get("output", {}).get("opaque_reasoning_items", 0) for a in audits),
                                      "effective_contexts": sorted({a["effective_context"] for a in audits if a.get("effective_context")})}
        result["scored_usage"] = dict(result["usage"])
        for failure in generation_failures:
            for key in result["usage"]:
                result["usage"][key] += (failure.get("usage") or {}).get(key, 0) or 0
            entry = result["episodes"][failure["episode"] - 1]
            entry["generation_failure"] = failure
            for row in entry["rounds"]:
                if row["round"] >= failure["round"]:
                    row["status"] = "generation_failed" if row["round"] == failure["round"] else "excluded"
        result["excluded_rounds"] = sum(row["status"] == "excluded" for episode in result["episodes"] for row in episode["rounds"])
        return result
    if not (directory / "summary.json").exists():
        save_json(directory / "summary.json", snapshot_result())
    for episode in range(1, config["episodes"] + 1):
        problem = get_problem(episode, problems)
        abandoned = False
        history = [{"role": "user", "content": prompt["user"]}]
        api_history = deepcopy(history)
        for number in range(1, 6):
            policy = policies[number - 1]
            required = policy["success_streak_required"]
            limit = policy["max_attempts"]
            streak = 0
            reporter.round_start(episode, number)
            for attempt in count(1):
                if limit is not None and attempt > limit:
                    break
                if (directory / "stop.request").exists():
                    raise ExperimentStopped("현재까지 받은 응답을 저장하고 중단했습니다. 재개할 수 있습니다.")
                call += 1
                progress = submission_message(problem.rules[number - 1], number, attempt)
                next_history = history + [progress]
                next_api_history = api_history + [deepcopy(progress)]
                request = make_request(next_history, config, next_api_history)
                trial = directory / "trials" / f"{call:05d}_e{episode:03d}_r{number}_a{attempt:02d}"
                recovered = (trial / "record.json").exists()
                if not recovered:
                    save_json(directory / f"episode_{episode:03d}_history.json", history)
                    save_text(directory / "submissions.jsonl", "".join(canonical(r) + "\n" for r in records))
                    save_json(directory / "summary.json", snapshot_result())
                def request_started(transport_attempt):
                    fields = {"call": call, "episode": episode, "problem_id": problem.id, "condition": problem.condition,
                              "round": number, "attempt": attempt}
                    publish_state(directory, "requesting", **fields, success_streak=streak,
                                  success_streak_required=required, message="모델 응답을 기다리는 중입니다.")
                    append_event(directory, "request_started", **fields, transport_attempt=transport_attempt)
                    reporter.waiting(attempt, transport_attempt)
                envelope = obtain_response(trial, request, client, retry_pending, request_started)
                # The native response is already checkpointed. Derived logs never cause resampling.
                summary_text, summary_metadata = summary_artifact(
                    envelope, provider=config["provider"], model=config["model"], call=call,
                    episode=episode, problem_id=problem.id, condition=problem.condition,
                    round_number=number, attempt=attempt, trial_path=trial.relative_to(directory).as_posix(),
                )
                summary_file = trial / "reasoning_summary.txt"
                if not summary_file.exists() or summary_file.read_text(encoding="utf-8") != summary_text:
                    save_text(summary_file, summary_text)
                reasoning_summaries.append(summary_metadata)
                save_text(directory / "reasoning_summaries.jsonl",
                          "".join(canonical(item) + "\n" for item in reasoning_summaries))
                try:
                    raw, refused = visible_response(envelope["response"])
                except GenerationFailure as exc:
                    failure = {"call": call, "episode": episode, "problem_id": problem.id,
                               "condition": problem.condition, "round": number, "attempt": attempt,
                               "reason": str(exc), "scored": False, "retry": False,
                               "usage": response_usage(envelope["response"]),
                               "usage_incomplete": bool(envelope["response"].get("usage_is_partial")) or envelope["response"].get("usage") is None,
                               "elapsed_seconds": envelope["elapsed_seconds"],
                               "response_sha256": envelope["response_sha256"],
                               "trial": str(trial.relative_to(directory))}
                    failure_file = trial / "generation_failure.json"
                    if failure_file.exists() and read_json(failure_file) != failure:
                        raise ExperimentError("생성 실패 checkpoint가 기존 기록과 다릅니다.")
                    if not failure_file.exists():
                        save_json(failure_file, failure)
                        append_event(directory, "generation_failed", **failure)
                    generation_failures.append(failure)
                    save_json(directory / "summary.json", snapshot_result())
                    reporter.generation_failure(failure)
                    abandoned = True
                    break
                native_items = continuation_items(envelope["response"])
                continuation = {"response_sha256": envelope["response_sha256"],
                                "items_sha256": digest(native_items), "items": native_items}
                continuation_file = trial / "continuation.json"
                if continuation_file.exists() and read_json(continuation_file) != continuation:
                    raise ExperimentError("기존 추론 상태와 응답 원본에서 복원한 상태가 다릅니다.")
                if not continuation_file.exists():
                    save_json(continuation_file, continuation)
                replay_audit = {"input": continuation_counts(next_api_history),
                                "output": continuation_counts(native_items),
                                "output_sha256": continuation["items_sha256"],
                                "effective_context": (envelope["response"].get("reasoning") or {}).get("context")}
                score = grade(raw, number, episode, problem=problem, feedback_policy=feedback_policy)
                if refused:
                    score = grade(None, number, episode, problem=problem, feedback_policy=feedback_policy)
                streak = streak + 1 if score["current_success"] else 0
                record = {
                    "call": call, "episode": episode, "problem_id": problem.id, "condition": problem.condition,
                    "round": number, "attempt": attempt,
                    "raw_output": raw, **score, "refusal": refused, "reasoning_replay": replay_audit,
                    "response_id": envelope["response"].get("id"),
                    "model_returned": envelope["response"].get("model"),
                    "usage": response_usage(envelope["response"]),
                    "elapsed_seconds": envelope["elapsed_seconds"],
                    "transport_attempt": envelope["transport_attempt"],
                    "request_sha256": envelope["request_sha256"],
                    "success_streak": streak, "round_success": streak >= required,
                    "success_streak_required": required,
                }
                if config.get("dataset"):
                    record.update(initial_positions=list(problem.rules[number - 1].initial_positions),
                                  initial_family=problem.rules[number - 1].initial_family,
                                  rule_snapshot=rule_snapshot(problem.rules[number - 1]),
                                  dataset_id=dataset_identity["dataset_id"], dataset_sha256=dataset_identity["sha256"],
                                  dataset_index=config.get("start_problem", 1) + episode - 1)
                record_file = trial / "record.json"
                if record_file.exists() and read_json(record_file) != record:
                    raise ExperimentError(f"기존 채점 기록과 재채점 결과가 다릅니다: {trial}")
                if not recovered:
                    save_json(record_file, record)
                records.append(record)
                api_history = next_api_history + native_items + [{"role": "user", "content": score["feedback"]}]
                history = next_history + [{"role": "assistant", "content": raw},
                                          {"role": "user", "content": score["feedback"]}]
                if not recovered:
                    save_json(directory / f"episode_{episode:03d}_history.json", history)
                    save_text(directory / "submissions.jsonl", "".join(canonical(r) + "\n" for r in records))
                    save_json(directory / "summary.json", snapshot_result())
                    append_event(directory, "submission_scored", call=call, episode=episode,
                                 problem_id=problem.id, condition=problem.condition, round=number,
                                 attempt=attempt, feedback=score["feedback"], persistent_error=score["persistent_error"],
                                 success_streak=streak, round_success=streak >= required,
                                 elapsed_seconds=envelope["elapsed_seconds"])
                    publish_state(directory, "running", call=call, episode=episode, problem_id=problem.id, condition=problem.condition,
                                  round=number, attempt=attempt,
                                  success_streak=streak, success_streak_required=required,
                                  message=f"{call}개 제출을 저장했습니다.")
                    refresh_reports(directory)
                reporter.submission(record, recovered=recovered)
                # Restore the streak from recovered responses to preserve the stopping point.
                if streak >= required:
                    break
            round_summary = snapshot_result()["episodes"][episode - 1]["rounds"][number - 1]
            reporter.round_end(round_summary)
            if abandoned:
                break
            if round_summary["status"] == "exhausted":
                if not recovered:
                    append_event(directory, "problem_exhausted", episode=episode, problem_id=problem.id,
                                 round=number, max_attempts=limit, excluded_rounds=list(range(number + 1, 6)))
                reporter.problem_exhausted(episode, number)
                break
        save_json(directory / f"episode_{episode:03d}_history.json", history)
    result = snapshot_result(finished=True)
    save_text(directory / "submissions.jsonl", "".join(canonical(r) + "\n" for r in records))
    save_json(directory / "summary.json", result)
    publish_state(directory, "complete" if result["complete"] else "finished_with_failures", call=call,
                  message=f"실행 종료 · 한도 소진 {result['exhausted_problems']}문제 · 미실행 {result['excluded_rounds']}라운드 · 생성 실패 {len(generation_failures)}문제 · 자동 재시도 없음")
    refresh_reports(directory)
    reporter.finish(result)
    return result


def make_client(config):
    """Construct and authenticate the selected provider before any run writes."""
    from .auth import checked_client
    return checked_client(config)


def main(argv=None, *, client_factory=None, check_only=False):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="새 결과 폴더 (기존 폴더 덮어쓰기 금지)")
    destination.add_argument("--resume", type=Path, help="동일 코드/설정으로 기존 실험 재개")
    parser.add_argument("--provider", choices=tuple(PROVIDER_MODELS), help="API 공급원: openai 또는 anthropic")
    parser.add_argument("--model", help="기본값 gpt-6.1-sol; 자동 모델 변경 없음")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--problems", "--episodes", dest="episodes", type=int,
                        help="실행할 문제 수. 기본값은 선택 데이터셋의 남은 전체 문제; --episodes는 호환 별칭")
    parser.add_argument("--dataset", type=Path, help="실행할 문제 JSON. 기본값: data/benchmark_300.json (3유형 각 100문제)")
    parser.add_argument("--start-problem", type=int, help="데이터셋 내 1부터 시작하는 실행 순번; 원래 Pxxx ID와 다를 수 있음")
    parser.add_argument("--context-format", choices=["messages"], help="내부 추론 상태 유지를 위해 역할별 메시지만 사용")
    parser.add_argument("--output-format", choices=["plain"], help="JSON은 프롬프트로 요청하고 로컬 파서로 채점")
    parser.add_argument("--feedback-policy", choices=FEEDBACK_POLICIES,
                        help="binary_v1: 정오답; goal_rule_4way_v1: 정답/오답/규칙 위반/정답이지만 규칙 위반")
    parser.add_argument("--T", "--t", "--max-attempts", dest="max_attempts", type=int, help="모든 라운드의 최대 제출 수 T (기본 10)")
    parser.add_argument("--k", "--success-streak", dest="success_streak", type=int, help="연속 정답 기준 k (기본 2)")
    parser.add_argument("--r1-apply-k", action=argparse.BooleanOptionalAction, default=None, help="R1에도 k 적용 (기본 true)")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--dry-run", action="store_true", help="첫 API 요청만 저장. API 호출/채점 없음")
    parser.add_argument("--retry-pending", action="store_true", help="결과 불명확 요청의 재호출을 명시적으로 허용")
    parser.add_argument("--preserve-stop-request", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.retry_pending and not args.resume:
        parser.error("--retry-pending은 --resume과 함께 사용하세요.")
    config = dict(DEFAULTS)
    selected_provider, selected_model = launch_settings(argv if argv is not None else sys.argv[1:])
    config.update(provider=selected_provider, model=selected_model)
    if args.resume:
        saved = read_json(args.resume / "protocol.json")
        frozen = saved.get("frozen", {})
        if saved.get("frozen_sha256") != digest(frozen) or not isinstance(frozen.get("config"), dict):
            raise ExperimentError("protocol.json 형식 또는 해시가 일치하지 않습니다.")
        config = dict(frozen["config"])
        if set(config) != set(DEFAULTS):
            raise ExperimentError("protocol.json의 설정 항목이 올바르지 않습니다.")
    else:
        config["dataset"] = str((args.dataset or Path(DEFAULTS["dataset"])).expanduser().resolve())
        pool, dataset, _ = load_candidate_problems(config["dataset"])
        settings = dataset.get("experiment_defaults", {})
        config.update(max_attempts=settings.get("T", DEFAULTS["max_attempts"]), success_streak=settings.get("k", 2),
                      r1_apply_k=settings.get("r1_apply_k", True),
                      feedback_policy=settings.get("feedback_policy", DEFAULTS["feedback_policy"]),
                      episodes=len(pool) - ((args.start_problem or 1) - 1))
    for key in DEFAULTS:
        value = getattr(args, key)
        if key == "dataset" and value is not None:
            value = str(value.expanduser().resolve())
        if value is not None:
            if args.resume and value != config[key]:
                raise ExperimentError(f"재개할 때 설정을 바꿀 수 없습니다: {key}")
            config[key] = value
    problems = configured_problems(config)
    if config["max_output_tokens"] < 1 or config["timeout"] <= 0:
        parser.error("max-output-tokens와 timeout은 양수여야 합니다.")
    round_policies(config)  # validate before client creation or output directory creation
    try:
        feedback_values(config["feedback_policy"])
    except ValueError as exc:
        raise ExperimentError(str(exc)) from exc
    client = None
    directory = (args.resume or args.output or ROOT / "runs" / datetime.now().strftime(f"api_{config['provider']}_%Y%m%d_%H%M%S_%f")).resolve()
    if args.resume:
        if not directory.is_dir():
            raise ExperimentError("재개할 결과 폴더가 없습니다.")
        prepare_protocol(directory, config, True)  # Read-only frozen-code validation before auth.
    elif directory.exists():
        raise ExperimentError("새 결과 폴더가 이미 존재합니다. --resume 또는 새 경로를 사용하세요.")
    if check_only and args.dry_run:
        raise ExperimentError("--check-only와 --dry-run은 함께 사용할 수 없습니다.")
    if not args.dry_run:
        client = (client_factory or make_client)(config)
    try:
        if check_only:
            return 0
        if not args.resume:
            directory.mkdir(parents=True, exist_ok=False)
        with run_lock(directory):
            prepare_protocol(directory, config, bool(args.resume))
            if args.dry_run:
                history = [{"role": "user", "content": task_prompt(config["feedback_policy"])["user"]},
                           submission_message(problems[0].rules[0], 1, 1)]
                save_json(directory / "first_request_preview.json", make_request(history, config))
                publish_state(directory, "preview", call=0, message="요청 미리보기입니다. 실제 API 호출은 없습니다.")
                refresh_reports(directory)
                print(f"Dry run 완료: API 호출 0회, 결과 점수 없음.\n{directory}")
                return 0
            save_json(directory / "runtime.json", {
                "python": sys.version, "provider": config["provider"],
                "sdk_version": importlib.metadata.version("anthropic" if config["provider"] == "anthropic" else "openai"),
                "context_format": config["context_format"], "started_at": utc_now(),
            })
            if not args.preserve_stop_request:
                (directory / "stop.request").unlink(missing_ok=True)
            publish_state(directory, "running", message="저장된 기록을 확인하고 실험을 시작합니다.")
            append_event(directory, "run_resumed" if (directory / "trials").exists() else "run_started", config=config)
            try:
                result = execute(directory, config, client, args.retry_pending)
                append_event(directory, "run_completed" if result["complete"] else "run_finished_with_failures",
                             recorded_submissions=result["recorded_submissions"], generation_failure_count=result["generation_failure_count"])
            except ExperimentStopped as exc:
                publish_state(directory, "stopped", message=str(exc))
                append_event(directory, "run_stopped", message=str(exc))
                refresh_reports(directory)
                print(str(exc), flush=True)
                return 0
            except (ExperimentError, KeyboardInterrupt) as exc:
                save_json(directory / "last_stop.json", {"at": utc_now(), "type": type(exc).__name__,
                                                          "message": str(exc)})
                publish_state(directory, "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                              message=str(exc) or "실행이 중단되었습니다.")
                append_event(directory, "run_error", error_type=type(exc).__name__, message=str(exc))
                refresh_reports(directory)
                raise
        return 0
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ExperimentError, OSError) as exc:
        print(f"중단: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("사용자 중단. 저장된 응답은 --resume으로 재사용할 수 있습니다.", file=sys.stderr)
        sys.exit(130)
