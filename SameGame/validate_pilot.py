"""Independently validate a pilot directory or grade one submitted action string."""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path


LETTERS = "ABCDE"
RUN = re.compile(r"([A-E])\1+")
TAPE = re.compile(r"[A-E]+(?: [A-E]+)?")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expect(actual, expected, field):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{field}: expected {expected!r}, found {actual!r}")


@dataclass(frozen=True)
class Setting:
    rule: str
    target: int
    priority: str


def letter_count(tape):
    return len(tape) - tape.count(" ")


def terminal_reason(tape, target, legacy):
    letters = tape.replace(" ", "")
    if len(letters) != target:
        return None
    if not legacy and target == 2 and letters[0] == letters[1]:
        return "duplicate_final_pair"
    return "target_reached"


@lru_cache(maxsize=32768)
def moves(tape, rule):
    """Map each letter to (result, original run length) for its leftmost eligible run."""
    choices = {}
    for run in RUN.finditer(tape):
        letter = run.group(1)
        if letter not in choices:
            stop = run.end() if rule == "whole" else run.start() + 2
            choices[letter] = (tape[:run.start()] + tape[stop:], run.end() - run.start())
    return choices


def transition_kind(old, new):
    changes = [kind for kind, left, right in (("rule", old.rule, new.rule),
               ("condition", old.priority, new.priority), ("goal", old.target, new.target))
               if left != right]
    require(len(changes) == 1, f"adjacent rounds must change exactly one setting: {old} -> {new}")
    return changes[0]


@dataclass(frozen=True)
class Statistics:
    count: int
    min_hits: int
    max_hits: int
    min_steps: int
    max_steps: int


def solution_statistics(start, setting, following, legacy):
    """All successful paths stop at the first target decision, including duplicate-pair failure."""
    kind = transition_kind(setting, following) if following else None
    empty = Statistics(0, 0, 0, 0, 0)
    success = Statistics(1, 0, 0, 0, 0)

    @lru_cache(maxsize=None)
    def visit(tape):
        terminal = terminal_reason(tape, setting.target, legacy)
        if terminal == "target_reached":
            return success
        if terminal or letter_count(tape) < setting.target + 2:
            return empty
        choices = moves(tape, setting.rule)
        actions = (setting.priority,) if setting.priority in choices else choices
        branches = []
        for action in actions:
            next_tape, run_length = choices[action]
            child = visit(next_tape)
            if child.count:
                if kind == "condition":
                    hit = int(action == setting.priority and following.priority in choices)
                elif kind == "rule":
                    hit = int(run_length >= 3)
                else:
                    hit = 0
                branches.append((child, hit))
        if not branches:
            return empty
        return Statistics(sum(c.count for c, _ in branches),
                          min(c.min_hits + h for c, h in branches),
                          max(c.max_hits + h for c, h in branches),
                          1 + min(c.min_steps for c, _ in branches),
                          1 + max(c.max_steps for c, _ in branches))

    result = visit(start)
    visit.cache_clear()
    return result


@dataclass
class Replay:
    states: list
    priority_violations: list
    reason: str
    stop_step: int

    @property
    def valid(self):
        return self.reason == "target_reached" and not self.priority_violations


def replay(tape, setting, answer, legacy=False):
    """An extra-rule violation is recorded; a physical failure or terminal state stops the trace."""
    require(isinstance(answer, str), "answer must be a string")
    states = [tape]
    violations = []
    for step, action in enumerate(answer, 1):
        terminal = terminal_reason(tape, setting.target, legacy)
        if terminal == "duplicate_final_pair":
            return Replay(states, violations, terminal, step - 1)
        if terminal:
            return Replay(states, violations, "action_after_target", step)
        if action not in LETTERS:
            return Replay(states, violations, "invalid_action_symbol", step)
        choices = moves(tape, setting.rule)
        if action not in choices:
            return Replay(states, violations, "no_removable_run", step)
        if setting.priority in choices and action != setting.priority:
            violations.append(step)
        tape = choices[action][0]
        states.append(tape)
    reason = terminal_reason(tape, setting.target, legacy) or "target_not_reached"
    return Replay(states, violations, reason, len(answer))


def old_priority_matches(result, answer, old, new):
    return [step for step, (tape, action) in enumerate(zip(result.states[:-1], answer), 1)
            if action == old.priority and new.priority in moves(tape, old.rule)]


def effect_steps(result, answer, old):
    return [step for step, (tape, action) in enumerate(zip(result.states[:-1], answer), 1)
            if moves(tape, old.rule)[action][1] >= 3]


def validate_space_probe(probe, tape, settings, rounds, legacy):
    if " " not in tape:
        expect(probe, None, "space_probe")
        return
    require(isinstance(probe, dict), "a spaced input needs a space_probe")
    number = probe["round"]
    require(type(number) is int and 1 <= number <= len(rounds), "invalid space_probe.round")
    setting = settings[number - 1]
    answer = probe["one_valid_answer"]
    expect(answer, rounds[number - 1]["one_valid_answer"], "space_probe.one_valid_answer")
    expect(probe["priority"], setting.priority, "space_probe.priority")
    with_space = replay(tape, setting, answer, legacy)
    require(with_space.valid, f"space_probe answer fails with the boundary: {with_space.reason}")
    joined = replay(tape.replace(" ", ""), setting, answer, legacy)
    require(not joined.valid, "space_probe answer also succeeds without the boundary")
    differences = [step for step, (original, other) in
                   enumerate(zip(with_space.states[1:], joined.states[1:]), 1)
                   if original.replace(" ", "") != other]
    differences.extend(joined.priority_violations)
    if joined.reason in {"no_removable_run", "action_after_target", "duplicate_final_pair"}:
        differences.append(joined.stop_step)
    require(differences, "space_probe never diverges during the answer")
    expect(probe["first_different_step"], min(differences), "space_probe.first_different_step")


def settings_for(public, key, legacy):
    base = public["target_length"]
    require(type(base) is int and base in range(4), "target_length must be 0, 1, 2 or 3")
    rounds = key["rounds"]
    expect(len(rounds), 5, "round count")
    if not legacy:
        expect(len(public["round_settings"]), 5, "round_settings count")
    settings = []
    for i, r in enumerate(rounds, 1):
        expect(r["round"], i, f"R{i}.round")
        require(r["priority"] in tuple(LETTERS), f"R{i}: unknown priority")
        if legacy:
            settings.append(Setting("whole", base, r["priority"]))
            continue
        advertised = public["round_settings"][i - 1]
        for field in ("round", "target_length", "deletion_rule"):
            expect(advertised[field], r[field], f"R{i}.public.{field}")
        require(type(r["target_length"]) is int and r["target_length"] in range(4),
                f"R{i}: invalid target_length")
        require(r["deletion_rule"] in ("whole", "pair"), f"R{i}: invalid deletion_rule")
        settings.append(Setting(r["deletion_rule"], r["target_length"], r["priority"]))
    if legacy:
        expect(sorted(s.priority for s in settings), list(LETTERS), "round priorities")
    else:
        forward = settings if settings[0].target == base else settings[::-1]
        p, q = forward[0].priority, forward[2].priority
        require(p != q, "condition transition must change priority")
        expect(forward, [Setting("whole", base, p), Setting("pair", base, p),
                         Setting("pair", base, q), Setting("whole", base, q),
                         Setting("whole", (base + 1) % 4, q)], "five-round schedule")
    return settings


def validate_transition(tape, rounds, settings, traces, stats, index, record, legacy):
    number = index + 1
    old, new = settings[index:index + 2]
    old_answer = rounds[index]["one_valid_answer"]
    trace = traces[index]
    kind = transition_kind(old, new)
    fields = {"from_round": number, "to_round": number + 1}
    if not legacy:
        replayed = replay(tape, new, old_answer, legacy)
        require(not replayed.valid, f"R{number} answer still succeeds under R{number + 1}")
        fields.update(kind=kind, reference_valid_under_new=False)
    if kind == "condition":
        hits = old_priority_matches(trace, old_answer, old, new)
        fields.update(old_priority=old.priority, new_priority=new.priority,
                      old_priority_matches_on_replayed_reference=hits,
                      min_over_all_previous_solutions=stats[index].min_hits,
                      max_over_all_previous_solutions=stats[index].max_hits)
        require(stats[index].min_hits >= 2, f"R{number}: a previous solution has fewer than two priority matches")
    elif kind == "rule":
        whole, pair = (stats[index:index + 2] if old.rule == "whole"
                       else (stats[index + 1], stats[index]))
        require(pair.min_steps == pair.max_steps and whole.max_steps < pair.min_steps,
                f"R{number}: whole and pair successful lengths must be disjoint")
        require(stats[index].min_hits >= 2,
                f"R{number}: a previous solution has fewer than two rule-effect opportunities")
        fields.update(old_rule=old.rule, new_rule=new.rule,
                      effect_steps_on_old_reference=effect_steps(trace, old_answer, old),
                      min_effect_steps_over_old_solutions=stats[index].min_hits,
                      max_whole_solution_steps=whole.max_steps, pair_solution_steps=pair.min_steps)
    else:
        fields.update(old_target_length=old.target, new_target_length=new.target)
    for field, expected in fields.items():
        expect(record[field], expected, f"R{number}->R{number + 1}.{field}")
    return kind


def validate_item(public, key, legacy):
    tape = public["input"]
    require(isinstance(tape, str) and TAPE.fullmatch(tape),
            "input must contain A–E and optionally one interior ASCII space")
    require(28 <= letter_count(tape) <= 32, "input must contain 28–32 letters")
    rounds = key["rounds"]
    settings = settings_for(public, key, legacy)
    expect(len(key["transitions"]), 4, "transition count")
    stats, traces = [], []
    for i, (record, setting) in enumerate(zip(rounds, settings)):
        following = settings[i + 1] if i + 1 < len(settings) else None
        s = solution_statistics(tape, setting, following, legacy)
        require(s.count > 0, f"R{i + 1}: no valid answers")
        expect(record["n_valid_answers"], s.count, f"R{i + 1}.n_valid_answers")
        r = replay(tape, setting, record["one_valid_answer"], legacy)
        require(r.valid, f"R{i + 1} reference: {r.reason}, priority violations {r.priority_violations}")
        expect(record["final_state"], r.states[-1], f"R{i + 1}.final_state")
        stats.append(s)
        traces.append(r)
    kinds = [validate_transition(tape, rounds, settings, traces, stats, i, entry, legacy)
             for i, entry in enumerate(key["transitions"])]
    validate_space_probe(key["space_probe"], tape, settings, rounds, legacy)
    return settings, kinds


def load_pilot(directory):
    public = json.loads((directory / "problems.json").read_text(encoding="utf-8"))
    key = json.loads((directory / "answer_key.json").read_text(encoding="utf-8"))
    expect(public["version"], key["version"], "public and key version")
    require(public["version"] in ("pilot-2", "pilot-3"), f"unsupported version {public['version']!r}")
    return public, key, public["version"] == "pilot-2"


def validate(directory):
    public, key, legacy = load_pilot(directory)
    items = public["items"]
    keys = {k["id"]: k for k in key["items"]}
    expect(len(items), 50, "public items")
    expect(len(key["items"]), 50, "answer-key items")
    ids = {p["id"] for p in items}
    require(len(ids) == 50 and ids == keys.keys(), "public and answer-key IDs must match uniquely")
    targets, spaces, kinds, pairs = Counter(), Counter(), Counter(), Counter()
    positions = [Counter() for _ in range(5)]
    ab_a_examples = []
    for item in items:
        try:
            settings, changes = validate_item(item, keys[item["id"]], legacy)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f"{item['id']}: {error}") from error
        targets[item["target_length"]] += 1
        spaces[item["input"].count(" ")] += 1
        kinds.update(changes)
        for counts, setting in zip(positions, settings):
            counts[setting.priority] += 1
        pairs.update(x.priority + y.priority for x, y, c in zip(settings, settings[1:], changes)
                     if c == "condition")
        ab_a_examples.extend([item["id"], r["round"]] for r in keys[item["id"]]["rounds"]
                             if r["final_state"] == "AB A")
        moves.cache_clear()
    expect(dict(targets), {0: 13, 1: 13, 2: 12, 3: 12}, "base target-length distribution")
    expect(dict(spaces), {0: 30, 1: 20}, "space distribution")
    for number, counts in enumerate(positions, 1):
        expect(dict(counts), dict.fromkeys(LETTERS, 10), f"R{number} priorities")
    if legacy:
        expect(dict(pairs), {"".join(p): 10 for p in permutations(LETTERS, 2)}, "transition pairs")
    else:
        expect(dict(kinds), {"rule": 100, "condition": 50, "goal": 50}, "transition kinds")
        require(len(pairs) == 20 and all(count in (2, 3) for count in pairs.values()),
                "each ordered priority switch must appear two or three times")
    require(ab_a_examples, "no saved reference answer ends with the three-letter state 'AB A'")
    return {
        "version": public["version"], "items": len(items), "rounds": 5 * len(items),
        "transitions": sum(kinds.values()), "transition_kinds": dict(sorted(kinds.items())),
        "target_length_counts": dict(sorted(targets.items())), "space_counts": dict(sorted(spaces.items())),
        "priority_counts_by_round": [dict(sorted(p.items())) for p in positions],
        "priority_pair_counts": dict(sorted(pairs.items())), "literal_AB_A_examples": ab_a_examples,
    }


def grade(directory, item_id, round_number, answer):
    public, key, legacy = load_pilot(directory)
    number = int(round_number)
    require(1 <= number <= 5, "round must be 1 through 5")
    items = {p["id"]: p for p in public["items"]}
    records = {k["id"]: k for k in key["items"]}
    require(item_id in items and item_id in records, f"unknown item {item_id!r}")
    item = items[item_id]
    settings = settings_for(item, records[item_id], legacy)
    current = settings[number - 1]
    tape = item["input"]
    result = replay(tape, current, answer, legacy)
    report = {
        "id": item_id, "round": number, "priority": current.priority, "target_length": current.target,
        "deletion_rule": current.rule, "valid": result.valid, "final_state": result.states[-1],
        "remaining_length": letter_count(result.states[-1]),
        "reason": "priority_violation" if result.reason == "target_reached" and result.priority_violations else result.reason,
        "stop_reason": result.reason, "stop_step": result.stop_step, "executed_steps": len(result.states) - 1,
        "priority_violation_steps": result.priority_violations,
    }
    if number > 1:
        previous = settings[number - 2]
        kind = transition_kind(previous, current)
        old_run = replay(tape, previous, answer, legacy)
        report.update(transition_kind=kind, valid_under_previous_round=old_run.valid,
                      previous_only=old_run.valid and not result.valid)
        if kind == "condition":
            report["old_priority_match_steps"] = old_priority_matches(result, answer, previous, current)
        elif kind == "rule":
            report["effect_steps_when_replayed_under_previous_rule"] = effect_steps(old_run, answer, previous)
    elif legacy:
        report["old_priority_match_steps"] = []
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent,
                        help="directory with problems.json and answer_key.json (default: beside this script)")
    parser.add_argument("--grade", nargs=3, metavar=("ID", "ROUND", "ACTIONS"),
                        help="grade one submitted A–E action string (one-based round and action indices)")
    args = parser.parse_args()
    try:
        summary = grade(args.directory, *args.grade) if args.grade else validate(args.directory)
    except (OSError, KeyError, TypeError, IndexError, ValueError) as error:
        parser.exit(1, f"Validation failed: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
