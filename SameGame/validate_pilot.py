"""Independently check the two pilot JSON files. Run: python3 validate_pilot.py [directory]."""

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


def letter_count(tape):
    return len(tape) - tape.count(" ")


@lru_cache(maxsize=32768)
def moves(tape):
    """Each choice deletes its whole leftmost eligible run, preserving literal spaces."""
    choices = {}
    for run in RUN.finditer(tape):
        letter = run.group(1)
        if letter not in choices:
            choices[letter] = tape[:run.start()] + tape[run.end():]
    return choices


def solution_statistics(start, target, priority, next_priority):
    """Count successful action strings and their extreme simultaneous-priority matches."""
    @lru_cache(maxsize=None)
    def visit(tape):
        remaining = letter_count(tape)
        if remaining == target:
            return 1, 0, 0
        if remaining < target:
            return 0, 0, 0
        choices = moves(tape)
        total = 0
        low = high = 0
        for action in ((priority,) if priority in choices else choices):
            count, child_low, child_high = visit(choices[action])
            if count:
                hit = int(action == priority and next_priority in choices)
                low = min(low, child_low + hit) if total else child_low + hit
                high = max(high, child_high + hit)
                total += count
        return total, low, high

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


def replay(tape, target, priority, answer):
    """Replay physical moves, keeping priority violations observable after the first."""
    require(isinstance(answer, str), "answer must be a string")
    states = [tape]
    violations = []
    for step, action in enumerate(answer, 1):
        if letter_count(tape) == target:
            return Replay(states, violations, "action_after_target", step)
        if action not in LETTERS:
            return Replay(states, violations, "invalid_action_symbol", step)
        choices = moves(tape)
        if action not in choices:
            return Replay(states, violations, "no_removable_run", step)
        if priority in choices and action != priority:
            violations.append(step)
        tape = choices[action]
        states.append(tape)
    reason = "target_reached" if letter_count(tape) == target else "target_not_reached"
    return Replay(states, violations, reason, len(answer))


def old_priority_matches(result, answer, old, new):
    return [step for step, (tape, action) in enumerate(zip(result.states[:-1], answer), 1)
            if action == old and new in moves(tape)]


def validate_space_probe(probe, tape, target, rounds):
    if " " not in tape:
        expect(probe, None, "space_probe")
        return
    require(isinstance(probe, dict), "a spaced input needs a space_probe")
    number = probe["round"]
    require(type(number) is int and 1 <= number <= len(rounds), "invalid space_probe.round")
    record = rounds[number - 1]
    answer = probe["one_valid_answer"]
    expect(answer, record["one_valid_answer"], "space_probe.one_valid_answer")
    priority = record["priority"]
    with_space = replay(tape, target, priority, answer)
    require(with_space.valid, f"space_probe answer fails with the boundary: {with_space.reason}")
    joined = replay(tape.replace(" ", ""), target, priority, answer)
    require(not joined.valid, "space_probe answer also succeeds without the boundary")
    differences = [step for step, (original, other) in
                   enumerate(zip(with_space.states[1:], joined.states[1:]), 1)
                   if original.replace(" ", "") != other]
    differences.extend(joined.priority_violations)
    if joined.reason in {"no_removable_run", "action_after_target"}:
        differences.append(joined.stop_step)
    require(differences, "space_probe never diverges during the answer")
    expect(probe["first_different_step"], min(differences), "space_probe.first_different_step")


def validate_item(public, key):
    tape = public["input"]
    target = public["target_length"]
    require(isinstance(tape, str) and TAPE.fullmatch(tape),
            "input must contain A–E and optionally one interior ASCII space")
    require(28 <= letter_count(tape) <= 32, "input must contain 28–32 letters")
    require(type(target) is int and target in range(4), "target_length must be 0, 1, 2 or 3")
    rounds = key["rounds"]
    transitions = key["transitions"]
    expect(len(rounds), 5, "round count")
    expect(len(transitions), 4, "transition count")
    priorities = [r["priority"] for r in rounds]
    expect(sorted(priorities), list(LETTERS), "priorities across the five rounds")
    for index, record in enumerate(rounds):
        number = index + 1
        tag = f"R{number}"
        old = priorities[index]
        new = priorities[number] if number < len(rounds) else None
        expect(record["round"], number, f"{tag}.round")
        count, low, high = solution_statistics(tape, target, old, new)
        require(count > 0, f"{tag}: no valid answers")
        expect(record["n_valid_answers"], count, f"{tag}.n_valid_answers")
        answer = record["one_valid_answer"]
        result = replay(tape, target, old, answer)
        require(result.valid, f"{tag}.one_valid_answer: {result.reason}, priority violations {result.priority_violations}")
        expect(record["final_state"], result.states[-1], f"{tag}.final_state")
        if new is not None:
            transition = transitions[index]
            hits = old_priority_matches(result, answer, old, new)
            for field, expected in (
                ("from_round", number), ("to_round", number + 1),
                ("old_priority", old), ("new_priority", new),
                ("old_priority_matches_on_replayed_reference", hits),
                ("min_over_all_previous_solutions", low), ("max_over_all_previous_solutions", high),
            ):
                expect(transition[field], expected, f"{tag}→R{number + 1}.{field}")
            require(low >= 2, f"{tag}→R{number + 1}: a previous solution has only {low} matches")
    validate_space_probe(key["space_probe"], tape, target, rounds)
    return priorities


def load_pilot(directory):
    public = json.loads((directory / "problems.json").read_text(encoding="utf-8"))
    key = json.loads((directory / "answer_key.json").read_text(encoding="utf-8"))
    return public, key


def validate(directory):
    public, key = load_pilot(directory)
    items = public["items"]
    keys = {k["id"]: k for k in key["items"]}
    expect(len(items), 50, "public items")
    expect(len(key["items"]), 50, "answer-key items")
    ids = {p["id"] for p in items}
    require(len(ids) == 50 and ids == keys.keys(), "public and answer-key IDs must match uniquely")
    targets = Counter()
    spaces = Counter()
    positions = [Counter() for _ in range(5)]
    pairs = Counter()
    ab_a_examples = []
    for item in items:
        try:
            order = validate_item(item, keys[item["id"]])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f"{item['id']}: {error}") from error
        targets[item["target_length"]] += 1
        spaces[item["input"].count(" ")] += 1
        for counts, priority in zip(positions, order):
            counts[priority] += 1
        ab_a_examples.extend([item["id"], r["round"]] for r in keys[item["id"]]["rounds"]
                             if r["final_state"] == "AB A")
        pairs.update(a + b for a, b in zip(order, order[1:]))
        moves.cache_clear()
    expect(dict(targets), {0: 13, 1: 13, 2: 12, 3: 12}, "target-length distribution")
    expect(dict(spaces), {0: 30, 1: 20}, "space distribution")
    for number, counts in enumerate(positions, 1):
        expect(dict(counts), dict.fromkeys(LETTERS, 10), f"R{number} priorities")
    expect(dict(pairs), {"".join(p): 10 for p in permutations(LETTERS, 2)}, "transition pairs")
    require(ab_a_examples, "no saved reference answer ends with the three-letter state 'AB A'")
    return {
        "items": len(items), "rounds": 5 * len(items), "transitions": sum(pairs.values()),
        "target_length_counts": dict(sorted(targets.items())), "space_counts": dict(sorted(spaces.items())),
        "priority_counts_by_round": [dict(sorted(p.items())) for p in positions],
        "priority_pair_counts": dict(sorted(pairs.items())), "literal_AB_A_examples": ab_a_examples,
    }


def grade(directory, item_id, round_number, answer):
    public, key = load_pilot(directory)
    number = int(round_number)
    require(1 <= number <= 5, "round must be 1 through 5")
    items = {p["id"]: p for p in public["items"]}
    records = {k["id"]: k["rounds"] for k in key["items"]}
    require(item_id in items and item_id in records, f"unknown item {item_id!r}")
    item = items[item_id]
    priorities = [r["priority"] for r in records[item_id]]
    current = priorities[number - 1]
    previous = priorities[number - 2] if number > 1 else None
    result = replay(item["input"], item["target_length"], current, answer)
    return {
        "id": item_id, "round": number, "priority": current, "valid": result.valid,
        "final_state": result.states[-1], "remaining_length": letter_count(result.states[-1]),
        "reason": "priority_violation" if result.reason == "target_reached" and result.priority_violations else result.reason,
        "stop_reason": result.reason, "stop_step": result.stop_step,
        "executed_steps": len(result.states) - 1,
        "priority_violation_steps": result.priority_violations,
        "old_priority_match_steps": old_priority_matches(result, answer, previous, current),
    }


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
