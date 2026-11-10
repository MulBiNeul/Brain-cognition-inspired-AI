"""Independently validate a pilot directory or grade one submitted action string."""

import argparse
import hashlib
import importlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations
from pathlib import Path


LETTERS = "ABCDE"
# SHA-256 of "id:priorities\n" in ID order in the released v3 key (fc1d2ac).
V3_PRIORITY_SCHEDULE_DIGEST = "0349ab0bab4cd567e5e96a88bd641fe970f1c8ea4fed9aa5a0ad16f69029332c"
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
    side: str = "left"


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
def moves(tape, rule, side="left"):
    """Map a letter to the result and length of the first or last eligible run."""
    choices = {}
    for run in RUN.finditer(tape):
        letter = run.group(1)
        if side == "right" or letter not in choices:
            stop = run.end() if rule == "whole" else run.start() + 2
            choices[letter] = (tape[:run.start()] + tape[stop:], run.end() - run.start())
    return choices


def transition_kind(old, new):
    changes = [kind for kind, left, right in (("rule", old.rule, new.rule),
               ("condition", old.priority, new.priority), ("side", old.side, new.side),
               ("goal", old.target, new.target))
               if left != right]
    if changes == ["condition", "goal"]:
        return "goal_and_condition"
    if changes == ["side", "goal"]:
        return "goal_and_rule"
    require(len(changes) == 1, f"unsupported changes in adjacent settings: {old} -> {new}")
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
        choices = moves(tape, setting.rule, setting.side)
        actions = (setting.priority,) if setting.priority in choices else choices
        branches = []
        for action in actions:
            next_tape, run_length = choices[action]
            child = visit(next_tape)
            if child.count:
                if kind in ("condition", "goal_and_condition"):
                    hit = int(action == setting.priority and following.priority in choices)
                elif kind == "rule":
                    hit = int(run_length >= 3)
                elif kind == "goal_and_rule":
                    hit = int(next_tape != moves(tape, following.rule, following.side)[action][0])
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
        choices = moves(tape, setting.rule, setting.side)
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
            if action == old.priority and new.priority in moves(tape, old.rule, old.side)]


def effect_steps(result, answer, old):
    return [step for step, (tape, action) in enumerate(zip(result.states[:-1], answer), 1)
            if moves(tape, old.rule, old.side)[action][1] >= 3]


def side_opportunities(result, answer, rule):
    """Compare both next states at each executed action of this particular trace."""
    return [step for step, (tape, action) in enumerate(zip(result.states[:-1], answer), 1)
            if moves(tape, rule, "left")[action][0] != moves(tape, rule, "right")[action][0]]


def joint_counterfactuals(tape, answer, old, new, kind, legacy):
    old_goal = replace(new, target=old.target)
    if kind == "goal_and_condition":
        old_policy = replace(new, priority=old.priority)
        policy_field = "valid_if_only_priority_reverted"
    else:
        old_policy = replace(new, side=old.side)
        policy_field = "valid_if_only_selection_side_reverted"
    return {
        "valid_if_only_goal_reverted": replay(tape, old_goal, answer, legacy).valid,
        policy_field: replay(tape, old_policy, answer, legacy).valid,
    }


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
    if joined.priority_violations:
        step = joined.priority_violations[0]
        reason, state = "priority_violation", joined.states[step - 1]
    else:
        step, state = joined.stop_step, joined.states[-1]
        reason = {"action_after_target": "continued_after_goal",
                  "target_not_reached": "wrong_final_length"}.get(joined.reason, joined.reason)
    expect(probe["failure_step_without_space"], step, "space_probe.failure_step_without_space")
    expect(probe["failure_without_space"], reason, "space_probe.failure_without_space")
    expect(probe["state_without_space_at_failure"], state, "space_probe.state_without_space_at_failure")


def settings_for(public, key, version):
    legacy = version == "pilot-2"
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
        public_fields = ["round", "target_length", "deletion_rule"]
        if version == "pilot-4":
            public_fields.extend(["priority", "selection_side"])
        for field in public_fields:
            expect(advertised[field], r[field], f"R{i}.public.{field}")
        require(type(r["target_length"]) is int and r["target_length"] in range(4),
                f"R{i}: invalid target_length")
        require(r["deletion_rule"] in ("whole", "pair"), f"R{i}: invalid deletion_rule")
        side = r["selection_side"] if version == "pilot-4" else "left"
        require(side in ("left", "right"), f"R{i}: invalid selection_side")
        settings.append(Setting(r["deletion_rule"], r["target_length"], r["priority"], side))
    if legacy:
        expect(sorted(s.priority for s in settings), list(LETTERS), "round priorities")
    elif version == "pilot-3":
        forward = settings if settings[0].target == base else settings[::-1]
        p, q = forward[0].priority, forward[2].priority
        require(p != q, "condition transition must change priority")
        expect(forward, [Setting("whole", base, p), Setting("pair", base, p),
                         Setting("pair", base, q), Setting("whole", base, q),
                         Setting("whole", (base + 1) % 4, q)], "five-round schedule")
    else:
        expect(base, settings[0].target, "target_length must advertise the first round goal")
        require(all(s.rule == "whole" for s in settings), "v4 requires whole-run deletion in every round")
        require({s.target for s in settings} == set(range(4)), "item must cover all four target lengths")
        require(all(a.target != b.target for a, b in zip(settings, settings[1:])),
                "adjacent rounds must have different target lengths")
        changes = [transition_kind(a, b) for a, b in zip(settings, settings[1:])]
        expect(dict(Counter(changes)), {"goal": 2, "goal_and_rule": 1, "goal_and_condition": 1},
               "per-item transition kinds")
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
    if kind in ("goal", "goal_and_condition", "goal_and_rule"):
        fields.update(old_target_length=old.target, new_target_length=new.target)
    if kind in ("condition", "goal_and_condition"):
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
    elif kind == "goal_and_rule":
        require(stats[index].min_hits >= 2, f"R{number}: a previous solution has fewer than two side opportunities")
        fields.update(old_selection_side=old.side, new_selection_side=new.side,
                      effect_steps_on_old_reference=side_opportunities(trace, old_answer, old.rule),
                      min_effect_steps_over_old_solutions=stats[index].min_hits,
                      max_effect_steps_over_old_solutions=stats[index].max_hits)
    for field, expected in fields.items():
        expect(record[field], expected, f"R{number}->R{number + 1}.{field}")
    if kind in ("goal_and_condition", "goal_and_rule"):
        answer = record["carryover_probe"]["answer"]
        probe_replayed = replay(tape, new, answer, legacy)
        require(not probe_replayed.valid and not replay(tape, old, answer, legacy).valid,
                f"R{number}: carryover probe must fail both the full previous and full current settings")
        for label, actions, current_trace in (("previous reference", old_answer, replayed),
                                              ("carryover probe", answer, probe_replayed)):
            if kind == "goal_and_condition":
                positions = old_priority_matches(current_trace, actions, old, new)
            else:
                positions = side_opportunities(current_trace, actions, new.rule)
            require(len(positions) >= 2, f"R{number}->R{number + 1}: {label} has fewer than two "
                    f"diagnostic steps in the current-round replay (found {len(positions)})")
        mixed = joint_counterfactuals(tape, answer, old, new, kind, legacy)
        policy_field = ("valid_if_only_priority_reverted" if kind == "goal_and_condition"
                        else "valid_if_only_selection_side_reverted")
        expect(mixed, {"valid_if_only_goal_reverted": False, policy_field: True},
               f"R{number}->R{number + 1}.carryover_probe")
    return kind


def validate_item(public, key, version):
    legacy = version == "pilot-2"
    tape = public["input"]
    require(isinstance(tape, str) and TAPE.fullmatch(tape),
            "input must contain A–E and optionally one interior ASCII space")
    require(28 <= letter_count(tape) <= 32, "input must contain 28–32 letters")
    rounds = key["rounds"]
    settings = settings_for(public, key, version)
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
    require(public["version"] in ("pilot-2", "pilot-3", "pilot-4", "pilot-5"),
            f"unsupported version {public['version']!r}")
    return public, key, public["version"] == "pilot-2"


def merge_validator():
    # Older versions can still be checked with this script alone.
    return importlib.import_module(".validate_merge", __package__) if __package__ else importlib.import_module("validate_merge")


def validate(directory):
    public, key, legacy = load_pilot(directory)
    version = public["version"]
    if version == "pilot-5":
        return merge_validator().validate(public, key)
    items = public["items"]
    keys = {k["id"]: k for k in key["items"]}
    expect(len(items), 50, "public items")
    expect(len(key["items"]), 50, "answer-key items")
    ids = {p["id"] for p in items}
    require(len(ids) == 50 and ids == keys.keys(), "public and answer-key IDs must match uniquely")
    targets, spaces, kinds, pairs = Counter(), Counter(), Counter(), Counter()
    positions = [Counter() for _ in range(5)]
    initial_sides = Counter()
    targets_by_round = [Counter() for _ in range(5)]
    all_targets = Counter()
    ab_a_examples = []
    for item in items:
        try:
            settings, changes = validate_item(item, keys[item["id"]], version)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f"{item['id']}: {error}") from error
        targets[item["target_length"]] += 1
        spaces[item["input"].count(" ")] += 1
        initial_sides[settings[0].side] += 1
        kinds.update(changes)
        for i, setting in enumerate(settings):
            positions[i][setting.priority] += 1
            targets_by_round[i][setting.target] += 1
            all_targets[setting.target] += 1
        pairs.update(x.priority + y.priority for x, y, c in zip(settings, settings[1:], changes)
                     if c in ("condition", "goal_and_condition"))
        ab_a_examples.extend([item["id"], r["round"]] for r in keys[item["id"]]["rounds"]
                             if r["final_state"] == "AB A")
        moves.cache_clear()
    if version == "pilot-4":
        expect(dict(initial_sides), {"left": 25, "right": 25}, "starting selection side")
        for i, counts in enumerate(targets_by_round, 1):
            require(all(counts[t] in (12, 13) for t in range(4)), f"R{i}: unbalanced target lengths")
        require(all(all_targets[t] in (62, 63) for t in range(4)), "unbalanced overall target lengths")
        schedule = "".join(item_id + ":" + "".join(r["priority"] for r in keys[item_id]["rounds"])
                           + "\n" for item_id in sorted(keys))
        expect(hashlib.sha256(schedule.encode("utf-8")).hexdigest(),
               V3_PRIORITY_SCHEDULE_DIGEST, "exact v3 priority schedule")
    else:
        expect(dict(targets), {0: 13, 1: 13, 2: 12, 3: 12}, "base target-length distribution")
    expect(dict(spaces), {0: 30, 1: 20}, "space distribution")
    for number, counts in enumerate(positions, 1):
        expect(dict(counts), dict.fromkeys(LETTERS, 10), f"R{number} priorities")
    if legacy:
        expect(dict(pairs), {"".join(p): 10 for p in permutations(LETTERS, 2)}, "transition pairs")
    else:
        expected_kinds = ({"goal": 100, "goal_and_condition": 50, "goal_and_rule": 50} if version == "pilot-4"
                          else {"rule": 100, "condition": 50, "goal": 50})
        expect(dict(kinds), expected_kinds, "transition kinds")
        require(len(pairs) == 20 and all(count in (2, 3) for count in pairs.values()),
                "each ordered priority switch must appear two or three times")
    require(ab_a_examples, "no saved reference answer ends with the three-letter state 'AB A'")
    return {
        "version": public["version"], "items": len(items), "rounds": 5 * len(items),
        "transitions": sum(kinds.values()), "transition_kinds": dict(sorted(kinds.items())),
        "carryover_probes_validated": kinds["goal_and_condition"] + kinds["goal_and_rule"],
        "target_length_counts": dict(sorted(targets.items())), "space_counts": dict(sorted(spaces.items())),
        "all_round_target_length_counts": dict(sorted(all_targets.items())),
        "target_length_counts_by_round": [dict(sorted(t.items())) for t in targets_by_round],
        "initial_selection_side_counts": dict(sorted(initial_sides.items())),
        "priority_counts_by_round": [dict(sorted(p.items())) for p in positions],
        "priority_pair_counts": dict(sorted(pairs.items())), "literal_AB_A_examples": ab_a_examples,
    }


def grade(directory, item_id, round_number, answer):
    public, key, legacy = load_pilot(directory)
    number = int(round_number)
    if public["version"] == "pilot-5":
        return merge_validator().grade(public, item_id, number, answer)
    require(1 <= number <= 5, "round must be 1 through 5")
    items = {p["id"]: p for p in public["items"]}
    records = {k["id"]: k for k in key["items"]}
    require(item_id in items and item_id in records, f"unknown item {item_id!r}")
    item = items[item_id]
    settings = settings_for(item, records[item_id], public["version"])
    current = settings[number - 1]
    tape = item["input"]
    result = replay(tape, current, answer, legacy)
    report = {
        "id": item_id, "round": number, "priority": current.priority, "target_length": current.target,
        "deletion_rule": current.rule, "selection_side": current.side,
        "valid": result.valid, "final_state": result.states[-1],
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
        if kind in ("goal_and_condition", "goal_and_rule"):
            report.update(joint_counterfactuals(tape, answer, previous, current, kind, legacy))
        if kind in ("condition", "goal_and_condition"):
            report["old_priority_match_steps"] = old_priority_matches(result, answer, previous, current)
        elif kind == "goal_and_rule":
            report["selection_side_opportunity_steps"] = side_opportunities(result, answer, current.rule)
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
