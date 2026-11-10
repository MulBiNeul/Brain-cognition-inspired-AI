"""Independent pilot-5 validator for whole-run deletion with permanent boundaries."""

import re
from collections import Counter
from functools import lru_cache
from itertools import groupby
from typing import NamedTuple


LETTERS = "ABCDE"
INPUT = re.compile(r"[A-E]+(?: [A-E]+)?")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def expect(actual, expected, field):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{field}: expected {expected!r}, found {actual!r}")


class Move(NamedTuple):
    mask: int
    merged: bool
    singleton_merge: bool


class Stats(NamedTuple):
    count: int
    min_merged: int
    min_singleton: int


class Game:
    """A bit identifies a maximal run in the original input. Whole runs disappear together."""

    def __init__(self, tape, target):
        self.original = tuple("".join(group) for _, group in groupby(tape))
        self.start = (1 << len(self.original)) - 1
        self.target = target
        self._cache = {}

    def state(self, mask):
        if mask in self._cache:
            return self._cache[mask]
        live = [i for i in range(len(self.original)) if mask & (1 << i)]
        text = "".join(self.original[i] for i in live)
        letters = text.replace(" ", "")
        terminal = None
        if len(letters) == self.target:
            terminal = ("duplicate_final_pair" if self.target == 2 and letters[0] == letters[1]
                        else "target_reached")
        elif len(letters) < self.target:
            terminal = "target_overshot"

        choices = {}
        for letter, group in groupby(live, key=lambda i: self.original[i][0]):
            if letter == " " or letter in choices:
                continue
            ids = tuple(group)
            if sum(len(self.original[i]) for i in ids) < 2:
                continue
            merged = len(ids) > 1
            singleton_merge = merged and all(len(self.original[i]) == 1 for i in ids)
            removed = sum(1 << i for i in ids)
            choices[letter] = Move(mask ^ removed, merged, singleton_merge)
        self._cache[mask] = text, terminal, choices
        return self._cache[mask]

    def statistics(self):
        @lru_cache(maxsize=None)
        def visit(mask):
            text, terminal, choices = self.state(mask)
            if terminal == "target_reached":
                return Stats(1, 0, 0)
            if terminal or len(text.replace(" ", "")) < self.target + 2:
                return Stats(0, 0, 0)
            branches = [(move, visit(move.mask)) for move in choices.values()]
            successes = [(m, child) for m, child in branches if child.count]
            if not successes:
                return Stats(0, 0, 0)
            return Stats(sum(c.count for _, c in successes),
                         min(m.merged + c.min_merged for m, c in successes),
                         min(m.singleton_merge + c.min_singleton for m, c in successes))

        result = visit(self.start)
        visit.cache_clear()
        return result

    def replay(self, answer):
        check(isinstance(answer, str), "submitted actions must be a string")
        mask = self.start
        states = [self.state(mask)[0]]
        merges, singletons = [], []
        for step, action in enumerate(answer, 1):
            _, terminal, choices = self.state(mask)
            if terminal == "target_reached":
                reason, stop_step = "action_after_target", step
                break
            if terminal:
                reason, stop_step = terminal, step - 1
                break
            if action not in LETTERS:
                reason, stop_step = "invalid_action_symbol", step
                break
            if action not in choices:
                reason, stop_step = "no_removable_run", step
                break
            move = choices[action]
            if move.merged:
                merges.append(step)
            if move.singleton_merge:
                singletons.append(step)
            mask = move.mask
            states.append(self.state(mask)[0])
        else:
            reason = self.state(mask)[1] or "target_not_reached"
            stop_step = len(answer)
        return {"valid": reason == "target_reached", "reason": reason, "stop_reason": reason,
                "stop_step": stop_step, "executed_steps": len(states) - 1, "states": states,
                "final_state": states[-1], "remaining_length": len(states[-1].replace(" ", "")),
                "merge_steps": merges, "singleton_merge_steps": singletons}


def game_for(record, number):
    expect(set(record), {"round", "input", "target_length"}, f"R{number} public fields")
    expect(record["round"], number, f"R{number}.round")
    tape, target = record["input"], record["target_length"]
    check(isinstance(tape, str) and INPUT.fullmatch(tape), f"R{number}: invalid input")
    letters = tape.replace(" ", "")
    check(28 <= len(letters) <= 32 and set(letters) == set(LETTERS),
          f"R{number}: input must contain 28–32 letters and include all of A–E")
    check(type(target) is int and target in range(4), f"R{number}: invalid target_length")
    return Game(tape, target)


def verify_round(record, key, number):
    game = game_for(record, number)
    expect(key["round"], number, f"R{number}.round in answer key")
    check(len(game.state(game.start)[2]) >= 2, f"R{number}: fewer than two initial letter choices")
    stats = game.statistics()
    check(stats.count > 0, f"R{number}: no solutions")
    for name, value in (("n_valid_answers", stats.count), ("min_merged_deletions", stats.min_merged),
                        ("min_singleton_merge_deletions", stats.min_singleton)):
        expect(key[name], value, f"R{number}.{name}")
    check(stats.min_merged >= 2 and stats.min_singleton >= 1,
          f"R{number}: some successful path has too few merge deletions")
    ref = game.replay(key["one_valid_answer"])
    check(ref["valid"], f"R{number}: reference failed: {ref['reason']} at {ref['stop_step']}")
    for name in ("final_state", "merge_steps", "singleton_merge_steps"):
        expect(key[name], ref[name], f"R{number}.{name}")
    tape = record["input"]
    if " " in tape:
        without = Game(tape.replace(" ", ""), game.target).replay(key["one_valid_answer"])
        check(not without["valid"], f"R{number}: reference still succeeds after erasing the boundary")
    return stats, ref["final_state"]


def verify_probe(probe, records, answers, spaced):
    if not spaced:
        expect(probe, None, "space_probe")
        return
    check(isinstance(probe, dict), "spaced item needs a space_probe")
    number = probe["round"]
    check(type(number) is int and 1 <= number <= 5, "invalid space_probe.round")
    record = records[number - 1]
    tape, target = record["input"], record["target_length"]
    answer = probe["answer"]
    expect(answer, answers[number - 1]["one_valid_answer"], "space_probe.answer")
    check(Game(tape, target).replay(answer)["valid"], "space_probe fails on its original input")
    check(not Game(tape.replace(" ", ""), target).replay(answer)["valid"],
          "space_probe still succeeds without its permanent boundary")


def validate(public, key):
    items, key_items = public["items"], key["items"]
    expect(len(items), 50, "public item count")
    expect(len(key_items), 50, "key item count")
    keyed = {item["id"]: item for item in key_items}
    ids = {item["id"] for item in items}
    check(len(ids) == 50 and ids == keyed.keys(), "public/key IDs must match uniquely")
    seen = set()
    targets = [Counter() for _ in range(5)]
    spaces = Counter()
    minima = []
    examples = []
    verified_spaced_rounds = 0
    for item in items:
        item_id = item["id"]
        try:
            expect(set(item), {"id", "round_settings"}, "public item fields")
            records = item["round_settings"]
            answers = keyed[item_id]["rounds"]
            expect(len(records), 5, "public rounds")
            expect(len(answers), 5, "key rounds")
            item_goals, item_spaces = [], []
            for number, (record, answer) in enumerate(zip(records, answers), 1):
                stats, final = verify_round(record, answer, number)
                tape, target = record["input"], record["target_length"]
                check(tape not in seen, f"R{number}: input already appears elsewhere in the pilot")
                seen.add(tape)
                targets[number - 1][target] += 1
                item_goals.append(target)
                item_spaces.append(tape.count(" "))
                verified_spaced_rounds += int(" " in tape)
                minima.append((stats.min_merged, stats.min_singleton))
                if final == "AB A" and target == 3:
                    examples.append([item_id, number])
            check(set(item_goals) == set(range(4)) and all(a != b for a, b in zip(item_goals, item_goals[1:])),
                  "each item must cover goals 0–3 with no consecutive equal goals")
            check(len(set(item_spaces)) == 1, "all five inputs must share boundary status")
            spaces[item_spaces[0]] += 1
            verify_probe(keyed[item_id]["space_probe"], records, answers, bool(item_spaces[0]))
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ValueError(f"{item_id}: {error}") from error
    expect(dict(spaces), {0: 30, 1: 20}, "space counts by item")
    for number, counts in enumerate(targets, 1):
        check(all(counts[t] in (12, 13) for t in range(4)), f"R{number}: unbalanced goals")
    all_targets = sum(targets, Counter())
    check(all(all_targets[t] in (62, 63) for t in range(4)), "unbalanced goals across all rounds")
    check(examples, "no saved reference ends in the three-letter state 'AB A'")
    return {"version": public["version"], "items": len(items), "rounds": len(minima),
            "distinct_inputs": len(seen), "space_counts": dict(sorted(spaces.items())),
            "spaced_references_failing_without_boundary": verified_spaced_rounds,
            "item_space_probes_validated": spaces[1], "all_round_target_length_counts": dict(sorted(all_targets.items())),
            "target_length_counts_by_round": [dict(sorted(counts.items())) for counts in targets],
            "min_merged_deletions_over_all_answers": min(m for m, s in minima),
            "min_singleton_merge_deletions_over_all_answers": min(s for m, s in minima),
            "literal_AB_A_examples": examples}


def grade(public, item_id, number, answer):
    check(1 <= number <= 5, "round must be 1 through 5")
    selected = next((i for i in public["items"] if i["id"] == item_id), None)
    check(selected is not None, f"unknown item {item_id!r}")
    record = selected["round_settings"][number - 1]
    game = game_for(record, number)
    return {"id": item_id, "round": number, "target_length": game.target, **game.replay(answer)}
