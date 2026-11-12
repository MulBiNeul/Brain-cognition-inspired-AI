"""Grade remaining strings, replay deletion sequences, and validate the forbidden-pair dataset."""

import argparse
import json
import re
from collections import Counter, defaultdict
from functools import cache
from itertools import groupby
from pathlib import Path
from statistics import median
from typing import NamedTuple

from ._paths import DATA_DIR


LETTERS = "ABCDE"
BOARD = re.compile(r"[A-E]+(?: [A-E]+)?")
INDEXED_MOVES = re.compile(r"[A-E][1-9][0-9]*(?: [A-E][1-9][0-9]*)*")
TYPES = {"same_letter", "different_letter"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_moves(moves):
    """Read numbered actions or the compact legacy notation for first runs."""
    if isinstance(moves, str):
        if re.fullmatch(r"[A-E]+", moves):
            return tuple(letter + "1" for letter in moves)
        if INDEXED_MOVES.fullmatch(moves):
            return tuple(moves.split(" "))
    raise ValueError("expected numbered moves separated by single spaces, e.g. 'A2 B1', or compact letters")


class Move(NamedTuple):
    mask: int
    merged: bool
    singleton_merge: bool


class Board(NamedTuple):
    text: str
    terminal: str | None
    moves: dict[str, Move]


class Witness(NamedTuple):
    recoverable: int
    hits: int
    answer: str

    def rank(self):
        steps = self.answer.count(" ") + 1 if self.answer else 0
        return self.recoverable, self.hits, -steps


class LegalEndings(NamedTuple):
    letters_by_length: tuple[set[str], ...]
    failing_actions: str | None


class Game:
    """Each surviving bit identifies a whole maximal run in the original input."""

    def __init__(self, tape, target):
        self.runs = tuple("".join(group) for _, group in groupby(tape))
        self.start = (1 << len(self.runs)) - 1
        self.target = target
        self._boards = {}
        self._finishes = {}

    def board(self, mask):
        if mask in self._boards:
            return self._boards[mask]
        live = [i for i in range(len(self.runs)) if mask & (1 << i)]
        text = "".join(self.runs[i] for i in live)
        letters = text.replace(" ", "")
        moves = {}
        ordinals = Counter()
        for letter, group in groupby(live, key=lambda i: self.runs[i][0]):
            if letter == " ":
                continue
            ids = tuple(group)
            if sum(len(self.runs[i]) for i in ids) < 2:
                continue
            merged = len(ids) > 1
            singleton = merged and all(len(self.runs[i]) == 1 for i in ids)
            removed = sum(1 << i for i in ids)
            ordinals[letter] += 1
            moves[f"{letter}{ordinals[letter]}"] = Move(mask ^ removed, merged, singleton)
        terminal = None
        if not moves:
            if len(letters) == self.target:
                terminal = "target_reached"
            elif len(letters) < self.target:
                terminal = "target_overshot"
            else:
                terminal = "target_not_reached"
        self._boards[mask] = Board(text, terminal, moves)
        return self._boards[mask]

    def physical_minima(self):
        """Minimize the two merge counts separately over all physical successes."""
        @cache
        def visit(mask):
            current = self.board(mask)
            if current.terminal:
                return (0, 0) if current.terminal == "target_reached" else None
            best_merged = best_singleton = None
            for move in current.moves.values():
                child = visit(move.mask)
                if child is not None:
                    merged = move.merged + child[0]
                    singleton = move.singleton_merge + child[1]
                    best_merged = merged if best_merged is None else min(best_merged, merged)
                    best_singleton = singleton if best_singleton is None else min(best_singleton, singleton)
            return None if best_merged is None else (best_merged, best_singleton)

        result = visit(self.start)
        visit.cache_clear()
        return result

    def can_finish(self, mask, last, forbidden):
        """Test legal continuations using the live board and last executed action."""
        key = (mask, last, forbidden)
        if key not in self._finishes:
            board = self.board(mask)
            if board.terminal:
                succeeds = board.terminal == "target_reached"
            else:
                succeeds = any(last + token[0] != forbidden and self.can_finish(move.mask, token[0], forbidden)
                               for token, move in board.moves.items())
            self._finishes[key] = succeeds
        return self._finishes[key]

    def legal_endings(self, forbidden):
        """Explore legal plays to completion for each candidate goal from zero to three."""
        letters_by_length = tuple(set() for _ in range(4))
        failing_actions = None
        seen = set()
        pending = [(self.start, "", "")]
        while pending:
            mask, last, path = pending.pop()
            if (mask, last) in seen:
                continue
            seen.add((mask, last))
            board = self.board(mask)
            if not board.moves:
                letters = board.text.replace(" ", "")
                if len(letters) < len(letters_by_length):
                    letters_by_length[len(letters)].add(letters)
                if failing_actions is None and len(letters) != self.target:
                    failing_actions = path
            pending.extend((move.mask, token[0], (path + " " + token).lstrip()) for token, move in board.moves.items()
                           if last + token[0] != forbidden)
        return LegalEndings(letters_by_length, failing_actions)

    def final_witness(self, final, forbidden, required_pair=None, flat=None):
        """Reach this exact final, optionally violating an old pair and failing on a flat copy."""
        if final and not BOARD.fullmatch(final):
            return None
        if len(final.replace(" ", "")) != self.target or re.search(r"([A-E])\1", final):
            return None

        @cache
        def search(mask, last, need_pair, flat_mask):
            board = self.board(mask)
            if board.terminal:
                if board.terminal != "target_reached" or need_pair or board.text.strip(" ") != final:
                    return None
                if flat is not None and flat_mask is not None and flat.board(flat_mask).terminal == "target_reached":
                    return None
                return ""
            if not self.can_finish(mask, last, forbidden):
                return None
            for token, move in board.moves.items():
                letter = token[0]
                pair = last + letter
                if pair == forbidden:
                    continue
                # None marks a flattened replay that already stopped before this path ended.
                next_flat = None
                if flat is not None and flat_mask is not None:
                    other = flat.board(flat_mask)
                    if token in other.moves:
                        next_flat = other.moves[token].mask
                suffix = search(move.mask, letter, need_pair and pair != required_pair, next_flat)
                if suffix is not None:
                    return token + " " + suffix if suffix else token
            return None

        path = search(self.start, "", required_pair is not None, None if flat is None else flat.start)
        search.cache_clear()
        return path

    def grade_final(self, answer, forbidden, previous=None):
        require(isinstance(answer, str), "final answer must be a string")
        final = answer.strip(" ")
        valid = self.final_witness(final, forbidden) is not None
        old_valid = None if previous is None else self.final_witness(final, previous) is not None
        # Previous-only here describes reachability of the submitted string. No path was observed.
        return {"mode": "final_only", "answer": final, "valid": valid,
                "reason": "reachable_final" if valid else "unreachable_final", "forbidden": forbidden,
                "previous_forbidden": previous, "valid_under_previous_rule": old_valid,
                "previous_only": None if previous is None else old_valid and not valid}

    def previous_rule_witness(self, forbidden, previous):
        """Find an old-rule success with the most locally recoverable violations."""
        @cache
        def best(mask, last):
            board = self.board(mask)
            if board.terminal:
                return Witness(0, 0, "") if board.terminal == "target_reached" else None
            winner = None
            for token in sorted(board.moves, key=lambda action: action[0]):
                letter = token[0]
                pair = last + letter
                if pair == previous:
                    continue
                suffix = best(board.moves[token].mask, letter)
                if suffix is None:
                    continue
                hit = pair == forbidden
                recoverable = hit and self.can_finish(mask, last, forbidden)
                answer = token + " " + suffix.answer if suffix.answer else token
                candidate = Witness(recoverable + suffix.recoverable, hit + suffix.hits, answer)
                if winner is None or candidate.rank() > winner.rank():
                    winner = candidate
            return winner

        result = best(self.start, "")
        best.cache_clear()
        return result

    def grade(self, answer, forbidden, previous=None, *, final_answer=None):
        tokens = parse_moves(answer)
        letters = "".join(token[0] for token in tokens)
        board = self.board(self.start)
        masks = [self.start]
        merged, singleton = [], []
        failed_step = None
        for step, token in enumerate(tokens, 1):
            if token not in board.moves:
                stop_reason = "no_removable_run"
                failed_step = stop_step = step
                break
            move = board.moves[token]
            if move.merged:
                merged.append(step)
            if move.singleton_merge:
                singleton.append(step)
            board = self.board(move.mask)
            masks.append(move.mask)
        else:
            stop_reason = board.terminal or "removable_run_remaining"
            stop_step = len(tokens)

        executed = len(masks) - 1
        # A recognized letter on the first impossible move still identifies a
        # directed pair. Its command is checked but its deletion is not executed.
        checked = executed + int(failed_step is not None)
        pairs = [letters[i - 1:i + 1] for i in range(1, checked)]
        violations = [step for step, pair in enumerate(pairs, 2) if pair == forbidden]
        completed = stop_reason == "target_reached"
        valid = completed and not violations
        previous_valid = old_allowed_steps = previous_only = None
        previous_violations = None
        if previous is not None:
            previous_violations = [step for step, pair in enumerate(pairs, 2) if pair == previous]
            previous_valid = completed and not previous_violations
            old_allowed_steps = [step for step in violations if pairs[step - 2] != previous]
            previous_only = previous_valid and not valid

        result = {
            "valid": valid, "reason": "forbidden_pair" if completed and violations else stop_reason,
            "stop_reason": stop_reason, "stop_step": stop_step,
            "answer": answer, "forbidden": forbidden, "previous_forbidden": previous,
            "executed_steps": executed, "checked_steps": checked, "first_basic_error_step": failed_step,
            "trajectory": [self.board(mask).text for mask in masks],
            "final_state": board.text.strip(" "),
            "board_exhausted": not board.moves,
            "remaining_length": len(board.text.replace(" ", "")),
            "merged_steps": merged, "singleton_merge_steps": singleton,
            "violation_steps": violations,
            "previous_violation_steps": previous_violations,
            "recoverable_violation_steps": [
                step for step in violations if not self.board(masks[step - 1]).terminal
                and self.can_finish(masks[step - 1], letters[step - 2], forbidden)
            ],
            "previous_allows_current_forbids_steps": old_allowed_steps,
            "valid_under_previous_rule": previous_valid, "previous_only": previous_only,
            "action_violation_rate": rate(len(violations), checked),
            "transition_violation_rate": rate(len(violations), len(pairs)),
        }
        if final_answer is not None:
            final = final_answer.strip(" ")
            matches = result["final_state"] == final
            result.update(actions=answer, answer=final, moves_valid=valid, final_matches_trace=matches)
            if not matches:
                if valid:
                    result["reason"] = "final_mismatch"
                result["valid"] = False
                if previous is not None:
                    result["valid_under_previous_rule"] = result["previous_only"] = False
        return result


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def read_round(record, number, item_type):
    require(type(record["round"]) is int and record["round"] == number, "incorrect round number")
    tape, target, forbidden = record["input"], record["target"], record["forbidden"]
    require(isinstance(tape, str) and BOARD.fullmatch(tape), "invalid input or boundary")
    letters = tape.replace(" ", "")
    require(28 <= len(letters) <= 32 and set(letters) == set(LETTERS),
            "input must have 28–32 letters and use all of A–E")
    require(type(target) is int and 0 <= target <= 3, "invalid target")
    require(isinstance(forbidden, str) and re.fullmatch("[A-E]{2}", forbidden), "invalid forbidden pair")
    require(item_type in TYPES and (forbidden[0] == forbidden[1]) == (item_type == "same_letter"),
            "forbidden pair disagrees with item type")
    return Game(tape, target)


def grade_item(public, item_id, number, actions=None, answer=None, *, group_counts=False):
    item = next((item for item in public["items"] if item["id"] == item_id), None)
    require(item is not None, f"unknown item {item_id!r}")
    require(1 <= number <= len(item["rounds"]), "round number out of range")
    record = item["rounds"][number - 1]
    game = read_round(record, number, item["type"])
    previous = item["rounds"][number - 2]["forbidden"] if number > 1 else None
    forbidden = record["forbidden"]
    if group_counts:
        require(actions is None and answer is None, "group counts take no actions or answer")
        counts = terminal_group_counts(game)
        finals = sorted(counts, key=lambda final: (len(final.replace(" ", "")), final))
        result = {
            "mode": "physical_group_counts",
            "total_groupings": sum(counts.values()),
            "target_length_groupings": sum(count for final, count in counts.items()
                                          if len(final.replace(" ", "")) == game.target),
            "groupings_by_final": {final: counts[final] for final in finals},
        }
    elif actions is None:
        result = game.grade_final(answer, forbidden, previous)
    else:
        result = game.grade(actions, forbidden, previous, final_answer=answer)
        result["mode"] = "actions_only" if answer is None else "actions_and_final"
    return {"id": item_id, "round": number, "target": game.target, **result}


def terminal_group_counts(game):
    """Count unordered deletion groupings by terminal string, regardless of goal or rule."""
    counts = Counter()
    # Original runs never split. A set of removed run masks identifies exactly
    # which original positions were deleted together, independent of move order.
    frontier = {(game.start, frozenset())}
    while frontier:
        following = set()
        for mask, groups in frontier:
            board = game.board(mask)
            if not board.moves:
                counts[board.text.strip(" ")] += 1
            for move in board.moves.values():
                removed = mask ^ move.mask
                following.add((move.mask, groups | {removed}))
        frontier = following
    return counts


def successful_paths(game, forbidden):
    """Count distinct moves per final string and carry the uniform policy's probability mass."""
    counts_by_final = Counter()
    masses_by_multiset = Counter()
    frontier = {(game.start, ""): (1, 1.0)}
    while frontier:
        arrivals = {}
        for (mask, last), (count, mass) in frontier.items():
            board = game.board(mask)
            if board.terminal == "target_reached":
                final = board.text.strip(" ")
                counts_by_final[final] += count
                multiset = "".join(sorted(final.replace(" ", "")))
                masses_by_multiset[multiset] += mass
            elif board.terminal is None:
                allowed = [(move.mask, token[0]) for token, move in board.moves.items()
                           if last + token[0] != forbidden]
                groups_per_letter = Counter(letter for _, letter in allowed)
                for child in allowed:
                    previous_count, previous_mass = arrivals.get(child, (0, 0.0))
                    choices = len(groups_per_letter) * groups_per_letter[child[1]]
                    arrivals[child] = previous_count + count, previous_mass + mass / choices
        frontier = arrivals
    return counts_by_final, masses_by_multiset


def summarize_solution_counts(counts_by_goal):
    summary = {}
    for goal, rounds in sorted(counts_by_goal.items()):
        by_field = {}
        for field in ("final_strings", "letter_multisets", "move_sequences"):
            values = [record[field] for record in rounds]
            by_field[field] = {"min": min(values), "median": median(values), "max": max(values),
                               "histogram": dict(sorted(Counter(values).items()))}
        rich_rounds = sum(record["letter_multisets"] >= 3 for record in rounds)
        if goal > 0:
            required = (len(rounds) + 3) // 4
            require(rich_rounds >= required,
                    f"target {goal}: only {rich_rounds} rounds have three letter multisets; need {required}")
        summary[goal] = {"rounds": len(rounds), "rounds_with_at_least_three_multisets": rich_rounds, **by_field}
    return summary


def validate_length_balance(length_counts):
    short_range_strata = full_range_strata = largest_difference = 0
    for (kind, number, goal), counts in length_counts.items():
        label = f"{kind} R{number} target {goal}"
        if kind == "different_letter" and number > 1:
            require(set(counts) == {31, 32}, f"{label}: both lengths 31 and 32 must be used exclusively")
            difference = abs(counts[31] - counts[32])
            require(difference <= 1, f"{label}: length frequencies 31/32 differ by {difference}")
            largest_difference = max(largest_difference, difference)
            short_range_strata += 1
        else:
            require(set(counts) == set(range(28, 33)), f"{label}: all five input lengths 28–32 must occur")
            full_range_strata += 1
    return {"strata_with_balanced_31_and_32": short_range_strata,
            "largest_31_32_count_difference": largest_difference,
            "strata_with_all_five_lengths": full_range_strata}


def validate_round(record, key, number, item_type, previous):
    require(set(record) == {"round", "input", "target", "forbidden"}, "expected round, input, target and forbidden")
    game = read_round(record, number, item_type)
    require(set(key) == {"round", "answer"}, "expected round and answer in key")
    require(type(key["round"]) is int and key["round"] == number, "incorrect key round number")
    forbidden = record["forbidden"]
    final = key["answer"]
    require(isinstance(final, str) and final == final.strip(" "), "expected a canonical final answer string")
    tape = record["input"]
    flat = Game(tape.replace(" ", ""), game.target) if " " in tape else None
    path = game.final_witness(final, forbidden, required_pair=previous, flat=flat)
    require(path is not None, f"no qualifying reference path for final {final!r}")
    ref = game.grade(path, forbidden, previous)
    require(ref["valid"], f"reference fails: {ref['reason']} at {ref['stop_step']}")
    require(ref["final_state"] == final, "reference path ends in a different answer")
    endings = game.legal_endings(forbidden)
    require(endings.failing_actions is not None,
            "need a legal path exhausting all removable runs at an incorrect length")
    final_letters = endings.letters_by_length[game.target]
    multiset_count = len({tuple(sorted(letters)) for letters in final_letters})
    if game.target > 0:
        require(multiset_count >= 2, "need two reachable letter multisets after removing spaces and order; "
                f"found {multiset_count} across {len(final_letters)} letter strings")
    require(any(letters for length, letters in enumerate(endings.letters_by_length) if length != game.target),
            "no complete legal final at another target length from zero through three")
    counts_by_final, masses = successful_paths(game, forbidden)
    move_count = sum(counts_by_final.values())
    require(move_count >= 3, f"need three distinct successful move sequences; found {move_count}")
    if game.target == 0:
        probability = masses[""]
        description = "probability of complete deletion"
    else:
        probability = max(masses.values()) / sum(masses.values())
        description = "probability of most frequent multiset conditioned on success"
    require(probability <= 0.95 + 1e-10, f"uniform legal choices: {description} {probability:.12g} exceeds 0.95")
    if previous is not None:
        require(ref["valid_under_previous_rule"] is False,
                "reference must also distinguish the new rule from the previous one")
    minima = game.physical_minima()
    require(minima is not None and minima[0] >= 4 and minima[1] >= 2,
            f"insufficient minimum merge counts across all physical successes: {minima}")
    if flat is not None:
        require(not flat.grade(path, forbidden)["valid"], "spaced reference still succeeds without its boundary")

    exposure = None
    if previous is not None:
        witness = game.previous_rule_witness(forbidden, previous)
        require(witness is not None, "no success under the previous rule on the current board and goal")
        diagnosis = game.grade(witness.answer, forbidden, previous)
        require(diagnosis["previous_only"], "recomputed witness must succeed under the previous rule only")
        recoverable = diagnosis["recoverable_violation_steps"]
        require(witness.recoverable == len(recoverable) and witness.hits == len(diagnosis["violation_steps"]),
                "witness search and replay disagree")
        require(len(recoverable) >= 3 and 5 * len(recoverable) >= 2 * diagnosis["executed_steps"],
                "witness must have at least three recoverable violations spanning at least 40% of actions")
        exposure = len(recoverable) / diagnosis["executed_steps"]
    counts = {"final_strings": len(counts_by_final), "letter_multisets": multiset_count,
              "move_sequences": move_count}
    return minima, ref["final_state"], exposure, len(final_letters), counts, probability


def validate(public, key):
    require(public["rounds_per_item"] == 5, "expected five rounds per item")
    require(set(key) == {"items"}, "expected only items in answer key")
    items, answers = public["items"], key["items"]
    require(len(items) == len(answers) == 50, "expected 50 public and key items")
    require(all(set(item) == {"id", "rounds"} for item in answers), "expected only id and rounds in key items")
    keyed = {item["id"]: item["rounds"] for item in answers}
    ids = {item["id"] for item in items}
    require(len(ids) == 50 and ids == keyed.keys(), "public and key IDs must match uniquely")
    shapes, minima = set(), []
    boundary_examples = {"A A": [], "AB A": []}
    boundaries, types = Counter(), Counter()
    minimum_distinct_finals, minimum_multisets = {}, {}
    maximum_probabilities = Counter()
    length_counts = defaultdict(Counter)
    solution_counts = defaultdict(list)
    goals_by_round = [Counter() for _ in range(5)]
    goals_by_type = {kind: [Counter() for _ in range(5)] for kind in sorted(TYPES)}
    probes = spaced_refs = 0
    exposures = {kind: [] for kind in TYPES}
    for item in items:
        item_id, kind, records = item["id"], item["type"], item["rounds"]
        require(kind in TYPES, f"{item_id}: unknown type")
        require(len(records) == len(keyed[item_id]) == 5, f"{item_id}: expected five rounds")
        types[kind] += 1
        goals, spaces, rules = [], set(), set()
        previous = None
        for number, (record, saved) in enumerate(zip(records, keyed[item_id]), 1):
            try:
                minimum, final_state, exposure, distinct_finals, counts, probability = validate_round(
                    record, saved, number, kind, previous)
                minima.append(minimum)
                tape, goal, rule = record["input"], record["target"], record["forbidden"]
                length_counts[kind, number, goal][len(tape.replace(" ", ""))] += 1
                solution_counts[goal].append(counts)
                maximum_probabilities[goal] = max(maximum_probabilities[goal], probability)
                if goal > 0:
                    multiset_count = counts["letter_multisets"]
                    minimum_distinct_finals[goal] = min(minimum_distinct_finals.get(goal, distinct_finals),
                                                        distinct_finals)
                    minimum_multisets[goal] = min(minimum_multisets.get(goal, multiset_count), multiset_count)
                shape = board_shape(tape)
                require(shape not in shapes, "repeated input structure up to renaming letters")
                require(rule not in rules, "repeated forbidden pair within item")
                shapes.add(shape)
                rules.add(rule)
                spaces.add(tape.count(" "))
                goals.append(goal)
                goals_by_round[number - 1][goal] += 1
                goals_by_type[kind][number - 1][goal] += 1
                spaced_refs += int(" " in tape)
                if previous is not None:
                    if kind == "different_letter":
                        require(sum(a != b for a, b in zip(rule, previous)) == 1,
                                "exactly one endpoint of the directed pair must change")
                    exposures[kind].append(exposure)
                    probes += 1
                if final_state in boundary_examples:
                    boundary_examples[final_state].append([item_id, number])
                previous = rule
            except (KeyError, TypeError, ValueError, IndexError) as error:
                raise ValueError(f"{item_id} R{number}: {error}") from error
        require(set(goals) == set(range(4)) and all(a != b for a, b in zip(goals, goals[1:])),
                f"{item_id}: use all four targets with no consecutive equal targets")
        require(len(spaces) == 1, f"{item_id}: inconsistent boundary status")
        boundaries[spaces.pop()] += 1
    require(types == {"same_letter": 25, "different_letter": 25}, "expected 25 items of each type")
    require(boundaries == {0: 30, 1: 20}, "expected 30 unspaced and 20 spaced items")
    goals_total = sum(goals_by_round, Counter())
    require(all(goals_total[t] in (62, 63) for t in range(4)), "unbalanced goals across all rounds")
    for number, counts in enumerate(goals_by_round, 1):
        require(all(counts[t] in (12, 13) for t in range(4)), f"R{number}: unbalanced goals")
        for kind, rounds in goals_by_type.items():
            require(all(rounds[number - 1][t] in (6, 7) for t in range(4)),
                    f"{kind} R{number}: unbalanced goals")
    for final, locations in boundary_examples.items():
        require(locations, f"no saved ending {final!r}")
    length_balance = validate_length_balance(length_counts)
    solution_summary = summarize_solution_counts(solution_counts)
    return {"valid": True, "items": len(items), "rounds": len(shapes), "boards": len(shapes), "type_counts": dict(types),
            "spaced_items": boundaries[1], "unspaced_items": boundaries[0],
            "previous_rule_probes": probes, "spaced_references_verified": spaced_refs,
            "recoverable_probe_fraction_of_actions": {
                kind: {"min": min(values), "max": max(values), "probes_at_least_40_percent":
                       sum(value >= 0.4 for value in values)} for kind, values in sorted(exposures.items())},
            "goals_by_round": goals_by_round, "goals_by_type_and_round": goals_by_type,
            "goals_total": dict(sorted(goals_total.items())),
            "input_length_balance": length_balance,
            "solution_counts_by_target": solution_summary,
            "uniform_legal_letter_policy": {
                "maximum_complete_deletion_probability_at_target_zero": maximum_probabilities[0],
                "maximum_conditional_multiset_probability_by_positive_target": {
                    goal: maximum_probabilities[goal] for goal in (1, 2, 3)}},
            "rounds_with_legal_failure": len(shapes),
            "rounds_with_alternative_goal": len(shapes),
            "minimum_distinct_normalized_finals_by_positive_target": dict(sorted(minimum_distinct_finals.items())),
            "minimum_distinct_letter_multisets_by_positive_target": dict(sorted(minimum_multisets.items())),
            "minimum_merged_deletions": min(m[0] for m in minima),
            "minimum_singleton_merge_deletions": min(m[1] for m in minima),
            "literal_AB_A_examples": boundary_examples["AB A"],
            "literal_A_A_examples": boundary_examples["A A"]}


def board_shape(text):
    """Normalize alphabet renamings while retaining the boundary and all run lengths."""
    names = {}
    shape = []
    for symbol in text:
        if symbol != " " and symbol not in names:
            names[symbol] = LETTERS[len(names)]
        shape.append(names.get(symbol, " "))
    return "".join(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate", action="store_true", help="check all public problems and the answer key")
    modes.add_argument("--group-counts", action="store_true",
                       help="count unordered deletion groupings for every terminal string regardless of goal or rule")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("id", nargs="?")
    parser.add_argument("round", nargs="?", type=int)
    parser.add_argument("actions", nargs="?", help="numbered deletions such as 'A2 B1' (legacy 'AB' means 'A1 B1')")
    parser.add_argument("--answer", help="the remaining string; quote spaces and use '' for the empty string")
    args = parser.parse_args()
    provided = (args.id, args.round, args.actions, args.answer)
    if args.validate and any(value is not None for value in provided):
        parser.error("--validate checks all boards and takes no item, round, actions, or answer")
    if args.group_counts and (args.actions is not None or args.answer is not None):
        parser.error("--group-counts takes an ID and ROUND without actions or an answer")
    if not args.validate and (args.id is None or args.round is None
                              or (not args.group_counts and args.actions is None and args.answer is None)):
        parser.error("provide ID ROUND with --group-counts, --answer TEXT, or ACTIONS; or use --validate")
    try:
        public = json.loads((args.data_dir / "problems.json").read_text(encoding="utf-8"))
        if args.validate:
            key = json.loads((args.data_dir / "answer_key.json").read_text(encoding="utf-8"))
            result = validate(public, key)
        else:
            result = grade_item(public, args.id, args.round, args.actions, args.answer, group_counts=args.group_counts)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
