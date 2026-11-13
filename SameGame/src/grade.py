"""Grade remaining strings, replay deletion sequences, and validate the forbidden-pair dataset."""

import argparse
import json
import re
from collections import Counter, defaultdict
from functools import cache
from itertools import groupby
from pathlib import Path
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
    """Read numbered actions or the compact notation selecting the first group."""
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


def grade_case(public, item_id, number, case_id, actions=None, answer=None, *, group_counts=False):
    """Inspect one input condition using its round's current and previous rules."""
    require(public.get("version") == 1, "unsupported problem schema")
    item = next((item for item in public["items"] if item["id"] == item_id), None)
    require(item is not None, f"unknown item {item_id!r}")
    require(type(number) is int and 1 <= number <= len(item["rounds"]), "round number out of range")
    record = item["rounds"][number - 1]
    case = next((case for case in record["cases"] if case["id"] == case_id), None)
    require(case is not None, f"unknown case {case_id!r} in {item_id} R{number}")
    game = read_round({**case, "round": record["round"], "forbidden": record["forbidden"]},
                      number, item["type"])
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
            "final_answer_types": len(counts),
            "groupings_by_final": {final: counts[final] for final in finals},
        }
    elif actions is None:
        result = game.grade_final(answer, forbidden, previous)
    else:
        result = game.grade(actions, forbidden, previous, final_answer=answer)
        result["mode"] = "actions_only" if answer is None else "actions_and_final"
    return {"id": item_id, "round": number, "case_id": case_id, "input": case["input"],
            "target": game.target, "forbidden": forbidden, "previous_forbidden": previous, **result}


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


def validate(public, key):
    """Validate the dataset with the same exact checks used during generation."""
    from .dataset import validate as validate_dataset
    return validate_dataset(public, key)


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
    parser.add_argument("--case", dest="case_id", help="input condition ID within the selected round, e.g. c01")
    parser.add_argument("actions", nargs="?", help="numbered deletions such as 'A2 B1' ('AB' also means 'A1 B1')")
    parser.add_argument("--answer", help="the remaining string; quote spaces and use '' for the empty string")
    args = parser.parse_intermixed_args()
    provided = (args.id, args.round, args.case_id, args.actions, args.answer)
    if args.validate and any(value is not None for value in provided):
        parser.error("--validate checks all problems and takes no item, round, case, actions, or answer")
    if args.group_counts and (args.actions is not None or args.answer is not None):
        parser.error("--group-counts takes ID ROUND --case CASE without actions or an answer")
    if not args.validate and (args.id is None or args.round is None or args.case_id is None
                              or (not args.group_counts and args.actions is None and args.answer is None)):
        parser.error("provide ID ROUND --case CASE with --group-counts, --answer TEXT, or ACTIONS; or use --validate")
    try:
        public = json.loads((args.data_dir / "problems.json").read_text(encoding="utf-8"))
        if args.validate:
            key = json.loads((args.data_dir / "answer_key.json").read_text(encoding="utf-8"))
            result = validate(public, key)
        else:
            result = grade_case(public, args.id, args.round, args.case_id, args.actions, args.answer,
                                group_counts=args.group_counts)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
