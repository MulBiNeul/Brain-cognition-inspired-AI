"""Rule engine for parameterized four-peg, three-disk Hanoi tasks.

This module does not import or alter the experiment runner.  A position tuple is
ordered by disk number (smallest first).  Search states include the previous departure peg and visit history needed
by the movement restrictions.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from itertools import permutations, product
import json
from typing import Any, Mapping, Sequence

PEGS = ("A", "B", "C", "D")
DISK_COUNT = 3
MAX_MOVES = 30
RULE_TYPES = ("peg", "occupancy", "source_sequence", "visit")
INITIAL_FAMILIES = ("stacked", "pair12", "pair13", "pair23", "separated")
Rule = Mapping[str, Any]
Signature = tuple[Any, ...]
POSITIONS = tuple(product(range(4), repeat=DISK_COUNT))
POSITION_INDEX = {positions: index for index, positions in enumerate(POSITIONS)}


def rule_pool(kind: str) -> list[dict[str, Any]]:
    """Return every distinct rule variant, in deterministic order."""
    if kind == "peg":
        return [dict(kind=kind, source=s, destination=d)
                for s, d in permutations(PEGS, 2)]
    if kind == "occupancy":
        return [dict(kind=kind, occupied_peg=x, source=y, destination=z)
                for x, y, z in permutations(PEGS, 3)]
    if kind == "source_sequence":
        return [dict(kind=kind, previous_source=x, source=y)
                for x, y in permutations(PEGS, 2)]
    if kind == "visit":
        return [dict(kind=kind, visitor_disk=disk, visited_peg=peg, locked_disk=3)
                for disk in (1, 2) for peg in PEGS]
    raise ValueError(f"Unknown rule type: {kind!r}")


def rule_key(rule: Rule) -> str:
    """Stable identity of one additional movement restriction."""
    normalized = dict(rule)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def initial_family(positions: Sequence[str]) -> str:
    _encode_positions(positions)
    a, b, c = positions
    if a == b == c:
        return "stacked"
    if a == b:
        return "pair12"
    if a == c:
        return "pair13"
    if b == c:
        return "pair23"
    return "separated"


def all_initial_positions(family: str | None = None) -> list[tuple[str, ...]]:
    if family is not None and family not in INITIAL_FAMILIES:
        raise ValueError(f"Unknown initial family: {family!r}")
    positions = [tuple(PEGS[p] for p in board) for board in POSITIONS]
    return [p for p in positions if family is None or initial_family(p) == family]


def _encode_positions(positions: Sequence[str]) -> int:
    if len(positions) != DISK_COUNT or any(p not in PEGS for p in positions):
        raise ValueError("positions must contain three pegs, ordered disk 1, 2, 3")
    return POSITION_INDEX[tuple(PEGS.index(p) for p in positions)]


def _signature(rule: Rule | None) -> Signature | None:
    if rule is None:
        return None
    key = rule_key(rule)
    kind = rule.get("kind")
    if kind not in RULE_TYPES or key not in _RULE_SIGNATURES:
        raise ValueError(f"Invalid additional rule: {dict(rule)!r}")
    return _RULE_SIGNATURES[key]


def _compile(rule: Rule) -> Signature:
    kind = rule["kind"]
    if kind == "peg":
        return kind, PEGS.index(rule["source"]), PEGS.index(rule["destination"])
    if kind == "occupancy":
        return (kind, PEGS.index(rule["occupied_peg"]), PEGS.index(rule["source"]),
                PEGS.index(rule["destination"]))
    if kind == "source_sequence":
        return kind, PEGS.index(rule["previous_source"]), PEGS.index(rule["source"])
    return kind, rule["visitor_disk"] - 1, PEGS.index(rule["visited_peg"]), rule["locked_disk"] - 1


_RULE_SIGNATURES = {rule_key(rule): _compile(rule)
                    for kind in RULE_TYPES for rule in rule_pool(kind)}


def _board_edges(board: tuple[int, ...]) -> tuple[tuple[str, int, int, int, int], ...]:
    top = {peg: next((d for d, p in enumerate(board) if p == peg), None)
           for peg in range(4)}
    result = []
    for source in range(4):
        disk = top[source]
        if disk is None:
            continue
        for destination in range(4):
            if source == destination:
                continue
            target_disk = top[destination]
            if target_disk is not None and target_disk < disk:
                continue
            changed = list(board)
            changed[disk] = destination
            result.append((PEGS[source] + PEGS[destination], disk, source,
                           destination, POSITION_INDEX[tuple(changed)]))
    return tuple(result)


EDGES = tuple(_board_edges(board) for board in POSITIONS)


def _initial_visit(rule: Signature | None, board: tuple[int, ...]) -> bool:
    return bool(rule and rule[0] == "visit" and board[rule[1]] == rule[2])


def _next_visit(rule: Signature | None, visited: bool, disk: int, destination: int) -> bool:
    return bool(rule and rule[0] == "visit" and
                (visited or (disk == rule[1] and destination == rule[2])))


def _forbids(rule: Signature | None, board: tuple[int, ...], disk: int,
             source: int, destination: int, previous_source: int, visited: bool) -> bool:
    if rule is None:
        return False
    if rule[0] == "peg":
        return source == rule[1] and destination == rule[2]
    if rule[0] == "occupancy":
        return rule[1] in board and source == rule[2] and destination == rule[3]
    if rule[0] == "source_sequence":
        return previous_source == rule[1] and source == rule[2]
    return disk == rule[3] and not visited


@lru_cache(maxsize=100_000)
def _bfs(start: int, goal: int, allowed: Signature | None,
         observed: Signature | None, require_violation: bool,
         max_moves: int) -> tuple[str, ...] | None:
    """Search on board + previous source + rule-specific visit bits + violation."""
    remembers_source = any(r and r[0] == "source_sequence" for r in (allowed, observed))
    initial = (start, 4, _initial_visit(allowed, POSITIONS[start]),
               _initial_visit(observed, POSITIONS[start]), False)
    parents: dict[tuple[Any, ...], tuple[tuple[Any, ...], str] | None] = {initial: None}
    frontier = deque([(initial, 0)])
    while frontier:
        state, depth = frontier.popleft()
        board_id, previous_source, allowed_visit, observed_visit, violated = state
        if POSITIONS[board_id] == (goal, goal, goal) and (violated or not require_violation):
            plan: list[str] = []
            while parents[state] is not None:
                state, move = parents[state]  # type: ignore[misc]
                plan.append(move)
            return tuple(reversed(plan))
        if depth >= max_moves:
            continue
        board = POSITIONS[board_id]
        for move, disk, source, destination, next_board in EDGES[board_id]:
            if _forbids(allowed, board, disk, source, destination, previous_source, allowed_visit):
                continue
            next_violated = violated or _forbids(
                observed, board, disk, source, destination, previous_source, observed_visit)
            next_state = (next_board, source if remembers_source else 4,
                          _next_visit(allowed, allowed_visit, disk, destination),
                          _next_visit(observed, observed_visit, disk, destination), next_violated)
            if next_state not in parents:
                parents[next_state] = (state, move)
                frontier.append((next_state, depth + 1))
    return None


def _search(initial_positions: Sequence[str], goal_peg: str, *,
            allowed: Rule | None, observed: Rule | None, require_violation: bool,
            max_moves: int) -> list[str] | None:
    if goal_peg not in PEGS or max_moves < 0:
        raise ValueError("Invalid goal peg or move limit")
    result = _bfs(_encode_positions(initial_positions), PEGS.index(goal_peg),
                  _signature(allowed), _signature(observed), require_violation, max_moves)
    return None if result is None else list(result)


def shortest_plan(initial_positions: Sequence[str], goal_peg: str,
                  rule: Rule | None = None, max_moves: int = MAX_MOVES) -> list[str] | None:
    """Shortest basic-legal solution satisfying the optional additional rule."""
    return _search(initial_positions, goal_peg, allowed=rule, observed=None,
                   require_violation=False, max_moves=max_moves)


def restriction_witness(initial_positions: Sequence[str], goal_peg: str,
                        rule: Rule, max_moves: int = MAX_MOVES) -> list[str] | None:
    """Shortest basic-legal goal-reaching plan violating the current rule."""
    return _search(initial_positions, goal_peg, allowed=None, observed=rule,
                   require_violation=True, max_moves=max_moves)


def transition_witness(initial_positions: Sequence[str], goal_peg: str,
                       previous_rule: Rule, current_rule: Rule,
                       max_moves: int = MAX_MOVES) -> list[str] | None:
    """Use the SAME current initial/goal; allow old rule and violate new rule."""
    return _search(initial_positions, goal_peg, allowed=previous_rule, observed=current_rule,
                   require_violation=True, max_moves=max_moves)


def permute_rule(rule: Rule, mapping: Mapping[str, str]) -> dict[str, Any]:
    result = dict(rule)
    for field in ("source", "destination", "occupied_peg", "previous_source", "visited_peg"):
        if field in result:
            result[field] = mapping[result[field]]
    return result


def canonical_episode(problem: Mapping[str, Any]) -> str:
    """Episode identity modulo all 24 global peg renamings; ignore witnesses."""
    representations = []
    for permutation in permutations(PEGS):
        mapping = dict(zip(PEGS, permutation))
        rounds = [{"initial_positions": [mapping[p] for p in round_["initial_positions"]],
                   "goal_peg": mapping[round_["goal_peg"]],
                   "rule": permute_rule(round_["rule"], mapping)}
                  for round_ in problem["rounds"]]
        representations.append(json.dumps({"rule_type": problem["rule_type"], "rounds": rounds},
                                           sort_keys=True, separators=(",", ":")))
    return min(representations)
