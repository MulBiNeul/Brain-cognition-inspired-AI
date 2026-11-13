"""Exact board searches and certificates used by dataset generation and validation."""

from functools import cache
from itertools import groupby

from .grade import Game, board_shape, require, successful_paths


def mechanical_shape(tape):
    """Ignore harmless extra copies inside an initially removable run."""
    return board_shape("".join(letter * min(2, len(list(group))) for letter, group in groupby(tape)))


def solution_lengths(game, forbidden):
    @cache
    def visit(mask, last):
        board = game.board(mask)
        if board.terminal:
            return (0, 0, "") if board.terminal == "target_reached" else None
        best = None
        for token, move in board.moves.items():
            if last + token[0] == forbidden:
                continue
            child = visit(move.mask, token[0])
            if child is None:
                continue
            path = token + (" " + child[2] if child[2] else "")
            if best is None:
                best = child[0] + 1, child[1] + 1, path
            else:
                best = (min(best[0], child[0] + 1), max(best[1], child[1] + 1),
                        path if child[0] + 1 < best[0] else best[2])
        return best
    result = visit(game.start, "")
    visit.cache_clear()
    return result


def repeat_certificate(tape, target, forbidden):
    game = Game(tape, target)
    @cache
    def visit(mask, last):
        board = game.board(mask)
        if board.terminal:
            return (0, "") if board.terminal == "target_reached" else None
        best = None
        for token, move in board.moves.items():
            if last + token[0] == forbidden:
                continue
            child = visit(move.mask, token[0])
            if child is None:
                continue
            cost = child[0] + int(last == token[0])
            if best is None or cost < best[0]:
                best = cost, token + (" " + child[1] if child[1] else "")
                if cost == 0:
                    break
        return best
    result = visit(game.start, "")
    visit.cache_clear()
    return {"minimum_same_letter_transitions": result[0] if result else None,
            "witness": result[1] if result else None,
            "alternating_solution_exists": result is not None and result[0] == 0}


def alternating_witness(game):
    @cache
    def visit(mask, last):
        board = game.board(mask)
        if board.terminal:
            return "" if board.terminal == "target_reached" else None
        for token, move in board.moves.items():
            if token[0] == last:
                continue
            child = visit(move.mask, token[0])
            if child is not None:
                return token + (" " + child if child else "")
        return None
    result = visit(game.start, "")
    visit.cache_clear()
    return result


def find_path(game, forbidden=None, *, wrong_goal=False, required_pair=None, current_failure_pair=None):
    """Find a complete path with explicit goal and pair-hit requirements."""
    @cache
    def visit(mask, last, required_hit, current_hit):
        board = game.board(mask)
        if board.terminal:
            goal = board.terminal == "target_reached"
            acceptable = (not goal and len(board.text.replace(" ", "")) <= 3) if wrong_goal else goal
            if current_failure_pair is not None:
                acceptable = required_hit and (not goal or current_hit)
            elif required_pair is not None:
                acceptable = acceptable and required_hit
            return "" if acceptable else None
        for token, move in board.moves.items():
            pair = last + token[0]
            if forbidden is not None and pair == forbidden:
                continue
            child = visit(move.mask, token[0], required_hit or pair == required_pair,
                          current_hit or pair == current_failure_pair)
            if child is not None:
                return token + (" " + child if child else "")
        return None
    result = visit(game.start, "", False, False)
    visit.cache_clear()
    return result


def physical_group_baseline(game, current, previous=None):
    """Exact outcome probabilities when every available physical group is equally likely."""
    @cache
    def visit(mask, last, current_hit, previous_hit):
        board = game.board(mask)
        if board.terminal:
            correct = board.terminal == "target_reached" and not current_hit
            persistent = previous is not None and not previous_hit and not correct
            return (float(correct), float(persistent), float(not correct and not persistent))
        totals = [0., 0., 0.]
        for token, move in board.moves.items():
            pair = last + token[0]
            outcome = visit(move.mask, token[0], current_hit or pair == current,
                            previous_hit or pair == previous)
            for index, probability in enumerate(outcome):
                totals[index] += probability / len(board.moves)
        return tuple(totals)
    correct, persistent, ordinary = visit(game.start, "", False, False)
    visit.cache_clear()
    failure = persistent + ordinary
    return {"correct": correct, "persistent_error": persistent if previous else None,
            "ordinary_error": ordinary if previous else None,
            "incorrect": failure,
            "persistent_error_given_incorrect": persistent / failure if previous and failure else None,
            "ordinary_error_given_incorrect": ordinary / failure if previous and failure else None}


def inspect_round(record, previous=None, *, item_type, solution_bounds):
    """Certify solvability, goal diversity, dependencies and diagnostic branches."""
    game = Game(record["input"], record["target"])
    rule, target = record["forbidden"], record["target"]
    if item_type == "same_letter" and alternating_witness(game) is not None:
        return None
    lengths = solution_lengths(game, rule)
    lower, upper, longest = solution_bounds
    if lengths is None or not (lower <= lengths[0] <= upper):
        return None
    if lengths[1] > longest:
        return None
    minima = game.physical_minima()
    if minima is None or minima[0] < 2 or minima[1] < 1:
        return None
    counts, masses = successful_paths(game, rule)
    if sum(counts.values()) < 3 or target and len(masses) < 2:
        return None
    dominance = masses.get("", 0) if target == 0 else max(masses.values()) / sum(masses.values())
    if dominance > .95 + 1e-10:
        return None
    alternative = find_path(game, rule, wrong_goal=True)
    if alternative is None:
        return None
    current_violation = find_path(game, required_pair=rule)
    if current_violation is None:
        return None
    trap = negative = None
    if previous is not None:
        trap = find_path(game, previous, required_pair=rule)
        negative = find_path(game, required_pair=previous, current_failure_pair=rule)
        if trap is None or negative is None:
            return None
    baseline = physical_group_baseline(game, rule, previous)
    if previous is not None and min(baseline["persistent_error_given_incorrect"],
                                    baseline["ordinary_error_given_incorrect"]) < .05 - 1e-12:
        return None
    flat = Game(record["input"].replace(" ", ""), target) if " " in record["input"] else None
    reference = final = None
    for ending in sorted(counts):
        reference = game.final_witness(ending, rule, flat=flat)
        if reference is not None:
            final = ending
            break
    if reference is None:
        return None

    def evidence(path):
        score = game.grade(path, rule, previous)
        return {"moves": path, "answer": score["final_state"],
                "current_violation_steps": score["violation_steps"],
                "previous_violation_steps": score["previous_violation_steps"]}

    result = {"letters": len(record["input"].replace(" ", "")),
              "shortest_success_moves": lengths[0], "longest_success_moves": lengths[1],
              "minimum_merged_deletions": minima[0], "minimum_singleton_merge_deletions": minima[1],
              "initial_choices": len(game.board(game.start).moves),
              "successful_move_sequences": sum(counts.values()), "successful_final_strings": len(counts),
              "successful_letter_multisets": len(masses), "success_probability": sum(masses.values()),
              "maximum_final_share": dominance,
              "reference": {"moves": reference, "answer": final},
              "alternative_goal": evidence(alternative), "current_rule_violation": evidence(current_violation),
              "space_changes_reference_outcome": flat is not None,
              "uniform_physical_group_baseline": baseline,
              "persistent_error": evidence(trap) if trap is not None else None,
              "ordinary_error": evidence(negative) if negative is not None else None}
    if item_type == "same_letter":
        result["repeat_certificate"] = repeat_certificate(record["input"], target, rule)
        require(result["repeat_certificate"]["minimum_same_letter_transitions"] >= 1,
                "same-letter alternation shortcut found")
    if trap is not None:
        require(not result["persistent_error"]["previous_violation_steps"]
                and result["persistent_error"]["current_violation_steps"], "invalid PE witness")
        require(result["ordinary_error"]["previous_violation_steps"], "invalid ordinary-error witness")
    return result
