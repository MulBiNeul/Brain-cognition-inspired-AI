"""Build and verify an easier first round without changing the original benchmark."""

from copy import deepcopy
from functools import cache
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
import random
from statistics import median

from ._paths import DATA_DIR
from .grade import Game, LETTERS, board_shape, read_round, require, successful_paths


SEED = 2026111119
VARIANT = "easy_r1"
MIN_MOVES, MAX_MOVES = 8, 10


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes(directory):
    return {name: sha256((directory / name).read_bytes()).hexdigest()
            for name in ("problems.json", "answer_key.json")}


def solution_lengths(game, forbidden):
    """Find exact shortest and longest successful plans without enumerating permutations."""
    @cache
    def visit(mask, last):
        board = game.board(mask)
        if board.terminal:
            return (0, 0, "") if board.terminal == "target_reached" else None
        shortest = longest = None
        witness = None
        for token, move in board.moves.items():
            if last + token[0] == forbidden:
                continue
            child = visit(move.mask, token[0])
            if child is None:
                continue
            if shortest is None or child[0] + 1 < shortest:
                shortest = child[0] + 1
                witness = token + (" " + child[2] if child[2] else "")
            longest = max(longest or 0, child[1] + 1)
        return None if shortest is None else (shortest, longest, witness)

    result = visit(game.start, "")
    visit.cache_clear()
    return result


def describe_round(record):
    game = Game(record["input"], record["target"])
    paths = solution_lengths(game, record["forbidden"])
    counts, masses = successful_paths(game, record["forbidden"])
    minima = game.physical_minima()
    return {
        "letters": len(record["input"].replace(" ", "")),
        "shortest_success_moves": paths[0] if paths else None,
        "longest_success_moves": paths[1] if paths else None,
        "minimum_merged_deletions": minima[0] if minima else None,
        "minimum_singleton_merge_deletions": minima[1] if minima else None,
        "initial_choices": len(game.board(game.start).moves),
        "successful_move_sequences": sum(counts.values()),
        "successful_final_strings": len(counts),
        "successful_letter_multisets": len(masses),
    }


def alternative_goal_witness(game, forbidden):
    """Find a legal exhausted board of another candidate goal size (zero to three)."""
    @cache
    def visit(mask, last):
        board = game.board(mask)
        if board.terminal:
            length = len(board.text.replace(" ", ""))
            return "" if length <= 3 and length != game.target else None
        for token, move in board.moves.items():
            if last + token[0] == forbidden:
                continue
            suffix = visit(move.mask, token[0])
            if suffix is not None:
                return token + (" " + suffix if suffix else "")
        return None

    witness = visit(game.start, "")
    visit.cache_clear()
    return witness


def violating_goal_witness(game, forbidden):
    """Find a physical goal-reaching plan that demonstrates the current hidden rule."""
    @cache
    def visit(mask, last):
        board = game.board(mask)
        if board.terminal:
            return (0, "") if board.terminal == "target_reached" else None
        best = None
        for token, move in board.moves.items():
            child = visit(move.mask, token[0])
            if child is None:
                continue
            hits = child[0] + (last + token[0] == forbidden)
            if best is None or hits > best[0]:
                best = hits, token + (" " + child[1] if child[1] else "")
        return best

    result = visit(game.start, "")
    visit.cache_clear()
    return result[1] if result is not None and result[0] else None


def inspect_r1(record, *, minimum_multisets=2, exact=None):
    """Return private diagnostics for a qualifying easy R1, or None for a candidate."""
    game = Game(record["input"], record["target"])
    forbidden, target = record["forbidden"], record["target"]
    lengths = solution_lengths(game, forbidden)
    if lengths is None or not (MIN_MOVES <= lengths[0] <= lengths[1] <= MAX_MOVES):
        return None
    minima = game.physical_minima()
    if minima is None or not (2 <= minima[0] <= 3 and 1 <= minima[1] <= 2):
        return None
    endings = game.legal_endings(forbidden)
    if endings.failing_actions is None or not any(
            finals for length, finals in enumerate(endings.letters_by_length) if length != target):
        return None
    counts, masses = successful_paths(game, forbidden)
    if sum(counts.values()) < 3 or (target and len(masses) < minimum_multisets):
        return None
    dominance = masses.get("", 0) if target == 0 else max(masses.values()) / sum(masses.values())
    if dominance > 0.95 + 1e-10:
        return None
    violating = violating_goal_witness(game, forbidden)
    if violating is None:
        return None
    violating_replay = game.grade(violating, forbidden)
    require(violating_replay["stop_reason"] == "target_reached" and violating_replay["violation_steps"],
            "current-rule witness does not reach the goal with a rule violation")
    flat = Game(record["input"].replace(" ", ""), target) if " " in record["input"] else None
    finals = [exact] if exact is not None else sorted(counts)
    witness = final = None
    for candidate in finals:
        witness = game.final_witness(candidate, forbidden, flat=flat)
        if witness is not None:
            final = candidate
            break
    if witness is None:
        return None
    replay = game.grade(witness, forbidden, final_answer=final)
    require(replay["valid"], "generated reference failed physical replay")
    if flat is not None:
        require(not flat.grade(witness, forbidden)["valid"], "space does not affect reference")
    alternative = alternative_goal_witness(game, forbidden)
    require(alternative is not None, "no alternative candidate goal witness")
    return {
        "letters": len(record["input"].replace(" ", "")),
        "shortest_success_moves": lengths[0], "longest_success_moves": lengths[1],
        "minimum_merged_deletions": minima[0], "minimum_singleton_merge_deletions": minima[1],
        "initial_choices": len(game.board(game.start).moves),
        "successful_move_sequences": sum(counts.values()),
        "successful_final_strings": len(counts), "successful_letter_multisets": len(masses),
        "success_probability": sum(masses.values()), "maximum_final_share": dominance,
        "reachable_goal_sizes": [size for size, finals in enumerate(endings.letters_by_length) if finals],
        "space_changes_reference_outcome": None if flat is None else True,
        "reference": {"moves": witness, "answer": final},
        "alternative_legal_ending": {
            "moves": alternative,
            "answer": game.grade(alternative, forbidden)["final_state"],
        },
        "current_rule_violation_witness": {
            "moves": violating, "answer": violating_replay["final_state"],
            "violation_steps": violating_replay["violation_steps"],
        },
    }


def candidate_tape(rng, size, target, spaced, rich, exact=None):
    """Generate a short topology, then enlarge only groups that are already removable."""
    parts = exact.split(" ") if exact is not None else [""] * (2 if spaced else 1)
    if exact is None:
        for _ in range(target):
            index = rng.randrange(len(parts))
            choices = [c for c in LETTERS if not parts[index].endswith(c)]
            parts[index] += rng.choice(choices)
    if target == 0:
        x, y = rng.sample(LETTERS, 2)
        parts[rng.randrange(len(parts))] = "".join({"X": x, "Y": y}[c] for c in "XXYXXYX")
    else:
        index = rng.choice([i for i, text in enumerate(parts) if text])
        at = rng.randrange(len(parts[index]))
        x = parts[index][at]
        y, z = rng.sample([c for c in LETTERS if c != x], 2)
        pattern = rng.choice(["ZXYYXZZXYYXXY", "XZZXXZYYXXY", "XYYXXYZZXXZ"] if rich else
                             ["XYYXXYXX", "XYYXXYXX", "XXYXXYX", "XYYXXY"])
        core = "".join({"X": x, "Y": y, "Z": z}[c] for c in pattern)
        parts[index] = parts[index][:at] + core + parts[index][at + 1:]

    # At most ten actions can occur on this scaffold. Inflating a run of size
    # >=2 preserves exactly which moves exist and all terminal single letters.
    left = 20 + target - sum(map(len, parts))
    if left < 0 or left == 1:
        return None
    while left:
        width = 3 if left == 3 else 2
        left -= width
        index = rng.randrange(len(parts))
        text = parts[index]
        joins = [i for i in range(1, len(text)) if text[i - 1] == text[i]]
        at = rng.choice(joins) if joins and rng.random() < 0.6 else rng.randrange(len(text) + 1)
        neighbors = {text[at - 1] if at else "", text[at] if at < len(text) else ""}
        choices = [c for c in LETTERS if c not in neighbors]
        missing = [c for c in choices if c not in "".join(parts)]
        letter = rng.choice(missing or choices)
        parts[index] = text[:at] + letter * width + text[at:]
    if any(not text for text in parts) or set("".join(parts)) != set(LETTERS):
        return None
    runs = ["".join(group) for _, group in groupby(" ".join(parts))]
    growable = [i for i, run in enumerate(runs) if run != " " and len(run) > 1]
    for _ in range(size - sum(map(len, parts))):
        index = rng.choice(growable)
        runs[index] += runs[index][0]
    return "".join(runs)


def summarize(rows):
    fields = ("shortest_success_moves", "longest_success_moves", "minimum_merged_deletions",
              "minimum_singleton_merge_deletions", "initial_choices")
    result = {}
    for name in ("original", "easy"):
        result[name] = {field: {"min": min(values), "median": median(values), "max": max(values)}
                        for field in fields for values in [[row[name][field] for row in rows]]}
    return result


def validate_variant(public, answers, source, source_answers, *, progress=None):
    require(len(public["items"]) == len(source["items"]) == 50, "expected 50 items")
    require(set(public) == set(source) and public["rules_en"] == source["rules_en"], "dataset schema changed")
    require(public["rounds_per_item"] == source["rounds_per_item"] == 5, "expected five rounds")
    require(set(answers) == {"items"} and len(answers["items"]) == 50, "invalid answer key")
    shapes = set()
    rows = []
    diversity = {goal: {"rounds": 0, "rounds_with_at_least_three_multisets": 0} for goal in range(4)}
    for item, old, key, old_key in zip(public["items"], source["items"], answers["items"], source_answers["items"]):
        require(set(item) == set(old) and item["id"] == old["id"] and item["type"] == old["type"], "item identity changed")
        require(set(key) == {"id", "rounds"} and key["id"] == item["id"], "answer identity changed")
        require(len(item["rounds"]) == len(key["rounds"]) == 5, "invalid round count")
        require(item["rounds"][1:] == old["rounds"][1:] and key["rounds"][1:] == old_key["rounds"][1:],
                "R2-R5 must remain exactly unchanged")
        for n, record in enumerate(item["rounds"], 1):
            read_round(record, n, item["type"])
            require(set(record) == {"round", "input", "target", "forbidden"}, "round schema changed")
            shape = board_shape(record["input"])
            require(shape not in shapes, "duplicate normalized board")
            shapes.add(shape)
        record, original = item["rounds"][0], old["rounds"][0]
        require(record["input"] != original["input"], "R1 was not replaced")
        require(all(record[k] == original[k] for k in ("round", "target", "forbidden")), "R1 rule/goal changed")
        require(len(record["input"].replace(" ", "")) == len(original["input"].replace(" ", "")), "R1 length changed")
        require(record["input"].count(" ") == original["input"].count(" "), "R1 boundary presence changed")
        baseline = describe_round(original)
        required = min(3, baseline["successful_letter_multisets"]) if record["target"] else 0
        desired = old_key["rounds"][0]["answer"] if old_key["rounds"][0]["answer"] in {"A A", "AB A"} else None
        analysis = inspect_r1(record, minimum_multisets=required, exact=desired)
        require(analysis is not None, f"{item['id']}: easy R1 conditions failed")
        require(set(key["rounds"][0]) == {"round", "answer"} and key["rounds"][0]["round"] == 1,
                "invalid R1 answer key")
        require(key["rounds"][0]["answer"] == analysis["reference"]["answer"], "R1 answer does not match validated reference")
        require(analysis["longest_success_moves"] < baseline["shortest_success_moves"], "R1 plans are not strictly shorter")
        for number, round_record in enumerate(item["rounds"]):
            if number == 0:
                multiset_count = analysis["successful_letter_multisets"]
            else:
                _, masses = successful_paths(Game(round_record["input"], round_record["target"]),
                                             round_record["forbidden"])
                multiset_count = len(masses)
            entry = diversity[round_record["target"]]
            entry["rounds"] += 1
            entry["rounds_with_at_least_three_multisets"] += multiset_count >= 3
        rows.append({"id": item["id"], "type": item["type"], "target": record["target"], "forbidden": record["forbidden"],
                     "original": baseline, "easy": analysis})
        if progress:
            progress(item["id"])
    for goal, entry in diversity.items():
        entry["required_rich_rounds"] = (entry["rounds"] + 3) // 4 if goal else 0
        require(entry["rounds_with_at_least_three_multisets"] >= entry["required_rich_rounds"],
                f"goal {goal}: global solution diversity quota failed")
    return {"variant": VARIANT, "seed": SEED, "changed_r1": len(rows), "unchanged_rounds": 200,
            "limits": {"success_moves_min": MIN_MOVES, "success_moves_max": MAX_MOVES,
                       "minimum_merged_deletions": 2, "minimum_singleton_merge_deletions": 1,
                       "synthetic_previous_rule_in_r1": False},
            "checks": {"same_item_id_type_and_round_schedule": True, "r2_to_r5_and_answers_unchanged": True,
                       "r1_exact_letter_length_and_boundary_presence_preserved": True,
                       "all_250_normalized_boards_distinct": True,
                       "r1_all_physical_successes_require_merges": True,
                       "r1_all_legal_successes_take_8_to_10_moves": True,
                       "r1_all_legal_successes_shorter_than_original_shortest": True,
                       "r1_another_candidate_goal_is_legally_reachable": True,
                       "r1_minimum_three_successful_move_sequences": True,
                       "r1_current_rule_can_be_violated_on_goal_reaching_plan": True,
                       "r1_positive_goal_minimum_two_letter_multisets": True,
                       "r1_dominance_at_most_0_95": True,
                       "r1_spaced_reference_fails_without_boundary": True,
                       "original_literal_A_A_and_AB_A_references_preserved": True},
            "probability_policy": "Choose uniformly among legal letters, then uniformly among that letter's legal groups.",
            "solution_diversity_by_target": diversity,
            "summary": summarize(rows), "items": rows}


def generate(source, source_answers, *, progress=None, maximum_attempts=50000):
    public, answers = deepcopy(source), deepcopy(source_answers)
    seen = {board_shape(record["input"]) for item in source["items"] for record in item["rounds"]}
    attempts_by_item = {}
    for index, (item, key) in enumerate(zip(public["items"], answers["items"])):
        original = item["rounds"][0]
        baseline = describe_round(original)
        rich = original["target"] > 0 and baseline["successful_letter_multisets"] >= 3
        exact = key["rounds"][0]["answer"] if key["rounds"][0]["answer"] in {"A A", "AB A"} else None
        rng = random.Random(SEED + index)
        for attempt in range(1, maximum_attempts + 1):
            tape = candidate_tape(rng, baseline["letters"], original["target"], " " in original["input"], rich, exact)
            if tape is None or board_shape(tape) in seen:
                continue
            candidate = dict(original, input=tape)
            analysis = inspect_r1(candidate, minimum_multisets=3 if rich else 2, exact=exact)
            if analysis is None or analysis["longest_success_moves"] >= baseline["shortest_success_moves"]:
                continue
            item["rounds"][0] = candidate
            key["rounds"][0] = {"round": 1, "answer": analysis["reference"]["answer"]}
            seen.add(board_shape(tape))
            attempts_by_item[item["id"]] = attempt
            if progress:
                progress(item["id"], attempt, analysis)
            break
        else:
            raise ValueError(f"No qualifying easy R1 for {item['id']} after {maximum_attempts} candidates")
    return public, answers, attempts_by_item


def build(output_dir=None, *, validate_only=False):
    output = Path(output_dir or DATA_DIR / VARIANT).resolve()
    require(output != DATA_DIR.resolve(), "the original dataset directory cannot be overwritten")
    before = source_hashes(DATA_DIR)
    source, source_answers = read_json(DATA_DIR / "problems.json"), read_json(DATA_DIR / "answer_key.json")
    attempts = None
    if validate_only:
        public, answers = read_json(output / "problems.json"), read_json(output / "answer_key.json")
    else:
        def progress(identifier, attempt, stats):
            print(f"{identifier}: accepted candidate {attempt}, success moves "
                  f"{stats['shortest_success_moves']}–{stats['longest_success_moves']}, "
                  f"merge minima {stats['minimum_merged_deletions']}/{stats['minimum_singleton_merge_deletions']}", flush=True)
        public, answers, attempts = generate(source, source_answers, progress=progress)
    report = validate_variant(public, answers, source, source_answers)
    report["source_sha256"] = before
    if attempts is not None:
        report["candidate_attempts"] = attempts
    require(source_hashes(DATA_DIR) == before, "original data changed during generation")
    if not validate_only:
        output.mkdir(parents=True, exist_ok=True)
        for name, value in (("problems.json", public), ("answer_key.json", answers), ("design_report.json", report)):
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
