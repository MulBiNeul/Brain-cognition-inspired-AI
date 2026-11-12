"""Generate a separate easier benchmark while preserving hidden-rule transitions."""

from copy import deepcopy
from itertools import groupby
import json
from pathlib import Path
import random

from ._paths import DATA_DIR
from .easy_dataset import (alternative_goal_witness, candidate_tape, describe_round,
                           inspect_r1, read_json, solution_lengths, source_hashes, summarize)
from .grade import Game, LETTERS, board_shape, read_round, require, successful_paths


SEED = 2026111201
VARIANT = "easy_all"


def inflate(tape, size, rng):
    """Enlarge existing removable runs without changing deletion choices or finals."""
    runs = ["".join(group) for _, group in groupby(tape)]
    choices = [i for i, run in enumerate(runs) if run != " " and len(run) >= 2]
    if not choices:
        return None
    for _ in range(size - len(tape.replace(" ", ""))):
        index = rng.choice(choices)
        runs[index] += runs[index][0]
    return "".join(runs)


def reduced_candidate(game, original, rng):
    mask = game.start
    for _ in range(rng.choice((1, 2, 2, 3, 3, 4))):
        moves = game.board(mask).moves
        if not moves:
            break
        mask = moves[rng.choice(tuple(moves))].mask
    tape = game.board(mask).text.strip(" ")
    if set(tape.replace(" ", "")) != set(LETTERS) or tape.count(" ") != original.count(" "):
        return None
    return inflate(tape, len(original.replace(" ", "")), rng)


def inspect_transition(record, previous, baseline, minimum_multisets=2, exact=None):
    game = Game(record["input"], record["target"])
    forbidden, target = record["forbidden"], record["target"]
    lengths = solution_lengths(game, forbidden)
    if lengths is None or not (8 <= lengths[0] <= min(12, baseline["shortest_success_moves"] - 1)):
        return None
    minima = game.physical_minima()
    if minima is None or minima[0] < 2 or minima[1] < 1:
        return None
    counts, masses = successful_paths(game, forbidden)
    if sum(counts.values()) < 3 or (target and len(masses) < minimum_multisets):
        return None
    dominance = masses.get("", 0) if target == 0 else max(masses.values()) / sum(masses.values())
    if dominance > 0.95 + 1e-10:
        return None
    alternative = alternative_goal_witness(game, forbidden)
    if alternative is None:
        return None
    probe = game.previous_rule_witness(forbidden, previous)
    if probe is None or probe.recoverable < 3:
        return None
    old_replay = game.grade(probe.answer, forbidden, previous)
    require(old_replay["previous_only"], "old-rule witness must succeed only under the old rule")
    require(probe.recoverable == len(old_replay["recoverable_violation_steps"]), "probe replay mismatch")
    flat = Game(record["input"].replace(" ", ""), target) if " " in record["input"] else None
    finals = [exact] if exact is not None else sorted(counts)
    reference = final = None
    for value in finals:
        reference = game.final_witness(value, forbidden, required_pair=previous, flat=flat)
        if reference is not None:
            final = value
            break
    if reference is None:
        return None
    replay = game.grade(reference, forbidden, previous, final_answer=final)
    require(replay["valid"] and replay["valid_under_previous_rule"] is False, "reference must distinguish rules")
    if flat is not None:
        require(not flat.grade(reference, forbidden)["valid"], "reference must depend on space boundary")
    endings = game.legal_endings(forbidden)
    return {
        "letters": len(record["input"].replace(" ", "")),
        "shortest_success_moves": lengths[0], "longest_success_moves": lengths[1],
        "minimum_merged_deletions": minima[0], "minimum_singleton_merge_deletions": minima[1],
        "initial_choices": len(game.board(game.start).moves),
        "successful_move_sequences": sum(counts.values()), "successful_final_strings": len(counts),
        "successful_letter_multisets": len(masses), "success_probability": sum(masses.values()),
        "maximum_final_share": dominance,
        "reachable_goal_sizes": [size for size, values in enumerate(endings.letters_by_length) if values],
        "space_changes_reference_outcome": None if flat is None else True,
        "reference": {"moves": reference, "answer": final},
        "alternative_legal_ending": {"moves": alternative,
                                     "answer": game.grade(alternative, forbidden)["final_state"]},
        "previous_rule_probe": {
            "moves": probe.answer, "answer": old_replay["final_state"],
            "violation_steps": old_replay["violation_steps"],
            "recoverable_violation_steps": old_replay["recoverable_violation_steps"],
            "executed_steps": old_replay["executed_steps"],
            "recoverable_fraction": probe.recoverable / old_replay["executed_steps"],
        },
    }


def generate(source, source_keys, first_rounds, first_keys, *, maximum_attempts=10000):
    public, keys = deepcopy(source), deepcopy(source_keys)
    seen = {board_shape(r["input"]) for item in source["items"] for r in item["rounds"]}
    seen.update(board_shape(item["rounds"][0]["input"]) for item in first_rounds["items"])
    rows, attempts = [], {}
    for index, (item, key, first, first_key) in enumerate(zip(
            public["items"], keys["items"], first_rounds["items"], first_keys["items"])):
        item["rounds"][0] = deepcopy(first["rounds"][0])
        key["rounds"][0] = deepcopy(first_key["rounds"][0])
        for number in range(2, 6):
            original = source["items"][index]["rounds"][number - 1]
            baseline = describe_round(original)
            original_game = Game(original["input"], original["target"])
            previous = item["rounds"][number - 2]["forbidden"]
            required = min(3, baseline["successful_letter_multisets"]) if original["target"] else 0
            exact = key["rounds"][number - 1]["answer"] if key["rounds"][number - 1]["answer"] in {"A A", "AB A"} else None
            rng = random.Random(SEED + 5 * index + number)
            preferred_shortest = min(10, baseline["shortest_success_moves"] -
                                     (2 if baseline["shortest_success_moves"] >= 12 else 1))
            for attempt in range(1, maximum_attempts + 1):
                if attempt % 4:
                    tape = reduced_candidate(original_game, original["input"], rng)
                else:
                    tape = candidate_tape(rng, baseline["letters"], original["target"],
                                          " " in original["input"], required >= 3, exact)
                if tape is None or board_shape(tape) in seen:
                    continue
                candidate = dict(original, input=tape)
                analysis = inspect_transition(candidate, previous, baseline, required, exact)
                if analysis is None:
                    continue
                if attempt <= 100 and analysis["shortest_success_moves"] > preferred_shortest:
                    continue
                item["rounds"][number - 1] = candidate
                key["rounds"][number - 1] = {"round": number, "answer": analysis["reference"]["answer"]}
                seen.add(board_shape(tape))
                attempts[f"{item['id']}/R{number}"] = attempt
                rows.append({"id": item["id"], "round": number, "type": item["type"],
                             "target": original["target"], "forbidden": original["forbidden"],
                             "previous_forbidden": previous, "original": baseline, "easy": analysis})
                print(f"{item['id']}/R{number}: candidate {attempt}, shortest "
                      f"{baseline['shortest_success_moves']}->{analysis['shortest_success_moves']}, "
                      f"recoverable {len(analysis['previous_rule_probe']['recoverable_violation_steps'])}/"
                      f"{analysis['previous_rule_probe']['executed_steps']}", flush=True)
                break
            else:
                raise ValueError(f"No qualifying board for {item['id']}/R{number} after {maximum_attempts} candidates")
    return public, keys, rows, attempts


def validate_variant(public, keys, source, source_keys, first_rounds, first_keys):
    require(len(public["items"]) == len(keys["items"]) == len(source["items"]) == 50, "expected 50 items")
    require(set(keys) == {"items"}, "invalid answer-key schema")
    require({k: v for k, v in public.items() if k != "items"} ==
            {k: v for k, v in source.items() if k != "items"}, "public schema changed")
    shapes, rows = set(), []
    diversity = {goal: {"rounds": 0, "rich_rounds": 0} for goal in range(4)}
    for item, key, old, old_key, first, first_key in zip(public["items"], keys["items"], source["items"],
            source_keys["items"], first_rounds["items"], first_keys["items"]):
        require(item["id"] == key["id"] == old["id"] == first["id"], "item identity changed")
        require(set(item) == set(old) and set(key) == {"id", "rounds"}, "invalid item schema")
        require(item["type"] == old["type"] and len(item["rounds"]) == len(key["rounds"]) == 5, "invalid item")
        require(item["rounds"][0] == first["rounds"][0] and key["rounds"][0] == first_key["rounds"][0],
                "easy R1 must remain unchanged")
        for record, answer, original, original_answer in zip(item["rounds"], key["rounds"], old["rounds"], old_key["rounds"]):
            number = record["round"]
            require(set(record) == {"round", "input", "target", "forbidden"}, "invalid round schema")
            require(set(answer) == {"round", "answer"} and answer["round"] == number, "invalid answer schema")
            read_round(record, number, item["type"])
            require(all(record[k] == original[k] for k in ("round", "target", "forbidden")), "rule/goal schedule changed")
            require(len(record["input"].replace(" ", "")) == len(original["input"].replace(" ", "")), "length changed")
            require(record["input"].count(" ") == original["input"].count(" "), "space structure changed")
            shape = board_shape(record["input"])
            require(shape not in shapes, "duplicate normalized board")
            shapes.add(shape)
            baseline = describe_round(original)
            required = min(3, baseline["successful_letter_multisets"]) if record["target"] else 0
            exact = original_answer["answer"] if original_answer["answer"] in {"A A", "AB A"} else None
            previous = item["rounds"][number - 2]["forbidden"] if number > 1 else None
            analysis = (inspect_transition(record, previous, baseline, required, exact) if previous is not None
                        else inspect_r1(record, minimum_multisets=required, exact=exact))
            require(analysis is not None, f"{item['id']}/R{number}: validation failed")
            require(answer["answer"] == analysis["reference"]["answer"], "answer does not match reference")
            require(record["input"] != original["input"], "original round was not replaced")
            require(analysis["shortest_success_moves"] < baseline["shortest_success_moves"], "shortest plan is not easier")
            entry = diversity[record["target"]]
            entry["rounds"] += 1
            entry["rich_rounds"] += analysis["successful_letter_multisets"] >= 3
            rows.append({"id": item["id"], "round": number, "type": item["type"], "target": record["target"],
                         "forbidden": record["forbidden"], "previous_forbidden": previous,
                         "original": baseline, "easy": analysis})
    for target, entry in diversity.items():
        entry["required_rich_rounds"] = (entry["rounds"] + 3) // 4 if target else 0
        require(entry["rich_rounds"] >= entry["required_rich_rounds"], "global final-answer diversity quota failed")
    transitions = [row for row in rows if row["round"] > 1]
    return {"variant": VARIANT, "seed": SEED, "changed_from_original": 250,
            "r1_copied_from_easy_r1": 50, "regenerated_r2_to_r5": 200,
            "selection_policy": {"shortest_success_moves_min": 8, "shortest_success_moves_max": 12,
                                 "minimum_merged_deletions": 2, "minimum_singleton_merge_deletions": 1,
                                 "minimum_recoverable_old_rule_violations": 3,
                                 "minimum_recoverable_action_fraction": None,
                                 "per_mpe_scoring_changed": False},
            "summary": summarize(rows), "transition_summary": summarize(transitions),
            "solution_diversity_by_target": diversity, "items": rows}


def build(output_dir=None, *, validate_only=False):
    output = Path(output_dir or DATA_DIR / VARIANT).resolve()
    require(output not in {DATA_DIR.resolve(), (DATA_DIR / "easy_r1").resolve()}, "existing datasets cannot be overwritten")
    original_hashes, first_hashes = source_hashes(DATA_DIR), source_hashes(DATA_DIR / "easy_r1")
    source, source_keys = read_json(DATA_DIR / "problems.json"), read_json(DATA_DIR / "answer_key.json")
    first, first_keys = read_json(DATA_DIR / "easy_r1" / "problems.json"), read_json(DATA_DIR / "easy_r1" / "answer_key.json")
    attempts = None
    if validate_only:
        public, keys = read_json(output / "problems.json"), read_json(output / "answer_key.json")
    else:
        public, keys, _, attempts = generate(source, source_keys, first, first_keys)
    report = validate_variant(public, keys, source, source_keys, first, first_keys)
    report["source_sha256"] = original_hashes
    report["easy_r1_sha256"] = first_hashes
    if attempts is not None:
        report["candidate_attempts"] = attempts
    require(source_hashes(DATA_DIR) == original_hashes and source_hashes(DATA_DIR / "easy_r1") == first_hashes,
            "existing datasets changed")
    if not validate_only:
        output.mkdir(parents=True, exist_ok=True)
        for filename, value in (("problems.json", public), ("answer_key.json", keys), ("design_report.json", report)):
            (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
