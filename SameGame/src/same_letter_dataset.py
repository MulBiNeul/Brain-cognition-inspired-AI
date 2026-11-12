"""Generate and certify same-letter rounds that require a repeated deletion letter."""

from copy import deepcopy
from collections import Counter
from functools import cache
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
import random
from statistics import median

from ._paths import DATA_DIR
from .easy_dataset import (alternative_goal_witness, describe_round, read_json, solution_lengths,
                           source_hashes)
from .easy_all_dataset import inflate
from .grade import Game, LETTERS, board_shape, read_round, require, successful_paths

SEED = 2026111203
VARIANT = "easy_same_letter_v2"


def repeat_certificate(tape, target, forbidden):
    """Minimize repeated deletion letters over every current-legal successful plan.

    The state includes the last deletion letter. A ``None`` minimum means there
    is no successful current-legal plan. For a same-letter prohibition, a zero
    minimum is exactly the all-letter-alternation shortcut being excluded.
    """
    game = Game(tape, target)

    @cache
    def visit(mask, last):
        board = game.board(mask)
        if board.terminal:
            return (0, "") if board.terminal == "target_reached" else None
        best = None
        for token, move in board.moves.items():
            letter = token[0]
            if last + letter == forbidden:
                continue
            child = visit(move.mask, letter)
            if child is None:
                continue
            cost = child[0] + int(last == letter)
            if best is None or cost < best[0]:
                best = cost, token + (" " + child[1] if child[1] else "")
                if cost == 0:
                    break
        return best

    result = visit(game.start, "")
    states = visit.cache_info().currsize
    visit.cache_clear()
    return {
        "minimum_same_letter_transitions": None if result is None else result[0],
        "witness": None if result is None else result[1],
        "alternating_solution_exists": result is not None and result[0] == 0,
        "explored_states": states,
    }


def alternating_witness(game):
    """Find any goal-reaching path with every adjacent deletion letter different."""
    @cache
    def visit(mask, last):
        board = game.board(mask)
        if board.terminal:
            return "" if board.terminal == "target_reached" else None
        for token, move in board.moves.items():
            if token[0] == last:
                continue
            suffix = visit(move.mask, token[0])
            if suffix is not None:
                return token + (" " + suffix if suffix else "")
        return None

    result = visit(game.start, "")
    visit.cache_clear()
    return result


def inspect_round(record, previous=None, *, first=False, minimum_multisets=2):
    """Exact private checks, including absence of *every* alternating solution."""
    game = Game(record["input"], record["target"])
    forbidden, target = record["forbidden"], record["target"]
    if alternating_witness(game) is not None:
        return None
    lengths = solution_lengths(game, forbidden)
    if lengths is None or not (8 <= lengths[0] <= (10 if first else 12)):
        return None
    r1_maximum = 11 if target == 1 and " " in record["input"] else 10
    if lengths[1] > (r1_maximum if first else 13):
        return None
    minima = game.physical_minima()
    if minima is None or minima[0] < 2 or minima[1] < 1:
        return None
    counts, masses = successful_paths(game, forbidden)
    if sum(counts.values()) < 3 or target and len(masses) < minimum_multisets:
        return None
    dominance = masses.get("", 0) if target == 0 else max(masses.values()) / sum(masses.values())
    if dominance > .95 + 1e-10:
        return None
    alternative = alternative_goal_witness(game, forbidden)
    if alternative is None:
        return None
    probe = game.previous_rule_witness(forbidden, previous or "")
    if probe is None or probe.recoverable < 1:
        return None
    old_replay = game.grade(probe.answer, forbidden, previous)
    flat = Game(record["input"].replace(" ", ""), target) if " " in record["input"] else None
    reference = final = None
    current_only = None
    if previous is not None:
        for value in sorted(counts):
            current_only = game.final_witness(value, forbidden, required_pair=previous, flat=flat)
            if current_only is not None:
                reference, final = current_only, value
                break
    if reference is None:
        for value in sorted(counts):
            reference = game.final_witness(value, forbidden, flat=flat)
            if reference is not None:
                final = value
                break
    if reference is None:
        return None
    require(game.grade(reference, forbidden, final_answer=final)["valid"], "invalid reference")
    if flat is not None:
        require(not flat.grade(reference, forbidden)["valid"], "space must affect reference")
    certificate = repeat_certificate(record["input"], target, forbidden)
    require(certificate["minimum_same_letter_transitions"] >= 1, "alternating solution survives")
    endings = game.legal_endings(forbidden)
    result = {
        "letters": len(record["input"].replace(" ", "")),
        "shortest_success_moves": lengths[0], "longest_success_moves": lengths[1],
        "minimum_merged_deletions": minima[0], "minimum_singleton_merge_deletions": minima[1],
        "initial_choices": len(game.board(game.start).moves),
        "successful_move_sequences": sum(counts.values()), "successful_final_strings": len(counts),
        "successful_final_strings_with_space": sum(" " in final for final in counts),
        "successful_letter_multisets": len(masses), "success_probability": sum(masses.values()),
        "maximum_final_share": dominance,
        "reachable_goal_sizes": [size for size, values in enumerate(endings.letters_by_length) if values],
        "space_changes_reference_outcome": None if flat is None else True,
        "reference": {"moves": reference, "answer": final},
        "alternative_legal_ending": {"moves": alternative, "answer": game.grade(alternative, forbidden)["final_state"]},
        "repeat_certificate": certificate,
        "current_only_available": current_only is not None,
    }
    detail = {"moves": probe.answer, "answer": old_replay["final_state"],
              "violation_steps": old_replay["violation_steps"],
              "recoverable_violation_steps": old_replay["recoverable_violation_steps"],
              "executed_steps": old_replay["executed_steps"],
              "recoverable_fraction": probe.recoverable / old_replay["executed_steps"]}
    if previous is None:
        result["current_rule_violation_witness"] = detail
    else:
        require(old_replay["previous_only"], "probe is not previous-rule-only")
        result["previous_rule_probe"] = detail
    return result


def inverse_candidate(rng, target, spaced=False):
    """Reverse-insert a mixture of pairs and an odd group to create dependencies.

    Canonical C is the prohibited repeated deletion letter; A is a high-count
    alternative letter. A triple B insertion permits alternative terminal goals.
    Missing letters are inserted, while letters already in the intended ending
    do not have to be given an additional independent removable group.
    """
    parts = [""] * (2 if spaced else 1)
    for _ in range(target):
        at = rng.randrange(len(parts))
        parts[at] += rng.choice([c for c in LETTERS if not parts[at].endswith(c)])
    tape = "".join(parts)
    letters = list("A" * rng.choice((3, 4, 5, 6)) + "B" + "C" * rng.choice((2, 2, 3, 4))
                   + ("D" if "D" not in tape or rng.random() < .3 else "")
                   + ("E" if "E" not in tape or rng.random() < .3 else ""))
    if 2 * len(letters) + target + 1 > 30:
        return None
    rng.shuffle(letters)
    for letter in letters:
        choices = []
        for index, text in enumerate(parts):
            choices.extend((index, at) for at in range(len(text) + 1)
                           if (at == 0 or text[at - 1] != letter)
                           and (at == len(text) or text[at] != letter))
        if not choices:
            return None
        inside = [(index, at) for index, at in choices
                  if 0 < at < len(parts[index]) and parts[index][at - 1] == parts[index][at]]
        index, at = rng.choice(inside if inside and rng.random() < .85 else choices)
        parts[index] = parts[index][:at] + letter * (3 if letter == "B" else 2) + parts[index][at:]
    if any(not text for text in parts) or set("".join(parts)) != set(LETTERS):
        return None
    return " ".join(parts)


def canonical_tape(tape):
    """Cap removable original runs at two; deletion choices and terminals persist."""
    return "".join(letter * min(2, len(tuple(group))) if letter != " " else " "
                   for letter, group in groupby(tape))


def make_catalog(bank):
    """Build verified canonical topologies and meaningful boundary variants."""
    catalog = {(target, spaced): [] for target in range(4) for spaced in (False, True)}
    seen = set()
    for entry in bank["items"]:
        core = canonical_tape(entry["input"])
        target = entry["target"]
        tapes = [core, core[::-1]]
        if " " not in core:
            tapes += ["AA " + core, core + " AA", "AA " + core[::-1], core[::-1] + " AA"]
        for tape in tapes:
            identity = (tape, target)
            if identity in seen:
                continue
            seen.add(identity)
            record = {"input": tape, "target": target, "forbidden": "CC"}
            analysis = inspect_round(record, "BB")
            if analysis is not None:
                catalog[(target, " " in tape)].append((record, analysis))
    return catalog


def renamed(record, current, previous, rng):
    mapping = {"C": current[0]}
    if previous is not None:
        mapping["B"] = previous[0]
    available = [letter for letter in LETTERS if letter not in mapping.values()]
    rng.shuffle(available)
    for letter in LETTERS:
        if letter not in mapping:
            mapping[letter] = available.pop()
    return record["input"].translate(str.maketrans(mapping))


def generate(source, source_keys, bank):
    public, keys = deepcopy(source), deepcopy(source_keys)
    catalog = make_catalog(bank)
    print("Verified template catalog: " + ", ".join(
        f"goal {goal}, {'space' if spaced else 'no space'}: {len(values)}"
        for (goal, spaced), values in catalog.items()), flush=True)
    seen = {board_shape(r["input"]) for item in source["items"] for r in item["rounds"]}
    uses, template_uses = {}, {}
    missing_first_round = []
    for item in source["items"]:
        if item["type"] != "same_letter":
            continue
        first = item["rounds"][0]
        spaced = " " in first["input"]
        maximum = 11 if first["target"] == 1 and spaced else 10
        eligible = [value for _, value in catalog[(first["target"], spaced)]
                    if value["shortest_success_moves"] <= 10 and value["longest_success_moves"] <= maximum]
        if not eligible:
            missing_first_round.append(f"{item['id']} (goal {first['target']}, space {spaced})")
    require(not missing_first_round, "Missing R1 templates: " + ", ".join(missing_first_round))
    for item, key in zip(public["items"], keys["items"]):
        if item["type"] != "same_letter":
            continue
        for index, old in enumerate(item["rounds"]):
            number = old["round"]
            previous = item["rounds"][index - 1]["forbidden"] if index else None
            rng = random.Random(SEED + int(item["id"].split("-")[-1]) * 10 + number)
            candidates = list(catalog[(old["target"], " " in old["input"])])
            rng.shuffle(candidates)
            candidates.sort(key=lambda pair: (
                template_uses.get(board_shape(pair[0]["input"]), 0),
                -pair[1]["successful_letter_multisets"],
                -len(pair[1]["previous_rule_probe"]["recoverable_violation_steps"]),
                pair[1]["shortest_success_moves"], pair[1]["longest_success_moves"]))
            chosen = None
            for canonical, canonical_analysis in candidates:
                if number == 1:
                    maximum = 11 if old["target"] == 1 and " " in old["input"] else 10
                    if (canonical_analysis["shortest_success_moves"] > 10
                            or canonical_analysis["longest_success_moves"] > maximum):
                        continue
                if len(canonical["input"].replace(" ", "")) > len(old["input"].replace(" ", "")):
                    continue
                for _ in range(12):
                    tape = renamed(canonical, old["forbidden"], previous, rng)
                    tape = inflate(tape, len(old["input"].replace(" ", "")), rng)
                    if board_shape(tape) in seen:
                        continue
                    candidate = dict(old, input=tape)
                    analysis = inspect_round(candidate, previous, first=number == 1)
                    if analysis is not None:
                        chosen = candidate, analysis, canonical
                        break
                if chosen is not None:
                    break
            require(chosen is not None, f"No qualifying template for {item['id']}/R{number}, "
                    f"target {old['target']}, spaced {' ' in old['input']}")
            candidate, analysis, canonical = chosen
            item["rounds"][index] = candidate
            key["rounds"][index] = {"round": number, "answer": analysis["reference"]["answer"]}
            seen.add(board_shape(candidate["input"]))
            template_shape = board_shape(canonical["input"])
            template_uses[template_shape] = template_uses.get(template_shape, 0) + 1
            uses[f"{item['id']}/R{number}"] = canonical["input"]
            print(f"{item['id']}/R{number}: {analysis['shortest_success_moves']}–"
                  f"{analysis['longest_success_moves']} moves, minimum repeated transitions "
                  f"{analysis['repeat_certificate']['minimum_same_letter_transitions']}", flush=True)
    return public, keys, {"templates_by_round": uses,
                          "distinct_template_shapes": len(template_uses),
                          "maximum_template_reuse": max(template_uses.values())}


def validate_variant(public, keys, source, source_keys, source_report):
    require(len(public["items"]) == len(keys["items"]) == len(source["items"]) == 50, "expected 50 items")
    require({k: v for k, v in public.items() if k != "items"} ==
            {k: v for k, v in source.items() if k != "items"}, "public schema changed")
    require(set(keys) == {"items"}, "answer-key schema changed")
    old_rows = {(row["id"], row["round"]): row for row in source_report["items"]}
    shapes, rows, changed = set(), [], []
    diversity = {goal: {"rounds": 0, "rich_rounds": 0} for goal in range(4)}
    for item, key, before, before_key in zip(public["items"], keys["items"], source["items"], source_keys["items"]):
        require(item["id"] == key["id"] == before["id"] == before_key["id"], "item identity changed")
        require(set(item) == set(before) and set(key) == {"id", "rounds"}, "item schema changed")
        require(item["type"] == before["type"] and len(item["rounds"]) == len(key["rounds"]) == 5, "invalid item")
        if item["type"] == "different_letter":
            require(item == before and key == before_key, "different-letter data must remain identical")
        for record, answer, old in zip(item["rounds"], key["rounds"], before["rounds"]):
            number = record["round"]
            require(set(record) == {"round", "input", "target", "forbidden"}, "round schema changed")
            require(set(answer) == {"round", "answer"} and answer["round"] == number, "answer schema changed")
            read_round(record, number, item["type"])
            require(all(record[k] == old[k] for k in ("round", "target", "forbidden")), "rule/goal schedule changed")
            require(len(record["input"].replace(" ", "")) == len(old["input"].replace(" ", "")), "letter length changed")
            require(record["input"].count(" ") == old["input"].count(" "), "space presence changed")
            shape = board_shape(record["input"])
            require(shape not in shapes, "duplicate normalized input")
            shapes.add(shape)
            if item["type"] == "same_letter":
                require(record["input"] != old["input"], "same-letter round not replaced")
                previous = item["rounds"][number - 2]["forbidden"] if number > 1 else None
                analysis = inspect_round(record, previous, first=number == 1)
                require(analysis is not None, f"{item['id']}/R{number}: design validation failed")
                require(answer["answer"] == analysis["reference"]["answer"], "representative answer mismatch")
                row = {"id": item["id"], "round": number, "type": item["type"], "target": record["target"],
                       "forbidden": record["forbidden"], "previous_forbidden": previous,
                       "original": describe_round(old), "easy": analysis}
                changed.append(row)
            else:
                row = deepcopy(old_rows[(item["id"], number)])
                analysis = row["easy"]
            diversity[record["target"]]["rounds"] += 1
            diversity[record["target"]]["rich_rounds"] += analysis["successful_letter_multisets"] >= 3
            rows.append(row)
    require(len(changed) == 125, "expected 125 revised rounds")
    later = [row for row in changed if row["round"] > 1]
    summary = {}
    for phase in ("original", "easy"):
        summary[phase] = {}
        for field in ("shortest_success_moves", "longest_success_moves", "minimum_merged_deletions",
                      "minimum_singleton_merge_deletions", "initial_choices"):
            values = [row[phase][field] for row in changed]
            summary[phase][field] = {"min": min(values), "median": median(values), "max": max(values)}
    for goal, entry in diversity.items():
        entry["previous_quarter_quota"] = (entry["rounds"] + 3) // 4 if goal else 0
        entry["quarter_quota_met"] = entry["rich_rounds"] >= entry["previous_quarter_quota"]
    exception_rows = [{"id": row["id"], "round": row["round"], "target": row["target"],
                       "longest_success_moves": row["easy"]["longest_success_moves"]}
                      for row in changed if row["round"] == 1 and row["easy"]["longest_success_moves"] > 10]
    return {"variant": VARIANT, "seed": SEED, "changed_from_easy_all": 125,
            "unchanged_different_letter_problems": 25, "revised_same_letter_problems": 25,
            "selection_policy": {"all_same_letter_rounds_require_repeated_deletion_letter": True,
                "shortest_success_moves_min": 8, "r1_shortest_success_moves_max": 10,
                "r1_longest_success_moves_max": 10, "r1_goal1_spaced_longest_exception_max": 11,
                "later_shortest_success_moves_max": 12, "later_longest_success_moves_max": 13,
                "minimum_merged_deletions": 2, "minimum_singleton_merge_deletions": 1,
                "r1_merge_count_upper_bounds_removed": True, "minimum_positive_goal_letter_multisets": 2,
                "three_multiset_quarter_quota_enforced": False, "minimum_recoverable_old_rule_violations": 1,
                "previous_minimum_recoverable_old_rule_violations": 3, "current_only_reference_required": False,
                "per_mpe_scoring_changed": False},
            "summary": summary, "r1_longest_exceptions": exception_rows,
            "repeated_transition_minimum_distribution": dict(Counter(
                str(row['easy']['repeat_certificate']['minimum_same_letter_transitions']) for row in changed)),
            "recoverable_old_rule_violation_distribution": dict(Counter(
                str(len(row['easy']['previous_rule_probe']['recoverable_violation_steps'])) for row in later)),
            "current_only_available_rounds": sum(row["easy"]["current_only_available"] for row in later),
            "revised_rounds_with_spaced_successful_final": sum(
                row["easy"]["successful_final_strings_with_space"] > 0 for row in changed),
            "solution_diversity_by_target": diversity, "items": rows}


def build(output_dir=None, *, validate_only=False):
    output = Path(output_dir or DATA_DIR / VARIANT).resolve()
    require(output not in {DATA_DIR.resolve(), (DATA_DIR / "easy_r1").resolve(), (DATA_DIR / "easy_all").resolve()},
            "existing datasets cannot be overwritten")
    source_dir = DATA_DIR / "easy_all"
    before = source_hashes(source_dir)
    source = read_json(source_dir / "problems.json")
    source_keys = read_json(source_dir / "answer_key.json")
    source_report = read_json(source_dir / "design_report.json")
    bank_path = DATA_DIR / VARIANT / "candidate_bank.json"
    bank = read_json(bank_path)
    selection = None
    if validate_only:
        public, keys = read_json(output / "problems.json"), read_json(output / "answer_key.json")
    else:
        public, keys, selection = generate(source, source_keys, bank)
    report = validate_variant(public, keys, source, source_keys, source_report)
    report["source_sha256"] = before
    report["candidate_bank_sha256"] = sha256(bank_path.read_bytes()).hexdigest()
    if selection is not None:
        report["selection"] = selection
    require(source_hashes(source_dir) == before, "existing easy_all dataset changed")
    if not validate_only:
        output.mkdir(parents=True, exist_ok=True)
        for name, value in (("problems.json", public), ("answer_key.json", keys), ("design_report.json", report)):
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        subset = output / "same_letter"
        subset.mkdir(exist_ok=True)
        subpublic = dict(public, items=[item for item in public["items"] if item["type"] == "same_letter"])
        identifiers = {item["id"] for item in subpublic["items"]}
        subkeys = {"items": [item for item in keys["items"] if item["id"] in identifiers]}
        for name, value in (("problems.json", subpublic), ("answer_key.json", subkeys)):
            (subset / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
