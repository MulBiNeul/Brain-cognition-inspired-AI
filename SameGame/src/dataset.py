"""Generate fresh, certified input pools for trials with changing hidden goals.

The evaluator's grading and persistent-error formulas are deliberately reused,
not reimplemented here. A round fixes its rule and supplies three distinct
inputs for each goal. Selecting a case between actual submissions belongs to
the evaluation session, not to this offline generator.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import random
from statistics import mean, median
import tempfile

from ._paths import DATA_DIR
from .dataset_checks import inspect_round, mechanical_shape
from .grade import LETTERS, read_round, require


SEED = 2026111301
KINDS = ("different_letter", "same_letter")
MAX_STRUCTURE_USES = 2


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _compact(tape):
    return "".join(letter * min(2, len(list(run))) for letter, run in groupby(tape))


def inflate(tape, size, rng):
    runs = ["".join(group) for _, group in groupby(tape)]
    choices = [i for i, run in enumerate(runs) if run != " " and len(run) >= 2]
    if not choices or len(tape.replace(" ", "")) > size:
        return None
    for _ in range(size - len(tape.replace(" ", ""))):
        index = rng.choice(choices)
        runs[index] += runs[index][0]
    return "".join(runs)


def renamed(tape, current, previous, source_current, source_previous, rng):
    mapping = {}
    for before, after in ((source_current, current), (source_previous or "", previous or "")):
        for source, destination in zip(before, after):
            if source in mapping and mapping[source] != destination:
                return None
            mapping[source] = destination
    if len(set(mapping.values())) != len(mapping):
        return None
    available = [letter for letter in LETTERS if letter not in mapping.values()]
    rng.shuffle(available)
    for letter in LETTERS:
        if letter not in mapping:
            mapping[letter] = available.pop()
    return tape.translate(str.maketrans(mapping))


@lru_cache(maxsize=1)
def _seed_structures():
    document = json.loads((DATA_DIR / "seed_structures.json").read_text(encoding="utf-8"))
    sources = document["structures"]
    return ({target: [s for s in sources if s["target"] == target] for target in range(4)},
            {mechanical_shape(s["input"]) for s in sources})


def rule_schedule(rng, kind):
    """Only adjacent rules must differ; both directed endpoints may change."""
    pool = [a + b for a in LETTERS for b in LETTERS
            if (a == b) == (kind == "same_letter")]
    rules = []
    for _ in range(5):
        rules.append(rng.choice([r for r in pool if not rules or r != rules[-1]]))
    return rules


def _insert_groups(rng, target, spaced):
    """Construct a board from a terminal seed through random inverse deletions.

    No fixed X/Y pattern is used. Triple insertions and splits within existing
    groups create alternative outcomes and dependencies; exact checks below
    determine whether the resulting board has the required properties.
    """
    parts = [""] * (2 if spaced else 1)
    for _ in range(target):
        n = rng.randrange(len(parts))
        parts[n] += rng.choice([c for c in LETTERS if not parts[n].endswith(c)])
    budget = rng.choice((22, 23, 24, 25, 26)) - target
    while budget > 0:
        width = 3 if budget == 3 or budget > 3 and rng.random() < .15 else 2
        if budget < width or budget - width == 1:
            width = budget
        budget -= width
        n = rng.randrange(len(parts))
        part = parts[n]
        splits = [i for i in range(1, len(part)) if part[i - 1] == part[i]]
        at = rng.choice(splits) if splits and rng.random() < .7 else rng.randrange(len(part) + 1)
        neighbors = {part[at - 1] if at else "", part[at] if at < len(part) else ""}
        choices = [c for c in LETTERS if c not in neighbors]
        missing = [c for c in choices if c not in "".join(parts)]
        letter = rng.choice(missing or choices)
        parts[n] = part[:at] + letter * width + part[at:]
    return " ".join(parts) if all(parts) else None


def _mutate_structure(tape, rng):
    """Move, recolor or insert whole groups, changing the deletion graph."""
    runs = ["".join(group) for _, group in groupby(_compact(tape))]
    removable = [i for i, run in enumerate(runs) if run != " " and len(run) >= 2]
    if not removable:
        return None, "none"
    operation = rng.choices(("relocate", "replace_and_relocate", "insert", "recolor"),
                            weights=(4, 3, 2, 1))[0]
    if operation == "insert":
        block = rng.choice(LETTERS) * 2
    elif operation == "recolor":
        at = rng.choice(removable)
        runs[at] = rng.choice([c for c in LETTERS if c != runs[at][0]]) * 2
        return "".join(runs), operation
    else:
        block = runs.pop(rng.choice(removable))
        if operation == "replace_and_relocate":
            block = rng.choice(LETTERS) * 2
    text = "".join(runs)
    at = rng.randrange(len(text) + 1)
    return text[:at] + block + text[at:], operation


def solution_bounds(kind, number):
    if kind == "different_letter":
        return 8, 10, 11
    return (9, 11, 12) if number == 1 else (9, 12, 13)


def inspect_case(case, forbidden, previous, kind, number):
    row = {"round": number, "input": case["input"], "target": case["target"], "forbidden": forbidden}
    diagnostic = inspect_round(row, previous, item_type=kind,
                               solution_bounds=solution_bounds(kind, number))
    if diagnostic is None:
        return None
    p = diagnostic["uniform_physical_group_baseline"]["correct"]
    return diagnostic if .06 <= p <= .60 else None


def _make_case(job):
    seed, item_id, kind, number, forbidden, previous, target, ordinal, retry = job
    rng = random.Random(f"{seed}:{item_id}:{number}:{target}:{ordinal}:{retry}")
    size = 28 + (int(item_id.split("-")[-1]) + number + target * 3 + ordinal) % 5
    sources, seed_shapes = _seed_structures() if kind == "same_letter" else ({}, set())
    rejected = Counter()
    for attempt in range(1, 30001):
        if kind == "different_letter":
            core = _insert_groups(rng, target, rng.random() < .25)
            construction = {"method": "inverse_group_insertions"}
        else:
            source = rng.choice(sources[target])
            core = renamed(source["input"], forbidden, previous, source["forbidden"],
                           source.get("previous") if previous else None, rng)
            operations = []
            for _ in range(2 if rng.random() < .25 else 1):
                core, operation = _mutate_structure(core, rng)
                operations.append(operation)
                if core is None:
                    break
            construction = {"method": "structural_mutation", "operations": operations,
                            "source_structure_sha256": _digest(mechanical_shape(source["input"]))}
        if core is None or set(core.replace(" ", "")) != set(LETTERS):
            rejected["alphabet_or_empty_segment"] += 1
            continue
        core = core.strip(" ")
        if "  " in core:
            rejected["boundary"] += 1
            continue
        shape = mechanical_shape(core)
        if shape in seed_shapes:
            rejected["unchanged_seed_structure"] += 1
            continue
        tape = inflate(core, size, rng)
        if tape is None:
            rejected["length"] += 1
            continue
        case = {"id": f"c{target * 3 + ordinal:02d}", "input": tape, "target": target}
        diagnostic = inspect_case(case, forbidden, previous, kind, number)
        if diagnostic is None:
            rejected["design_constraints"] += 1
            continue
        return {"case": case, "diagnostic": diagnostic, "construction": construction,
                "search_attempts": attempt, "rejections": dict(rejected)}
    raise ValueError(f"No qualifying case for {item_id}/R{number}/goal{target}/{ordinal}: {dict(rejected)}")


@lru_cache(maxsize=1)
def _cache_signature():
    paths = [Path(__file__), Path(__file__).with_name("dataset_checks.py"),
             Path(__file__).with_name("grade.py"), DATA_DIR / "seed_structures.json"]
    return _digest([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])


def _cached_case(job):
    """Persist expensive candidate searches outside the repository.

    The code and seed-library fingerprint invalidates stale results. Atomic
    writes also let an interrupted 300-item build reuse completed worker jobs.
    """
    directory = Path(tempfile.gettempdir()) / "samegame-candidates" / _cache_signature()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (_digest(job) + ".json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    result = _make_case(job)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return result


def _jobs(public, seed):
    for item in public["items"]:
        previous = None
        for row in item["rounds"]:
            for target in range(4):
                for ordinal in range(1, 4):
                    yield (seed, item["id"], item["type"], row["round"], row["forbidden"],
                           previous, target, ordinal, 0)
            previous = row["forbidden"]


def design(seed=SEED, count=50):
    """Build extensible schedules without generating the expensive cases."""
    require(isinstance(count, int) and count > 0 and count % 2 == 0, "count must be a positive even number")
    rng = random.Random(seed)
    public = {"version": 1, "seed": seed, "rounds_per_item": 5, "cases_per_target": 3, "items": []}
    for index in range(1, count + 1):
        # Extend in balanced fifty-item batches: --count 300 preserves the
        # initial fifty IDs, rule schedules and case-generation seeds.
        batch_start = ((index - 1) // 50) * 50
        batch_size = min(50, count - batch_start)
        kind = KINDS[int(index - batch_start > batch_size // 2)]
        public["items"].append({"id": f"SG-{index:03d}", "type": kind,
                                "rounds": [{"round": n, "forbidden": rule, "cases": []}
                                           for n, rule in enumerate(rule_schedule(rng, kind), 1)]})
    return public


def generate(seed=SEED, count=50, workers=4):
    public = design(seed, count)
    jobs = list(_jobs(public, seed))
    usage, round_shapes, verified = Counter(), {}, {}
    key = {"version": 1, "items": [{"id": i["id"], "rounds": [{"round": r["round"], "cases": []}
                                                                         for r in i["rounds"]]}
                                   for i in public["items"]]}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for job, result in zip(jobs, executor.map(_cached_case, jobs, chunksize=1)):
            local_shapes = round_shapes.setdefault((job[1], job[3]), set())
            retry = 0
            while (usage[mechanical_shape(result["case"]["input"])] >= MAX_STRUCTURE_USES
                   or mechanical_shape(result["case"]["input"]) in local_shapes):
                retry += 1
                require(retry <= 1000, f"Unique structural search exhausted for {job[1]}/R{job[3]}; expand the mutation search")
                result = _cached_case((*job[:-1], retry))
            shape = mechanical_shape(result["case"]["input"])
            usage[shape] += 1
            local_shapes.add(shape)
            _, item_id, kind, number, rule, previous, _, _, _ = job
            index = int(item_id.split("-")[-1]) - 1
            case = result["case"]
            public["items"][index]["rounds"][number - 1]["cases"].append(case)
            key["items"][index]["rounds"][number - 1]["cases"].append(
                {"id": case["id"], "answer": result["diagnostic"]["reference"]["answer"]})
            verified[item_id, number, case["id"]] = result
            if case["id"] == "c12":
                print(f"{item_id}/R{number}: 12 cases certified", flush=True)
    return public, key, validate(public, key, verified=verified)


def validate(public, key, *, verified=None):
    """Check schema and exact case certificates, preserving the shared scorer."""
    require(set(public) == {"version", "seed", "rounds_per_item", "cases_per_target", "items"}, "invalid dataset fields")
    require(public["version"] == 1 and public["rounds_per_item"] == 5 and public["cases_per_target"] == 3,
            "unsupported dataset design")
    count = len(public["items"])
    require(count > 0 and count % 2 == 0, "expected balanced types")
    require(set(key) == {"version", "items"} and key["version"] == 1, "invalid key schema")
    require(len(key["items"]) == count, "key item count differs")
    require(all(set(item) == {"id", "rounds"} for item in key["items"]), "invalid key item fields")
    keyed = {item["id"]: item for item in key["items"]}
    require(len(keyed) == count, "duplicate key IDs")
    rows, usage, kinds = [], Counter(), Counter()
    _, seed_shapes = _seed_structures()
    for index, item in enumerate(public["items"], 1):
        item_id, kind = item["id"], item["type"]
        require(set(item) == {"id", "type", "rounds"} and item_id == f"SG-{index:03d}" and kind in KINDS,
                "invalid problem identity")
        require(len(item["rounds"]) == len(keyed[item_id]["rounds"]) == 5, "expected five rounds")
        kinds[kind] += 1
        previous = None
        for number, (row, answer_row) in enumerate(zip(item["rounds"], keyed[item_id]["rounds"]), 1):
            rule = row["forbidden"]
            require(set(row) == {"round", "forbidden", "cases"} and row["round"] == number, "invalid round")
            require(rule != previous, "adjacent rules must differ")
            require(set(answer_row) == {"round", "cases"} and answer_row["round"] == number, "invalid key round")
            require(len(row["cases"]) == len(answer_row["cases"]) == 12, "expected twelve cases")
            answers = {c["id"]: c["answer"] for c in answer_row["cases"]}
            require(len(answers) == 12 and all(set(c) == {"id", "answer"} for c in answer_row["cases"]), "invalid key cases")
            require(Counter(c["target"] for c in row["cases"]) == {t: 3 for t in range(4)}, "expected three inputs per target")
            require({c["id"] for c in row["cases"]} == {f"c{n:02d}" for n in range(1, 13)}, "invalid case IDs")
            local_shapes = set()
            for case in row["cases"]:
                location = item_id, number, case["id"]
                require(set(case) == {"id", "input", "target"}, f"{location}: invalid case")
                record = {"round": number, "input": case["input"], "target": case["target"], "forbidden": rule}
                game = read_round(record, number, kind)
                shape = mechanical_shape(case["input"])
                require(shape not in local_shapes, f"{location}: repeated mechanical structure within a round")
                require(usage[shape] < MAX_STRUCTURE_USES, f"{location}: mechanical structure overused")
                require(kind != "same_letter" or shape not in seed_shapes,
                        f"{location}: an unchanged seed structure cannot be an output case")
                usage[shape] += 1
                local_shapes.add(shape)
                generated = verified.get(location) if verified is not None else None
                require(generated is None or generated["case"] == case, "certificate/input mismatch")
                diagnostic = generated["diagnostic"] if generated else inspect_case(case, rule, previous, kind, number)
                require(diagnostic is not None, f"{location}: unsatisfied design constraints")
                reference = diagnostic["reference"]
                require(answers[case["id"]] == reference["answer"], "key differs from certified reference")
                score = game.grade(reference["moves"], rule, previous, final_answer=reference["answer"])
                require(score["valid"], "invalid reference replay")
                for name in ("persistent_error", "ordinary_error"):
                    witness = diagnostic[name]
                    if previous is None:
                        require(witness is None, "R1 has no prior-rule witness")
                        continue
                    require(witness is not None, f"missing {name} witness")
                    result = game.grade(witness["moves"], rule, previous, final_answer=witness["answer"])
                    require(result["first_basic_error_step"] is None and result["board_exhausted"]
                            and result["final_matches_trace"] and not result["valid"], "invalid failure witness")
                    require(bool(result["previous_violation_steps"]) == (name == "ordinary_error"),
                            "wrong previous-rule witness classification")
                rows.append({"id": item_id, "type": kind, "round": number, "case_id": case["id"],
                             "target": case["target"], "forbidden": rule, "previous_forbidden": previous,
                             **diagnostic})
            previous = rule
    require(kinds == {kind: count // 2 for kind in KINDS}, "rule types must be balanced")
    summaries = {}
    for kind in KINDS:
        selected = [r for r in rows if r["type"] == kind]
        summaries[kind] = {"cases": len(selected),
                           "shortest_moves_mean": mean(r["shortest_success_moves"] for r in selected),
                           "shortest_moves_median": median(r["shortest_success_moves"] for r in selected),
                           "shortest_moves_histogram": dict(sorted(Counter(r["shortest_success_moves"] for r in selected).items())),
                           "uniform_physical_group_correct_probability_mean": mean(r["uniform_physical_group_baseline"]["correct"] for r in selected)}
    return {"valid": True, "version": 1, "dataset_sha256": _digest({"problems": public, "answer_key": key}),
            "items": count, "rounds": count * 5, "cases": len(rows), "type_counts": dict(kinds),
            "cases_per_target": 3, "hidden_goal_and_rule": True,
            "rule_schedule_constraint": "adjacent rounds differ; nonadjacent repeats are allowed",
            "difficulty": "default",
            "solution_bounds": {kind: {"round_1": solution_bounds(kind, 1), "later_rounds": solution_bounds(kind, 2)} for kind in KINDS},
            "input_lengths": dict(sorted(Counter(r["letters"] for r in rows).items())),
            "targets": dict(sorted(Counter(r["target"] for r in rows).items())),
            "spaced_cases": sum(r["space_changes_reference_outcome"] for r in rows),
            "unique_mechanical_structures": len(usage), "maximum_mechanical_structure_uses": max(usage.values()),
            "maximum_mechanical_structure_uses_allowed": MAX_STRUCTURE_USES,
            "same_letter_outputs_reusing_seed_structures": 0,
            "nonadjacent_rule_repeat_items": sum(len({r["forbidden"] for r in item["rounds"]}) < 5 for item in public["items"]),
            "different_letter_transitions_changing_both_endpoints": sum(
                all(a != b for a, b in zip(before["forbidden"], after["forbidden"]))
                for item in public["items"] if item["type"] == "different_letter"
                for before, after in zip(item["rounds"], item["rounds"][1:])),
            "transition_cases_with_persistent_and_ordinary_errors": sum(r["previous_forbidden"] is not None for r in rows),
            "same_letter_cases_without_alternating_goal_paths": sum(r["type"] == "same_letter" for r in rows),
            "difficulty_summary": summaries, "case_certificates": rows}
