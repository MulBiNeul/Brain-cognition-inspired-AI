"""Build the offline researcher review for SameGame problems."""

import argparse
from collections import Counter
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.grade import BOARD, Game, terminal_group_counts  # noqa: E402


TEMPLATE = PROJECT_DIR / "scripts/templates/review.html"
SCORING = PROJECT_DIR / "scripts/templates/scoring.js"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value):
    path = Path(value)
    return (PROJECT_DIR / path).resolve() if not path.is_absolute() else path.resolve()


def physical_witnesses(game):
    """Keep one complete physical deletion path for each reachable final string."""
    witnesses = {}
    seen = set()
    pending = [(game.start, ())]
    while pending:
        mask, path = pending.pop()
        if mask in seen:
            continue
        seen.add(mask)
        board = game.board(mask)
        if not board.moves:
            witnesses.setdefault(board.text.strip(" "), " ".join(path))
        else:
            pending.extend((move.mask, path + (token,))
                           for token, move in reversed(tuple(board.moves.items())))
    return witnesses


def diagnostic_witness(game, forbidden, previous, kind):
    """Find a complete trace with the specified current/previous-rule relation."""
    require(kind in {"previousOnly", "generalError", "both", "currentOnly"},
            f"Unknown example type: {kind}")

    @cache
    def search(mask, last, current_hit, previous_hit):
        board = game.board(mask)
        if board.terminal:
            goal_met = board.terminal == "target_reached"
            current_correct = goal_met and not current_hit
            matches = {
                "previousOnly": goal_met and current_hit and not previous_hit,
                "generalError": not current_correct and previous_hit,
                "both": current_correct and not previous_hit,
                "currentOnly": current_correct and previous_hit,
            }
            return "" if matches[kind] else None
        for token, move in board.moves.items():
            pair = last + token[0]
            hit_current = current_hit or pair == forbidden
            hit_previous = previous_hit or pair == previous
            if kind in {"both", "currentOnly"} and hit_current:
                continue
            if kind in {"both", "previousOnly"} and hit_previous:
                continue
            suffix = search(move.mask, token[0], hit_current, hit_previous)
            if suffix is not None:
                return token + (" " + suffix if suffix else "")
        return None

    moves = search(game.start, "", False, False)
    search.cache_clear()
    if moves is None:
        return None
    score = game.grade(moves, forbidden, previous)
    require(score["board_exhausted"] and score["first_basic_error_step"] is None,
            f"Incomplete {kind} example")
    return {"moves": moves, "answer": score["final_state"]}


def case_metadata(case, number, forbidden, previous):
    """Describe complete outcomes and diagnostic paths for one input condition."""
    tape, target = case["input"], case["target"]
    require(isinstance(tape, str) and BOARD.fullmatch(tape), "Invalid input string")
    require(type(target) is int and target in range(4), "Target must be 0, 1, 2 or 3")
    game = Game(tape, target)
    counts = terminal_group_counts(game)
    witnesses = physical_witnesses(game)
    require(counts.keys() == witnesses.keys(), "Terminal counts and witnesses disagree")
    outcomes = []
    correct = None
    for final in sorted(counts, key=lambda text: (len(text.replace(" ", "")) != target,
                                                len(text.replace(" ", "")), text)):
        legal = game.final_witness(final, forbidden)
        moves = legal if legal is not None else witnesses[final]
        score = game.grade(moves, forbidden, previous, final_answer=final)
        require(score["board_exhausted"] and score["final_state"] == final,
                "Incomplete final-outcome witness")
        outcomes.append({"text": final, "groups": counts[final], "moves": moves,
                         "correct": legal is not None})
        if correct is None and legal is not None:
            correct = {"moves": legal, "answer": final}
    require(correct is not None, f"No current answer for case {case['id']}")
    examples = {"correct": correct}
    if previous is not None:
        for name in ("previousOnly", "generalError", "both", "currentOnly"):
            examples[name] = diagnostic_witness(game, forbidden, previous, name)
    return {**case, "round": number, "forbidden": forbidden, "key": correct["answer"],
            "examples": examples, "outcomes": outcomes}


def scoring_javascript():
    """Read the browser's single-submission replay implementation."""
    return SCORING.read_text(encoding="utf-8")


def build(data_dir, output, label="문제 검토"):
    data_dir, output = Path(data_dir), Path(output)
    problem_path = data_dir / "problems.json"
    document = read_json(problem_path)
    require(document.get("version") == 1, "Unsupported problem schema")
    require(document.get("items"), "The dataset has no items")
    report_path = data_dir / "design_report.json"
    report = read_json(report_path) if report_path.exists() else {}
    certificates = {(row["id"], row["round"], row["case_id"]): row
                    for row in report.get("case_certificates", [])}
    if report.get("dataset_sha256"):
        key = read_json(data_dir / "answer_key.json")
        canonical = json.dumps({"problems": document, "answer_key": key}, sort_keys=True,
                               ensure_ascii=False, separators=(",", ":"))
        require(hashlib.sha256(canonical.encode()).hexdigest() == report["dataset_sha256"],
                "Generator report does not match the selected problem and answer files")
    items, item_ids = [], set()
    for item in document["items"]:
        require(item["id"] not in item_ids, "Duplicate item ID")
        item_ids.add(item["id"])
        require(item["type"] in {"different_letter", "same_letter"}, "Unknown rule type")
        rounds, previous = [], None
        for number, record in enumerate(item["rounds"], 1):
            require(record["round"] == number, "Round numbers must be consecutive")
            forbidden = record["forbidden"]
            require(isinstance(forbidden, str) and len(forbidden) == 2
                    and set(forbidden) <= set("ABCDE"), "Invalid forbidden pair")
            require(record.get("cases"), "Each round needs cases")
            case_ids, cases = set(), []
            for case in record["cases"]:
                require(isinstance(case["id"], str) and case["id"] not in case_ids,
                        "Case IDs must be unique strings within their round")
                case_ids.add(case["id"])
                metadata = case_metadata(case, number, forbidden, previous)
                certificate = certificates.get((item["id"], number, case["id"]))
                if certificate is not None:
                    require(certificate["target"] == case["target"]
                            and certificate["forbidden"] == forbidden, "Case certificate differs from problem")
                    fields = ("shortest_success_moves", "longest_success_moves", "minimum_merged_deletions",
                              "minimum_singleton_merge_deletions", "initial_choices", "successful_letter_multisets")
                    metadata["certificate"] = {field: certificate[field] for field in fields if field in certificate}
                cases.append(metadata)
            require(all(any(other["target"] != case["target"] and other["input"] != case["input"]
                            for other in cases) for case in cases),
                    "Every case needs a next case with a different input and target")
            rounds.append({"round": number, "forbidden": forbidden, "cases": cases})
            previous = forbidden
        require(len(rounds) == document["rounds_per_item"], "Round count differs from dataset metadata")
        items.append({"id": item["id"], "type": item["type"], "rounds": rounds})
        print(f"  {item['id']} · {len(rounds)} rounds · {sum(len(r['cases']) for r in rounds)} cases", flush=True)
    rounds = [r for item in items for r in item["rounds"]]
    cases = [c for r in rounds for c in r["cases"]]
    transitions = [c for item in items for r in item["rounds"][1:] for c in r["cases"]]
    dataset = {
        "version": 1, "label": label, "seed": document.get("seed"),
        "casesPerTarget": document.get("cases_per_target"), "items": items,
        "sha": hashlib.sha256(problem_path.read_bytes()).hexdigest(),
        "reportHref": os.path.relpath(report_path.resolve(), output.parent.resolve()) if report_path.exists() else None,
        "summary": {
            "items": len(items), "rounds": len(rounds), "cases": len(cases),
            "transitions": len(transitions), "transitionRounds": sum(len(i["rounds"])-1 for i in items),
            "casesPerTarget": {str(target): count for target, count in
                               sorted(Counter(c["target"] for c in cases).items())},
            "ruleCompatible": sum(c["examples"].get("both") is not None for c in transitions),
            "previousRuleProbes": sum(c["examples"].get("previousOnly") is not None for c in transitions),
            "generalErrorProbes": sum(c["examples"].get("generalError") is not None for c in transitions),
        },
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    require(template.count("__DATA__") == template.count("__SCORING_JAVASCRIPT__") == 1,
            "Expected one dataset and one shared scoring placeholder")
    serialized = json.dumps(dataset, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("__DATA__", serialized).replace("__SCORING_JAVASCRIPT__", scoring_javascript())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Review: {output}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="web/review.html")
    parser.add_argument("--label", default="문제 검토")
    args = parser.parse_args()
    build(project_path(args.data_dir), project_path(args.output), args.label)


if __name__ == "__main__":
    main()
