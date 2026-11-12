"""Build offline dataset reviews without changing existing versioned review pages."""

import argparse
from copy import deepcopy
from functools import cache
import hashlib
import json
from pathlib import Path
import re
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.grade import Game, terminal_group_counts  # noqa: E402


DATA_SCRIPT = re.compile(r'(<script\b[^>]*\bid="data"[^>]*>)(.*?)(</script>)', re.DOTALL)
ROUND_FIELDS = ("round", "input", "target", "forbidden")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value):
    path = Path(value)
    return (PROJECT_DIR / path).resolve() if not path.is_absolute() else path.resolve()


def index_items(document):
    result = {item["id"]: item for item in document["items"]}
    require(len(result) == len(document["items"]), "Duplicate item IDs")
    return result


def physical_witnesses(game):
    """Keep one complete physical deletion path for every reachable final string."""
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


def first_round_metadata(record, answer):
    game = Game(record["input"], record["target"])
    forbidden = record["forbidden"]
    correct = game.final_witness(answer, forbidden)
    require(correct is not None, f"Representative answer is unreachable: {answer!r}")

    # These counts group identical original deletions together, ignoring their
    # ordering. They include wrong goals and forbidden-pair violations.
    counts = terminal_group_counts(game)
    witnesses = physical_witnesses(game)
    require(counts.keys() == witnesses.keys(), "Terminal counts and witnesses disagree")
    outcomes = []
    order = lambda text: (len(text.replace(" ", "")) != game.target,
                          len(text.replace(" ", "")), text)
    for final in sorted(counts, key=order):
        legal = game.final_witness(final, forbidden)
        moves = legal if legal is not None else witnesses[final]
        result = game.grade(moves, forbidden, final_answer=final)
        require(result["board_exhausted"] and result["final_state"] == final,
                f"Incomplete outcome witness: {final!r}")
        outcomes.append({"text": final, "groups": counts[final], "moves": moves})
    require(game.grade(correct, forbidden, final_answer=answer)["valid"],
            "Representative witness failed replay")
    return {
        **record,
        "key": answer,
        "examples": {
            "correct": {"moves": correct, "answer": answer},
            "previousOnly": None,
            "currentOnly": None,
            "both": None,
            "unfinished": None,
        },
        "recoverableSteps": [],
        "outcomes": outcomes,
    }


def both_rules_witness(game, forbidden, previous):
    """Find a complete goal-reaching sequence obeying both adjacent-pair rules."""
    @cache
    def search(mask, last):
        board = game.board(mask)
        if board.terminal:
            return "" if board.terminal == "target_reached" else None
        for token, move in board.moves.items():
            if last + token[0] in {forbidden, previous}:
                continue
            suffix = search(move.mask, token[0])
            if suffix is not None:
                return token + (" " + suffix if suffix else "")
        return None

    result = search(game.start, "")
    search.cache_clear()
    return result


def transition_metadata(record, answer, previous, *, minimum_recoverable=3, allow_common_reference=False):
    metadata = first_round_metadata(record, answer)
    game = Game(record["input"], record["target"])
    forbidden = record["forbidden"]
    flat = Game(record["input"].replace(" ", ""), record["target"]) if " " in record["input"] else None
    current = game.final_witness(answer, forbidden, required_pair=previous, flat=flat)
    if current is None and allow_common_reference:
        current = game.final_witness(answer, forbidden, flat=flat)
    require(current is not None, "Reference must satisfy the current rule and any boundary condition")
    current_score = game.grade(current, forbidden, previous, final_answer=answer)
    require(current_score["valid"] and (allow_common_reference or not current_score["valid_under_previous_rule"]),
            "Current-rule example failed replay")
    probe = game.previous_rule_witness(forbidden, previous)
    require(probe is not None, "Missing previous-rule witness")
    previous_score = game.grade(probe.answer, forbidden, previous)
    recoverable = previous_score["recoverable_violation_steps"]
    require(previous_score["previous_only"] and len(recoverable) >= minimum_recoverable,
            f"Previous-rule witness must retain {minimum_recoverable} recoverable violations")
    both = both_rules_witness(game, forbidden, previous)
    both_example = None
    if both is not None:
        both_score = game.grade(both, forbidden, previous)
        require(both_score["valid"] and both_score["valid_under_previous_rule"],
                "Both-rules example failed replay")
        both_example = {"moves": both, "answer": both_score["final_state"]}
    current_example = {"moves": current, "answer": answer}
    metadata["examples"] = {
        "correct": current_example,
        "previousOnly": {"moves": probe.answer, "answer": previous_score["final_state"]},
        "currentOnly": None if current_score["valid_under_previous_rule"] else current_example,
        "both": both_example,
        "unfinished": None,
    }
    metadata["recoverableSteps"] = recoverable
    return metadata


def build(data_dir, output, variant="easy_r1"):
    require(variant in {"easy_r1", "easy_all", "easy_same_letter_v2"}, "Unknown review variant")
    template_path = PROJECT_DIR / "web" / "review.html"
    require(output != template_path.resolve(), "The original review.html must not be overwritten")
    if variant == "easy_all":
        require(output != (PROJECT_DIR / "web" / "review_easy_r1.html").resolve(),
                "The existing R1 review must not be overwritten by the all-round variant")
    if variant == "easy_same_letter_v2":
        require(output not in {(PROJECT_DIR / "web" / name).resolve()
                               for name in ("review_easy_r1.html", "review_easy_all.html")},
                "Existing dataset reviews must not be overwritten by the same-letter revision")
    original = read_json(PROJECT_DIR / "data" / "problems.json")
    original_keys = index_items(read_json(PROJECT_DIR / "data" / "answer_key.json"))
    problem_path = data_dir / "problems.json"
    problems = read_json(problem_path)
    keys = index_items(read_json(data_dir / "answer_key.json"))
    template = template_path.read_text(encoding="utf-8")
    scripts = list(DATA_SCRIPT.finditer(template))
    require(len(scripts) == 1, "Expected exactly one embedded dataset in original review.html")
    match = scripts[0]
    embedded = json.loads(match.group(2))
    original_items = index_items(original)
    embedded_items = index_items(embedded)
    new_items = index_items(problems)
    require(list(new_items) == list(original_items) == list(embedded_items),
            "Variant, original data and review must have identical item IDs and order")
    require(keys.keys() == original_keys.keys() == original_items.keys(), "Answer-key IDs differ")
    require({k: v for k, v in original.items() if k != "items"} ==
            {k: v for k, v in problems.items() if k != "items"},
            "Only round inputs may change in the variant problem file")
    easy_r1_items = easy_r1_keys = None
    if variant == "easy_all":
        easy_r1_items = index_items(read_json(PROJECT_DIR / "data" / "easy_r1" / "problems.json"))
        easy_r1_keys = index_items(read_json(PROJECT_DIR / "data" / "easy_r1" / "answer_key.json"))
    easy_all_items = easy_all_keys = None
    if variant == "easy_same_letter_v2":
        easy_all_items = index_items(read_json(PROJECT_DIR / "data" / "easy_all" / "problems.json"))
        easy_all_keys = index_items(read_json(PROJECT_DIR / "data" / "easy_all" / "answer_key.json"))

    dataset = deepcopy(embedded)
    for item in dataset["items"]:
        item_id = item["id"]
        base, modified = original_items[item_id], new_items[item_id]
        require(base["type"] == modified["type"] == item["type"], f"Type changed: {item_id}")
        if variant == "easy_same_letter_v2" and item["type"] == "different_letter":
            require(modified == easy_all_items[item_id] and keys[item_id] == easy_all_keys[item_id],
                    f"Different-letter items must match easy_all: {item_id}")
        require(len(base["rounds"]) == len(modified["rounds"]) == len(item["rounds"]) ==
                len(keys[item_id]["rounds"]) == len(original_keys[item_id]["rounds"]) == 5,
                f"Expected five rounds: {item_id}")
        for index, (old_round, new_round, review_round, key, old_key) in enumerate(zip(
                base["rounds"], modified["rounds"], item["rounds"],
                keys[item_id]["rounds"], original_keys[item_id]["rounds"]), start=1):
            label = f"{item_id}/R{index}"
            require({field: review_round[field] for field in ROUND_FIELDS} == old_round and
                    review_round["key"] == old_key["answer"], f"Original review is stale: {label}")
            require(new_round["round"] == key["round"] == index, f"Round numbers differ: {label}")
            if index > 1 and variant == "easy_r1":
                require(new_round == old_round and key == old_key, f"R2–R5 changed: {label}")
            else:
                require({k: v for k, v in new_round.items() if k != "input"} ==
                        {k: v for k, v in old_round.items() if k != "input"},
                        f"Goal or rule changed: {label}")
                require(new_round["input"] != old_round["input"], f"Input did not change: {label}")
                require(len(new_round["input"].replace(" ", "")) ==
                        len(old_round["input"].replace(" ", "")) and
                        new_round["input"].count(" ") == old_round["input"].count(" "),
                        f"Input length or boundary count changed: {label}")
                if variant == "easy_all" and index == 1:
                    require(new_round == easy_r1_items[item_id]["rounds"][0]
                            and key == easy_r1_keys[item_id]["rounds"][0],
                            f"R1 must match the existing easy R1 variant: {label}")
                if variant == "easy_same_letter_v2" and item["type"] == "same_letter":
                    require(new_round["input"] != easy_all_items[item_id]["rounds"][index - 1]["input"],
                            f"Same-letter round must be regenerated: {label}")
                item["rounds"][index - 1] = (first_round_metadata(new_round, key["answer"]) if index == 1
                                             else transition_metadata(new_round, key["answer"],
                                                                      modified["rounds"][index - 2]["forbidden"],
                                                                      minimum_recoverable=(1 if variant == "easy_same_letter_v2"
                                                                                           and item["type"] == "same_letter" else 3),
                                                                      allow_common_reference=(variant == "easy_same_letter_v2"
                                                                                              and item["type"] == "same_letter")))
        description = "R1 계산 완료 · R2–R5 원본 유지" if variant == "easy_r1" else "R1–R5 계산 완료"
        print(f"{item_id}: {description}", flush=True)

    dataset["sha"] = hashlib.sha256(problem_path.read_bytes()).hexdigest()
    dataset["variant"] = variant
    # Only easy_r1 preserves the original transition summary. Other variants
    # recompute it from their witnesses, including any changed selection floor.
    if variant != "easy_r1":
        transitions = [record for item in dataset["items"] for record in item["rounds"][1:]]
        exposures = [len(record["recoverableSteps"]) / len(record["examples"]["previousOnly"]["moves"].split())
                     for record in transitions]
        dataset["summary"] = {
            "items": len(dataset["items"]),
            "rounds": sum(len(item["rounds"]) for item in dataset["items"]),
            "transitions": len(transitions),
            "ruleCompatible": sum(record["examples"]["both"] is not None for record in transitions),
            "previousRuleProbes": sum(record["examples"]["previousOnly"] is not None for record in transitions),
            "minimumProbeRecoverable": min(len(record["recoverableSteps"]) for record in transitions),
            "minimumProbeExposure": min(exposures),
            "maximumProbeExposure": max(exposures),
        }
    encoded = json.dumps(dataset, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = template[:match.start(2)] + encoded + template[match.end(2):]
    variant_label = "R1 완화 버전" if variant == "easy_r1" else "전체 라운드 완화 버전"
    round_label = "R1 완화 버전 · R2–R5 원본 유지" if variant == "easy_r1" else "전체 라운드 완화 버전 · R1–R5"
    if variant == "easy_same_letter_v2":
        variant_label = "같은 문자형 수정 v2"
        round_label = "같은 문자형 125라운드 수정 · 다른 문자형 유지"
    replacements = {
        "<title>SameGame · 문항 검토</title>": f"<title>SameGame · {variant_label} · 문항 검토</title>",
        "</span>SameGame</div>": f"</span>SameGame <span class=\"type-label\">{variant_label}</span></div>",
        '<div class="eyebrow">라운드 검토</div>':
            f'<div class="eyebrow">{round_label}</div>',
        '<div class="eyebrow">데이터셋</div><h1>전체 검증</h1>':
            f'<div class="eyebrow">{variant_label} · 데이터셋</div><h1>전체 검증</h1>',
    }
    if variant != "easy_r1":
        replacements["number(s.previousRuleProbes)+'개 · R2–R5 모든 라운드에 회복 가능한 위반 3회 이상, 해당 수순 행동의 40% 이상인 예가 있습니다.'"] = (
            "number(s.previousRuleProbes)+'개 · R2–R5 모든 라운드에 회복 가능한 위반 '+s.minimumProbeRecoverable+"
            "'회 이상인 예가 있습니다. 해당 수순 행동 중 위반 비율은 '+(100*s.minimumProbeExposure).toFixed(1)+"
            "'–'+(100*s.maximumProbeExposure).toFixed(1)+'%입니다.'")
    for before, after in replacements.items():
        require(page.count(before) == 1, f"Template marker changed: {before}")
        page = page.replace(before, after, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(f"저장: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Variant data directory; defaults to data/<variant>")
    parser.add_argument("--output", help="Standalone review HTML; defaults to web/review_<variant>.html")
    parser.add_argument("--variant", choices=("easy_r1", "easy_all", "easy_same_letter_v2"), default="easy_r1",
                        help="Which dataset variant to validate and render")
    args = parser.parse_args()
    build(project_path(args.data_dir or f"data/{args.variant}"),
          project_path(args.output or f"web/review_{args.variant}.html"), args.variant)


if __name__ == "__main__":
    main()
