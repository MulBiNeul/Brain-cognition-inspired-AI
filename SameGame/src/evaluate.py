"""Run the four-way feedback experiment over JSON lines, without a model provider."""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

from ._paths import DATA_DIR
from .grade import Game, parse_moves, read_round


INSTRUCTIONS = (
    "This is a letter-deletion puzzle. Use the feedback after each attempt to solve the puzzle.\n\n"
    "Use only the instructions and conversation. Do not use tools, files, code execution, "
    "external information, or other agents.\n\n"
    "Start each attempt from the given string. Delete an entire group of two or more identical letters "
    "next to each other. The remaining letters join, but never across spaces. Newly formed groups "
    "require another move. Continue until no identical letters remain next to each other.\n\n"
    "For each letter, number its deletable groups from left to right across the whole string. "
    "Write A2 for the second A-group. Recount after each deletion.\n\n"
    "Submit only JSON in this format:\n"
    "{\"moves\": \"<deletion moves>\", \"answer\": \"<remaining string>\"}\n\n"
    "List your moves in order, separated by spaces. Write the actual remaining string as \"answer\", "
    "preserving internal spaces and removing outer spaces. Use \"\" if nothing remains.\n\n"
    "Examples of deletion mechanics and JSON formatting:\n\n"
    "Example 1\n"
    "Initial string: \"ABBBA\"\n"
    "Trace: \"ABBBA\" --B1--> \"AA\" --A1--> \"\"\n"
    "B1 removes all three B letters. The joined A letters need a separate move.\n"
    "Submission: {\"moves\": \"B1 A1\", \"answer\": \"\"}\n\n"
    "Example 2\n"
    "Initial string: \"ACAADAAEAAB\"\n"
    "Trace: \"ACAADAAEAAB\" --A2--> \"ACAADEAAB\" --A2--> \"ACAADEB\" --A1--> \"ACDEB\"\n"
    "The single A at the start is not numbered. After the first deletion, the last AA becomes A2.\n"
    "Submission: {\"moves\": \"A2 A2 A1\", \"answer\": \"ACDEB\"}\n\n"
    "Example 3\n"
    "Initial string: \"ABB ABB\"\n"
    "Trace: \"ABB ABB\" --B2--> \"ABB A\" --B1--> \"A A\"\n"
    "B2 selects the BB after the space. The remaining A letters stay separated by the space.\n"
    "Submission: {\"moves\": \"B2 B1\", \"answer\": \"A A\"}"
)
MAX_MOVES = 30
RULE_OUTCOMES = {(True, True): "both", (True, False): "current_only",
                 (False, True): "previous_only", (False, False): "neither"}


def grade_response(raw, record, previous=None):
    """Replay the whole submitted plan. Keep attempted and executed actions distinct."""
    try:
        submission = json.loads(raw)
        tokens = (parse_moves(submission["moves"]) if isinstance(submission, dict)
                  and set(submission) == {"answer", "moves"} and isinstance(submission["answer"], str) else ())
    except (ValueError, TypeError):
        tokens = ()
    parsed = 1 <= len(tokens) <= MAX_MOVES
    valid_plan = False
    final_matches = physical_goal = goal = None
    exhausted = None
    previous_success = None if previous is None else False
    violations, recoverable = [], []
    executed, failed_step = 0, None
    final_state = None
    if parsed:
        game = Game(record["input"], record["target"])
        result = game.grade(submission["moves"], record["forbidden"], previous, final_answer=submission["answer"])
        executed, failed_step = result["executed_steps"], result["first_basic_error_step"]
        final_state = result["final_state"]
        exhausted = result["board_exhausted"]
        violations, recoverable = result["violation_steps"], result["recoverable_violation_steps"]
        previous_success = result["valid_under_previous_rule"]
        valid_plan = failed_step is None
        if valid_plan:
            physical_goal = result["stop_reason"] == "target_reached"
            final_matches = result["final_matches_trace"]
            goal = physical_goal and final_matches

    correct = bool(goal and not violations)
    if correct:
        feedback = "Correct"
    elif violations:
        feedback = "Correct, but rule violation" if goal else "Incorrect, also rule violation"
        # Steps are 1-based submitted moves; a forbidden pair is attributed to
        # its second move. Keep every violation below for private scoring only.
        feedback += f". Rule violation: move {violations[0]}."
    else:
        feedback = "Incorrect, but no rule violation"
    return {
        "feedback": feedback, "correct": correct, "parse_valid": parsed, "base_valid": valid_plan,
        "moves": " ".join(tokens) if parsed else None,
        "answer": submission["answer"].strip(" ") if parsed else None,
        "physical_goal_reached": physical_goal, "final_matches": final_matches, "goal_reached": goal,
        "final_state": final_state, "board_exhausted": exhausted, "executed_steps": executed,
        "first_basic_error_step": failed_step,
        "violation_steps": violations, "recoverable_violation_steps": recoverable,
        "persistent_error": None if previous is None else bool(violations),
        "previous_success": previous_success,
        "previous_only": None if previous is None else previous_success and not correct,
        "rule_outcome": None if previous is None else RULE_OUTCOMES[correct, previous_success],
    }


def copy_match(signature, references):
    if signature is None or not references:
        return {"moves": None, "response": None}
    return {"moves": any(moves == signature[0] for moves, _ in references),
            "response": signature in references}


def round_summary(number, records, status):
    count = len(records)
    errors = sum(not record["correct"] for record in records)
    persistent = None if number == 1 else sum(record["persistent_error"] for record in records)
    return {"round": number, "status": status, "submissions": count, "errors": errors,
            "rule_outcomes": {s: sum(r["rule_outcome"] == s for r in records) for s in RULE_OUTCOMES.values()},
            "persistent_errors": persistent, "per": persistent / count if number > 1 and count else None,
            "error_rate": errors / count if count else None}


def observed_mean(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return mean(values) if values else None


def metrics(problems):
    rounds = [r for problem in problems for r in problem["rounds"]]
    outcomes = {s: sum(r["rule_outcomes"][s] for r in rounds) for s in RULE_OUTCOMES.values()}
    return {"scheduled_problems": len(problems),
            "mpe_eligible_problems": sum(problem["mpe"] is not None for problem in problems),
            "mpe": observed_mean(problems, "mpe"),
            "error_eligible_problems": sum(problem["error_rate"] is not None for problem in problems),
            "error_rate": observed_mean(problems, "error_rate"),
            "rule_outcomes": {"denominator": sum(outcomes.values()), "counts": outcomes}}


class EvaluationSession:
    """One scoring protocol shared by model requests, human play, and JSONL replay."""

    def __init__(self, items, *, max_attempts=10, required_streak=2):
        if not 1 <= required_streak <= max_attempts:
            raise ValueError("require 1 <= required_streak <= max_attempts")
        self.items = items
        self.max_attempts = max_attempts
        self.required_streak = required_streak
        self.item_index = 0
        self.round_index = 0
        self.responses = []
        self.records = []
        self._problems = []
        self._start_item()

    @property
    def complete(self):
        return self.item_index >= len(self.items)

    @property
    def current_item(self):
        return None if self.complete else self.items[self.item_index]

    @property
    def current_record(self):
        return None if self.complete else self.current_item["rounds"][self.round_index]

    @property
    def attempt(self):
        return len(self._attempts) + 1

    def _start_item(self):
        self.round_index = 0
        self._history = [{"role": "system", "content": INSTRUCTIONS}]
        self._rounds = []
        self._attempts = []
        self._streak = 0
        self._last_submission = None
        self._seen = set()
        self._successes = set()
        self._previous_successes = set()

    def messages(self):
        if self.complete:
            return []
        prompt = "Initial string: " + json.dumps(self.current_record["input"])
        return [dict(message) for message in self._history] + [{"role": "user", "content": prompt}]

    def submit(self, raw):
        if self.complete:
            raise ValueError("this evaluation is already complete")
        if not isinstance(raw, str):
            raise ValueError("a submission must be a string")
        if not raw.strip():
            self.responses.append(raw)
            self._finish_round("generation_failure")
            return None
        record = self.current_record
        previous = self.current_item["rounds"][self.round_index-1]["forbidden"] if self.round_index else None
        score = grade_response(raw, record, previous)
        signature = (score["moves"], score["answer"]) if score["parse_valid"] else None
        copies = {
            "previous_submission": copy_match(signature, [self._last_submission] if self._last_submission else []),
            "same_round_success": copy_match(signature, self._successes),
            "previous_round_success": copy_match(signature, self._previous_successes),
            "any_in_problem": copy_match(signature, self._seen),
        }
        self._streak = self._streak + 1 if score["correct"] else 0
        row = {"id": self.current_item["id"], "type": self.current_item["type"],
               "round": record["round"], "attempt": self.attempt, "raw": raw,
               "success_streak": self._streak, "input": record["input"], "copy": copies, **score}
        self._history = self.messages()
        self._history.extend([{"role": "assistant", "content": raw},
                              {"role": "user", "content": score["feedback"]}])
        self._last_submission = signature
        if signature is not None:
            self._seen.add(signature)
        if score["correct"]:
            self._successes.add(signature)
        self.responses.append(raw)
        self.records.append(row)
        self._attempts.append(row)
        if self._streak >= self.required_streak:
            self._finish_round("success")
        elif len(self._attempts) >= self.max_attempts:
            self._finish_round("exhausted")
        return row

    def _finish_round(self, status):
        self._rounds.append(round_summary(self.current_record["round"], self._attempts, status))
        self.round_index += 1
        if status in ("exhausted", "generation_failure"):
            for record in self.current_item["rounds"][self.round_index:]:
                self._rounds.append(round_summary(record["round"], [], "excluded_after_" + status))
            self.round_index = len(self.current_item["rounds"])
        if self.round_index == len(self.current_item["rounds"]):
            self._problems.append(self._problem_report(self.current_item, self._rounds))
            self.item_index += 1
            self._start_item()
        else:
            self._previous_successes = self._successes
            self._successes = set()
            self._attempts = []
            self._streak = 0

    @staticmethod
    def _problem_report(item, rounds):
        return {"id": item["id"], "type": item["type"], "rounds": rounds,
                "mpe": observed_mean(rounds, "per"), "error_rate": observed_mean(rounds, "error_rate")}

    def report(self):
        problems = list(self._problems)
        for index in range(self.item_index, len(self.items)):
            item = self.items[index]
            rounds = list(self._rounds) if index == self.item_index else []
            for record in item["rounds"][len(rounds):]:
                active = index == self.item_index and record is self.current_record
                rounds.append(round_summary(record["round"], self._attempts if active else [],
                                            "incomplete" if active else "unattempted"))
            problems.append(self._problem_report(item, rounds))
        types = sorted({item["type"] for item in self.items})
        return {"complete": self.complete, "max_attempts": self.max_attempts,
                "required_streak": self.required_streak, "summary": metrics(problems),
                "by_type": {kind: metrics([p for p in problems if p["type"] == kind]) for kind in types},
                "problems": problems, "submissions": list(self.records)}


def load_items(data_dir, item_ids=None, limit=None):
    """Load and check round fields without exposing evaluator metadata to callers' prompts."""
    path = Path(data_dir)
    if path.suffix != ".json":
        path = path / "problems.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("the dataset must contain at least one item")
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item IDs in the dataset")
    for item in items:
        if not item["rounds"]:
            raise ValueError("each item must contain rounds")
        for number, record in enumerate(item["rounds"], 1):
            read_round(record, number, item["type"])
    if item_ids:
        unknown = set(item_ids) - set(ids)
        if unknown:
            raise ValueError("unknown item(s): " + ", ".join(sorted(unknown)))
        items = [item for item in items if item["id"] in item_ids]
    items.sort(key=lambda item: (item["type"], item["id"]))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        items = items[:limit]
    return items


def run(items, respond, deliver_feedback, *, max_attempts=10, required_streak=2):
    """respond(messages) returns raw JSON, None at EOF, or blank for a failed generation."""
    session = EvaluationSession(items, max_attempts=max_attempts, required_streak=required_streak)
    while not session.complete:
        raw = respond(session.messages())
        if raw is None:
            break
        row = session.submit(raw)
        if row is not None:
            deliver_feedback(row["feedback"])
    return session.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--item", action="append", help="restrict the run to an item ID; repeat to select more")
    parser.add_argument("--max-attempts", type=int, default=10, metavar="T")
    parser.add_argument("--required-streak", type=int, default=2, metavar="K")
    parser.add_argument("--responses", type=Path, help="replay one raw submission per line from a file")
    parser.add_argument("--report", type=Path, help="write private metrics and scored submissions as JSON")
    args = parser.parse_args()
    if not 1 <= args.required_streak <= args.max_attempts:
        parser.error("require 1 <= K <= T")
    data_file = args.data_dir / "problems.json"
    inputs = {data_file.resolve(), (args.data_dir / "answer_key.json").resolve()}
    if args.responses is not None:
        inputs.add(args.responses.resolve())
    if args.report is not None and args.report.resolve() in inputs:
        parser.error("report path must be separate from the input files")
    try:
        items = load_items(args.data_dir, args.item)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    stream = sys.stdin if args.responses is None else args.responses.open(encoding="utf-8")

    def emit(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    def respond(messages):
        if args.responses is None:
            emit({"type": "request", "messages": messages})
        try:
            line = stream.readline()
        except KeyboardInterrupt:
            return None
        return line.rstrip("\r\n") if line else None

    try:
        report = run(items, respond, lambda text: emit({"type": "feedback", "feedback": text}),
                     max_attempts=args.max_attempts, required_streak=args.required_streak)
        if args.responses is not None and stream.readline():
            parser.error("response file contains submissions beyond the end of the scheduled run")
    finally:
        if stream is not sys.stdin:
            stream.close()
    if args.report is not None:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": report["complete"], **report["summary"],
                      "by_type": report["by_type"]}, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
