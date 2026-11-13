"""Select a fresh hidden-goal case per submission using the shared evaluator."""

from copy import deepcopy
from hashlib import sha256
import json

from .evaluate import EvaluationSession


class ExperimentSession(EvaluationSession):
    """Run a validated problem catalog without changing scoring or metrics.

    A pending case stays fixed until ``submit`` consumes a response. Calling
    ``messages`` or retrying an API request therefore cannot select a new case.
    Replaying the same raw responses with the same seed restores the selection.
    The caller's catalog remains untouched for protocol snapshots and resumes.
    """

    def __init__(self, items, *, seed, max_attempts=10, required_streak=2):
        if type(seed) is not int:
            raise ValueError("the case-selection seed must be an integer")
        self.seed = seed
        self._last_case = None
        self._used_cases = {}
        super().__init__(deepcopy(items), max_attempts=max_attempts,
                         required_streak=required_streak)
        self._select_case()

    def _select_case(self):
        if self.complete:
            return
        item, record = self.current_item, self.current_record
        scope = (item["id"], record["round"])
        eligible = [case for case in record["cases"]
                    if self._last_case is None
                    or (case["input"] != self._last_case["input"]
                        and case["target"] != self._last_case["target"])]
        if not eligible:
            raise ValueError(f"{scope[0]} R{scope[1]}: no case changes both input and target")
        used = self._used_cases.get(scope, set())
        unused = [case for case in eligible if case["id"] not in used]
        pool = sorted(unused or eligible, key=lambda case: case["id"])
        selection_key = json.dumps([self.seed, item["id"], record["round"], self.attempt],
                                   ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        index = int.from_bytes(sha256(selection_key).digest(), "big") % len(pool)
        case = pool[index]
        # Preserve the round object's identity: EvaluationSession.report uses it
        # to associate active submissions with the current round.
        record.update(case_id=case["id"], input=case["input"], target=case["target"])

    def submit(self, raw):
        if self.complete:
            return super().submit(raw)  # Preserve the shared completed-session error.
        record = self.current_record
        scope = (self.current_item["id"], record["round"])
        consumed = {key: record[key] for key in ("case_id", "input", "target", "forbidden")}
        row = super().submit(raw)
        if row is not None:
            row.update({key: consumed[key] for key in ("case_id", "target", "forbidden")})
        self._used_cases.setdefault(scope, set()).add(consumed["case_id"])
        # Keep this across round and item boundaries, including a generation
        # failure. Only the conversation history resets at a new item.
        self._last_case = consumed
        self._select_case()
        return row
