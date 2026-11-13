"""Load and validate the SameGame problem catalog."""

from collections import Counter
import json
from pathlib import Path

from .grade import read_round


def load_catalog(path, identifiers=None, limit=None):
    path = Path(path)
    document = json.loads((path / "problems.json" if path.is_dir() else path).read_text(encoding="utf-8"))
    if document.get("version") != 1 or type(document.get("seed")) is not int:
        raise ValueError("시도별 문자열·목표 데이터(version=1, seed)가 필요합니다.")
    for field in ("rounds_per_item", "cases_per_target"):
        if type(document.get(field)) is not int or document[field] < 1:
            raise ValueError(f"{field}는 양의 정수여야 합니다.")
    items = document.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("문제 목록이 비어 있습니다.")
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("문제 ID가 중복됩니다.")
    for item in items:
        if len(item.get("rounds", [])) != document["rounds_per_item"]:
            raise ValueError("실제 라운드 수가 rounds_per_item과 다릅니다.")
        previous = None
        for number, record in enumerate(item["rounds"], 1):
            cases = record.get("cases")
            if not isinstance(cases, list) or not cases:
                raise ValueError("각 라운드에는 시도용 문자열 후보가 필요합니다.")
            case_ids = [case["id"] for case in cases]
            if (any(not isinstance(identifier, str) or not identifier for identifier in case_ids)
                    or len(set(case_ids)) != len(case_ids)):
                raise ValueError("라운드 안의 후보 ID는 중복되지 않는 문자열이어야 합니다.")
            if {case["target"] for case in cases} != {0, 1, 2, 3}:
                raise ValueError("각 라운드는 목표 0·1·2·3의 후보가 모두 필요합니다.")
            if Counter(case["target"] for case in cases) != {goal: document["cases_per_target"] for goal in range(4)}:
                raise ValueError("목표별 실제 후보 수가 cases_per_target과 다릅니다.")
            if len({case["input"] for case in cases}) != len(cases):
                raise ValueError("같은 라운드의 후보 문자열은 서로 달라야 합니다.")
            if record["forbidden"] == previous:
                raise ValueError("인접 라운드의 규칙은 서로 달라야 합니다.")
            for case in cases:
                read_round({"round": record["round"], "forbidden": record["forbidden"],
                            "input": case["input"], "target": case["target"]}, number, item["type"])
            previous = record["forbidden"]
    if identifiers:
        unknown = set(identifiers) - set(ids)
        if unknown:
            raise ValueError("없는 문제 ID: " + ", ".join(sorted(unknown)))
        items = [item for item in items if item["id"] in identifiers]
    # Keep catalog order so expanding a fifty-problem batch preserves its prefix.
    if limit is not None:
        if limit < 1:
            raise ValueError("limit은 1 이상이어야 합니다.")
        items = items[:limit]
    return items, document["seed"]

