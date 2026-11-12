"""Replay saved submissions under the clarified PE definition, without API calls."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import EvaluationSession, PER_DENOMINATOR, PERSISTENT_ERROR_BASIS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes(directory):
    return {name: sha256((directory / name).read_bytes()).hexdigest()
            for name in ("protocol.json", "checkpoint.json", "report.json")}


def recalculate(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or output == directory / "submissions":
        raise ValueError("재집계 결과는 원본 실행과 다른 분석 폴더에 저장해야 합니다.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("분석 폴더가 비어 있지 않습니다. 새 분석 폴더를 지정하세요.")
    before = hashes(directory)
    protocol, checkpoint, original = (read(directory / name)
                                      for name in ("protocol.json", "checkpoint.json", "report.json"))
    settings = protocol["settings"]
    session = EvaluationSession(protocol["items"], max_attempts=settings["max_attempts"],
                                required_streak=settings["required_streak"])
    for raw in checkpoint["responses"]:
        if session.complete:
            raise ValueError("저장된 응답이 실험 종료 이후까지 포함되어 있습니다.")
        session.submit(raw)
    report = session.report()
    if len(report["submissions"]) != len(original["submissions"]):
        raise ValueError("원본과 재채점된 제출 수가 다릅니다.")
    unchanged = ("id", "round", "attempt", "raw", "feedback", "correct", "success_streak", "input")
    for index, (old, new) in enumerate(zip(original["submissions"], report["submissions"]), 1):
        differences = [key for key in unchanged if old[key] != new[key]]
        if differences:
            raise ValueError(f"제출 {index}: 정답·피드백·진행 재현 실패: {differences}")
    if report["complete"] != original["complete"]:
        raise ValueError("실험 완료 상태가 달라졌습니다.")
    for old, new in zip(original["problems"], report["problems"]):
        if [(r["status"], r["submissions"]) for r in old["rounds"]] != [
                (r["status"], r["submissions"]) for r in new["rounds"]]:
            raise ValueError(f"{old['id']}: 라운드 진행이 달라졌습니다.")

    eligible = [row for row in report["submissions"] if row["persistent_error"] is not None]
    observed = [r for p in report["problems"] for r in p["rounds"] if r["per"] is not None]
    errors = sum(row["persistent_error"] for row in eligible)
    comparison = {
        "source_run": str(directory), "source_sha256": before,
        "persistent_error_basis": PERSISTENT_ERROR_BASIS, "per_denominator": PER_DENOMINATOR,
        "max_attempts": settings["max_attempts"], "required_streak": settings["required_streak"],
        "definition": "current input; complete truthful plan obeys previous rule; current response is incorrect",
        "aggregation": "mean of observed R2+ PER per problem, then mean over eligible problems",
        "unobserved_round_policy": "missing, not zero; early-success rounds retain the fixed T denominator",
        "model_calls": 0, "unchanged_scored_submissions": len(report["submissions"]),
        "current_correctness_feedback_and_progress_identical": True,
        "original_summary": original["summary"], "recalculated_summary": report["summary"],
        "by_type": report["by_type"],
        "persistent_errors": errors, "eligible_submissions": len(eligible),
        "pooled_submission_rate": errors / len(eligible) if eligible else None,
        "flat_observed_round_mean_fixed_T": sum(r["per"] for r in observed) / len(observed) if observed else None,
        "flat_observed_round_mean_actual_n": sum(r["observed_per"] for r in observed) / len(observed) if observed else None,
        "errors_that_also_reached_current_goal": sum(row["persistent_error"] and bool(row["goal_reached"])
                                                    for row in eligible),
        "errors_that_missed_current_goal": sum(row["persistent_error"] and not row["goal_reached"]
                                               for row in eligible),
    }
    if hashes(directory) != before:
        raise ValueError("재집계 중 원본 파일이 변경되었습니다.")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("report.json", report), ("comparison.json", comparison)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        comparison = recalculate(args.run, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({key: comparison[key] for key in (
        "persistent_error_basis", "max_attempts", "unchanged_scored_submissions", "persistent_errors",
        "eligible_submissions", "recalculated_summary", "by_type")}, ensure_ascii=False, indent=2))
    print(f"저장: {args.output.resolve()}")


if __name__ == "__main__":
    main()
