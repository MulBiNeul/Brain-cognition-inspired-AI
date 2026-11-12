"""Optional API reasoning summaries for evaluator-only diagnostic logs."""

from . import providers


SUMMARY_MODES = ("off", "auto", "concise", "detailed")


def mode(settings):
    selected = settings.get("reasoning_summary", "off")
    if selected not in SUMMARY_MODES:
        raise ValueError("reasoning_summary는 off, auto, concise, detailed 중 하나여야 합니다.")
    if selected != "off" and settings["provider"] != "openai":
        raise ValueError("추론 요약 옵션은 OpenAI 실행에서 지원합니다.")
    return selected


def make_request(settings, system, history):
    selected = mode(settings)
    request = providers.make_request(settings, system, history)
    if selected != "off":
        request["reasoning"]["summary"] = selected
    return request


def extract_summary(raw):
    """Read only API-provided summary text; never decode opaque reasoning state."""
    parts = []
    for item in raw.get("output", []):
        if item.get("type") != "reasoning":
            continue
        for part in item.get("summary") or []:
            text = part.get("text")
            if part.get("type") == "summary_text" and isinstance(text, str) and text.strip():
                parts.append(text)
    return "\n\n".join(parts)


def render_log(records):
    sections = [
        "SameGame · 추론 요약 진단 기록\n"
        "API가 제공한 요약이며 내부 추론 원문을 복호화한 기록이 아닙니다.\n"
        "요약을 반환하지 않은 응답은 '반환된 추론 요약 없음'으로 표시합니다."
    ]
    for record in sorted(records, key=lambda value: value["submission"]):
        feedback = record["feedback"] or f"미채점 ({record['generation_failure']})"
        violations = ", ".join(map(str, record["violation_steps"])) or "없음"
        persistence = ""
        if record.get("persistent_error_basis") == "previous_rule_compliant_current_incorrect":
            previous = ", ".join(map(str, record.get("previous_violation_steps") or [])) or "없음"
            counted = record.get("persistent_error")
            persistence = (f"\n이전 규칙 위반 행동: {previous}\n"
                           f"이전 규칙·기본 조작 준수: {record.get('previous_rule_compliant')}\n"
                           f"지속 오류: {'미집계' if counted is None else int(counted)} (이전 규칙 준수 + 현재 오답)")
        sections.append(
            f"[{record['submission']:05d}] {record['id']} · R{record['round']} · 시도 {record['attempt']}\n"
            f"입력: {record['input']}\n"
            f"목표: {record['target']}글자 남기기\n"
            f"이전 금지쌍: {record['previous_forbidden'] or '없음 (첫 라운드)'}\n"
            f"현재 금지쌍: {record['forbidden']}\n\n"
            f"추론 요약 ({record['summary_mode']}):\n"
            f"{record['summary'] or '[반환된 추론 요약 없음]'}\n\n"
            f"제출 원문:\n{record['raw_output'] or '[채점 가능한 응답 없음]'}\n\n"
            f"피드백: {feedback}\n"
            f"현재 규칙 위반 행동: {violations}{persistence}"
        )
    return "\n\n".join(sections) + "\n"
