"""Run or resume SameGame model evaluations."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import sys
from threading import Event, Thread
import time
import tomllib

from .evaluate import INSTRUCTIONS, PER_DENOMINATOR, PERSISTENT_ERROR_BASIS
from .grade import Game, successful_paths
from .catalog import load_catalog
from .session import ExperimentSession
from . import diagnostics, error_log, providers
from ._paths import PACKAGE_DIR, PROJECT_DIR

DEFAULT_CONFIGS = {
    "sol": PROJECT_DIR / "configs/openai.toml",
    "opus": PROJECT_DIR / "configs/anthropic.toml",
    "sonnet": PROJECT_DIR / "configs/anthropic.toml",
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_text(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_settings(path=None, profile=None):
    if path is None and profile is not None and profile not in DEFAULT_CONFIGS:
        raise ValueError("모델 프로필은 sol, opus, sonnet 중에서 선택하세요.")
    path = Path(path or DEFAULT_CONFIGS.get(profile or "sol", "")).expanduser().resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("config_version") != 1:
        raise ValueError("config_version은 1이어야 합니다.")
    profile = profile or config.get("default_profile")
    selected = config.get("profiles", {}).get(profile)
    experiment = config.get("experiment", {})
    if not isinstance(selected, dict) or not isinstance(experiment, dict):
        raise ValueError("설정 파일에서 선택한 모델 프로필을 찾을 수 없습니다.")
    required = {"dataset", "max_attempts", "required_streak", "max_output_tokens", "timeout"}
    if set(experiment) != required or set(selected) - {"reasoning_summary"} != {"label", "provider", "model", "effort"}:
        raise ValueError("설정 항목을 확인하세요. API 키는 환경 변수에 설정합니다.")
    settings = {**experiment, **selected, "profile": profile}
    if settings["provider"] not in providers.KEY_NAMES:
        raise ValueError("provider는 openai 또는 anthropic이어야 합니다.")
    if any(not isinstance(settings[k], str) or not settings[k].strip()
           for k in ("label", "model", "effort", "dataset")):
        raise ValueError("모델 설정의 문자열 항목을 확인하세요.")
    if settings["effort"] not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("지원하지 않는 effort입니다.")
    for key in ("max_attempts", "required_streak", "max_output_tokens"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key}는 1 이상의 정수여야 합니다.")
    if settings["required_streak"] > settings["max_attempts"]:
        raise ValueError("required_streak은 max_attempts 이하여야 합니다.")
    if (type(settings["timeout"]) not in (int, float)
            or not math.isfinite(settings["timeout"]) or settings["timeout"] <= 0):
        raise ValueError("timeout은 양수여야 합니다.")
    settings["dataset"] = str((path.parent / settings["dataset"]).resolve())
    diagnostics.mode(settings)
    return settings


def select_items(settings, identifiers=None, limit=None):
    items, seed = load_catalog(settings["dataset"], identifiers, limit)
    settings["selection_seed"] = seed
    return items


def make_session(items, settings, responses=()):
    session = ExperimentSession(items, seed=settings["selection_seed"],
        max_attempts=settings["max_attempts"], required_streak=settings["required_streak"])
    for raw in responses:
        if session.complete:
            raise ValueError("실험 종료 이후의 응답이 저장되어 있습니다.")
        session.submit(raw)
    return session


def implementation_fingerprint(settings=None):
    return digest({"instructions": INSTRUCTIONS, "sources": {
        name: sha256((PACKAGE_DIR / name).read_bytes()).hexdigest()
        for name in ("evaluate.py", "grade.py", "providers.py", "diagnostics.py",
                     "model_runner.py", "session.py", "catalog.py")
    }})


def compatible_fingerprint(frozen, settings):
    return frozen == implementation_fingerprint(settings)


def require_protocol(frozen):
    if frozen.get("version") != 1 or "selection_seed" not in frozen.get("settings", {}):
        raise ValueError("저장된 실행 형식과 후보 선택 seed를 확인하세요.")


def transport_options(settings, retries=10, delay=5.0, timeout=None):
    timeout = settings["timeout"] if timeout is None else timeout
    if type(retries) is not int or retries < 0:
        raise ValueError("통신 재시도 횟수는 0 이상의 정수여야 합니다.")
    if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
        raise ValueError("통신 재시도 대기는 0 이상의 유한한 숫자여야 합니다.")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout은 양수여야 합니다.")
    return {"max_retries": retries, "retry_delay": delay, "timeout": timeout}


def append_run_event(directory, event):
    with (directory / "run_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": timestamp(), **event}, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def evaluation_context(session):
    if session.complete:
        return {"submission": len(session.responses), "id": None, "round": None, "attempt": None}
    return {"submission": len(session.responses) + 1, "id": session.current_item["id"],
            "round": session.current_record["round"], "attempt": session.attempt}


def record_error(directory, event, error=None):
    """Persist a readable error without letting logging mask the original failure."""
    details = error_log.exception_details(error) if error is not None else {}
    diagnostic = {"at": timestamp(), **event, **details}
    try:
        append_run_event(directory, diagnostic)
        error_log.append_error(directory, diagnostic)
    except OSError as logging_error:
        print(f"오류 기록 저장 실패 · {type(logging_error).__name__}", file=sys.stderr, flush=True)
    return diagnostic


class Terminal:
    """Compact evaluator logs with separate correctness and persistence results."""

    def __init__(self, settings, items, directory):
        from shutil import get_terminal_size

        self.settings = settings
        self.total = len(items)
        self._possible_answers = {}
        self.width = min(100, max(40, get_terminal_size(fallback=(96, 24)).columns))
        self.color = sys.stderr.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
        self.pe_rule = "이전 규칙 준수 + 현재 오답"
        self._heading("SameGame · 모델 평가")
        self._line("모델", settings["label"])
        self._line("진행 설정", f"{self.total}문제 · 최대 {settings['max_attempts']}회 · 연속 정답 {settings['required_streak']}회")
        self._line("지속 오류 기준", f"{self.pe_rule} · R1 집계 제외")
        self._line("PER 분모", f"최대 시도 수 T={settings['max_attempts']} · 미진입 라운드는 미측정")
        self._line("토큰 안내", "입력은 요청 문맥, 출력은 모델 생성량(추론 포함)")
        if diagnostics.mode(settings) != "off":
            self._line("추론 요약", f"{diagnostics.mode(settings)} · reasoning_log.txt")
        self._line("저장 위치", directory)

    @staticmethod
    def clean(value):
        return "".join(char for char in str(value) if char.isprintable())

    @staticmethod
    def display_width(value):
        """Count Hangul as two terminal cells and combining marks as zero."""
        from unicodedata import combining, east_asian_width

        return sum(0 if combining(char) else 2 if east_asian_width(char) in {"W", "F"} else 1
                   for char in value)

    @classmethod
    def wrapped(cls, value, width):
        remaining = cls.clean(value)
        lines = []
        while cls.display_width(remaining) > width:
            used = cut = 0
            for index, char in enumerate(remaining):
                size = cls.display_width(char)
                if used + size > width:
                    break
                used += size
                cut = index + 1
            split = remaining.rfind(" ", 0, cut + 1)
            cut = split if split > 0 else max(cut, 1)
            lines.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip()
        return lines + [remaining]

    def _styled(self, text, tone):
        return f"\033[{tone}m{text}\033[0m" if self.color and tone else text

    def _heading(self, title):
        title = self.clean(title)
        print(file=sys.stderr)
        for line in self.wrapped("── " + title + " ", self.width):
            tail = "─" * max(0, self.width - self.display_width(line))
            print(self._styled(line + tail, "1;36"), file=sys.stderr, flush=True)

    def _line(self, label, value, tone=None):
        prefix = "  " + label + " " * max(2, 18 - self.display_width(label))
        indent = self.display_width(prefix)
        for index, line in enumerate(self.wrapped(value, max(10, self.width - indent))):
            lead = " " * indent if index else prefix
            print(lead + self._styled(line, tone), file=sys.stderr, flush=True)

    def attempt_label(self, session):
        return (f"[{session.item_index+1:02d}/{self.total:02d}] {session.current_item['id']} "
                f"· R{session.round_index+1} · 시도 {session.attempt}/{self.settings['max_attempts']}")

    @staticmethod
    def format_answer(answer):
        if answer is None:
            return "해석 불가"
        if answer == "":
            return "∅ (빈 문자열)"
        return json.dumps(answer, ensure_ascii=False)

    def possible_answers(self, record):
        key = (record["input"], record["target"], record["forbidden"])
        if key not in self._possible_answers:
            counts, _ = successful_paths(Game(key[0], key[1]), key[2])
            self._possible_answers[key] = tuple(sorted(counts))
        return self._possible_answers[key]

    def start_attempt(self, session):
        current = session.current_record["forbidden"]
        previous = (session.current_item["rounds"][session.round_index - 1]["forbidden"]
                    if session.round_index else None)
        previous_label = (f"{previous[0]} → {previous[1]} 연속 삭제 금지"
                          if previous else "없음 (첫 라운드)")
        self._heading(self.attempt_label(session))
        self._line("현재 문자열", self.format_answer(session.current_record["input"]))
        self._line("이전 규칙", previous_label)
        self._line("현재 규칙", f"{current[0]} → {current[1]} 연속 삭제 금지")
        answers = self.possible_answers(session.current_record)
        finals = ", ".join(self.format_answer(answer) for answer in answers) or "없음"
        self._line("현재 가능 정답", f"{len(answers)}개 · {finals}")
        self._line("현재 목표", f"{session.current_record['target']}글자 남기기 · 평가자용")
        self._line("현재 조건", session.current_record["case_id"])

    @contextmanager
    def waiting(self, session):
        stop = Event()
        begin = time.monotonic()

        def animate():
            frames = ("◐", "◓", "◑", "◒")
            index = 0
            while not stop.wait(0.15):
                elapsed = int(time.monotonic() - begin)
                print(f"\r\033[2K  {frames[index % 4]} 응답 대기  {elapsed//60:02d}:{elapsed%60:02d}",
                      file=sys.stderr, end="", flush=True)
                index += 1

        thread = Thread(target=animate, daemon=True) if self.color else None
        if thread:
            thread.start()
        try:
            yield
        finally:
            stop.set()
            if thread:
                thread.join()
                print("\r\033[2K", file=sys.stderr, end="", flush=True)

    def submission(self, row, elapsed, usage, failure=None):
        if failure:
            self._line("결과", "생성 미완료 · 미채점, 다음 문제로 이동", "33")
            self._line("원인", failure)
            self._line("지속 오류", "집계 제외 · 채점 가능한 응답 없음", "2")
            return
        self._line("정답 판정", "정답" if row["correct"] else "오답", "1;32" if row["correct"] else "1;33")
        persistent = row.get("persistent_error")
        if persistent is None:
            self._line("지속 오류", "집계 제외 · R1 (직전 규칙 없음)", "2")
        else:
            self._line("지속 오류", f"{'발생' if persistent else '없음'} · PE={int(persistent)} ({self.pe_rule} 기준)",
                       "1;31" if persistent else "32")
        self._line("피드백", row["feedback"])
        self._line("제출한 삭제", row.get("moves") or "해석 불가")
        self._line("제출한 답안", self.format_answer(row.get("answer")))
        current = row.get("violation_steps") or []
        previous = row.get("previous_violation_steps")
        self._line("현재 규칙 위반", ", ".join(map(str, current)) + "번째 행동" if current else "없음")
        self._line("이전 규칙 위반", "해당 없음 (R1)" if previous is None else
                   ", ".join(map(str, previous)) + "번째 행동" if previous else "없음")
        compliant = row.get("previous_rule_compliant")
        self._line("이전 기준 조작", "해당 없음 (R1)" if compliant is None else
                   "완결·답안 일치·이전 규칙 준수" if compliant else "이전 규칙 또는 기본 조작 미충족")
        self._line("진행", f"연속 정답 {row['success_streak']}/{self.settings['required_streak']} · 응답 {elapsed:.1f}초")
        self._line("토큰 · API 집계", f"입력 {usage['input_tokens']:,} / 출력 {usage['output_tokens']:,}")
        print(file=sys.stderr, flush=True)

    def retry(self, error, number, limit, delay):
        self._line("통신 오류", error, "33")
        self._line("자동 재시도", f"{number}/{limit} · {delay:.1f}초 후 · 같은 요청 유지 · 풀이 시도 미차감")

    def summary(self, report):
        summary = report.get("summary", {})
        self._heading("평가 완료" if report.get("complete") else "진행 기록 저장")
        self._line("전체 제출", f"{len(report.get('submissions', [])):,}건")
        eligible = [row for row in report.get("submissions", []) if row.get("persistent_error") is not None]
        self._line("지속 오류", f"{sum(bool(row['persistent_error']) for row in eligible):,}건 / 집계 제출 {len(eligible):,}건")
        mpe = summary.get("mpe")
        self._line("MPE", f"{100*mpe:.2f}% · 고정 T 기준 · 문제별 평균" if mpe is not None else "집계 대상 없음")
        if "observed_transition_rounds" in summary:
            self._line("측정 라운드", f"{summary['observed_transition_rounds']}/{summary['scheduled_transition_rounds']} · R2 이후")



@contextmanager
def lock_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("다른 프로세스가 이 결과 폴더를 사용하고 있습니다.") from None
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def save_progress(directory, session, native_item, native_history):
    write_json(directory / "checkpoint.json", {
        "version": 1, "saved_at": timestamp(), "responses": session.responses,
        "native_item": native_item, "native_history": native_history,
    })
    write_json(directory / "report.json", session.report())


def recover_stream_response(trials, number, request, provider):
    """A completed stream survives a crash before the response file is saved."""
    request_hash = digest(request)
    for stream_path in sorted(trials.glob(f"{number:05d}_transport_*_stream.jsonl"), reverse=True):
        started_path = stream_path.with_name(stream_path.name.replace("_stream.jsonl", "_started.json"))
        if not started_path.exists():
            continue
        started = read_json(started_path)
        if started.get("request_sha256") != request_hash:
            raise ValueError("저장된 스트림의 요청 식별자가 일치하지 않습니다.")
        raw = providers.recover_response(stream_path, provider)
        if raw is not None:
            elapsed = max(0.0, stream_path.stat().st_mtime - datetime.fromisoformat(started["at"]).timestamp())
            return {"response": raw, "elapsed_seconds": elapsed, "elapsed_estimated": True,
                    "request_sha256": request_hash, "received_at": timestamp(),
                    "recovered_from_stream": stream_path.name}
    return None


def receive_with_retries(client, settings, request, trials, number, terminal, session, receive, options):
    """Retry a transport failure without submitting, advancing, or adding partial context."""
    for retry in range(options["max_retries"] + 1):
        existing = trials.glob(f"{number:05d}_transport_*_started.json")
        index = max((int(path.name.split("_")[-2]) for path in existing), default=0) + 1
        transport = trials / f"{number:05d}_transport_{index:03d}"
        context = {**evaluation_context(session), "provider": settings["provider"], "model": settings["model"],
                   "transport_attempt": index, "retry_number": retry, "retry_limit": options["max_retries"]}
        write_json(Path(str(transport) + "_started.json"), {
            "at": timestamp(), "request_sha256": digest(request), "transport": options, **context,
        })
        begin = time.monotonic()
        try:
            with terminal.waiting(session):
                raw = receive(client, settings["provider"], request, Path(str(transport) + "_stream.jsonl"))
        except providers.ProviderError as error:
            # A full terminal event may already have arrived before a trailing
            # network failure. Reuse it before issuing a potentially billable request.
            recovered = recover_stream_response(trials, number, request, settings["provider"])
            will_retry = recovered is None and error.retryable and retry < options["max_retries"]
            delay = max(error.retry_after or 0, options["retry_delay"] * 2**min(retry, 8)
                        + random.uniform(0, 1)) if will_retry else 0
            diagnostic = {
                "at": timestamp(), "request_sha256": digest(request), **context,
                "kind": error.kind, "message": str(error), "status_code": error.status_code,
                "retryable": error.retryable, "retry_after": error.retry_after,
                "elapsed_seconds": time.monotonic() - begin, "will_retry": will_retry,
                "retry_delay_seconds": delay, "recovered_from_stream": recovered is not None,
                "timeout_seconds": options["timeout"], **error_log.exception_details(error),
            }
            write_json(Path(str(transport) + "_error.json"), diagnostic)
            record_error(trials.parent, {"event": "transport_error", **diagnostic})
            if recovered is not None:
                print(f"완료된 스트림에서 응답 복구 · {number:05d}", file=sys.stderr)
                event = {"at": timestamp(), "event": "transport_success", **context,
                         "recovered_from_stream": recovered["recovered_from_stream"]}
                append_run_event(trials.parent, event)
                error_log.append_error(trials.parent, event)
                return recovered
            if not will_retry:
                raise
            terminal.retry(error, retry + 1, options["max_retries"], delay)
            time.sleep(delay)
            continue
        event = {"at": timestamp(), "event": "transport_success", **context,
                 "elapsed_seconds": time.monotonic() - begin, "recovered_from_stream": False}
        append_run_event(trials.parent, event)
        if index > 1:
            error_log.append_error(trials.parent, event)
        return {"response": raw, "elapsed_seconds": event["elapsed_seconds"],
                "request_sha256": digest(request), "received_at": timestamp()}


def execute(items, settings, directory, *, resume=False, client=None, receive=providers.receive,
            transport_retries=10, retry_delay=5.0, timeout=None):
    """Every saved response is reused before another provider request is made."""
    summary_mode = diagnostics.mode(settings)
    options = transport_options(settings, transport_retries, retry_delay, timeout)
    owned_client = client is None
    with lock_directory(directory):
        if resume:
            frozen = read_json(directory / "protocol.json")
            require_protocol(frozen)
            if not compatible_fingerprint(frozen["fingerprint"], settings):
                raise ValueError("실행 중 지시문·채점기·API 코드가 변경되었습니다. 원래 코드로 재개하세요.")
            if frozen["items"] != items or frozen["settings"] != settings:
                raise ValueError("저장된 실행의 문제·모델 설정과 일치하지 않습니다.")
            saved = read_json(directory / "checkpoint.json")
            session = make_session(items, settings, saved["responses"])
            native_item = saved["native_item"]
            native_history = saved["native_history"]
        else:
            if any(path.name != ".run.lock" for path in directory.iterdir()):
                raise ValueError("결과 폴더가 비어 있지 않습니다. 새 폴더 또는 --resume을 사용하세요.")
            session = make_session(items, settings)
            native_item, native_history = None, []
            write_json(directory / "protocol.json", {
                "version": 1, "created_at": timestamp(), "settings": settings, "items": items,
                "persistent_error_basis": PERSISTENT_ERROR_BASIS,
                "per_denominator": PER_DENOMINATOR,
                "fingerprint": implementation_fingerprint(settings),
            })
            save_progress(directory, session, native_item, native_history)
        terminal = Terminal(settings, items, directory)
        append_run_event(directory, {"event": "resume" if resume else "start", "transport": options,
                                     "implementation_fingerprint": implementation_fingerprint(settings)})
        print(f"통신 설정 · timeout {options['timeout']:g}초 · 자동 재시도 최대 {options['max_retries']}회\n",
              file=sys.stderr, flush=True)
        trials = directory / "submissions"
        trials.mkdir(exist_ok=True)
        summaries = {}
        if summary_mode != "off":
            for path in sorted(trials.glob("*_reasoning_summary.json")):
                saved_summary = read_json(path)
                summaries[saved_summary["submission"]] = saved_summary
            write_text(directory / "reasoning_log.txt", diagnostics.render_log(summaries.values()))
        context = evaluation_context(session)
        phase = "prepare_request"
        try:
            while not session.complete:
                context = evaluation_context(session)
                phase = "prepare_request"
                current_id = session.current_item["id"]
                if summary_mode != "off":
                    current = session.current_record
                    diagnostic_context = {
                        "id": current_id, "round": current["round"], "attempt": session.attempt,
                        "input": current["input"], "target": current["target"], "case_id": current["case_id"],
                        "forbidden": current["forbidden"],
                        "previous_forbidden": (session.current_item["rounds"][session.round_index-1]["forbidden"]
                                               if session.round_index else None),
                    }
                messages = session.messages()
                history = native_history if native_item == current_id else messages[1:-1]
                submitted_history = history + [messages[-1]]
                request = diagnostics.make_request(settings, messages[0]["content"], submitted_history)
                number = len(session.responses) + 1
                stem = trials / f"{number:05d}"
                request_path = Path(str(stem) + "_request.json")
                response_path = Path(str(stem) + "_response.json")
                if request_path.exists():
                    if read_json(request_path) != request:
                        raise ValueError("저장된 요청과 재개할 요청이 다릅니다.")
                else:
                    write_json(request_path, request)
                terminal.start_attempt(session)
                if response_path.exists():
                    envelope = read_json(response_path)
                    if envelope["request_sha256"] != digest(request):
                        raise ValueError("저장된 응답의 요청 식별자가 일치하지 않습니다.")
                    print(f"저장된 응답 재사용 · {number:05d}", file=sys.stderr)
                else:
                    envelope = recover_stream_response(trials, number, request, settings["provider"])
                    if envelope is not None:
                        print(f"완료된 스트림에서 응답 복구 · {number:05d}", file=sys.stderr)
                    else:
                        phase = "receive"
                        if client is None:
                            client = providers.make_client({**settings, "timeout": options["timeout"]})
                        envelope = receive_with_retries(client, settings, request, trials, number,
                                                        terminal, session, receive, options)
                    write_json(response_path, envelope)
                phase = "grade"
                raw = envelope["response"]
                text, failure = providers.visible_text(raw, settings["provider"])
                continuation = [] if failure else providers.continuation_items(raw, settings["provider"])
                if failure:
                    reason = (raw.get("incomplete_details") or {}).get("reason")
                    if reason not in {"max_output_tokens", "content_filter"}:
                        reason = None
                    record_error(directory, {"event": "generation_failure", **context,
                                            "kind": "generation_failure", "failure": failure,
                                            "generation_reason": reason,
                                            "message": "응답 생성이 미완료되어 채점 없이 다음 문제로 이동합니다."})
                row = session.submit(text)
                if failure:
                    native_item, native_history = None, []
                else:
                    native_item = current_id
                    native_history = submitted_history + continuation + [{"role": "user", "content": row["feedback"]}]
                usage = providers.token_usage(raw, settings["provider"])
                phase = "save_progress"
                write_json(Path(str(stem) + "_record.json"), {
                    "submission": number, "generation_failure": failure,
                    "usage": usage, "elapsed_seconds": envelope["elapsed_seconds"], "score": row,
                })
                if summary_mode != "off":
                    summary_text = diagnostics.extract_summary(raw)
                    diagnostic = {
                        "submission": number, **diagnostic_context, "summary_mode": summary_mode,
                        "summary_status": "available" if summary_text else "empty",
                        "summary": summary_text, "raw_output": text,
                        "feedback": row["feedback"] if row else None,
                        "violation_steps": row["violation_steps"] if row else [],
                        "previous_violation_steps": row["previous_violation_steps"] if row else None,
                        "previous_rule_compliant": row["previous_rule_compliant"] if row else None,
                        "persistent_error": row["persistent_error"] if row else None,
                        "persistent_error_basis": PERSISTENT_ERROR_BASIS,
                        "generation_failure": failure,
                    }
                    write_json(Path(str(stem) + "_reasoning_summary.json"), diagnostic)
                    summaries[number] = diagnostic
                    write_text(directory / "reasoning_log.txt", diagnostics.render_log(summaries.values()))
                save_progress(directory, session, native_item, native_history)
                append_run_event(directory, {"event": "submission_graded", **context,
                                             "generation_failure": failure, "correct": row["correct"] if row else None})
                terminal.submission(row, envelope["elapsed_seconds"], usage, failure)
                if summary_mode != "off" and not summary_text:
                    print("  추론 요약 없음 · API가 요약 텍스트를 반환하지 않았습니다.\n", file=sys.stderr)
        except KeyboardInterrupt as error:
            record_error(directory, {"event": "interrupted", "kind": "KeyboardInterrupt", **context,
                                     "phase": phase, "message": "사용자 중단 신호(Ctrl+C)를 받았습니다."}, error)
            raise
        except Exception as error:
            record_error(directory, {"event": "stopped", **context, "phase": phase,
                                     "kind": error.kind if isinstance(error, providers.ProviderError) else "local_error",
                                     "message": str(error) if isinstance(error, providers.ProviderError)
                                     else "실행 코드에서 오류가 발생했습니다. 예외 유형과 코드 위치를 확인하세요."}, error)
            print(f"재개 명령 · python run.py --resume {directory} --timeout {options['timeout']:g}",
                  file=sys.stderr, flush=True)
            raise
        finally:
            unwinding = sys.exc_info()[0] is not None
            cleanup_failure = None
            cleanup = [("save_report", lambda: write_json(directory / "report.json", session.report())),
                       ("terminal_summary", lambda: terminal.summary(session.report()))]
            if owned_client and client is not None:
                cleanup.insert(0, ("close_client", client.close))
            for cleanup_phase, action in cleanup:
                try:
                    action()
                except Exception as error:
                    record_error(directory, {"event": "stopped", **context, "phase": cleanup_phase,
                                             "kind": "local_error", "message": "종료 정리 중 오류가 발생했습니다."}, error)
                    cleanup_failure = cleanup_failure or error
            if cleanup_failure is not None and not unwinding:
                raise cleanup_failure
        append_run_event(directory, {"event": "complete", "submissions": len(session.responses)})
    return session.report()


def main(argv=None):
    parser = argparse.ArgumentParser(description="SameGame · 모델 평가")
    parser.add_argument("--config", type=Path, help="OpenAI 또는 Anthropic TOML 설정")
    parser.add_argument("--dataset", type=Path, help="시도별 조건 데이터 경로; 생략하면 설정 파일 사용")
    parser.add_argument("--profile", choices=tuple(DEFAULT_CONFIGS), help="sol, opus, sonnet")
    parser.add_argument("--item", action="append", help="문제 ID, 여러 번 지정 가능")
    parser.add_argument("--limit", type=int, help="실행할 문제 수")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="새 결과 폴더")
    destination.add_argument("--resume", type=Path, help="이전 결과 폴더에서 재개")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 첫 요청 확인")
    parser.add_argument("--timeout", type=float, help="이번 실행의 통신 timeout(초), 재개 시에도 지정 가능")
    parser.add_argument("--transport-retries", type=int, default=10, help="일시적 통신 실패 자동 재시도 횟수(기본 10)")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="재시도 기본 대기 초(기본 5, 이후 지수 증가)")
    parser.add_argument("--reasoning-summary", nargs="?", const="auto", choices=diagnostics.SUMMARY_MODES,
                        help="OpenAI 추론 요약 저장 (값 생략 시 auto, 기본 off)")
    args = parser.parse_args(argv)
    try:
        if args.resume:
            if args.config or args.dataset or args.profile or args.item or args.limit is not None or args.reasoning_summary is not None:
                raise ValueError("--resume은 저장된 설정과 문제 목록을 사용합니다.")
            directory = args.resume.expanduser().resolve()
            frozen = read_json(directory / "protocol.json")
            require_protocol(frozen)
            settings, items = frozen["settings"], frozen["items"]
        else:
            settings = load_settings(args.config, args.profile)
            if args.dataset is not None:
                settings["dataset"] = str(args.dataset.expanduser().resolve())
            if args.reasoning_summary is not None:
                settings["reasoning_summary"] = args.reasoning_summary
            items = select_items(settings, args.item, args.limit)
            default = PROJECT_DIR / "runs" / f"{settings['profile']}_{datetime.now():%Y%m%d_%H%M%S}"
            directory = (args.output or default).expanduser().resolve()
        diagnostics.mode(settings)
        options = transport_options(settings, args.transport_retries, args.retry_delay, args.timeout)
        if args.resume and not compatible_fingerprint(frozen["fingerprint"], settings):
            raise ValueError("실행 중 지시문·채점기·API 코드가 변경되었습니다. 원래 코드로 재개하세요.")
        if args.dry_run:
            session = make_session(items, settings)
            if args.resume:
                saved = read_json(directory / "checkpoint.json")
                session = make_session(items, settings, saved["responses"])
            if session.complete:
                print("이미 완료된 실행입니다.")
                return 0
            messages = session.messages()
            history = messages[1:-1]
            if args.resume and saved["native_item"] == session.current_item["id"]:
                history = saved["native_history"]
            request = diagnostics.make_request(settings, messages[0]["content"], history + [messages[-1]])
            print(json.dumps({"profile": settings["profile"], "problems": len(items), "transport": options,
                              "persistent_error_basis": PERSISTENT_ERROR_BASIS,
                              "per_denominator": PER_DENOMINATOR,
                              "request": request}, ensure_ascii=False, indent=2))
            return 0
        execute(items, settings, directory, resume=bool(args.resume), transport_retries=args.transport_retries,
                retry_delay=args.retry_delay, timeout=args.timeout)
        print(f"결과 저장 · {directory / 'report.json'}")
        return 0
    except KeyboardInterrupt:
        print("\n중단했습니다. --resume으로 저장된 실행을 이어갈 수 있습니다.", file=sys.stderr)
        return 130
    except (providers.ProviderError, ValueError, OSError, KeyError) as exc:
        print(f"실행 중단 · {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"실행 중단 · {type(exc).__name__}. 실행 폴더의 errors.log를 확인하세요.", file=sys.stderr)
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
