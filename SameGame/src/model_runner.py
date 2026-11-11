"""Run or resume provider-backed SameGame experiments."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from threading import Event, Thread
import time
import tomllib

from .evaluate import EvaluationSession, INSTRUCTIONS, load_items
from .grade import Game, successful_paths
from . import diagnostics, providers
from ._paths import PACKAGE_DIR, PROJECT_DIR


DEFAULT_CONFIGS = {
    "sol": PROJECT_DIR / "configs" / "openai.toml",
    "opus": PROJECT_DIR / "configs" / "anthropic.toml",
    "sonnet": PROJECT_DIR / "configs" / "anthropic.toml",
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
    return load_items(settings["dataset"], identifiers, limit)


def make_session(items, settings, responses=()):
    session = EvaluationSession(items, max_attempts=settings["max_attempts"],
                                required_streak=settings["required_streak"])
    for raw in responses:
        if session.complete:
            raise ValueError("저장된 응답이 실험 종료 이후까지 포함되어 있습니다.")
        session.submit(raw)
    return session


def implementation_fingerprint(settings=None):
    sources = {
        "instructions": INSTRUCTIONS,
        "grader": sha256((PACKAGE_DIR / "grade.py").read_bytes()).hexdigest(),
        "evaluator": sha256((PACKAGE_DIR / "evaluate.py").read_bytes()).hexdigest(),
        "providers": sha256((PACKAGE_DIR / "providers.py").read_bytes()).hexdigest(),
    }
    if settings is not None and diagnostics.mode(settings) != "off":
        sources["diagnostics"] = sha256((PACKAGE_DIR / "diagnostics.py").read_bytes()).hexdigest()
        sources["runner"] = sha256((PACKAGE_DIR / "model_runner.py").read_bytes()).hexdigest()
    return digest(sources)


class Terminal:
    """Readable research logs, with a timer only on an interactive terminal."""

    def __init__(self, settings, items, directory):
        self.settings = settings
        self.total = len(items)
        self._possible_answers = {}
        self.color = sys.stderr.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
        print(f"\n╭─ SameGame · {settings['label']}", file=sys.stderr)
        print(f"│  {self.total}문제 · 최대 {settings['max_attempts']}회 · 연속 정답 {settings['required_streak']}회", file=sys.stderr)
        if diagnostics.mode(settings) != "off":
            print(f"│  추론 요약 {diagnostics.mode(settings)} · {directory / 'reasoning_log.txt'}", file=sys.stderr)
        print(f"╰─ {directory}\n", file=sys.stderr)

    @staticmethod
    def clean(value):
        return "".join(char for char in str(value) if char.isprintable())

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
        print(self.attempt_label(session), file=sys.stderr)
        print(f"  이전 규칙  {previous_label}", file=sys.stderr)
        print(f"  현재 규칙  {current[0]} → {current[1]} 연속 삭제 금지", file=sys.stderr, flush=True)
        answers = self.possible_answers(session.current_record)
        finals = ", ".join(self.format_answer(answer) for answer in answers) or "없음"
        print(f"  현재 가능 정답 ({len(answers)}개)  {finals}", file=sys.stderr, flush=True)

    @contextmanager
    def waiting(self, session):
        label = self.attempt_label(session)
        stop = Event()
        begin = time.monotonic()

        def animate():
            frames = ("◐", "◓", "◑", "◒")
            index = 0
            while not stop.wait(0.15):
                elapsed = int(time.monotonic() - begin)
                print(f"\r\033[2K{label}  {frames[index % 4]} 응답 대기 {elapsed//60:02d}:{elapsed%60:02d}",
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
            print(f"  생성 미완료 · {self.clean(failure)} · 미채점, 다음 문제로 이동", file=sys.stderr)
            return
        feedback = row["feedback"]
        color = "\033[32m" if row["correct"] else "\033[33m"
        label = f"{color}{feedback}\033[0m" if self.color else feedback
        final = self.format_answer(row.get("answer"))
        print(f"  {label}  ·  {elapsed:.1f}s  ·  토큰 {usage['input_tokens']:,} → {usage['output_tokens']:,}", file=sys.stderr)
        print(f"  제출한 삭제  {self.clean(row.get('moves') or '해석 불가')}", file=sys.stderr)
        print(f"  제출한 답안  {final}  ·  연속 정답 {row['success_streak']}/{self.settings['required_streak']}", file=sys.stderr)
        if row["violation_steps"]:
            print("  위반  " + ", ".join(map(str, row["violation_steps"])) + "번째 행동", file=sys.stderr)
        print(file=sys.stderr, flush=True)

    def summary(self, report):
        summary = report.get("summary", {})
        print("\n─ 평가 완료 ─" if report.get("complete") else "\n─ 진행 기록 저장 ─", file=sys.stderr)
        print(f"제출 {len(report.get('submissions', [])):,}건", file=sys.stderr)
        mpe = summary.get("mpe")
        if mpe is not None:
            print(f"MPE {100*mpe:.2f}%", file=sys.stderr)


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


def execute(items, settings, directory, *, resume=False, client=None, receive=providers.receive):
    """Every saved response is reused before another provider request is made."""
    summary_mode = diagnostics.mode(settings)
    owned_client = client is None
    with lock_directory(directory):
        if resume:
            frozen = read_json(directory / "protocol.json")
            if frozen["fingerprint"] != implementation_fingerprint(settings):
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
                "fingerprint": implementation_fingerprint(settings),
            })
            save_progress(directory, session, native_item, native_history)
        terminal = Terminal(settings, items, directory)
        trials = directory / "submissions"
        trials.mkdir(exist_ok=True)
        summaries = {}
        if summary_mode != "off":
            for path in sorted(trials.glob("*_reasoning_summary.json")):
                saved_summary = read_json(path)
                summaries[saved_summary["submission"]] = saved_summary
            write_text(directory / "reasoning_log.txt", diagnostics.render_log(summaries.values()))
        try:
            while not session.complete:
                current_id = session.current_item["id"]
                if summary_mode != "off":
                    current = session.current_record
                    diagnostic_context = {
                        "id": current_id, "round": current["round"], "attempt": session.attempt,
                        "input": current["input"], "target": current["target"],
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
                        if client is None:
                            client = providers.make_client(settings)
                        attempts = len(list(trials.glob(f"{number:05d}_transport_*_started.json"))) + 1
                        transport = Path(str(stem) + f"_transport_{attempts:03d}")
                        write_json(Path(str(transport) + "_started.json"), {"at": timestamp(), "request_sha256": digest(request)})
                        begin = time.monotonic()
                        with terminal.waiting(session):
                            raw = receive(client, settings["provider"], request, Path(str(transport) + "_stream.jsonl"))
                        envelope = {"response": raw, "elapsed_seconds": time.monotonic()-begin,
                                    "request_sha256": digest(request), "received_at": timestamp()}
                    write_json(response_path, envelope)
                raw = envelope["response"]
                text, failure = providers.visible_text(raw, settings["provider"])
                continuation = [] if failure else providers.continuation_items(raw, settings["provider"])
                row = session.submit(text)
                if failure:
                    native_item, native_history = None, []
                else:
                    native_item = current_id
                    native_history = submitted_history + continuation + [{"role": "user", "content": row["feedback"]}]
                usage = providers.token_usage(raw, settings["provider"])
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
                        "generation_failure": failure,
                    }
                    write_json(Path(str(stem) + "_reasoning_summary.json"), diagnostic)
                    summaries[number] = diagnostic
                    write_text(directory / "reasoning_log.txt", diagnostics.render_log(summaries.values()))
                save_progress(directory, session, native_item, native_history)
                terminal.submission(row, envelope["elapsed_seconds"], usage, failure)
                if summary_mode != "off" and not summary_text:
                    print("  추론 요약 없음 · API가 요약 텍스트를 반환하지 않았습니다.\n", file=sys.stderr)
        finally:
            if owned_client and client is not None:
                client.close()
            write_json(directory / "report.json", session.report())
            terminal.summary(session.report())
    return session.report()


def main(argv=None):
    parser = argparse.ArgumentParser(description="SameGame 모델 평가")
    parser.add_argument("--config", type=Path, help="OpenAI 또는 Anthropic TOML 설정")
    parser.add_argument("--profile", choices=tuple(DEFAULT_CONFIGS), help="sol, opus, sonnet")
    parser.add_argument("--item", action="append", help="문제 ID, 여러 번 지정 가능")
    parser.add_argument("--limit", type=int, help="실행할 문제 수")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="새 결과 폴더")
    destination.add_argument("--resume", type=Path, help="이전 결과 폴더에서 재개")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 첫 요청 확인")
    parser.add_argument("--reasoning-summary", nargs="?", const="auto", choices=diagnostics.SUMMARY_MODES,
                        help="OpenAI 추론 요약 저장 (값 생략 시 auto, 기본 off)")
    args = parser.parse_args(argv)
    try:
        if args.resume:
            if args.config or args.profile or args.item or args.limit is not None or args.reasoning_summary is not None:
                raise ValueError("--resume은 저장된 설정과 문제 목록을 사용합니다.")
            directory = args.resume.expanduser().resolve()
            frozen = read_json(directory / "protocol.json")
            settings, items = frozen["settings"], frozen["items"]
        else:
            settings = load_settings(args.config, args.profile)
            if args.reasoning_summary is not None:
                settings["reasoning_summary"] = args.reasoning_summary
            items = select_items(settings, args.item, args.limit)
            default = PROJECT_DIR / "runs" / f"{settings['profile']}_{datetime.now():%Y%m%d_%H%M%S}"
            directory = (args.output or default).expanduser().resolve()
        diagnostics.mode(settings)
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
            print(json.dumps({"profile": settings["profile"], "problems": len(items),
                              "request": request}, ensure_ascii=False, indent=2))
            return 0
        execute(items, settings, directory, resume=bool(args.resume))
        print(f"결과 저장 · {directory / 'report.json'}")
        return 0
    except KeyboardInterrupt:
        print("\n중단했습니다. --resume으로 저장된 실행을 이어갈 수 있습니다.", file=sys.stderr)
        return 130
    except (providers.ProviderError, ValueError, OSError, KeyError) as exc:
        print(f"실행 중단 · {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
