"""Serve the participant HTML and persist private evaluation sessions locally."""

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .catalog import load_catalog
from .evaluate import INSTRUCTIONS, PER_DENOMINATOR, PERSISTENT_ERROR_BASIS
from .session import ExperimentSession
from ._paths import DATA_DIR, PACKAGE_DIR, PROJECT_DIR, WEB_DIR


TOKEN = re.compile(r"[a-f0-9]{48}")
REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def fingerprint(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RequestError(Exception):
    def __init__(self, status, message, session=None):
        self.status = status
        self.payload = {"error": message}
        if session is not None:
            self.payload["session"] = session


class HumanApp:
    def __init__(self, items, output_dir, max_attempts, required_streak, *, seed):
        self.items = deepcopy(items)
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.protocol = {"max_attempts": max_attempts, "required_streak": required_streak}
        self.implementation = fingerprint({
            name: hashlib.sha256((PACKAGE_DIR / name).read_bytes()).hexdigest()
            for name in ("evaluate.py", "grade.py", "session.py", "human_server.py")
        })
        ExperimentSession(self.items, seed=seed, **self.protocol)
        self.lock = threading.RLock()
        self.sessions = {}

    @staticmethod
    def restore(data, count=None):
        engine = ExperimentSession(data["items"], seed=data["selection_seed"], **data["protocol"])
        for raw in data["responses"][:count]:
            engine.submit(raw)
        return engine

    @staticmethod
    def public(data, engine, *, turn_id=None, last_feedback=None):
        current = engine.current_record
        if current is not None:
            item_id, number = engine.current_item["id"], current["round"]
        elif engine.records:
            item_id, number = engine.records[-1]["id"], engine.records[-1]["round"]
        else:
            item_id, number = None, None
        history = []
        for record in engine.records:
            if (record["id"], record["round"]) == (item_id, number):
                submitted = json.loads(record["raw"])
                history.append({"attempt": record["attempt"], "input": record["input"], "moves": submitted["moves"],
                                "answer": submitted["answer"], "feedback": record["feedback"]})
        return {
            "token": data["token"], "turn_id": turn_id or data["turn_id"],
            "instructions": INSTRUCTIONS, "input": current["input"] if current else None,
            # Group attempts without exposing the problem ID, goal, or rule.
            "round_token": fingerprint([data["token"], item_id, number])[:32],
            "history": history, "complete": engine.complete,
            "last_feedback": data["last_feedback"] if last_feedback is None else last_feedback,
        }

    def path(self, token):
        if not isinstance(token, str) or not TOKEN.fullmatch(token):
            raise RequestError(404, "Session not found.")
        return self.output_dir / "sessions" / (token + ".json")

    def save(self, data, engine):
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        data["responses"] = list(engine.responses)
        data["report"] = engine.report()
        atomic_json(self.path(data["token"]), data)

    def load(self, token):
        path = self.path(token)
        if token not in self.sessions:
            if not path.is_file():
                raise RequestError(404, "Session not found.")
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data.get("schema_version") != 1 or data["token"] != token
                    or data["dataset_sha256"] != fingerprint(data["items"])):
                raise RequestError(409, "The saved session could not be verified.")
            if data["instructions_sha256"] != fingerprint(INSTRUCTIONS):
                raise RequestError(409, "The instructions have changed. Start a new session.")
            if data["implementation_sha256"] != self.implementation:
                raise RequestError(409, "The evaluation code has changed. Start a new session.")
            self.sessions[token] = (data, self.restore(data))
        return self.sessions[token]

    def create(self, body):
        participant = body.get("participant_id", "")
        if not isinstance(participant, str) or len(participant) > 100:
            raise RequestError(400, "Participant ID must contain at most 100 characters.")
        with self.lock:
            token = secrets.token_hex(24)
            data = {
                "schema_version": 1, "token": token, "turn_id": secrets.token_hex(16),
                "selection_seed": self.seed,
                "participant_id": participant, "created_at": datetime.now(timezone.utc).isoformat(),
                "items": deepcopy(self.items), "dataset_sha256": fingerprint(self.items),
                "instructions_sha256": fingerprint(INSTRUCTIONS), "protocol": deepcopy(self.protocol),
                "implementation_sha256": self.implementation,
                "persistent_error_basis": PERSISTENT_ERROR_BASIS,
                "per_denominator": PER_DENOMINATOR,
                "responses": [], "receipts": {}, "last_feedback": None,
            }
            engine = ExperimentSession(data["items"], seed=self.seed, **self.protocol)
            self.save(data, engine)
            self.sessions[token] = (data, engine)
            return self.public(data, engine)

    def get(self, token):
        with self.lock:
            return self.public(*self.load(token))

    def submit(self, body):
        required = {"token", "turn_id", "request_id", "moves", "answer"}
        if set(body) != required or not all(isinstance(body[key], str) for key in required):
            raise RequestError(400, "Provide token, turn_id, request_id, moves, and answer as strings.")
        if not REQUEST_ID.fullmatch(body["request_id"]):
            raise RequestError(400, "Invalid request ID.")
        with self.lock:
            data, engine = self.load(body["token"])
            digest = fingerprint(body)
            receipt = data["receipts"].get(body["request_id"])
            if receipt is not None:
                if receipt["digest"] != digest:
                    raise RequestError(409, "This request ID was already used.", self.public(data, engine))
                earlier = self.restore(data, receipt["response_count"])
                return self.public(data, earlier, turn_id=receipt["turn_id"],
                                   last_feedback=receipt["feedback"])
            if engine.complete or body["turn_id"] != data["turn_id"]:
                raise RequestError(409, "This attempt has already been submitted.", self.public(data, engine))
            raw = json.dumps({"moves": body["moves"], "answer": body["answer"]}, ensure_ascii=False)
            try:
                record = engine.submit(raw)
                data["turn_id"] = secrets.token_hex(16)
                data["last_feedback"] = record["feedback"]
                data["receipts"][body["request_id"]] = {
                    "digest": digest, "response_count": len(engine.responses),
                    "turn_id": data["turn_id"], "feedback": record["feedback"],
                }
                self.save(data, engine)
            except Exception:
                self.sessions.pop(data["token"], None)
                raise
            print(f"  {data['token'][:8]}  │  {record['id']} R{record['round']}  "
                  f"│  attempt {record['attempt']:2d}  │  {record['feedback']}", flush=True)
            return self.public(data, engine)


class HumanHandler(BaseHTTPRequestHandler):
    server_version = "SameGame"

    def log_message(self, *_):
        pass

    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, value):
        self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        request = urlsplit(self.path)
        try:
            if request.path in ("/", "/human.html"):
                self.send_bytes(200, (WEB_DIR / "human.html").read_bytes(), "text/html; charset=utf-8")
            elif request.path == "/api/info":
                self.send_json(200, {"instructions": INSTRUCTIONS})
            elif request.path == "/api/session":
                token = parse_qs(request.query).get("token", [""])[0]
                self.send_json(200, self.server.app.get(token))
            else:
                self.send_json(404, {"error": "Not found."})
        except RequestError as error:
            self.send_json(error.status, error.payload)
        except (OSError, ValueError, KeyError):
            self.send_json(500, {"error": "The session could not be loaded. Please retry."})

    def do_POST(self):
        try:
            if self.path not in ("/api/session", "/api/submit"):
                raise RequestError(404, "Not found.")
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise RequestError(403, "Open the evaluation page from this server.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 131072:
                raise RequestError(400, "Invalid request size.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise RequestError(400, "Expected a JSON object.")
            result = self.server.app.create(body) if self.path == "/api/session" else self.server.app.submit(body)
            self.send_json(200, result)
        except RequestError as error:
            self.send_json(error.status, error.payload)
        except (ValueError, TypeError):
            self.send_json(400, {"error": "Invalid JSON request."})
        except (OSError, KeyError):
            self.send_json(500, {"error": "The response could not be saved. Retry the same submission."})


class HumanServer(ThreadingHTTPServer):
    def server_close(self):
        super().server_close()
        lock = getattr(self, "output_lock", None)
        if lock is not None:
            lock.close()
            self.output_lock = None


def create_server(items, output_dir, host="127.0.0.1", port=0, *, seed,
                  max_attempts=10, required_streak=2):
    app = HumanApp(items, output_dir, max_attempts, required_streak, seed=seed)
    app.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (app.output_dir / ".server.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        server = HumanServer((host, port), HumanHandler)
    except Exception:
        lock.close()
        raise
    server.output_lock = lock
    server.app = app
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "runs" / "human")
    parser.add_argument("--item", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--required-streak", type=int, default=2)
    args = parser.parse_args()
    try:
        items, seed = load_catalog(args.data_dir, args.item, args.limit)
        server = create_server(items, args.output, args.host, args.port, seed=seed,
                               max_attempts=args.max_attempts, required_streak=args.required_streak)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    host, port = server.server_address
    print(f"\n  SameGame · Human evaluation\n"
          f"  Open     http://{host}:{port}\n"
          f"  Problems {len(items)}  ·  Attempts {args.max_attempts}  ·  Streak {args.required_streak}\n"
          f"  Results  {args.output.resolve()}\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped. Saved participants can resume.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
