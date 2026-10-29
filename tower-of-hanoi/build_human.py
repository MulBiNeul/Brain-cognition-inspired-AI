#!/usr/bin/env python3
"""Build a self-contained human assessment HTML; no server or API required."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hanoi.human import INSTRUCTIONS, ROOT, load_config

ASSETS = ("index.html", "style.css", "engine.js", "offline.js", "app.js")


def make_bundle(config: Path) -> dict:
    settings = load_config(config)
    assets = {name: hashlib.sha256((ROOT / "human_web" / name).read_bytes()).hexdigest()
              for name in ASSETS}
    bundle = {
        "format": "hanoi_human_standalone_v1",
        "dataset": settings["dataset"],
        "dataset_sha256": settings["dataset_sha256"],
        "config": {"max_attempts": settings["T"], "success_streak": settings["k"],
                   "r1_apply_k": settings["r1_apply_k"]},
        "feedback_policy": settings["feedback_policy"],
        "instructions": INSTRUCTIONS,
        "source_sha256": assets,
    }
    encoded = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    bundle["bundle_id"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return bundle


def render_html(bundle: dict) -> str:
    source = ROOT / "human_web"
    html = (source / "index.html").read_text(encoding="utf-8")
    css = (source / "style.css").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="/style.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="/app.js" defer></script>', "")
    html = html.replace('class="brand" href="/"', 'class="brand" href="#"')
    policy = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
              "img-src data: blob:; connect-src 'none'; base-uri 'none'; form-action 'none'")
    html = html.replace("<head>", f'<head>\n<meta http-equiv="Content-Security-Policy" content="{policy}">', 1)
    html = html.replace("<main>", '<main>\n<noscript>이 과제는 JavaScript를 켠 브라우저에서 열어 주세요.</noscript>', 1)
    encoded = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    scripts = [f"window.HANOI_BUNDLE={encoded};"]
    scripts.extend((source / name).read_text(encoding="utf-8")
                   for name in ("engine.js", "offline.js", "app.js"))
    for script in scripts:
        if "</script" in script.lower():
            raise ValueError("Inline JavaScript contains an unsafe script closing tag")
    return html.replace("</body>", "\n".join(f"<script>\n{s}\n</script>" for s in scripts) + "\n</body>")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/human.toml")
    parser.add_argument("--output", type=Path, default=ROOT / "Hanoi_Human_Evaluation.html")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".html":
        parser.error("--output must end in .html")
    content = render_html(make_bundle(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"HTML: {args.output.resolve()}")
    print("이 파일 하나를 브라우저에서 열면 됩니다. 결과는 화면의 JSON 저장 버튼으로 내려받습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
