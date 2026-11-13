"""Generate or validate the SameGame benchmark."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from src.dataset import DATA_DIR, SEED, generate, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if args.validate_only:
        public = json.loads((directory / "problems.json").read_text())
        key = json.loads((directory / "answer_key.json").read_text())
        report = validate(public, key)
    else:
        public, key, report = generate(args.seed, args.count, args.workers)
    directory.mkdir(parents=True, exist_ok=True)
    documents = {"design_report.json": report}
    if not args.validate_only:
        documents.update({"problems.json": public, "answer_key.json": key})
    for name, document in documents.items():
        (directory / name).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {report['items']} items / {report['rounds']} rounds / {report['cases']} cases")


if __name__ == "__main__":
    main()
