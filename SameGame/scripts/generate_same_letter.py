"""Generate or validate same-letter v2 while preserving different-letter items."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.same_letter_dataset import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="defaults to data/easy_same_letter_v2")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        report = build(args.output_dir, validate_only=args.validate_only)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({key: report[key] for key in ("variant", "changed_from_easy_all", "summary",
          "r1_longest_exceptions", "recoverable_old_rule_violation_distribution")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
