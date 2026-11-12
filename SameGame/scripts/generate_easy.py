"""Generate or validate the separate 50-item easier-R1 benchmark."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.easy_dataset import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="defaults to data/easy_r1")
    parser.add_argument("--validate-only", action="store_true", help="verify existing files without changing them")
    args = parser.parse_args()
    try:
        report = build(args.output_dir, validate_only=args.validate_only)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({key: report[key] for key in ("variant", "changed_r1", "unchanged_rounds", "summary")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
