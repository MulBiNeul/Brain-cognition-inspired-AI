"""Generate or validate the separate 50-item all-round easier benchmark."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.easy_all_dataset import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="defaults to data/easy_all")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        report = build(args.output_dir, validate_only=args.validate_only)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({key: report[key] for key in ("variant", "changed_from_original", "regenerated_r2_to_r5", "summary")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
