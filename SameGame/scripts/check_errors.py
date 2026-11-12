"""Inspect saved SameGame errors without issuing API requests or changing run files."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.error_log import inspect_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="실행 결과 폴더, 예: runs/sol_easy_r1")
    parser.add_argument("--limit", type=int, default=10, help="표시할 최근 오류·중단 기록 수(기본 10)")
    args = parser.parse_args()
    try:
        print(inspect_run(args.directory, args.limit), end="")
    except (ValueError, OSError) as error:
        parser.exit(1, f"확인 실패: {error}\n")


if __name__ == "__main__":
    main()
