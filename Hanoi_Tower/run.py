#!/usr/bin/env python3
"""Validate configuration and authentication, then start or resume a Hanoi run."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import tempfile

from hanoi import auth, config, experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="TOML 설정 (기본: configs/sol.toml)")
    parser.add_argument("--profile", help="모델 프로필: sol, opus, sonnet (기본: sol)")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="새 결과 폴더 (기본: runs/프로필)")
    destination.add_argument("--resume", type=Path, help="저장된 응답을 복구하고 응답이 없는 요청부터 재개")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="인증·모델 조회만 확인; 생성 요청·파일 저장 없음")
    mode.add_argument("--dry-run", action="store_true", help="API 없이 요청 미리보기; 기본 저장 위치는 시스템 임시 폴더")
    # Accept older commands; --resume now includes pending-request recovery.
    parser.add_argument("--retry-pending", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.retry_pending and not args.resume:
        parser.error("--retry-pending은 --resume과 함께 사용하세요.")
    try:
        arguments = config.resolve_arguments(args.config, args.profile, args.resume)
        if not args.resume:
            settings = config.load_settings(args.config, args.profile)
            default_output = (Path(tempfile.gettempdir()) / datetime.now().strftime("hanoi_preview_%Y%m%d_%H%M%S_%f")
                              if args.dry_run else experiment.ROOT / "runs" / settings["profile"])
            directory = (args.output or default_output).expanduser().resolve()
            arguments.extend(("--output", str(directory)))
        if args.dry_run:
            arguments.append("--dry-run")
        # Keep the frozen experiment core unchanged for in-progress runs.
        # An explicit resume authorizes one new transport attempt, not a retry loop.
        if args.resume and not (args.dry_run or args.check_only):
            arguments.append("--retry-pending")
        return experiment.main(arguments, client_factory=auth.checked_client, check_only=args.check_only)
    except auth.AuthenticationCheckError as error:
        print(f"시작 전 확인 중단: {error}", file=sys.stderr)
        return error.exit_code
    except config.ConfigError as error:
        print(f"설정 오류: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"설정·데이터 확인 실패: {error}", file=sys.stderr)
        return 2
    except experiment.ExperimentError as error:
        print(f"실험 중단: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("파일 또는 실행 환경 오류로 중단했습니다. 경로와 권한을 확인하세요.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("사용자 중단. 저장된 응답은 재개할 때 재사용합니다.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
