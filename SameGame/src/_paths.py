"""Project locations independent of the current working directory."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
WEB_DIR = PROJECT_DIR / "web"
