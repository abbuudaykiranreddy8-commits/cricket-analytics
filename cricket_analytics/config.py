from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DB_DIR = DATA_DIR / "db"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DB_DIR / "cricket_analytics.db"
DEFAULT_CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"
DEFAULT_REGISTER_URL = "https://cricsheet.org/register/people.csv"
REGISTER_NAMES_PATH = CACHE_DIR / "names.csv"
REGISTER_PEOPLE_PATH = CACHE_DIR / "people.csv"
SCHEDULE_PDF_PATH = CACHE_DIR / "ipl_2026_schedule.pdf"
DEFAULT_SCHEDULE_PDF_URL = "https://documents.iplt20.com/smart-images/1774525332894_TATA_IPL_2026-Schedule.pdf"


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DIR, DB_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
