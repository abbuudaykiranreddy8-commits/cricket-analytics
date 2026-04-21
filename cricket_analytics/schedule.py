from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader

from cricket_analytics.config import DEFAULT_SCHEDULE_PDF_URL, SCHEDULE_PDF_PATH
from cricket_analytics.db import db_session, initialize_database
from cricket_analytics.normalize import normalize_team_name, normalize_venue_name


MATCH_HEADER = "Match No Date Day Start Home Away Venue"
DATE_PATTERN = re.compile(r"^\d{2}-[A-Z]{3}-\d{2}$")


def download_schedule_pdf(force: bool = False, url: str = DEFAULT_SCHEDULE_PDF_URL) -> Path:
    destination = SCHEDULE_PDF_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not destination.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        destination.write_bytes(response.content)
    return destination


def _parse_schedule_page(page_text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    header_idx = next((index for index, line in enumerate(lines) if line.startswith(MATCH_HEADER)), None)
    if header_idx is None:
        return []

    numbers: list[str] = []
    idx = header_idx - 1
    while idx >= 0 and lines[idx].isdigit():
        numbers.append(lines[idx])
        idx -= 1
    numbers.reverse()
    if not numbers:
        return []

    count = len(numbers)
    dates = lines[header_idx - (2 * count) : header_idx - count]
    days = lines[header_idx - (3 * count) : header_idx - (2 * count)]
    starts = lines[header_idx - (4 * count) : header_idx - (3 * count)]
    away = lines[header_idx - (5 * count) : header_idx - (4 * count)]
    venues = lines[header_idx - (6 * count) : header_idx - (5 * count)]
    home = lines[header_idx - (7 * count) : header_idx - (6 * count)]

    rows: list[dict[str, str]] = []
    for match_no, match_date, day_name, start_time, away_team, home_team, venue_display in zip(
        numbers,
        dates,
        days,
        starts,
        away,
        home,
        venues,
        strict=True,
    ):
        if not DATE_PATTERN.match(match_date):
            continue
        rows.append(
            {
                "match_no": match_no,
                "match_date": match_date,
                "day_name": "Sat" if day_name == "Satt" else day_name,
                "start_time": start_time,
                "team_a": normalize_team_name(home_team),
                "team_b": normalize_team_name(away_team),
                "venue_display": venue_display,
                "venue": normalize_venue_name(venue_display),
            }
        )
    return rows


def parse_schedule_pdf(pdf_path: Path | str = SCHEDULE_PDF_PATH, season: str = "2026") -> pd.DataFrame:
    reader = PdfReader(str(pdf_path))
    rows: list[dict[str, str]] = []
    for page in reader.pages:
        rows.extend(_parse_schedule_page(page.extract_text() or ""))
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return pd.DataFrame(
            columns=[
                "schedule_match_id",
                "season",
                "match_no",
                "team_a",
                "team_b",
                "venue",
                "venue_display",
                "match_date",
                "day_name",
                "start_time",
                "status",
                "source",
            ]
        )
    schedule["season"] = season
    schedule["match_no"] = schedule["match_no"].astype(int)
    schedule["match_date"] = pd.to_datetime(schedule["match_date"], format="%d-%b-%y").dt.strftime("%Y-%m-%d")
    schedule["schedule_match_id"] = schedule["season"] + "-" + schedule["match_no"].astype(str).str.zfill(2)
    schedule["status"] = "Not Started"
    schedule["source"] = str(pdf_path)
    return schedule.sort_values(["match_date", "match_no"]).reset_index(drop=True)


def store_upcoming_matches(schedule_df: pd.DataFrame, as_of_date: date | None = None) -> pd.DataFrame:
    initialize_database()
    as_of_date = as_of_date or date.today()
    upcoming = schedule_df[pd.to_datetime(schedule_df["match_date"]).dt.date > as_of_date].copy()
    ingested_at = datetime.now(timezone.utc).isoformat()
    with db_session() as conn:
        conn.execute("DELETE FROM upcoming_matches")
        for row in upcoming.to_dict("records"):
            conn.execute(
                """
                INSERT INTO upcoming_matches (
                    schedule_match_id, season, match_no, team_a, team_b, venue, venue_display,
                    match_date, day_name, start_time, status, source, ingested_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["schedule_match_id"],
                    row["season"],
                    int(row["match_no"]),
                    row["team_a"],
                    row["team_b"],
                    row["venue"],
                    row["venue_display"],
                    row["match_date"],
                    row["day_name"],
                    row["start_time"],
                    row["status"],
                    row["source"],
                    ingested_at,
                ),
            )
    return upcoming.reset_index(drop=True)


def refresh_upcoming_matches(force_download: bool = False, as_of_date: date | None = None) -> pd.DataFrame:
    pdf_path = download_schedule_pdf(force=force_download)
    schedule_df = parse_schedule_pdf(pdf_path)
    return store_upcoming_matches(schedule_df, as_of_date=as_of_date)
