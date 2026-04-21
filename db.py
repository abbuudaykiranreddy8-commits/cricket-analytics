from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from cricket_analytics.config import DB_PATH, ensure_directories


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    ensure_directories()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


@contextmanager
def db_session(db_path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    event_name TEXT,
    season TEXT,
    match_type TEXT,
    team1 TEXT,
    team2 TEXT,
    venue TEXT,
    city TEXT,
    match_date TEXT,
    toss_winner TEXT,
    toss_decision TEXT,
    winner TEXT,
    result_type TEXT,
    result_margin INTEGER,
    source_file TEXT,
    data_hash TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    unique_name TEXT,
    role TEXT,
    batting_style TEXT,
    bowling_style TEXT,
    bowling_type TEXT,
    bowling_arm TEXT
);

CREATE TABLE IF NOT EXISTS appearances (
    match_id TEXT NOT NULL,
    team TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    PRIMARY KEY (match_id, team, player_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL,
    innings INTEGER NOT NULL,
    over INTEGER NOT NULL,
    ball INTEGER NOT NULL,
    ball_label TEXT NOT NULL,
    phase TEXT NOT NULL,
    batting_team TEXT NOT NULL,
    bowling_team TEXT NOT NULL,
    batter_id TEXT,
    batter TEXT NOT NULL,
    bowler_id TEXT,
    bowler TEXT NOT NULL,
    non_striker_id TEXT,
    non_striker TEXT,
    batter_runs INTEGER NOT NULL,
    extras INTEGER NOT NULL,
    total_runs INTEGER NOT NULL,
    is_legal_ball INTEGER NOT NULL,
    is_dot_ball INTEGER NOT NULL,
    boundary_value INTEGER NOT NULL,
    dismissal INTEGER NOT NULL,
    dismissal_kind TEXT,
    player_out_id TEXT,
    player_out TEXT,
    venue TEXT,
    season TEXT,
    match_date TEXT,
    bowler_style TEXT,
    bowling_type TEXT,
    bowling_arm TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    source_file TEXT PRIMARY KEY,
    data_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS upcoming_matches (
    schedule_match_id TEXT PRIMARY KEY,
    season TEXT NOT NULL,
    match_no INTEGER NOT NULL,
    team_a TEXT NOT NULL,
    team_b TEXT NOT NULL,
    venue TEXT NOT NULL,
    venue_display TEXT NOT NULL,
    match_date TEXT NOT NULL,
    day_name TEXT,
    start_time TEXT,
    status TEXT NOT NULL,
    source TEXT,
    ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date DESC);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(team1, team2);
CREATE INDEX IF NOT EXISTS idx_appearances_team_match ON appearances(team, match_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_match ON deliveries(match_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_batter_bowler ON deliveries(batter_id, bowler_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_teams ON deliveries(batting_team, bowling_team);
CREATE INDEX IF NOT EXISTS idx_deliveries_venue ON deliveries(venue);
CREATE INDEX IF NOT EXISTS idx_deliveries_phase ON deliveries(phase);
CREATE INDEX IF NOT EXISTS idx_upcoming_matches_date ON upcoming_matches(match_date, start_time);
"""


def initialize_database(db_path: Path | str = DB_PATH) -> None:
    with db_session(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def reset_database(db_path: Path | str = DB_PATH) -> None:
    with db_session(db_path) as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS deliveries;
            DROP TABLE IF EXISTS appearances;
            DROP TABLE IF EXISTS ingestion_log;
            DROP TABLE IF EXISTS players;
            DROP TABLE IF EXISTS matches;
            """
        )
        conn.executescript(SCHEMA_SQL)
