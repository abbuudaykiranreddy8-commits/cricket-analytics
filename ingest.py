from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

from cricket_analytics.config import (
    CACHE_DIR,
    DEFAULT_CRICSHEET_URL,
    DEFAULT_REGISTER_URL,
    RAW_DIR,
    ensure_directories,
)
from cricket_analytics.db import db_session, reset_database
from cricket_analytics.normalize import normalize_team_name


LOGGER = logging.getLogger(__name__)


@dataclass
class IngestResult:
    matches: int = 0
    deliveries: int = 0
    players: int = 0
    skipped_files: int = 0
    zip_path: str = ""


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def ensure_register_csv(register_csv: Path | None = None, register_url: str = DEFAULT_REGISTER_URL) -> Path:
    ensure_directories()
    register_csv = register_csv or CACHE_DIR / "people.csv"
    if not register_csv.exists():
        download_file(register_url, register_csv)
    return register_csv


def download_latest_ipl_zip(force: bool = False, url: str = DEFAULT_CRICSHEET_URL) -> Path:
    ensure_directories()
    zip_path = RAW_DIR / "ipl_json.zip"
    if force or not zip_path.exists():
        LOGGER.info("Downloading IPL dataset from %s", url)
        download_file(url, zip_path)
    return zip_path


def classify_phase(over_number: int) -> str:
    if over_number <= 6:
        return "Powerplay"
    if over_number <= 15:
        return "Middle"
    return "Death"


def load_register_profiles(register_csv: Path) -> dict[str, dict[str, str]]:
    profiles: dict[str, dict[str, str]] = {}
    if not register_csv.exists():
        return profiles
    with register_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            identifier = row.get("identifier")
            if not identifier:
                continue
            profiles[identifier] = {
                "name": row.get("name", ""),
                "unique_name": row.get("unique_name", ""),
            }
    return profiles


def upsert_player(conn, player_id: str, name: str, profile: dict[str, str] | None = None) -> None:
    profile = profile or {}
    conn.execute(
        """
        INSERT INTO players (player_id, name, unique_name)
        VALUES (?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            name = excluded.name,
            unique_name = COALESCE(excluded.unique_name, players.unique_name)
        """,
        (player_id, name, profile.get("unique_name") or None),
    )


def parse_result(info: dict) -> tuple[str | None, str | None, int | None]:
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    by = outcome.get("by", {})
    if "runs" in by:
        return winner, "runs", int(by["runs"])
    if "wickets" in by:
        return winner, "wickets", int(by["wickets"])
    if outcome.get("result"):
        return winner, outcome["result"], None
    return winner, None, None


def parse_match_date(info: dict) -> str | None:
    dates = info.get("dates", [])
    if not dates:
        return None
    return min(dates)


def ingest_json_bytes(conn, payload: bytes, source_name: str, profiles: dict[str, dict[str, str]]) -> tuple[int, int, int]:
    data = json.loads(payload.decode("utf-8"))
    info = data.get("info", {})
    registry = info.get("registry", {}).get("people", {})
    teams = info.get("teams", [])
    match_id = Path(source_name).stem
    event = info.get("event", {})
    team1 = normalize_team_name(teams[0]) if len(teams) > 0 else None
    team2 = normalize_team_name(teams[1]) if len(teams) > 1 else None
    match_date = parse_match_date(info)
    winner, result_type, result_margin = parse_result(info)
    winner = normalize_team_name(winner)
    data_hash = hashlib.sha256(payload).hexdigest()

    conn.execute(
        """
        INSERT OR REPLACE INTO matches (
            match_id, event_name, season, match_type, team1, team2, venue, city, match_date,
            toss_winner, toss_decision, winner, result_type, result_margin, source_file, data_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            match_id,
            event.get("name"),
            info.get("season"),
            info.get("match_type"),
            team1,
            team2,
            info.get("venue"),
            info.get("city"),
            match_date,
            normalize_team_name(info.get("toss", {}).get("winner")),
            info.get("toss", {}).get("decision"),
            winner,
            result_type,
            result_margin,
            source_name,
            data_hash,
        ),
    )

    player_count = 0
    for team, members in info.get("players", {}).items():
        normalized_team = normalize_team_name(team)
        for name in members:
            player_id = registry.get(name, name)
            upsert_player(conn, player_id, name, profiles.get(player_id))
            conn.execute(
                """
                INSERT OR REPLACE INTO appearances (match_id, team, player_id, player_name)
                VALUES (?, ?, ?, ?)
                """,
                (match_id, normalized_team, player_id, name),
            )
            player_count += 1

    delivery_count = 0
    for innings_index, innings in enumerate(data.get("innings", []), start=1):
        batting_team = normalize_team_name(innings.get("team"))
        bowling_team = next((team for team in [team1, team2] if team != batting_team), None)
        for over_block in innings.get("overs", []):
            over_number = int(over_block["over"]) + 1
            deliveries = over_block.get("deliveries", [])
            for ball_index, delivery in enumerate(deliveries, start=1):
                batter = delivery.get("batter")
                bowler = delivery.get("bowler")
                non_striker = delivery.get("non_striker")
                batter_id = registry.get(batter, batter)
                bowler_id = registry.get(bowler, bowler)
                non_striker_id = registry.get(non_striker, non_striker)
                upsert_player(conn, batter_id, batter, profiles.get(batter_id))
                upsert_player(conn, bowler_id, bowler, profiles.get(bowler_id))
                if non_striker:
                    upsert_player(conn, non_striker_id, non_striker, profiles.get(non_striker_id))
                extras_map = delivery.get("extras", {})
                legal = 1 if not any(key in extras_map for key in ("wides", "noballs")) else 0
                batter_runs = int(delivery.get("runs", {}).get("batter", 0))
                extras = int(delivery.get("runs", {}).get("extras", 0))
                total_runs = int(delivery.get("runs", {}).get("total", 0))
                wickets = delivery.get("wickets", [])
                dismissal_kind = wickets[0].get("kind") if wickets else None
                player_out = wickets[0].get("player_out") if wickets else None
                player_out_id = registry.get(player_out, player_out) if player_out else None
                phase = classify_phase(over_number)
                boundary_value = batter_runs if batter_runs in (4, 6) else 0
                delivery_id = f"{match_id}_{innings_index}_{over_number}_{ball_index}"

                conn.execute(
                    """
                    INSERT OR REPLACE INTO deliveries (
                        delivery_id, match_id, innings, over, ball, ball_label, phase, batting_team, bowling_team,
                        batter_id, batter, bowler_id, bowler, non_striker_id, non_striker, batter_runs, extras,
                        total_runs, is_legal_ball, is_dot_ball, boundary_value, dismissal, dismissal_kind,
                        player_out_id, player_out, venue, season, match_date
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        delivery_id,
                        match_id,
                        innings_index,
                        over_number,
                        ball_index,
                        f"{over_number}.{ball_index}",
                        phase,
                        batting_team,
                        bowling_team,
                        batter_id,
                        batter,
                        bowler_id,
                        bowler,
                        non_striker_id,
                        non_striker,
                        batter_runs,
                        extras,
                        total_runs,
                        legal,
                        1 if total_runs == 0 else 0,
                        boundary_value,
                        1 if wickets else 0,
                        dismissal_kind,
                        player_out_id,
                        player_out,
                        info.get("venue"),
                        info.get("season"),
                        match_date,
                    ),
                )
                delivery_count += 1

    conn.execute(
        """
        INSERT OR REPLACE INTO ingestion_log (source_file, data_hash, ingested_at)
        VALUES (?, ?, ?)
        """,
        (source_name, data_hash, datetime.now(timezone.utc).isoformat()),
    )
    return 1, delivery_count, player_count


def ingest_zip(zip_path: Path, register_csv: Path | None = None) -> IngestResult:
    reset_database()
    profiles = load_register_profiles(register_csv) if register_csv else {}
    result = IngestResult(zip_path=str(zip_path))
    with db_session() as conn:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".json"):
                    continue
                payload = archive.read(info.filename)
                matches, deliveries, players = ingest_json_bytes(conn, payload, info.filename, profiles)
                result.matches += matches
                result.deliveries += deliveries
                result.players += players
    infer_roles_and_styles()
    LOGGER.info(
        "Rebuilt database from %s with %s matches and %s deliveries.",
        zip_path,
        result.matches,
        result.deliveries,
    )
    return result


def infer_roles_and_styles() -> None:
    with db_session() as conn:
        stats = conn.execute(
            """
            SELECT
                p.player_id,
                p.name,
                SUM(CASE WHEN d.batter_id = p.player_id THEN d.is_legal_ball ELSE 0 END) AS balls_faced,
                SUM(CASE WHEN d.batter_id = p.player_id THEN d.batter_runs ELSE 0 END) AS batting_runs,
                SUM(CASE WHEN d.bowler_id = p.player_id THEN d.is_legal_ball ELSE 0 END) AS balls_bowled,
                SUM(CASE WHEN d.bowler_id = p.player_id THEN d.total_runs ELSE 0 END) AS runs_conceded
            FROM players p
            LEFT JOIN deliveries d ON d.batter_id = p.player_id OR d.bowler_id = p.player_id
            GROUP BY p.player_id, p.name
            """
        ).fetchall()
        for row in stats:
            balls_faced = row["balls_faced"] or 0
            balls_bowled = row["balls_bowled"] or 0
            if balls_bowled >= 24 and balls_faced >= 24:
                role = "All-rounder"
            elif balls_bowled >= max(12, balls_faced * 1.5) or (balls_bowled >= 6 and balls_bowled > balls_faced):
                role = "Bowler"
            elif balls_faced >= max(12, balls_bowled * 1.5) or (balls_faced >= 6 and balls_faced > balls_bowled):
                role = "Batsman"
            else:
                role = "Player"
            conn.execute(
                "UPDATE players SET role = ? WHERE player_id = ?",
                (role, row["player_id"]),
            )


def build_database(
    zip_path: Path | None = None,
    register_csv: Path | None = None,
    download: bool = False,
    force_download: bool = False,
) -> IngestResult:
    ensure_directories()
    register_csv = ensure_register_csv(register_csv)
    if force_download or download or not (zip_path or (RAW_DIR / "ipl_json.zip").exists()):
        zip_path = download_latest_ipl_zip(force=force_download)
    else:
        zip_path = zip_path or (RAW_DIR / "ipl_json.zip")
    result = ingest_zip(zip_path, register_csv)
    print(
        f"Rebuilt SQLite database from {zip_path} with {result.matches} matches and {result.deliveries} deliveries."
    )
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest Cricsheet JSON into SQLite.")
    parser.add_argument("--zip-path", type=Path, default=None, help="Path to Cricsheet JSON zip file.")
    parser.add_argument("--register-csv", type=Path, default=None, help="Path to Cricsheet register people.csv.")
    parser.add_argument("--download", action="store_true", help="Download the IPL zip automatically if it is missing.")
    parser.add_argument("--force-download", action="store_true", help="Always fetch a fresh IPL zip before rebuilding.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = build_database(
        args.zip_path,
        args.register_csv,
        download=args.download,
        force_download=args.force_download,
    )
    print(
        json.dumps(
            {
                "zip_path": result.zip_path,
                "matches_ingested": result.matches,
                "deliveries_ingested": result.deliveries,
                "players_seen": result.players,
                "skipped_files": result.skipped_files,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
