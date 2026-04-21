from __future__ import annotations

from functools import lru_cache
from typing import Any
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from cricket_analytics.config import REGISTER_NAMES_PATH, REGISTER_PEOPLE_PATH
from cricket_analytics.current_squads import CURRENT_SQUADS
from cricket_analytics.db import get_connection
from cricket_analytics.normalize import normalize_team_name, normalize_venue_name


MANUAL_PLAYER_ALIASES = {
    "virat kohli": "ba607b88",
    "rohit sharma": "740742ef",
    "surya kumar yadav": "271f83cd",
    "suryakumar yadav": "271f83cd",
    "ruturaj gaikwad": "45a43fe2",
    "nitish kumar reddy": "aad0c365",
    "matheesha pathirana": "64839cb3",
    "rahul tripathi": "77255a9e",
    "tim seifert": "4663bd23",
    "varun chakaravarthy": "5b7ab5a9",
    "varun chakravarthy": "5b7ab5a9",
    "rasikh dar": "b8527c3d",
    "shahbaz ahamad": "f9e6e7ef",
    "shahbaz ahmed": "f9e6e7ef",
    "vyshak vijaykumar": "54e52590",
    "luke wood": "65b6943c",
}


def fetch_df(query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    df = fetch_df(
        """
        SELECT match_id, event_name, season, team1, team2, venue, city, match_date, winner
        FROM matches
        WHERE match_type = 'T20'
        ORDER BY match_date DESC
        """
    )
    if not df.empty:
        df["team1"] = df["team1"].map(normalize_team_name)
        df["team2"] = df["team2"].map(normalize_team_name)
        df["winner"] = df["winner"].map(normalize_team_name)
        df["venue"] = df["venue"].map(normalize_venue_name)
    return df


@lru_cache(maxsize=1)
def load_deliveries() -> pd.DataFrame:
    df = fetch_df(
        """
        SELECT
            d.*,
            COALESCE(p.role, 'Player') AS bowler_role,
            COALESCE(pb.role, 'Player') AS batter_role
        FROM deliveries d
        LEFT JOIN players p ON d.bowler_id = p.player_id
        LEFT JOIN players pb ON d.batter_id = pb.player_id
        """
    )
    if df.empty:
        return df
    for column in ("batting_team", "bowling_team"):
        df[column] = df[column].map(normalize_team_name)
    df["venue"] = df["venue"].map(normalize_venue_name)
    return df


@lru_cache(maxsize=1)
def get_upcoming_matches() -> pd.DataFrame:
    df = fetch_df(
        """
        SELECT
            schedule_match_id,
            season,
            match_no,
            team_a,
            team_b,
            venue,
            venue_display,
            match_date,
            day_name,
            start_time,
            status
        FROM upcoming_matches
        WHERE status = 'Not Started'
        ORDER BY match_date ASC, match_no ASC
        """
    )
    if not df.empty:
        df["team_a"] = df["team_a"].map(normalize_team_name)
        df["team_b"] = df["team_b"].map(normalize_team_name)
        df["venue"] = df["venue"].map(normalize_venue_name)
    return df


def get_upcoming_match_context(schedule_match_id: str) -> dict[str, Any] | None:
    upcoming = get_upcoming_matches()
    if upcoming.empty:
        return None
    row = upcoming[upcoming["schedule_match_id"] == schedule_match_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


@lru_cache(maxsize=1)
def load_appearances() -> pd.DataFrame:
    df = fetch_df(
        """
        SELECT match_id, team, player_id, player_name
        FROM appearances
        """
    )
    if not df.empty:
        df["team"] = df["team"].map(normalize_team_name)
    return df


def get_all_teams() -> list[str]:
    return sorted(CURRENT_SQUADS.keys())


def recent_match_pool(limit: int = 8) -> pd.DataFrame:
    matches = load_matches().copy()
    if matches.empty:
        return matches
    latest_season = matches["season"].dropna().astype(str).max()
    season_df = matches[matches["season"].astype(str) == latest_season]
    if season_df.empty:
        season_df = matches
    return season_df.head(limit)


def normalize_person_key(name: str | None) -> str:
    if not name:
        return ""
    collapsed = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return " ".join(collapsed.split())


def initials_key(name: str | None) -> str:
    tokens = normalize_person_key(name).split()
    return "".join(token[0] for token in tokens if token)


@lru_cache(maxsize=1)
def load_player_reference() -> pd.DataFrame:
    df = fetch_df("SELECT player_id, name, unique_name FROM players")
    if df.empty:
        return df
    df["key_name"] = df["name"].map(normalize_person_key)
    df["key_unique"] = df["unique_name"].map(normalize_person_key)
    return df


@lru_cache(maxsize=1)
def load_register_lookup() -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}

    def add_alias(alias: str, identifier: str) -> None:
        key = normalize_person_key(alias)
        if not key:
            return
        lookup.setdefault(key, [])
        if identifier not in lookup[key]:
            lookup[key].append(identifier)

    if REGISTER_PEOPLE_PATH.exists():
        people = pd.read_csv(REGISTER_PEOPLE_PATH)
        for _, row in people.iterrows():
            identifier = row.get("identifier")
            if not isinstance(identifier, str):
                continue
            add_alias(row.get("name"), identifier)
            add_alias(row.get("unique_name"), identifier)

    if REGISTER_NAMES_PATH.exists():
        names = pd.read_csv(REGISTER_NAMES_PATH)
        for _, row in names.iterrows():
            identifier = row.get("identifier")
            if not isinstance(identifier, str):
                continue
            add_alias(row.get("name"), identifier)
    return lookup


@lru_cache(maxsize=1)
def load_team_player_candidates() -> dict[str, pd.DataFrame]:
    matches = load_matches()
    appearances = load_appearances()
    players = load_player_reference()
    if matches.empty or appearances.empty or players.empty:
        return {}
    latest_seasons = matches["season"].dropna().astype(str).unique().tolist()[:3]
    scope_match_ids = matches[matches["season"].astype(str).isin(latest_seasons)]["match_id"].tolist()
    team_candidates: dict[str, pd.DataFrame] = {}
    scoped = appearances[appearances["match_id"].isin(scope_match_ids)].copy()
    for team in get_all_teams():
        merged = scoped[scoped["team"] == team].merge(players, on="player_id", how="left")
        if merged.empty:
            team_candidates[team] = merged
            continue
        merged["surname"] = merged["name"].fillna(merged["player_name"]).map(
            lambda value: normalize_person_key(value).split()[-1] if normalize_person_key(value) else ""
        )
        merged["candidate_initials"] = merged["name"].fillna(merged["player_name"]).map(initials_key)
        team_candidates[team] = merged
    return team_candidates


def resolve_current_player_id(team_name: str, official_name: str) -> str | None:
    register_lookup = load_register_lookup()
    reference = load_player_reference()
    key = normalize_person_key(official_name)
    if key in MANUAL_PLAYER_ALIASES:
        return MANUAL_PLAYER_ALIASES[key]
    ids = register_lookup.get(key, [])
    if len(ids) == 1:
        return ids[0]
    if len(ids) > 1 and not reference.empty:
        ref_ids = set(reference["player_id"].tolist())
        scoped = [identifier for identifier in ids if identifier in ref_ids]
        if len(scoped) == 1:
            return scoped[0]

    if not reference.empty:
        exact = reference[(reference["key_name"] == key) | (reference["key_unique"] == key)]
        if len(exact) == 1:
            return exact.iloc[0]["player_id"]
        token_set = set(key.split())
        token_exact = reference[
            reference["key_name"].map(lambda value: set(value.split()) == token_set if isinstance(value, str) else False)
            | reference["key_unique"].map(lambda value: set(value.split()) == token_set if isinstance(value, str) else False)
        ]
        if len(token_exact) == 1:
            return token_exact.iloc[0]["player_id"]

    candidates = load_team_player_candidates().get(team_name)
    if candidates is None or candidates.empty:
        candidates = reference.copy()

    official_tokens = normalize_person_key(official_name).split()
    if not official_tokens:
        return None
    surname = official_tokens[-1]
    initials = initials_key(official_name)
    first_initial = initials[:1]
    scored_rows: list[tuple[int, str]] = []
    for _, row in candidates.drop_duplicates("player_id").iterrows():
        candidate_name = row.get("name") or row.get("player_name") or ""
        candidate_key = normalize_person_key(candidate_name)
        candidate_tokens = candidate_key.split()
        candidate_initials = row.get("candidate_initials") or initials_key(candidate_name)
        score = 0
        candidate_surname = row.get("surname")
        if not candidate_surname and candidate_tokens:
            candidate_surname = candidate_tokens[-1]
        surname_similarity = SequenceMatcher(None, candidate_surname or "", surname).ratio()
        if surname_similarity < 0.72:
            continue
        if candidate_key == key:
            score += 200
        score += int(surname_similarity * 40)
        if first_initial and candidate_initials.startswith(first_initial):
            score += 40
        overlap = len(set(initials) & set(candidate_initials))
        score += overlap * 8
        score += int(row.get("match_id") is not None) * 10
        score += len(set(official_tokens) & set(candidate_tokens)) * 5
        score += int(SequenceMatcher(None, candidate_key, key).ratio() * 30)
        scored_rows.append((score, row["player_id"]))
    if not scored_rows:
        return None
    scored_rows.sort(reverse=True)
    return scored_rows[0][1] if scored_rows[0][0] > 0 else None


def _empty_squad_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "player_id",
            "name",
            "role",
            "source_name",
            "historical_data_status",
        ]
    )


def _classify_role(balls_faced: int, balls_bowled: int) -> str:
    if balls_faced >= 12 and balls_bowled >= 12:
        return "All-rounder"
    if balls_bowled >= max(12, int(balls_faced * 1.2)) or (balls_bowled >= 6 and balls_bowled > balls_faced):
        return "Bowler"
    if balls_faced >= max(12, int(balls_bowled * 1.2)) or (balls_faced >= 6 and balls_faced > balls_bowled):
        return "Batsman"
    if balls_bowled > 0 and balls_faced > 0:
        return "All-rounder"
    if balls_bowled > 0:
        return "Bowler"
    return "Batsman"


@lru_cache(maxsize=32)
def get_current_squad(team_name: str) -> pd.DataFrame:
    team_name = normalize_team_name(team_name)
    squad = CURRENT_SQUADS.get(team_name)
    if not squad:
        return _empty_squad_frame()
    rows: list[dict[str, Any]] = []
    for role_bucket, names in squad.items():
        role = role_bucket[:-1] if role_bucket.endswith("s") else role_bucket
        for official_name in names:
            player_id = resolve_current_player_id(team_name, official_name)
            rows.append(
                {
                    "player_id": player_id,
                    "name": official_name,
                    "role": role,
                    "source_name": official_name,
                    "historical_data_status": "mapped" if player_id else "no historical data",
                }
            )
    return pd.DataFrame(rows)


def build_current_player_id_map() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team_name in get_all_teams():
        squad = get_current_squad(team_name)
        if squad.empty:
            continue
        team_rows = squad.copy()
        team_rows.insert(0, "team", team_name)
        rows.extend(team_rows.to_dict("records"))
    if not rows:
        return pd.DataFrame(
            columns=["team", "name", "player_id", "role", "source_name", "historical_data_status"]
        )
    return pd.DataFrame(rows)


def get_current_squad_mapping_summary() -> pd.DataFrame:
    mapping_df = build_current_player_id_map()
    if mapping_df.empty:
        return pd.DataFrame(columns=["team", "mapped_players", "unmapped_players", "total_players"])
    summary = (
        mapping_df.groupby("team", dropna=False)
        .agg(
            mapped_players=("player_id", lambda values: int(pd.Series(values).notna().sum())),
            unmapped_players=("player_id", lambda values: int(pd.Series(values).isna().sum())),
            total_players=("team", "size"),
        )
        .reset_index()
        .sort_values("team")
    )
    return summary


def derive_recent_squad(
    team: str,
    lookback_matches: int = 5,
    seasons_back: int = 2,
    min_players: int = 15,
    max_players: int = 20,
) -> pd.DataFrame:
    del lookback_matches, seasons_back, min_players, max_players
    return get_current_squad(team)


def split_squad_roles(squad_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if squad_df.empty:
        return {"Batsman": squad_df, "Bowler": squad_df, "All-rounder": squad_df}
    mapped = squad_df.copy()
    mapped["role"] = mapped["role"].fillna("Batsman")
    return {
        "Batsman": mapped[mapped["role"].isin(["Batter", "All-rounder"])].reset_index(drop=True),
        "Bowler": mapped[mapped["role"].isin(["Bowler", "All-rounder"])].reset_index(drop=True),
        "All-rounder": mapped[mapped["role"] == "All-rounder"].reset_index(drop=True),
    }


def _summarize_matchup(df: pd.DataFrame) -> dict[str, Any]:
    batter_dismissals = df[(df["dismissal"] == 1) & (df["player_out_id"] == df["batter_id"])]
    balls = int(df["is_legal_ball"].sum())
    runs = int(df["batter_runs"].sum())
    dismissals = int(len(batter_dismissals))
    dot_balls = int(((df["is_legal_ball"] == 1) & (df["total_runs"] == 0)).sum())
    summary = {
        "runs": runs,
        "balls": balls,
        "strike_rate": round((runs / balls) * 100, 2) if balls else 0.0,
        "dot_balls": dot_balls,
        "dot_pct": round((dot_balls / balls) * 100, 2) if balls else 0.0,
        "ones": int((df["batter_runs"] == 1).sum()),
        "twos": int((df["batter_runs"] == 2).sum()),
        "threes": int((df["batter_runs"] == 3).sum()),
        "fours": int((df["batter_runs"] == 4).sum()),
        "sixes": int((df["batter_runs"] == 6).sum()),
        "dismissals": dismissals,
        "dismissal_types": dict(batter_dismissals["dismissal_kind"].fillna("unknown").value_counts()),
    }
    return summary


def _empty_matchup_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "batter_id",
            "batter",
            "batting_team",
            "bowler_id",
            "bowler",
            "bowling_team",
            "history_status",
            "sample_label",
            "runs",
            "balls",
            "strike_rate",
            "dot_balls",
            "dot_pct",
            "ones",
            "twos",
            "threes",
            "fours",
            "sixes",
            "dismissals",
            "dismissal_types",
            "dismissal_caught",
            "dismissal_bowled",
            "dismissal_lbw",
            "dismissal_run_out",
            "dismissal_stumped",
            "dismissal_caught_and_bowled",
            "dismissal_other",
            "match_count",
            "venue_history_status",
            "venue_sample_label",
            "venue_runs",
            "venue_balls",
            "venue_sr",
            "venue_dot_balls",
            "venue_dot_pct",
            "venue_ones",
            "venue_twos",
            "venue_threes",
            "venue_fours",
            "venue_sixes",
            "venue_dismissals",
            "venue_dismissal_types",
            "venue_dismissal_caught",
            "venue_dismissal_bowled",
            "venue_dismissal_lbw",
            "venue_dismissal_run_out",
            "venue_dismissal_stumped",
            "venue_dismissal_caught_and_bowled",
            "venue_dismissal_other",
            "sample_matches",
        ]
    )


def _apply_no_data_markers(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    status_column = f"{prefix}history_status" if prefix else "history_status"
    if prefix == "venue_":
        numeric_columns = [
            "venue_runs",
            "venue_balls",
            "venue_sr",
            "venue_dot_balls",
            "venue_dot_pct",
            "venue_ones",
            "venue_twos",
            "venue_threes",
            "venue_fours",
            "venue_sixes",
            "venue_dismissals",
            "venue_dismissal_caught",
            "venue_dismissal_bowled",
            "venue_dismissal_lbw",
            "venue_dismissal_run_out",
            "venue_dismissal_stumped",
            "venue_dismissal_caught_and_bowled",
            "venue_dismissal_other",
        ]
        count_columns: list[str] = []
        dismissal_types_column = "venue_dismissal_types"
        sample_label_column = "venue_sample_label"
    else:
        numeric_columns = [
            "runs",
            "balls",
            "strike_rate",
            "dot_balls",
            "dot_pct",
            "ones",
            "twos",
            "threes",
            "fours",
            "sixes",
            "dismissals",
            "dismissal_caught",
            "dismissal_bowled",
            "dismissal_lbw",
            "dismissal_run_out",
            "dismissal_stumped",
            "dismissal_caught_and_bowled",
            "dismissal_other",
        ]
        count_columns = ["match_count"]
        dismissal_types_column = "dismissal_types"
        sample_label_column = "sample_label"
    no_data_mask = df[status_column] == "no data"
    for column in [name for name in numeric_columns + count_columns if name in df.columns]:
        df.loc[no_data_mask, column] = pd.NA
    if dismissal_types_column in df.columns:
        df.loc[no_data_mask, dismissal_types_column] = "no data"
    if sample_label_column in df.columns:
        df.loc[no_data_mask, sample_label_column] = "no data"
    return df


def _dismissal_breakdown_columns(dismissal_types: dict[str, int], prefix: str = "") -> dict[str, int]:
    dismissal_types = dismissal_types or {}
    return {
        f"{prefix}dismissal_caught": int(dismissal_types.get("caught", 0)),
        f"{prefix}dismissal_bowled": int(dismissal_types.get("bowled", 0)),
        f"{prefix}dismissal_lbw": int(dismissal_types.get("lbw", 0)),
        f"{prefix}dismissal_run_out": int(dismissal_types.get("run out", 0)),
        f"{prefix}dismissal_stumped": int(dismissal_types.get("stumped", 0)),
        f"{prefix}dismissal_caught_and_bowled": int(dismissal_types.get("caught and bowled", 0)),
        f"{prefix}dismissal_other": int(
            sum(
                count
                for kind, count in dismissal_types.items()
                if kind not in {"caught", "bowled", "lbw", "run out", "stumped", "caught and bowled"}
            )
        ),
    }


def _sample_label(history_status: str, balls: Any) -> str:
    if history_status != "has data" or pd.isna(balls):
        return "no data"
    return "low sample" if float(balls) < 10 else "sufficient sample"


def get_matchup_stats_by_player_ids(batter_id: str, bowler_id: str, venue: str | None = None) -> pd.DataFrame:
    venue = normalize_venue_name(venue)
    deliveries = load_deliveries()
    df = deliveries[(deliveries["batter_id"] == batter_id) & (deliveries["bowler_id"] == bowler_id)].copy()
    venue_df = df[df["venue"] == venue].copy() if venue else pd.DataFrame()
    if df.empty:
        result = pd.DataFrame(
            [
                {
                    "batter_id": batter_id,
                    "bowler_id": bowler_id,
                    "history_status": "no data",
                    "sample_label": "no data",
                    "runs": pd.NA,
                    "balls": pd.NA,
                    "strike_rate": pd.NA,
                    "dot_balls": pd.NA,
                    "dot_pct": pd.NA,
                    "ones": pd.NA,
                    "twos": pd.NA,
                    "threes": pd.NA,
                    "fours": pd.NA,
                    "sixes": pd.NA,
                    "dismissals": pd.NA,
                    "dismissal_types": "no data",
                    "dismissal_caught": pd.NA,
                    "dismissal_bowled": pd.NA,
                    "dismissal_lbw": pd.NA,
                    "dismissal_run_out": pd.NA,
                    "dismissal_stumped": pd.NA,
                    "dismissal_caught_and_bowled": pd.NA,
                    "dismissal_other": pd.NA,
                    "match_count": pd.NA,
                    "venue_history_status": "no data",
                    "venue_sample_label": "no data",
                    "venue_runs": pd.NA,
                    "venue_balls": pd.NA,
                    "venue_sr": pd.NA,
                    "venue_dot_balls": pd.NA,
                    "venue_dot_pct": pd.NA,
                    "venue_ones": pd.NA,
                    "venue_twos": pd.NA,
                    "venue_threes": pd.NA,
                    "venue_fours": pd.NA,
                    "venue_sixes": pd.NA,
                    "venue_dismissals": pd.NA,
                    "venue_dismissal_types": "no data",
                    "venue_dismissal_caught": pd.NA,
                    "venue_dismissal_bowled": pd.NA,
                    "venue_dismissal_lbw": pd.NA,
                    "venue_dismissal_run_out": pd.NA,
                    "venue_dismissal_stumped": pd.NA,
                    "venue_dismissal_caught_and_bowled": pd.NA,
                    "venue_dismissal_other": pd.NA,
                    "sample_matches": pd.NA,
                }
            ]
        )
        return result

    overall = _summarize_matchup(df)
    venue_summary = _summarize_matchup(venue_df) if venue and not venue_df.empty else None
    batter_name = df["batter"].iloc[0]
    bowler_name = df["bowler"].iloc[0]
    result = pd.DataFrame(
        [
            {
                "batter_id": batter_id,
                "batter": batter_name,
                "bowler_id": bowler_id,
                "bowler": bowler_name,
                "history_status": "has data",
                "sample_label": _sample_label("has data", overall["balls"]),
                "runs": overall["runs"],
                "balls": overall["balls"],
                "strike_rate": overall["strike_rate"],
                "dot_balls": overall["dot_balls"],
                "dot_pct": overall["dot_pct"],
                "ones": overall["ones"],
                "twos": overall["twos"],
                "threes": overall["threes"],
                "fours": overall["fours"],
                "sixes": overall["sixes"],
                "dismissals": overall["dismissals"],
                "dismissal_types": overall["dismissal_types"],
                **_dismissal_breakdown_columns(overall["dismissal_types"]),
                "match_count": int(df["match_id"].nunique()),
                "venue_history_status": "has data" if venue_summary else "no data",
                "venue_sample_label": _sample_label("has data" if venue_summary else "no data", venue_summary["balls"] if venue_summary else pd.NA),
                "venue_runs": venue_summary["runs"] if venue_summary else pd.NA,
                "venue_balls": venue_summary["balls"] if venue_summary else pd.NA,
                "venue_sr": venue_summary["strike_rate"] if venue_summary else pd.NA,
                "venue_dot_balls": venue_summary["dot_balls"] if venue_summary else pd.NA,
                "venue_dot_pct": venue_summary["dot_pct"] if venue_summary else pd.NA,
                "venue_ones": venue_summary["ones"] if venue_summary else pd.NA,
                "venue_twos": venue_summary["twos"] if venue_summary else pd.NA,
                "venue_threes": venue_summary["threes"] if venue_summary else pd.NA,
                "venue_fours": venue_summary["fours"] if venue_summary else pd.NA,
                "venue_sixes": venue_summary["sixes"] if venue_summary else pd.NA,
                "venue_dismissals": venue_summary["dismissals"] if venue_summary else pd.NA,
                "venue_dismissal_types": venue_summary["dismissal_types"] if venue_summary else "no data",
                **(_dismissal_breakdown_columns(venue_summary["dismissal_types"], prefix="venue_") if venue_summary else _dismissal_breakdown_columns({}, prefix="venue_")),
                "sample_matches": int(df["match_id"].nunique()),
            }
        ]
    )
    return result


@lru_cache(maxsize=64)
def compute_matchup_matrix(team_a: str, team_b: str, venue: str | None = None) -> pd.DataFrame:
    team_a = normalize_team_name(team_a)
    team_b = normalize_team_name(team_b)
    venue = normalize_venue_name(venue)
    deliveries = load_deliveries()
    if deliveries.empty:
        return _empty_matchup_frame()
    squad_a = get_current_squad(team_a)
    squad_b = get_current_squad(team_b)
    if squad_a.empty or squad_b.empty:
        return _empty_matchup_frame()
    roles_a = split_squad_roles(squad_a)
    roles_b = split_squad_roles(squad_b)
    batters_a = set(roles_a["Batsman"]["player_id"].dropna().tolist())
    bowlers_a = set(roles_a["Bowler"]["player_id"].dropna().tolist())
    batters_b = set(roles_b["Batsman"]["player_id"].dropna().tolist())
    bowlers_b = set(roles_b["Bowler"]["player_id"].dropna().tolist())
    if not batters_a or not bowlers_a or not batters_b or not bowlers_b:
        return _empty_matchup_frame()

    matchups = deliveries[
        (
            deliveries["batter_id"].isin(batters_a)
            & deliveries["bowler_id"].isin(bowlers_b)
            & (deliveries["batting_team"] == team_a)
            & (deliveries["bowling_team"] == team_b)
        )
        | (
            deliveries["batter_id"].isin(batters_b)
            & deliveries["bowler_id"].isin(bowlers_a)
            & (deliveries["batting_team"] == team_b)
            & (deliveries["bowling_team"] == team_a)
        )
    ].copy()
    if venue:
        matchups["is_selected_venue"] = matchups["venue"].eq(venue)
    else:
        matchups["is_selected_venue"] = False

    if matchups.empty:
        matchups = deliveries.iloc[0:0].copy()

    batter_pool = pd.concat(
        [
            roles_a["Batsman"][["player_id", "name"]].assign(batting_team=team_a),
            roles_b["Batsman"][["player_id", "name"]].assign(batting_team=team_b),
        ],
        ignore_index=True,
    ).rename(columns={"player_id": "batter_id", "name": "batter"})
    batter_pool = batter_pool[batter_pool["batter_id"].notna()].drop_duplicates(["batter_id", "batting_team"])
    bowler_pool = pd.concat(
        [
            roles_b["Bowler"][["player_id", "name"]].assign(bowling_team=team_b),
            roles_a["Bowler"][["player_id", "name"]].assign(bowling_team=team_a),
        ],
        ignore_index=True,
    ).rename(columns={"player_id": "bowler_id", "name": "bowler"})
    bowler_pool = bowler_pool[bowler_pool["bowler_id"].notna()].drop_duplicates(["bowler_id", "bowling_team"])

    matchup_pairs = pd.concat(
        [
            batter_pool[batter_pool["batting_team"] == team_a].merge(
                bowler_pool[bowler_pool["bowling_team"] == team_b], how="cross"
            ),
            batter_pool[batter_pool["batting_team"] == team_b].merge(
                bowler_pool[bowler_pool["bowling_team"] == team_a], how="cross"
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["batter_id", "bowler_id", "batting_team", "bowling_team"])

    grouped_rows: list[dict[str, Any]] = []
    for (batter_id, bowler_id), pair_df in matchups.groupby(["batter_id", "bowler_id"]):
        overall = _summarize_matchup(pair_df)
        venue_df = pair_df[pair_df["is_selected_venue"]].copy() if venue else pd.DataFrame()
        venue_stats = _summarize_matchup(venue_df) if venue and not venue_df.empty else None
        grouped_rows.append(
            {
                "batter_id": batter_id,
                "history_status": "has data",
                "sample_label": _sample_label("has data", overall["balls"]),
                "batting_team": pair_df["batting_team"].iloc[0],
                "bowler_id": bowler_id,
                "bowling_team": pair_df["bowling_team"].iloc[0],
                "runs": overall["runs"],
                "balls": overall["balls"],
                "strike_rate": overall["strike_rate"],
                "dot_balls": overall["dot_balls"],
                "dot_pct": overall["dot_pct"],
                "ones": overall["ones"],
                "twos": overall["twos"],
                "threes": overall["threes"],
                "fours": overall["fours"],
                "sixes": overall["sixes"],
                "dismissals": overall["dismissals"],
                "dismissal_types": overall["dismissal_types"],
                **_dismissal_breakdown_columns(overall["dismissal_types"]),
                "match_count": int(pair_df["match_id"].nunique()),
                "venue_history_status": "has data" if venue_stats else "no data",
                "venue_sample_label": _sample_label("has data" if venue_stats else "no data", venue_stats["balls"] if venue_stats else pd.NA),
                "venue_runs": venue_stats["runs"] if venue_stats else pd.NA,
                "venue_balls": venue_stats["balls"] if venue_stats else pd.NA,
                "venue_sr": venue_stats["strike_rate"] if venue_stats else pd.NA,
                "venue_dot_balls": venue_stats["dot_balls"] if venue_stats else pd.NA,
                "venue_dot_pct": venue_stats["dot_pct"] if venue_stats else pd.NA,
                "venue_ones": venue_stats["ones"] if venue_stats else pd.NA,
                "venue_twos": venue_stats["twos"] if venue_stats else pd.NA,
                "venue_threes": venue_stats["threes"] if venue_stats else pd.NA,
                "venue_fours": venue_stats["fours"] if venue_stats else pd.NA,
                "venue_sixes": venue_stats["sixes"] if venue_stats else pd.NA,
                "venue_dismissals": venue_stats["dismissals"] if venue_stats else pd.NA,
                "venue_dismissal_types": venue_stats["dismissal_types"] if venue_stats else "no data",
                **(_dismissal_breakdown_columns(venue_stats["dismissal_types"], prefix="venue_") if venue_stats else _dismissal_breakdown_columns({}, prefix="venue_")),
                "sample_matches": int(pair_df["match_id"].nunique()),
            }
        )
    summarized = pd.DataFrame(grouped_rows)
    if summarized.empty:
        summarized = pd.DataFrame(
            columns=[
                "batter_id",
                "batting_team",
                "bowler_id",
                "bowling_team",
                "history_status",
                "sample_label",
                "runs",
                "balls",
                "strike_rate",
                "dot_balls",
                "dot_pct",
                "ones",
                "twos",
                "threes",
                "fours",
                "sixes",
                "dismissals",
                "dismissal_types",
                "dismissal_caught",
                "dismissal_bowled",
                "dismissal_lbw",
                "dismissal_run_out",
                "dismissal_stumped",
                "dismissal_caught_and_bowled",
                "dismissal_other",
                "match_count",
                "venue_history_status",
                "venue_sample_label",
                "venue_runs",
                "venue_balls",
                "venue_sr",
                "venue_dot_balls",
                "venue_dot_pct",
                "venue_ones",
                "venue_twos",
                "venue_threes",
                "venue_fours",
                "venue_sixes",
                "venue_dismissals",
                "venue_dismissal_types",
                "venue_dismissal_caught",
                "venue_dismissal_bowled",
                "venue_dismissal_lbw",
                "venue_dismissal_run_out",
                "venue_dismissal_stumped",
                "venue_dismissal_caught_and_bowled",
                "venue_dismissal_other",
                "sample_matches",
            ]
        )
    full = matchup_pairs.merge(
        summarized,
        on=["batter_id", "batting_team", "bowler_id", "bowling_team"],
        how="left",
    )
    fill_zero_columns = [
        "runs",
        "balls",
        "strike_rate",
        "dot_balls",
        "dot_pct",
        "ones",
        "twos",
        "threes",
        "fours",
        "sixes",
        "dismissals",
        "match_count",
        "sample_matches",
    ]
    for column in fill_zero_columns:
        if column in full.columns:
            full[column] = full[column].fillna(0)
    full["history_status"] = full["history_status"].fillna("no data")
    if "venue_history_status" in full.columns:
        full["venue_history_status"] = full["venue_history_status"].fillna("no data")
    if "sample_label" in full.columns:
        full["sample_label"] = full["sample_label"].fillna("no data")
    if "venue_sample_label" in full.columns:
        full["venue_sample_label"] = full["venue_sample_label"].fillna("no data")
    if "dismissal_types" not in full.columns:
        full["dismissal_types"] = [{} for _ in range(len(full))]
    else:
        full["dismissal_types"] = full["dismissal_types"].apply(lambda value: value if isinstance(value, dict) else {})
    if "venue_dismissal_types" not in full.columns:
        full["venue_dismissal_types"] = ["no data" for _ in range(len(full))]
    else:
        full["venue_dismissal_types"] = full["venue_dismissal_types"].apply(
            lambda value: value if isinstance(value, dict) else "no data"
        )
    full = _apply_no_data_markers(full)
    if "venue_history_status" in full.columns:
        full = _apply_no_data_markers(full, prefix="venue_")
    return full.sort_values(["dismissals", "balls", "batter", "bowler"], ascending=[False, False, True, True]).reset_index(drop=True)


def build_upcoming_match_analysis(schedule_match_id: str) -> dict[str, Any]:
    selected_match = get_upcoming_match_context(schedule_match_id)
    if not selected_match:
        return {"match": None, "matchups": _empty_matchup_frame()}
    matchup_df = compute_matchup_matrix(
        selected_match["team_a"],
        selected_match["team_b"],
        selected_match["venue"],
    )
    return {"match": selected_match, "matchups": matchup_df}


def matchup_phase_breakdown(batter_id: str, bowler_id: str, venue: str | None = None) -> pd.DataFrame:
    deliveries = load_deliveries()
    df = deliveries[(deliveries["batter_id"] == batter_id) & (deliveries["bowler_id"] == bowler_id)].copy()
    if venue:
        df = df[df["venue"] == venue]
    if df.empty:
        return pd.DataFrame(columns=["phase", "runs", "balls", "strike_rate", "dismissals"])
    rows = []
    for phase, phase_df in df.groupby("phase"):
        summary = _summarize_matchup(phase_df)
        rows.append(
            {
                "phase": phase,
                "runs": summary["runs"],
                "balls": summary["balls"],
                "strike_rate": summary["strike_rate"],
                "dot_pct": summary["dot_pct"],
                "dismissals": summary["dismissals"],
            }
        )
    phase_order = pd.CategoricalDtype(categories=["Powerplay", "Middle", "Death"], ordered=True)
    result = pd.DataFrame(rows)
    result["phase"] = result["phase"].astype(phase_order)
    return result.sort_values("phase")


def matchup_history(batter_id: str, bowler_id: str, venue: str | None = None) -> pd.DataFrame:
    deliveries = load_deliveries()
    df = deliveries[(deliveries["batter_id"] == batter_id) & (deliveries["bowler_id"] == bowler_id)].copy()
    if venue:
        df = df[df["venue"] == venue]
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (match_id, match_date, venue_name), match_df in df.groupby(["match_id", "match_date", "venue"]):
        summary = _summarize_matchup(match_df)
        rows.append(
            {
                "match_id": match_id,
                "match_date": match_date,
                "venue": venue_name,
                "runs": summary["runs"],
                "balls": summary["balls"],
                "strike_rate": summary["strike_rate"],
                "dismissals": summary["dismissals"],
                "timeline_events": len(match_df),
            }
        )
    return pd.DataFrame(rows).sort_values("match_date", ascending=False)


def matchup_timeline(batter_id: str, bowler_id: str) -> pd.DataFrame:
    deliveries = load_deliveries()
    df = deliveries[(deliveries["batter_id"] == batter_id) & (deliveries["bowler_id"] == bowler_id)].copy()
    if df.empty:
        return pd.DataFrame()
    df["result"] = np.where(df["dismissal"] == 1, df["dismissal_kind"].fillna("wicket"), df["total_runs"].astype(str))
    return df[
        [
            "match_date",
            "match_id",
            "venue",
            "innings",
            "over",
            "ball",
            "ball_label",
            "result",
            "batter_runs",
            "extras",
            "dismissal_kind",
        ]
    ].sort_values(["match_date", "innings", "over", "ball"], ascending=[False, True, True, True])


def player_type_weaknesses(player_id: str) -> pd.DataFrame:
    deliveries = load_deliveries()
    df = deliveries[deliveries["batter_id"] == player_id].copy()
    if df.empty or "bowling_type" not in df.columns:
        return pd.DataFrame(columns=["bucket", "runs", "balls", "strike_rate", "dismissals"])
    buckets = {
        "Spin": df["bowling_type"].fillna("Unknown").eq("Spin"),
        "Pace": df["bowling_type"].fillna("Unknown").eq("Pace"),
        "Left-arm": df["bowling_arm"].fillna("Unknown").eq("Left"),
        "Right-arm": df["bowling_arm"].fillna("Unknown").eq("Right"),
    }
    rows = []
    for bucket, mask in buckets.items():
        bucket_df = df[mask]
        if bucket_df.empty:
            continue
        summary = _summarize_matchup(bucket_df)
        rows.append(
            {
                "bucket": bucket,
                "runs": summary["runs"],
                "balls": summary["balls"],
                "strike_rate": summary["strike_rate"],
                "dismissals": summary["dismissals"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dismissals", "strike_rate"], ascending=[False, True])


def key_battles(matchup_df: pd.DataFrame) -> pd.DataFrame:
    if matchup_df.empty:
        return matchup_df
    rated = matchup_df[matchup_df["sample_matches"] > 0].copy()
    if rated.empty:
        return rated
    rated["battle_score"] = (
        rated["dismissals"] * 18
        + rated["balls"].clip(upper=60) * 0.7
        - rated["strike_rate"] * 0.08
        + rated["venue_dismissals"] * 10
    )
    return rated.sort_values("battle_score", ascending=False).head(5)


def strong_and_weak_matchups(matchup_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matchup_df.empty:
        return matchup_df, matchup_df
    eligible = matchup_df[(matchup_df["sample_matches"] > 0) & (matchup_df["balls"] >= 6)].copy()
    weak = eligible.sort_values(["dismissals", "strike_rate"], ascending=[False, True]).head(5)
    strong = eligible.sort_values(["strike_rate", "balls"], ascending=[False, False]).head(5)
    return strong, weak


def strategy_suggestions(matchup_df: pd.DataFrame) -> list[str]:
    suggestions: list[str] = []
    if matchup_df.empty:
        return ["No historical matchup data is available for the current squad pool."]
    for batter, batter_df in matchup_df.groupby("batter"):
        eligible = batter_df[
            (batter_df["sample_matches"] > 0) & (batter_df["balls"] >= 6)
        ].sort_values(["dismissals", "strike_rate"], ascending=[False, True])
        if eligible.empty:
            continue
        best = eligible.iloc[0]
        suggestions.append(
            f"Use {best['bowler']} early against {batter}: {int(best['runs'])} runs off {int(best['balls'])} balls, SR {best['strike_rate']}, dismissals {int(best['dismissals'])}."
        )
    return suggestions[:8] or ["No historical matchup data is available for the current squad pool."]


def clear_analytics_caches() -> None:
    load_matches.cache_clear()
    load_deliveries.cache_clear()
    load_appearances.cache_clear()
    get_upcoming_matches.cache_clear()
    load_player_reference.cache_clear()
    load_register_lookup.cache_clear()
    load_team_player_candidates.cache_clear()
    get_current_squad.cache_clear()
    compute_matchup_matrix.cache_clear()
