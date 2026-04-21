from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cricket_analytics.analytics import (
    clear_analytics_caches,
    compute_matchup_matrix,
    get_current_squad,
    get_all_teams,
    key_battles,
    load_matches,
    matchup_history,
    matchup_phase_breakdown,
    matchup_timeline,
    player_type_weaknesses,
    recent_match_pool,
    strategy_suggestions,
    strong_and_weak_matchups,
)
from cricket_analytics.config import DB_PATH
from cricket_analytics.db import initialize_database
from cricket_analytics.ingest import build_database


st.set_page_config(
    page_title="IPL Strategy Engine",
    page_icon=":cricket_bat_and_ball:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def style_app() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f1e8;
            --card: #fffaf2;
            --ink: #102a43;
            --muted: #627d98;
            --accent: #d62828;
            --accent-2: #f77f00;
            --good: #2a9d8f;
            --bad: #d62828;
            --border: rgba(16, 42, 67, 0.10);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(247,127,0,0.14), transparent 24%),
                radial-gradient(circle at top right, rgba(42,157,143,0.10), transparent 22%),
                linear-gradient(180deg, #f8f4ec 0%, #efe6d8 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero, .card {
            background: rgba(255, 250, 242, 0.92);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(16, 42, 67, 0.08);
        }
        .hero h1, .section-title {
            color: var(--ink);
            margin-bottom: 0.2rem;
        }
        .meta {
            color: var(--muted);
            font-size: 0.92rem;
        }
        .pill-good, .pill-bad, .pill-neutral {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 0.35rem;
        }
        .pill-good { background: rgba(42,157,143,0.16); color: var(--good); }
        .pill-bad { background: rgba(214,40,40,0.12); color: var(--bad); }
        .pill-neutral { background: rgba(16,42,67,0.08); color: var(--ink); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def prepare_database() -> str:
    initialize_database()
    return str(DB_PATH)


@st.cache_data(show_spinner=False)
def load_venues() -> list[str]:
    matches = load_matches()
    if matches.empty:
        return []
    return sorted(matches["venue"].dropna().unique().tolist())


def render_matchup_heatmap(matchup_df: pd.DataFrame) -> None:
    if matchup_df.empty:
        st.info("No data available")
        return
    heatmap_df = matchup_df.copy()
    heatmap_df["display_sr"] = heatmap_df["strike_rate"].where(heatmap_df["sample_matches"] > 0, other=float("nan"))
    matrix = heatmap_df.pivot(index="batter", columns="bowler", values="display_sr")
    if matrix.empty:
        st.info("No data available")
        return
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="RdYlGn",
            colorbar={"title": "Strike Rate"},
            hoverongaps=False,
            zmin=0,
            zmax=max(200, float(pd.Series(heatmap_df["strike_rate"]).max() or 0)),
        )
    )
    fig.update_layout(height=max(420, len(matrix.index) * 28), margin={"l": 10, "r": 10, "t": 40, "b": 10}, title="Matchup Heatmap")
    st.plotly_chart(fig, use_container_width=True)


def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>IPL Pre-Match Strategy Engine</h1>
            <div class="meta">All player-vs-player matchups, venue intelligence, phase splits, and analyst-style strategy suggestions from Cricsheet ball-by-ball history.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.subheader("Match Planner")
    cards = recent_match_pool()
    if cards.empty:
        st.info("No IPL data is loaded yet. Use the sidebar to ingest a Cricsheet zip first.")
        return
    cols = st.columns(4)
    for idx, row in cards.iterrows():
        with cols[idx % 4]:
            st.markdown(
                f"""
                <div class="card">
                    <div class="pill-neutral">{row['season']}</div>
                    <h4>{row['team1']} vs {row['team2']}</h4>
                    <div class="meta">{row['venue'] or 'Venue TBD'}</div>
                    <div class="meta">{row['match_date'] or 'Date unavailable'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_match_dashboard(team_a: str, team_b: str, venue: str | None) -> None:
    matchup_df = compute_matchup_matrix(team_a, team_b, venue)
    st.markdown(
        f"""
        <div class="hero">
            <h1>{team_a} vs {team_b}</h1>
            <div class="meta">{venue or 'All venues'} | Current IPL squad source of truth with historical IPL matchup stats</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if matchup_df.empty:
        st.warning("No data available")
        return
    if not (matchup_df["sample_matches"] > 0).any():
        st.warning("No data available")

    key_df = key_battles(matchup_df)
    strong_df, weak_df = strong_and_weak_matchups(matchup_df)
    suggestions = strategy_suggestions(matchup_df)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Top 5 Key Battles")
        st.dataframe(
            key_df[["batter", "bowler", "runs", "balls", "strike_rate", "dismissals", "venue_sr"]],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Strategy Suggestions")
        for line in suggestions:
            st.markdown(f"- {line}")

    st.subheader("Full Matchup Matrix")
    matrix = matchup_df.assign(
        display_sr=matchup_df["strike_rate"].where(matchup_df["sample_matches"] > 0)
    ).pivot_table(
        index="batter",
        columns="bowler",
        values="display_sr",
        aggfunc="mean",
    )
    st.dataframe(
        matrix.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.1f}", na_rep="-"),
        use_container_width=True,
    )

    render_matchup_heatmap(matchup_df)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Strong Matchups")
        st.dataframe(
            strong_df[["batter", "bowler", "runs", "balls", "strike_rate", "dismissals"]],
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.subheader("Weak Matchups")
        st.dataframe(
            weak_df[["batter", "bowler", "runs", "balls", "strike_rate", "dismissals"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Detailed Matchup Table")
    st.dataframe(matchup_df, use_container_width=True, hide_index=True)

    top_runs = matchup_df.nlargest(12, "runs").copy()
    top_runs["label"] = top_runs["batter"] + " vs " + top_runs["bowler"]
    bar = px.bar(top_runs, x="label", y="runs", color="dismissals", title="Highest Volume Player-vs-Player Histories")
    bar.update_layout(xaxis_title="", yaxis_title="Runs")
    st.plotly_chart(bar, use_container_width=True)


def render_player_detail(matchup_df: pd.DataFrame, venue: str | None) -> None:
    if matchup_df.empty:
        st.info("No data available")
        return
    labels = matchup_df.apply(lambda row: f"{row['batter']} vs {row['bowler']}", axis=1).tolist()
    selected_label = st.selectbox("Select matchup", labels, key="matchup_picker")
    selected = matchup_df.iloc[labels.index(selected_label)]
    batter_id = selected["batter_id"]
    bowler_id = selected["bowler_id"]
    if int(selected["sample_matches"]) == 0:
        st.info("No data available for this matchup yet.")
        return

    st.subheader("Player Detail View")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Runs", int(selected["runs"]))
    m2.metric("Balls", int(selected["balls"]))
    m3.metric("Strike Rate", f"{selected['strike_rate']:.1f}")
    m4.metric("4s / 6s", f"{int(selected['fours'])} / {int(selected['sixes'])}")
    m5.metric("Dot %", f"{selected['dot_pct']:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        phase_df = matchup_phase_breakdown(batter_id, bowler_id, venue)
        st.markdown("**Phase Breakdown**")
        st.dataframe(phase_df, use_container_width=True, hide_index=True)
        if not phase_df.empty:
            phase_chart = px.bar(phase_df, x="phase", y="strike_rate", color="dismissals", title="Strike Rate by Phase")
            st.plotly_chart(phase_chart, use_container_width=True)
    with c2:
        weakness_df = player_type_weaknesses(batter_id)
        st.markdown("**Player Weaknesses**")
        if weakness_df.empty:
            st.caption("Bowling-style metadata is not enriched yet, so spin/pace and arm-based weakness splits are unavailable.")
        else:
            st.dataframe(weakness_df, use_container_width=True, hide_index=True)
            weakness_chart = px.pie(weakness_df, names="bucket", values="dismissals", title="Dismissals by Bowling Bucket")
            st.plotly_chart(weakness_chart, use_container_width=True)

    comparison_df = pd.DataFrame(
        [
            {"scope": "Overall", "runs": selected["runs"], "balls": selected["balls"], "strike_rate": selected["strike_rate"], "dismissals": selected["dismissals"]},
            {"scope": "Selected Venue", "runs": selected["venue_runs"], "balls": selected["venue_balls"], "strike_rate": selected["venue_sr"], "dismissals": selected["venue_dismissals"]},
        ]
    )
    st.markdown("**Overall vs Venue**")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    venue_chart = px.bar(comparison_df, x="scope", y="strike_rate", color="dismissals", title="Overall vs Venue Strike Rate")
    st.plotly_chart(venue_chart, use_container_width=True)

    history_df = matchup_history(batter_id, bowler_id, venue)
    st.markdown("**Match History**")
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    timeline_df = matchup_timeline(batter_id, bowler_id)
    st.markdown("**Ball-by-Ball Timeline**")
    st.dataframe(timeline_df, use_container_width=True, hide_index=True)


def sidebar_controls() -> tuple[str | None, str | None, str | None, bool]:
    st.sidebar.title("Controls")
    db_path = prepare_database()
    st.sidebar.caption(f"DB: {db_path}")
    st.sidebar.caption("Dataset: data/raw/ipl_json.zip")
    download_data = st.sidebar.checkbox("Refresh IPL zip from Cricsheet before rebuild", value=False)
    if st.sidebar.button("Rebuild IPL Database", use_container_width=True):
        try:
            result = build_database(
                zip_path=Path("data/raw/ipl_json.zip"),
                download=False,
                force_download=download_data,
            )
            clear_analytics_caches()
            load_venues.clear()
            st.sidebar.success(
                f"Rebuilt from {result.zip_path}: {result.matches} matches and {result.deliveries} deliveries processed."
            )
        except Exception as exc:
            st.sidebar.error(str(exc))

    teams = get_all_teams()
    venues = load_venues()
    if not teams:
        st.sidebar.info("Rebuild the IPL database to unlock team and venue selectors.")
        return None, None, None, False

    team_a = st.sidebar.selectbox("Team A", teams, index=0, help="Type to search teams.")
    team_b_choices = [team for team in teams if team != team_a]
    if not team_b_choices:
        st.sidebar.warning("No opponent teams available.")
        return None, None, None, False
    team_b = st.sidebar.selectbox("Team B", team_b_choices, index=0, help="Type to search teams.")
    venue = st.sidebar.selectbox("Venue", ["All venues"] + venues, help="Type to search venues.")
    show_player_view = st.sidebar.toggle("Show player detail view", value=True)
    return team_a, team_b, None if venue == "All venues" else venue, show_player_view


def main() -> None:
    style_app()
    team_a, team_b, venue, show_player_view = sidebar_controls()
    tab_home, tab_dashboard = st.tabs(["Home", "Match Dashboard"])
    with tab_home:
        render_home()
        if team_a:
            st.subheader("Current Squad Pools")
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(get_current_squad(team_a), use_container_width=True, hide_index=True)
            with c2:
                st.dataframe(get_current_squad(team_b), use_container_width=True, hide_index=True)
    with tab_dashboard:
        if not team_a or not team_b:
            st.info("Load data from the sidebar to start analyzing a match.")
            return
        matchup_df = compute_matchup_matrix(team_a, team_b, venue)
        render_match_dashboard(team_a, team_b, venue)
        if show_player_view:
            render_player_detail(matchup_df, venue)


if __name__ == "__main__":
    main()
