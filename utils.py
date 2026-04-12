"""
IPL Smart Analyst — shared data loading, cleaning, and feature engineering.

Keeps logic readable for interviews: explicit steps, minimal magic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MATCHES_PATH = DATA_DIR / "matches.csv"
DELIVERIES_PATH = DATA_DIR / "deliveries.csv"

# Historical franchise renames — align names across seasons
TEAM_REPLACEMENTS = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


def normalize_team_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply canonical team names so matches and deliveries join cleanly."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace(TEAM_REPLACEMENTS)
    return out


def load_matches(path: Path | None = None) -> pd.DataFrame:
    """Load matches CSV; parse dates; fill a few safe defaults for missing text."""
    p = path or MATCHES_PATH
    df = pd.read_csv(p, low_memory=False)
    # Missing venue / umpires: keep rows but fill venue for encoding
    if "venue" in df.columns:
        df["venue"] = df["venue"].fillna("Unknown")
    for col in ["team1", "team2", "toss_winner", "winner", "toss_decision"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    df = normalize_team_columns(df, ["team1", "team2", "toss_winner", "winner"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_deliveries(path: Path | None = None) -> pd.DataFrame:
    """Load ball-by-ball data; coerce numeric wicket flag."""
    p = path or DELIVERIES_PATH
    df = pd.read_csv(p, low_memory=False)
    df = normalize_team_columns(df, ["batting_team", "bowling_team"])
    if "is_wicket" in df.columns:
        df["is_wicket"] = pd.to_numeric(df["is_wicket"], errors="coerce").fillna(0).astype(int)
    return df


def compute_team_win_percentage(matches: pd.DataFrame) -> dict[str, float]:
    """
    Overall win rate per team (matches played as team1 or team2).
    Used as a simple strength signal for the model and UI.
    """
    m = matches.dropna(subset=["team1", "team2", "winner"]).copy()
    m = m[m["team1"] != m["team2"]]
    m = m[(m["winner"] == m["team1"]) | (m["winner"] == m["team2"])]
    teams = pd.concat([m["team1"], m["team2"]]).unique()
    stats: dict[str, float] = {}
    for t in teams:
        played = m[(m["team1"] == t) | (m["team2"] == t)].shape[0]
        wins = m[m["winner"] == t].shape[0]
        stats[str(t)] = float(wins / played) if played > 0 else 0.0
    return stats


def compute_batsman_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Per-batter: runs, balls faced, dismissals -> batting_avg, strike_rate.
    """
    d = deliveries.copy()
    # Balls faced: rows where this player is the striker (standard IPL ball count)
    grp = d.groupby("batter", as_index=False).agg(
        runs=("batsman_runs", "sum"),
        balls=("batter", "count"),
    )
    # Dismissals credited to batter (out or run out on their account)
    dismissals = (
        d[d["is_wicket"] == 1]
        .dropna(subset=["player_dismissed"])
        .groupby("player_dismissed")
        .size()
        .reset_index(name="dismissals")
        .rename(columns={"player_dismissed": "batter"})
    )
    grp = grp.merge(dismissals, on="batter", how="left")
    grp["dismissals"] = grp["dismissals"].fillna(0).astype(int)
    # Avoid divide-by-zero; average undefined if never out — use runs only for SR
    grp["batting_avg"] = np.where(
        grp["dismissals"] > 0,
        grp["runs"] / grp["dismissals"],
        np.nan,
    )
    grp["strike_rate"] = np.where(grp["balls"] > 0, (grp["runs"] / grp["balls"]) * 100.0, np.nan)
    return grp


def compute_bowler_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Per-bowler: runs conceded, legal balls, wickets -> economy_rate.
    """
    d = deliveries.copy()
    grp = d.groupby("bowler", as_index=False).agg(
        runs_conceded=("total_runs", "sum"),
        balls=("bowler", "count"),
        wickets=("is_wicket", "sum"),
    )
    overs = grp["balls"] / 6.0
    grp["economy_rate"] = np.where(overs > 0, grp["runs_conceded"] / overs, np.nan)
    return grp


def team_batting_strike_rate(deliveries: pd.DataFrame) -> dict[str, float]:
    """Team-level batting strike rate (all their runs / all balls while batting)."""
    d = deliveries.copy()
    agg = d.groupby("batting_team", as_index=False).agg(
        runs=("total_runs", "sum"),
        balls=("match_id", "count"),
    )
    agg["sr"] = np.where(agg["balls"] > 0, (agg["runs"] / agg["balls"]) * 100.0, 0.0)
    return dict(zip(agg["batting_team"].astype(str), agg["sr"].astype(float)))


def team_bowling_economy(deliveries: pd.DataFrame) -> dict[str, float]:
    """Team-level economy while bowling."""
    d = deliveries.copy()
    agg = d.groupby("bowling_team", as_index=False).agg(
        runs=("total_runs", "sum"),
        balls=("match_id", "count"),
    )
    overs = agg["balls"] / 6.0
    agg["econ"] = np.where(overs > 0, agg["runs"] / overs, 0.0)
    return dict(zip(agg["bowling_team"].astype(str), agg["econ"].astype(float)))


def top_batsmen_by_strike_rate(
    bat_df: pd.DataFrame,
    min_balls: int = 500,
    n: int = 10,
) -> pd.DataFrame:
    """Descriptive: qualified batsmen sorted by strike rate."""
    q = bat_df[bat_df["balls"] >= min_balls].dropna(subset=["strike_rate"])
    return q.nlargest(n, "strike_rate")[
        ["batter", "runs", "balls", "dismissals", "batting_avg", "strike_rate"]
    ].reset_index(drop=True)


def top_bowlers_by_economy(
    bowl_df: pd.DataFrame,
    min_balls: int = 500,
    n: int = 10,
) -> pd.DataFrame:
    """Lower economy is better — ascending sort."""
    q = bowl_df[bowl_df["balls"] >= min_balls].dropna(subset=["economy_rate"])
    return q.nsmallest(n, "economy_rate")[
        ["bowler", "runs_conceded", "balls", "wickets", "economy_rate"]
    ].reset_index(drop=True)


def prepare_model_table(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    One row per match with numeric features + target team1_win (1 if team1 won).
    """
    m = matches[
        ["team1", "team2", "toss_winner", "toss_decision", "winner", "venue"]
    ].dropna(subset=["team1", "team2", "winner"])
    m = m[m["team1"] != m["team2"]]
    # Winner must be one of the two sides in that fixture
    m = m[(m["winner"] == m["team1"]) | (m["winner"] == m["team2"])]

    win_pct = compute_team_win_percentage(m)
    bat_sr = team_batting_strike_rate(deliveries)
    bowl_econ = team_bowling_economy(deliveries)

    m = m.copy()
    m["team1_win"] = (m["winner"] == m["team1"]).astype(int)
    m["team_win_pct_team1"] = m["team1"].map(lambda t: win_pct.get(str(t), 0.0))
    m["team_win_pct_team2"] = m["team2"].map(lambda t: win_pct.get(str(t), 0.0))
    m["team_bat_sr_1"] = m["team1"].map(lambda t: bat_sr.get(str(t), 0.0))
    m["team_bat_sr_2"] = m["team2"].map(lambda t: bat_sr.get(str(t), 0.0))
    m["team_bowl_econ_1"] = m["team1"].map(lambda t: bowl_econ.get(str(t), 0.0))
    m["team_bowl_econ_2"] = m["team2"].map(lambda t: bowl_econ.get(str(t), 0.0))

    m["toss_decision"] = (
        m["toss_decision"].astype(str).str.lower().replace({"batting": "bat", "fielding": "field"})
    )
    m["toss_advantage"] = (m["toss_winner"] == m["team1"]).astype(int)
    return m.reset_index(drop=True)
