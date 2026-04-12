"""
🏏 IPL Smart Analyst — Match Insights & Prediction System
Streamlit dashboard: prediction, team analytics, player leaderboards.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = PROJECT_ROOT / "model" / "match_winner_bundle.joblib"


# -----------------------------------------------------------------------------
# Cached loads (data + model bundle)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_matches_df() -> pd.DataFrame:
    return utils.load_matches()


@st.cache_data(show_spinner=False)
def load_deliveries_df() -> pd.DataFrame:
    return utils.load_deliveries()


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


@st.cache_data(show_spinner=False)
def batsman_table(deliveries: pd.DataFrame) -> pd.DataFrame:
    return utils.compute_batsman_features(deliveries)


@st.cache_data(show_spinner=False)
def bowler_table(deliveries: pd.DataFrame) -> pd.DataFrame:
    return utils.compute_bowler_features(deliveries)


def plotly_theme() -> dict:
    """Dark, minimal Plotly defaults."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,15,30,0.6)",
        font=dict(family="Inter, system-ui, sans-serif", color="#e8e6df"),
        title_font_size=16,
    )


def inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
.hero-title {
    font-size: 2.35rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem 0;
    color: #f8f6f2;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #7a7f92;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.75rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #f0b429;
    margin: 1.5rem 0 0.75rem 0;
    letter-spacing: -0.02em;
}
.card {
    background: linear-gradient(145deg, #0e1224 0%, #0a0d18 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 0.85rem;
}
.card-muted { color: #8b90a5; font-size: 0.88rem; line-height: 1.45; }
.win-banner {
    background: linear-gradient(120deg, rgba(34,197,94,0.15), rgba(240,180,41,0.08));
    border: 1px solid rgba(74,222,128,0.35);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0 1rem 0;
}
.win-banner strong { color: #4ade80; font-size: 1.2rem; }
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #5c6178;
}
.stat-value { font-size: 1.45rem; font-weight: 700; color: #f8f6f2; }
hr.sep { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.25rem 0; }
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #22c55e, #f0b429) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def build_prediction_row(
    bundle: dict,
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
    venue: str,
) -> pd.DataFrame:
    """One row as a DataFrame with the same column names used in training (avoids sklearn warnings)."""
    te = bundle["team_encoder"]
    toss_e = bundle["toss_encoder"]
    ven_e = bundle["venue_encoder"]
    wp = bundle["team_win_pct"]
    sr = bundle["team_bat_sr"]
    be = bundle["team_bowl_econ"]

    venue_clean = venue if venue in set(ven_e.classes_) else (
        "Unknown" if "Unknown" in set(ven_e.classes_) else ven_e.classes_[0]
    )

    row = {
        "team1_enc": te.transform([team1])[0],
        "team2_enc": te.transform([team2])[0],
        "toss_winner_enc": te.transform([toss_winner])[0],
        "toss_decision_enc": toss_e.transform([str(toss_decision).strip().lower()])[0],
        "venue_enc": ven_e.transform([venue_clean])[0],
        "team_win_pct_team1": float(wp.get(str(team1), 0.0)),
        "team_win_pct_team2": float(wp.get(str(team2), 0.0)),
        "team_bat_sr_1": float(sr.get(str(team1), 0.0)),
        "team_bat_sr_2": float(sr.get(str(team2), 0.0)),
        "team_bowl_econ_1": float(be.get(str(team1), 0.0)),
        "team_bowl_econ_2": float(be.get(str(team2), 0.0)),
        "toss_advantage": int(toss_winner == team1),
    }
    cols = bundle["feature_columns"]
    return pd.DataFrame([[row[c] for c in cols]], columns=cols)


def team_wins_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Total wins per team (for bar charts)."""
    m = matches.dropna(subset=["winner", "team1", "team2"]).copy()
    m = m[m["team1"] != m["team2"]]
    m = m[(m["winner"] == m["team1"]) | (m["winner"] == m["team2"])]
    wc = m["winner"].value_counts().reset_index()
    wc.columns = ["team", "wins"]
    return wc.sort_values("wins", ascending=False)


def wins_lookup(table: pd.DataFrame, team_name: str) -> int:
    sub = table[table["team"] == team_name]
    return int(sub["wins"].iloc[0]) if len(sub) else 0


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IPL Smart Analyst",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_styles()

st.markdown(
    """
<div class="hero-sub">Match insights & prediction system</div>
<h1 class="hero-title">🏏 IPL Smart Analyst</h1>
    """,
    unsafe_allow_html=True,
)

matches_raw = load_matches_df()
deliveries_raw = load_deliveries_df()
bundle = load_model_bundle()

bat_df = batsman_table(deliveries_raw)
bowl_df = bowler_table(deliveries_raw)
wins_df = team_wins_table(matches_raw)

tab_pred, tab_team, tab_players = st.tabs(
    ["Match Prediction", "Team Analysis", "Player Insights"]
)

# ----- Tab: Match Prediction -----
with tab_pred:
    st.markdown('<p class="section-title">Match Prediction</p>', unsafe_allow_html=True)

    if bundle is None:
        st.error(
            "Model bundle not found. Run `python model/train_model.py` from the project root, then refresh."
        )
    else:
        teams = list(bundle["team_encoder"].classes_)
        venues = list(bundle["venue_encoder"].classes_)
        toss_opts = list(bundle["toss_encoder"].classes_)

        c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
        with c1:
            st.markdown('<span class="stat-label">Team A (reference side)</span>', unsafe_allow_html=True)
            team1 = st.selectbox("Team A", teams, key="t1", label_visibility="visible")
        with c2:
            st.markdown('<span class="stat-label">Team B</span>', unsafe_allow_html=True)
            team2_opts = [t for t in teams if t != team1]
            team2 = st.selectbox("Team B", team2_opts, key="t2")
        with c3:
            st.markdown('<span class="stat-label">Venue</span>', unsafe_allow_html=True)
            venue = st.selectbox("Venue", venues, key="ven")

        t1, t2 = st.columns(2)
        with t1:
            toss_winner = st.selectbox("Toss winner", [team1, team2])
        with t2:
            toss_decision = st.selectbox("Toss decision", toss_opts)

        st.markdown("<br/>", unsafe_allow_html=True)
        predict = st.button("Predict Winner", type="primary", use_container_width=False)

        if team1 == team2:
            st.warning("Pick two different teams.")

        if predict and team1 != team2:
            X = build_prediction_row(bundle, team1, team2, toss_winner, toss_decision, venue)
            proba = bundle["model"].predict_proba(X)[0]
            p_team1 = float(proba[1])
            p_team2 = float(proba[0])
            pred_team = team1 if p_team1 >= p_team2 else team2
            win_p = max(p_team1, p_team2)

            st.markdown(
                f"""
<div class="win-banner">
<strong>Predicted winner · {pred_team}</strong>
<p style="margin:0.5rem 0 0 0; color:#b8bcc8; font-size:0.92rem;">
Win probability for this pick: <b>{win_p*100:.1f}%</b>
</p>
</div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(f"Confidence bar for predicted winner: {pred_team}")
            st.progress(win_p)

            st.markdown('<hr class="sep"/>', unsafe_allow_html=True)
            st.markdown("**Key stats comparison**", unsafe_allow_html=True)
            wp = bundle["team_win_pct"]
            sr = bundle["team_bat_sr"]
            be = bundle["team_bowl_econ"]

            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(
                    f'<div class="card"><span class="stat-label">{team1}</span>'
                    f'<p class="stat-value">{wp.get(team1,0)*100:.1f}%</p>'
                    f'<p class="card-muted">Historical win rate · SR {sr.get(team1,0):.1f} · Econ {be.get(team1,0):.2f}</p></div>',
                    unsafe_allow_html=True,
                )
            with sc2:
                st.markdown(
                    f'<div class="card"><span class="stat-label">{team2}</span>'
                    f'<p class="stat-value">{wp.get(team2,0)*100:.1f}%</p>'
                    f'<p class="card-muted">Historical win rate · SR {sr.get(team2,0):.1f} · Econ {be.get(team2,0):.2f}</p></div>',
                    unsafe_allow_html=True,
                )

            with st.expander("Model evaluation (test set)"):
                m = bundle["metrics"]
                st.write(f"**Algorithm:** {m['model_name']}")
                st.write(f"**Hold-out accuracy:** {m['accuracy']:.4f}")
                cm = np.array(m["confusion_matrix"])
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=["Pred: Team B", "Pred: Team A"],
                        y=["Actual: Team B won", "Actual: Team A won"],
                        colorscale="Viridis",
                        text=cm,
                        texttemplate="%{text}",
                        showscale=False,
                    )
                )
                fig_cm.update_layout(
                    title="Confusion matrix",
                    height=320,
                    margin=dict(l=40, r=20, t=50, b=40),
                    **plotly_theme(),
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption("Rows = true outcome (whether Team A won). This is a tough baseline problem; accuracy near 55% is common.")

# ----- Tab: Team Analysis -----
with tab_team:
    st.markdown('<p class="section-title">Team Analysis</p>', unsafe_allow_html=True)
    all_teams = sorted(wins_df["team"].astype(str).unique())
    if len(all_teams) < 2:
        st.info("Need at least two teams in the dataset for comparison charts.")
    else:
        a1, a2 = st.columns(2)
        with a1:
            compare_a = st.selectbox("Team 1", all_teams, key="cmp1")
        rest = [t for t in all_teams if t != compare_a]
        with a2:
            compare_b = st.selectbox("Team 2", rest, key="cmp2")

        wp_map = utils.compute_team_win_percentage(matches_raw)
        sr_map = utils.team_batting_strike_rate(deliveries_raw)
        econ_map = utils.team_bowling_economy(deliveries_raw)

        comp = pd.DataFrame(
            {
                "team": [compare_a, compare_b],
                "wins": [wins_lookup(wins_df, compare_a), wins_lookup(wins_df, compare_b)],
                "win_pct": [wp_map.get(compare_a, 0) * 100, wp_map.get(compare_b, 0) * 100],
                "strike_rate": [sr_map.get(compare_a, 0), sr_map.get(compare_b, 0)],
                "economy": [econ_map.get(compare_a, 0), econ_map.get(compare_b, 0)],
            }
        )

        bc1, bc2 = st.columns(2)
        with bc1:
            fig_w = px.bar(
                comp,
                x="team",
                y="wins",
                title="Total wins (dataset)",
                color="team",
                color_discrete_sequence=["#f0b429", "#3b82f6"],
            )
            fig_w.update_layout(showlegend=False, height=360, **plotly_theme())
            st.plotly_chart(fig_w, use_container_width=True)
        with bc2:
            fig_m = go.Figure()
            fig_m.add_trace(
                go.Bar(name="Win %", x=comp["team"], y=comp["win_pct"], marker_color="#22c55e")
            )
            fig_m.update_layout(barmode="group", title="Historical win %", height=360, **plotly_theme())
            st.plotly_chart(fig_m, use_container_width=True)

        fig_adv = go.Figure()
        fig_adv.add_trace(
            go.Bar(name="Batting SR", x=comp["team"], y=comp["strike_rate"], marker_color="#f0b429")
        )
        fig_adv.add_trace(
            go.Bar(name="Bowling economy", x=comp["team"], y=comp["economy"], marker_color="#6366f1")
        )
        fig_adv.update_layout(
            barmode="group",
            title="Squad tempo: team batting SR vs bowling economy (all IPL balls in data)",
            height=400,
            **plotly_theme(),
        )
        st.plotly_chart(fig_adv, use_container_width=True)

        max_n = max(1, min(20, len(wins_df)))
        default_top = min(10, max_n)
        top_n = st.slider("Show top N teams by wins", 1, max_n, default_top)
        top_side = wins_df.head(top_n)
        fig_top = px.bar(
            top_side,
            x="wins",
            y="team",
            orientation="h",
            title=f"Top {top_n} teams by wins",
            color="wins",
            color_continuous_scale="Sunset",
        )
        fig_top.update_layout(height=420, yaxis=dict(autorange="reversed"), **plotly_theme())
        st.plotly_chart(fig_top, use_container_width=True)

# ----- Tab: Player Insights -----
with tab_players:
    st.markdown('<p class="section-title">Player Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="card-muted">Features from ball-by-ball data: <code>batting_avg</code>, '
        "<code>strike_rate</code>, <code>economy_rate</code>; squad strength uses <code>team_win_percentage</code> elsewhere.</p>",
        unsafe_allow_html=True,
    )

    q_balls_bat, q_balls_bowl = st.columns(2)
    with q_balls_bat:
        min_b = st.slider("Min balls for batting leaderboard", 200, 2000, 500, 50)
    with q_balls_bowl:
        min_bo = st.slider("Min balls for bowling leaderboard", 200, 2000, 500, 50)

    top_bat = utils.top_batsmen_by_strike_rate(bat_df, min_balls=min_b, n=10)
    top_bowl = utils.top_bowlers_by_economy(bowl_df, min_balls=min_bo, n=10)

    p1, p2 = st.columns(2)
    with p1:
        if len(top_bat) == 0:
            st.warning("No batsmen meet the minimum balls filter — lower the threshold.")
        else:
            fig_bs = px.bar(
                top_bat.sort_values("strike_rate"),
                x="strike_rate",
                y="batter",
                orientation="h",
                title="Top 10 batsmen by strike rate",
                text="strike_rate",
                color="strike_rate",
                color_continuous_scale="YlOrRd",
            )
            fig_bs.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_bs.update_layout(height=440, yaxis=dict(autorange="reversed"), **plotly_theme())
            st.plotly_chart(fig_bs, use_container_width=True)
    with p2:
        if len(top_bowl) == 0:
            st.warning("No bowlers meet the minimum balls filter — lower the threshold.")
        else:
            fig_bo = px.bar(
                top_bowl.sort_values("economy_rate", ascending=False),
                x="economy_rate",
                y="bowler",
                orientation="h",
                title="Top 10 bowlers by economy (lower is better)",
                text="economy_rate",
                color="economy_rate",
                color_continuous_scale="Blues_r",
            )
            fig_bo.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_bo.update_layout(height=440, yaxis=dict(autorange="reversed"), **plotly_theme())
            st.plotly_chart(fig_bo, use_container_width=True)

    with st.expander("Raw leaderboards (table)"):
        st.subheader("Batsmen")
        b_show = top_bat.copy()
        if len(b_show):
            b_show["batting_avg"] = b_show["batting_avg"].round(2)
            b_show["strike_rate"] = b_show["strike_rate"].round(2)
        st.dataframe(b_show, use_container_width=True, hide_index=True)
        st.subheader("Bowlers")
        bo_show = top_bowl.copy()
        if len(bo_show):
            bo_show["economy_rate"] = bo_show["economy_rate"].round(2)
        st.dataframe(bo_show, use_container_width=True, hide_index=True)

st.markdown(
    """
<div style="text-align:center; margin-top:2.5rem; font-size:0.72rem; color:#3d4154; letter-spacing:0.08em;">
IPL Smart Analyst · Random Forest · Pandas · scikit-learn · Plotly
</div>
    """,
    unsafe_allow_html=True,
)
