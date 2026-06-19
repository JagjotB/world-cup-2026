"""
FIFA World Cup 2026 Prediction Dashboard
Run: streamlit run dashboard.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import (
    GROUP_STAGE_SCHEDULE_FILE,
    MODEL_FILE,
    UPCOMING_GROUP_STAGE_PREDICTIONS_FILE,
)
from worldcup2026.live import apply_played_results_to_predictor, played_schedule_results
from worldcup2026.match_simulator import (
    outcome_probabilities,
    run_simulations,
    scoreline_distribution,
    sim_predict_result,
)
from worldcup2026.model import MatchPredictor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="World Cup 2026 Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TODAY = date.today().isoformat()

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    predictor = MatchPredictor.load(MODEL_FILE)
    schedule = pd.read_csv(GROUP_STAGE_SCHEDULE_FILE)
    results = played_schedule_results(schedule)
    apply_played_results_to_predictor(predictor, results)
    return predictor


@st.cache_data(ttl=300, show_spinner="Loading schedule...")
def load_schedule():
    return pd.read_csv(GROUP_STAGE_SCHEDULE_FILE)


@st.cache_data(ttl=300, show_spinner="Loading predictions...")
def load_predictions():
    if UPCOMING_GROUP_STAGE_PREDICTIONS_FILE.exists():
        return pd.read_csv(UPCOMING_GROUP_STAGE_PREDICTIONS_FILE)
    return pd.DataFrame()


def result_label(h, a):
    if h > a:
        return "H"
    elif a > h:
        return "A"
    return "D"


def prob_bar(p: float, color: str = "#1f77b4") -> str:
    pct = int(p * 100)
    return f"""
    <div style="background:#eee;border-radius:4px;height:8px;margin:2px 0">
      <div style="background:{color};width:{pct}%;height:8px;border-radius:4px"></div>
    </div>
    <small>{pct}%</small>
    """


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚽ FIFA World Cup 2026")
st.caption(f"Predictions powered by XGBoost + Poisson simulation · Updated {TODAY}")

predictor = load_predictor()
schedule = load_schedule()
predictions = load_predictions()

played = schedule[schedule["status"] == "played"].copy()
upcoming = schedule[schedule["status"] != "played"].copy()
today_matches = schedule[schedule["local_date"] == TODAY].copy()

# ---------------------------------------------------------------------------
# Accuracy banner
# ---------------------------------------------------------------------------
if not played.empty:
    played["home_score"] = pd.to_numeric(played["home_score"], errors="coerce")
    played["away_score"] = pd.to_numeric(played["away_score"], errors="coerce")

    correct = total = 0
    for r in played.itertuples(index=False):
        try:
            pred = predictor.predict_match(r.home_team, r.away_team, neutral=True)
            dec = predictor.decision_for_prediction(pred)
            actual = result_label(r.home_score, r.away_score)
            rec = dec.recommended_result
            if rec == r.home_team or rec == "H":
                predicted = "H"
            elif rec == r.away_team or rec == "A":
                predicted = "A"
            else:
                predicted = "D"
            # Draw-aware sim rule
            sims = run_simulations(r.home_team, r.away_team, pred.expected_home_goals, pred.expected_away_goals, n=5000, seed=42)
            probs = outcome_probabilities(sims)
            predicted = sim_predict_result(probs, pred.expected_home_goals, pred.expected_away_goals)
            correct += int(predicted == actual)
            total += 1
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches Played", total)
    c2.metric("Correct Predictions", correct)
    c3.metric("Accuracy", f"{correct/total:.1%}" if total else "—")
    c4.metric("Remaining", len(upcoming))
    st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_today, tab_upcoming, tab_played, tab_simulate, tab_teams = st.tabs([
    "📅 Today",
    "🔮 Upcoming",
    "📊 Played",
    "🎲 Simulate",
    "🏆 Teams",
])

# ===========================================================================
# TODAY TAB
# ===========================================================================
with tab_today:
    st.subheader(f"Matches — {TODAY}")

    if today_matches.empty:
        st.info("No matches today.")
    else:
        for row in today_matches.itertuples(index=False):
            home, away = str(row.home_team), str(row.away_team)
            is_played = str(row.status) == "played"

            try:
                pred = predictor.predict_match(home, away, neutral=True)
                sims = run_simulations(home, away, pred.expected_home_goals, pred.expected_away_goals, n=10000, seed=42)
                probs = outcome_probabilities(sims)
                pick = sim_predict_result(probs, pred.expected_home_goals, pred.expected_away_goals)
                top_scorelines = scoreline_distribution(sims, top_n=5)

                with st.container(border=True):
                    col_left, col_mid, col_right = st.columns([2, 3, 2])

                    with col_left:
                        st.markdown(f"**{row.local_time}** · Group {row.group}")
                        st.markdown(f"### {home}")
                        st.markdown(f"xG **{pred.expected_home_goals:.2f}**")

                    with col_mid:
                        if is_played:
                            st.markdown(f"<h2 style='text-align:center'>{int(row.home_score)} – {int(row.away_score)}</h2>", unsafe_allow_html=True)
                            actual = result_label(row.home_score, row.away_score)
                            correct_pick = pick == actual
                            badge = "✅" if correct_pick else "❌"
                            st.markdown(f"<p style='text-align:center'>{badge} Predicted: **{pick}**</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
                            pick_label = home if pick == "H" else (away if pick == "A" else "Draw")
                            st.markdown(f"<p style='text-align:center'>Pick: <b>{pick_label}</b></p>", unsafe_allow_html=True)

                        cols = st.columns(3)
                        cols[0].markdown(prob_bar(probs["H"], "#2196F3"), unsafe_allow_html=True)
                        cols[0].caption(home[:12])
                        cols[1].markdown(prob_bar(probs["D"], "#9E9E9E"), unsafe_allow_html=True)
                        cols[1].caption("Draw")
                        cols[2].markdown(prob_bar(probs["A"], "#F44336"), unsafe_allow_html=True)
                        cols[2].caption(away[:12])

                    with col_right:
                        st.markdown(f"### {away}")
                        st.markdown(f"xG **{pred.expected_away_goals:.2f}**")
                        st.markdown("**Top scorelines**")
                        for sl, p in top_scorelines[:3]:
                            st.caption(f"{sl} — {p:.0%}")

            except Exception as e:
                st.error(f"Could not predict {home} vs {away}: {e}")

# ===========================================================================
# UPCOMING TAB
# ===========================================================================
with tab_upcoming:
    st.subheader("Upcoming Group Stage Matches")

    if predictions.empty:
        st.warning("No predictions loaded. Run `python scripts/predict_group_stage.py` first.")
    else:
        groups = ["All"] + sorted(predictions["group"].unique().tolist())
        sel_group = st.selectbox("Filter by group", groups)

        df_show = predictions.copy()
        if sel_group != "All":
            df_show = df_show[df_show["group"] == sel_group]

        df_show = df_show[df_show["local_date"] >= TODAY].copy()

        cols_display = [
            "local_date", "local_time", "group",
            "home_team", "away_team",
            "predicted_result",
            "p_home_win", "p_draw", "p_away_win",
            "expected_home_goals", "expected_away_goals",
            "sim_top_scoreline", "sim_top_scoreline_prob",
        ]
        available = [c for c in cols_display if c in df_show.columns]
        df_show = df_show[available].copy()

        rename = {
            "local_date": "Date", "local_time": "Time", "group": "Grp",
            "home_team": "Home", "away_team": "Away",
            "predicted_result": "Pick",
            "p_home_win": "P(H)", "p_draw": "P(D)", "p_away_win": "P(A)",
            "expected_home_goals": "xG(H)", "expected_away_goals": "xG(A)",
            "sim_top_scoreline": "Top Score", "sim_top_scoreline_prob": "Score%",
        }
        df_show = df_show.rename(columns=rename)

        for col in ["P(H)", "P(D)", "P(A)", "Score%"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].apply(lambda x: f"{float(x):.0%}" if pd.notna(x) else "—")
        for col in ["xG(H)", "xG(A)"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "—")

        st.dataframe(df_show, use_container_width=True, hide_index=True)

# ===========================================================================
# PLAYED TAB
# ===========================================================================
with tab_played:
    st.subheader("Played Matches — Model vs Actual")

    rows = []
    for r in played.itertuples(index=False):
        try:
            pred = predictor.predict_match(r.home_team, r.away_team, neutral=True)
            sims = run_simulations(r.home_team, r.away_team, pred.expected_home_goals, pred.expected_away_goals, n=5000, seed=42)
            probs = outcome_probabilities(sims)
            predicted = sim_predict_result(probs, pred.expected_home_goals, pred.expected_away_goals)
            actual = result_label(r.home_score, r.away_score)
            rows.append({
                "Date": r.local_date,
                "Home": r.home_team,
                "Score": f"{int(r.home_score)}-{int(r.away_score)}",
                "Away": r.away_team,
                "Actual": actual,
                "Predicted": predicted,
                "Result": "✅" if predicted == actual else "❌",
                "P(H)": f"{probs['H']:.0%}",
                "P(D)": f"{probs['D']:.0%}",
                "P(A)": f"{probs['A']:.0%}",
            })
        except Exception:
            pass

    if rows:
        df_played = pd.DataFrame(rows)
        correct = (df_played["Result"] == "✅").sum()
        total = len(df_played)

        c1, c2, c3 = st.columns(3)
        c1.metric("Correct", f"{correct}/{total}")
        c2.metric("Accuracy", f"{correct/total:.1%}")
        c3.metric("Wrong", f"{total - correct}/{total}")

        st.dataframe(df_played, use_container_width=True, hide_index=True)

        # Breakdown
        st.markdown("**By result type**")
        cols = st.columns(3)
        for i, (result, label) in enumerate([("H", "Home Wins"), ("D", "Draws"), ("A", "Away Wins")]):
            sub = df_played[df_played["Actual"] == result]
            ok = (sub["Result"] == "✅").sum()
            cols[i].metric(label, f"{ok}/{len(sub)}", f"{ok/len(sub):.0%}" if len(sub) else "—")

# ===========================================================================
# SIMULATE TAB
# ===========================================================================
with tab_simulate:
    st.subheader("Match Simulator")

    all_teams = sorted(set(schedule["home_team"].tolist() + schedule["away_team"].tolist()))

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        home_sel = st.selectbox("Home Team", all_teams, index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
    with col2:
        st.markdown("<br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
    with col3:
        away_options = [t for t in all_teams if t != home_sel]
        away_sel = st.selectbox("Away Team", away_options, index=away_options.index("Argentina") if "Argentina" in away_options else 0)

    n_sims = st.slider("Simulations", 10_000, 100_000, 50_000, step=10_000)

    if st.button("Run Simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            try:
                pred = predictor.predict_match(home_sel, away_sel, neutral=True)
                sims = run_simulations(home_sel, away_sel, pred.expected_home_goals, pred.expected_away_goals, n=n_sims, seed=42)
                probs = outcome_probabilities(sims)
                pick = sim_predict_result(probs, pred.expected_home_goals, pred.expected_away_goals)
                dist = scoreline_distribution(sims, top_n=12)

                pick_label = home_sel if pick == "H" else (away_sel if pick == "A" else "Draw")

                st.markdown(f"### {home_sel} vs {away_sel}")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("xG " + home_sel[:10], f"{pred.expected_home_goals:.2f}")
                c2.metric(f"{home_sel[:10]} win", f"{probs['H']:.1%}")
                c3.metric("Draw", f"{probs['D']:.1%}")
                c4.metric(f"{away_sel[:10]} win", f"{probs['A']:.1%}")
                c5.metric("Pick", pick_label)

                st.markdown("**Scoreline distribution (top 12)**")
                dist_df = pd.DataFrame(dist, columns=["Scoreline", "Probability"])
                dist_df["Winner"] = dist_df["Scoreline"].apply(
                    lambda s: home_sel if int(s.split("-")[0]) > int(s.split("-")[1])
                    else (away_sel if int(s.split("-")[1]) > int(s.split("-")[0]) else "Draw")
                )
                dist_df["Probability"] = dist_df["Probability"].apply(lambda x: f"{x:.1%}")
                dist_df["Bar"] = dist_df["Probability"]
                st.dataframe(dist_df[["Scoreline", "Probability", "Winner"]], use_container_width=True, hide_index=True)

                # Sample timeline
                st.markdown("**Sample match timeline**")
                rng = np.random.default_rng(99)
                idx = int(rng.integers(0, len(sims)))
                sim = sims[idx]
                result_str = home_sel if sim.result() == "H" else (away_sel if sim.result() == "A" else "Draw")
                st.code(f"{home_sel} {sim.home_goals}-{sim.away_goals} {away_sel}  ({result_str})\n\n{sim.timeline_str()}")

            except Exception as e:
                st.error(f"Simulation failed: {e}")

# ===========================================================================
# TEAMS TAB
# ===========================================================================
with tab_teams:
    st.subheader("Team Ratings & Form")

    all_team_names = sorted(set(schedule["home_team"].tolist() + schedule["away_team"].tolist()))

    rows = []
    for team in all_team_names:
        try:
            rating = predictor.team_rating(team)
            form = predictor.artifact.team_form.get(predictor.model_team_name(team), {})
            rows.append({
                "Team": team,
                "Elo Rating": round(rating, 0),
                "Form (pts)": round(form.get("points", 1.0), 2),
                "Form (GD)": round(form.get("goal_diff", 0.0), 2),
            })
        except Exception:
            pass

    if rows:
        df_teams = pd.DataFrame(rows).sort_values("Elo Rating", ascending=False).reset_index(drop=True)
        df_teams.index += 1
        st.dataframe(df_teams, use_container_width=True)
