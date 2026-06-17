import pytest
import pandas as pd

from worldcup2026.player_projections import (
    project_player_match_performances,
    summarize_team_player_projections,
)


class ProjectionPredictor:
    def predict_match(self, home_team, away_team, neutral=True):
        class Prediction:
            p_home_win = 0.45
            p_draw = 0.30
            p_away_win = 0.25
            expected_home_goals = 1.6
            expected_away_goals = 1.1

        return Prediction()


def _players(team):
    rows = []
    for idx in range(20):
        position = "GK" if idx < 2 else "DF" if idx < 8 else "MF" if idx < 14 else "FW"
        rows.append(
            {
                "team": team,
                "player_name": f"{team} Player {idx}",
                "position": position,
                "club": "Club",
                "height_cm": 180,
                "international_caps": 20 - idx,
                "international_goals": idx % 5,
                "has_club_stats": True,
                "min": 1800 - idx * 20,
                "starts": 18,
                "gls": idx % 7,
                "ast": idx % 4,
                "sh": 30 + idx,
                "sot": 10 + idx,
                "tklw": 15,
                "int": 12,
                "saves": 20 if position == "GK" else 0,
                "cs": 6 if position == "GK" else 0,
                "availability_status": "available",
                "availability_multiplier": 1.0,
                "expected_minutes_share": 1.0,
                "attacking_score": 1.0 + (idx % 5) * 0.2,
                "creativity_score": 0.8 + (idx % 3) * 0.2,
                "shot_volume_score": 1.1 + (idx % 4) * 0.2,
                "ball_winning_score": 0.9,
                "defensive_score": 1.0,
                "keeper_score": 1.5 if position == "GK" else 0.0,
                "discipline_risk": 0.1,
            }
        )
    return rows


def test_player_match_projections_allocate_team_expected_goals(tmp_path):
    schedule = pd.DataFrame(
        [
            {
                "match_number": 1,
                "group": "A",
                "status": "upcoming",
                "local_date": "2026-06-18",
                "local_time": "12:00",
                "home_team": "Home",
                "away_team": "Away",
                "venue": "Venue",
            }
        ]
    )
    players = pd.DataFrame(_players("Home") + _players("Away"))

    projections = project_player_match_performances(
        ProjectionPredictor(),
        schedule,
        players,
        output_path=tmp_path / "players.csv",
        from_date="2026-06-18",
    )

    assert len(projections) == 36
    assert set(projections["roster_role"]) == {"starter", "bench"}
    assert projections[projections["team"].eq("Home")]["projected_goals"].sum() == pytest.approx(
        1.6,
        abs=0.02,
    )
    assert projections[projections["team"].eq("Away")]["projected_goals"].sum() == pytest.approx(
        1.1,
        abs=0.02,
    )


def test_projection_summaries_create_model_ready_team_features():
    players = pd.DataFrame(_players("Home") + _players("Away"))

    summaries = summarize_team_player_projections(players)
    home = summaries[summaries["team"].eq("Home")].iloc[0]

    assert home["projection_starter_count"] == 11
    assert home["projection_bench_count"] == 7
    assert home["projection_minutes_total"] > 700
    assert home["projection_goal_threat"] > 0
    assert home["projection_balance_score"] > 0
