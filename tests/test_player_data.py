import pandas as pd
import pytest

from worldcup2026.features import team_player_feature_row
from worldcup2026.player_data import build_player_feature_tables, split_club


def test_split_club_extracts_country_code():
    assert split_club("Manchester United FC (ENG)") == ("Manchester United FC", "ENG")


def test_build_player_feature_tables_from_squad_rows():
    squad = pd.DataFrame(
        [
            {
                "team": "Example",
                "position": "FW",
                "player_name": "A Player",
                "date_of_birth": pd.Timestamp("2000-01-01"),
                "club": "Club A",
                "height_cm": 180,
                "international_caps": 20,
                "international_goals": 5,
                "player_key": "aplayer",
                "club_key": "cluba",
                "age_on_2026_06_11": 26.45,
            },
            {
                "team": "Example",
                "position": "GK",
                "player_name": "B Keeper",
                "date_of_birth": pd.Timestamp("1995-01-01"),
                "club": "Club B",
                "height_cm": 190,
                "international_caps": 10,
                "international_goals": 0,
                "player_key": "bkeeper",
                "club_key": "clubb",
                "age_on_2026_06_11": 31.45,
            },
        ]
    )

    _, team_features = build_player_feature_tables(squad)

    assert team_features.loc[0, "squad_players"] == 2
    assert team_features.loc[0, "international_caps_sum"] == 30
    assert team_features.loc[0, "players_fw"] == 1
    assert team_features.loc[0, "players_gk"] == 1
    assert "top_11_attacking_score" in team_features.columns
    assert "starting_gk_keeper_score" in team_features.columns
    assert "projection_goal_threat" in team_features.columns
    assert team_features.loc[0, "top_11_experience_score"] > 0


def test_team_player_feature_row_adds_matchup_edges():
    lookup = {
        "home": {
            "starting_fw_attacking_score": 2.0,
            "starting_mf_creativity_score": 1.5,
            "starting_def_defensive_score": 1.0,
            "starting_gk_keeper_score": 0.8,
            "top_11_creativity_score": 1.2,
            "top_11_ball_winning_score": 0.9,
            "starting_fw_shot_volume_score": 1.4,
            "top_11_physical_score": 1.1,
            "bench_depth_score": 3.0,
            "top_11_discipline_risk": 0.2,
        },
        "away": {
            "starting_fw_attacking_score": 1.0,
            "starting_mf_creativity_score": 0.5,
            "starting_def_defensive_score": 0.7,
            "starting_gk_keeper_score": 0.4,
            "top_11_creativity_score": 0.8,
            "top_11_ball_winning_score": 1.4,
            "starting_fw_shot_volume_score": 0.9,
            "top_11_physical_score": 0.9,
            "bench_depth_score": 2.0,
            "top_11_discipline_risk": 0.5,
        },
    }

    row = team_player_feature_row("Home", "Away", lookup, defaults={})

    assert row["home_attack_vs_away_defense"] == 2.4
    assert row["keeper_edge"] == 0.4
    assert row["bench_depth_edge"] == 1.0
    assert row["discipline_risk_edge"] == 0.3


def test_team_player_feature_row_adds_projection_matchup_edges():
    lookup = {
        "home": {
            "projection_goal_threat": 3.0,
            "projection_assist_threat": 2.0,
            "projection_shot_threat": 4.0,
            "projection_defensive_work": 1.0,
            "projection_keeper_coverage": 0.8,
            "projection_card_risk": 0.4,
            "projection_bench_goal_threat": 0.8,
            "projection_bench_assist_threat": 0.4,
            "projection_balance_score": 9.0,
        },
        "away": {
            "projection_goal_threat": 2.0,
            "projection_assist_threat": 1.0,
            "projection_shot_threat": 3.0,
            "projection_defensive_work": 0.6,
            "projection_keeper_coverage": 0.5,
            "projection_card_risk": 0.7,
            "projection_bench_goal_threat": 0.3,
            "projection_bench_assist_threat": 0.2,
            "projection_balance_score": 6.0,
        },
    }

    row = team_player_feature_row("Home", "Away", lookup, defaults={})

    assert row["home_projection_attack_vs_away_resistance"] == pytest.approx(5.3)
    assert row["away_projection_attack_vs_home_resistance"] == pytest.approx(2.25)
    assert row["projection_attack_matchup_diff"] == pytest.approx(3.05)
    assert row["projection_balance_edge"] == 3.0


def test_build_player_features_uses_availability_lineups_and_tactics():
    squad = pd.DataFrame(
        [
            {
                "team": "Example",
                "position": "FW" if idx else "GK",
                "player_name": f"Player {idx}",
                "date_of_birth": pd.Timestamp("2000-01-01"),
                "club": "Club",
                "height_cm": 180,
                "international_caps": 10 + idx,
                "international_goals": idx,
                "player_key": f"player{idx}",
                "club_key": "club",
                "age_on_2026_06_11": 26,
            }
            for idx in range(12)
        ]
    )
    availability = pd.DataFrame(
        [
            {
                "team": "Example",
                "player_name": "Player 11",
                "status": "out",
            }
        ]
    )
    lineups = pd.DataFrame(
        [
            {
                "team": "Example",
                "player_name": "Player 10",
                "lineup_slot": 1,
                "lineup_position": "FW",
                "lineup_status": "confirmed",
                "expected_minutes_share": 0.9,
                "formation": "4-3-3",
            }
        ]
    )
    tactics = pd.DataFrame(
        [
            {
                "team": "Example",
                "formation": "4-3-3",
                "pressing_intensity": 0.8,
                "defensive_line": "high",
                "tempo": 0.6,
                "directness": 0.4,
                "possession_style": "possession",
                "width": "wide",
                "set_piece_strength": 0.7,
                "transition_speed": 0.8,
            }
        ]
    )

    players, team_features = build_player_feature_tables(
        squad,
        availability=availability,
        manual_lineups=lineups,
        team_tactics=tactics,
    )

    unavailable = players[players["player_name"].eq("Player 11")].iloc[0]
    assert unavailable["availability_multiplier"] == 0.0
    assert unavailable["is_unavailable"] == 1.0
    assert team_features.loc[0, "is_unavailable_sum"] == 1.0
    assert team_features.loc[0, "tactics_available"] == 1.0
    assert team_features.loc[0, "formation_back_line"] == 4.0
    assert team_features.loc[0, "tactic_pressing_intensity"] == 0.8
