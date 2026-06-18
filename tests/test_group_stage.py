import pandas as pd

from worldcup2026.group_stage import add_match_usefulness_filters, canonical_schedule_team


def test_canonical_schedule_team_maps_turkey_to_project_team_name():
    assert canonical_schedule_team("Turkey") == "Turkiye"
    assert canonical_schedule_team(" Türkiye ") == "Turkiye"


def test_add_match_usefulness_filters_adds_double_chance_and_btts():
    matches = pd.DataFrame(
        [
            {
                "home_team": "Home",
                "away_team": "Away",
                "predicted_result": "Home",
                "top_probability": 0.52,
                "p_home_win": 0.52,
                "p_draw": 0.24,
                "p_away_win": 0.24,
                "expected_home_goals": 1.2,
                "expected_away_goals": 0.8,
            },
            {
                "home_team": "Low",
                "away_team": "Edge",
                "predicted_result": "Low",
                "top_probability": 0.41,
                "p_home_win": 0.41,
                "p_draw": 0.31,
                "p_away_win": 0.28,
                "expected_home_goals": 1.4,
                "expected_away_goals": 0.5,
            },
        ]
    )

    filtered = add_match_usefulness_filters(matches)

    assert filtered.iloc[0]["double_chance_pick"] == "Home or Draw"
    assert filtered.iloc[0]["double_chance_probability"] == 0.76
    assert bool(filtered.iloc[0]["double_chance_useful"]) is True
    assert filtered.iloc[0]["either_team_wins_probability"] == 0.76
    assert bool(filtered.iloc[0]["either_team_wins_useful"]) is False
    assert filtered.iloc[0]["btts_pick"] == "Yes"
    assert bool(filtered.iloc[0]["btts_useful"]) is True
    assert bool(filtered.iloc[1]["double_chance_useful"]) is False
    assert filtered.iloc[1]["btts_pick"] == "No play"
