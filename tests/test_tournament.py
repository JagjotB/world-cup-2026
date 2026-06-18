import pandas as pd

from worldcup2026.data import load_teams
from worldcup2026.model import RatingOnlyPredictor
from worldcup2026.tournament import generate_group_matches, simulate_tournament_once


def test_group_fixture_generation_has_72_matches():
    teams = load_teams()
    fixtures = generate_group_matches(teams)

    assert len(teams) == 48
    assert len(fixtures) == 72
    assert pd.DataFrame(fixtures).groupby("group").size().eq(6).all()


def test_single_tournament_simulation_completes_104_matches():
    teams = load_teams()
    ratings = {row.team: 1600 - idx for idx, row in enumerate(teams.itertuples(index=False))}
    predictor = RatingOnlyPredictor(ratings)

    result = simulate_tournament_once(predictor, teams, seed=7)

    assert result["champion"] in set(teams["team"])
    assert len(result["matches"]) == 104
    assert set(result["stage_reached"]).issubset(set(teams["team"]))


def test_tournament_simulation_uses_supplied_group_predictions():
    teams = load_teams()
    predictor = RatingOnlyPredictor({team: 1800 for team in teams["team"]})
    first_fixture = generate_group_matches(teams)[0]
    group_predictions = {
        (
            str(first_fixture["group"]),
            str(first_fixture["home_team"]),
            str(first_fixture["away_team"]),
        ): {
            "p_home_win": 0.0,
            "p_draw": 0.0,
            "p_away_win": 1.0,
            "expected_home_goals": 0.5,
            "expected_away_goals": 2.5,
        }
    }

    result = simulate_tournament_once(
        predictor,
        teams,
        group_prediction_lookup=group_predictions,
        seed=7,
    )
    first_match = result["matches"].iloc[0]

    assert first_match["home_team"] == first_fixture["home_team"]
    assert first_match["away_team"] == first_fixture["away_team"]
    assert first_match["away_score"] > first_match["home_score"]
