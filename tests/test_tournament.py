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
