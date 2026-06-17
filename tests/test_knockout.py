import pytest

from worldcup2026.knockout import predict_knockout_resolution
from worldcup2026.model import RatingOnlyPredictor


class SimpleKnockoutPredictor(RatingOnlyPredictor):
    def predict_match(self, home_team, away_team, neutral=True):
        class Prediction:
            p_home_win = 0.45
            p_draw = 0.30
            p_away_win = 0.25
            expected_home_goals = 1.4
            expected_away_goals = 1.0

        return Prediction()


def test_knockout_probabilities_sum_to_one():
    predictor = SimpleKnockoutPredictor({"A": 1600, "B": 1500})
    result = predict_knockout_resolution(predictor, "A", "B")

    assert result["p_home_advance_total"] + result["p_away_advance_total"] == pytest.approx(1.0)
    assert result["p_extra_time"] == pytest.approx(result["p_draw_regulation"])
    assert result["p_penalty_shootout"] <= result["p_extra_time"]
    assert result["golden_goal_rule_active"] is False
    assert result["p_golden_goal"] == 0.0
