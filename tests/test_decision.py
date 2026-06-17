import numpy as np
import pandas as pd

from worldcup2026.decision import choose_recommended_result, tune_decision_policy


def test_marginal_draw_can_be_overridden_to_best_team():
    decision = choose_recommended_result(
        "Home",
        "Away",
        p_home=0.38,
        p_draw=0.39,
        p_away=0.23,
        policy={"draw_min_probability": 0.40, "draw_margin_over_team": 0.03},
    )

    assert decision.raw_top_result == "Draw"
    assert decision.recommended_result == "Home"
    assert decision.draw_override_applied is True


def test_strong_draw_is_kept():
    decision = choose_recommended_result(
        "Home",
        "Away",
        p_home=0.27,
        p_draw=0.45,
        p_away=0.28,
        policy={"draw_min_probability": 0.40, "draw_margin_over_team": 0.03},
    )

    assert decision.recommended_result == "Draw"
    assert decision.draw_override_applied is False


def test_policy_tuning_returns_thresholds():
    probabilities = np.array(
        [
            [0.40, 0.38, 0.22],
            [0.25, 0.45, 0.30],
            [0.20, 0.30, 0.50],
        ]
    )
    labels = ["H", "D", "A"]
    y_true = pd.Series(["H", "D", "A"])

    policy = tune_decision_policy(probabilities, y_true, labels)

    assert "draw_min_probability" in policy
    assert "draw_margin_over_team" in policy
