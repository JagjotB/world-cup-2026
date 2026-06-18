import numpy as np
import pytest

from worldcup2026.train import blend_with_poisson_draw_probability


def test_draw_blend_weight_controls_poisson_draw_influence():
    probabilities = np.array([[0.2, 0.1, 0.7]])
    no_blend = blend_with_poisson_draw_probability(
        probabilities,
        home_lambdas=np.array([1.0]),
        away_lambdas=np.array([1.0]),
        draw_blend_weight=0.0,
    )
    blended = blend_with_poisson_draw_probability(
        probabilities,
        home_lambdas=np.array([1.0]),
        away_lambdas=np.array([1.0]),
        draw_blend_weight=1.0,
    )

    assert no_blend[0, 1] == pytest.approx(probabilities[0, 1])
    assert blended[0, 1] > no_blend[0, 1]
    assert blended.sum() == pytest.approx(1.0)
    assert blended[0, 0] < probabilities[0, 0]
    assert blended[0, 2] < probabilities[0, 2]
