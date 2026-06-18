import numpy as np

from worldcup2026.tactical_embeddings import style_matchup_features


def test_style_matchup_features_keep_raw_dimension_edges():
    vectors = {
        "Pressers": np.array([0.9, 0.7, 0.8, 0.3, 0.6, 0.6, 0.7, 0.8]),
        "Block": np.array([0.4, 0.3, 0.5, 0.8, 0.3, 0.5, 0.4, 0.5]),
    }

    features = style_matchup_features("Pressers", "Block", vectors, {})

    assert features["home_style_pressing"] == 0.9
    assert features["away_style_pressing"] == 0.4
    assert features["pressing_edge"] == 0.5
    assert features["home_style_directness"] == 0.3
    assert features["away_style_directness"] == 0.8
    assert features["directness_edge"] == -0.5
    assert features["style_distance"] > 0.0
