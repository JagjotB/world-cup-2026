from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TEAM_TACTICS_FILE

STYLE_DIMS = [
    "pressing_intensity",
    "defensive_line",
    "tempo",
    "directness",
    "possession_style",
    "width",
    "set_piece_strength",
    "transition_speed",
]

STYLE_FEATURE_COLUMNS = [
    "style_similarity",
    "style_distance",
    "pressing_battle",
    "possession_battle",
    "tempo_battle",
    "home_style_directness",
    "away_style_directness",
    "directness_edge",
    "home_style_pressing",
    "away_style_pressing",
    "pressing_edge",
    "home_set_piece_strength",
    "away_set_piece_strength",
    "set_piece_edge",
    "transition_edge",
    "style_cluster_home",
    "style_cluster_away",
    "style_cluster_mismatch",
]

N_STYLE_CLUSTERS = 5


def load_style_vectors(
    path: Path = TEAM_TACTICS_FILE,
) -> dict[str, np.ndarray]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    if df.empty:
        return {}
    missing = [c for c in STYLE_DIMS if c not in df.columns]
    if missing:
        return {}
    df = df.dropna(subset=STYLE_DIMS + ["team"])
    vectors: dict[str, np.ndarray] = {}
    for row in df.itertuples(index=False):
        team = str(row.team)
        vec = np.array([float(getattr(row, dim)) for dim in STYLE_DIMS], dtype=float)
        vectors[team] = vec
    return vectors


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))


def fit_style_clusters(
    vectors: dict[str, np.ndarray],
    n_clusters: int = N_STYLE_CLUSTERS,
    random_state: int = 42,
) -> dict[str, int]:
    if len(vectors) < n_clusters:
        return {team: 0 for team in vectors}
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {team: 0 for team in vectors}

    teams = list(vectors.keys())
    matrix = np.stack([vectors[t] for t in teams])
    matrix = StandardScaler().fit_transform(matrix)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(matrix)
    return {team: int(label) for team, label in zip(teams, labels)}


def style_matchup_features(
    home_team: str,
    away_team: str,
    vectors: dict[str, np.ndarray],
    clusters: dict[str, int],
) -> dict[str, float]:
    default_vec = np.array([0.6, 0.5, 0.6, 0.55, 0.5, 0.6, 0.55, 0.65])

    home_vec = vectors.get(home_team, default_vec)
    away_vec = vectors.get(away_team, default_vec)

    def dim(vec: np.ndarray, name: str) -> float:
        idx = STYLE_DIMS.index(name)
        return float(vec[idx])

    home_cluster = clusters.get(home_team, 0)
    away_cluster = clusters.get(away_team, 0)

    return {
        "style_similarity": _cosine_similarity(home_vec, away_vec),
        "style_distance": float(np.linalg.norm(home_vec - away_vec)),
        "pressing_battle": dim(home_vec, "pressing_intensity") * dim(away_vec, "pressing_intensity"),
        "possession_battle": dim(home_vec, "possession_style") * dim(away_vec, "possession_style"),
        "tempo_battle": dim(home_vec, "tempo") * dim(away_vec, "tempo"),
        "home_style_directness": dim(home_vec, "directness"),
        "away_style_directness": dim(away_vec, "directness"),
        "directness_edge": dim(home_vec, "directness") - dim(away_vec, "directness"),
        "home_style_pressing": dim(home_vec, "pressing_intensity"),
        "away_style_pressing": dim(away_vec, "pressing_intensity"),
        "pressing_edge": dim(home_vec, "pressing_intensity") - dim(away_vec, "pressing_intensity"),
        "home_set_piece_strength": dim(home_vec, "set_piece_strength"),
        "away_set_piece_strength": dim(away_vec, "set_piece_strength"),
        "set_piece_edge": dim(home_vec, "set_piece_strength") - dim(away_vec, "set_piece_strength"),
        "transition_edge": dim(home_vec, "transition_speed") - dim(away_vec, "transition_speed"),
        "style_cluster_home": float(home_cluster),
        "style_cluster_away": float(away_cluster),
        "style_cluster_mismatch": float(abs(home_cluster - away_cluster)),
    }
