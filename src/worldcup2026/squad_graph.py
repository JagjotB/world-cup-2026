"""
GNN-lite squad chemistry module.

Builds a club co-appearance graph per national team squad and applies
one round of message passing to produce chemistry-aware player embeddings.
Team-level embeddings are then aggregated as match features.

No training required — the graph structure (club partnerships) acts as
inductive bias and enriches the raw player feature aggregation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PLAYER_FEATURES_FILE

NODE_FEATURE_COLS = [
    "international_caps",
    "gls",
    "ast",
    "sh",
    "sot",
    "tklw",
    "saves",
    "attacking_score",
    "creativity_score",
    "defensive_score",
    "keeper_score",
    "physical_score",
    "availability_multiplier",
    "expected_minutes_share",
]

GNN_EMBED_DIM = 8
GNN_FEATURE_COLUMNS = [
    "gnn_attack_embed",
    "gnn_defense_embed",
    "gnn_creativity_embed",
    "gnn_keeper_embed",
    "gnn_experience_embed",
    "gnn_cohesion",
    "gnn_top5_league_ratio",
    "gnn_depth_embed",
]

GNN_MATCH_FEATURE_COLUMNS = [
    *[f"home_{c}" for c in GNN_FEATURE_COLUMNS],
    *[f"away_{c}" for c in GNN_FEATURE_COLUMNS],
    *[f"gnn_edge_{c}" for c in GNN_FEATURE_COLUMNS],
    "gnn_chemistry_advantage",
]

TOP5_LEAGUE_CODES = {"ENG", "ESP", "GER", "ITA", "FRA"}


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _node_features(player_row) -> np.ndarray:
    return np.array(
        [_safe_float(getattr(player_row, col, 0.0)) for col in NODE_FEATURE_COLS],
        dtype=float,
    )


def _build_adjacency(players: list, alpha: float = 0.5) -> np.ndarray:
    """
    Weighted adjacency matrix. Players from the same club get edge weight alpha;
    all players have a self-loop of weight 1.0.
    """
    n = len(players)
    clubs = [str(getattr(p, "club", "")) for p in players]
    adj = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if clubs[i] and clubs[i] == clubs[j]:
                adj[i, j] = alpha
                adj[j, i] = alpha
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return adj / row_sums


def _message_pass(node_matrix: np.ndarray, adj_norm: np.ndarray) -> np.ndarray:
    """One round of mean-aggregation message passing."""
    return adj_norm @ node_matrix


def _top5_ratio(players: list) -> float:
    codes = [str(getattr(p, "club_country_code", "")) for p in players]
    if not codes:
        return 0.0
    return sum(1 for c in codes if c in TOP5_LEAGUE_CODES) / len(codes)


def _club_cohesion(players: list) -> float:
    clubs: dict[str, int] = {}
    for p in players:
        c = str(getattr(p, "club", "UNKNOWN"))
        clubs[c] = clubs.get(c, 0) + 1
    n = len(players)
    if n == 0:
        return 0.0
    return max(clubs.values()) / n


def _bench_depth(players: list) -> float:
    scores = [
        _safe_float(getattr(p, "attacking_score", 0.0))
        + _safe_float(getattr(p, "defensive_score", 0.0))
        + _safe_float(getattr(p, "creativity_score", 0.0))
        for p in players
    ]
    scores_sorted = sorted(scores, reverse=True)
    bench_scores = scores_sorted[11:] if len(scores_sorted) > 11 else scores_sorted
    return float(np.mean(bench_scores)) if bench_scores else 0.0


def team_gnn_embedding(
    team_players: pd.DataFrame,
    alpha: float = 0.5,
    global_mean: np.ndarray | None = None,
    global_std: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute GNN-lite team embedding from squad data.

    alpha controls the club-partner message weight (0 = no graph, 1 = full blend).
    global_mean/std should come from the full player dataset for proper normalisation.
    """
    if team_players.empty:
        return {col: 0.0 for col in GNN_FEATURE_COLUMNS}

    players = list(team_players.itertuples(index=False))
    node_matrix = np.stack([_node_features(p) for p in players])

    # Normalise using global stats if available, else per-batch
    mean = global_mean if global_mean is not None else node_matrix.mean(axis=0)
    std = global_std if global_std is not None else node_matrix.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    node_norm = (node_matrix - mean) / std

    adj_norm = _build_adjacency(players, alpha=alpha)
    aggregated = _message_pass(node_norm, adj_norm)

    dim_names = NODE_FEATURE_COLS

    def agg_mean(col_name: str) -> float:
        idx = dim_names.index(col_name) if col_name in dim_names else None
        if idx is None:
            return 0.0
        return float(np.mean(aggregated[:, idx]))

    cohesion = _club_cohesion(players)
    top5 = _top5_ratio(players)
    depth = _bench_depth(players)

    return {
        "gnn_attack_embed": agg_mean("attacking_score"),
        "gnn_defense_embed": agg_mean("defensive_score"),
        "gnn_creativity_embed": agg_mean("creativity_score"),
        "gnn_keeper_embed": agg_mean("keeper_score"),
        "gnn_experience_embed": agg_mean("international_caps"),
        "gnn_cohesion": cohesion,
        "gnn_top5_league_ratio": top5,
        "gnn_depth_embed": depth,
    }


def build_team_gnn_lookup(
    player_features_path: Path = PLAYER_FEATURES_FILE,
    alpha: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Build GNN embedding for every team using global player normalisation."""
    try:
        df = pd.read_csv(player_features_path)
    except FileNotFoundError:
        return {}
    if df.empty:
        return {}

    # Compute global mean/std across all 1248 players for proper normalisation
    all_features = np.stack([
        _node_features(row) for row in df.itertuples(index=False)
    ])
    global_mean = all_features.mean(axis=0)
    global_std = all_features.std(axis=0)

    lookup: dict[str, dict[str, float]] = {}
    for team, group in df.groupby("team"):
        lookup[str(team)] = team_gnn_embedding(
            group, alpha=alpha, global_mean=global_mean, global_std=global_std
        )
    return lookup


def gnn_match_features(
    home_team: str,
    away_team: str,
    gnn_lookup: dict[str, dict[str, float]],
) -> dict[str, float]:
    default = {col: 0.0 for col in GNN_FEATURE_COLUMNS}
    home_emb = gnn_lookup.get(home_team, default)
    away_emb = gnn_lookup.get(away_team, default)

    row: dict[str, float] = {}
    for col in GNN_FEATURE_COLUMNS:
        row[f"home_{col}"] = home_emb.get(col, 0.0)
        row[f"away_{col}"] = away_emb.get(col, 0.0)
        row[f"gnn_edge_{col}"] = home_emb.get(col, 0.0) - away_emb.get(col, 0.0)

    home_composite = sum(home_emb.get(c, 0.0) for c in GNN_FEATURE_COLUMNS)
    away_composite = sum(away_emb.get(c, 0.0) for c in GNN_FEATURE_COLUMNS)
    row["gnn_chemistry_advantage"] = home_composite - away_composite
    return row
