from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import canonicalize_team_name, normalized_key
from .decision import DecisionResult, choose_recommended_result
from .features import (
    EloSettings,
    tournament_importance,
    team_player_feature_row,
)
from .tactical_embeddings import style_matchup_features
from .squad_graph import gnn_match_features


@dataclass
class MatchModelArtifact:
    classifier: Any | None
    home_goal_model: Any
    away_goal_model: Any
    feature_columns: list[str]
    team_ratings: dict[str, float]
    team_form: dict[str, dict[str, float]]
    aliases: dict[str, str]
    metadata: dict[str, Any]
    team_player_feature_lookup: dict[str, dict[str, float]] | None = None
    team_player_feature_defaults: dict[str, float] | None = None
    draw_model: Any | None = None
    non_draw_classifier: Any | None = None
    team_rating_uncertainties: dict[str, float] | None = None
    style_vectors: dict[str, object] | None = None
    style_clusters: dict[str, int] | None = None
    gnn_lookup: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True)
class MatchPrediction:
    home_team: str
    away_team: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    expected_home_goals: float
    expected_away_goals: float


def sample_score_for_outcome(
    rng: np.random.Generator,
    home_lambda: float,
    away_lambda: float,
    outcome: str,
    max_attempts: int = 80,
) -> tuple[int, int]:
    for _ in range(max_attempts):
        home_score = int(rng.poisson(home_lambda))
        away_score = int(rng.poisson(away_lambda))
        if (
            (outcome == "H" and home_score > away_score)
            or (outcome == "D" and home_score == away_score)
            or (outcome == "A" and away_score > home_score)
        ):
            return home_score, away_score

    home_score = int(np.clip(round(home_lambda), 0, 6))
    away_score = int(np.clip(round(away_lambda), 0, 6))
    if outcome == "H" and home_score <= away_score:
        home_score = away_score + 1
    elif outcome == "A" and away_score <= home_score:
        away_score = home_score + 1
    elif outcome == "D":
        draw_score = int(np.clip(round((home_lambda + away_lambda) / 2.0), 0, 5))
        home_score = draw_score
        away_score = draw_score
    return home_score, away_score


def class_probability(model: Any, x: pd.DataFrame, positive_class: object, default: float = 0.0) -> float:
    if model is None or not hasattr(model, "predict_proba"):
        return float(default)
    probabilities = model.predict_proba(x)[0]
    classes = list(getattr(model, "classes_", []))
    if positive_class not in classes:
        return float(default)
    return float(probabilities[classes.index(positive_class)])


def two_stage_result_probabilities(
    draw_model: Any,
    non_draw_classifier: Any,
    x: pd.DataFrame,
) -> dict[str, float]:
    p_draw = float(np.clip(class_probability(draw_model, x, 1, default=0.0), 0.0, 1.0))
    p_home_cond = float(
        np.clip(class_probability(non_draw_classifier, x, "H", default=0.5), 0.0, 1.0)
    )
    p_away_cond = float(
        np.clip(class_probability(non_draw_classifier, x, "A", default=1.0 - p_home_cond), 0.0, 1.0)
    )
    conditional_total = max(p_home_cond + p_away_cond, 1e-12)
    p_home_cond /= conditional_total
    p_away_cond /= conditional_total
    p_non_draw = 1.0 - p_draw
    probs = {
        "H": p_non_draw * p_home_cond,
        "D": p_draw,
        "A": p_non_draw * p_away_cond,
    }
    total = max(sum(probs.values()), 1e-12)
    return {label: value / total for label, value in probs.items()}


class MatchPredictor:
    def __init__(self, artifact: MatchModelArtifact):
        self.artifact = artifact
        self._rating_lookup = {
            normalized_key(team): team for team in artifact.team_ratings.keys()
        }
        self._form_lookup = {
            normalized_key(team): team for team in artifact.team_form.keys()
        }
        uncertainties = artifact.team_rating_uncertainties or {}
        self._uncertainty_lookup = {
            normalized_key(team): unc for team, unc in uncertainties.items()
        }
        self._prediction_cache: dict[tuple[str, str, bool, str], MatchPrediction] = {}

    @classmethod
    def load(cls, path: Path) -> "MatchPredictor":
        artifact = joblib.load(path)
        return cls(artifact)

    def model_team_name(self, team: str) -> str:
        aliased = canonicalize_team_name(team, self.artifact.aliases)
        if aliased in self.artifact.team_ratings:
            return aliased
        key = normalized_key(aliased)
        return self._rating_lookup.get(key, aliased)

    def team_rating(self, team: str) -> float:
        model_name = self.model_team_name(team)
        return float(self.artifact.team_ratings.get(model_name, 1500.0))

    def _team_uncertainty(self, team: str) -> float:
        model_name = self.model_team_name(team)
        key = normalized_key(model_name)
        return float(self._uncertainty_lookup.get(key, 50.0))

    def _team_form(self, team: str) -> dict[str, float]:
        model_name = self.model_team_name(team)
        if model_name in self.artifact.team_form:
            return self.artifact.team_form[model_name]
        key = normalized_key(model_name)
        if key in self._form_lookup:
            return self.artifact.team_form[self._form_lookup[key]]
        return {"points": 1.0, "goal_diff": 0.0}

    def _feature_row(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
    ) -> pd.DataFrame:
        settings = EloSettings()
        home_elo = self.team_rating(home_team)
        away_elo = self.team_rating(away_team)
        home_advantage = 0.0 if neutral else settings.home_advantage
        home_form = self._team_form(home_team)
        away_form = self._team_form(away_team)
        importance = tournament_importance(tournament)

        home_uncertainty = self._team_uncertainty(home_team)
        away_uncertainty = self._team_uncertainty(away_team)

        row = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo + home_advantage - away_elo,
            "home_form_points": home_form["points"],
            "away_form_points": away_form["points"],
            "form_points_diff": home_form["points"] - away_form["points"],
            "home_form_goal_diff": home_form["goal_diff"],
            "away_form_goal_diff": away_form["goal_diff"],
            "form_goal_diff": home_form["goal_diff"] - away_form["goal_diff"],
            "neutral": int(neutral),
            "home_advantage": home_advantage,
            "is_world_cup": int(importance >= 0.95),
            "is_qualifier": int(0.75 <= importance < 0.95),
            "is_friendly": int(importance <= 0.4),
            "tournament_importance": importance,
            "home_elo_uncertainty": home_uncertainty,
            "away_elo_uncertainty": away_uncertainty,
            "elo_uncertainty_diff": home_uncertainty - away_uncertainty,
            "combined_uncertainty": home_uncertainty + away_uncertainty,
        }

        team_player_lookup = getattr(
            self.artifact, "team_player_feature_lookup", None
        )
        team_player_defaults = getattr(
            self.artifact, "team_player_feature_defaults", None
        )
        if team_player_lookup is not None and team_player_defaults is not None:
            row.update(
                team_player_feature_row(
                    home_team,
                    away_team,
                    team_player_lookup,
                    team_player_defaults,
                )
            )

        style_vecs = getattr(self.artifact, "style_vectors", None) or {}
        style_clus = getattr(self.artifact, "style_clusters", None) or {}
        if style_vecs:
            row.update(
                style_matchup_features(
                    home_team, away_team, style_vecs, style_clus
                )
            )

        gnn_lookup = getattr(self.artifact, "gnn_lookup", None) or {}
        if gnn_lookup:
            row.update(gnn_match_features(home_team, away_team, gnn_lookup))

        return pd.DataFrame([row], columns=self.artifact.feature_columns)

    @staticmethod
    def _poisson_draw_probability(home_lambda: float, away_lambda: float, max_goals: int = 10) -> float:
        p = 0.0
        for k in range(max_goals + 1):
            p += (
                exp(-home_lambda) * home_lambda ** k / factorial(k)
                * exp(-away_lambda) * away_lambda ** k / factorial(k)
            )
        return float(np.clip(p, 0.0, 1.0))

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
    ) -> MatchPrediction:
        cache_key = (home_team, away_team, neutral, tournament)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        x = self._feature_row(home_team, away_team, neutral=neutral, tournament=tournament)
        if self.artifact.draw_model is not None and self.artifact.non_draw_classifier is not None:
            probs = two_stage_result_probabilities(
                self.artifact.draw_model,
                self.artifact.non_draw_classifier,
                x,
            )
        else:
            if self.artifact.classifier is None:
                raise ValueError("Model artifact has neither two-stage models nor a multiclass classifier.")
            raw_probs = self.artifact.classifier.predict_proba(x)[0]
            probs = {label: 0.0 for label in ["H", "D", "A"]}
            for label, prob in zip(self.artifact.classifier.classes_, raw_probs):
                probs[str(label)] = float(prob)

        home_goals = float(np.clip(self.artifact.home_goal_model.predict(x)[0], 0.15, 5.5))
        away_goals = float(np.clip(self.artifact.away_goal_model.predict(x)[0], 0.15, 5.5))

        draw_blend_weight = float(
            self.artifact.metadata.get("draw_blend_weight", 0.0)
            if isinstance(getattr(self.artifact, "metadata", None), dict) else 0.0
        )
        if draw_blend_weight > 0.0:
            p_draw_poisson = self._poisson_draw_probability(home_goals, away_goals)
            p_draw_blended = draw_blend_weight * p_draw_poisson + (1.0 - draw_blend_weight) * probs["D"]
            ha_total = max(probs["H"] + probs["A"], 1e-9)
            scale = (1.0 - p_draw_blended) / ha_total
            probs = {
                "H": float(probs["H"] * scale),
                "D": float(p_draw_blended),
                "A": float(probs["A"] * scale),
            }

        prediction = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            p_home_win=probs["H"],
            p_draw=probs["D"],
            p_away_win=probs["A"],
            expected_home_goals=home_goals,
            expected_away_goals=away_goals,
        )
        self._prediction_cache[cache_key] = prediction
        return prediction

    def decision_for_prediction(self, prediction: MatchPrediction) -> DecisionResult:
        policy = getattr(self.artifact, "metadata", {}).get("decision_policy", {})
        return choose_recommended_result(
            prediction.home_team,
            prediction.away_team,
            prediction.p_home_win,
            prediction.p_draw,
            prediction.p_away_win,
            policy,
        )

    def recommended_result(self, prediction: MatchPrediction) -> str:
        return self.decision_for_prediction(prediction).recommended_result

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        rng: np.random.Generator,
        neutral: bool = True,
        allow_draw: bool = True,
    ) -> dict[str, object]:
        prediction = self.predict_match(home_team, away_team, neutral=neutral)
        if allow_draw:
            outcome = str(
                rng.choice(
                    ["H", "D", "A"],
                    p=[prediction.p_home_win, prediction.p_draw, prediction.p_away_win],
                )
            )
        else:
            decisive_home = prediction.p_home_win / max(
                prediction.p_home_win + prediction.p_away_win, 1e-9
            )
            outcome = "H" if rng.random() < decisive_home else "A"

        home_score, away_score = sample_score_for_outcome(
            rng,
            prediction.expected_home_goals,
            prediction.expected_away_goals,
            outcome,
        )
        winner = home_team if outcome == "H" else away_team if outcome == "A" else None

        return {
            "home_score": home_score,
            "away_score": away_score,
            "winner": winner,
            "prediction": prediction,
        }


class RatingOnlyPredictor:
    """Small deterministic fallback used by tests and smoke simulations."""

    def __init__(self, ratings: dict[str, float] | None = None):
        self.ratings = ratings or {}

    def team_rating(self, team: str) -> float:
        return float(self.ratings.get(team, 1500.0))

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        rng: np.random.Generator,
        neutral: bool = True,
        allow_draw: bool = True,
    ) -> dict[str, object]:
        home_rating = self.team_rating(home_team)
        away_rating = self.team_rating(away_team)
        home_lambda = np.clip(1.25 + (home_rating - away_rating) / 650.0, 0.25, 4.5)
        away_lambda = np.clip(1.25 + (away_rating - home_rating) / 650.0, 0.25, 4.5)
        home_score = int(rng.poisson(home_lambda))
        away_score = int(rng.poisson(away_lambda))

        winner = None
        if home_score > away_score:
            winner = home_team
        elif away_score > home_score:
            winner = away_team
        elif not allow_draw:
            p_home = 1.0 / (1.0 + 10.0 ** (-(home_rating - away_rating) / 400.0))
            winner = home_team if rng.random() < p_home else away_team

        return {"home_score": home_score, "away_score": away_score, "winner": winner}


def save_artifact(artifact: MatchModelArtifact, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path
