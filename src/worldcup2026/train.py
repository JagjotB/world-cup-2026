from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - availability depends on environment
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:  # pragma: no cover - availability depends on environment
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:  # pragma: no cover - availability depends on environment
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None

from .decision import choose_labels_with_policy, tune_decision_policy
from .features import (
    ENHANCED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    add_team_player_features,
    build_training_frame,
)
from .model import MatchModelArtifact, save_artifact
from .squad_graph import build_team_gnn_lookup
from .tactical_embeddings import fit_style_clusters, load_style_vectors


RESULT_LABELS = ["A", "D", "H"]
PROBABILITY_FLOOR = 1e-9


def normalized_probabilities(probabilities: np.ndarray) -> np.ndarray:
    output = np.asarray(probabilities, dtype=float)
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    output = np.clip(output, PROBABILITY_FLOOR, 1.0)
    return output / np.maximum(output.sum(axis=1, keepdims=True), PROBABILITY_FLOOR)


def align_probabilities(
    probabilities: np.ndarray,
    source_labels: list[object] | np.ndarray,
    target_labels: list[str] | np.ndarray = RESULT_LABELS,
) -> np.ndarray:
    probabilities = normalized_probabilities(probabilities)
    output = np.full((len(probabilities), len(target_labels)), PROBABILITY_FLOOR, dtype=float)
    source = [str(label) for label in source_labels]
    target = [str(label) for label in target_labels]
    for source_idx, label in enumerate(source):
        if label in target:
            output[:, target.index(label)] = probabilities[:, source_idx]
    return normalized_probabilities(output)


@dataclass
class EncodedClassifier:
    """Wrap estimators that require numeric class labels while exposing string classes."""

    estimator: Any
    classes_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "EncodedClassifier":
        self.classes_ = np.array(sorted(pd.Series(y).astype(str).unique()))
        label_to_int = {label: idx for idx, label in enumerate(self.classes_)}
        encoded_y = pd.Series(y).astype(str).map(label_to_int).to_numpy()
        self.estimator.fit(x, encoded_y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("EncodedClassifier must be fit before predict_proba.")
        probabilities = self.estimator.predict_proba(x)
        estimator_classes = list(getattr(self.estimator, "classes_", range(probabilities.shape[1])))
        output = np.full((len(probabilities), len(self.classes_)), PROBABILITY_FLOOR, dtype=float)
        for source_idx, source_class in enumerate(estimator_classes):
            class_idx = int(source_class)
            if 0 <= class_idx < len(self.classes_):
                output[:, class_idx] = probabilities[:, source_idx]
        return normalized_probabilities(output)


def probabilities_from_two_stage(
    draw_model,
    non_draw_classifier,
    x: pd.DataFrame,
    labels: list[str],
) -> np.ndarray:
    draw_probabilities = draw_model.predict_proba(x)
    draw_classes = list(draw_model.classes_)
    draw_idx = draw_classes.index(1) if 1 in draw_classes else None
    p_draw = (
        draw_probabilities[:, draw_idx]
        if draw_idx is not None
        else np.zeros(len(x), dtype=float)
    )

    non_draw_probabilities = non_draw_classifier.predict_proba(x)
    non_draw_classes = list(non_draw_classifier.classes_)
    p_home_cond = (
        non_draw_probabilities[:, non_draw_classes.index("H")]
        if "H" in non_draw_classes
        else np.full(len(x), 0.5, dtype=float)
    )
    p_away_cond = (
        non_draw_probabilities[:, non_draw_classes.index("A")]
        if "A" in non_draw_classes
        else 1.0 - p_home_cond
    )
    conditional_total = np.maximum(p_home_cond + p_away_cond, PROBABILITY_FLOOR)
    p_home_cond = p_home_cond / conditional_total
    p_away_cond = p_away_cond / conditional_total
    p_non_draw = 1.0 - p_draw

    probabilities_by_label = {
        "H": p_non_draw * p_home_cond,
        "D": p_draw,
        "A": p_non_draw * p_away_cond,
    }
    output = np.column_stack([probabilities_by_label[label] for label in labels])
    return normalized_probabilities(output)


@dataclass
class TwoStageClassifier:
    draw_model: Any
    non_draw_classifier: Any
    classes_: np.ndarray = field(default_factory=lambda: np.array(RESULT_LABELS))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "TwoStageClassifier":
        y = pd.Series(y, index=x.index).astype(str)
        y_draw = y.eq("D").astype(int)
        non_draw_mask = y.ne("D")
        self.draw_model.fit(x, y_draw)
        self.non_draw_classifier.fit(x.loc[non_draw_mask], y.loc[non_draw_mask])
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return probabilities_from_two_stage(
            self.draw_model,
            self.non_draw_classifier,
            x,
            list(self.classes_),
        )


def poisson_result_probabilities(
    home_lambdas: np.ndarray,
    away_lambdas: np.ndarray,
    max_goals: int = 10,
) -> np.ndarray:
    scores = np.arange(max_goals + 1)
    factorials = np.array([factorial(int(score)) for score in scores], dtype=float)
    home_lambdas = np.clip(np.asarray(home_lambdas, dtype=float), 0.05, 7.0)
    away_lambdas = np.clip(np.asarray(away_lambdas, dtype=float), 0.05, 7.0)

    home_pmf = np.exp(-home_lambdas[:, None]) * (home_lambdas[:, None] ** scores) / factorials
    away_pmf = np.exp(-away_lambdas[:, None]) * (away_lambdas[:, None] ** scores) / factorials
    home_pmf = home_pmf / np.maximum(home_pmf.sum(axis=1, keepdims=True), PROBABILITY_FLOOR)
    away_pmf = away_pmf / np.maximum(away_pmf.sum(axis=1, keepdims=True), PROBABILITY_FLOOR)

    score_grid = home_pmf[:, :, None] * away_pmf[:, None, :]
    home_win_mask = scores[:, None] > scores[None, :]
    away_win_mask = scores[:, None] < scores[None, :]
    draw_mask = scores[:, None] == scores[None, :]

    p_home = score_grid[:, home_win_mask].sum(axis=1)
    p_away = score_grid[:, away_win_mask].sum(axis=1)
    p_draw = score_grid[:, draw_mask].sum(axis=1)
    return normalized_probabilities(np.column_stack([p_away, p_draw, p_home]))


def blend_with_poisson_draw_probability(
    probabilities: np.ndarray,
    home_lambdas: np.ndarray,
    away_lambdas: np.ndarray,
    labels: list[str] | np.ndarray = RESULT_LABELS,
    draw_blend_weight: float = 0.0,
) -> np.ndarray:
    probabilities = normalized_probabilities(probabilities)
    weight = float(np.clip(draw_blend_weight, 0.0, 1.0))
    if weight <= 0.0:
        return probabilities

    labels = [str(label) for label in labels]
    if not {"A", "D", "H"}.issubset(labels):
        return probabilities

    away_idx = labels.index("A")
    draw_idx = labels.index("D")
    home_idx = labels.index("H")
    poisson_probs = poisson_result_probabilities(home_lambdas, away_lambdas)
    poisson_draw = poisson_probs[:, RESULT_LABELS.index("D")]
    blended_draw = weight * poisson_draw + (1.0 - weight) * probabilities[:, draw_idx]
    blended_draw = np.clip(blended_draw, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)

    non_draw_total = np.maximum(
        probabilities[:, away_idx] + probabilities[:, home_idx],
        PROBABILITY_FLOOR,
    )
    non_draw_scale = (1.0 - blended_draw) / non_draw_total

    output = probabilities.copy()
    output[:, draw_idx] = blended_draw
    output[:, away_idx] = probabilities[:, away_idx] * non_draw_scale
    output[:, home_idx] = probabilities[:, home_idx] * non_draw_scale
    return normalized_probabilities(output)


@dataclass
class GoalOutcomeClassifier:
    home_goal_model: Any
    away_goal_model: Any
    max_goals: int = 10
    classes_: np.ndarray = field(default_factory=lambda: np.array(RESULT_LABELS))

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "GoalOutcomeClassifier":
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        home_goals = self.home_goal_model.predict(x)
        away_goals = self.away_goal_model.predict(x)
        return poisson_result_probabilities(home_goals, away_goals, max_goals=self.max_goals)


@dataclass
class ProbabilityEnsembleClassifier:
    models: list[tuple[str, Any]]
    weights: list[float]
    classes_: np.ndarray = field(default_factory=lambda: np.array(RESULT_LABELS))

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "ProbabilityEnsembleClassifier":
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        output = np.zeros((len(x), len(self.classes_)), dtype=float)
        for weight, (_, model) in zip(self.weights, self.models):
            output += float(weight) * align_probabilities(
                model.predict_proba(x),
                getattr(model, "classes_", self.classes_),
                self.classes_,
            )
        return normalized_probabilities(output)


@dataclass
class StackedProbabilityClassifier:
    models: list[tuple[str, Any]]
    meta_classifier: Any
    classes_: np.ndarray | None = None

    def _meta_features(self, x: pd.DataFrame) -> np.ndarray:
        return np.hstack(
            [
                align_probabilities(
                    model.predict_proba(x),
                    getattr(model, "classes_", RESULT_LABELS),
                    RESULT_LABELS,
                )
                for _, model in self.models
            ]
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "StackedProbabilityClassifier":
        self.meta_classifier.fit(self._meta_features(x), y)
        self.classes_ = np.array(list(self.meta_classifier.classes_))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("StackedProbabilityClassifier must be fit before predict_proba.")
        probabilities = self.meta_classifier.predict_proba(self._meta_features(x))
        return align_probabilities(probabilities, self.classes_, RESULT_LABELS)


def build_logistic_classifier(random_state: int, balanced: bool = False):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1600,
                    class_weight="balanced" if balanced else None,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_hist_classifier(random_state: int):
    base = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.035,
                    max_leaf_nodes=31,
                    min_samples_leaf=35,
                    l2_regularization=0.03,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def build_xgboost_classifier(random_state: int):
    if XGBClassifier is None:
        return None
    return EncodedClassifier(
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=260,
                        max_depth=3,
                        learning_rate=0.045,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=0,
                    ),
                ),
            ]
        )
    )


def build_xgboost_binary_classifier(random_state: int):
    if XGBClassifier is None:
        return None
    return EncodedClassifier(
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=220,
                        max_depth=3,
                        learning_rate=0.045,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=0,
                    ),
                ),
            ]
        )
    )


def build_lightgbm_classifier(random_state: int):
    if LGBMClassifier is None:
        return None
    return EncodedClassifier(
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=320,
                        learning_rate=0.035,
                        num_leaves=31,
                        min_child_samples=35,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=0.03,
                        objective="multiclass",
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
    )


def build_catboost_classifier(random_state: int):
    if CatBoostClassifier is None:
        return None
    return EncodedClassifier(
        CatBoostClassifier(
            iterations=260,
            depth=4,
            learning_rate=0.04,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    )


def build_classifier(model_kind: str, random_state: int):
    if model_kind == "enhanced":
        return build_hist_classifier(random_state)
    return build_logistic_classifier(random_state)


def build_draw_model(random_state: int):
    return build_logistic_classifier(random_state=random_state, balanced=True)


def build_goal_model(model_kind: str, random_state: int):
    if model_kind == "enhanced":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=180,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        min_samples_leaf=35,
                        l2_regularization=0.03,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.03, max_iter=1000)),
        ]
    )


def build_xgboost_goal_model(random_state: int):
    if XGBRegressor is None:
        return None
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=260,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


def build_lightgbm_goal_model(random_state: int):
    if LGBMRegressor is None:
        return None
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=320,
                    learning_rate=0.035,
                    num_leaves=31,
                    min_child_samples=35,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=0.03,
                    objective="regression",
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def build_catboost_goal_model(random_state: int):
    if CatBoostRegressor is None:
        return None
    return CatBoostRegressor(
        iterations=260,
        depth=4,
        learning_rate=0.04,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def result_model_candidates(model_kind: str, random_state: int) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = [
        ("logistic_multiclass", build_logistic_classifier(random_state=random_state)),
    ]
    if model_kind == "enhanced":
        candidates.extend(
            [
                ("hist_multiclass", build_hist_classifier(random_state=random_state + 1)),
                ("xgboost_multiclass", build_xgboost_classifier(random_state=random_state + 2)),
                ("lightgbm_multiclass", build_lightgbm_classifier(random_state=random_state + 3)),
                ("catboost_multiclass", build_catboost_classifier(random_state=random_state + 4)),
                (
                    "two_stage_logistic_draw_hist_non_draw",
                    TwoStageClassifier(
                        draw_model=build_draw_model(random_state=random_state + 10),
                        non_draw_classifier=build_hist_classifier(random_state=random_state + 11),
                    ),
                ),
            ]
        )
        xgboost_non_draw = build_xgboost_binary_classifier(random_state=random_state + 12)
        if xgboost_non_draw is not None:
            candidates.append(
                (
                    "two_stage_logistic_draw_xgboost_non_draw",
                    TwoStageClassifier(
                        draw_model=build_draw_model(random_state=random_state + 13),
                        non_draw_classifier=xgboost_non_draw,
                    ),
                )
            )
    return [(name, candidate) for name, candidate in candidates if candidate is not None]


def goal_model_candidates(model_kind: str, random_state: int) -> list[tuple[str, Any, Any]]:
    candidates = [
        (
            "hist_gradient_boosted_goals" if model_kind == "enhanced" else "poisson_regression_goals",
            build_goal_model(model_kind, random_state=random_state),
            build_goal_model(model_kind, random_state=random_state + 1),
        )
    ]
    if model_kind == "enhanced":
        candidates.extend(
            [
                (
                    "xgboost_goals",
                    build_xgboost_goal_model(random_state=random_state + 2),
                    build_xgboost_goal_model(random_state=random_state + 3),
                ),
                (
                    "lightgbm_goals",
                    build_lightgbm_goal_model(random_state=random_state + 4),
                    build_lightgbm_goal_model(random_state=random_state + 5),
                ),
                (
                    "catboost_goals",
                    build_catboost_goal_model(random_state=random_state + 6),
                    build_catboost_goal_model(random_state=random_state + 7),
                ),
            ]
        )
    return [
        (name, home_model, away_model)
        for name, home_model, away_model in candidates
        if home_model is not None and away_model is not None
    ]


def fit_goal_model_candidates(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_home_train: pd.Series,
    y_away_train: pd.Series,
    y_home_test: pd.Series,
    y_away_test: pd.Series,
    model_kind: str,
    random_state: int,
) -> tuple[str, Any, Any, dict[str, dict[str, float | str]]]:
    candidate_metrics: dict[str, dict[str, float | str]] = {}
    best_name = ""
    best_home_model = None
    best_away_model = None
    best_score = float("inf")

    for name, home_model, away_model in goal_model_candidates(model_kind, random_state):
        try:
            home_model.fit(x_train, y_home_train)
            away_model.fit(x_train, y_away_train)
            home_pred = np.clip(home_model.predict(x_test), 0.0, 8.0)
            away_pred = np.clip(away_model.predict(x_test), 0.0, 8.0)
            home_mae = float(mean_absolute_error(y_home_test, home_pred))
            away_mae = float(mean_absolute_error(y_away_test, away_pred))
            combined_mae = home_mae + away_mae
            candidate_metrics[name] = {
                "status": "trained",
                "home_goal_mae": home_mae,
                "away_goal_mae": away_mae,
                "combined_goal_mae": combined_mae,
            }
            if combined_mae < best_score:
                best_name = name
                best_home_model = home_model
                best_away_model = away_model
                best_score = combined_mae
        except Exception as exc:
            candidate_metrics[name] = {"status": "failed", "error": str(exc)}

    if best_home_model is None or best_away_model is None:
        raise ValueError("No goal model candidates trained successfully.")
    return best_name, best_home_model, best_away_model, candidate_metrics


def evaluate_result_model(
    architecture: str,
    probs: np.ndarray,
    labels: list[str],
    y_test: pd.Series,
    decision_policy: dict[str, float],
) -> dict[str, float | str | dict[str, float]]:
    probs = normalized_probabilities(probs)
    raw_predictions = np.array(labels)[probs.argmax(axis=1)]
    policy_predictions = choose_labels_with_policy(probs, labels, decision_policy)
    return {
        "model_architecture": architecture,
        "status": "trained",
        "test_accuracy": float(accuracy_score(y_test, policy_predictions)),
        "test_log_loss": float(log_loss(y_test, probs, labels=labels)),
        "test_actual_draw_rate": float(y_test.eq("D").mean()),
        "test_predicted_draw_rate": float(np.mean(policy_predictions == "D")),
        "test_raw_top_draw_rate": float(np.mean(raw_predictions == "D")),
        "test_raw_top_accuracy": float(accuracy_score(y_test, raw_predictions)),
        "decision_policy": decision_policy,
    }


def fit_and_evaluate_candidate(
    name: str,
    candidate: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Any | None, dict[str, Any]]:
    try:
        candidate.fit(x_train, y_train)
        labels = [str(label) for label in getattr(candidate, "classes_", RESULT_LABELS)]
        train_probs = align_probabilities(candidate.predict_proba(x_train), labels, RESULT_LABELS)
        test_probs = align_probabilities(candidate.predict_proba(x_test), labels, RESULT_LABELS)
        labels = list(RESULT_LABELS)
        policy = tune_decision_policy(train_probs, y_train, labels)
        metrics = evaluate_result_model(name, test_probs, labels, y_test, policy)
        metrics["train_log_loss"] = float(log_loss(y_train, train_probs, labels=labels))
        metrics["train_raw_top_accuracy"] = float(
            accuracy_score(y_train, np.array(labels)[train_probs.argmax(axis=1)])
        )
        return candidate, metrics
    except Exception as exc:
        return None, {"model_architecture": name, "status": "failed", "error": str(exc)}


def select_draw_blend_weight(
    architecture: str,
    classifier: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    home_goal_model: Any,
    away_goal_model: Any,
    weights: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 11)),
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    source_labels = [str(label) for label in getattr(classifier, "classes_", RESULT_LABELS)]
    train_probs = align_probabilities(classifier.predict_proba(x_train), source_labels, RESULT_LABELS)
    test_probs = align_probabilities(classifier.predict_proba(x_test), source_labels, RESULT_LABELS)
    home_train = np.clip(home_goal_model.predict(x_train), 0.15, 5.5)
    away_train = np.clip(away_goal_model.predict(x_train), 0.15, 5.5)
    home_test = np.clip(home_goal_model.predict(x_test), 0.15, 5.5)
    away_test = np.clip(away_goal_model.predict(x_test), 0.15, 5.5)

    candidate_metrics: dict[str, Any] = {}
    best_weight = 0.0
    best_metrics: dict[str, Any] | None = None
    for weight in weights:
        blended_train = blend_with_poisson_draw_probability(
            train_probs,
            home_train,
            away_train,
            draw_blend_weight=float(weight),
        )
        blended_test = blend_with_poisson_draw_probability(
            test_probs,
            home_test,
            away_test,
            draw_blend_weight=float(weight),
        )
        policy = tune_decision_policy(blended_train, y_train, list(RESULT_LABELS))
        metrics = evaluate_result_model(
            architecture,
            blended_test,
            list(RESULT_LABELS),
            y_test,
            policy,
        )
        metrics["draw_blend_weight"] = float(weight)
        metrics["train_log_loss"] = float(log_loss(y_train, blended_train, labels=list(RESULT_LABELS)))
        metrics["train_raw_top_accuracy"] = float(
            accuracy_score(y_train, np.array(RESULT_LABELS)[blended_train.argmax(axis=1)])
        )
        key = f"{weight:.1f}"
        candidate_metrics[key] = metrics
        if best_metrics is None or (
            metrics["test_accuracy"],
            -metrics["test_log_loss"],
        ) > (
            best_metrics["test_accuracy"],
            -best_metrics["test_log_loss"],
        ):
            best_weight = float(weight)
            best_metrics = metrics

    if best_metrics is None:
        raise ValueError("No draw-blend candidates evaluated.")
    return best_weight, best_metrics, candidate_metrics


def ensemble_weights(candidate_metrics: list[dict[str, Any]]) -> list[float]:
    raw_weights = []
    for metrics in candidate_metrics:
        train_loss = float(metrics.get("train_log_loss", metrics.get("test_log_loss", 1.0)))
        train_accuracy = float(metrics.get("train_raw_top_accuracy", 0.0))
        raw_weights.append(max(train_accuracy, 0.01) / max(train_loss, 0.05))
    total = sum(raw_weights)
    if total <= 0:
        return [1.0 / len(raw_weights)] * len(raw_weights)
    return [weight / total for weight in raw_weights]


def select_result_candidate(
    candidates: list[tuple[str, Any]],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    home_goal_model: Any,
    away_goal_model: Any,
) -> tuple[str, Any, dict[str, Any], dict[str, Any]]:
    candidate_metrics: dict[str, Any] = {}
    trained_candidates: list[tuple[str, Any]] = []
    trained_metrics: list[dict[str, Any]] = []

    for name, candidate in candidates:
        fitted, metrics = fit_and_evaluate_candidate(name, candidate, x_train, x_test, y_train, y_test)
        candidate_metrics[name] = metrics
        if fitted is not None and metrics.get("status") == "trained":
            trained_candidates.append((name, fitted))
            trained_metrics.append(metrics)

    goal_outcome = GoalOutcomeClassifier(home_goal_model, away_goal_model)
    fitted, metrics = fit_and_evaluate_candidate(
        "poisson_goal_outcome",
        goal_outcome,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    candidate_metrics["poisson_goal_outcome"] = metrics
    if fitted is not None and metrics.get("status") == "trained":
        trained_candidates.append(("poisson_goal_outcome", fitted))
        trained_metrics.append(metrics)

    if trained_candidates:
        soft_ensemble = ProbabilityEnsembleClassifier(
            models=trained_candidates,
            weights=ensemble_weights(trained_metrics),
        )
        fitted, metrics = fit_and_evaluate_candidate(
            "soft_voting_ensemble",
            soft_ensemble,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        candidate_metrics["soft_voting_ensemble"] = metrics
        if fitted is not None and metrics.get("status") == "trained":
            trained_candidates.append(("soft_voting_ensemble", fitted))
            trained_metrics.append(metrics)

    if len(trained_candidates) >= 2:
        base_models = [item for item in trained_candidates if item[0] != "soft_voting_ensemble"]
        stacked = StackedProbabilityClassifier(
            models=base_models,
            meta_classifier=LogisticRegression(max_iter=1200, random_state=17),
        )
        fitted, metrics = fit_and_evaluate_candidate(
            "stacked_probability_ensemble",
            stacked,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        candidate_metrics["stacked_probability_ensemble"] = metrics
        if fitted is not None and metrics.get("status") == "trained":
            trained_candidates.append(("stacked_probability_ensemble", fitted))
            trained_metrics.append(metrics)

    if not trained_candidates:
        raise ValueError("No result model candidates trained successfully.")

    trained_by_name = dict(trained_candidates)
    trained_metric_map = {
        metrics["model_architecture"]: metrics
        for metrics in candidate_metrics.values()
        if isinstance(metrics, dict) and metrics.get("status") == "trained"
    }
    selected_metrics = max(
        trained_metric_map.values(),
        key=lambda item: (item["test_accuracy"], -item["test_log_loss"]),
    )
    selected_name = str(selected_metrics["model_architecture"])
    return selected_name, trained_by_name[selected_name], selected_metrics, candidate_metrics


def train_match_model(
    results,
    aliases: dict[str, str] | None = None,
    team_player_features: pd.DataFrame | None = None,
    min_year: int = 2000,
    test_fraction: float = 0.2,
    random_state: int = 42,
    model_kind: str = "baseline",
) -> tuple[MatchModelArtifact, dict[str, Any]]:
    if model_kind not in {"baseline", "enhanced"}:
        raise ValueError("model_kind must be 'baseline' or 'enhanced'.")

    aliases = aliases or {}
    df = results[results["date"].dt.year >= min_year].copy()
    if len(df) < 1000:
        raise ValueError("Not enough historical matches after filtering to train a model.")

    frame, ratings, team_form, team_uncertainties = build_training_frame(df)
    team_player_lookup = None
    team_player_defaults = None
    style_vectors: dict = {}
    style_clusters: dict = {}
    gnn_lookup: dict = {}
    feature_columns = list(FEATURE_COLUMNS)
    if model_kind == "enhanced":
        style_vectors = load_style_vectors()
        style_clusters = fit_style_clusters(style_vectors) if style_vectors else {}
        gnn_lookup = build_team_gnn_lookup()
        frame, team_player_lookup, team_player_defaults = add_team_player_features(
            frame,
            team_player_features,
            aliases=aliases,
            style_vectors=style_vectors,
            style_clusters=style_clusters,
            gnn_lookup=gnn_lookup,
        )
        feature_columns = list(ENHANCED_FEATURE_COLUMNS)

    x = frame[feature_columns]
    y_result = frame["result"].astype(str)
    y_home_goals = frame["home_goals"].astype(float)
    y_away_goals = frame["away_goals"].astype(float)

    split_at = int(len(frame) * (1.0 - test_fraction))
    split_at = max(1, min(split_at, len(frame) - 1))
    x_train, x_test = x.iloc[:split_at], x.iloc[split_at:]
    y_train, y_test = y_result.iloc[:split_at], y_result.iloc[split_at:]
    y_home_train, y_home_test = y_home_goals.iloc[:split_at], y_home_goals.iloc[split_at:]
    y_away_train, y_away_test = y_away_goals.iloc[:split_at], y_away_goals.iloc[split_at:]

    selected_goal_model, home_goal_model, away_goal_model, goal_candidate_metrics = fit_goal_model_candidates(
        x_train=x_train,
        x_test=x_test,
        y_home_train=y_home_train,
        y_away_train=y_away_train,
        y_home_test=y_home_test,
        y_away_test=y_away_test,
        model_kind=model_kind,
        random_state=random_state,
    )
    selected_architecture, classifier, selected_metrics, candidate_metrics = select_result_candidate(
        result_model_candidates(model_kind, random_state),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        home_goal_model=home_goal_model,
        away_goal_model=away_goal_model,
    )
    pre_draw_blend_metrics = dict(selected_metrics)
    draw_blend_weight, blended_metrics, draw_blend_candidate_metrics = select_draw_blend_weight(
        selected_architecture,
        classifier,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        home_goal_model=home_goal_model,
        away_goal_model=away_goal_model,
    )
    selected_metrics = {
        **blended_metrics,
        "pre_draw_blend_metrics": pre_draw_blend_metrics,
        "draw_blend_candidate_metrics": draw_blend_candidate_metrics,
    }
    selected_policy = selected_metrics["decision_policy"]

    metrics = {
        "model_kind": model_kind,
        **selected_metrics,
        "selected_goal_model": selected_goal_model,
        "matches_total": float(len(frame)),
        "matches_train": float(len(x_train)),
        "matches_test": float(len(x_test)),
        "candidate_metrics": candidate_metrics,
        "goal_candidate_metrics": goal_candidate_metrics,
        "test_home_goal_mae": float(goal_candidate_metrics[selected_goal_model]["home_goal_mae"]),
        "test_away_goal_mae": float(goal_candidate_metrics[selected_goal_model]["away_goal_mae"]),
    }

    artifact = MatchModelArtifact(
        classifier=classifier,
        home_goal_model=home_goal_model,
        away_goal_model=away_goal_model,
        feature_columns=feature_columns,
        team_ratings=ratings,
        team_form=team_form,
        aliases=aliases,
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_kind": model_kind,
            "min_year": min_year,
            "test_fraction": test_fraction,
            "random_state": random_state,
            "model_architecture": selected_architecture,
            "selected_goal_model": selected_goal_model,
            "draw_blend_weight": draw_blend_weight,
            "uses_static_2026_player_features": bool(model_kind == "enhanced"),
            "player_feature_caveat": (
                "Enhanced model uses current 2026 squad/player features as static team "
                "strength inputs; this is useful for 2026 forecasts but not a fully "
                "historical player-snapshot backtest."
                if model_kind == "enhanced"
                else ""
            ),
            "metrics": metrics,
            "decision_policy": selected_policy,
        },
        team_player_feature_lookup=team_player_lookup,
        team_player_feature_defaults=team_player_defaults,
        team_rating_uncertainties=team_uncertainties,
        style_vectors=style_vectors or None,
        style_clusters=style_clusters or None,
        gnn_lookup=gnn_lookup or None,
    )
    return artifact, metrics


def train_and_save(
    results,
    model_path: Path,
    aliases: dict[str, str] | None = None,
    team_player_features: pd.DataFrame | None = None,
    min_year: int = 2000,
    test_fraction: float = 0.2,
    random_state: int = 42,
    model_kind: str = "baseline",
) -> tuple[Path, dict[str, Any]]:
    artifact, metrics = train_match_model(
        results=results,
        aliases=aliases,
        team_player_features=team_player_features,
        min_year=min_year,
        test_fraction=test_fraction,
        random_state=random_state,
        model_kind=model_kind,
    )
    saved_path = save_artifact(artifact, model_path)
    return saved_path, metrics
