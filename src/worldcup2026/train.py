from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .decision import choose_labels_with_policy, tune_decision_policy
from .features import (
    ENHANCED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    add_team_player_features,
    build_training_frame,
)
from .model import MatchModelArtifact, save_artifact


RESULT_LABELS = ["A", "D", "H"]


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
    conditional_total = np.maximum(p_home_cond + p_away_cond, 1e-12)
    p_home_cond = p_home_cond / conditional_total
    p_away_cond = p_away_cond / conditional_total
    p_non_draw = 1.0 - p_draw

    probabilities_by_label = {
        "H": p_non_draw * p_home_cond,
        "D": p_draw,
        "A": p_non_draw * p_away_cond,
    }
    output = np.column_stack([probabilities_by_label[label] for label in labels])
    return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-12)


def build_classifier(model_kind: str, random_state: int):
    if model_kind == "enhanced":
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

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_draw_model(random_state: int):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


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
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.03, max_iter=1000)),
        ]
    )


def evaluate_result_model(
    architecture: str,
    probs: np.ndarray,
    labels: list[str],
    y_test: pd.Series,
    decision_policy: dict[str, float],
) -> dict[str, float]:
    raw_predictions = np.array(labels)[probs.argmax(axis=1)]
    policy_predictions = choose_labels_with_policy(probs, labels, decision_policy)
    return {
        "model_architecture": architecture,
        "test_accuracy": float(accuracy_score(y_test, policy_predictions)),
        "test_log_loss": float(log_loss(y_test, probs, labels=labels)),
        "test_actual_draw_rate": float(y_test.eq("D").mean()),
        "test_predicted_draw_rate": float(np.mean(policy_predictions == "D")),
        "test_raw_top_draw_rate": float(np.mean(raw_predictions == "D")),
        "test_raw_top_accuracy": float(accuracy_score(y_test, raw_predictions)),
        "decision_policy": decision_policy,
    }


def train_match_model(
    results,
    aliases: dict[str, str] | None = None,
    team_player_features: pd.DataFrame | None = None,
    min_year: int = 2000,
    test_fraction: float = 0.2,
    random_state: int = 42,
    model_kind: str = "baseline",
) -> tuple[MatchModelArtifact, dict[str, float]]:
    if model_kind not in {"baseline", "enhanced"}:
        raise ValueError("model_kind must be 'baseline' or 'enhanced'.")

    aliases = aliases or {}
    df = results[results["date"].dt.year >= min_year].copy()
    if len(df) < 1000:
        raise ValueError("Not enough historical matches after filtering to train a model.")

    frame, ratings, team_form = build_training_frame(df)
    team_player_lookup = None
    team_player_defaults = None
    feature_columns = list(FEATURE_COLUMNS)
    if model_kind == "enhanced":
        frame, team_player_lookup, team_player_defaults = add_team_player_features(
            frame,
            team_player_features,
            aliases=aliases,
        )
        feature_columns = list(ENHANCED_FEATURE_COLUMNS)

    x = frame[feature_columns]
    y_result = frame["result"]
    y_home_goals = frame["home_goals"].astype(float)
    y_away_goals = frame["away_goals"].astype(float)

    split_at = int(len(frame) * (1.0 - test_fraction))
    split_at = max(1, min(split_at, len(frame) - 1))
    x_train, x_test = x.iloc[:split_at], x.iloc[split_at:]
    y_train, y_test = y_result.iloc[:split_at], y_result.iloc[split_at:]

    classifier = build_classifier(model_kind, random_state=random_state)
    draw_model = build_draw_model(random_state=random_state + 10)
    non_draw_classifier = build_classifier(model_kind, random_state=random_state + 20)
    home_goal_model = build_goal_model(model_kind, random_state=random_state)
    away_goal_model = build_goal_model(model_kind, random_state=random_state + 1)

    y_draw_train = y_train.eq("D").astype(int)
    non_draw_train_mask = y_train.ne("D")
    classifier.fit(x_train, y_train)
    draw_model.fit(x_train, y_draw_train)
    non_draw_classifier.fit(x_train[non_draw_train_mask], y_train[non_draw_train_mask])
    home_goal_model.fit(x_train, y_home_goals.iloc[:split_at])
    away_goal_model.fit(x_train, y_away_goals.iloc[:split_at])

    multiclass_labels = list(classifier.classes_)
    multiclass_train_probs = classifier.predict_proba(x_train)
    multiclass_policy = tune_decision_policy(multiclass_train_probs, y_train, multiclass_labels)
    multiclass_probs = classifier.predict_proba(x_test)
    multiclass_metrics = evaluate_result_model(
        "multiclass_home_draw_away",
        multiclass_probs,
        multiclass_labels,
        y_test,
        multiclass_policy,
    )

    two_stage_labels = list(RESULT_LABELS)
    two_stage_train_probs = probabilities_from_two_stage(
        draw_model,
        non_draw_classifier,
        x_train,
        two_stage_labels,
    )
    two_stage_policy = tune_decision_policy(two_stage_train_probs, y_train, two_stage_labels)
    two_stage_probs = probabilities_from_two_stage(
        draw_model,
        non_draw_classifier,
        x_test,
        two_stage_labels,
    )
    two_stage_metrics = evaluate_result_model(
        "two_stage_draw_then_home_away",
        two_stage_probs,
        two_stage_labels,
        y_test,
        two_stage_policy,
    )

    selected_metrics = max(
        [multiclass_metrics, two_stage_metrics],
        key=lambda item: (item["test_accuracy"], -item["test_log_loss"]),
    )
    selected_architecture = str(selected_metrics["model_architecture"])
    selected_policy = selected_metrics["decision_policy"]

    metrics = {
        "model_kind": model_kind,
        **selected_metrics,
        "matches_total": float(len(frame)),
        "matches_train": float(len(x_train)),
        "matches_test": float(len(x_test)),
        "candidate_metrics": {
            "multiclass_home_draw_away": multiclass_metrics,
            "two_stage_draw_then_home_away": two_stage_metrics,
        },
        "test_home_goal_mae": float(
            mean_absolute_error(y_home_goals.iloc[split_at:], home_goal_model.predict(x_test))
        ),
        "test_away_goal_mae": float(
            mean_absolute_error(y_away_goals.iloc[split_at:], away_goal_model.predict(x_test))
        ),
    }

    artifact = MatchModelArtifact(
        classifier=classifier if selected_architecture == "multiclass_home_draw_away" else None,
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
        draw_model=draw_model if selected_architecture == "two_stage_draw_then_home_away" else None,
        non_draw_classifier=(
            non_draw_classifier if selected_architecture == "two_stage_draw_then_home_away" else None
        ),
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
) -> tuple[Path, dict[str, float]]:
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
