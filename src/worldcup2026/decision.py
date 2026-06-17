from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_DECISION_POLICY = {
    "draw_min_probability": 0.40,
    "draw_margin_over_team": 0.0,
    "low_confidence_probability": 0.45,
    "medium_confidence_probability": 0.55,
    "low_confidence_margin": 0.06,
    "medium_confidence_margin": 0.12,
}


@dataclass(frozen=True)
class DecisionResult:
    recommended_result: str
    raw_top_result: str
    confidence: str
    top_probability: float
    runner_up_probability: float
    probability_margin: float
    draw_override_applied: bool


def result_probabilities(home_team: str, away_team: str, p_home: float, p_draw: float, p_away: float) -> dict[str, float]:
    return {
        home_team: float(p_home),
        "Draw": float(p_draw),
        away_team: float(p_away),
    }


def classify_confidence(top_probability: float, margin: float, policy: dict[str, float] | None = None) -> str:
    policy = {**DEFAULT_DECISION_POLICY, **(policy or {})}
    if (
        top_probability < policy["low_confidence_probability"]
        or margin < policy["low_confidence_margin"]
    ):
        return "low"
    if (
        top_probability < policy["medium_confidence_probability"]
        or margin < policy["medium_confidence_margin"]
    ):
        return "medium"
    return "high"


def choose_recommended_result(
    home_team: str,
    away_team: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    policy: dict[str, float] | None = None,
) -> DecisionResult:
    policy = {**DEFAULT_DECISION_POLICY, **(policy or {})}
    probabilities = result_probabilities(home_team, away_team, p_home, p_draw, p_away)
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    raw_top_result, top_probability = ranked[0]
    runner_up_probability = ranked[1][1]
    probability_margin = top_probability - runner_up_probability
    recommended_result = raw_top_result
    draw_override_applied = False

    if raw_top_result == "Draw":
        best_team = home_team if p_home >= p_away else away_team
        best_team_probability = max(p_home, p_away)
        draw_edge = p_draw - best_team_probability
        if p_draw < policy["draw_min_probability"] and draw_edge <= policy["draw_margin_over_team"]:
            recommended_result = best_team
            draw_override_applied = True

    return DecisionResult(
        recommended_result=recommended_result,
        raw_top_result=raw_top_result,
        confidence=classify_confidence(top_probability, probability_margin, policy),
        top_probability=top_probability,
        runner_up_probability=runner_up_probability,
        probability_margin=probability_margin,
        draw_override_applied=draw_override_applied,
    )


def choose_labels_with_policy(
    probabilities: np.ndarray,
    labels: list[str],
    policy: dict[str, float],
) -> np.ndarray:
    output = []
    for row in probabilities:
        probs = {label: float(prob) for label, prob in zip(labels, row)}
        decision = choose_recommended_result(
            "H",
            "A",
            probs.get("H", 0.0),
            probs.get("D", 0.0),
            probs.get("A", 0.0),
            policy,
        )
        output.append("H" if decision.recommended_result == "H" else "A" if decision.recommended_result == "A" else "D")
    return np.array(output)


def tune_decision_policy(probabilities: np.ndarray, y_true, labels: list[str]) -> dict[str, float]:
    best_policy = dict(DEFAULT_DECISION_POLICY)
    best_accuracy = -1.0
    best_draw_gap = float("inf")
    actual_draw_rate = float(y_true.eq("D").mean())

    for draw_min in np.linspace(0.36, 0.70, 18):
        for draw_margin in np.linspace(0.00, 0.25, 11):
            policy = {
                **DEFAULT_DECISION_POLICY,
                "draw_min_probability": float(draw_min),
                "draw_margin_over_team": float(draw_margin),
            }
            predicted = choose_labels_with_policy(probabilities, labels, policy)
            accuracy = float(np.mean(predicted == np.array(y_true)))
            draw_gap = abs(float(np.mean(predicted == "D")) - actual_draw_rate)
            if accuracy > best_accuracy or (
                accuracy == best_accuracy and draw_gap < best_draw_gap
            ):
                best_accuracy = accuracy
                best_draw_gap = draw_gap
                best_policy = policy

    best_policy["training_policy_accuracy"] = best_accuracy
    best_policy["training_policy_draw_rate_gap"] = best_draw_gap
    return best_policy
