from __future__ import annotations

from math import exp, factorial

import numpy as np

from .model import sample_score_for_outcome


GOLDEN_GOAL_RULE_ACTIVE = False
EXTRA_TIME_GOAL_RATE = 0.32
MAX_GOALS = 10


def poisson_pmf(lam: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    lam = max(float(lam), 0.01)
    probs = np.array([exp(-lam) * lam**goals / factorial(goals) for goals in range(max_goals + 1)])
    probs[-1] += max(0.0, 1.0 - probs.sum())
    return probs / probs.sum()


def poisson_outcome_probabilities(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = MAX_GOALS,
) -> tuple[float, float, float]:
    home_probs = poisson_pmf(home_lambda, max_goals=max_goals)
    away_probs = poisson_pmf(away_lambda, max_goals=max_goals)
    matrix = np.outer(home_probs, away_probs)
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    total = max(home_win + draw + away_win, 1e-12)
    return home_win / total, draw / total, away_win / total


def penalty_home_win_probability(predictor, home_team: str, away_team: str) -> float:
    home_rating = float(predictor.team_rating(home_team))
    away_rating = float(predictor.team_rating(away_team))
    rating_edge = (home_rating - away_rating) / 2200.0
    return float(np.clip(0.5 + rating_edge, 0.35, 0.65))


def predict_knockout_resolution(predictor, home_team: str, away_team: str) -> dict[str, object]:
    prediction = predictor.predict_match(home_team, away_team, neutral=True)
    extra_home_goals = prediction.expected_home_goals * EXTRA_TIME_GOAL_RATE
    extra_away_goals = prediction.expected_away_goals * EXTRA_TIME_GOAL_RATE
    et_home_cond, et_draw_cond, et_away_cond = poisson_outcome_probabilities(
        extra_home_goals,
        extra_away_goals,
    )
    p_penalty_shootout = prediction.p_draw * et_draw_cond
    p_home_pen_cond = penalty_home_win_probability(predictor, home_team, away_team)

    p_home_reg = prediction.p_home_win
    p_away_reg = prediction.p_away_win
    p_home_et = prediction.p_draw * et_home_cond
    p_away_et = prediction.p_draw * et_away_cond
    p_home_pen = p_penalty_shootout * p_home_pen_cond
    p_away_pen = p_penalty_shootout * (1.0 - p_home_pen_cond)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "p_home_win_regulation": p_home_reg,
        "p_draw_regulation": prediction.p_draw,
        "p_away_win_regulation": p_away_reg,
        "p_extra_time": prediction.p_draw,
        "p_home_advance_extra_time": p_home_et,
        "p_away_advance_extra_time": p_away_et,
        "p_penalty_shootout": p_penalty_shootout,
        "p_home_advance_penalties": p_home_pen,
        "p_away_advance_penalties": p_away_pen,
        "p_home_advance_total": p_home_reg + p_home_et + p_home_pen,
        "p_away_advance_total": p_away_reg + p_away_et + p_away_pen,
        "p_golden_goal": 0.0,
        "golden_goal_rule_active": GOLDEN_GOAL_RULE_ACTIVE,
        "expected_home_goals_regulation": prediction.expected_home_goals,
        "expected_away_goals_regulation": prediction.expected_away_goals,
        "expected_home_goals_extra_time": extra_home_goals,
        "expected_away_goals_extra_time": extra_away_goals,
        "p_home_win_penalty_shootout_conditional": p_home_pen_cond,
        "p_away_win_penalty_shootout_conditional": 1.0 - p_home_pen_cond,
    }


def _simulate_penalty_shootout(
    rng: np.random.Generator,
    home_team: str,
    away_team: str,
    home_win_probability: float,
) -> tuple[str, int, int]:
    home_conversion = float(np.clip(0.76 + (home_win_probability - 0.5) * 0.16, 0.68, 0.84))
    away_conversion = float(np.clip(0.76 - (home_win_probability - 0.5) * 0.16, 0.68, 0.84))
    home_score = 0
    away_score = 0

    for kick in range(5):
        if rng.random() < home_conversion:
            home_score += 1
        if home_score + (4 - kick) < away_score:
            return away_team, home_score, away_score
        if rng.random() < away_conversion:
            away_score += 1
        if away_score + (4 - kick) < home_score:
            return home_team, home_score, away_score

    while home_score == away_score:
        if rng.random() < home_conversion:
            home_score += 1
        if rng.random() < away_conversion:
            away_score += 1

    winner = home_team if home_score > away_score else away_team
    return winner, home_score, away_score


def simulate_knockout_match(
    predictor,
    rng: np.random.Generator,
    home_team: str,
    away_team: str,
) -> dict[str, object]:
    if hasattr(predictor, "predict_match"):
        prediction = predictor.predict_match(home_team, away_team, neutral=True)
        reg_home_lambda = prediction.expected_home_goals
        reg_away_lambda = prediction.expected_away_goals
        home_penalty_win = penalty_home_win_probability(predictor, home_team, away_team)
        regulation_outcome = str(
            rng.choice(
                ["H", "D", "A"],
                p=[prediction.p_home_win, prediction.p_draw, prediction.p_away_win],
            )
        )
        regulation_home_score, regulation_away_score = sample_score_for_outcome(
            rng,
            reg_home_lambda,
            reg_away_lambda,
            regulation_outcome,
        )
    else:
        home_rating = float(predictor.team_rating(home_team))
        away_rating = float(predictor.team_rating(away_team))
        reg_home_lambda = float(np.clip(1.25 + (home_rating - away_rating) / 650.0, 0.25, 4.5))
        reg_away_lambda = float(np.clip(1.25 + (away_rating - home_rating) / 650.0, 0.25, 4.5))
        home_penalty_win = float(np.clip(0.5 + (home_rating - away_rating) / 2200.0, 0.35, 0.65))
        regulation_home_score = int(rng.poisson(reg_home_lambda))
        regulation_away_score = int(rng.poisson(reg_away_lambda))

    extra_time_home_goals = 0
    extra_time_away_goals = 0
    home_penalties = None
    away_penalties = None

    if regulation_home_score > regulation_away_score:
        winner = home_team
        decided_by = "regulation"
    elif regulation_away_score > regulation_home_score:
        winner = away_team
        decided_by = "regulation"
    else:
        extra_time_home_goals = int(rng.poisson(reg_home_lambda * EXTRA_TIME_GOAL_RATE))
        extra_time_away_goals = int(rng.poisson(reg_away_lambda * EXTRA_TIME_GOAL_RATE))
        home_total = regulation_home_score + extra_time_home_goals
        away_total = regulation_away_score + extra_time_away_goals
        if home_total > away_total:
            winner = home_team
            decided_by = "extra_time"
        elif away_total > home_total:
            winner = away_team
            decided_by = "extra_time"
        else:
            winner, home_penalties, away_penalties = _simulate_penalty_shootout(
                rng,
                home_team,
                away_team,
                home_penalty_win,
            )
            decided_by = "penalties"

    return {
        "winner": winner,
        "decided_by": decided_by,
        "regulation_home_score": regulation_home_score,
        "regulation_away_score": regulation_away_score,
        "extra_time_home_goals": extra_time_home_goals,
        "extra_time_away_goals": extra_time_away_goals,
        "home_score": regulation_home_score + extra_time_home_goals,
        "away_score": regulation_away_score + extra_time_away_goals,
        "home_penalties": home_penalties,
        "away_penalties": away_penalties,
        "golden_goal_rule_active": GOLDEN_GOAL_RULE_ACTIVE,
        "golden_goal_scored": False,
    }
