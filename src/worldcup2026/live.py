from __future__ import annotations

import pandas as pd

from .features import (
    EloSettings,
    _goal_margin_multiplier,
    elo_k_factor,
    expected_score,
    score_points,
)


def played_schedule_results(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule is None or schedule.empty:
        return pd.DataFrame(
            columns=["stage", "group", "home_team", "away_team", "home_score", "away_score"]
        )

    played = schedule[schedule["status"].eq("played")].copy()
    if played.empty:
        return pd.DataFrame(
            columns=["stage", "group", "home_team", "away_team", "home_score", "away_score"]
        )

    return pd.DataFrame(
        {
            "stage": "group",
            "group": played["group"],
            "home_team": played["home_team"],
            "away_team": played["away_team"],
            "home_score": played["home_score"].astype(int),
            "away_score": played["away_score"].astype(int),
        }
    )


def _update_form_average(current: float, new_value: float, window: int) -> float:
    return ((float(current) * (window - 1)) + float(new_value)) / window


def apply_played_results_to_predictor(
    predictor,
    results: pd.DataFrame,
    tournament: str = "FIFA World Cup",
) -> int:
    if results is None or results.empty:
        return 0

    settings = EloSettings()
    applied = 0

    for row in results.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home_model_team = predictor.model_team_name(home_team)
        away_model_team = predictor.model_team_name(away_team)
        home_score = int(row.home_score)
        away_score = int(row.away_score)

        ratings = predictor.artifact.team_ratings
        ratings.setdefault(home_model_team, settings.base_rating)
        ratings.setdefault(away_model_team, settings.base_rating)

        home_elo = float(ratings[home_model_team])
        away_elo = float(ratings[away_model_team])
        home_actual = score_points(home_score, away_score)
        home_expected = expected_score(home_elo, away_elo)
        delta = (
            elo_k_factor(tournament, settings)
            * _goal_margin_multiplier(home_score, away_score)
            * (home_actual - home_expected)
        )

        ratings[home_model_team] = home_elo + delta
        ratings[away_model_team] = away_elo - delta

        team_form = predictor.artifact.team_form
        team_form.setdefault(home_model_team, {"points": 1.0, "goal_diff": 0.0})
        team_form.setdefault(away_model_team, {"points": 1.0, "goal_diff": 0.0})

        home_points = 3.0 if home_score > away_score else 1.0 if home_score == away_score else 0.0
        away_points = 3.0 if away_score > home_score else 1.0 if home_score == away_score else 0.0
        team_form[home_model_team]["points"] = _update_form_average(
            team_form[home_model_team].get("points", 1.0),
            home_points,
            settings.form_window,
        )
        team_form[away_model_team]["points"] = _update_form_average(
            team_form[away_model_team].get("points", 1.0),
            away_points,
            settings.form_window,
        )
        team_form[home_model_team]["goal_diff"] = _update_form_average(
            team_form[home_model_team].get("goal_diff", 0.0),
            float(home_score - away_score),
            settings.form_window,
        )
        team_form[away_model_team]["goal_diff"] = _update_form_average(
            team_form[away_model_team].get("goal_diff", 0.0),
            float(away_score - home_score),
            settings.form_window,
        )
        applied += 1

    predictor._prediction_cache.clear()
    predictor.artifact.metadata["live_results_applied"] = (
        int(predictor.artifact.metadata.get("live_results_applied", 0)) + applied
    )
    return applied
