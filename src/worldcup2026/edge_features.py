from __future__ import annotations

from math import asin, cos, log1p, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    MATCH_CONTEXT_OVERRIDES_FILE,
    PLAYER_READINESS_SIGNALS_FILE,
    TEAM_CONTEXT_FILE,
    TEAM_PLAYER_FEATURES_FILE,
    VENUE_CONTEXT_FILE,
)
from .data import normalized_key
from .decision import choose_recommended_result


DEFAULT_REST_DAYS = 5.0
EDGE_PROBABILITY_LOGIT_SCALE = 0.22
EDGE_GOAL_SHIFT_PER_SIGNAL = 0.05
EDGE_SIGNAL_CLIP = 2.5

EDGE_FEATURE_COLUMNS = [
    "edge_home_travel_km",
    "edge_away_travel_km",
    "edge_home_travel_origin",
    "edge_away_travel_origin",
    "edge_home_body_clock_hours",
    "edge_away_body_clock_hours",
    "edge_home_rest_days",
    "edge_away_rest_days",
    "edge_rest_edge",
    "edge_home_weather_stress",
    "edge_away_weather_stress",
    "edge_travel_fatigue_edge",
    "edge_body_clock_edge",
    "edge_weather_stress_edge",
    "edge_crowd_support_edge",
    "edge_lineup_chemistry_edge",
    "edge_set_piece_mismatch_edge",
    "edge_keeper_shot_style_edge",
    "edge_press_resistance_edge",
    "edge_second_half_pressure_edge",
    "edge_home_readiness_signal",
    "edge_away_readiness_signal",
    "edge_readiness_edge",
    "edge_home_readiness_samples",
    "edge_away_readiness_samples",
    "edge_referee_tempo",
    "edge_referee_card_strictness",
    "edge_referee_card_risk_edge",
    "edge_total_signal",
    "edge_total_signal_pick",
    "edge_total_signal_strength",
    "edge_flags",
]


def load_edge_context(
    team_context_path: Path = TEAM_CONTEXT_FILE,
    venue_context_path: Path = VENUE_CONTEXT_FILE,
    match_context_path: Path = MATCH_CONTEXT_OVERRIDES_FILE,
    team_player_features_path: Path = TEAM_PLAYER_FEATURES_FILE,
    player_readiness_signals_path: Path = PLAYER_READINESS_SIGNALS_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _read_csv_if_exists(team_context_path),
        _read_csv_if_exists(venue_context_path),
        _read_csv_if_exists(match_context_path),
        _read_csv_if_exists(team_player_features_path),
        _read_csv_if_exists(player_readiness_signals_path),
    )


def add_edge_feature_columns(
    matches: pd.DataFrame,
    schedule: pd.DataFrame | None = None,
    team_context: pd.DataFrame | None = None,
    venue_context: pd.DataFrame | None = None,
    match_context: pd.DataFrame | None = None,
    team_player_features: pd.DataFrame | None = None,
    player_readiness_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    enriched = matches.copy()
    for column in EDGE_FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = pd.NA

    if enriched.empty:
        return enriched

    team_lookup, team_defaults = _build_team_context_lookup(team_context)
    venue_lookup = _build_venue_lookup(venue_context)
    override_lookup = _build_override_lookup(match_context)
    player_lookup, player_defaults = _build_player_feature_lookup(team_player_features)
    readiness_lookup = _build_readiness_lookup(player_readiness_signals)
    rest_lookup = _build_rest_lookup(schedule if schedule is not None else matches)

    edge_rows: list[dict[str, object]] = []
    for row in enriched.to_dict("records"):
        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        home_key = normalized_key(home_team)
        away_key = normalized_key(away_team)

        venue = _venue_for_row(row, venue_lookup)
        venue_offset = _parse_offset_hours(row.get("utc_offset"), default=_venue_offset_fallback(venue))
        home_context = _team_context_for_key(team_lookup, team_defaults, home_key, venue, venue_offset)
        away_context = _team_context_for_key(team_lookup, team_defaults, away_key, venue, venue_offset)
        home_features = _feature_for_key(player_lookup, player_defaults, home_key)
        away_features = _feature_for_key(player_lookup, player_defaults, away_key)
        overrides = override_lookup.get(_match_key(row), {})
        home_readiness = _readiness_for_match(readiness_lookup, row, home_key)
        away_readiness = _readiness_for_match(readiness_lookup, row, away_key)

        rest_days = rest_lookup.get(_match_key(row), (DEFAULT_REST_DAYS, DEFAULT_REST_DAYS))
        home_rest_days, away_rest_days = rest_days

        travel = _travel_edges(
            home_context,
            away_context,
            venue,
            venue_offset,
            home_rest_days,
            away_rest_days,
        )
        weather = _weather_edges(home_context, away_context, venue, overrides)
        crowd_edge = _crowd_support_edge(home_context, away_context, venue, overrides)
        lineup_chemistry_edge = _lineup_chemistry_edge(home_features, away_features)
        set_piece_edge = _set_piece_mismatch_edge(row, home_features, away_features)
        keeper_style_edge = _keeper_shot_style_edge(row, home_features, away_features)
        press_edge = _press_resistance_edge(home_features, away_features)
        second_half_edge = _second_half_pressure_edge(
            home_features,
            away_features,
            home_rest_days,
            away_rest_days,
            travel["home_travel_load"],
            travel["away_travel_load"],
            weather["home_weather_stress"],
            weather["away_weather_stress"],
        )
        readiness_edge = home_readiness["signal"] - away_readiness["signal"]
        referee_tempo = _num(overrides, "referee_tempo", 0.5)
        referee_card_strictness = _num(overrides, "referee_card_strictness", 0.5)
        referee_card_risk_edge = _referee_card_risk_edge(
            home_features,
            away_features,
            referee_card_strictness,
        )

        total_signal = (
            travel["travel_fatigue_edge"] * 0.28
            + travel["body_clock_edge"] * 0.08
            + weather["weather_stress_edge"] * 0.24
            + crowd_edge * 0.30
            + lineup_chemistry_edge * 0.50
            + set_piece_edge * 0.28
            + keeper_style_edge * 0.30
            + press_edge * 0.18
            + second_half_edge * 0.25
            + readiness_edge * 0.35
            + referee_card_risk_edge * 0.16
        )

        edge_rows.append(
            {
                "edge_home_travel_km": round(travel["home_travel_km"], 1),
                "edge_away_travel_km": round(travel["away_travel_km"], 1),
                "edge_home_travel_origin": str(home_context.get("travel_origin", "base")),
                "edge_away_travel_origin": str(away_context.get("travel_origin", "base")),
                "edge_home_body_clock_hours": round(travel["home_body_clock_hours"], 2),
                "edge_away_body_clock_hours": round(travel["away_body_clock_hours"], 2),
                "edge_home_rest_days": round(home_rest_days, 2),
                "edge_away_rest_days": round(away_rest_days, 2),
                "edge_rest_edge": round(home_rest_days - away_rest_days, 2),
                "edge_home_weather_stress": round(weather["home_weather_stress"], 3),
                "edge_away_weather_stress": round(weather["away_weather_stress"], 3),
                "edge_travel_fatigue_edge": round(travel["travel_fatigue_edge"], 3),
                "edge_body_clock_edge": round(travel["body_clock_edge"], 3),
                "edge_weather_stress_edge": round(weather["weather_stress_edge"], 3),
                "edge_crowd_support_edge": round(crowd_edge, 3),
                "edge_lineup_chemistry_edge": round(lineup_chemistry_edge, 3),
                "edge_set_piece_mismatch_edge": round(set_piece_edge, 3),
                "edge_keeper_shot_style_edge": round(keeper_style_edge, 3),
                "edge_press_resistance_edge": round(press_edge, 3),
                "edge_second_half_pressure_edge": round(second_half_edge, 3),
                "edge_home_readiness_signal": round(home_readiness["signal"], 3),
                "edge_away_readiness_signal": round(away_readiness["signal"], 3),
                "edge_readiness_edge": round(readiness_edge, 3),
                "edge_home_readiness_samples": int(home_readiness["samples"]),
                "edge_away_readiness_samples": int(away_readiness["samples"]),
                "edge_referee_tempo": round(referee_tempo, 3),
                "edge_referee_card_strictness": round(referee_card_strictness, 3),
                "edge_referee_card_risk_edge": round(referee_card_risk_edge, 3),
                "edge_total_signal": round(total_signal, 3),
                "edge_total_signal_pick": _total_signal_pick(total_signal, home_team, away_team),
                "edge_total_signal_strength": _total_signal_strength(total_signal),
                "edge_flags": _edge_flags(
                    travel["travel_fatigue_edge"],
                    travel["body_clock_edge"],
                    weather["weather_stress_edge"],
                    crowd_edge,
                    lineup_chemistry_edge,
                    set_piece_edge,
                    keeper_style_edge,
                    press_edge,
                    second_half_edge,
                    readiness_edge,
                    referee_card_risk_edge,
                ),
            }
        )

    edge_frame = pd.DataFrame(edge_rows, index=enriched.index)
    for column in EDGE_FEATURE_COLUMNS:
        enriched[column] = edge_frame[column]
    return enriched


def apply_edge_probability_adjustments(
    matches: pd.DataFrame,
    decision_policy: dict[str, float] | None = None,
) -> pd.DataFrame:
    adjusted = matches.copy()
    required_columns = {
        "home_team",
        "away_team",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "expected_home_goals",
        "expected_away_goals",
        "edge_total_signal",
    }
    if adjusted.empty or not required_columns.issubset(adjusted.columns):
        return adjusted

    base_columns = {
        "p_home_win": "base_p_home_win",
        "p_draw": "base_p_draw",
        "p_away_win": "base_p_away_win",
        "expected_home_goals": "base_expected_home_goals",
        "expected_away_goals": "base_expected_away_goals",
    }
    for source, target in base_columns.items():
        if target not in adjusted.columns:
            adjusted[target] = adjusted[source]

    for index, row in adjusted.iterrows():
        row_dict = row.to_dict()
        p_home = _num(row_dict, "p_home_win", 0.0)
        p_draw = _num(row_dict, "p_draw", 0.0)
        p_away = _num(row_dict, "p_away_win", 0.0)
        total = max(p_home + p_draw + p_away, 1e-9)
        p_home /= total
        p_draw /= total
        p_away /= total

        non_draw_total = max(p_home + p_away, 1e-9)
        home_share = _clip(p_home / non_draw_total, 0.001, 0.999)
        edge_signal = _clip(_num(row_dict, "edge_total_signal", 0.0), -EDGE_SIGNAL_CLIP, EDGE_SIGNAL_CLIP)
        logit_shift = edge_signal * EDGE_PROBABILITY_LOGIT_SCALE
        home_logit = np.log(home_share / (1.0 - home_share)) + logit_shift
        adjusted_home_share = 1.0 / (1.0 + np.exp(-home_logit))
        adjusted_p_home = float(non_draw_total * adjusted_home_share)
        adjusted_p_away = float(non_draw_total * (1.0 - adjusted_home_share))

        home_goal_delta = edge_signal * EDGE_GOAL_SHIFT_PER_SIGNAL
        base_home_goals = _num(row_dict, "expected_home_goals", 1.2)
        base_away_goals = _num(row_dict, "expected_away_goals", 1.2)
        adjusted_home_goals = float(max(base_home_goals + home_goal_delta, 0.15))
        adjusted_away_goals = float(max(base_away_goals - home_goal_delta, 0.15))

        decision = choose_recommended_result(
            str(row_dict.get("home_team", "")),
            str(row_dict.get("away_team", "")),
            adjusted_p_home,
            p_draw,
            adjusted_p_away,
            decision_policy,
        )

        adjusted.at[index, "p_home_win"] = adjusted_p_home
        adjusted.at[index, "p_draw"] = float(p_draw)
        adjusted.at[index, "p_away_win"] = adjusted_p_away
        adjusted.at[index, "expected_home_goals"] = adjusted_home_goals
        adjusted.at[index, "expected_away_goals"] = adjusted_away_goals
        adjusted.at[index, "predicted_result"] = decision.recommended_result
        adjusted.at[index, "raw_top_result"] = decision.raw_top_result
        adjusted.at[index, "pick_confidence"] = decision.confidence
        adjusted.at[index, "top_probability"] = decision.top_probability
        adjusted.at[index, "runner_up_probability"] = decision.runner_up_probability
        adjusted.at[index, "probability_margin"] = decision.probability_margin
        adjusted.at[index, "draw_override_applied"] = decision.draw_override_applied
        adjusted.at[index, "edge_probability_logit_shift"] = round(logit_shift, 4)
        adjusted.at[index, "edge_home_win_probability_delta"] = round(adjusted_p_home - p_home, 4)
        adjusted.at[index, "edge_away_win_probability_delta"] = round(adjusted_p_away - p_away, 4)
        adjusted.at[index, "edge_home_expected_goals_delta"] = round(adjusted_home_goals - base_home_goals, 3)
        adjusted.at[index, "edge_away_expected_goals_delta"] = round(adjusted_away_goals - base_away_goals, 3)

    return adjusted


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_team_context_lookup(
    team_context: pd.DataFrame | None,
) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
    if team_context is None or team_context.empty or "team" not in team_context.columns:
        return {}, {}
    frame = team_context.copy()
    frame["_key"] = frame["team"].map(normalized_key)
    numeric_defaults = {
        column: _series_median(frame[column], 0.0)
        for column in frame.columns
        if column not in {"team", "country_code", "_key"}
    }
    lookup = {
        str(row["_key"]): row.drop(labels=["_key"]).to_dict()
        for _, row in frame.iterrows()
        if str(row["_key"])
    }
    return lookup, numeric_defaults


def _build_venue_lookup(venue_context: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if venue_context is None or venue_context.empty:
        return {}
    frame = venue_context.copy()
    keys: list[str] = []
    for row in frame.to_dict("records"):
        key = normalized_key(row.get("venue_key") or row.get("venue") or "")
        if not key:
            key = normalized_key(row.get("venue", ""))
        keys.append(key)
    frame["_key"] = keys
    lookup: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        row_dict = row.drop(labels=["_key"]).to_dict()
        for candidate in {row["_key"], normalized_key(row_dict.get("venue", ""))}:
            if candidate:
                lookup[str(candidate)] = row_dict
    return lookup


def _build_override_lookup(match_context: pd.DataFrame | None) -> dict[tuple[str, str, str], dict[str, object]]:
    if match_context is None or match_context.empty:
        return {}
    required = {"local_date", "home_team", "away_team"}
    if not required.issubset(match_context.columns):
        return {}
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in match_context.fillna("").to_dict("records"):
        key = (
            str(row.get("local_date", "")),
            normalized_key(row.get("home_team", "")),
            normalized_key(row.get("away_team", "")),
        )
        if all(key):
            lookup[key] = row
    return lookup


def _build_player_feature_lookup(
    team_player_features: pd.DataFrame | None,
) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
    if (
        team_player_features is None
        or team_player_features.empty
        or "team" not in team_player_features.columns
    ):
        return {}, {}
    frame = team_player_features.copy()
    frame["_key"] = frame["team"].map(normalized_key)
    defaults = {
        column: _series_median(frame[column], 0.0)
        for column in frame.columns
        if column not in {"team", "_key"} and pd.api.types.is_numeric_dtype(frame[column])
    }
    lookup = {
        str(row["_key"]): row.drop(labels=["_key"]).to_dict()
        for _, row in frame.iterrows()
        if str(row["_key"])
    }
    return lookup, defaults


def _build_readiness_lookup(
    player_readiness_signals: pd.DataFrame | None,
) -> dict[tuple[str, str], dict[str, float]]:
    if (
        player_readiness_signals is None
        or player_readiness_signals.empty
        or "local_date" not in player_readiness_signals.columns
        or "team" not in player_readiness_signals.columns
    ):
        return {}

    frame = player_readiness_signals.copy().fillna("")
    if "include_signal" in frame.columns:
        include = frame["include_signal"].astype(str).str.casefold().str.strip()
        frame = frame[~include.isin({"0", "false", "no", "n", "exclude"})].copy()
    if frame.empty:
        return {}

    frame["_team_key"] = frame["team"].map(normalized_key)
    frame["_signal"] = frame.apply(_readiness_row_signal, axis=1)
    frame["_confidence"] = [
        _clip(_num(row, "confidence", 0.65), 0.0, 1.0) for row in frame.to_dict("records")
    ]
    frame["_weight"] = frame["_confidence"].clip(lower=0.05)

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for (local_date, team_key), group in frame.groupby(["local_date", "_team_key"], dropna=False):
        team_key = str(team_key)
        if not str(local_date) or not team_key:
            continue
        weights = pd.to_numeric(group["_weight"], errors="coerce").fillna(0.0)
        signals = pd.to_numeric(group["_signal"], errors="coerce").fillna(0.0)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            signal = 0.0
        else:
            signal = float((signals * weights).sum() / total_weight)
        lookup[(str(local_date), team_key)] = {
            "signal": float(np.clip(signal, -1.5, 1.5)),
            "samples": float(len(group)),
        }
    return lookup


def _readiness_row_signal(row: pd.Series) -> float:
    row_dict = row.to_dict()
    components: list[tuple[float, float]] = []
    readiness_score = _num(row_dict, "readiness_score", np.nan)
    recovery_score = _num(row_dict, "recovery_score", np.nan)
    sleep_hours = _num(row_dict, "sleep_hours", np.nan)
    sleep_quality = _num(row_dict, "sleep_quality_score", np.nan)
    hrv_delta = _num(row_dict, "hrv_delta_pct", np.nan)
    resting_hr_delta = _num(row_dict, "resting_hr_delta_pct", np.nan)
    strain_7d = _num(row_dict, "strain_7d", np.nan)
    acute_load_delta = _num(row_dict, "acute_load_delta_pct", np.nan)
    last_activity_distance = _num(row_dict, "last_activity_distance_km", np.nan)
    minutes_adjustment = _num(row_dict, "minutes_adjustment", np.nan)

    if not np.isnan(readiness_score):
        components.append((_clip((readiness_score - 50.0) / 35.0, -1.5, 1.5), 0.34))
    if not np.isnan(recovery_score):
        components.append((_clip((recovery_score - 50.0) / 35.0, -1.5, 1.5), 0.30))
    if not np.isnan(sleep_hours):
        components.append((_clip((sleep_hours - 7.0) / 2.0, -1.2, 1.2), 0.18))
    if not np.isnan(sleep_quality):
        components.append((_clip((sleep_quality - 50.0) / 35.0, -1.2, 1.2), 0.14))
    if not np.isnan(hrv_delta):
        components.append((_clip(hrv_delta / 20.0, -1.2, 1.2), 0.16))
    if not np.isnan(resting_hr_delta):
        components.append((_clip(-resting_hr_delta / 15.0, -1.2, 1.2), 0.14))
    if not np.isnan(strain_7d):
        components.append((_clip((12.0 - strain_7d) / 8.0, -0.8, 0.8), 0.08))
    if not np.isnan(acute_load_delta):
        components.append((_clip(-acute_load_delta / 35.0, -1.0, 1.0), 0.10))
    if not np.isnan(last_activity_distance):
        components.append((_clip((6.0 - last_activity_distance) / 12.0, -0.6, 0.6), 0.04))
    if not np.isnan(minutes_adjustment):
        components.append((_clip(minutes_adjustment, -1.0, 1.0), 0.18))

    if not components:
        return 0.0
    total_weight = sum(weight for _, weight in components)
    return float(sum(value * weight for value, weight in components) / total_weight)


def _readiness_for_match(
    readiness_lookup: dict[tuple[str, str], dict[str, float]],
    row: dict[str, object],
    team_key: str,
) -> dict[str, float]:
    exact = readiness_lookup.get((str(row.get("local_date", "")), team_key))
    if exact:
        return exact
    return {"signal": 0.0, "samples": 0.0}


def _build_rest_lookup(schedule: pd.DataFrame | None) -> dict[tuple[str, str, str], tuple[float, float]]:
    if schedule is None or schedule.empty:
        return {}
    required = {"local_date", "home_team", "away_team"}
    if not required.issubset(schedule.columns):
        return {}

    frame = schedule.copy()
    frame["_kickoff"] = frame.apply(_kickoff_timestamp, axis=1)
    frame = frame.dropna(subset=["_kickoff"]).sort_values("_kickoff")

    previous_by_team: dict[str, pd.Timestamp] = {}
    rest_lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in frame.to_dict("records"):
        key = _match_key(row)
        home_key = key[1]
        away_key = key[2]
        kickoff = row["_kickoff"]
        home_rest = _days_since(previous_by_team.get(home_key), kickoff)
        away_rest = _days_since(previous_by_team.get(away_key), kickoff)
        rest_lookup[key] = (home_rest, away_rest)
        rest_lookup[(key[0], away_key, home_key)] = (away_rest, home_rest)
        previous_by_team[home_key] = kickoff
        previous_by_team[away_key] = kickoff
    return rest_lookup


def _venue_for_row(row: dict[str, object], venue_lookup: dict[str, dict[str, object]]) -> dict[str, object]:
    venue_text = str(row.get("venue", ""))
    key = normalized_key(venue_text)
    if key in venue_lookup:
        return venue_lookup[key]
    for lookup_key, venue in venue_lookup.items():
        if key and (key in lookup_key or lookup_key in key):
            return venue
    return {
        "venue": venue_text,
        "country_code": "",
        "latitude": 39.5,
        "longitude": -98.35,
        "altitude_m": 150.0,
        "roof_type": "open",
        "surface": "grass",
        "avg_june_temp_c": 25.0,
        "avg_june_humidity": 0.60,
        "wind_exposure": 0.35,
    }


def _team_context_for_key(
    lookup: dict[str, dict[str, object]],
    defaults: dict[str, float],
    key: str,
    venue: dict[str, object],
    venue_offset: float,
) -> dict[str, object]:
    context = dict(lookup.get(key, {}))
    context.setdefault("country_code", "")
    base_latitude = _num(context, "base_latitude", _num(venue, "latitude", 39.5))
    base_longitude = _num(context, "base_longitude", _num(venue, "longitude", -98.35))
    home_utc_offset = _num(context, "home_utc_offset", venue_offset)
    has_camp_location = _has_value(context.get("camp_latitude")) and _has_value(
        context.get("camp_longitude")
    )
    context["base_latitude"] = base_latitude
    context["base_longitude"] = base_longitude
    context["home_utc_offset"] = home_utc_offset
    context["travel_latitude"] = (
        _num(context, "camp_latitude", base_latitude) if has_camp_location else base_latitude
    )
    context["travel_longitude"] = (
        _num(context, "camp_longitude", base_longitude) if has_camp_location else base_longitude
    )
    context["travel_utc_offset"] = (
        _num(context, "camp_utc_offset", home_utc_offset)
        if has_camp_location
        else home_utc_offset
    )
    context["travel_origin"] = "camp" if has_camp_location else "base"
    for column, default in {
        "heat_acclimation": defaults.get("heat_acclimation", 0.50),
        "altitude_acclimation": defaults.get("altitude_acclimation", 0.20),
        "travel_resilience": defaults.get("travel_resilience", 0.55),
        "crowd_support_base": defaults.get("crowd_support_base", 0.50),
    }.items():
        context[column] = _num(context, column, default)
    return context


def _feature_for_key(
    lookup: dict[str, dict[str, object]],
    defaults: dict[str, float],
    key: str,
) -> dict[str, object]:
    features = dict(defaults)
    features.update(lookup.get(key, {}))
    return features


def _travel_edges(
    home_context: dict[str, object],
    away_context: dict[str, object],
    venue: dict[str, object],
    venue_offset: float,
    home_rest_days: float,
    away_rest_days: float,
) -> dict[str, float]:
    venue_lat = _num(venue, "latitude", 39.5)
    venue_lon = _num(venue, "longitude", -98.35)
    home_travel_km = _haversine_km(
        _num(home_context, "travel_latitude", venue_lat),
        _num(home_context, "travel_longitude", venue_lon),
        venue_lat,
        venue_lon,
    )
    away_travel_km = _haversine_km(
        _num(away_context, "travel_latitude", venue_lat),
        _num(away_context, "travel_longitude", venue_lon),
        venue_lat,
        venue_lon,
    )
    home_body_clock = _clock_shift_hours(_num(home_context, "travel_utc_offset", venue_offset), venue_offset)
    away_body_clock = _clock_shift_hours(_num(away_context, "travel_utc_offset", venue_offset), venue_offset)
    home_load = _travel_load(
        home_travel_km,
        home_body_clock,
        home_rest_days,
        _num(home_context, "travel_resilience", 0.55),
    )
    away_load = _travel_load(
        away_travel_km,
        away_body_clock,
        away_rest_days,
        _num(away_context, "travel_resilience", 0.55),
    )
    return {
        "home_travel_km": home_travel_km,
        "away_travel_km": away_travel_km,
        "home_body_clock_hours": home_body_clock,
        "away_body_clock_hours": away_body_clock,
        "home_travel_load": home_load,
        "away_travel_load": away_load,
        "travel_fatigue_edge": away_load - home_load,
        "body_clock_edge": away_body_clock - home_body_clock,
    }


def _weather_edges(
    home_context: dict[str, object],
    away_context: dict[str, object],
    venue: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, float]:
    temp_c = _num(overrides, "weather_temp_c", _num(venue, "avg_june_temp_c", 25.0))
    humidity = _humidity(_num(overrides, "weather_humidity", _num(venue, "avg_june_humidity", 0.60)))
    wind_kph = _num(overrides, "weather_wind_kph", 0.0)
    roof = str(venue.get("roof_type", "open")).casefold()
    roof_mitigation = 0.0
    if "retract" in roof:
        roof_mitigation = 0.30
    elif "canopy" in roof or "indoor" in roof or "dome" in roof:
        roof_mitigation = 0.20
    turf_heat = 0.08 if str(venue.get("surface", "grass")).casefold() == "turf" and temp_c >= 24 else 0.0
    wind_stress = min(wind_kph / 45.0, 0.4) * _num(venue, "wind_exposure", 0.35)
    altitude_m = _num(venue, "altitude_m", 150.0)

    home_stress = _weather_stress(
        temp_c,
        humidity,
        altitude_m,
        roof_mitigation,
        turf_heat,
        wind_stress,
        _num(home_context, "heat_acclimation", 0.50),
        _num(home_context, "altitude_acclimation", 0.20),
    )
    away_stress = _weather_stress(
        temp_c,
        humidity,
        altitude_m,
        roof_mitigation,
        turf_heat,
        wind_stress,
        _num(away_context, "heat_acclimation", 0.50),
        _num(away_context, "altitude_acclimation", 0.20),
    )
    return {
        "home_weather_stress": home_stress,
        "away_weather_stress": away_stress,
        "weather_stress_edge": away_stress - home_stress,
    }


def _crowd_support_edge(
    home_context: dict[str, object],
    away_context: dict[str, object],
    venue: dict[str, object],
    overrides: dict[str, object],
) -> float:
    venue_country = str(venue.get("country_code", "")).upper()
    home_country = str(home_context.get("country_code", "")).upper()
    away_country = str(away_context.get("country_code", "")).upper()
    home_support = _num(home_context, "crowd_support_base", 0.50)
    away_support = _num(away_context, "crowd_support_base", 0.50)
    if venue_country and home_country == venue_country:
        home_support += 0.35
    if venue_country and away_country == venue_country:
        away_support += 0.35
    home_support += _num(overrides, "home_crowd_boost", 0.0)
    away_support += _num(overrides, "away_crowd_boost", 0.0)
    return float(np.clip(home_support - away_support, -1.25, 1.25))


def _lineup_chemistry_edge(home_features: dict[str, object], away_features: dict[str, object]) -> float:
    return _lineup_chemistry(home_features) - _lineup_chemistry(away_features)


def _lineup_chemistry(features: dict[str, object]) -> float:
    return (
        _num(features, "projected_lineup_club_stats_share", 0.55) * 0.35
        + _clip(_num(features, "top_11_availability_score", 0.55), 0.0, 1.0) * 0.35
        + _clip(_num(features, "projected_lineup_score_mean", 7.0) / 12.0, 0.0, 1.4) * 0.25
        + _clip(_num(features, "projected_lineup_caps_sum", 420.0) / 800.0, 0.0, 1.4) * 0.20
        + _clip(_num(features, "projection_starter_minutes", 800.0) / 900.0, 0.0, 1.1) * 0.12
        + _num(features, "tactics_available", 0.0) * 0.06
    )


def _set_piece_mismatch_edge(
    row: dict[str, object],
    home_features: dict[str, object],
    away_features: dict[str, object],
) -> float:
    home_attack = _set_piece_attack(home_features) + _num(row, "projected_home_corners", 3.5) * 0.035
    away_attack = _set_piece_attack(away_features) + _num(row, "projected_away_corners", 3.5) * 0.035
    home_defense = _set_piece_defense(home_features)
    away_defense = _set_piece_defense(away_features)
    return (home_attack - away_defense) - (away_attack - home_defense)


def _set_piece_attack(features: dict[str, object]) -> float:
    return (
        _num(features, "tactic_set_piece_strength", 0.0) * 0.20
        + _num(features, "top_11_crossing_score", 1.0) * 0.22
        + _num(features, "top_11_physical_score", 1.0) * 0.16
        + _clip(_num(features, "height_cm_mean", 181.0) / 200.0, 0.0, 1.1) * 0.14
        + _num(features, "projection_starter_assist_threat", 7.0) / 40.0
    )


def _set_piece_defense(features: dict[str, object]) -> float:
    return (
        _num(features, "starting_def_defensive_score", 1.2) * 0.24
        + _num(features, "starting_gk_keeper_score", 1.0) * 0.14
        + _num(features, "starting_def_physical_score", 1.0) * 0.16
        + _clip(_num(features, "height_cm_mean", 181.0) / 205.0, 0.0, 1.1) * 0.12
        + _num(features, "projection_starter_defensive_work", 24.0) / 110.0
    )


def _keeper_shot_style_edge(
    row: dict[str, object],
    home_features: dict[str, object],
    away_features: dict[str, object],
) -> float:
    home_attack = _shot_style_attack(home_features) + _num(row, "projected_home_shots_on_target", 3.0) * 0.08
    away_attack = _shot_style_attack(away_features) + _num(row, "projected_away_shots_on_target", 3.0) * 0.08
    home_resistance = _keeper_resistance(home_features)
    away_resistance = _keeper_resistance(away_features)
    return (home_attack - away_resistance) - (away_attack - home_resistance)


def _shot_style_attack(features: dict[str, object]) -> float:
    return (
        _num(features, "starting_fw_shot_volume_score", 1.1) * 0.22
        + _num(features, "top_11_shot_volume_score", 1.0) * 0.18
        + _num(features, "starting_fw_attacking_score", 1.0) * 0.16
        + _num(features, "projection_starter_shot_threat", 12.0) / 65.0
        + _num(features, "projection_starter_goal_threat", 10.0) / 70.0
    )


def _keeper_resistance(features: dict[str, object]) -> float:
    return (
        _num(features, "starting_gk_keeper_score", 1.0) * 0.26
        + _num(features, "top_11_keeper_score", 0.1) * 0.14
        + _num(features, "starting_def_defensive_score", 1.2) * 0.18
        + _num(features, "projection_keeper_coverage", 1.0) / 8.0
    )


def _press_resistance_edge(home_features: dict[str, object], away_features: dict[str, object]) -> float:
    return (
        (_press_score(home_features) - _resistance_score(away_features))
        - (_press_score(away_features) - _resistance_score(home_features))
    )


def _press_score(features: dict[str, object]) -> float:
    return (
        _num(features, "tactic_pressing_intensity", 0.0) * 0.18
        + _num(features, "tactic_defensive_line", 0.0) * 0.08
        + _num(features, "top_11_ball_winning_score", 1.0) * 0.18
        + _num(features, "projection_starter_defensive_work", 25.0) / 120.0
        + _num(features, "projection_defensive_work", 32.0) / 180.0
    )


def _resistance_score(features: dict[str, object]) -> float:
    return (
        _num(features, "tactic_possession_style", 0.0) * 0.16
        + _num(features, "starting_mf_creativity_score", 0.9) * 0.18
        + _num(features, "top_11_creativity_score", 0.8) * 0.14
        + _clip(_num(features, "international_caps_mean", 30.0) / 65.0, 0.0, 1.3) * 0.10
        + _num(features, "projection_starter_assist_threat", 7.0) / 80.0
    )


def _second_half_pressure_edge(
    home_features: dict[str, object],
    away_features: dict[str, object],
    home_rest_days: float,
    away_rest_days: float,
    home_travel_load: float,
    away_travel_load: float,
    home_weather_stress: float,
    away_weather_stress: float,
) -> float:
    home_late = _late_pressure_score(home_features, home_rest_days, home_travel_load, home_weather_stress)
    away_late = _late_pressure_score(away_features, away_rest_days, away_travel_load, away_weather_stress)
    return home_late - away_late


def _late_pressure_score(
    features: dict[str, object],
    rest_days: float,
    travel_load: float,
    weather_stress: float,
) -> float:
    fatigue_drag = max(0.0, 4.0 - rest_days) * 0.05 + travel_load * 0.08 + weather_stress * 0.08
    return (
        _num(features, "bench_depth_score", 2.5) * 0.10
        + _num(features, "projection_bench_goal_threat", 1.5) / 16.0
        + _num(features, "projection_bench_assist_threat", 1.2) / 18.0
        + _num(features, "projection_bench_shot_threat", 2.0) / 30.0
        + _num(features, "tactic_transition_speed", 0.0) * 0.08
        + _num(features, "tactic_tempo", 0.0) * 0.05
        - _num(features, "projection_card_risk", 3.0) / 65.0
        - fatigue_drag
    )


def _referee_card_risk_edge(
    home_features: dict[str, object],
    away_features: dict[str, object],
    strictness: float,
) -> float:
    home_risk = _num(home_features, "projection_card_risk", 3.0) + _num(
        home_features,
        "top_11_discipline_risk",
        0.2,
    ) * 8.0
    away_risk = _num(away_features, "projection_card_risk", 3.0) + _num(
        away_features,
        "top_11_discipline_risk",
        0.2,
    ) * 8.0
    return (away_risk - home_risk) * _clip(strictness, 0.0, 1.0) / 8.0


def _travel_load(distance_km: float, body_clock_hours: float, rest_days: float, travel_resilience: float) -> float:
    return (
        log1p(max(distance_km, 0.0) / 1000.0)
        + max(body_clock_hours - 2.0, 0.0) / 6.0
        + max(4.0 - rest_days, 0.0) * 0.12
        - _clip(travel_resilience, 0.0, 1.0) * 0.35
    )


def _weather_stress(
    temp_c: float,
    humidity: float,
    altitude_m: float,
    roof_mitigation: float,
    turf_heat: float,
    wind_stress: float,
    heat_acclimation: float,
    altitude_acclimation: float,
) -> float:
    heat_stress = max(temp_c - 24.0, 0.0) / 12.0 + max(humidity - 0.55, 0.0) * 0.85
    altitude_stress = max(altitude_m - 500.0, 0.0) / 2200.0
    stress = (
        heat_stress
        + altitude_stress
        + turf_heat
        + wind_stress
        - _clip(heat_acclimation, 0.0, 1.0) * 0.42
        - _clip(altitude_acclimation, 0.0, 1.0) * 0.35
        - roof_mitigation
    )
    return float(max(stress, 0.0))


def _total_signal_pick(total_signal: float, home_team: str, away_team: str) -> str:
    if total_signal >= 0.65:
        return home_team
    if total_signal <= -0.65:
        return away_team
    return "No clear edge"


def _total_signal_strength(total_signal: float) -> str:
    absolute = abs(total_signal)
    if absolute >= 1.20:
        return "strong"
    if absolute >= 0.65:
        return "useful"
    return "thin"


def _edge_flags(
    travel_edge: float,
    body_edge: float,
    weather_edge: float,
    crowd_edge: float,
    chemistry_edge: float,
    set_piece_edge: float,
    keeper_style_edge: float,
    press_edge: float,
    second_half_edge: float,
    readiness_edge: float,
    referee_card_edge: float,
) -> str:
    checks = [
        ("travel", travel_edge, 0.45),
        ("body_clock", body_edge, 2.0),
        ("weather", weather_edge, 0.22),
        ("crowd", crowd_edge, 0.25),
        ("chemistry", chemistry_edge, 0.14),
        ("set_piece", set_piece_edge, 0.22),
        ("keeper_shot_style", keeper_style_edge, 0.22),
        ("press", press_edge, 0.18),
        ("late_pressure", second_half_edge, 0.18),
        ("readiness", readiness_edge, 0.22),
        ("card_risk", referee_card_edge, 0.18),
    ]
    flags: list[str] = []
    for name, value, threshold in checks:
        if value >= threshold:
            flags.append(f"home_{name}")
        elif value <= -threshold:
            flags.append(f"away_{name}")
    return ";".join(flags) if flags else "none"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2.0) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_km * asin(sqrt(a))


def _clock_shift_hours(team_offset: float, venue_offset: float) -> float:
    shift = abs(team_offset - venue_offset) % 24.0
    return min(shift, 24.0 - shift)


def _parse_offset_hours(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return float(default)
    text = str(value).strip()
    if text.startswith("UTC"):
        text = text.replace("UTC", "").replace(":", "")
        if not text:
            return 0.0
        try:
            return float(text[:3])
        except ValueError:
            return float(default)
    try:
        return float(text)
    except ValueError:
        return float(default)


def _venue_offset_fallback(venue: dict[str, object]) -> float:
    country = str(venue.get("country_code", "")).upper()
    city = str(venue.get("city", "")).casefold()
    if country == "MEX":
        return -6.0
    if country == "CAN" and "vancouver" in city:
        return -7.0
    if country == "CAN":
        return -4.0
    if "seattle" in city or "santa clara" in city or "inglewood" in city:
        return -7.0
    if "arlington" in city or "houston" in city or "kansas" in city:
        return -5.0
    return -4.0


def _kickoff_timestamp(row: pd.Series) -> pd.Timestamp | pd.NaT:
    kickoff = pd.to_datetime(row.get("kickoff_utc", ""), utc=True, errors="coerce")
    if not pd.isna(kickoff):
        return kickoff
    local_date = str(row.get("local_date", ""))
    local_time = str(row.get("local_time", "00:00")) or "00:00"
    return pd.to_datetime(f"{local_date} {local_time}", utc=True, errors="coerce")


def _days_since(previous: pd.Timestamp | None, current: pd.Timestamp) -> float:
    if previous is None or pd.isna(previous):
        return DEFAULT_REST_DAYS
    return max((current - previous).total_seconds() / 86400.0, 0.0)


def _match_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("local_date", "")),
        normalized_key(row.get("home_team", "")),
        normalized_key(row.get("away_team", "")),
    )


def _series_median(series: pd.Series, default: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.median())


def _num(row: dict[str, object], column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    if pd.isna(value) or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def _humidity(value: float) -> float:
    if value > 1.5:
        return value / 100.0
    return value


def _clip(value: float, lower: float = 0.0, upper: float | None = None) -> float:
    value = max(float(value), lower)
    if upper is not None:
        value = min(value, upper)
    return value
