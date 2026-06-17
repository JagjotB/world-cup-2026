from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PLAYER_MATCH_PROJECTIONS_FILE
from .lineups import OUT_STATUSES, player_lineup_score, project_starting_lineup


BENCH_SIZE = 7
STARTER_BASE_MINUTES = {
    "GK": 90.0,
    "DF": 84.0,
    "MF": 76.0,
    "FW": 72.0,
}
BENCH_BASE_MINUTES = {
    "GK": 0.0,
    "DF": 12.0,
    "MF": 24.0,
    "FW": 28.0,
}


def _num(row, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def _safe_share(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return float(value / total)


def _clip(value: float, lower: float = 0.0, upper: float | None = None) -> float:
    value = max(float(value), lower)
    if upper is not None:
        value = min(value, upper)
    return value


def _available_team_players(players: pd.DataFrame, team: str) -> pd.DataFrame:
    team_players = players[players["team"].eq(team)].copy()
    if team_players.empty:
        return pd.DataFrame()
    if "availability_status" not in team_players.columns:
        team_players["availability_status"] = "available"
    if "availability_multiplier" not in team_players.columns:
        team_players["availability_multiplier"] = 1.0
    if "expected_minutes_share" not in team_players.columns:
        team_players["expected_minutes_share"] = team_players["availability_multiplier"]
    team_players = team_players[
        ~team_players["availability_status"].astype(str).str.casefold().isin(OUT_STATUSES)
    ].copy()
    if team_players.empty:
        return pd.DataFrame()
    team_players["lineup_score"] = team_players.apply(player_lineup_score, axis=1)
    return team_players


def _projection_pool(players: pd.DataFrame, team: str) -> pd.DataFrame:
    team_players = _available_team_players(players, team)
    if team_players.empty:
        return pd.DataFrame()

    starters = project_starting_lineup(players, team)
    starter_names = set(starters["player_name"]) if not starters.empty else set()
    starter_rows = team_players[team_players["player_name"].isin(starter_names)].copy()
    starter_rows["roster_role"] = "starter"

    bench = (
        team_players[~team_players["player_name"].isin(starter_names)]
        .sort_values("lineup_score", ascending=False)
        .head(BENCH_SIZE)
        .copy()
    )
    bench["roster_role"] = "bench"

    pool = pd.concat([starter_rows, bench], ignore_index=True, sort=False)
    if pool.empty:
        return pool
    lineup_positions = dict(zip(starters.get("player_name", []), starters.get("lineup_position", [])))
    selection_reasons = dict(zip(starters.get("player_name", []), starters.get("selection_reason", [])))
    pool["lineup_position"] = [
        lineup_positions.get(row.player_name, row.position)
        for row in pool.itertuples(index=False)
    ]
    pool["selection_reason"] = [
        selection_reasons.get(row.player_name, "bench_depth")
        for row in pool.itertuples(index=False)
    ]
    return pool


def _expected_minutes(row) -> float:
    position = str(row.get("lineup_position") or row.get("position") or "")
    role = str(row.get("roster_role", "bench"))
    base = (
        STARTER_BASE_MINUTES.get(position, 72.0)
        if role == "starter"
        else BENCH_BASE_MINUTES.get(position, 18.0)
    )
    availability = np.clip(_num(row, "availability_multiplier", 1.0), 0.0, 1.0)
    minutes_share = np.clip(_num(row, "expected_minutes_share", availability), 0.0, 1.0)
    if role == "bench" and minutes_share == 1.0:
        minutes_share = 0.75
    return float(np.clip(base * ((availability * 0.6) + (minutes_share * 0.4)), 0.0, 90.0))


def _add_projection_weights(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["projected_minutes"] = out.apply(_expected_minutes, axis=1)
    minutes_factor = out["projected_minutes"] / 90.0
    position = out["lineup_position"].astype(str)

    attacking = pd.to_numeric(out.get("attacking_score", 0.0), errors="coerce").fillna(0.0)
    creativity = pd.to_numeric(out.get("creativity_score", 0.0), errors="coerce").fillna(0.0)
    shot_volume = pd.to_numeric(out.get("shot_volume_score", 0.0), errors="coerce").fillna(0.0)
    defensive = pd.to_numeric(out.get("defensive_score", 0.0), errors="coerce").fillna(0.0)
    ball_winning = pd.to_numeric(out.get("ball_winning_score", 0.0), errors="coerce").fillna(0.0)
    keeper = pd.to_numeric(out.get("keeper_score", 0.0), errors="coerce").fillna(0.0)
    discipline = pd.to_numeric(out.get("discipline_risk", 0.0), errors="coerce").fillna(0.0)

    goal_position_bonus = np.select(
        [position.eq("FW"), position.eq("MF"), position.eq("DF"), position.eq("GK")],
        [1.35, 0.75, 0.20, 0.02],
        default=0.5,
    )
    assist_position_bonus = np.select(
        [position.eq("FW"), position.eq("MF"), position.eq("DF"), position.eq("GK")],
        [0.75, 1.20, 0.55, 0.03],
        default=0.6,
    )
    defensive_position_bonus = np.select(
        [position.eq("DF"), position.eq("MF"), position.eq("FW"), position.eq("GK")],
        [1.30, 1.05, 0.35, 0.05],
        default=0.7,
    )

    out["goal_weight"] = minutes_factor * (0.25 + attacking + shot_volume * 0.45) * goal_position_bonus
    out["assist_weight"] = minutes_factor * (0.20 + creativity + attacking * 0.25) * assist_position_bonus
    out["shot_weight"] = minutes_factor * (0.30 + shot_volume + attacking * 0.45) * goal_position_bonus
    out["defensive_weight"] = minutes_factor * (0.25 + defensive + ball_winning) * defensive_position_bonus
    out["keeper_weight"] = minutes_factor * (0.10 + keeper) * position.eq("GK").astype(float)
    out["card_weight"] = minutes_factor * (0.05 + discipline + ball_winning * 0.08)
    return out


def _project_team_players(
    players: pd.DataFrame,
    team: str,
    opponent: str,
    match_context: dict[str, object],
    team_expected_goals: float,
    opponent_expected_goals: float,
    team_win_probability: float,
    draw_probability: float,
) -> list[dict[str, object]]:
    pool = _projection_pool(players, team)
    if pool.empty:
        return []

    pool = _add_projection_weights(pool)
    goal_weight_total = pool["goal_weight"].sum()
    assist_weight_total = pool["assist_weight"].sum()
    shot_weight_total = pool["shot_weight"].sum()
    defensive_weight_total = pool["defensive_weight"].sum()
    keeper_weight_total = pool["keeper_weight"].sum()
    card_weight_total = pool["card_weight"].sum()

    expected_team_assists = team_expected_goals * 0.72
    expected_team_shots = 7.2 + team_expected_goals * 3.2
    expected_team_sot = max(team_expected_goals * 1.75, expected_team_shots * 0.28)
    expected_defensive_actions = 18.0 + opponent_expected_goals * 5.0
    expected_cards = 1.3 + (1.0 - team_win_probability) * 0.45
    clean_sheet_probability = float(np.exp(-max(opponent_expected_goals, 0.05)))

    rows: list[dict[str, object]] = []
    for player in pool.itertuples(index=False):
        goal_share = _safe_share(player.goal_weight, goal_weight_total)
        assist_share = _safe_share(player.assist_weight, assist_weight_total)
        shot_share = _safe_share(player.shot_weight, shot_weight_total)
        defensive_share = _safe_share(player.defensive_weight, defensive_weight_total)
        keeper_share = _safe_share(player.keeper_weight, keeper_weight_total)
        card_share = _safe_share(player.card_weight, card_weight_total)

        projected_goals = team_expected_goals * goal_share
        projected_assists = expected_team_assists * assist_share
        projected_shots = expected_team_shots * shot_share
        projected_sot = min(projected_shots, expected_team_sot * shot_share * 1.15)
        projected_defensive_actions = expected_defensive_actions * defensive_share
        projected_saves = max(opponent_expected_goals * 2.35, 0.0) * keeper_share
        projected_card_risk = min(expected_cards * card_share, 0.95)
        impact_score = (
            projected_goals * 4.0
            + projected_assists * 3.0
            + projected_sot * 0.6
            + projected_defensive_actions * 0.16
            + projected_saves * 0.25
            + clean_sheet_probability * keeper_share * 2.0
            - projected_card_risk * 0.9
        )

        rows.append(
            {
                **match_context,
                "team": team,
                "opponent": opponent,
                "player_name": player.player_name,
                "position": player.position,
                "lineup_position": player.lineup_position,
                "roster_role": player.roster_role,
                "selection_reason": player.selection_reason,
                "club": getattr(player, "club", ""),
                "availability_status": getattr(player, "availability_status", "available"),
                "projected_minutes": round(_clip(player.projected_minutes, 0.0, 90.0), 1),
                "projected_goals": round(projected_goals, 3),
                "projected_assists": round(projected_assists, 3),
                "projected_shots": round(projected_shots, 2),
                "projected_shots_on_target": round(projected_sot, 2),
                "projected_defensive_actions": round(projected_defensive_actions, 2),
                "projected_saves": round(projected_saves, 2),
                "clean_sheet_probability": round(clean_sheet_probability * keeper_share, 3),
                "card_risk": round(projected_card_risk, 3),
                "impact_score": round(impact_score, 3),
                "team_expected_goals": round(team_expected_goals, 3),
                "opponent_expected_goals": round(opponent_expected_goals, 3),
                "team_win_probability": round(team_win_probability, 4),
                "draw_probability": round(draw_probability, 4),
            }
        )
    return rows


def project_player_match_performances(
    predictor,
    schedule: pd.DataFrame,
    players: pd.DataFrame,
    output_path: Path = PLAYER_MATCH_PROJECTIONS_FILE,
    upcoming_only: bool = True,
    from_date: str | None = None,
) -> pd.DataFrame:
    if players is None or players.empty:
        return pd.DataFrame()

    matches = schedule[schedule["status"].eq("upcoming")].copy() if upcoming_only else schedule.copy()
    if from_date:
        matches = matches[matches["local_date"].ge(from_date)].copy()

    rows: list[dict[str, object]] = []
    for match in matches.itertuples(index=False):
        prediction = predictor.predict_match(match.home_team, match.away_team, neutral=True)
        match_context = {
            "match_number": match.match_number,
            "group": match.group,
            "local_date": match.local_date,
            "local_time": match.local_time,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "venue": match.venue,
        }
        rows.extend(
            _project_team_players(
                players,
                str(match.home_team),
                str(match.away_team),
                match_context,
                team_expected_goals=prediction.expected_home_goals,
                opponent_expected_goals=prediction.expected_away_goals,
                team_win_probability=prediction.p_home_win,
                draw_probability=prediction.p_draw,
            )
        )
        rows.extend(
            _project_team_players(
                players,
                str(match.away_team),
                str(match.home_team),
                match_context,
                team_expected_goals=prediction.expected_away_goals,
                opponent_expected_goals=prediction.expected_home_goals,
                team_win_probability=prediction.p_away_win,
                draw_probability=prediction.p_draw,
            )
        )

    projections = pd.DataFrame(rows)
    if not projections.empty:
        projections = projections.sort_values(
            ["local_date", "local_time", "match_number", "team", "roster_role", "impact_score"],
            ascending=[True, True, True, True, False, False],
        ).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    projections.to_csv(output_path, index=False)
    return projections
