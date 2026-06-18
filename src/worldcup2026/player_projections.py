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

PLAYER_PROJECTION_TEAM_FEATURE_COLUMNS = [
    "projection_player_count",
    "projection_starter_count",
    "projection_bench_count",
    "projection_minutes_total",
    "projection_starter_minutes",
    "projection_bench_minutes",
    "projection_goal_threat",
    "projection_assist_threat",
    "projection_shot_threat",
    "projection_defensive_work",
    "projection_keeper_coverage",
    "projection_card_risk",
    "projection_starter_goal_threat",
    "projection_starter_assist_threat",
    "projection_starter_shot_threat",
    "projection_starter_defensive_work",
    "projection_bench_goal_threat",
    "projection_bench_assist_threat",
    "projection_bench_shot_threat",
    "projection_bench_defensive_work",
    "projection_top3_impact",
    "projection_top5_impact",
    "projection_attacking_impact",
    "projection_defensive_impact",
    "projection_balance_score",
]

MATCH_TEAM_PROJECTION_COLUMNS = [
    "projected_home_shots",
    "projected_away_shots",
    "projected_home_shots_on_target",
    "projected_away_shots_on_target",
    "projected_sot_edge",
    "projected_more_shots_on_target",
    "more_shots_on_target_useful",
    "more_shots_on_target_strength",
    "projected_home_corners",
    "projected_away_corners",
    "projected_home_first_half_corners",
    "projected_home_second_half_corners",
    "projected_away_first_half_corners",
    "projected_away_second_half_corners",
    "p_home_1plus_corners_each_half",
    "p_away_1plus_corners_each_half",
    "home_1plus_corners_each_half_pick",
    "away_1plus_corners_each_half_pick",
    "home_1plus_corners_each_half_useful_pick",
    "away_1plus_corners_each_half_useful_pick",
    "home_1plus_corners_each_half_strength",
    "away_1plus_corners_each_half_strength",
]

CORNER_EACH_HALF_PICK_THRESHOLD = 0.55
SOT_EDGE_USEFUL_THRESHOLD = 0.50
SOT_EDGE_STRONG_THRESHOLD = 1.25
CORNER_YES_USEFUL_THRESHOLD = 0.62
CORNER_YES_STRONG_THRESHOLD = 0.70
CORNER_NO_USEFUL_MAX_PROBABILITY = 0.52


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


def build_player_projection_pool(players: pd.DataFrame, team: str) -> pd.DataFrame:
    pool = _projection_pool(players, team)
    if pool.empty:
        return pool
    weighted = _add_projection_weights(pool)
    weighted["baseline_projection_impact"] = (
        weighted["goal_weight"] * 4.0
        + weighted["assist_weight"] * 3.0
        + weighted["shot_weight"] * 0.6
        + weighted["defensive_weight"] * 0.16
        + weighted["keeper_weight"] * 0.25
        - weighted["card_weight"] * 0.9
    )
    return weighted


def _pool_sum(pool: pd.DataFrame, column: str) -> float:
    if pool.empty or column not in pool.columns:
        return 0.0
    value = pd.to_numeric(pool[column], errors="coerce").sum()
    return 0.0 if pd.isna(value) else float(value)


def _top_sum(pool: pd.DataFrame, column: str, count: int) -> float:
    if pool.empty or column not in pool.columns:
        return 0.0
    values = pd.to_numeric(pool[column], errors="coerce").fillna(0.0)
    return float(values.sort_values(ascending=False).head(count).sum())


def summarize_team_player_projections(players: pd.DataFrame) -> pd.DataFrame:
    if players is None or players.empty or "team" not in players.columns:
        return pd.DataFrame(columns=["team", *PLAYER_PROJECTION_TEAM_FEATURE_COLUMNS])

    rows: list[dict[str, object]] = []
    for team in sorted(players["team"].dropna().astype(str).unique()):
        pool = build_player_projection_pool(players, team)
        if pool.empty:
            rows.append({"team": team, **{column: 0.0 for column in PLAYER_PROJECTION_TEAM_FEATURE_COLUMNS}})
            continue

        starters = pool[pool["roster_role"].eq("starter")]
        bench = pool[pool["roster_role"].eq("bench")]
        attacking_impact = (
            _pool_sum(pool, "goal_weight") * 4.0
            + _pool_sum(pool, "assist_weight") * 3.0
            + _pool_sum(pool, "shot_weight") * 0.6
        )
        defensive_impact = _pool_sum(pool, "defensive_weight") * 0.16 + _pool_sum(pool, "keeper_weight") * 0.25
        card_drag = _pool_sum(pool, "card_weight") * 0.9
        rows.append(
            {
                "team": team,
                "projection_player_count": float(len(pool)),
                "projection_starter_count": float(len(starters)),
                "projection_bench_count": float(len(bench)),
                "projection_minutes_total": _pool_sum(pool, "projected_minutes"),
                "projection_starter_minutes": _pool_sum(starters, "projected_minutes"),
                "projection_bench_minutes": _pool_sum(bench, "projected_minutes"),
                "projection_goal_threat": _pool_sum(pool, "goal_weight"),
                "projection_assist_threat": _pool_sum(pool, "assist_weight"),
                "projection_shot_threat": _pool_sum(pool, "shot_weight"),
                "projection_defensive_work": _pool_sum(pool, "defensive_weight"),
                "projection_keeper_coverage": _pool_sum(pool, "keeper_weight"),
                "projection_card_risk": _pool_sum(pool, "card_weight"),
                "projection_starter_goal_threat": _pool_sum(starters, "goal_weight"),
                "projection_starter_assist_threat": _pool_sum(starters, "assist_weight"),
                "projection_starter_shot_threat": _pool_sum(starters, "shot_weight"),
                "projection_starter_defensive_work": _pool_sum(starters, "defensive_weight"),
                "projection_bench_goal_threat": _pool_sum(bench, "goal_weight"),
                "projection_bench_assist_threat": _pool_sum(bench, "assist_weight"),
                "projection_bench_shot_threat": _pool_sum(bench, "shot_weight"),
                "projection_bench_defensive_work": _pool_sum(bench, "defensive_weight"),
                "projection_top3_impact": _top_sum(pool, "baseline_projection_impact", 3),
                "projection_top5_impact": _top_sum(pool, "baseline_projection_impact", 5),
                "projection_attacking_impact": attacking_impact,
                "projection_defensive_impact": defensive_impact,
                "projection_balance_score": attacking_impact + defensive_impact - card_drag,
            }
        )

    return pd.DataFrame(rows)


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
    pool = build_player_projection_pool(players, team)
    if pool.empty:
        return []

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


def _project_corner_each_half_probability(
    projected_shots: float,
    projected_shots_on_target: float,
    expected_goals: float,
    win_probability: float,
    draw_probability: float,
) -> tuple[float, float, float, float]:
    projected_shots = _clip(projected_shots, 0.0)
    projected_shots_on_target = _clip(projected_shots_on_target, 0.0)
    expected_goals = _clip(expected_goals, 0.0)
    win_probability = float(np.clip(win_probability, 0.0, 1.0))
    draw_probability = float(np.clip(draw_probability, 0.0, 1.0))

    expected_corners = (
        0.95
        + projected_shots * 0.12
        + projected_shots_on_target * 0.18
        + expected_goals * 0.45
        + max(win_probability - 0.25, 0.0) * 0.95
        + draw_probability * 0.10
    )
    expected_corners = _clip(expected_corners, 0.8, 9.5)

    first_half_corners = expected_corners * 0.43
    second_half_corners = expected_corners * 0.57
    raw_probability = (1.0 - np.exp(-first_half_corners)) * (
        1.0 - np.exp(-second_half_corners)
    )
    probability = float(np.clip(raw_probability * 0.86, 0.0, 0.95))
    return (
        round(expected_corners, 2),
        round(first_half_corners, 2),
        round(second_half_corners, 2),
        round(probability, 4),
    )


def _sot_strength(edge: float) -> str:
    abs_edge = abs(edge)
    if abs_edge >= SOT_EDGE_STRONG_THRESHOLD:
        return "strong"
    if abs_edge >= SOT_EDGE_USEFUL_THRESHOLD:
        return "useful"
    return "no_play"


def _corner_useful_pick(probability: float) -> str:
    if probability >= CORNER_YES_USEFUL_THRESHOLD:
        return "Yes"
    if probability <= CORNER_NO_USEFUL_MAX_PROBABILITY:
        return "No"
    return "No play"


def _corner_strength(probability: float) -> str:
    if probability >= CORNER_YES_STRONG_THRESHOLD:
        return "strong_yes"
    if probability >= CORNER_YES_USEFUL_THRESHOLD:
        return "useful_yes"
    if probability <= CORNER_NO_USEFUL_MAX_PROBABILITY:
        return "useful_no"
    return "no_play"


def add_match_team_projection_totals(
    matches: pd.DataFrame,
    player_projections: pd.DataFrame,
) -> pd.DataFrame:
    enriched = matches.copy()
    for column in MATCH_TEAM_PROJECTION_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = pd.NA

    if matches.empty or player_projections is None or player_projections.empty:
        return enriched

    required_columns = {
        "match_number",
        "group",
        "local_date",
        "local_time",
        "home_team",
        "away_team",
        "team",
        "projected_shots",
        "projected_shots_on_target",
    }
    if not required_columns.issubset(player_projections.columns):
        return enriched

    key_columns = ["match_number", "group", "local_date", "local_time", "home_team", "away_team"]
    projections = player_projections.copy()
    projections["projected_shots_on_target"] = pd.to_numeric(
        projections["projected_shots_on_target"],
        errors="coerce",
    ).fillna(0.0)
    projections["projected_shots"] = pd.to_numeric(
        projections["projected_shots"],
        errors="coerce",
    ).fillna(0.0)
    team_totals = (
        projections.groupby([*key_columns, "team"], dropna=False)
        .agg(
            projected_shots=("projected_shots", "sum"),
            projected_shots_on_target=("projected_shots_on_target", "sum"),
        )
        .reset_index()
    )

    def projected_team_totals(row) -> tuple[float, float, float, float]:
        match_totals = team_totals
        for column in key_columns:
            value = row[column]
            if pd.isna(value):
                match_totals = match_totals[match_totals[column].isna()]
            else:
                match_totals = match_totals[match_totals[column].eq(value)]
        home = match_totals[match_totals["team"].eq(row["home_team"])]
        away = match_totals[match_totals["team"].eq(row["away_team"])]
        return (
            round(float(home["projected_shots"].iloc[0]), 2) if not home.empty else 0.0,
            round(float(away["projected_shots"].iloc[0]), 2) if not away.empty else 0.0,
            round(float(home["projected_shots_on_target"].iloc[0]), 2) if not home.empty else 0.0,
            round(float(away["projected_shots_on_target"].iloc[0]), 2) if not away.empty else 0.0,
        )

    home_shots_values: list[float] = []
    away_shots_values: list[float] = []
    home_values: list[float] = []
    away_values: list[float] = []
    edges: list[float] = []
    picks: list[str] = []
    sot_useful: list[bool] = []
    sot_strengths: list[str] = []
    home_corners: list[float] = []
    away_corners: list[float] = []
    home_first_half_corners: list[float] = []
    home_second_half_corners: list[float] = []
    away_first_half_corners: list[float] = []
    away_second_half_corners: list[float] = []
    home_corner_probabilities: list[float] = []
    away_corner_probabilities: list[float] = []
    home_corner_picks: list[bool] = []
    away_corner_picks: list[bool] = []
    home_corner_useful_picks: list[str] = []
    away_corner_useful_picks: list[str] = []
    home_corner_strengths: list[str] = []
    away_corner_strengths: list[str] = []
    for row in enriched.itertuples(index=False):
        row_dict = row._asdict()
        home_shots, away_shots, home_sot, away_sot = projected_team_totals(row_dict)
        edge = round(home_sot - away_sot, 2)
        home_expected_goals = float(row_dict.get("expected_home_goals") or 0.0)
        away_expected_goals = float(row_dict.get("expected_away_goals") or 0.0)
        p_home_win = float(row_dict.get("p_home_win") or 0.0)
        p_draw = float(row_dict.get("p_draw") or 0.0)
        p_away_win = float(row_dict.get("p_away_win") or 0.0)
        home_corner_total, home_corner_first, home_corner_second, home_corner_probability = (
            _project_corner_each_half_probability(
                home_shots,
                home_sot,
                home_expected_goals,
                p_home_win,
                p_draw,
            )
        )
        away_corner_total, away_corner_first, away_corner_second, away_corner_probability = (
            _project_corner_each_half_probability(
                away_shots,
                away_sot,
                away_expected_goals,
                p_away_win,
                p_draw,
            )
        )
        home_shots_values.append(home_shots)
        away_shots_values.append(away_shots)
        home_values.append(home_sot)
        away_values.append(away_sot)
        edges.append(edge)
        sot_strength = _sot_strength(edge)
        sot_strengths.append(sot_strength)
        sot_useful.append(sot_strength != "no_play")
        home_corners.append(home_corner_total)
        away_corners.append(away_corner_total)
        home_first_half_corners.append(home_corner_first)
        home_second_half_corners.append(home_corner_second)
        away_first_half_corners.append(away_corner_first)
        away_second_half_corners.append(away_corner_second)
        home_corner_probabilities.append(home_corner_probability)
        away_corner_probabilities.append(away_corner_probability)
        home_corner_picks.append(
            home_corner_probability >= CORNER_EACH_HALF_PICK_THRESHOLD
        )
        away_corner_picks.append(
            away_corner_probability >= CORNER_EACH_HALF_PICK_THRESHOLD
        )
        home_corner_useful_picks.append(_corner_useful_pick(home_corner_probability))
        away_corner_useful_picks.append(_corner_useful_pick(away_corner_probability))
        home_corner_strengths.append(_corner_strength(home_corner_probability))
        away_corner_strengths.append(_corner_strength(away_corner_probability))
        if edge > 0:
            picks.append(str(row_dict["home_team"]))
        elif edge < 0:
            picks.append(str(row_dict["away_team"]))
        else:
            picks.append("Even")

    enriched["projected_home_shots"] = home_shots_values
    enriched["projected_away_shots"] = away_shots_values
    enriched["projected_home_shots_on_target"] = home_values
    enriched["projected_away_shots_on_target"] = away_values
    enriched["projected_sot_edge"] = edges
    enriched["projected_more_shots_on_target"] = picks
    enriched["more_shots_on_target_useful"] = sot_useful
    enriched["more_shots_on_target_strength"] = sot_strengths
    enriched["projected_home_corners"] = home_corners
    enriched["projected_away_corners"] = away_corners
    enriched["projected_home_first_half_corners"] = home_first_half_corners
    enriched["projected_home_second_half_corners"] = home_second_half_corners
    enriched["projected_away_first_half_corners"] = away_first_half_corners
    enriched["projected_away_second_half_corners"] = away_second_half_corners
    enriched["p_home_1plus_corners_each_half"] = home_corner_probabilities
    enriched["p_away_1plus_corners_each_half"] = away_corner_probabilities
    enriched["home_1plus_corners_each_half_pick"] = home_corner_picks
    enriched["away_1plus_corners_each_half_pick"] = away_corner_picks
    enriched["home_1plus_corners_each_half_useful_pick"] = home_corner_useful_picks
    enriched["away_1plus_corners_each_half_useful_pick"] = away_corner_useful_picks
    enriched["home_1plus_corners_each_half_strength"] = home_corner_strengths
    enriched["away_1plus_corners_each_half_strength"] = away_corner_strengths
    return enriched
