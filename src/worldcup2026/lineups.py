from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PLAYER_FEATURES_FILE

DEFAULT_FORMATION = {"GK": 1, "DF": 4, "MF": 3, "FW": 3}
OUT_STATUSES = {"out", "injured", "suspended", "unavailable"}
STARTER_STATUSES = {"confirmed", "probable", "projected"}
LINEUP_STYLE_COLUMNS = [
    "availability_score",
    "experience_score",
    "attacking_score",
    "creativity_score",
    "shot_volume_score",
    "crossing_score",
    "ball_winning_score",
    "defensive_score",
    "keeper_score",
    "physical_score",
    "discipline_risk",
]


def load_player_features(path: Path = PLAYER_FEATURES_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(row, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def player_lineup_score(row) -> float:
    position = str(row.get("position", ""))
    status = str(row.get("availability_status", "available")).strip().casefold()
    if status in OUT_STATUSES:
        return 0.0

    score = 0.0
    score += _num(row, "international_caps") * 0.035
    score += _num(row, "international_goals") * 0.12
    score += _num(row, "min") / 900.0
    score += _num(row, "starts") * 0.07
    score += _num(row, "gls") * 0.28
    score += _num(row, "ast") * 0.22
    score += _num(row, "mp") * 0.025
    score += 0.4 if bool(row.get("has_club_stats", False)) else 0.0

    if position == "GK":
        score += _num(row, "height_cm") / 100.0
        score += _num(row, "cs") * 0.2
        score += _num(row, "saves") * 0.015
    elif position == "DF":
        score += _num(row, "height_cm") / 250.0
        score += _num(row, "tklw") * 0.04
        score += _num(row, "int") * 0.04
    elif position == "MF":
        score += _num(row, "tklw") * 0.03
        score += _num(row, "int") * 0.03
        score += _num(row, "crs") * 0.02
    elif position == "FW":
        score += _num(row, "sh") * 0.025
        score += _num(row, "sot") * 0.04

    age = _num(row, "age_on_2026_06_11", 27.0)
    score -= abs(age - 28.0) * 0.025
    availability_multiplier = np.clip(_num(row, "availability_multiplier", 1.0), 0.0, 1.0)
    minutes_share = np.clip(_num(row, "expected_minutes_share", availability_multiplier), 0.0, 1.0)
    return float(score * ((availability_multiplier * 0.7) + (minutes_share * 0.3)))


def project_starting_lineup(
    players: pd.DataFrame,
    team: str,
    formation: dict[str, int] | None = None,
) -> pd.DataFrame:
    formation = formation or DEFAULT_FORMATION
    team_players = players[players["team"].eq(team)].copy()
    if team_players.empty:
        return pd.DataFrame()
    if "availability_status" not in team_players.columns:
        team_players["availability_status"] = "available"
    if "availability_multiplier" not in team_players.columns:
        team_players["availability_multiplier"] = 1.0
    if "expected_minutes_share" not in team_players.columns:
        team_players["expected_minutes_share"] = team_players["availability_multiplier"]
    if "lineup_status" not in team_players.columns:
        team_players["lineup_status"] = ""

    team_players["lineup_score"] = team_players.apply(player_lineup_score, axis=1)
    selected_indexes: list[int] = []
    slot = 1
    rows: list[dict[str, object]] = []

    if "manual_lineup_slot" in team_players.columns:
        manual = team_players[
            team_players["manual_lineup_slot"].notna()
            & team_players["lineup_status"].astype(str).str.casefold().isin(STARTER_STATUSES)
            & ~team_players["availability_status"].astype(str).str.casefold().isin(OUT_STATUSES)
        ].copy()
        if not manual.empty:
            manual["manual_lineup_slot"] = pd.to_numeric(
                manual["manual_lineup_slot"],
                errors="coerce",
            )
            manual = manual.sort_values(["manual_lineup_slot", "lineup_score"], ascending=[True, False])
            for idx, player in manual.head(11).iterrows():
                selected_indexes.append(idx)
                lineup_position = str(player.get("manual_lineup_position") or player.get("position"))
                rows.append(_lineup_row(player, slot, lineup_position, str(player.get("lineup_status"))))
                slot += 1

    for position, count in formation.items():
        current_count = sum(1 for row in rows if row["lineup_position"] == position)
        remaining_count = max(count - current_count, 0)
        if remaining_count == 0:
            continue
        candidates = (
            team_players[
                team_players["position"].eq(position)
                & ~team_players.index.isin(selected_indexes)
                & ~team_players["availability_status"].astype(str).str.casefold().isin(OUT_STATUSES)
            ]
            .sort_values("lineup_score", ascending=False)
            .head(remaining_count)
        )
        for idx, player in candidates.iterrows():
            selected_indexes.append(idx)
            rows.append(_lineup_row(player, slot, position, "position_fit"))
            slot += 1

    if len(rows) < 11:
        remaining = (
            team_players[
                ~team_players.index.isin(selected_indexes)
                & ~team_players["availability_status"].astype(str).str.casefold().isin(OUT_STATUSES)
            ]
            .sort_values("lineup_score", ascending=False)
            .head(11 - len(rows))
        )
        for _, player in remaining.iterrows():
            rows.append(_lineup_row(player, slot, str(player["position"]), "best_remaining"))
            slot += 1

    return pd.DataFrame(rows)


def _lineup_row(player, slot: int, lineup_position: str, selection_reason: str) -> dict[str, object]:
    row = {
        "team": player["team"],
        "lineup_slot": slot,
        "lineup_position": lineup_position,
        "player_name": player["player_name"],
        "squad_number": player.get("squad_number"),
        "position": player["position"],
        "club": player.get("club"),
        "international_caps": player.get("international_caps"),
        "international_goals": player.get("international_goals"),
        "club_minutes": player.get("min", np.nan),
        "club_goals": player.get("gls", np.nan),
        "club_assists": player.get("ast", np.nan),
        "lineup_score": player["lineup_score"],
        "selection_reason": selection_reason,
        "availability_status": player.get("availability_status"),
        "availability_multiplier": player.get("availability_multiplier"),
        "expected_minutes_share": player.get("expected_minutes_share"),
        "lineup_status": player.get("lineup_status"),
        "manual_formation": player.get("manual_formation"),
    }
    for column in LINEUP_STYLE_COLUMNS:
        row[column] = player.get(column, np.nan)
    return row


def project_all_starting_lineups(players: pd.DataFrame) -> pd.DataFrame:
    frames = [
        project_starting_lineup(players, team)
        for team in sorted(players["team"].dropna().unique())
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def lineup_string(lineups: pd.DataFrame, team: str) -> str:
    rows = lineups[lineups["team"].eq(team)].sort_values("lineup_slot")
    if rows.empty:
        return ""
    return " | ".join(
        f"{row.lineup_position}:{row.player_name}"
        for row in rows.itertuples(index=False)
    )
