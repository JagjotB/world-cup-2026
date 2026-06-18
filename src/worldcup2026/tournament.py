from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .knockout import predict_knockout_resolution, simulate_knockout_match
from .model import sample_score_for_outcome


GROUP_MATCH_PAIR_ORDER = [(0, 1), (2, 3), (3, 0), (1, 2), (0, 2), (1, 3)]

R32_SPECS = [
    (73, "A2", "B2"),
    (74, "E1", "T74"),
    (75, "F1", "C2"),
    (76, "C1", "F2"),
    (77, "I1", "T77"),
    (78, "E2", "I2"),
    (79, "A1", "T79"),
    (80, "L1", "T80"),
    (81, "D1", "T81"),
    (82, "G1", "T82"),
    (83, "K2", "L2"),
    (84, "H1", "J2"),
    (85, "B1", "T85"),
    (86, "J1", "H2"),
    (87, "K1", "T87"),
    (88, "D2", "G2"),
]

THIRD_SLOT_CANDIDATES = {
    "T74": set("ABCDF"),
    "T77": set("CDFGH"),
    "T79": set("CEFHI"),
    "T80": set("EHIJK"),
    "T81": set("BEFIJ"),
    "T82": set("AEHIJ"),
    "T85": set("EFGIJ"),
    "T87": set("DEIJL"),
}

ROUND_OF_16_SPECS = [
    (89, 73, 75),
    (90, 74, 77),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]

QUARTERFINAL_SPECS = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
SEMIFINAL_SPECS = [(101, 97, 98), (102, 99, 100)]

STAGE_ORDER = [
    "group_stage",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "champion",
]
STAGE_RANK = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}


@dataclass(frozen=True)
class PlayedMatch:
    stage: str
    match_number: int | None
    group: str | None
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    winner: str | None
    decided_by: str = "regulation"
    regulation_home_score: int | None = None
    regulation_away_score: int | None = None
    extra_time_home_goals: int = 0
    extra_time_away_goals: int = 0
    home_penalties: int | None = None
    away_penalties: int | None = None
    golden_goal_rule_active: bool = False
    golden_goal_scored: bool = False
    used_actual: bool = False


def generate_group_matches(teams: pd.DataFrame) -> list[dict[str, object]]:
    fixtures: list[dict[str, object]] = []
    match_number = 1
    for group, group_df in teams.sort_values(["group", "seed"]).groupby("group"):
        group_teams = group_df["team"].tolist()
        if len(group_teams) != 4:
            raise ValueError(f"Group {group} must have exactly four teams.")
        for home_idx, away_idx in GROUP_MATCH_PAIR_ORDER:
            fixtures.append(
                {
                    "stage": "group",
                    "match_number": match_number,
                    "group": group,
                    "home_team": group_teams[home_idx],
                    "away_team": group_teams[away_idx],
                }
            )
            match_number += 1
    return fixtures


def _actual_result_lookup(actual_results: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    if actual_results is None or actual_results.empty:
        return lookup

    for row in actual_results.itertuples(index=False):
        group = str(getattr(row, "group", "") or "").upper()
        home = str(row.home_team)
        away = str(row.away_team)
        key = ("group", group, "|".join(sorted([home, away])))
        lookup[key] = {
            "home_team": home,
            "away_team": away,
            "home_score": int(row.home_score),
            "away_score": int(row.away_score),
        }
    return lookup


def _result_from_actual(
    fixture: dict[str, object],
    lookup: dict[tuple[str, str, str], dict[str, object]],
) -> tuple[int, int] | None:
    home = str(fixture["home_team"])
    away = str(fixture["away_team"])
    group = str(fixture["group"]).upper()
    key = ("group", group, "|".join(sorted([home, away])))
    actual = lookup.get(key)
    if not actual:
        return None
    if actual["home_team"] == home and actual["away_team"] == away:
        return int(actual["home_score"]), int(actual["away_score"])
    return int(actual["away_score"]), int(actual["home_score"])


def _group_prediction_key(group: object, home_team: object, away_team: object) -> tuple[str, str, str]:
    return (str(group).upper(), str(home_team), str(away_team))


def _build_group_prediction_lookup(
    group_predictions: pd.DataFrame | None,
) -> dict[tuple[str, str, str], dict[str, object]]:
    if group_predictions is None or group_predictions.empty:
        return {}

    required_columns = {
        "group",
        "home_team",
        "away_team",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "expected_home_goals",
        "expected_away_goals",
    }
    if not required_columns.issubset(group_predictions.columns):
        return {}

    return {
        _group_prediction_key(row["group"], row["home_team"], row["away_team"]): row
        for row in group_predictions.to_dict("records")
    }


def _simulate_group_match_from_prediction(
    rng: np.random.Generator,
    prediction_row: dict[str, object],
) -> tuple[int, int]:
    probabilities = np.array(
        [
            float(prediction_row["p_home_win"]),
            float(prediction_row["p_draw"]),
            float(prediction_row["p_away_win"]),
        ],
        dtype=float,
    )
    total = max(float(probabilities.sum()), 1e-12)
    outcome = str(rng.choice(["H", "D", "A"], p=probabilities / total))
    return sample_score_for_outcome(
        rng,
        float(prediction_row["expected_home_goals"]),
        float(prediction_row["expected_away_goals"]),
        outcome,
    )


def _empty_table(teams: pd.DataFrame, predictor) -> dict[str, dict[str, object]]:
    return {
        row.team: {
            "group": row.group,
            "team": row.team,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "points": 0,
            "rating": float(predictor.team_rating(row.team)),
        }
        for row in teams.itertuples(index=False)
    }


def _apply_group_result(
    table: dict[str, dict[str, object]],
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
) -> None:
    home = table[home_team]
    away = table[away_team]

    home["played"] += 1
    away["played"] += 1
    home["goals_for"] += home_score
    home["goals_against"] += away_score
    away["goals_for"] += away_score
    away["goals_against"] += home_score
    home["goal_diff"] = home["goals_for"] - home["goals_against"]
    away["goal_diff"] = away["goals_for"] - away["goals_against"]

    if home_score > away_score:
        home["wins"] += 1
        away["losses"] += 1
        home["points"] += 3
    elif away_score > home_score:
        away["wins"] += 1
        home["losses"] += 1
        away["points"] += 3
    else:
        home["draws"] += 1
        away["draws"] += 1
        home["points"] += 1
        away["points"] += 1


def rank_group_table(table: dict[str, dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(table.values())
    return df.sort_values(
        ["group", "points", "goal_diff", "goals_for", "wins", "rating", "team"],
        ascending=[True, False, False, False, False, False, True],
    ).reset_index(drop=True)


def rank_third_place_teams(third_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        third_rows,
        key=lambda row: (
            -int(row["points"]),
            -int(row["goal_diff"]),
            -int(row["goals_for"]),
            -int(row["wins"]),
            -float(row["rating"]),
            str(row["team"]),
        ),
    )


def _assign_third_slots(qualified_thirds: dict[str, str]) -> dict[str, str]:
    remaining_groups = set(qualified_thirds.keys())
    slots = sorted(
        THIRD_SLOT_CANDIDATES.items(),
        key=lambda item: (len(item[1].intersection(remaining_groups)), item[0]),
    )

    def backtrack(
        idx: int,
        remaining: set[str],
        assignments: dict[str, str],
    ) -> dict[str, str] | None:
        if idx == len(slots):
            return assignments.copy()

        slot, candidates = slots[idx]
        for group in sorted(candidates.intersection(remaining)):
            assignments[slot] = group
            result = backtrack(idx + 1, remaining - {group}, assignments)
            if result is not None:
                return result
            assignments.pop(slot, None)
        return None

    assignment = backtrack(0, remaining_groups, {})
    if assignment is None:
        groups = ", ".join(sorted(remaining_groups))
        raise ValueError(f"Could not assign third-place teams to Round of 32 slots: {groups}")
    return assignment


def _set_stage(stage_reached: dict[str, str], team: str, stage: str) -> None:
    current = stage_reached.get(team, "group_stage")
    if STAGE_RANK[stage] > STAGE_RANK[current]:
        stage_reached[team] = stage


def _play_knockout_match(
    predictor,
    rng: np.random.Generator,
    stage: str,
    match_number: int,
    home_team: str,
    away_team: str,
) -> PlayedMatch:
    simulated = simulate_knockout_match(predictor, rng, home_team, away_team)
    winner = str(simulated["winner"])
    return PlayedMatch(
        stage=stage,
        match_number=match_number,
        group=None,
        home_team=home_team,
        away_team=away_team,
        home_score=int(simulated["home_score"]),
        away_score=int(simulated["away_score"]),
        winner=winner,
        decided_by=str(simulated["decided_by"]),
        regulation_home_score=int(simulated["regulation_home_score"]),
        regulation_away_score=int(simulated["regulation_away_score"]),
        extra_time_home_goals=int(simulated["extra_time_home_goals"]),
        extra_time_away_goals=int(simulated["extra_time_away_goals"]),
        home_penalties=simulated["home_penalties"],
        away_penalties=simulated["away_penalties"],
        golden_goal_rule_active=bool(simulated["golden_goal_rule_active"]),
        golden_goal_scored=bool(simulated["golden_goal_scored"]),
        used_actual=False,
    )


def simulate_tournament_once(
    predictor,
    teams: pd.DataFrame,
    actual_results: pd.DataFrame | None = None,
    group_prediction_lookup: dict[tuple[str, str, str], dict[str, object]] | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    collect_matches: bool = True,
) -> dict[str, object]:
    rng = rng or np.random.default_rng(seed)
    table = _empty_table(teams, predictor)
    played_matches: list[PlayedMatch] = []
    actual_lookup = _actual_result_lookup(actual_results)

    for fixture in generate_group_matches(teams):
        home = str(fixture["home_team"])
        away = str(fixture["away_team"])
        actual_score = _result_from_actual(fixture, actual_lookup)
        used_actual = actual_score is not None
        if actual_score is None:
            prediction_row = (group_prediction_lookup or {}).get(
                _group_prediction_key(fixture["group"], home, away)
            )
            if prediction_row:
                home_score, away_score = _simulate_group_match_from_prediction(rng, prediction_row)
            else:
                simulated = predictor.simulate_match(home, away, rng=rng, neutral=True, allow_draw=True)
                home_score = int(simulated["home_score"])
                away_score = int(simulated["away_score"])
        else:
            home_score, away_score = actual_score

        _apply_group_result(table, home, away, home_score, away_score)
        if collect_matches:
            played_matches.append(
                PlayedMatch(
                    stage="group",
                    match_number=int(fixture["match_number"]),
                    group=str(fixture["group"]),
                    home_team=home,
                    away_team=away,
                    home_score=home_score,
                    away_score=away_score,
                    winner=home if home_score > away_score else away if away_score > home_score else None,
                    decided_by="regulation",
                    regulation_home_score=home_score,
                    regulation_away_score=away_score,
                    used_actual=used_actual,
                )
            )

    standings = rank_group_table(table)
    positions: dict[str, str] = {}
    third_rows: list[dict[str, object]] = []
    stage_reached = {team: "group_stage" for team in teams["team"].tolist()}

    for group, group_table in standings.groupby("group", sort=True):
        ordered = group_table.to_dict("records")
        positions[f"{group}1"] = str(ordered[0]["team"])
        positions[f"{group}2"] = str(ordered[1]["team"])
        positions[f"{group}3"] = str(ordered[2]["team"])
        third_rows.append(ordered[2])
        _set_stage(stage_reached, str(ordered[0]["team"]), "round_of_32")
        _set_stage(stage_reached, str(ordered[1]["team"]), "round_of_32")

    best_thirds = rank_third_place_teams(third_rows)[:8]
    qualified_thirds = {str(row["group"]): str(row["team"]) for row in best_thirds}
    for team in qualified_thirds.values():
        _set_stage(stage_reached, team, "round_of_32")

    third_slot_groups = _assign_third_slots(qualified_thirds)
    for slot, group in third_slot_groups.items():
        positions[slot] = qualified_thirds[group]

    winners: dict[int, str] = {}
    losers: dict[int, str] = {}

    for match_number, home_ref, away_ref in R32_SPECS:
        home = positions[home_ref]
        away = positions[away_ref]
        match = _play_knockout_match(predictor, rng, "round_of_32", match_number, home, away)
        winners[match_number] = match.winner or home
        losers[match_number] = away if winners[match_number] == home else home
        _set_stage(stage_reached, winners[match_number], "round_of_16")
        if collect_matches:
            played_matches.append(match)

    for match_number, left_match, right_match in ROUND_OF_16_SPECS:
        home = winners[left_match]
        away = winners[right_match]
        match = _play_knockout_match(predictor, rng, "round_of_16", match_number, home, away)
        winners[match_number] = match.winner or home
        losers[match_number] = away if winners[match_number] == home else home
        _set_stage(stage_reached, winners[match_number], "quarterfinal")
        if collect_matches:
            played_matches.append(match)

    for match_number, left_match, right_match in QUARTERFINAL_SPECS:
        home = winners[left_match]
        away = winners[right_match]
        match = _play_knockout_match(predictor, rng, "quarterfinal", match_number, home, away)
        winners[match_number] = match.winner or home
        losers[match_number] = away if winners[match_number] == home else home
        _set_stage(stage_reached, winners[match_number], "semifinal")
        if collect_matches:
            played_matches.append(match)

    for match_number, left_match, right_match in SEMIFINAL_SPECS:
        home = winners[left_match]
        away = winners[right_match]
        match = _play_knockout_match(predictor, rng, "semifinal", match_number, home, away)
        winners[match_number] = match.winner or home
        losers[match_number] = away if winners[match_number] == home else home
        _set_stage(stage_reached, winners[match_number], "final")
        if collect_matches:
            played_matches.append(match)

    third_place = _play_knockout_match(
        predictor,
        rng,
        "third_place",
        103,
        losers[101],
        losers[102],
    )
    if collect_matches:
        played_matches.append(third_place)

    final = _play_knockout_match(
        predictor,
        rng,
        "final",
        104,
        winners[101],
        winners[102],
    )
    champion = final.winner or final.home_team
    runner_up = final.away_team if champion == final.home_team else final.home_team
    _set_stage(stage_reached, champion, "champion")
    if collect_matches:
        played_matches.append(final)

    return {
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third_place.winner,
        "matches": pd.DataFrame([match.__dict__ for match in played_matches]),
        "standings": standings,
        "stage_reached": stage_reached,
    }


def simulate_many(
    predictor,
    teams: pd.DataFrame,
    actual_results: pd.DataFrame | None = None,
    group_predictions: pd.DataFrame | None = None,
    simulations: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    teams_list = teams["team"].tolist()
    counts = {team: {stage: 0 for stage in STAGE_ORDER} for team in teams_list}
    last_matches = pd.DataFrame()
    group_prediction_lookup = _build_group_prediction_lookup(group_predictions)

    for sim_idx in range(simulations):
        collect_matches = sim_idx == simulations - 1
        result = simulate_tournament_once(
            predictor,
            teams,
            actual_results=actual_results,
            group_prediction_lookup=group_prediction_lookup,
            rng=rng,
            collect_matches=collect_matches,
        )
        if collect_matches:
            last_matches = result["matches"]
        reached = result["stage_reached"]
        for team in teams_list:
            reached_rank = STAGE_RANK[reached[team]]
            for stage in STAGE_ORDER:
                if STAGE_RANK[stage] <= reached_rank:
                    counts[team][stage] += 1

    rows = []
    group_by_team = dict(zip(teams["team"], teams["group"]))
    for team in teams_list:
        row = {"team": team, "group": group_by_team[team]}
        row.update({stage: counts[team][stage] / simulations for stage in STAGE_ORDER})
        rows.append(row)

    probabilities = pd.DataFrame(rows).sort_values(
        ["champion", "final", "semifinal", "quarterfinal"],
        ascending=False,
    )
    return probabilities.reset_index(drop=True), last_matches


def predict_group_match_probabilities(predictor, teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fixture in generate_group_matches(teams):
        pred = predictor.predict_match(str(fixture["home_team"]), str(fixture["away_team"]))
        decision = predictor.decision_for_prediction(pred)
        rows.append(
            {
                **fixture,
                "predicted_result": decision.recommended_result,
                "raw_top_result": decision.raw_top_result,
                "pick_confidence": decision.confidence,
                "top_probability": decision.top_probability,
                "runner_up_probability": decision.runner_up_probability,
                "probability_margin": decision.probability_margin,
                "draw_override_applied": decision.draw_override_applied,
                "p_home_win": pred.p_home_win,
                "p_draw": pred.p_draw,
                "p_away_win": pred.p_away_win,
                "expected_home_goals": pred.expected_home_goals,
                "expected_away_goals": pred.expected_away_goals,
            }
        )
    return pd.DataFrame(rows)


def predict_sampled_knockout_matches(predictor, sampled_matches: pd.DataFrame) -> pd.DataFrame:
    if sampled_matches.empty:
        return pd.DataFrame()

    rows = []
    knockout_matches = sampled_matches[sampled_matches["stage"].ne("group")].copy()
    for row in knockout_matches.itertuples(index=False):
        prediction = predict_knockout_resolution(predictor, row.home_team, row.away_team)
        rows.append(
            {
                "stage": row.stage,
                "match_number": row.match_number,
                **prediction,
            }
        )
    return pd.DataFrame(rows)
