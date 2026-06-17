from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import log

import pandas as pd


@dataclass(frozen=True)
class EloSettings:
    base_rating: float = 1500.0
    home_advantage: float = 65.0
    k_default: float = 28.0
    k_friendly: float = 16.0
    k_world_cup: float = 42.0
    k_qualifier: float = 32.0
    form_window: int = 5


FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_points",
    "away_form_points",
    "form_points_diff",
    "home_form_goal_diff",
    "away_form_goal_diff",
    "form_goal_diff",
    "neutral",
    "home_advantage",
    "is_world_cup",
    "is_qualifier",
    "is_friendly",
    "tournament_importance",
]

TEAM_PLAYER_FEATURE_SOURCE_COLUMNS = [
    "club_stats_matched_players",
    "age_on_2026_06_11_mean",
    "height_cm_mean",
    "international_caps_sum",
    "international_caps_mean",
    "international_goals_sum",
    "international_goals_mean",
    "min_sum",
    "starts_sum",
    "gls_sum",
    "ast_sum",
    "sh_sum",
    "sot_sum",
    "tklw_sum",
    "int_sum",
    "saves_sum",
    "cs_sum",
    "players_df",
    "players_fw",
    "players_gk",
    "players_mf",
    "projected_experience_top_11_caps",
    "availability_multiplier_sum",
    "availability_multiplier_mean",
    "expected_minutes_share_sum",
    "expected_minutes_share_mean",
    "is_unavailable_sum",
    "is_questionable_sum",
    "is_suspended_sum",
    "is_injured_sum",
    "projected_lineup_score_sum",
    "projected_lineup_score_mean",
    "projected_lineup_club_stats_share",
    "projected_lineup_minutes_sum",
    "projected_lineup_caps_sum",
    "top_11_availability_score",
    "top_11_experience_score",
    "top_11_attacking_score",
    "top_11_creativity_score",
    "top_11_shot_volume_score",
    "top_11_crossing_score",
    "top_11_ball_winning_score",
    "top_11_defensive_score",
    "top_11_keeper_score",
    "top_11_physical_score",
    "top_11_discipline_risk",
    "starting_gk_keeper_score",
    "starting_def_defensive_score",
    "starting_def_physical_score",
    "starting_mf_creativity_score",
    "starting_mf_ball_winning_score",
    "starting_fw_attacking_score",
    "starting_fw_shot_volume_score",
    "bench_depth_score",
    "tactics_available",
    "formation_back_line",
    "formation_midfield_line",
    "formation_forward_line",
    "formation_defensive_density",
    "formation_midfield_density",
    "formation_attacking_density",
    "tactic_pressing_intensity",
    "tactic_defensive_line",
    "tactic_tempo",
    "tactic_directness",
    "tactic_possession_style",
    "tactic_width",
    "tactic_set_piece_strength",
    "tactic_transition_speed",
]

PLAYER_MATCHUP_FEATURE_COLUMNS = [
    "home_attack_vs_away_defense",
    "away_attack_vs_home_defense",
    "attack_matchup_diff",
    "home_creativity_vs_away_ball_winning",
    "away_creativity_vs_home_ball_winning",
    "creativity_matchup_diff",
    "home_shot_volume_vs_away_keeper",
    "away_shot_volume_vs_home_keeper",
    "finishing_keeper_matchup_diff",
    "home_physical_vs_away_physical",
    "away_physical_vs_home_physical",
    "physical_matchup_diff",
    "keeper_edge",
    "bench_depth_edge",
    "discipline_risk_edge",
    "availability_edge",
    "unavailable_player_edge",
    "pressing_edge",
    "defensive_line_edge",
    "tempo_edge",
    "directness_edge",
    "possession_edge",
    "width_edge",
    "set_piece_edge",
    "transition_edge",
    "formation_attack_density_edge",
    "formation_defensive_density_edge",
]

TEAM_PLAYER_FEATURE_COLUMNS = [
    "home_has_player_features",
    "away_has_player_features",
    *[f"home_player_{column}" for column in TEAM_PLAYER_FEATURE_SOURCE_COLUMNS],
    *[f"away_player_{column}" for column in TEAM_PLAYER_FEATURE_SOURCE_COLUMNS],
    *[f"player_diff_{column}" for column in TEAM_PLAYER_FEATURE_SOURCE_COLUMNS],
    *PLAYER_MATCHUP_FEATURE_COLUMNS,
]

ENHANCED_FEATURE_COLUMNS = FEATURE_COLUMNS + TEAM_PLAYER_FEATURE_COLUMNS


def tournament_importance(tournament: str) -> float:
    text = str(tournament).casefold()
    if "fifa world cup" in text and "qualification" not in text:
        return 1.0
    if "qualification" in text or "qualifier" in text:
        return 0.82
    if "uefa euro" in text or "copa america" in text or "africa cup" in text:
        return 0.88
    if "friendly" in text:
        return 0.35
    return 0.65


def elo_k_factor(tournament: str, settings: EloSettings) -> float:
    importance = tournament_importance(tournament)
    if importance >= 0.95:
        return settings.k_world_cup
    if importance >= 0.8:
        return settings.k_qualifier
    if importance <= 0.4:
        return settings.k_friendly
    return settings.k_default


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / 400.0))


def score_points(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def result_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score == away_score:
        return "D"
    return "A"


def _average(values: deque[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def _goal_margin_multiplier(home_score: int, away_score: int) -> float:
    goal_diff = abs(home_score - away_score)
    if goal_diff <= 1:
        return 1.0
    return 1.0 + log(goal_diff)


def build_training_frame(
    results: pd.DataFrame,
    settings: EloSettings | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, dict[str, float]]]:
    settings = settings or EloSettings()
    ratings: defaultdict[str, float] = defaultdict(lambda: settings.base_rating)
    points_form: defaultdict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=settings.form_window)
    )
    gd_form: defaultdict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=settings.form_window)
    )

    rows: list[dict[str, object]] = []

    for match in results.sort_values("date").itertuples(index=False):
        home = str(match.home_team)
        away = str(match.away_team)
        home_score = int(match.home_score)
        away_score = int(match.away_score)
        tournament = str(match.tournament)
        neutral = bool(match.neutral)
        importance = tournament_importance(tournament)
        home_advantage = 0.0 if neutral else settings.home_advantage
        home_elo = float(ratings[home])
        away_elo = float(ratings[away])
        home_form_points = _average(points_form[home])
        away_form_points = _average(points_form[away])
        home_form_gd = _average(gd_form[home])
        away_form_gd = _average(gd_form[away])

        rows.append(
            {
                "date": match.date,
                "home_team": home,
                "away_team": away,
                "tournament": tournament,
                "home_goals": home_score,
                "away_goals": away_score,
                "result": result_label(home_score, away_score),
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo + home_advantage - away_elo,
                "home_form_points": home_form_points,
                "away_form_points": away_form_points,
                "form_points_diff": home_form_points - away_form_points,
                "home_form_goal_diff": home_form_gd,
                "away_form_goal_diff": away_form_gd,
                "form_goal_diff": home_form_gd - away_form_gd,
                "neutral": int(neutral),
                "home_advantage": home_advantage,
                "is_world_cup": int(importance >= 0.95),
                "is_qualifier": int(0.75 <= importance < 0.95),
                "is_friendly": int(importance <= 0.4),
                "tournament_importance": importance,
            }
        )

        home_actual = score_points(home_score, away_score)
        away_actual = 1.0 - home_actual
        home_expected = expected_score(home_elo + home_advantage, away_elo)
        multiplier = _goal_margin_multiplier(home_score, away_score)
        delta = elo_k_factor(tournament, settings) * multiplier * (home_actual - home_expected)

        ratings[home] += delta
        ratings[away] -= delta

        points_form[home].append(3.0 if home_score > away_score else 1.0 if home_score == away_score else 0.0)
        points_form[away].append(3.0 if away_score > home_score else 1.0 if home_score == away_score else 0.0)
        gd_form[home].append(float(home_score - away_score))
        gd_form[away].append(float(away_score - home_score))

    team_form = {
        team: {
            "points": _average(points_form[team]),
            "goal_diff": _average(gd_form[team]),
        }
        for team in ratings.keys()
    }
    return pd.DataFrame(rows), dict(ratings), team_form


def build_team_player_feature_lookup(
    team_player_features: pd.DataFrame | None,
    aliases: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    aliases = aliases or {}
    if team_player_features is None or team_player_features.empty:
        defaults = {column: 0.0 for column in TEAM_PLAYER_FEATURE_SOURCE_COLUMNS}
        return {}, defaults

    df = team_player_features.copy()
    for column in TEAM_PLAYER_FEATURE_SOURCE_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce")

    defaults = {
        column: float(df[column].median(skipna=True))
        if not pd.isna(df[column].median(skipna=True))
        else 0.0
        for column in TEAM_PLAYER_FEATURE_SOURCE_COLUMNS
    }

    lookup: dict[str, dict[str, float]] = {}
    for row in df.itertuples(index=False):
        team = str(getattr(row, "team"))
        values = {
            column: float(getattr(row, column))
            if not pd.isna(getattr(row, column))
            else defaults[column]
            for column in TEAM_PLAYER_FEATURE_SOURCE_COLUMNS
        }
        keys = {normalized_team_key(team)}
        model_name = aliases.get(team)
        if model_name:
            keys.add(normalized_team_key(model_name))
        for tournament_name, alias_model_name in aliases.items():
            if alias_model_name == team:
                keys.add(normalized_team_key(tournament_name))
        for key in keys:
            lookup[key] = values

    return lookup, defaults


def normalized_team_key(team: str) -> str:
    return "".join(char for char in str(team).casefold() if char.isalnum())


def team_player_feature_row(
    home_team: str,
    away_team: str,
    lookup: dict[str, dict[str, float]],
    defaults: dict[str, float],
) -> dict[str, float]:
    home_values = lookup.get(normalized_team_key(home_team), defaults)
    away_values = lookup.get(normalized_team_key(away_team), defaults)
    home_has_features = float(normalized_team_key(home_team) in lookup)
    away_has_features = float(normalized_team_key(away_team) in lookup)

    row = {
        "home_has_player_features": home_has_features,
        "away_has_player_features": away_has_features,
    }
    home_feature_values: dict[str, float] = {}
    away_feature_values: dict[str, float] = {}
    for column in TEAM_PLAYER_FEATURE_SOURCE_COLUMNS:
        home_value = float(home_values.get(column, defaults.get(column, 0.0)))
        away_value = float(away_values.get(column, defaults.get(column, 0.0)))
        home_feature_values[column] = home_value
        away_feature_values[column] = away_value
        row[f"home_player_{column}"] = home_value
        row[f"away_player_{column}"] = away_value
        row[f"player_diff_{column}"] = home_value - away_value

    def home(column: str) -> float:
        return home_feature_values.get(column, 0.0)

    def away(column: str) -> float:
        return away_feature_values.get(column, 0.0)

    home_attack_vs_away_defense = (
        home("starting_fw_attacking_score")
        + home("starting_mf_creativity_score")
        - away("starting_def_defensive_score")
        - away("starting_gk_keeper_score")
    )
    away_attack_vs_home_defense = (
        away("starting_fw_attacking_score")
        + away("starting_mf_creativity_score")
        - home("starting_def_defensive_score")
        - home("starting_gk_keeper_score")
    )
    home_creativity_vs_away_ball_winning = (
        home("top_11_creativity_score") - away("top_11_ball_winning_score")
    )
    away_creativity_vs_home_ball_winning = (
        away("top_11_creativity_score") - home("top_11_ball_winning_score")
    )
    home_shot_volume_vs_away_keeper = (
        home("starting_fw_shot_volume_score") - away("starting_gk_keeper_score")
    )
    away_shot_volume_vs_home_keeper = (
        away("starting_fw_shot_volume_score") - home("starting_gk_keeper_score")
    )
    home_physical_vs_away_physical = (
        home("top_11_physical_score") - away("top_11_physical_score")
    )
    away_physical_vs_home_physical = -home_physical_vs_away_physical

    row.update(
        {
            "home_attack_vs_away_defense": home_attack_vs_away_defense,
            "away_attack_vs_home_defense": away_attack_vs_home_defense,
            "attack_matchup_diff": home_attack_vs_away_defense - away_attack_vs_home_defense,
            "home_creativity_vs_away_ball_winning": home_creativity_vs_away_ball_winning,
            "away_creativity_vs_home_ball_winning": away_creativity_vs_home_ball_winning,
            "creativity_matchup_diff": (
                home_creativity_vs_away_ball_winning - away_creativity_vs_home_ball_winning
            ),
            "home_shot_volume_vs_away_keeper": home_shot_volume_vs_away_keeper,
            "away_shot_volume_vs_home_keeper": away_shot_volume_vs_home_keeper,
            "finishing_keeper_matchup_diff": (
                home_shot_volume_vs_away_keeper - away_shot_volume_vs_home_keeper
            ),
            "home_physical_vs_away_physical": home_physical_vs_away_physical,
            "away_physical_vs_home_physical": away_physical_vs_home_physical,
            "physical_matchup_diff": home_physical_vs_away_physical - away_physical_vs_home_physical,
            "keeper_edge": home("starting_gk_keeper_score") - away("starting_gk_keeper_score"),
            "bench_depth_edge": home("bench_depth_score") - away("bench_depth_score"),
            "discipline_risk_edge": away("top_11_discipline_risk") - home("top_11_discipline_risk"),
            "availability_edge": (
                home("availability_multiplier_mean") - away("availability_multiplier_mean")
            ),
            "unavailable_player_edge": away("is_unavailable_sum") - home("is_unavailable_sum"),
            "pressing_edge": home("tactic_pressing_intensity") - away("tactic_pressing_intensity"),
            "defensive_line_edge": home("tactic_defensive_line") - away("tactic_defensive_line"),
            "tempo_edge": home("tactic_tempo") - away("tactic_tempo"),
            "directness_edge": home("tactic_directness") - away("tactic_directness"),
            "possession_edge": home("tactic_possession_style") - away("tactic_possession_style"),
            "width_edge": home("tactic_width") - away("tactic_width"),
            "set_piece_edge": home("tactic_set_piece_strength") - away("tactic_set_piece_strength"),
            "transition_edge": home("tactic_transition_speed") - away("tactic_transition_speed"),
            "formation_attack_density_edge": (
                home("formation_attacking_density") - away("formation_attacking_density")
            ),
            "formation_defensive_density_edge": (
                home("formation_defensive_density") - away("formation_defensive_density")
            ),
        }
    )
    return row


def add_team_player_features(
    frame: pd.DataFrame,
    team_player_features: pd.DataFrame | None,
    aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, float]]:
    lookup, defaults = build_team_player_feature_lookup(team_player_features, aliases=aliases)
    enhanced_rows = [
        team_player_feature_row(str(row.home_team), str(row.away_team), lookup, defaults)
        for row in frame.itertuples(index=False)
    ]
    enhanced = pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(enhanced_rows)],
        axis=1,
    )
    return enhanced, lookup, defaults
