from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import (
    GROUP_STAGE_SCHEDULE_FILE,
    MODEL_FILE,
    OUTPUT_DIR,
    PLAYER_FEATURES_FILE,
    PROJECTED_LINEUPS_FILE,
    RESULTS_2026_FILE,
    SAMPLED_KNOCKOUT_PREDICTIONS_FILE,
    TEAMS_2026_FILE,
)
from worldcup2026.data import load_actual_results, load_teams, normalized_key
from worldcup2026.edge_features import (
    add_edge_feature_columns,
    apply_edge_probability_adjustments,
    load_edge_context,
)
from worldcup2026.group_stage import add_match_usefulness_filters, fetch_group_stage_schedule
from worldcup2026.lineups import lineup_string, load_player_features, project_all_starting_lineups
from worldcup2026.live import (
    apply_played_results_to_predictor,
    played_schedule_results,
)
from worldcup2026.model import MatchPredictor
from worldcup2026.news_signals import add_news_signal_columns
from worldcup2026.tournament import (
    predict_group_match_probabilities,
    predict_sampled_knockout_matches,
    simulate_many,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict World Cup 2026 matches and winner.")
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--teams", type=Path, default=TEAMS_2026_FILE)
    parser.add_argument("--results", type=Path, default=RESULTS_2026_FILE)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--player-features", type=Path, default=PLAYER_FEATURES_FILE)
    parser.add_argument("--schedule", type=Path, default=GROUP_STAGE_SCHEDULE_FILE)
    parser.add_argument("--refresh-schedule", action="store_true")
    parser.add_argument("--no-live-results", action="store_true")
    parser.add_argument("--sims", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_schedule_context(group_predictions: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    if group_predictions.empty or schedule.empty:
        return group_predictions

    context_columns = [
        "status",
        "local_date",
        "local_time",
        "utc_offset",
        "kickoff_utc",
        "venue",
        "source_url",
    ]
    schedule_lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in schedule.to_dict("records"):
        key = (
            str(row.get("group", "")),
            normalized_key(row.get("home_team", "")),
            normalized_key(row.get("away_team", "")),
        )
        reverse_key = (key[0], key[2], key[1])
        context = {column: row.get(column, "") for column in context_columns}
        schedule_lookup[key] = context
        schedule_lookup.setdefault(reverse_key, context)

    enriched = group_predictions.copy()
    for column in context_columns:
        enriched[column] = ""
    for index, row in enriched.iterrows():
        key = (
            str(row.get("group", "")),
            normalized_key(row.get("home_team", "")),
            normalized_key(row.get("away_team", "")),
        )
        context = schedule_lookup.get(key)
        if context:
            for column in context_columns:
                enriched.at[index, column] = context.get(column, "")
    return enriched


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise SystemExit(
            f"Model not found at {args.model_path}. Run: python scripts/train_model.py --download"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictor = MatchPredictor.load(args.model_path)
    teams = load_teams(args.teams)
    actual_results = load_actual_results(args.results)
    schedule = (
        fetch_group_stage_schedule(args.schedule)
        if args.refresh_schedule or not args.schedule.exists()
        else pd.read_csv(args.schedule)
    )
    live_results_applied = 0
    if not args.no_live_results:
        schedule_results = played_schedule_results(schedule)
        if not schedule_results.empty:
            live_results_applied = apply_played_results_to_predictor(predictor, schedule_results)
            actual_results = pd.concat([actual_results, schedule_results], ignore_index=True)
            if not actual_results.empty:
                # Schedule results are the live source of truth; drop any manual-log
                # rows for the same match so played results are not double-counted.
                match_key = (
                    actual_results["group"].astype(str).str.upper().str.strip()
                    + "|" + actual_results["home_team"].map(normalized_key)
                    + "|" + actual_results["away_team"].map(normalized_key)
                )
                actual_results = actual_results[~match_key.duplicated(keep="last")].reset_index(drop=True)

    group_predictions = predict_group_match_probabilities(predictor, teams)
    group_predictions = add_schedule_context(group_predictions, schedule)
    team_context, venue_context, match_context, team_player_features, player_readiness_signals = load_edge_context()
    group_predictions = add_edge_feature_columns(
        group_predictions,
        schedule=schedule,
        team_context=team_context,
        venue_context=venue_context,
        match_context=match_context,
        team_player_features=team_player_features,
        player_readiness_signals=player_readiness_signals,
    )
    group_predictions = apply_edge_probability_adjustments(
        group_predictions,
        decision_policy=predictor.artifact.metadata.get("decision_policy", {}),
    )
    group_predictions = add_match_usefulness_filters(group_predictions)
    group_predictions = add_news_signal_columns(group_predictions)
    probabilities, sampled_matches = simulate_many(
        predictor,
        teams,
        actual_results=actual_results,
        group_predictions=group_predictions,
        simulations=args.sims,
        seed=args.seed,
    )

    group_predictions_path = args.out_dir / "group_match_probabilities.csv"
    tournament_path = args.out_dir / "tournament_probabilities.csv"
    sample_path = args.out_dir / "sampled_tournament_matches.csv"
    lineups_path = args.out_dir / PROJECTED_LINEUPS_FILE.name
    knockout_path = args.out_dir / SAMPLED_KNOCKOUT_PREDICTIONS_FILE.name

    lineups = None
    if args.player_features.exists():
        players = load_player_features(args.player_features)
        lineups = project_all_starting_lineups(players)
        lineups.to_csv(lineups_path, index=False)

    knockout_predictions = predict_sampled_knockout_matches(predictor, sampled_matches)
    if lineups is not None and not knockout_predictions.empty:
        knockout_predictions["home_projected_lineup"] = knockout_predictions["home_team"].map(
            lambda team: lineup_string(lineups, team)
        )
        knockout_predictions["away_projected_lineup"] = knockout_predictions["away_team"].map(
            lambda team: lineup_string(lineups, team)
        )

    group_predictions.to_csv(group_predictions_path, index=False)
    probabilities.to_csv(tournament_path, index=False)
    sampled_matches.to_csv(sample_path, index=False)
    knockout_predictions.to_csv(knockout_path, index=False)

    print(f"Wrote group match predictions: {group_predictions_path}")
    print(f"Wrote tournament probabilities: {tournament_path}")
    print(f"Wrote sampled bracket: {sample_path}")
    print(f"Wrote sampled knockout predictions: {knockout_path}")
    if lineups is not None:
        print(f"Wrote projected lineups: {lineups_path}")
    print(f"Live results applied: {live_results_applied}")
    print("\nTop champion probabilities:")
    print(probabilities[["team", "group", "champion", "final", "semifinal"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
