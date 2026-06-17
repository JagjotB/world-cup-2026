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
from worldcup2026.data import load_actual_results, load_teams
from worldcup2026.group_stage import fetch_group_stage_schedule
from worldcup2026.lineups import lineup_string, load_player_features, project_all_starting_lineups
from worldcup2026.live import (
    apply_played_results_to_predictor,
    played_schedule_results,
)
from worldcup2026.model import MatchPredictor
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
    live_results_applied = 0
    if not args.no_live_results:
        schedule = (
            fetch_group_stage_schedule(args.schedule)
            if args.refresh_schedule or not args.schedule.exists()
            else pd.read_csv(args.schedule)
        )
        schedule_results = played_schedule_results(schedule)
        if not schedule_results.empty:
            live_results_applied = apply_played_results_to_predictor(predictor, schedule_results)
            actual_results = pd.concat([actual_results, schedule_results], ignore_index=True)

    group_predictions = predict_group_match_probabilities(predictor, teams)
    probabilities, sampled_matches = simulate_many(
        predictor,
        teams,
        actual_results=actual_results,
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
