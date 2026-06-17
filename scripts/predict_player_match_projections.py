from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import (
    GROUP_STAGE_SCHEDULE_FILE,
    MODEL_FILE,
    PLAYER_FEATURES_FILE,
    PLAYER_MATCH_PROJECTIONS_FILE,
)
from worldcup2026.group_stage import fetch_group_stage_schedule
from worldcup2026.lineups import load_player_features
from worldcup2026.live import apply_played_results_to_predictor, played_schedule_results
from worldcup2026.model import MatchPredictor
from worldcup2026.player_projections import project_player_match_performances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project player performance for World Cup matches.")
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--schedule", type=Path, default=GROUP_STAGE_SCHEDULE_FILE)
    parser.add_argument("--player-features", type=Path, default=PLAYER_FEATURES_FILE)
    parser.add_argument("--output", type=Path, default=PLAYER_MATCH_PROJECTIONS_FILE)
    parser.add_argument("--refresh-schedule", action="store_true")
    parser.add_argument(
        "--from-date",
        default=datetime.now().date().isoformat(),
        help="Only include unplayed matches on or after this local date.",
    )
    parser.add_argument("--all", action="store_true", help="Project played and upcoming group matches.")
    parser.add_argument("--no-live-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = (
        fetch_group_stage_schedule(args.schedule)
        if args.refresh_schedule or not args.schedule.exists()
        else pd.read_csv(args.schedule)
    )
    predictor = MatchPredictor.load(args.model_path)
    if not args.no_live_results:
        apply_played_results_to_predictor(predictor, played_schedule_results(schedule))

    players = load_player_features(args.player_features)
    projections = project_player_match_performances(
        predictor,
        schedule,
        players,
        output_path=args.output,
        upcoming_only=not args.all,
        from_date=None if args.all else args.from_date,
    )

    print(f"Wrote player match projections: {args.output}")
    print(f"Rows: {len(projections)}")
    if not projections.empty:
        preview_columns = [
            "match_number",
            "team",
            "opponent",
            "player_name",
            "roster_role",
            "projected_minutes",
            "projected_goals",
            "projected_assists",
            "projected_shots",
            "impact_score",
        ]
        print(projections[preview_columns].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
