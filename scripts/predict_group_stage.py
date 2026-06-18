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
    UPCOMING_GROUP_STAGE_PREDICTIONS_FILE,
)
from worldcup2026.edge_features import (
    add_edge_feature_columns,
    apply_edge_probability_adjustments,
    load_edge_context,
)
from worldcup2026.news_signals import add_news_signal_columns
from worldcup2026.group_stage import (
    add_match_usefulness_filters,
    fetch_group_stage_schedule,
    predict_group_stage_matches,
)
from worldcup2026.lineups import load_player_features
from worldcup2026.live import (
    apply_played_results_to_predictor,
    played_schedule_results,
)
from worldcup2026.model import MatchPredictor
from worldcup2026.player_projections import (
    add_match_team_projection_totals,
    project_player_match_performances,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict upcoming FIFA World Cup 2026 group matches.")
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--schedule-output", type=Path, default=GROUP_STAGE_SCHEDULE_FILE)
    parser.add_argument("--predictions-output", type=Path, default=UPCOMING_GROUP_STAGE_PREDICTIONS_FILE)
    parser.add_argument("--player-features", type=Path, default=PLAYER_FEATURES_FILE)
    parser.add_argument("--player-projections-output", type=Path, default=PLAYER_MATCH_PROJECTIONS_FILE)
    parser.add_argument("--refresh-schedule", action="store_true")
    parser.add_argument(
        "--from-date",
        default=datetime.now().date().isoformat(),
        help="Only include unplayed matches on or after this local date.",
    )
    parser.add_argument("--all", action="store_true", help="Predict played and upcoming group matches.")
    parser.add_argument(
        "--no-live-results",
        action="store_true",
        help="Do not update ratings/form with played World Cup group results before predicting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise SystemExit(
            f"Model not found at {args.model_path}. Run: python scripts/train_model.py --download"
        )

    schedule = (
        fetch_group_stage_schedule(args.schedule_output)
        if args.refresh_schedule or not args.schedule_output.exists()
        else pd.read_csv(args.schedule_output)
    )
    predictor = MatchPredictor.load(args.model_path)
    live_results_applied = 0
    if not args.no_live_results:
        schedule_results = played_schedule_results(schedule)
        live_results_applied = apply_played_results_to_predictor(predictor, schedule_results)
    predictions = predict_group_stage_matches(
        predictor,
        schedule,
        args.predictions_output,
        upcoming_only=not args.all,
        from_date=None if args.all else args.from_date,
    )
    player_projections = pd.DataFrame()
    if args.player_features.exists():
        player_projections = project_player_match_performances(
            predictor,
            schedule,
            load_player_features(args.player_features),
            output_path=args.player_projections_output,
            upcoming_only=not args.all,
            from_date=None if args.all else args.from_date,
        )
        predictions = add_match_team_projection_totals(predictions, player_projections)
    team_context, venue_context, match_context, team_player_features, player_readiness_signals = load_edge_context()
    predictions = add_edge_feature_columns(
        predictions,
        schedule=schedule,
        team_context=team_context,
        venue_context=venue_context,
        match_context=match_context,
        team_player_features=team_player_features,
        player_readiness_signals=player_readiness_signals,
    )
    predictions = apply_edge_probability_adjustments(
        predictions,
        decision_policy=predictor.artifact.metadata.get("decision_policy", {}),
    )
    predictions = add_match_usefulness_filters(predictions)
    predictions = add_news_signal_columns(predictions)
    predictions.to_csv(args.predictions_output, index=False)

    print(f"Wrote group-stage schedule: {args.schedule_output}")
    print(f"Wrote group-stage predictions: {args.predictions_output}")
    if not player_projections.empty:
        print(f"Wrote player match projections: {args.player_projections_output}")
    print(f"Live results applied: {live_results_applied}")
    print(f"Upcoming matches predicted from {args.from_date}: {len(predictions)}")
    if not predictions.empty:
        preview_columns = [
            "match_number",
            "group",
            "local_date",
            "local_time",
            "home_team",
            "away_team",
            "predicted_result",
            "p_home_win",
            "p_draw",
            "p_away_win",
            "edge_home_win_probability_delta",
            "edge_away_win_probability_delta",
            "double_chance_pick",
            "double_chance_useful",
            "either_team_wins_probability",
            "either_team_wins_useful",
            "btts_pick",
            "btts_useful",
            "projected_more_shots_on_target",
            "projected_home_shots_on_target",
            "projected_away_shots_on_target",
            "more_shots_on_target_useful",
            "p_home_1plus_corners_each_half",
            "p_away_1plus_corners_each_half",
            "home_1plus_corners_each_half_useful_pick",
            "away_1plus_corners_each_half_useful_pick",
            "edge_total_signal",
            "edge_total_signal_pick",
            "edge_total_signal_strength",
            "edge_home_travel_origin",
            "edge_away_travel_origin",
            "edge_home_travel_km",
            "edge_away_travel_km",
            "edge_readiness_edge",
            "edge_home_readiness_samples",
            "edge_away_readiness_samples",
            "edge_flags",
            "news_narrative_home",
            "news_narrative_away",
            "news_injury_edge",
            "news_morale_edge",
            "news_upset_risk",
            "news_signal_confidence",
        ]
        available_preview_columns = [
            column for column in preview_columns if column in predictions.columns
        ]
        print(predictions[available_preview_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
