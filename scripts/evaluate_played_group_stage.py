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
    RAW_RESULTS_FILE,
    TEAM_PLAYER_FEATURES_FILE,
)
from worldcup2026.data import load_aliases, load_historical_results
from worldcup2026.group_stage import fetch_group_stage_schedule
from worldcup2026.live import apply_played_results_to_predictor
from worldcup2026.model import MatchPredictor
from worldcup2026.train import train_match_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions for played group-stage matches.")
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--schedule", type=Path, default=GROUP_STAGE_SCHEDULE_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "played_group_stage_evaluation.csv")
    parser.add_argument("--model-kind", choices=["baseline", "enhanced"], default="enhanced")
    parser.add_argument("--team-player-features", type=Path, default=TEAM_PLAYER_FEATURES_FILE)
    parser.add_argument(
        "--pre-world-cup-model",
        action="store_true",
        help="Retrain a temporary model excluding all 2026 World Cup matches before evaluating.",
    )
    parser.add_argument("--refresh-schedule", action="store_true")
    parser.add_argument(
        "--sequential-live",
        action="store_true",
        help="Before each played match, update ratings/form using only earlier played matches.",
    )
    return parser.parse_args()


def load_predictor(args: argparse.Namespace) -> MatchPredictor:
    if not args.pre_world_cup_model:
        return MatchPredictor.load(args.model_path)

    results = load_historical_results(RAW_RESULTS_FILE)
    pre_world_cup = results[results["date"] < pd.Timestamp("2026-06-11")].copy()
    team_player_features = (
        pd.read_csv(args.team_player_features)
        if args.model_kind == "enhanced" and args.team_player_features.exists()
        else None
    )
    artifact, _ = train_match_model(
        pre_world_cup,
        aliases=load_aliases(),
        team_player_features=team_player_features,
        min_year=2000,
        test_fraction=0.2,
        model_kind=args.model_kind,
    )
    return MatchPredictor(artifact)


def actual_result(row) -> str:
    if row.home_score > row.away_score:
        return row.home_team
    if row.away_score > row.home_score:
        return row.away_team
    return "Draw"


def result_row(row) -> dict[str, object]:
    return {
        "stage": "group",
        "group": row.group,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "home_score": int(row.home_score),
        "away_score": int(row.away_score),
    }


def sort_played_matches(played: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ["kickoff_utc", "local_date", "local_time", "group", "group_match_index"]
        if column in played.columns
    ]
    return played.sort_values(sort_columns).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.refresh_schedule or not args.schedule.exists():
        schedule = fetch_group_stage_schedule(args.schedule)
    else:
        schedule = pd.read_csv(args.schedule)

    played = schedule[schedule["status"].eq("played")].copy()
    predictor = load_predictor(args)

    rows = []
    played = sort_played_matches(played)
    for row in played.itertuples(index=False):
        pred = predictor.predict_match(row.home_team, row.away_team, neutral=True)
        decision = predictor.decision_for_prediction(pred)
        probs = {
            row.home_team: pred.p_home_win,
            "Draw": pred.p_draw,
            row.away_team: pred.p_away_win,
        }
        pick = decision.recommended_result
        actual = actual_result(row)
        rows.append(
            {
                "date": row.local_date,
                "group": row.group,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "score": f"{int(row.home_score)}-{int(row.away_score)}",
                "actual": actual,
                "pick": pick,
                "raw_top_result": decision.raw_top_result,
                "pick_confidence": decision.confidence,
                "draw_override_applied": decision.draw_override_applied,
                "correct": pick == actual,
                "p_home": pred.p_home_win,
                "p_draw": pred.p_draw,
                "p_away": pred.p_away_win,
                "picked_probability": probs[pick],
                "actual_probability": probs[actual],
            }
        )
        if args.sequential_live:
            current_result = pd.DataFrame([result_row(row)])
            apply_played_results_to_predictor(predictor, current_result)

    evaluation = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(args.output, index=False)

    correct = int(evaluation["correct"].sum()) if not evaluation.empty else 0
    total = len(evaluation)
    accuracy = correct / total if total else 0.0
    print(f"Wrote evaluation: {args.output}")
    print(f"Played matches: {total}")
    print(f"Correct picks: {correct}")
    print(f"Accuracy: {accuracy:.1%}")
    if not evaluation.empty:
        print(f"Average probability assigned to actual result: {evaluation['actual_probability'].mean():.1%}")
        print(evaluation.to_string(index=False))


if __name__ == "__main__":
    main()
