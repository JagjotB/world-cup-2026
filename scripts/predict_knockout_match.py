from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import MODEL_FILE, PLAYER_FEATURES_FILE
from worldcup2026.knockout import predict_knockout_resolution
from worldcup2026.lineups import lineup_string, load_player_features, project_all_starting_lineups
from worldcup2026.model import MatchPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a knockout-stage World Cup match.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--player-features", type=Path, default=PLAYER_FEATURES_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = MatchPredictor.load(args.model_path)
    prediction = predict_knockout_resolution(predictor, args.home, args.away)

    print(f"{args.home} vs {args.away}")
    print(f"Regulation: {args.home} {prediction['p_home_win_regulation']:.1%}, "
          f"Draw {prediction['p_draw_regulation']:.1%}, "
          f"{args.away} {prediction['p_away_win_regulation']:.1%}")
    print(f"Extra time: {prediction['p_extra_time']:.1%}")
    print(f"Penalties: {prediction['p_penalty_shootout']:.1%}")
    print(f"Golden goal active: {prediction['golden_goal_rule_active']}")
    print(f"Advance: {args.home} {prediction['p_home_advance_total']:.1%}, "
          f"{args.away} {prediction['p_away_advance_total']:.1%}")

    if args.player_features.exists():
        lineups = project_all_starting_lineups(load_player_features(args.player_features))
        print(f"\n{args.home} projected XI:")
        print(lineup_string(lineups, args.home))
        print(f"\n{args.away} projected XI:")
        print(lineup_string(lineups, args.away))


if __name__ == "__main__":
    main()
