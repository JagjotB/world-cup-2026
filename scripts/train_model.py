from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from worldcup2026.config import MODEL_FILE, RAW_RESULTS_FILE, TEAM_PLAYER_FEATURES_FILE
from worldcup2026.data import download_historical_results, load_aliases, load_historical_results
from worldcup2026.train import train_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the World Cup match model.")
    parser.add_argument("--raw-results", type=Path, default=RAW_RESULTS_FILE)
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--download", action="store_true", help="Download historical results first.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--model-kind", choices=["baseline", "enhanced"], default="enhanced")
    parser.add_argument("--team-player-features", type=Path, default=TEAM_PLAYER_FEATURES_FILE)
    parser.add_argument(
        "--exclude-2026-world-cup",
        action="store_true",
        help="Exclude 2026 World Cup matches from training for clean pre-tournament audits.",
    )
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download or not args.raw_results.exists():
        download_historical_results(args.raw_results, force=args.force_download)

    results = load_historical_results(args.raw_results)
    if args.exclude_2026_world_cup:
        results = results[
            ~(
                results["tournament"].eq("FIFA World Cup")
                & results["date"].ge(pd.Timestamp("2026-06-11"))
            )
        ].copy()

    aliases = load_aliases()
    team_player_features = None
    if args.model_kind == "enhanced":
        if args.team_player_features.exists():
            team_player_features = pd.read_csv(args.team_player_features)
        else:
            print(
                f"Warning: {args.team_player_features} not found; enhanced model will use default player features."
            )

    model_path, metrics = train_and_save(
        results=results,
        model_path=args.model_path,
        aliases=aliases,
        team_player_features=team_player_features,
        min_year=args.min_year,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
        model_kind=args.model_kind,
    )

    metrics_path = model_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
