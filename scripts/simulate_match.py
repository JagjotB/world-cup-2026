"""
Simulate a match and print a report with scoreline distribution and sample timelines.

Usage:
    python scripts/simulate_match.py "Mexico" "South Korea"
    python scripts/simulate_match.py "Brazil" "France" --sims 100000 --examples 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import MODEL_FILE  # noqa: E402
from worldcup2026.live import apply_played_results_to_predictor, played_schedule_results  # noqa: E402
from worldcup2026.match_simulator import (  # noqa: E402
    goal_timing_distribution,
    outcome_probabilities,
    run_simulations,
    scoreline_distribution,
)
from worldcup2026.model import MatchPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a World Cup match.")
    parser.add_argument("home_team", help="Home / first-listed team")
    parser.add_argument("away_team", help="Away / second-listed team")
    parser.add_argument("--sims", type=int, default=50_000, help="Number of simulations (default 50000)")
    parser.add_argument("--examples", type=int, default=3, help="Number of sample timelines to print")
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "#" * filled + "." * (width - filled)


def main() -> None:
    args = parse_args()

    predictor = MatchPredictor.load(args.model_path)

    # Apply live WC results so ratings are up to date
    try:
        results = played_schedule_results()
        if results:
            apply_played_results_to_predictor(predictor, results)
    except Exception:
        pass

    prediction = predictor.predict_match(args.home_team, args.away_team, neutral=True)
    home_xg = prediction.expected_home_goals
    away_xg = prediction.expected_away_goals

    divider = "-" * 52

    print(f"\n{divider}")
    print(f"  {args.home_team:<22} vs  {args.away_team}")
    print(f"  xG  {home_xg:.2f}{'':>17}{away_xg:.2f}")
    print(divider)

    sims = run_simulations(
        args.home_team, args.away_team, home_xg, away_xg,
        n=args.sims, seed=args.seed,
    )

    probs = outcome_probabilities(sims)
    print(f"\nOutcomes  ({args.sims:,} simulations)")
    print(f"  {args.home_team} win  {probs['H']:>5.1%}  {bar(probs['H'])}")
    print(f"  Draw         {probs['D']:>5.1%}  {bar(probs['D'])}")
    print(f"  {args.away_team} win  {probs['A']:>5.1%}  {bar(probs['A'])}")

    print("\nTop scorelines")
    dist = scoreline_distribution(sims, top_n=12)
    for scoreline, p in dist:
        h_g, a_g = map(int, scoreline.split("-"))
        if h_g > a_g:
            label = f"{args.home_team} win"
        elif a_g > h_g:
            label = f"{args.away_team} win"
        else:
            label = "Draw"
        print(f"  {scoreline}   {p:>5.1%}  {bar(p, 16)}  {label}")

    print("\nGoal timing")
    timing = goal_timing_distribution(sims)
    for band, p in timing.items():
        print(f"  {band:<8}  {p:>5.1%}  {bar(p, 14)}")

    rng = np.random.default_rng(args.seed + 1)
    indices = rng.choice(len(sims), size=args.examples, replace=False)

    print("\nSample timelines")
    for i, idx in enumerate(indices):
        sim = sims[int(idx)]
        result_label = (
            f"{args.home_team} win" if sim.result() == "H"
            else f"{args.away_team} win" if sim.result() == "A"
            else "Draw"
        )
        print(f"\n  [{i+1}]  {args.home_team} {sim.home_goals}-{sim.away_goals} {args.away_team}  ({result_label})")
        print(sim.timeline_str())

    print(f"\n{divider}\n")


if __name__ == "__main__":
    main()
