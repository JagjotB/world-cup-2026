"""
Simulate every group stage match and compare to actual results where available.

Usage:
    python scripts/simulate_all_matches.py
    python scripts/simulate_all_matches.py --sims 20000 --played-only
    python scripts/simulate_all_matches.py --upcoming-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import GROUP_STAGE_SCHEDULE_FILE, MODEL_FILE  # noqa: E402
from worldcup2026.live import apply_played_results_to_predictor, played_schedule_results  # noqa: E402
from worldcup2026.match_simulator import (  # noqa: E402
    SIM_DRAW_PROMOTE_MARGIN,
    SIM_DRAW_PROMOTE_THRESHOLD,
    outcome_probabilities,
    run_simulations,
    scoreline_distribution,
    sim_predict_result,
)
from worldcup2026.model import MatchPredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate all group stage matches.")
    parser.add_argument("--sims", type=int, default=20_000)
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--played-only", action="store_true")
    parser.add_argument("--upcoming-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def result_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    elif away_score > home_score:
        return "A"
    return "D"


def bar(p: float, width: int = 14) -> str:
    filled = round(p * width)
    return "#" * filled + "." * (width - filled)


def main() -> None:
    args = parse_args()

    schedule = pd.read_csv(GROUP_STAGE_SCHEDULE_FILE)
    predictor = MatchPredictor.load(args.model_path)

    wc_results = played_schedule_results(schedule)
    if not wc_results.empty:
        apply_played_results_to_predictor(predictor, wc_results)

    played = schedule[schedule["status"] == "played"].copy()
    upcoming = schedule[schedule["status"] != "played"].copy()

    if args.played_only:
        matches = played
    elif args.upcoming_only:
        matches = upcoming
    else:
        matches = pd.concat([played, upcoming]).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Per-match simulation
    # -----------------------------------------------------------------------
    correct_naive = 0      # old: max probability
    correct_fixed = 0      # new: draw boost + margin rule
    correct_scoreline = 0
    total_played = 0

    rng_master = np.random.default_rng(args.seed)

    for row in matches.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        is_played = str(row.status) == "played"
        seed_i = int(rng_master.integers(0, 1_000_000))

        try:
            prediction = predictor.predict_match(home, away, neutral=True)
        except Exception as e:
            print(f"  [SKIP] {home} vs {away}: {e}")
            continue

        sims = run_simulations(
            home, away,
            prediction.expected_home_goals,
            prediction.expected_away_goals,
            n=args.sims,
            seed=seed_i,
        )

        probs = outcome_probabilities(sims)
        naive_result = max(probs, key=lambda k: probs[k])
        fixed_result = sim_predict_result(
            probs,
            home_xg=prediction.expected_home_goals,
            away_xg=prediction.expected_away_goals,
        )
        top_scorelines = scoreline_distribution(sims, top_n=5)

        date_str = str(row.local_date) if hasattr(row, "local_date") else ""
        group_str = str(row.group)

        print(f"\n{'=' * 58}")
        print(f"  Group {group_str}  |  {date_str}")
        print(f"  {home:<24} vs  {away}")
        print(f"  xG: {prediction.expected_home_goals:.2f} — {prediction.expected_away_goals:.2f}")
        print(f"  {home} win {probs['H']:>5.1%}  |  Draw {probs['D']:>5.1%}  |  {away} win {probs['A']:>5.1%}")
        print(f"  Predicted (fixed): {fixed_result}   (naive: {naive_result})")

        print("  Top scorelines:")
        for sl, p in top_scorelines:
            h_g, a_g = map(int, sl.split("-"))
            winner = home if h_g > a_g else (away if a_g > h_g else "Draw")
            print(f"    {sl}  {p:>5.1%}  {bar(p)}  {winner}")

        if is_played:
            actual_h = int(float(row.home_score))
            actual_a = int(float(row.away_score))
            actual_result = result_label(actual_h, actual_a)
            actual_sl = f"{actual_h}-{actual_a}"

            naive_ok = naive_result == actual_result
            fixed_ok = fixed_result == actual_result
            sl_match = top_scorelines[0][0] == actual_sl if top_scorelines else False

            correct_naive += int(naive_ok)
            correct_fixed += int(fixed_ok)
            correct_scoreline += int(sl_match)
            total_played += 1

            actual_winner = home if actual_h > actual_a else (away if actual_a > actual_h else "Draw")
            print(f"  Actual:    {actual_sl}  ({actual_winner})  [{actual_result}]")
            print(f"  Naive:     [{'OK' if naive_ok else 'XX'}]  predicted {naive_result}")
            thresh = f"threshold={SIM_DRAW_PROMOTE_THRESHOLD:.2f}, margin={SIM_DRAW_PROMOTE_MARGIN}"
            print(f"  Fixed:     [{'OK' if fixed_ok else 'XX'}]  predicted {fixed_result}  ({thresh})")
            top_sl_str = top_scorelines[0][0] if top_scorelines else "N/A"
            print(f"  Scoreline: [{'OK' if sl_match else 'XX'}]  sim top {top_sl_str}, actual {actual_sl}")
        else:
            print("  (upcoming)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    if total_played > 0:
        print(f"\n{'=' * 58}")
        print(f"  ACCURACY COMPARISON  ({total_played} played matches)")
        print(f"{'=' * 58}")
        print(f"  Naive (max prob)     : {correct_naive}/{total_played}  ({correct_naive/total_played:.1%})")
        print(f"  Fixed (boost+margin) : {correct_fixed}/{total_played}  ({correct_fixed/total_played:.1%})")
        print(f"  Exact scoreline      : {correct_scoreline}/{total_played}  ({correct_scoreline/total_played:.1%})")
        delta = correct_fixed - correct_naive
        sign = "+" if delta >= 0 else ""
        print(f"  Improvement          : {sign}{delta} results")
        print(f"{'=' * 58}\n")


if __name__ == "__main__":
    main()
