"""
Fetch weather for all group stage matches and analyse how conditions
correlated with actual results on played matches.

Usage:
    python scripts/fetch_weather.py            # fetch + analyse
    python scripts/fetch_weather.py --analyse  # analyse cached data only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from worldcup2026.config import GROUP_STAGE_SCHEDULE_FILE  # noqa: E402
from worldcup2026.weather import (  # noqa: E402
    fetch_and_cache_weather,
    load_weather_cache,
    weather_goal_multiplier,
)


def bar(v: float, lo: float, hi: float, width: int = 12) -> str:
    pct = (v - lo) / max(hi - lo, 1e-6)
    filled = round(pct * width)
    return "#" * filled + "." * (width - filled)


def analyse(weather_df: pd.DataFrame, schedule: pd.DataFrame) -> None:
    played = schedule[schedule["status"] == "played"].copy()
    played["home_score"] = pd.to_numeric(played["home_score"], errors="coerce")
    played["away_score"] = pd.to_numeric(played["away_score"], errors="coerce")
    played["total_goals"] = played["home_score"] + played["away_score"]
    played["result"] = played.apply(
        lambda r: "H" if r.home_score > r.away_score
        else ("A" if r.away_score > r.home_score else "D"),
        axis=1,
    )

    merged = played.merge(
        weather_df,
        on=["local_date", "home_team", "away_team"],
        how="left",
    )
    merged = merged.dropna(subset=["temp_c"])

    print(f"\n{'=' * 62}")
    print("  WEATHER vs MATCH RESULTS  (played matches)")
    print(f"{'=' * 62}")

    print(f"\n  {'Match':<38} {'Temp':>5} {'Rain':>5} {'Wind':>5}  {'Goals':>5}  Conditions")
    print(f"  {'-' * 38} {'-' * 5} {'-' * 5} {'-' * 5}  {'-' * 5}  ----------")
    for _, r in merged.iterrows():
        match = f"{r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}"
        label = str(r.get("weather_label", ""))
        print(
            f"  {match:<38} {r.temp_c:>4.0f}C "
            f"{r.precip_mm:>4.0f}mm {r.wind_kmh:>4.0f}kph "
            f"  {int(r.total_goals):>2} gls  {label}"
        )

    # -----------------------------------------------------------------------
    # Temperature bands
    # -----------------------------------------------------------------------
    print(f"\n  {'Temperature bands':}")
    print(f"  {'Band':<18} {'Matches':>7} {'Avg goals':>9} {'Draws':>7} {'Draw%':>7}")
    print(f"  {'-' * 18} {'-' * 7} {'-' * 9} {'-' * 7} {'-' * 7}")
    bands = [
        ("< 15C (cool)",    merged["temp_c"] < 15),
        ("15-21C (mild)",  (merged["temp_c"] >= 15) & (merged["temp_c"] < 21)),
        ("21-27C (warm)",  (merged["temp_c"] >= 21) & (merged["temp_c"] < 27)),
        ("27-32C (hot)",   (merged["temp_c"] >= 27) & (merged["temp_c"] < 32)),
        (">= 32C (extreme)", merged["temp_c"] >= 32),
    ]
    for label, mask in bands:
        sub = merged[mask]
        if sub.empty:
            continue
        avg_g = sub["total_goals"].mean()
        draws = (sub["result"] == "D").sum()
        draw_pct = draws / len(sub)
        bar_str = bar(avg_g, 0, 6)
        print(
            f"  {label:<18} {len(sub):>7} {avg_g:>9.2f} {bar_str} "
            f"{draws:>7} {draw_pct:>7.0%}"
        )

    # -----------------------------------------------------------------------
    # Rain effect
    # -----------------------------------------------------------------------
    print(f"\n  {'Precipitation effect':}")
    print(f"  {'Band':<20} {'Matches':>7} {'Avg goals':>9} {'Draw%':>7}")
    print(f"  {'-' * 20} {'-' * 7} {'-' * 9} {'-' * 7}")
    rain_bands = [
        ("Dry (0mm)",      merged["precip_mm"] == 0),
        ("Light (0-4mm)", (merged["precip_mm"] > 0) & (merged["precip_mm"] < 4)),
        ("Rain (4mm+)",    merged["precip_mm"] >= 4),
    ]
    for label, mask in rain_bands:
        sub = merged[mask]
        if sub.empty:
            continue
        avg_g = sub["total_goals"].mean()
        draw_pct = (sub["result"] == "D").mean()
        bar_str = bar(avg_g, 0, 6)
        print(f"  {label:<20} {len(sub):>7} {avg_g:>9.2f} {bar_str} {draw_pct:>7.0%}")

    # -----------------------------------------------------------------------
    # Summary correlations
    # -----------------------------------------------------------------------
    if len(merged) >= 5:
        corr_temp  = merged["temp_c"].corr(merged["total_goals"])
        corr_rain  = merged["precip_mm"].corr(merged["total_goals"])
        corr_wind  = merged["wind_kmh"].corr(merged["total_goals"])
        print(f"\n  Correlations with total goals:")
        print(f"    Temperature : {corr_temp:+.3f}")
        print(f"    Precipitation: {corr_rain:+.3f}")
        print(f"    Wind speed  : {corr_wind:+.3f}")

    # -----------------------------------------------------------------------
    # Weather multipliers for upcoming matches
    # -----------------------------------------------------------------------
    upcoming = schedule[schedule["status"] == "upcoming"].copy()
    upcoming_w = upcoming.merge(
        weather_df,
        on=["local_date", "home_team", "away_team"],
        how="left",
    ).dropna(subset=["temp_c"])

    if not upcoming_w.empty:
        notable = upcoming_w[
            (upcoming_w["temp_c"] > 27)
            | (upcoming_w["precip_mm"] > 3)
            | (upcoming_w["wind_kmh"] > 25)
        ]
        if not notable.empty:
            print(f"\n  Upcoming matches with notable conditions:")
            for _, r in notable.iterrows():
                mults = weather_goal_multiplier(r.to_dict())
                print(
                    f"  {r.home_team} v {r.away_team} ({r.local_date})"
                    f"  {r.temp_c:.0f}C  {r.precip_mm:.0f}mm"
                    f"  2H mult={mults['second_half_mult']:.3f}"
                    f"  [{mults['label']}]"
                )
        else:
            print("\n  No upcoming matches with notable weather conditions.")

    print(f"\n{'=' * 62}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse", action="store_true", help="Skip fetch, analyse cached data")
    args = parser.parse_args()

    schedule = pd.read_csv(GROUP_STAGE_SCHEDULE_FILE)

    if not args.analyse:
        print("Fetching weather for all matches...")
        weather_df = fetch_and_cache_weather(schedule)
    else:
        weather_df = load_weather_cache()

    if weather_df.empty:
        print("No weather data available.")
        return

    analyse(weather_df, schedule)


if __name__ == "__main__":
    main()
