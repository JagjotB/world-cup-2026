"""
Fetch pre-match news for upcoming group stage games and extract signals via Claude.

Usage:
    python scripts/fetch_news_signals.py [--date YYYY-MM-DD] [--all]

Requires ANTHROPIC_API_KEY env variable. Falls back to neutral defaults if not set.
Output: data/manual/llm_news_signals_2026.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from worldcup2026.config import GROUP_STAGE_SCHEDULE_FILE, NEWS_SIGNALS_FILE
from worldcup2026.news_signals import (
    build_news_feature_row,
    extract_signals_with_llm,
    fetch_match_news_text,
)

def load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=str(date.today()))
    parser.add_argument("--all", action="store_true", help="Process all upcoming matches")
    args = parser.parse_args()

    schedule = pd.read_csv(GROUP_STAGE_SCHEDULE_FILE)
    upcoming = schedule[schedule["status"] == "upcoming"].copy()

    if not args.all:
        upcoming = upcoming[upcoming["local_date"] == args.date]

    if upcoming.empty:
        print(f"No upcoming matches found for {args.date}")
        return

    existing = load_existing(NEWS_SIGNALS_FILE)
    existing_keys = set()
    if not existing.empty and "home_team" in existing.columns:
        existing_keys = set(
            zip(existing["local_date"], existing["home_team"], existing["away_team"])
        )

    rows = []
    for _, row in upcoming.iterrows():
        key = (row["local_date"], row["home_team"], row["away_team"])
        if key in existing_keys:
            print(f"Skipping {row['home_team']} v {row['away_team']} (already processed)")
            continue

        print(f"Fetching: {row['home_team']} v {row['away_team']} ({row['local_date']})...")
        text = fetch_match_news_text(row["home_team"], row["away_team"], row["local_date"])
        print(f"  Text length: {len(text)} chars")

        signals = extract_signals_with_llm(
            home_team=row["home_team"],
            away_team=row["away_team"],
            date=row["local_date"],
            text=text,
        )
        features = build_news_feature_row(signals, row["home_team"], row["away_team"])

        record = {
            "local_date": row["local_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "text_length": len(text),
            "narrative_edge": signals.get("narrative_edge", "neutral"),
            "key_signals": "; ".join(signals.get("key_signals", [])),
            **features,
        }
        rows.append(record)
        print(f"  narrative_edge={signals['narrative_edge']}  confidence={signals['confidence']:.2f}")

    if rows:
        new_df = pd.DataFrame(rows)
        combined = (
            pd.concat([existing, new_df], ignore_index=True)
            if not existing.empty
            else new_df
        )
        NEWS_SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(NEWS_SIGNALS_FILE, index=False)
        print(f"\nSaved {len(rows)} new records to {NEWS_SIGNALS_FILE}")
    else:
        print("No new records to save.")


if __name__ == "__main__":
    main()
