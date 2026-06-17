from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import CLUB_PLAYER_STATS_FILE, CLUB_STAT_SOURCES_FILE
from worldcup2026.player_data import fetch_fbref_player_season_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch club player season stats from FBref.")
    parser.add_argument("--sources", type=Path, default=CLUB_STAT_SOURCES_FILE)
    parser.add_argument("--output", type=Path, default=CLUB_PLAYER_STATS_FILE)
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument(
        "--stat-types",
        default="standard",
        help="Comma-separated FBref stat types, for example standard,shooting,passing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = pd.read_csv(args.sources)
    fbref_sources = sources[
        (sources["source"].str.lower() == "fbref") & (sources["enabled"].astype(bool))
    ]
    leagues = fbref_sources["league"].dropna().astype(str).tolist()
    if not leagues:
        raise SystemExit(f"No enabled FBref leagues found in {args.sources}")

    stat_types = [item.strip() for item in args.stat_types.split(",") if item.strip()]
    stats = fetch_fbref_player_season_stats(leagues=leagues, season=args.season, stat_types=stat_types)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output, index=False)
    print(f"Wrote club player stats: {args.output}")
    print(f"Rows: {len(stats)}")


if __name__ == "__main__":
    main()
