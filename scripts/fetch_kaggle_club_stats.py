from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import CLUB_PLAYER_STATS_FILE
from worldcup2026.player_data import download_kaggle_club_player_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download 2025-26 club player stats from Kaggle.")
    parser.add_argument("--output", type=Path, default=CLUB_PLAYER_STATS_FILE)
    parser.add_argument("--full", action="store_true", help="Use the wider full CSV instead of light CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = download_kaggle_club_player_stats(args.output, light=not args.full)
    print(f"Wrote club player stats: {output_path}")


if __name__ == "__main__":
    main()
