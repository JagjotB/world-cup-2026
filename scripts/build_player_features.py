from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import (
    ADDITIONAL_CLUB_PLAYER_STATS_FILE,
    CLUB_PLAYER_STATS_FILE,
    FIFA_SQUAD_PLAYERS_FILE,
    MANUAL_PROJECTED_LINEUPS_FILE,
    PLAYER_AVAILABILITY_FILE,
    TEAM_TACTICS_FILE,
    TEAM_PLAYER_FEATURES_FILE,
)
from worldcup2026.player_data import save_team_player_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player and team player-feature tables.")
    parser.add_argument("--squad", type=Path, default=FIFA_SQUAD_PLAYERS_FILE)
    parser.add_argument("--club-stats", type=Path, default=CLUB_PLAYER_STATS_FILE)
    parser.add_argument("--additional-club-stats", type=Path, default=ADDITIONAL_CLUB_PLAYER_STATS_FILE)
    parser.add_argument("--availability", type=Path, default=PLAYER_AVAILABILITY_FILE)
    parser.add_argument("--lineups", type=Path, default=MANUAL_PROJECTED_LINEUPS_FILE)
    parser.add_argument("--team-tactics", type=Path, default=TEAM_TACTICS_FILE)
    parser.add_argument("--output", type=Path, default=TEAM_PLAYER_FEATURES_FILE)
    return parser.parse_args()


def main() -> None:
    player_path, team_path = save_team_player_features(
        squad_path=args.squad,
        club_stats_path=args.club_stats,
        additional_club_stats_path=args.additional_club_stats,
        availability_path=args.availability,
        manual_lineups_path=args.lineups,
        team_tactics_path=args.team_tactics,
        output_path=args.output,
    )
    print(f"Wrote player features: {player_path}")
    print(f"Wrote team player features: {team_path}")


if __name__ == "__main__":
    args = parse_args()
    main()
