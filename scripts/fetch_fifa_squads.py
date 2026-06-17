from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import FIFA_SQUAD_PDF_FILE, FIFA_SQUAD_PLAYERS_FILE
from worldcup2026.player_data import download_fifa_squad_pdf, save_fifa_squad_players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and parse official FIFA 2026 squads.")
    parser.add_argument("--pdf", type=Path, default=FIFA_SQUAD_PDF_FILE)
    parser.add_argument("--output", type=Path, default=FIFA_SQUAD_PLAYERS_FILE)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_fifa_squad_pdf(args.pdf, force=args.force_download)
    output_path = save_fifa_squad_players(args.pdf, args.output)
    print(f"Wrote FIFA squad players: {output_path}")


if __name__ == "__main__":
    main()
