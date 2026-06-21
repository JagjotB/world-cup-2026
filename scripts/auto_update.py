"""Auto-update the World Cup 2026 pipeline after every match.

On each cycle this:
  1. Re-scrapes the live group-stage schedule from Wikipedia and
     regenerates predictions (delegates to predict_group_stage.py).
  2. Syncs newly played matches into the manual results log
     (data/manual/results_2026.csv).
  3. Optionally commits + pushes the results log when it changes.

Run once:
    python scripts/auto_update.py

Watch on an interval (default 600s) until interrupted:
    python scripts/auto_update.py --watch --interval 600 --commit

A running Streamlit dashboard picks up the refreshed files automatically
(its caches are keyed on file modification time).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import GROUP_STAGE_SCHEDULE_FILE, RESULTS_2026_FILE
from worldcup2026.data import load_actual_results, normalized_key

RESULTS_COLUMNS = ["stage", "group", "home_team", "away_team", "home_score", "away_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-update the World Cup 2026 pipeline after matches.")
    parser.add_argument("--watch", action="store_true", help="Loop forever instead of running once.")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between cycles in --watch mode.")
    parser.add_argument("--commit", action="store_true", help="Commit + push the results log when it changes.")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def regenerate_predictions() -> None:
    """Re-scrape the schedule and regenerate predictions via predict_group_stage."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "predict_group_stage.py"), "--refresh-schedule"]
    env_path = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={**_env(), "PYTHONPATH": env_path},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log("predict_group_stage.py failed:")
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(result.returncode)


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _match_key(group: str, home: str, away: str) -> tuple[str, str, str]:
    return (str(group).strip().upper(), normalized_key(home), normalized_key(away))


def sync_results_log() -> list[str]:
    """Append played schedule matches missing from the manual results log.

    Returns a list of human-readable descriptions of newly recorded matches.
    """
    schedule = pd.read_csv(GROUP_STAGE_SCHEDULE_FILE)
    played = schedule[schedule["status"].eq("played")].copy()
    played["home_score"] = pd.to_numeric(played["home_score"], errors="coerce")
    played["away_score"] = pd.to_numeric(played["away_score"], errors="coerce")
    played = played.dropna(subset=["home_score", "away_score"])

    existing = load_actual_results(RESULTS_2026_FILE)
    known = {
        _match_key(row.group, row.home_team, row.away_team)
        for row in existing.itertuples(index=False)
    } if not existing.empty else set()

    new_rows: list[dict[str, object]] = []
    added: list[str] = []
    for row in played.itertuples(index=False):
        key = _match_key(row.group, row.home_team, row.away_team)
        if key in known:
            continue
        known.add(key)
        home_score = int(row.home_score)
        away_score = int(row.away_score)
        new_rows.append(
            {
                "stage": "group",
                "group": str(row.group).strip().upper(),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
        added.append(f"{row.home_team} {home_score}-{away_score} {row.away_team} (Grp {row.group})")

    if not new_rows:
        return []

    new_frame = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_frame], ignore_index=True) if not existing.empty else new_frame
    combined = combined[RESULTS_COLUMNS]
    combined.to_csv(RESULTS_2026_FILE, index=False)
    return added


def commit_and_push(added: list[str]) -> None:
    rel = str(RESULTS_2026_FILE.relative_to(PROJECT_ROOT))
    subprocess.run(["git", "add", rel], cwd=str(PROJECT_ROOT), check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", rel], cwd=str(PROJECT_ROOT), capture_output=True, text=True
    )
    if not status.stdout.strip():
        return
    summary = added[0] if len(added) == 1 else f"{len(added)} new match results"
    message = f"Auto-record {summary}"
    subprocess.run(["git", "commit", "-m", message], cwd=str(PROJECT_ROOT), check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=str(PROJECT_ROOT), check=True)
    log("Committed and pushed results log.")


def run_cycle(commit: bool) -> None:
    log("Refreshing schedule and regenerating predictions...")
    regenerate_predictions()
    added = sync_results_log()
    if added:
        log(f"Recorded {len(added)} new result(s):")
        for line in added:
            log(f"  + {line}")
        if commit:
            commit_and_push(added)
    else:
        log("No new results.")


def main() -> None:
    args = parse_args()
    if not args.watch:
        run_cycle(args.commit)
        return

    log(f"Watching every {args.interval}s (Ctrl+C to stop).")
    try:
        while True:
            run_cycle(args.commit)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("Stopped.")


if __name__ == "__main__":
    main()
