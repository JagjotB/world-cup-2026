from __future__ import annotations

import argparse
import sys
import tempfile
import unicodedata
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from worldcup2026.config import (  # noqa: E402
    GROUP_STAGE_SCHEDULE_FILE,
    MODEL_FILE,
    OUTPUT_DIR,
    PLAYER_FEATURES_FILE,
)
from worldcup2026.group_stage import fetch_group_stage_schedule  # noqa: E402
from worldcup2026.lineups import load_player_features  # noqa: E402
from worldcup2026.live import apply_played_results_to_predictor  # noqa: E402
from worldcup2026.model import MatchPredictor  # noqa: E402
from worldcup2026.player_projections import (  # noqa: E402
    add_match_team_projection_totals,
    project_player_match_performances,
)

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"
)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 world-cup-2026 predictor"}

TEAM_ALIASES = {
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "cape verde islands": "Cape Verde",
    "congo dr": "DR Congo",
    "czech republic": "Czechia",
    "curacao": "Curacao",
    "cote d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "turkey": "Turkiye",
    "turkiye": "Turkiye",
    "usa": "United States",
    "united states": "United States",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 1+ corner in each half projections for played group-stage matches."
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE)
    parser.add_argument("--schedule", type=Path, default=GROUP_STAGE_SCHEDULE_FILE)
    parser.add_argument("--player-features", type=Path, default=PLAYER_FEATURES_FILE)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "played_corner_prop_evaluation.csv",
    )
    parser.add_argument("--refresh-schedule", action="store_true")
    parser.add_argument(
        "--scoreboard-dates",
        help="ESPN scoreboard date range in YYYYMMDD or YYYYMMDD-YYYYMMDD form.",
    )
    parser.add_argument(
        "--no-live-results",
        action="store_true",
        help="Do not update ratings/form after each evaluated match.",
    )
    return parser.parse_args()


def strip_accents(value: object) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", str(value))
        if not unicodedata.combining(char)
    )


def canonical_team(team: object) -> str:
    clean = " ".join(strip_accents(team).replace("\xa0", " ").split()).strip()
    return TEAM_ALIASES.get(clean.casefold(), clean)


def scoreboard_date_range(played: pd.DataFrame) -> str:
    dates = played["local_date"].dropna().astype(str)
    if dates.empty:
        return ""
    start = dates.min().replace("-", "")
    end = dates.max().replace("-", "")
    return start if start == end else f"{start}-{end}"


def fetch_espn_json(url: str, params: dict[str, object]) -> dict:
    response = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def completed_espn_event_ids(date_range: str) -> list[str]:
    scoreboard = fetch_espn_json(
        ESPN_SCOREBOARD_URL,
        {"dates": date_range, "limit": 100},
    )
    event_ids: list[str] = []
    for event in scoreboard.get("events", []):
        status = event.get("status", {}).get("type", {})
        if status.get("completed"):
            event_ids.append(str(event["id"]))
    return event_ids


def competitor_teams(summary: dict) -> tuple[str, str] | None:
    competitions = summary.get("header", {}).get("competitions", [])
    if not competitions:
        return None
    competitors = competitions[0].get("competitors", [])
    if len(competitors) < 2:
        return None
    by_home_away = {competitor.get("homeAway"): competitor for competitor in competitors}
    home = by_home_away.get("home") or competitors[0]
    away = by_home_away.get("away") or competitors[1]
    return canonical_team(home["team"]["displayName"]), canonical_team(
        away["team"]["displayName"]
    )


def total_corner_stats(summary: dict) -> dict[str, int]:
    totals: dict[str, int] = {}
    for team in summary.get("boxscore", {}).get("teams", []):
        team_name = canonical_team(team["team"]["displayName"])
        for stat in team.get("statistics", []):
            if stat.get("name") != "wonCorners":
                continue
            try:
                totals[team_name] = int(float(str(stat.get("displayValue", "")).strip()))
            except ValueError:
                pass
    return totals


def corner_counts_by_half(summary: dict) -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = {}
    for item in summary.get("commentary", []):
        play = item.get("play") or {}
        play_type = play.get("type") or {}
        if play_type.get("type") != "corner-awarded" and play_type.get("text") != "Corner Awarded":
            continue
        team = canonical_team((play.get("team") or {}).get("displayName", ""))
        period = (play.get("period") or {}).get("number")
        if period not in (1, 2):
            continue
        counts.setdefault(team, {1: 0, 2: 0})
        counts[team][period] += 1
    return counts


def fetch_actual_corner_props(date_range: str) -> dict[tuple[str, str], dict[str, object]]:
    actuals: dict[tuple[str, str], dict[str, object]] = {}
    for event_id in completed_espn_event_ids(date_range):
        summary = fetch_espn_json(ESPN_SUMMARY_URL, {"event": event_id})
        teams = competitor_teams(summary)
        if teams is None:
            continue
        home_team, away_team = teams
        half_counts = corner_counts_by_half(summary)
        total_corners = total_corner_stats(summary)
        actuals[(home_team, away_team)] = {
            "event_id": event_id,
            "home_first_half_corners": half_counts.get(home_team, {}).get(1, 0),
            "home_second_half_corners": half_counts.get(home_team, {}).get(2, 0),
            "away_first_half_corners": half_counts.get(away_team, {}).get(1, 0),
            "away_second_half_corners": half_counts.get(away_team, {}).get(2, 0),
            "home_total_corners": total_corners.get(home_team),
            "away_total_corners": total_corners.get(away_team),
        }
    return actuals


def sort_played_matches(played: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ["kickoff_utc", "local_date", "local_time", "group", "group_match_index"]
        if column in played.columns
    ]
    return played.sort_values(sort_columns).reset_index(drop=True)


def match_prediction_frame(predictor: MatchPredictor, row) -> pd.DataFrame:
    prediction = predictor.predict_match(row.home_team, row.away_team, neutral=True)
    return pd.DataFrame(
        [
            {
                "match_number": row.match_number,
                "group": row.group,
                "local_date": row.local_date,
                "local_time": row.local_time,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "p_home_win": prediction.p_home_win,
                "p_draw": prediction.p_draw,
                "p_away_win": prediction.p_away_win,
                "expected_home_goals": prediction.expected_home_goals,
                "expected_away_goals": prediction.expected_away_goals,
            }
        ]
    )


def result_row(row) -> dict[str, object]:
    return {
        "stage": "group",
        "group": row.group,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "home_score": int(float(row.home_score)),
        "away_score": int(float(row.away_score)),
    }


def add_team_rows(rows: list[dict[str, object]], projection, actual: dict[str, object], row) -> None:
    match_label = f"{row.home_team} vs {row.away_team}"
    for side in ["home", "away"]:
        team = row.home_team if side == "home" else row.away_team
        projected_first = projection[f"projected_{side}_first_half_corners"]
        projected_second = projection[f"projected_{side}_second_half_corners"]
        probability = projection[f"p_{side}_1plus_corners_each_half"]
        predicted = bool(projection[f"{side}_1plus_corners_each_half_pick"])
        actual_first = int(actual[f"{side}_first_half_corners"])
        actual_second = int(actual[f"{side}_second_half_corners"])
        actual_hit = actual_first >= 1 and actual_second >= 1
        rows.append(
            {
                "date": row.local_date,
                "group": row.group,
                "match": match_label,
                "team": team,
                "opponent": row.away_team if side == "home" else row.home_team,
                "predicted_1plus_corners_each_half": predicted,
                "probability_1plus_corners_each_half": probability,
                "projected_first_half_corners": projected_first,
                "projected_second_half_corners": projected_second,
                "actual_1plus_corners_each_half": actual_hit,
                "actual_first_half_corners": actual_first,
                "actual_second_half_corners": actual_second,
                "actual_total_corners": actual.get(f"{side}_total_corners"),
                "correct": predicted == actual_hit,
                "true_positive": predicted and actual_hit,
            }
        )


def main() -> None:
    args = parse_args()
    schedule = (
        fetch_group_stage_schedule(args.schedule)
        if args.refresh_schedule or not args.schedule.exists()
        else pd.read_csv(args.schedule)
    )
    played = sort_played_matches(schedule[schedule["status"].eq("played")].copy())
    if played.empty:
        raise SystemExit("No played group-stage matches found.")

    date_range = args.scoreboard_dates or scoreboard_date_range(played)
    actuals = fetch_actual_corner_props(date_range)
    predictor = MatchPredictor.load(args.model_path)
    players = load_player_features(args.player_features)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        player_projection_output = Path(temporary_directory) / "player_match_projections.csv"
        for match in played.itertuples(index=False):
            actual = actuals.get((canonical_team(match.home_team), canonical_team(match.away_team)))
            if actual is None:
                missing.append(f"{match.home_team} vs {match.away_team}")
            else:
                match_frame = match_prediction_frame(predictor, match)
                player_projections = project_player_match_performances(
                    predictor,
                    pd.DataFrame([match._asdict()]),
                    players,
                    output_path=player_projection_output,
                    upcoming_only=False,
                )
                projection = add_match_team_projection_totals(
                    match_frame,
                    player_projections,
                ).iloc[0]
                add_team_rows(rows, projection, actual, match)

            if not args.no_live_results:
                apply_played_results_to_predictor(
                    predictor,
                    pd.DataFrame([result_row(match)]),
                )

    evaluation = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(args.output, index=False)

    print(f"Wrote evaluation: {args.output}")
    print(f"Played matches evaluated: {len(evaluation) // 2}")
    print(f"Team entries evaluated: {len(evaluation)}")
    if missing:
        print(f"Missing actual corner events: {len(missing)}")
        print("\n".join(missing))
    if evaluation.empty:
        return

    true_positive_count = int(evaluation["true_positive"].sum())
    predicted_positive_count = int(evaluation["predicted_1plus_corners_each_half"].sum())
    actual_positive_count = int(evaluation["actual_1plus_corners_each_half"].sum())
    correct_count = int(evaluation["correct"].sum())
    print(
        "Correct yes-picks: "
        f"{true_positive_count}/{predicted_positive_count} "
        f"({true_positive_count / predicted_positive_count:.1%})"
        if predicted_positive_count
        else "Correct yes-picks: 0/0"
    )
    print(
        "Overall yes/no accuracy: "
        f"{correct_count}/{len(evaluation)} ({correct_count / len(evaluation):.1%})"
    )
    print(
        "Actual 1+ corner each half rate: "
        f"{actual_positive_count}/{len(evaluation)} ({actual_positive_count / len(evaluation):.1%})"
    )

    preview_columns = [
        "date",
        "match",
        "team",
        "predicted_1plus_corners_each_half",
        "probability_1plus_corners_each_half",
        "actual_1plus_corners_each_half",
        "actual_first_half_corners",
        "actual_second_half_corners",
        "correct",
    ]
    print(evaluation[preview_columns].to_string(index=False))


if __name__ == "__main__":
    main()
