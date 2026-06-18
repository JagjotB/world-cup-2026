from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import GROUP_STAGE_SCHEDULE_FILE, UPCOMING_GROUP_STAGE_PREDICTIONS_FILE

GROUP_PAGE_URLS = {
    group: f"https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{group}"
    for group in "ABCDEFGHIJKL"
}

SCHEDULE_TEAM_ALIASES = {
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte D'Ivoire": "Ivory Coast",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Turkey": "Turkiye",
    "Türkiye": "Turkiye",
    "USA": "United States",
}

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 world-cup-2026 predictor"}

DOUBLE_CHANCE_USEFUL_TOP_PROBABILITY = 0.45
BTTS_YES_MIN_TEAM_XG = 0.70

VENUE_CITY_UTC_OFFSETS = {
    "Arlington": -5,
    "Atlanta": -4,
    "East Rutherford": -4,
    "Foxborough": -4,
    "Guadalajara": -6,
    "Houston": -5,
    "Inglewood": -7,
    "Kansas City": -5,
    "Mexico City": -6,
    "Miami Gardens": -4,
    "Monterrey": -6,
    "Philadelphia": -4,
    "Santa Clara": -7,
    "Seattle": -7,
    "Toronto": -4,
    "Vancouver": -7,
}


def canonical_schedule_team(team: str) -> str:
    clean = " ".join(str(team).split()).strip()
    return SCHEDULE_TEAM_ALIASES.get(clean, clean)


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\xa0", " ").split()).strip()


def _parse_score_or_match_number(score_text: str) -> tuple[str, int | None, int | None, int | None]:
    clean = _clean_text(score_text)
    match_number_match = re.search(r"Match\s+(\d+)", clean, flags=re.IGNORECASE)
    if match_number_match:
        return "upcoming", int(match_number_match.group(1)), None, None

    score_match = re.search(r"(\d+)\s*[–-]\s*(\d+)", clean)
    if score_match:
        return "played", None, int(score_match.group(1)), int(score_match.group(2))

    return "unknown", None, None, None


def _infer_offset_hours(venue: str) -> int | None:
    for city, offset in VENUE_CITY_UTC_OFFSETS.items():
        if city in venue:
            return offset
    return None


def _parse_kickoff(date_text: str, time_text: str, venue: str) -> tuple[str, str, str, str]:
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", date_text)
    local_date = date_match.group(0) if date_match else ""

    time_clean = _clean_text(time_text)
    time_match = re.search(r"(\d{1,2}:\d{2})\s*([ap])\.?m\.?", time_clean, flags=re.IGNORECASE)
    offset_match = re.search(r"UTC\s*([+−-])\s*(\d{1,2})", time_clean)

    if not local_date or not time_match:
        return local_date, time_clean, "", ""

    hour, minute = [int(part) for part in time_match.group(1).split(":")]
    meridiem = time_match.group(2).lower()
    if meridiem == "p" and hour != 12:
        hour += 12
    if meridiem == "a" and hour == 12:
        hour = 0

    if offset_match:
        sign_text = offset_match.group(1)
        sign = -1 if sign_text in {"−", "-"} else 1
        offset_hours = sign * int(offset_match.group(2))
    else:
        offset_hours = _infer_offset_hours(venue)
        if offset_hours is None:
            return local_date, f"{hour:02d}:{minute:02d}", "", ""

    offset = timezone(timedelta(hours=offset_hours))
    local_dt = datetime.fromisoformat(local_date).replace(hour=hour, minute=minute, tzinfo=offset)

    return (
        local_date,
        f"{hour:02d}:{minute:02d}",
        f"UTC{offset_hours:+03d}:00",
        local_dt.astimezone(timezone.utc).isoformat(),
    )


def _parse_venue(box) -> str:
    location = box.select_one(".fright [itemprop='name address']")
    if location is None:
        location = box.select_one(".fright")
    return _clean_text(location.get_text(" ", strip=True)) if location else ""


def fetch_group_stage_schedule(
    output_path: Path = GROUP_STAGE_SCHEDULE_FILE,
    page_urls: dict[str, str] | None = None,
) -> pd.DataFrame:
    page_urls = page_urls or GROUP_PAGE_URLS
    rows: list[dict[str, object]] = []

    for group, url in page_urls.items():
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for group_match_index, box in enumerate(soup.select(".footballbox"), start=1):
            date_text = _clean_text(box.select_one(".fdate").get_text(" ", strip=True))
            time_text = _clean_text(box.select_one(".ftime").get_text(" ", strip=True))
            venue = _parse_venue(box)
            local_date, local_time, utc_offset, kickoff_utc = _parse_kickoff(
                date_text,
                time_text,
                venue,
            )

            home = canonical_schedule_team(box.select_one(".fhome").get_text(" ", strip=True))
            away = canonical_schedule_team(box.select_one(".faway").get_text(" ", strip=True))
            score_text = _clean_text(box.select_one(".fscore").get_text(" ", strip=True))
            status, match_number, home_score, away_score = _parse_score_or_match_number(score_text)

            rows.append(
                {
                    "group": group,
                    "group_match_index": group_match_index,
                    "match_number": match_number,
                    "status": status,
                    "local_date": local_date,
                    "local_time": local_time,
                    "utc_offset": utc_offset,
                    "kickoff_utc": kickoff_utc,
                    "home_team": home,
                    "away_team": away,
                    "score_text": score_text,
                    "home_score": home_score,
                    "away_score": away_score,
                    "venue": venue,
                    "source_url": url,
                }
            )

    schedule = pd.DataFrame(rows).sort_values(["kickoff_utc", "group", "group_match_index"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(output_path, index=False)
    return schedule.reset_index(drop=True)


def predict_group_stage_matches(
    predictor,
    schedule: pd.DataFrame,
    output_path: Path = UPCOMING_GROUP_STAGE_PREDICTIONS_FILE,
    upcoming_only: bool = True,
    from_date: str | None = None,
) -> pd.DataFrame:
    rows = []
    matches = schedule[schedule["status"].eq("upcoming")].copy() if upcoming_only else schedule.copy()
    if from_date:
        matches = matches[matches["local_date"].ge(from_date)].copy()

    for row in matches.itertuples(index=False):
        prediction = predictor.predict_match(row.home_team, row.away_team, neutral=True)
        decision = predictor.decision_for_prediction(prediction)
        rows.append(
            {
                "match_number": row.match_number,
                "group": row.group,
                "local_date": row.local_date,
                "local_time": row.local_time,
                "utc_offset": row.utc_offset,
                "kickoff_utc": row.kickoff_utc,
                "venue": row.venue,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "predicted_result": decision.recommended_result,
                "raw_top_result": decision.raw_top_result,
                "pick_confidence": decision.confidence,
                "top_probability": decision.top_probability,
                "runner_up_probability": decision.runner_up_probability,
                "probability_margin": decision.probability_margin,
                "draw_override_applied": decision.draw_override_applied,
                "p_home_win": prediction.p_home_win,
                "p_draw": prediction.p_draw,
                "p_away_win": prediction.p_away_win,
                "expected_home_goals": prediction.expected_home_goals,
                "expected_away_goals": prediction.expected_away_goals,
                "source_url": row.source_url,
            }
        )

    predictions = pd.DataFrame(rows).sort_values(["kickoff_utc", "match_number"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions.reset_index(drop=True)


def add_match_usefulness_filters(matches: pd.DataFrame) -> pd.DataFrame:
    filtered = matches.copy()
    if filtered.empty:
        return filtered

    required_columns = {
        "home_team",
        "away_team",
        "predicted_result",
        "top_probability",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "expected_home_goals",
        "expected_away_goals",
    }
    if not required_columns.issubset(filtered.columns):
        return filtered

    double_chance_picks: list[str] = []
    double_chance_probabilities: list[float] = []
    double_chance_useful: list[bool] = []
    either_team_wins_probabilities: list[float] = []
    btts_picks: list[str] = []
    btts_useful: list[bool] = []
    for row in filtered.itertuples(index=False):
        row_dict = row._asdict()
        home_team = str(row_dict["home_team"])
        away_team = str(row_dict["away_team"])
        predicted_result = str(row_dict["predicted_result"])
        p_home = float(row_dict["p_home_win"])
        p_draw = float(row_dict["p_draw"])
        p_away = float(row_dict["p_away_win"])
        top_probability = float(row_dict["top_probability"])
        either_team_wins_probabilities.append(round(p_home + p_away, 4))

        if predicted_result == home_team:
            double_chance_picks.append(f"{home_team} or Draw")
            double_chance_probabilities.append(round(p_home + p_draw, 4))
        elif predicted_result == away_team:
            double_chance_picks.append(f"{away_team} or Draw")
            double_chance_probabilities.append(round(p_away + p_draw, 4))
        else:
            non_draw = home_team if p_home >= p_away else away_team
            non_draw_probability = p_home if p_home >= p_away else p_away
            double_chance_picks.append(f"{non_draw} or Draw")
            double_chance_probabilities.append(round(non_draw_probability + p_draw, 4))
        double_chance_useful.append(top_probability >= DOUBLE_CHANCE_USEFUL_TOP_PROBABILITY)

        home_xg = float(row_dict["expected_home_goals"])
        away_xg = float(row_dict["expected_away_goals"])
        if home_xg >= BTTS_YES_MIN_TEAM_XG and away_xg >= BTTS_YES_MIN_TEAM_XG:
            btts_picks.append("Yes")
            btts_useful.append(True)
        else:
            btts_picks.append("No play")
            btts_useful.append(False)

    filtered["double_chance_pick"] = double_chance_picks
    filtered["double_chance_probability"] = double_chance_probabilities
    filtered["double_chance_useful"] = double_chance_useful
    filtered["either_team_wins_pick"] = "Either team wins / No draw"
    filtered["either_team_wins_probability"] = either_team_wins_probabilities
    filtered["either_team_wins_useful"] = False
    filtered["btts_pick"] = btts_picks
    filtered["btts_useful"] = btts_useful
    filtered["btts_min_team_xg"] = filtered[
        ["expected_home_goals", "expected_away_goals"]
    ].min(axis=1)
    return filtered
