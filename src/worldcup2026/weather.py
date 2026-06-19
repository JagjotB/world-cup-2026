"""
Match weather fetching via Open-Meteo (free, no API key, historical + forecast).

Historical archive: archive-api.open-meteo.com  (any past date)
Forecast:          api.open-meteo.com            (up to ~16 days out)

Weather effects on goals (empirical football research):
  - High temperature (>28C): fatigue suppresses second-half scoring ~8%
  - Heavy rain (>5mm):       slippery pitch reduces total goals ~6%
  - Strong wind (>30 km/h):  disrupts passing/shooting, reduces goals ~5%
  - Humidity (>80%):         amplifies heat fatigue, additional ~4% second-half drop
"""
from __future__ import annotations

import time
from datetime import date as date_type
from pathlib import Path

import pandas as pd
import requests

from .config import MANUAL_DIR

WEATHER_CACHE_FILE = MANUAL_DIR / "match_weather_2026.csv"

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 world-cup-2026 predictor research"}

TODAY = date_type.today()

# Venue city fragment -> (city_name, latitude, longitude)
VENUE_COORDS: dict[str, tuple[str, float, float]] = {
    "Mexico City":     ("Mexico City",   19.43,  -99.13),
    "Zapopan":         ("Guadalajara",   20.65, -103.35),
    "Guadalupe":       ("Monterrey",     25.67, -100.31),
    "Guadalajara":     ("Guadalajara",   20.65, -103.35),
    "Monterrey":       ("Monterrey",     25.67, -100.31),
    "Toronto":         ("Toronto",       43.65,  -79.38),
    "Inglewood":       ("Los Angeles",   34.05, -118.24),
    "Santa Clara":     ("San Jose",      37.34, -121.89),
    "East Rutherford": ("New York",      40.75,  -74.00),
    "Foxborough":      ("Boston",        42.36,  -71.06),
    "Vancouver":       ("Vancouver",     49.25, -123.12),
    "Houston":         ("Houston",       29.76,  -95.37),
    "Arlington":       ("Dallas",        32.78,  -96.80),
    "Philadelphia":    ("Philadelphia",  39.95,  -75.17),
    "Atlanta":         ("Atlanta",       33.75,  -84.39),
    "Seattle":         ("Seattle",       47.61, -122.33),
    "Miami Gardens":   ("Miami",         25.77,  -80.19),
    "Kansas City":     ("Kansas City",   39.10,  -94.58),
}


def _coords_from_venue(venue: str) -> tuple[str, float, float] | None:
    for fragment, info in VENUE_COORDS.items():
        if fragment in venue:
            return info
    return None


def _fetch_open_meteo(lat: float, lon: float, match_date: str) -> dict | None:
    """
    Fetch hourly weather for a single date via Open-Meteo.
    Uses historical archive for past dates, forecast API for future dates.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": match_date,
        "end_date": match_date,
        "hourly": "temperature_2m,precipitation,windspeed_10m,relativehumidity_2m",
        "timezone": "auto",
    }
    try:
        d = date_type.fromisoformat(match_date)
        base = (
            "https://archive-api.open-meteo.com/v1/archive"
            if d <= TODAY
            else "https://api.open-meteo.com/v1/forecast"
        )
        r = requests.get(base, params=params, headers=REQUEST_HEADERS, timeout=15)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def _parse_open_meteo(data: dict, kickoff_hour: int = 18) -> dict:
    """Extract weather at the kickoff hour from Open-Meteo hourly response."""
    try:
        hourly = data["hourly"]
        times = hourly["time"]
        temps = hourly["temperature_2m"]
        precips = hourly["precipitation"]
        winds = hourly["windspeed_10m"]
        humids = hourly["relativehumidity_2m"]

        # Find index closest to kickoff_hour
        best_idx = min(
            range(len(times)),
            key=lambda i: abs(int(times[i].split("T")[1].split(":")[0]) - kickoff_hour),
        )
        return {
            "temp_c": round(float(temps[best_idx] or 20), 1),
            "feels_like_c": round(float(temps[best_idx] or 20), 1),
            "precip_mm": round(float(precips[best_idx] or 0), 1),
            "humidity_pct": round(float(humids[best_idx] or 60), 1),
            "wind_kmh": round(float(winds[best_idx] or 10), 1),
            "weather_desc": _describe(
                float(temps[best_idx] or 20),
                float(precips[best_idx] or 0),
                float(winds[best_idx] or 10),
            ),
        }
    except Exception:
        return _default_weather()


def _describe(temp: float, precip: float, wind: float) -> str:
    parts = []
    if precip >= 5:
        parts.append("Rain")
    elif precip >= 1:
        parts.append("Light rain")
    if temp >= 32:
        parts.append("Extreme heat")
    elif temp >= 28:
        parts.append("Hot")
    elif temp < 12:
        parts.append("Cold")
    if wind >= 40:
        parts.append("Strong wind")
    elif wind >= 25:
        parts.append("Windy")
    return ", ".join(parts) if parts else "Clear"


def _default_weather() -> dict:
    return {
        "temp_c": 20.0,
        "feels_like_c": 20.0,
        "precip_mm": 0.0,
        "humidity_pct": 60.0,
        "wind_kmh": 10.0,
        "weather_desc": "Unknown",
    }


def fetch_match_weather(venue: str, match_date: str, local_time: str = "18:00") -> dict:
    """Fetch weather for a venue on a given date."""
    info = _coords_from_venue(venue)
    if info is None:
        return {"city": venue, **_default_weather()}
    city, lat, lon = info
    try:
        kickoff_hour = int(local_time.split(":")[0])
    except Exception:
        kickoff_hour = 18
    data = _fetch_open_meteo(lat, lon, match_date)
    if data is None:
        return {"city": city, **_default_weather()}
    return {"city": city, **_parse_open_meteo(data, kickoff_hour)}


def weather_goal_multiplier(w: dict) -> dict:
    """
    Per-half goal-rate multipliers derived from weather conditions.

    Returns:
        first_half_mult, second_half_mult, label
    """
    temp = float(w.get("temp_c", 20))
    precip = float(w.get("precip_mm", 0))
    wind = float(w.get("wind_kmh", 10))
    humidity = float(w.get("humidity_pct", 60))

    first_mult = 1.0
    second_mult = 1.0
    notes: list[str] = []

    if temp >= 32:
        second_mult *= 0.88
        notes.append(f"extreme heat {temp:.0f}C")
    elif temp >= 28:
        second_mult *= 0.93
        notes.append(f"hot {temp:.0f}C")

    if humidity >= 80 and temp >= 25:
        second_mult *= 0.96
        notes.append(f"high humidity {humidity:.0f}%")

    if precip >= 10:
        first_mult *= 0.92
        second_mult *= 0.92
        notes.append(f"heavy rain {precip:.0f}mm")
    elif precip >= 4:
        first_mult *= 0.96
        second_mult *= 0.96
        notes.append(f"rain {precip:.0f}mm")

    if wind >= 40:
        first_mult *= 0.93
        second_mult *= 0.93
        notes.append(f"strong wind {wind:.0f}km/h")
    elif wind >= 25:
        first_mult *= 0.97
        second_mult *= 0.97
        notes.append(f"windy {wind:.0f}km/h")

    return {
        "first_half_mult": round(first_mult, 4),
        "second_half_mult": round(second_mult, 4),
        "label": ", ".join(notes) if notes else "normal conditions",
    }


# ---------------------------------------------------------------------------
# Batch fetch + cache
# ---------------------------------------------------------------------------

def load_weather_cache(path: Path | None = None) -> pd.DataFrame:
    p = path or WEATHER_CACHE_FILE
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def fetch_and_cache_weather(
    schedule: pd.DataFrame,
    path: Path | None = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch weather for every match in schedule and write to cache CSV."""
    p = path or WEATHER_CACHE_FILE
    existing = load_weather_cache(p)
    existing_keys: set[tuple] = set()
    if not existing.empty and "local_date" in existing.columns:
        existing_keys = set(zip(existing["local_date"], existing["home_team"], existing["away_team"]))

    rows: list[dict] = []
    for row in schedule.itertuples(index=False):
        key = (str(row.local_date), str(row.home_team), str(row.away_team))
        if key in existing_keys:
            continue
        venue = str(getattr(row, "venue", ""))
        match_date = str(row.local_date)
        local_time = str(getattr(row, "local_time", "18:00"))
        w = fetch_match_weather(venue, match_date, local_time)
        mults = weather_goal_multiplier(w)
        record = {
            "local_date": match_date,
            "home_team": str(row.home_team),
            "away_team": str(row.away_team),
            "venue": venue,
            **w,
            "first_half_mult": mults["first_half_mult"],
            "second_half_mult": mults["second_half_mult"],
            "weather_label": mults["label"],
        }
        rows.append(record)
        print(
            f"  {row.home_team} v {row.away_team} ({match_date})"
            f"  {w['temp_c']}C  {w['precip_mm']}mm  {w['wind_kmh']}km/h"
            f"  -> {mults['label']}"
        )
        time.sleep(delay)

    if rows:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        p.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(p, index=False)
        return combined
    return existing
