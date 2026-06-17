from __future__ import annotations

import unicodedata
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .config import (
    ALIASES_FILE,
    RAW_RESULTS_FILE,
    RAW_RESULTS_URL,
    RESULTS_2026_FILE,
    TEAMS_2026_FILE,
)


def download_historical_results(
    destination: Path = RAW_RESULTS_FILE,
    url: str = RAW_RESULTS_URL,
    force: bool = False,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination

    urlretrieve(url, destination)
    return destination


def load_historical_results(path: Path = RAW_RESULTS_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "country",
        "neutral",
    }
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Historical results are missing columns: {missing_text}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    return df.sort_values("date").reset_index(drop=True)


def load_teams(path: Path = TEAMS_2026_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"group", "seed", "team"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Teams file is missing columns: {missing_text}")

    df = df.copy()
    df["group"] = df["group"].astype(str).str.upper().str.strip()
    df["seed"] = df["seed"].astype(int)
    df["team"] = df["team"].astype(str).str.strip()
    return df.sort_values(["group", "seed"]).reset_index(drop=True)


def load_aliases(path: Path = ALIASES_FILE) -> dict[str, str]:
    if not path.exists():
        return {}

    df = pd.read_csv(path).fillna("")
    aliases: dict[str, str] = {}
    for row in df.itertuples(index=False):
        tournament_name = str(row.tournament_name).strip()
        model_name = str(row.model_name).strip()
        if tournament_name and model_name:
            aliases[tournament_name] = model_name
    return aliases


def load_actual_results(path: Path = RESULTS_2026_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["stage", "group", "home_team", "away_team", "home_score", "away_score"]
        )

    df = pd.read_csv(path)
    if df.empty:
        return df

    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"]).copy()
    if df.empty:
        return df

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["group"] = df.get("group", "").fillna("").astype(str).str.upper().str.strip()
    df["stage"] = df.get("stage", "group").fillna("group").astype(str).str.lower().str.strip()
    return df


def canonicalize_team_name(name: str, aliases: dict[str, str] | None = None) -> str:
    clean = str(name).strip()
    if aliases and clean in aliases:
        return aliases[clean]
    return clean


def normalized_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold()
    return "".join(char for char in text if char.isalnum())
