from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from .config import NEWS_SIGNALS_FILE
from .data import normalized_key

NEWS_SIGNALS_SCHEMA = {
    "home_injury_concern": "float 0-1: probability a key home player is doubtful/injured",
    "away_injury_concern": "float 0-1: probability a key away player is doubtful/injured",
    "home_morale": "float 0-1: home team squad morale (0=low, 0.5=neutral, 1=high)",
    "away_morale": "float 0-1: away team squad morale",
    "home_fatigue": "float 0-1: mentions of tiredness/heavy schedule for home team",
    "away_fatigue": "float 0-1: mentions of tiredness/heavy schedule for away team",
    "home_manager_confidence": "float 0-1: manager press conference tone (0=cautious, 1=confident)",
    "away_manager_confidence": "float 0-1: manager press conference tone",
    "narrative_edge": "string: 'home', 'away', or 'neutral' — which team the pre-match narrative favors",
    "upset_risk": "float 0-1: probability narrative suggests an upset is possible",
    "confidence": "float 0-1: how much useful text was available (0=no text, 1=rich coverage)",
    "key_signals": "list of strings: up to 3 most important signals found in the text",
}

EXTRACT_PROMPT = """You are a football (soccer) analyst extracting pre-match signals from news text.

Match: {home_team} vs {away_team} on {date}

News text:
---
{text}
---

Extract the following signals as a JSON object. Use null for any signal you cannot determine from the text.

Schema:
{schema}

Return ONLY valid JSON, no other text."""

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 world-cup-2026 predictor research"}


def _search_bbc_sport(home_team: str, away_team: str) -> str:
    """Try to fetch BBC Sport match preview text."""
    query = f"{home_team} {away_team} World Cup 2026"
    try:
        url = f"https://www.bbc.co.uk/sport/football/search?q={query.replace(' ', '+')}"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        if response.ok:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
            return " ".join(paragraphs[:30])
    except Exception:
        pass
    return ""


def _search_google_news(home_team: str, away_team: str, date: str) -> str:
    """Try to fetch Google News snippets for the match preview."""
    query = f'"{home_team}" "{away_team}" World Cup preview {date}'
    try:
        url = f"https://news.google.com/search?q={requests.utils.quote(query)}&hl=en"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        if response.ok:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            snippets = [a.get_text(" ", strip=True) for a in soup.select("article h3, article a")]
            return " ".join(snippets[:40])
    except Exception:
        pass
    return ""


def fetch_match_news_text(home_team: str, away_team: str, date: str) -> str:
    text = _search_bbc_sport(home_team, away_team)
    if len(text) < 200:
        text = _search_google_news(home_team, away_team, date)
    return text[:4000]


def extract_signals_with_llm(
    home_team: str,
    away_team: str,
    date: str,
    text: str,
    api_key: str | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _default_signals()

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        schema_str = json.dumps(NEWS_SIGNALS_SCHEMA, indent=2)
        prompt = EXTRACT_PROMPT.format(
            home_team=home_team,
            away_team=away_team,
            date=date,
            text=text if text.strip() else "(no text available)",
            schema=schema_str,
        )
        message = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        signals = json.loads(raw)
        return _normalise_signals(signals)
    except Exception:
        return _default_signals()


def _default_signals() -> dict:
    return {
        "home_injury_concern": 0.1,
        "away_injury_concern": 0.1,
        "home_morale": 0.5,
        "away_morale": 0.5,
        "home_fatigue": 0.1,
        "away_fatigue": 0.1,
        "home_manager_confidence": 0.5,
        "away_manager_confidence": 0.5,
        "narrative_edge": "neutral",
        "upset_risk": 0.2,
        "confidence": 0.0,
        "key_signals": [],
    }


def _normalise_signals(raw: dict) -> dict:
    defaults = _default_signals()
    result = {}
    for key, default in defaults.items():
        val = raw.get(key, None)
        if val is None:
            result[key] = default
        elif isinstance(default, float):
            try:
                result[key] = float(max(0.0, min(1.0, float(val))))
            except (TypeError, ValueError):
                result[key] = default
        else:
            result[key] = val
    return result


def build_news_feature_row(signals: dict, home_team: str, away_team: str) -> dict:
    """Convert extracted signals to model-ready float features."""
    narrative_home = 1.0 if signals.get("narrative_edge") == "home" else 0.0
    narrative_away = 1.0 if signals.get("narrative_edge") == "away" else 0.0
    return {
        "news_home_injury_concern": float(signals.get("home_injury_concern", 0.1)),
        "news_away_injury_concern": float(signals.get("away_injury_concern", 0.1)),
        "news_injury_edge": float(signals.get("away_injury_concern", 0.1)) - float(signals.get("home_injury_concern", 0.1)),
        "news_home_morale": float(signals.get("home_morale", 0.5)),
        "news_away_morale": float(signals.get("away_morale", 0.5)),
        "news_morale_edge": float(signals.get("home_morale", 0.5)) - float(signals.get("away_morale", 0.5)),
        "news_home_fatigue": float(signals.get("home_fatigue", 0.1)),
        "news_away_fatigue": float(signals.get("away_fatigue", 0.1)),
        "news_fatigue_edge": float(signals.get("away_fatigue", 0.1)) - float(signals.get("home_fatigue", 0.1)),
        "news_home_confidence": float(signals.get("home_manager_confidence", 0.5)),
        "news_away_confidence": float(signals.get("away_manager_confidence", 0.5)),
        "news_confidence_edge": float(signals.get("home_manager_confidence", 0.5)) - float(signals.get("away_manager_confidence", 0.5)),
        "news_narrative_home": narrative_home,
        "news_narrative_away": narrative_away,
        "news_upset_risk": float(signals.get("upset_risk", 0.2)),
        "news_signal_confidence": float(signals.get("confidence", 0.0)),
    }


NEWS_FEATURE_COLUMNS = list(build_news_feature_row(_default_signals(), "", "").keys())

def load_news_signals(path: Path | None = None) -> dict[tuple[str, str, str], dict]:
    """Load cached news signals keyed by (date, home_team, away_team)."""
    p = path or NEWS_SIGNALS_FILE
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    lookup = {}
    for row in df.itertuples(index=False):
        key = (
            str(row.local_date),
            normalized_key(row.home_team),
            normalized_key(row.away_team),
        )
        lookup[key] = {col: getattr(row, col, None) for col in NEWS_FEATURE_COLUMNS}
    return lookup


def flip_news_feature_row(signals: dict, defaults: dict) -> dict:
    flipped = dict(defaults)
    flipped.update(signals)
    pairs = [
        ("news_home_injury_concern", "news_away_injury_concern"),
        ("news_home_morale", "news_away_morale"),
        ("news_home_fatigue", "news_away_fatigue"),
        ("news_home_confidence", "news_away_confidence"),
        ("news_narrative_home", "news_narrative_away"),
    ]
    for home_col, away_col in pairs:
        flipped[home_col], flipped[away_col] = flipped.get(away_col, defaults[away_col]), flipped.get(
            home_col,
            defaults[home_col],
        )
    for edge_col in [
        "news_injury_edge",
        "news_morale_edge",
        "news_fatigue_edge",
        "news_confidence_edge",
    ]:
        flipped[edge_col] = -float(flipped.get(edge_col, defaults[edge_col]))
    return flipped


def add_news_signal_columns(
    predictions: "pd.DataFrame",
    news_path: Path | None = None,
) -> "pd.DataFrame":
    lookup = load_news_signals(news_path)
    defaults = build_news_feature_row(_default_signals(), "", "")
    rows = []
    for row in predictions.itertuples(index=False):
        key = (
            str(row.local_date),
            normalized_key(row.home_team),
            normalized_key(row.away_team),
        )
        reverse_key = (key[0], key[2], key[1])
        signals = lookup.get(key)
        if signals is None and reverse_key in lookup:
            signals = flip_news_feature_row(lookup[reverse_key], defaults)
        if signals is None:
            signals = defaults
        rows.append({col: signals.get(col, defaults[col]) for col in NEWS_FEATURE_COLUMNS})
    news_df = pd.DataFrame(rows)
    return pd.concat([predictions.reset_index(drop=True), news_df], axis=1)
