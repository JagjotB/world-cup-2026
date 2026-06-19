from __future__ import annotations

import json
import os
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


def _search_google_news_rss(home_team: str, away_team: str) -> str:
    """Google News RSS — robot-friendly, returns article titles and snippets."""
    import xml.etree.ElementTree as ET
    queries = [
        f"{home_team} {away_team} World Cup 2026",
        f"{home_team} injury World Cup 2026",
        f"{away_team} injury World Cup 2026",
    ]
    texts: list[str] = []
    for query in queries:
        try:
            url = (
                "https://news.google.com/rss/search"
                f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            if response.ok:
                root = ET.fromstring(response.text)
                for item in root.iter("item"):
                    title = item.findtext("title") or ""
                    desc = item.findtext("description") or ""
                    if title:
                        texts.append(title)
                    if desc:
                        from bs4 import BeautifulSoup
                        texts.append(BeautifulSoup(desc, "html.parser").get_text(" ", strip=True))
        except Exception:
            pass
    return " ".join(texts)


def _search_bing_news(home_team: str, away_team: str) -> str:
    """Bing News search — often less aggressive about blocking scrapers."""
    query = f"{home_team} {away_team} World Cup 2026 injury preview"
    try:
        url = f"https://www.bing.com/news/search?q={requests.utils.quote(query)}&format=rss"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        if response.ok and "<rss" in response.text:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            texts = []
            for item in root.iter("item"):
                title = item.findtext("title") or ""
                desc = item.findtext("description") or ""
                if title:
                    texts.append(title)
                if desc:
                    from bs4 import BeautifulSoup
                    texts.append(BeautifulSoup(desc, "html.parser").get_text(" ", strip=True))
            return " ".join(texts)
    except Exception:
        pass
    return ""


def fetch_match_news_text(home_team: str, away_team: str, date: str) -> str:
    text = _search_google_news_rss(home_team, away_team)
    if len(text) < 200:
        text = _search_bing_news(home_team, away_team)
    return text[:6000]


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
        # Free fallback: keyword proximity matching — no API cost
        if text.strip():
            return extract_signals_with_keywords(home_team, away_team, text)
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


_INJURY_KEYWORDS = [
    "injured", "injury", "doubtful", "ruled out", "fitness doubt",
    "fitness concern", "fitness test", "sidelined", "unavailable",
    "won't play", "will not play", "misses out", "withdrawn",
    "hamstring", "ankle injury", "knee injury", "calf injury",
    "thigh injury", "groin injury", "muscle injury",
    "limping", "substituted off injured",
]
_POSITIVE_KEYWORDS = [
    "confident", "motivated", "in great form", "great form",
    "unbeaten", "winning run", "momentum", "fired up", "raring to go",
    "full strength", "all available", "good spirits",
    "sharp", "clinical", "dominant", "impressive run",
]
_NEGATIVE_KEYWORDS = [
    "under pressure", "poor form", "struggling",
    "crisis", "disarray", "turmoil", "unhappy",
    "frustration", "disappointing form", "no wins",
]
_FATIGUE_KEYWORDS = [
    "tired", "fatigue", "tired legs", "heavy schedule",
    "jet lag", "load management", "overworked",
    "fourth game", "fifth game", "three games in",
]
_UPSET_KEYWORDS = [
    "surprise", "shock result", "upset", "underdog",
    "nothing to lose", "giant killing", "dark horse",
]


def _mentions(text_lower: str, team: str) -> list[int]:
    """Character positions of all team name occurrences in text."""
    positions: list[int] = []
    term = team.lower()
    pos = 0
    while True:
        idx = text_lower.find(term, pos)
        if idx == -1:
            break
        positions.append(idx)
        pos = idx + 1
    return positions


def _proximity_score(
    text_lower: str,
    positions: list[int],
    keywords: list[str],
    window: int = 250,
) -> float:
    """
    Returns a 0-1 score based on how many distinct keywords appear
    within *window* characters of any team name mention.
    """
    if not positions:
        return 0.0
    hit_keywords: set[str] = set()
    for kw in keywords:
        kw_pos = 0
        while True:
            idx = text_lower.find(kw, kw_pos)
            if idx == -1:
                break
            if any(abs(idx - tp) <= window for tp in positions):
                hit_keywords.add(kw)
                break
            kw_pos = idx + 1
    return min(len(hit_keywords) / max(len(keywords) * 0.25, 1), 1.0)


def extract_signals_with_keywords(
    home_team: str,
    away_team: str,
    text: str,
) -> dict:
    """
    Extract match signals from scraped news text using keyword proximity matching.
    Free — no API key required.
    """
    tl = text.lower()
    home_pos = _mentions(tl, home_team)
    away_pos = _mentions(tl, away_team)

    home_injury = _proximity_score(tl, home_pos, _INJURY_KEYWORDS)
    away_injury = _proximity_score(tl, away_pos, _INJURY_KEYWORDS)
    home_pos_s = _proximity_score(tl, home_pos, _POSITIVE_KEYWORDS)
    away_pos_s = _proximity_score(tl, away_pos, _POSITIVE_KEYWORDS)
    home_neg_s = _proximity_score(tl, home_pos, _NEGATIVE_KEYWORDS)
    away_neg_s = _proximity_score(tl, away_pos, _NEGATIVE_KEYWORDS)
    home_fatigue = _proximity_score(tl, home_pos, _FATIGUE_KEYWORDS)
    away_fatigue = _proximity_score(tl, away_pos, _FATIGUE_KEYWORDS)

    home_morale = float(max(0.0, min(1.0, 0.5 + 0.35 * home_pos_s - 0.35 * home_neg_s)))
    away_morale = float(max(0.0, min(1.0, 0.5 + 0.35 * away_pos_s - 0.35 * away_neg_s)))

    # Upset risk: direct upset keywords + away underdog having fewer injury concerns
    upset_base = _proximity_score(tl, home_pos + away_pos, _UPSET_KEYWORDS)
    upset_risk = float(min(upset_base * 1.4 + (away_injury > 0.3) * 0.15, 1.0))

    # Narrative edge: which team has more positive proximal keywords
    if home_pos_s > away_pos_s + 0.1:
        narrative_edge = "home"
    elif away_pos_s > home_pos_s + 0.1:
        narrative_edge = "away"
    else:
        narrative_edge = "neutral"

    # Confidence: driven by text length and how often teams are mentioned
    total_mentions = len(home_pos) + len(away_pos)
    confidence = float(min((total_mentions / 8) * (len(text) / 600), 1.0))

    key_signals: list[str] = []
    if home_injury > 0.25:
        key_signals.append(f"{home_team} injury concerns")
    if away_injury > 0.25:
        key_signals.append(f"{away_team} injury concerns")
    if narrative_edge != "neutral":
        key_signals.append(f"Narrative favors {narrative_edge} team")
    if upset_risk > 0.35:
        key_signals.append("Upset possible")

    return {
        "home_injury_concern":   float(min(home_injury, 1.0)),
        "away_injury_concern":   float(min(away_injury, 1.0)),
        "home_morale":           home_morale,
        "away_morale":           away_morale,
        "home_fatigue":          float(min(home_fatigue, 1.0)),
        "away_fatigue":          float(min(away_fatigue, 1.0)),
        "home_manager_confidence": 0.5,
        "away_manager_confidence": 0.5,
        "narrative_edge":        narrative_edge,
        "upset_risk":            upset_risk,
        "confidence":            confidence,
        "key_signals":           key_signals[:3],
    }


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
        "news_injury_edge": (
            float(signals.get("away_injury_concern", 0.1))
            - float(signals.get("home_injury_concern", 0.1))
        ),
        "news_home_morale": float(signals.get("home_morale", 0.5)),
        "news_away_morale": float(signals.get("away_morale", 0.5)),
        "news_morale_edge": (
            float(signals.get("home_morale", 0.5))
            - float(signals.get("away_morale", 0.5))
        ),
        "news_home_fatigue": float(signals.get("home_fatigue", 0.1)),
        "news_away_fatigue": float(signals.get("away_fatigue", 0.1)),
        "news_fatigue_edge": (
            float(signals.get("away_fatigue", 0.1))
            - float(signals.get("home_fatigue", 0.1))
        ),
        "news_home_confidence": float(signals.get("home_manager_confidence", 0.5)),
        "news_away_confidence": float(signals.get("away_manager_confidence", 0.5)),
        "news_confidence_edge": (
            float(signals.get("home_manager_confidence", 0.5))
            - float(signals.get("away_manager_confidence", 0.5))
        ),
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
