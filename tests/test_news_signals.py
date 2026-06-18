import pandas as pd

from worldcup2026.news_signals import add_news_signal_columns, load_news_signals


def test_load_news_signals_reads_cached_rows(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text(
        "local_date,home_team,away_team,news_home_injury_concern,news_away_injury_concern,"
        "news_injury_edge,news_home_morale,news_away_morale,news_morale_edge,"
        "news_home_fatigue,news_away_fatigue,news_fatigue_edge,news_home_confidence,"
        "news_away_confidence,news_confidence_edge,news_narrative_home,news_narrative_away,"
        "news_upset_risk,news_signal_confidence\n"
        "2026-06-18,Home,Away,0.2,0.4,0.2,0.7,0.5,0.2,0.1,0.3,0.2,0.8,0.4,0.4,1.0,0.0,0.25,0.9\n",
        encoding="utf-8",
    )

    signals = load_news_signals(path)

    assert ("2026-06-18", "home", "away") in signals
    assert signals[("2026-06-18", "home", "away")]["news_signal_confidence"] == 0.9


def test_add_news_signal_columns_flips_reversed_fixture(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text(
        "local_date,home_team,away_team,news_home_injury_concern,news_away_injury_concern,"
        "news_injury_edge,news_home_morale,news_away_morale,news_morale_edge,"
        "news_home_fatigue,news_away_fatigue,news_fatigue_edge,news_home_confidence,"
        "news_away_confidence,news_confidence_edge,news_narrative_home,news_narrative_away,"
        "news_upset_risk,news_signal_confidence\n"
        "2026-06-18,Home,Away,0.2,0.4,0.2,0.7,0.5,0.2,0.1,0.3,0.2,0.8,0.4,0.4,1.0,0.0,0.25,0.9\n",
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        [
            {
                "local_date": "2026-06-18",
                "home_team": "Away",
                "away_team": "Home",
            }
        ]
    )

    enriched = add_news_signal_columns(predictions, path)
    row = enriched.iloc[0]

    assert row["news_home_injury_concern"] == 0.4
    assert row["news_away_injury_concern"] == 0.2
    assert row["news_injury_edge"] == -0.2
    assert row["news_narrative_home"] == 0.0
    assert row["news_narrative_away"] == 1.0
