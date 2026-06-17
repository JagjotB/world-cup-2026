import pandas as pd

from worldcup2026.lineups import project_starting_lineup


def test_project_starting_lineup_returns_eleven_with_goalkeeper():
    rows = []
    for idx in range(26):
        position = "GK" if idx < 3 else "DF" if idx < 11 else "MF" if idx < 18 else "FW"
        rows.append(
            {
                "team": "Example",
                "squad_number": idx + 1,
                "position": position,
                "player_name": f"Player {idx}",
                "club": "Club",
                "height_cm": 180 + idx % 10,
                "international_caps": 26 - idx,
                "international_goals": idx % 5,
                "age_on_2026_06_11": 26,
                "has_club_stats": True,
                "min": 2000 - idx,
                "starts": 20,
                "gls": idx % 6,
                "ast": idx % 4,
            }
        )

    lineup = project_starting_lineup(pd.DataFrame(rows), "Example")

    assert len(lineup) == 11
    assert (lineup["lineup_position"] == "GK").sum() == 1


def test_project_starting_lineup_honors_manual_starter_and_excludes_out_player():
    rows = []
    for idx in range(14):
        position = "GK" if idx < 2 else "DF" if idx < 7 else "MF" if idx < 10 else "FW"
        rows.append(
            {
                "team": "Example",
                "squad_number": idx + 1,
                "position": position,
                "player_name": f"Player {idx}",
                "club": "Club",
                "height_cm": 180,
                "international_caps": 5,
                "international_goals": 0,
                "age_on_2026_06_11": 26,
                "has_club_stats": True,
                "min": 1000,
                "starts": 10,
                "gls": 0,
                "ast": 0,
                "availability_status": "available",
                "availability_multiplier": 1.0,
                "expected_minutes_share": 1.0,
                "lineup_status": "",
            }
        )
    rows[12]["availability_status"] = "out"
    rows[10]["manual_lineup_slot"] = 1
    rows[10]["manual_lineup_position"] = "FW"
    rows[10]["lineup_status"] = "confirmed"

    lineup = project_starting_lineup(pd.DataFrame(rows), "Example")

    assert "Player 10" in set(lineup["player_name"])
    assert "Player 12" not in set(lineup["player_name"])
