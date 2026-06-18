import pandas as pd

from worldcup2026.config import TEAM_CONTEXT_FILE, TEAMS_2026_FILE


def test_2026_teams_have_camp_coordinates():
    teams = pd.read_csv(TEAMS_2026_FILE)
    context = pd.read_csv(TEAM_CONTEXT_FILE)

    context_by_team = context.set_index("team")
    missing_teams = sorted(set(teams["team"]) - set(context_by_team.index))
    assert missing_teams == []

    required_columns = ["camp_city", "camp_latitude", "camp_longitude", "camp_utc_offset"]
    missing_camps = []
    for team in sorted(teams["team"]):
        row = context_by_team.loc[team]
        if row[required_columns].isna().any() or any(str(row[column]).strip() == "" for column in required_columns):
            missing_camps.append(team)

    assert missing_camps == []


def test_camp_coordinates_are_plausible_for_host_region():
    context = pd.read_csv(TEAM_CONTEXT_FILE)

    assert context["camp_latitude"].between(18.0, 50.0).all()
    assert context["camp_longitude"].between(-125.0, -70.0).all()
    assert context["camp_utc_offset"].between(-7, -4).all()
