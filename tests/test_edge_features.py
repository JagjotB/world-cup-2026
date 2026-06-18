import pandas as pd

from worldcup2026.edge_features import (
    EDGE_FEATURE_COLUMNS,
    add_edge_feature_columns,
    apply_edge_probability_adjustments,
)


def _team_features(team: str, strength: float) -> dict[str, object]:
    return {
        "team": team,
        "projected_lineup_club_stats_share": 0.45 + strength * 0.20,
        "top_11_availability_score": 0.50 + strength * 0.20,
        "projected_lineup_score_mean": 7.0 + strength * 3.0,
        "projected_lineup_caps_sum": 360.0 + strength * 260.0,
        "projection_starter_minutes": 760.0 + strength * 110.0,
        "tactics_available": 1.0,
        "top_11_crossing_score": 0.8 + strength * 0.5,
        "top_11_physical_score": 0.9 + strength * 0.3,
        "height_cm_mean": 179.0 + strength * 5.0,
        "projection_starter_assist_threat": 5.0 + strength * 4.0,
        "starting_def_defensive_score": 0.9 + strength * 0.6,
        "starting_gk_keeper_score": 0.8 + strength * 1.0,
        "starting_def_physical_score": 0.9 + strength * 0.4,
        "projection_starter_defensive_work": 20.0 + strength * 12.0,
        "starting_fw_shot_volume_score": 0.8 + strength * 0.5,
        "top_11_shot_volume_score": 0.8 + strength * 0.5,
        "starting_fw_attacking_score": 0.8 + strength * 0.5,
        "projection_starter_shot_threat": 10.0 + strength * 8.0,
        "projection_starter_goal_threat": 8.0 + strength * 6.0,
        "top_11_keeper_score": 0.1 + strength * 0.2,
        "projection_keeper_coverage": 0.8 + strength * 1.2,
        "tactic_pressing_intensity": 0.3 + strength * 0.4,
        "tactic_defensive_line": 0.3 + strength * 0.3,
        "top_11_ball_winning_score": 0.8 + strength * 0.6,
        "projection_defensive_work": 28.0 + strength * 12.0,
        "tactic_possession_style": 0.3 + strength * 0.4,
        "starting_mf_creativity_score": 0.8 + strength * 0.5,
        "top_11_creativity_score": 0.8 + strength * 0.4,
        "international_caps_mean": 24.0 + strength * 18.0,
        "bench_depth_score": 1.4 + strength * 3.0,
        "projection_bench_goal_threat": 0.8 + strength * 2.0,
        "projection_bench_assist_threat": 0.8 + strength * 1.8,
        "projection_bench_shot_threat": 1.0 + strength * 3.0,
        "tactic_transition_speed": 0.3 + strength * 0.4,
        "tactic_tempo": 0.3 + strength * 0.4,
        "projection_card_risk": 4.0 - strength,
        "top_11_discipline_risk": 0.25 - strength * 0.08,
    }


def test_edge_features_add_context_signal_and_flags():
    matches = pd.DataFrame(
        [
            {
                "local_date": "2026-06-18",
                "local_time": "20:00",
                "utc_offset": "UTC-04:00",
                "kickoff_utc": "2026-06-19T00:00:00+00:00",
                "venue": "BMO Field , Toronto",
                "home_team": "Home",
                "away_team": "Far Away",
                "projected_home_corners": 4.5,
                "projected_away_corners": 2.5,
                "projected_home_shots_on_target": 4.0,
                "projected_away_shots_on_target": 2.0,
            }
        ]
    )
    schedule = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "local_date": "2026-06-15",
                        "local_time": "20:00",
                        "kickoff_utc": "2026-06-16T00:00:00+00:00",
                        "home_team": "Home",
                        "away_team": "Previous",
                    },
                    {
                        "local_date": "2026-06-16",
                        "local_time": "20:00",
                        "kickoff_utc": "2026-06-17T00:00:00+00:00",
                        "home_team": "Other",
                        "away_team": "Far Away",
                    },
                ]
            ),
            matches,
        ],
        ignore_index=True,
        sort=False,
    )
    team_context = pd.DataFrame(
        [
            {
                "team": "Home",
                "country_code": "CAN",
                "base_latitude": 43.65,
                "base_longitude": -79.38,
                "home_utc_offset": -4,
                "heat_acclimation": 0.35,
                "altitude_acclimation": 0.10,
                "travel_resilience": 0.65,
                "crowd_support_base": 0.65,
            },
            {
                "team": "Far Away",
                "country_code": "JPN",
                "base_latitude": 35.68,
                "base_longitude": 139.76,
                "home_utc_offset": 9,
                "heat_acclimation": 0.55,
                "altitude_acclimation": 0.05,
                "travel_resilience": 0.45,
                "crowd_support_base": 0.45,
            },
        ]
    )
    venue_context = pd.DataFrame(
        [
            {
                "venue_key": "bmofieldtoronto",
                "venue": "BMO Field , Toronto",
                "country_code": "CAN",
                "city": "Toronto",
                "latitude": 43.63,
                "longitude": -79.42,
                "altitude_m": 76,
                "roof_type": "open",
                "surface": "grass",
                "avg_june_temp_c": 23,
                "avg_june_humidity": 0.63,
                "wind_exposure": 0.45,
            }
        ]
    )
    overrides = pd.DataFrame(
        [
            {
                "local_date": "2026-06-18",
                "home_team": "Home",
                "away_team": "Far Away",
                "referee_tempo": 0.6,
                "referee_card_strictness": 0.8,
                "weather_temp_c": 25,
                "weather_humidity": 0.65,
                "weather_wind_kph": 12,
                "home_crowd_boost": 0.05,
                "away_crowd_boost": 0.0,
            }
        ]
    )
    team_player_features = pd.DataFrame(
        [_team_features("Home", 1.0), _team_features("Far Away", 0.0)]
    )
    player_readiness_signals = pd.DataFrame(
        [
            {
                "local_date": "2026-06-18",
                "team": "Home",
                "player_name": "Home Player",
                "source_type": "whoop_public",
                "consent_status": "public",
                "include_signal": True,
                "confidence": 0.9,
                "recovery_score": 82,
                "sleep_hours": 8.1,
                "hrv_delta_pct": 12,
                "resting_hr_delta_pct": -4,
            },
            {
                "local_date": "2026-06-18",
                "team": "Far Away",
                "player_name": "Away Player",
                "source_type": "strava_public",
                "consent_status": "public",
                "include_signal": True,
                "confidence": 0.8,
                "recovery_score": 38,
                "sleep_hours": 5.9,
                "hrv_delta_pct": -8,
                "resting_hr_delta_pct": 6,
            },
        ]
    )

    enriched = add_edge_feature_columns(
        matches,
        schedule=schedule,
        team_context=team_context,
        venue_context=venue_context,
        match_context=overrides,
        team_player_features=team_player_features,
        player_readiness_signals=player_readiness_signals,
    )
    row = enriched.iloc[0]

    assert row["edge_away_travel_km"] > row["edge_home_travel_km"]
    assert row["edge_travel_fatigue_edge"] > 1.0
    assert row["edge_crowd_support_edge"] > 0.50
    assert row["edge_lineup_chemistry_edge"] > 0
    assert row["edge_readiness_edge"] > 0.8
    assert row["edge_home_readiness_samples"] == 1
    assert row["edge_away_readiness_samples"] == 1
    assert row["edge_total_signal_pick"] == "Home"
    assert "home_travel" in row["edge_flags"]
    assert "home_crowd" in row["edge_flags"]
    assert "home_readiness" in row["edge_flags"]


def test_edge_features_are_safe_without_optional_context():
    matches = pd.DataFrame(
        [
            {
                "local_date": "2026-06-18",
                "local_time": "12:00",
                "home_team": "Home",
                "away_team": "Away",
                "venue": "Unknown Venue",
            }
        ]
    )

    enriched = add_edge_feature_columns(matches)

    assert set(EDGE_FEATURE_COLUMNS).issubset(enriched.columns)
    assert enriched.iloc[0]["edge_total_signal_strength"] in {"thin", "useful", "strong"}


def test_edge_features_prefer_camp_location_for_travel():
    matches = pd.DataFrame(
        [
            {
                "local_date": "2026-06-18",
                "local_time": "20:00",
                "utc_offset": "UTC-04:00",
                "venue": "BMO Field , Toronto",
                "home_team": "Camped",
                "away_team": "Base Only",
            }
        ]
    )
    venue_context = pd.DataFrame(
        [
            {
                "venue_key": "bmofieldtoronto",
                "venue": "BMO Field , Toronto",
                "country_code": "CAN",
                "city": "Toronto",
                "latitude": 43.63,
                "longitude": -79.42,
                "altitude_m": 76,
                "roof_type": "open",
                "surface": "grass",
                "avg_june_temp_c": 23,
                "avg_june_humidity": 0.63,
                "wind_exposure": 0.45,
            }
        ]
    )
    team_context = pd.DataFrame(
        [
            {
                "team": "Camped",
                "country_code": "JPN",
                "base_latitude": 35.68,
                "base_longitude": 139.76,
                "home_utc_offset": 9,
                "camp_city": "Toronto",
                "camp_latitude": 43.66,
                "camp_longitude": -79.38,
                "camp_utc_offset": -4,
                "heat_acclimation": 0.55,
                "altitude_acclimation": 0.05,
                "travel_resilience": 0.45,
                "crowd_support_base": 0.45,
            },
            {
                "team": "Base Only",
                "country_code": "JPN",
                "base_latitude": 35.68,
                "base_longitude": 139.76,
                "home_utc_offset": 9,
                "camp_city": "",
                "camp_latitude": "",
                "camp_longitude": "",
                "camp_utc_offset": "",
                "heat_acclimation": 0.55,
                "altitude_acclimation": 0.05,
                "travel_resilience": 0.45,
                "crowd_support_base": 0.45,
            },
        ]
    )

    enriched = add_edge_feature_columns(
        matches,
        team_context=team_context,
        venue_context=venue_context,
    )
    row = enriched.iloc[0]

    assert row["edge_home_travel_origin"] == "camp"
    assert row["edge_away_travel_origin"] == "base"
    assert row["edge_home_travel_km"] < 10
    assert row["edge_away_travel_km"] > 10000
    assert row["edge_home_body_clock_hours"] == 0
    assert row["edge_away_body_clock_hours"] > 10


def test_edge_probability_adjustment_uses_signal_without_changing_draw_probability():
    matches = pd.DataFrame(
        [
            {
                "home_team": "Home",
                "away_team": "Away",
                "predicted_result": "Away",
                "raw_top_result": "Away",
                "pick_confidence": "low",
                "top_probability": 0.39,
                "runner_up_probability": 0.37,
                "probability_margin": 0.02,
                "draw_override_applied": False,
                "p_home_win": 0.37,
                "p_draw": 0.24,
                "p_away_win": 0.39,
                "expected_home_goals": 1.1,
                "expected_away_goals": 1.2,
                "edge_total_signal": 1.4,
            }
        ]
    )

    adjusted = apply_edge_probability_adjustments(matches)
    row = adjusted.iloc[0]

    assert row["p_home_win"] > matches.iloc[0]["p_home_win"]
    assert row["p_away_win"] < matches.iloc[0]["p_away_win"]
    assert row["p_draw"] == matches.iloc[0]["p_draw"]
    assert row["expected_home_goals"] > matches.iloc[0]["expected_home_goals"]
    assert row["expected_away_goals"] < matches.iloc[0]["expected_away_goals"]
    assert row["predicted_result"] == "Home"
    assert row["base_p_home_win"] == matches.iloc[0]["p_home_win"]
    assert row["edge_home_win_probability_delta"] > 0
