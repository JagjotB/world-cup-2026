from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANUAL_DIR = DATA_DIR / "manual"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"

RAW_RESULTS_FILE = RAW_DIR / "international_results.csv"
FIFA_SQUAD_PDF_URL = "https://fdp.fifa.org/assetspublic/ce281/pdf/SquadLists-English.pdf"
FIFA_SQUAD_PDF_FILE = RAW_DIR / "fifa_squad_lists_2026.pdf"
TEAMS_2026_FILE = MANUAL_DIR / "teams_2026.csv"
ALIASES_FILE = MANUAL_DIR / "team_aliases.csv"
FIFA_TEAM_ALIASES_FILE = MANUAL_DIR / "fifa_team_aliases.csv"
PLAYER_ALIASES_FILE = MANUAL_DIR / "player_aliases.csv"
CLUB_STAT_SOURCES_FILE = MANUAL_DIR / "club_stat_sources.csv"
ADDITIONAL_CLUB_PLAYER_STATS_FILE = MANUAL_DIR / "additional_club_player_stats_2025_2026.csv"
PLAYER_AVAILABILITY_FILE = MANUAL_DIR / "player_availability_2026.csv"
MANUAL_PROJECTED_LINEUPS_FILE = MANUAL_DIR / "projected_lineups_2026.csv"
TEAM_TACTICS_FILE = MANUAL_DIR / "team_tactics_2026.csv"
TEAM_CONTEXT_FILE = MANUAL_DIR / "team_context_2026.csv"
VENUE_CONTEXT_FILE = MANUAL_DIR / "venue_context_2026.csv"
MATCH_CONTEXT_OVERRIDES_FILE = MANUAL_DIR / "match_context_overrides_2026.csv"
PLAYER_READINESS_SIGNALS_FILE = MANUAL_DIR / "player_readiness_signals_2026.csv"
NEWS_SIGNALS_FILE = MANUAL_DIR / "llm_news_signals_2026.csv"
KAGGLE_CLUB_STATS_DATASET = "hubertsidorowicz/football-players-stats-2025-2026"
RESULTS_2026_FILE = MANUAL_DIR / "results_2026.csv"
FIFA_SQUAD_PLAYERS_FILE = PROCESSED_DIR / "fifa_squad_players_2026.csv"
CLUB_PLAYER_STATS_FILE = PROCESSED_DIR / "club_player_stats_2025_2026.csv"
PLAYER_FEATURES_FILE = PROCESSED_DIR / "player_features_2026.csv"
TEAM_PLAYER_FEATURES_FILE = PROCESSED_DIR / "team_player_features_2026.csv"
GROUP_STAGE_SCHEDULE_FILE = PROCESSED_DIR / "group_stage_schedule_2026.csv"
UPCOMING_GROUP_STAGE_PREDICTIONS_FILE = OUTPUT_DIR / "upcoming_group_stage_predictions.csv"
PROJECTED_LINEUPS_FILE = OUTPUT_DIR / "projected_starting_lineups.csv"
PLAYER_MATCH_PROJECTIONS_FILE = OUTPUT_DIR / "player_match_projections.csv"
SAMPLED_KNOCKOUT_PREDICTIONS_FILE = OUTPUT_DIR / "sampled_knockout_predictions.csv"
MODEL_FILE = MODELS_DIR / "world_cup_match_model.joblib"
