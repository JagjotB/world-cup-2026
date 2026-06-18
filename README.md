# FIFA World Cup 2026 Predictor

This project trains a match-level machine learning model on historical men's international results, then uses it to forecast FIFA World Cup 2026 group matches and simulate the full 104-match tournament.

The default model is now the enhanced ML path. It uses:

- pre-match Elo ratings updated chronologically from historical results
- recent form features from each team's previous matches
- tournament-type features for friendlies, qualifiers, and major tournaments
- 2026 squad/player-strength features where available
- individual player style scores for availability, experience, attacking, creativity, shot volume, crossing, ball-winning, defending, goalkeeping, physical profile, and discipline
- projected-XI matchup features such as attack versus opponent defense, creativity versus opponent ball-winning, keeper edge, depth edge, and discipline edge
- player-projection summary features such as expected minutes, goal threat, assist threat, shot pressure, defensive workload, keeper coverage, bench impact, card risk, and projection balance
- manual injury/suspension/minute-limit inputs, confirmed/probable lineup inputs, team tactical profiles, and extra non-Big-5 club-player stats when supplied
- non-odds edge features for venue/travel/body-clock stress, rest, heat/altitude acclimation, host/crowd support, lineup chemistry, set-piece mismatches, keeper-versus-shot style, press resistance, second-half pressure, and optional referee/weather overrides
- manual public/consented player readiness signals from wearable-style observations such as recovery, sleep, HRV delta, resting-heart-rate delta, strain/load, and minutes adjustments when supplied
- cached pre-match news signals for injuries, morale, fatigue, manager confidence, narrative edge, and upset risk when supplied
- a model-tournament training path that compares logistic regression, calibrated histogram gradient boosting, XGBoost, LightGBM, CatBoost, two-stage draw models, expected-goals-derived outcomes, soft voting, and stacked probability ensembles
- automatic selection of the best historical-validation result model and the best expected-goals model
- validation-selected blending between the learned draw probability and the expected-goals Poisson draw probability
- boosted goal models for sampled scorelines
- live World Cup result updates applied to ratings/form before future predictions
- Monte Carlo simulation for group qualification, Round of 32, and the knockout bracket, with sampled scorelines constrained to calibrated match outcomes

The model deliberately does not use Vegas lines, bookmaker odds, or market-implied probabilities.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/train_model.py --download
python scripts/predict_tournament.py --sims 5000
python scripts/predict_group_stage.py
```

For the current no-leakage workflow, train without 2026 World Cup rows and then apply played matches live:

```powershell
python scripts/train_model.py --model-kind enhanced --exclude-2026-world-cup
python scripts/predict_group_stage.py --from-date 2026-06-17
python scripts/predict_player_match_projections.py --from-date 2026-06-17
python scripts/predict_tournament.py --sims 5000
```

Use `python scripts/train_model.py --model-kind baseline` only when you want to compare against the older logistic-regression baseline.

The enhanced trainer writes candidate-level validation metrics to `models/world_cup_match_model.json`, including accuracy, log loss, predicted draw rate, and goal-model MAE. The saved artifact uses the candidate that performs best on the historical validation split; it does not tune on bookmaker odds or current-market prices.

Use `python scripts/predict_group_stage.py --from-date 2026-06-17` to pin the upcoming-match cutoff date. Add `--no-live-results` when you want static pre-tournament ratings/form.

Group-stage prediction rows include usefulness filters from the live tournament backtest:

- `double_chance_useful`: true when the model top result probability is at least 45%; use `double_chance_pick`.
- `either_team_wins_probability`: the home-or-away/no-draw probability; currently not marked useful because the live backtest has not found a reliable threshold.
- `more_shots_on_target_useful`: true when the projected shots-on-target edge is at least 0.50; `strong` starts at 1.25.
- `home_1plus_corners_each_half_useful_pick` / `away_1plus_corners_each_half_useful_pick`: `Yes` at 62%+, `No` at 52% or lower, otherwise `No play`.
- `btts_useful`: true when both teams project for at least 0.70 expected goals; use `btts_pick=Yes`.
- `edge_total_signal`: a non-odds context/style score. Positive favors the listed home team, negative favors the away team. `edge_flags` shows the largest drivers.
- `edge_readiness_edge`: a manual wearable-style readiness edge. Positive favors the home team, negative favors the away team. It stays neutral until rows are added to `player_readiness_signals_2026.csv`.
- `news_*`: cached pre-match news features. They stay neutral until `llm_news_signals_2026.csv` is populated.

For a known knockout-stage matchup:

```powershell
python scripts/predict_knockout_match.py --home Argentina --away France
```

To score played group-stage matches:

```powershell
python scripts/evaluate_played_group_stage.py --refresh-schedule
python scripts/evaluate_played_group_stage.py --sequential-live
python scripts/evaluate_played_group_stage.py --pre-world-cup-model --refresh-schedule
python scripts/evaluate_corner_props.py --refresh-schedule
```

Outputs are written to `outputs/predictions/`:

- `group_match_probabilities.csv`
- `tournament_probabilities.csv`
- `sampled_tournament_matches.csv`
- `upcoming_group_stage_predictions.csv`
- `sampled_knockout_predictions.csv`
- `projected_starting_lineups.csv`
- `player_match_projections.csv`
- `played_corner_prop_evaluation.csv`

## Player Data

The player-data pipeline has two sources:

- FIFA squad lists: official tournament roster data with player names, positions, clubs, heights, caps, and international goals.
- Club-season stats: 2025-26 Big 5 European league player stats from a public Kaggle mirror of FBref data.

It also supports manual enrichment files:

- `data/manual/additional_club_player_stats_2025_2026.csv`: non-Big-5 or manually sourced club player stats.
- `data/manual/player_availability_2026.csv`: injuries, suspensions, availability multipliers, and minute limits.
- `data/manual/projected_lineups_2026.csv`: confirmed/probable starters, slots, positions, and formation.
- `data/manual/team_tactics_2026.csv`: formation, pressing, defensive line, tempo, directness, width, set-piece strength, and transition speed.
- `data/manual/team_context_2026.csv`: team home/base location, tournament camp city/latitude/longitude/UTC offset, travel resilience, heat/altitude acclimation, and crowd-support priors. Camp coordinates override base coordinates in per-match travel/body-clock features and the camp-aware edge signal can nudge W/D/L probabilities; output columns show whether each side used `camp` or `base`. Camp assignments come from FIFA's published Team Base Camp list, with training-site coordinates from the NCBRT World Cup basecamp map.
- `data/manual/venue_context_2026.csv`: stadium location, altitude, roof/surface, and June weather priors.
- `data/manual/match_context_overrides_2026.csv`: optional match-specific referee tempo/card strictness, weather, wind, and crowd boosts.
- `data/manual/player_readiness_signals_2026.csv`: public or consented Strava/WHOOP-style observations entered manually. Use `include_signal=false` to keep a row for notes without affecting the model.
- `data/manual/llm_news_signals_2026.csv`: cached pre-match news/NLP signals, created manually or by `scripts/fetch_news_signals.py` when an LLM API key is configured.
- `data/manual/data_source_backlog.csv`: non-odds source candidates for squad, injury, lineup, tactic, and non-Big-5 player-stat enrichment.

`player_readiness_signals_2026.csv` accepts sparse rows. Fill only what you can verify:

```csv
local_date,team,player_name,source_type,source_url,consent_status,include_signal,confidence,readiness_score,recovery_score,sleep_hours,sleep_quality_score,hrv_delta_pct,resting_hr_delta_pct,strain_7d,acute_load_delta_pct,last_activity_distance_km,minutes_adjustment,notes
2026-06-18,Canada,Example Player,whoop_public,https://example.com,public,true,0.8,,78,8.0,,10,-3,9,,4.2,0.1,public post before match
```

Run:

```powershell
python scripts/fetch_fifa_squads.py
python scripts/fetch_kaggle_club_stats.py
python scripts/build_player_features.py
python scripts/fetch_news_signals.py --date 2026-06-18
```

Generated files:

- `data/processed/fifa_squad_players_2026.csv`
- `data/processed/club_player_stats_2025_2026.csv`
- `data/processed/player_features_2026.csv`
- `data/processed/team_player_features_2026.csv`

The FIFA roster covers all 48 teams and 1,248 players. `player_features_2026.csv` now includes individual style scores, availability fields, and manual-lineup fields. `team_player_features_2026.csv` rolls projected starting-XI, player-performance projection summaries, unit matchup strengths, availability, and tactical features into the match model. The club-stat table covers Big 5 European league players; players in MLS, Saudi Arabia, Brazil, Argentina, Japan, Qatar, and other leagues need another provider or manual import through `additional_club_player_stats_2025_2026.csv`.

## Updating Live Results

Add completed group-stage matches to `data/manual/results_2026.csv`:

```csv
stage,group,home_team,away_team,home_score,away_score
group,A,Mexico,South Africa,2,0
```

The simulator treats those scores as fixed and only samples unplayed matches.

## Files

- `data/manual/teams_2026.csv` contains the 48-team group draw used by the simulator.
- `data/manual/team_aliases.csv` maps tournament names to historical-data names.
- `scripts/train_model.py` downloads historical data and saves `models/world_cup_match_model.joblib`.
- `scripts/predict_group_stage.py` fetches group-stage fixtures/results, predicts unplayed matches, and adds team-level projected shots-on-target, corner-prop, and non-odds context/style edges when player projections are available.
- `scripts/predict_player_match_projections.py` projects per-player minutes, goals, assists, shots, defensive actions, saves, clean-sheet probability, card risk, and impact score for each upcoming group match.
- `scripts/evaluate_corner_props.py` evaluates projected 1+ corner in each half picks against ESPN event commentary for completed group matches.
- `scripts/predict_tournament.py` generates match probabilities, tournament probabilities, sampled knockout resolution paths, and lineup projections.
- `scripts/predict_knockout_match.py` predicts regulation, extra-time, penalty, and lineup details for one knockout matchup.
- `scripts/fetch_fifa_squads.py` downloads and parses the official FIFA squad PDF.
- `scripts/fetch_kaggle_club_stats.py` downloads a public Big 5 club-player stats CSV.
- `scripts/build_player_features.py` combines FIFA roster data with club stats.
- `src/worldcup2026/tournament.py` implements the expanded 2026 format.

## Notes

The Round of 32 uses the official slot structure for group winners, runners-up, and allowed third-place group candidates. The exact FIFA Annex C mapping for every possible third-place combination is represented with a valid backtracking assignment across the published slot candidate sets.

Knockout-stage predictions use FIFA's current 2026 rule path: if tied after normal time, play two 15-minute extra-time periods; if still tied, decide the match by penalties. Golden goal is not active in the current World Cup rules, so the model records `golden_goal_rule_active=false` and `p_golden_goal=0`.

Sources checked while scaffolding:

- FIFA format guidance: https://gpcustomersupportfwc2026.tickets.fifa.com/hc/en-gb/articles/28784798873117-10-What-is-the-format-for-the-FIFA-World-Cup-2026-tournament
- FIFA official squad list PDF: https://fdp.fifa.org/assetspublic/ce281/pdf/SquadLists-English.pdf
- Historical results dataset: https://github.com/martj42/international_results
- Kaggle Big 5 player stats mirror: https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2025-2026
- 2026 group list cross-check: https://www.foxsports.com/soccer/fifa-world-cup/standings
