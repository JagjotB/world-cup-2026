# FIFA World Cup 2026 Predictor

This project trains a match-level machine learning model on historical men's international results, then uses it to forecast FIFA World Cup 2026 group matches and simulate the full 104-match tournament.

The default model is now the enhanced ML path. It uses:

- pre-match Elo ratings updated chronologically from historical results
- recent form features from each team's previous matches
- tournament-type features for friendlies, qualifiers, and major tournaments
- 2026 squad/player-strength features where available
- individual player style scores for availability, experience, attacking, creativity, shot volume, crossing, ball-winning, defending, goalkeeping, physical profile, and discipline
- projected-XI matchup features such as attack versus opponent defense, creativity versus opponent ball-winning, keeper edge, depth edge, and discipline edge
- manual injury/suspension/minute-limit inputs, confirmed/probable lineup inputs, team tactical profiles, and extra non-Big-5 club-player stats when supplied
- a calibrated gradient-boosting classifier for home/draw/away probabilities
- an experimental two-stage draw-vs-non-draw then home-vs-away model; training evaluates it and keeps it only if it beats the multiclass model
- gradient-boosted goal models for sampled scorelines
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

Use `python scripts/predict_group_stage.py --from-date 2026-06-17` to pin the upcoming-match cutoff date. Add `--no-live-results` when you want static pre-tournament ratings/form.

For a known knockout-stage matchup:

```powershell
python scripts/predict_knockout_match.py --home Argentina --away France
```

To score played group-stage matches:

```powershell
python scripts/evaluate_played_group_stage.py --refresh-schedule
python scripts/evaluate_played_group_stage.py --sequential-live
python scripts/evaluate_played_group_stage.py --pre-world-cup-model --refresh-schedule
```

Outputs are written to `outputs/predictions/`:

- `group_match_probabilities.csv`
- `tournament_probabilities.csv`
- `sampled_tournament_matches.csv`
- `upcoming_group_stage_predictions.csv`
- `sampled_knockout_predictions.csv`
- `projected_starting_lineups.csv`
- `player_match_projections.csv`

## Player Data

The player-data pipeline has two sources:

- FIFA squad lists: official tournament roster data with player names, positions, clubs, heights, caps, and international goals.
- Club-season stats: 2025-26 Big 5 European league player stats from a public Kaggle mirror of FBref data.

It also supports manual enrichment files:

- `data/manual/additional_club_player_stats_2025_2026.csv`: non-Big-5 or manually sourced club player stats.
- `data/manual/player_availability_2026.csv`: injuries, suspensions, availability multipliers, and minute limits.
- `data/manual/projected_lineups_2026.csv`: confirmed/probable starters, slots, positions, and formation.
- `data/manual/team_tactics_2026.csv`: formation, pressing, defensive line, tempo, directness, width, set-piece strength, and transition speed.
- `data/manual/data_source_backlog.csv`: non-odds source candidates for squad, injury, lineup, tactic, and non-Big-5 player-stat enrichment.

Run:

```powershell
python scripts/fetch_fifa_squads.py
python scripts/fetch_kaggle_club_stats.py
python scripts/build_player_features.py
```

Generated files:

- `data/processed/fifa_squad_players_2026.csv`
- `data/processed/club_player_stats_2025_2026.csv`
- `data/processed/player_features_2026.csv`
- `data/processed/team_player_features_2026.csv`

The FIFA roster covers all 48 teams and 1,248 players. `player_features_2026.csv` now includes individual style scores, availability fields, and manual-lineup fields. `team_player_features_2026.csv` rolls projected starting-XI, unit matchup strengths, availability, and tactical features into the match model. The club-stat table covers Big 5 European league players; players in MLS, Saudi Arabia, Brazil, Argentina, Japan, Qatar, and other leagues need another provider or manual import through `additional_club_player_stats_2025_2026.csv`.

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
- `scripts/predict_group_stage.py` fetches group-stage fixtures/results and predicts unplayed matches.
- `scripts/predict_player_match_projections.py` projects per-player minutes, goals, assists, shots, defensive actions, saves, clean-sheet probability, card risk, and impact score for each upcoming group match.
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
