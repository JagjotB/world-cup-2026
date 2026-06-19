"""
Match simulator with time-varying goal rates.

Goals are modelled as a non-homogeneous Poisson process whose intensity
follows the empirical distribution of when goals are scored in professional
football — goals cluster at the end of each half and in stoppage time,
with a dip right after kick-off in each half.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Time-varying goal intensity
# ---------------------------------------------------------------------------
# 100 time slots:
#   indices  0–44  → minutes 1′–45′     (regular first half)
#   indices 45–49  → 45+1′ to 45+5′    (first-half stoppage)
#   indices 50–94  → minutes 46′–90′   (regular second half)
#   indices 95–99  → 90+1′ to 90+5′   (second-half stoppage)


def _build_minute_labels() -> list[str]:
    labels = [f"{m}'" for m in range(1, 46)]
    labels += [f"45+{i}'" for i in range(1, 6)]
    labels += [f"{m}'" for m in range(46, 91)]
    labels += [f"90+{i}'" for i in range(1, 6)]
    return labels


MINUTE_LABELS: list[str] = _build_minute_labels()  # 100 entries

# Raw per-minute intensity weights derived from empirical goal-timing research.
# Average weight across all 100 slots is 1.0 after normalisation.
# Pattern: gradual build in each half → spike in stoppage → reset at kick-off.
_RAW_WEIGHTS = np.array([
    # Minutes 1-45 (first half)
    0.62, 0.65, 0.68, 0.70, 0.72,   # 1-5    cautious start
    0.78, 0.80, 0.82, 0.84, 0.86,   # 6-10
    0.88, 0.89, 0.90, 0.91, 0.92,   # 11-15
    0.94, 0.95, 0.96, 0.97, 0.98,   # 16-20
    0.99, 1.00, 1.01, 1.02, 1.03,   # 21-25
    1.04, 1.05, 1.06, 1.07, 1.08,   # 26-30
    1.08, 1.10, 1.10, 1.12, 1.12,   # 31-35
    1.14, 1.15, 1.16, 1.18, 1.20,   # 36-40
    1.22, 1.24, 1.26, 1.28, 1.30,   # 41-45
    # 45+1 to 45+5 (first-half stoppage) — high intensity
    1.90, 1.80, 1.70, 1.60, 1.50,
    # Minutes 46-90 (second half)
    0.65, 0.68, 0.70, 0.72, 0.74,   # 46-50  reset after half-time
    0.80, 0.82, 0.85, 0.88, 0.90,   # 51-55
    0.92, 0.94, 0.96, 0.98, 1.00,   # 56-60
    1.02, 1.05, 1.07, 1.09, 1.11,   # 61-65
    1.13, 1.16, 1.18, 1.20, 1.23,   # 66-70
    1.25, 1.28, 1.31, 1.34, 1.37,   # 71-75
    1.40, 1.43, 1.46, 1.49, 1.52,   # 76-80
    1.55, 1.58, 1.61, 1.64, 1.67,   # 81-85
    1.70, 1.72, 1.74, 1.76, 1.78,   # 86-90
    # 90+1 to 90+5 (second-half stoppage) — peak intensity
    2.20, 2.10, 2.00, 1.90, 1.80,
], dtype=float)

assert len(_RAW_WEIGHTS) == 100, "intensity weights must have exactly 100 entries"

# Normalise so probabilities sum to 1
_SLOT_PROBS: np.ndarray = _RAW_WEIGHTS / _RAW_WEIGHTS.sum()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GoalEvent:
    slot: int
    minute_label: str
    team: str  # "home" or "away"


@dataclass
class RedCardEvent:
    slot: int
    minute_label: str
    team: str  # "home" or "away"


@dataclass
class InjuryEvent:
    slot: int
    minute_label: str
    team: str  # "home" or "away"


@dataclass
class SimulatedMatch:
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    goals: list[GoalEvent] = field(default_factory=list)
    red_cards: list[RedCardEvent] = field(default_factory=list)
    injuries: list[InjuryEvent] = field(default_factory=list)

    @property
    def home_goals(self) -> int:
        return sum(1 for g in self.goals if g.team == "home")

    @property
    def away_goals(self) -> int:
        return sum(1 for g in self.goals if g.team == "away")

    @property
    def scoreline(self) -> str:
        return f"{self.home_goals}-{self.away_goals}"

    def result(self) -> str:
        h, a = self.home_goals, self.away_goals
        if h > a:
            return "H"
        elif a > h:
            return "A"
        return "D"

    def timeline_str(self, home_label: str = "", away_label: str = "") -> str:
        home_label = home_label or self.home_team
        away_label = away_label or self.away_team

        # Unified event list: (slot, priority, kind, team)
        # priority keeps goals before cards/injuries at the same minute
        items: list[tuple[int, int, str, str]] = []
        for g in self.goals:
            items.append((g.slot, 0, "goal", g.team))
        for rc in self.red_cards:
            items.append((rc.slot, 2, "red_card", rc.team))
        for inj in self.injuries:
            items.append((inj.slot, 1, "injury", inj.team))
        items.sort(key=lambda x: (x[0], x[1]))

        if not items:
            return "  (no events)"

        lines = []
        home_g = away_g = home_reds = away_reds = 0
        for slot, _, kind, team in items:
            minute = MINUTE_LABELS[slot]
            label = home_label if team == "home" else away_label
            if kind == "goal":
                if team == "home":
                    home_g += 1
                else:
                    away_g += 1
                lines.append(f"  {minute:<8} GOAL  {label}  ({home_g}-{away_g})")
            elif kind == "red_card":
                if team == "home":
                    home_reds += 1
                    men = 11 - home_reds
                else:
                    away_reds += 1
                    men = 11 - away_reds
                lines.append(f"  {minute:<8} RED   {label}  (down to {men} men)")
            elif kind == "injury":
                lines.append(f"  {minute:<8} INJ   {label}  (key player off)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Game state dependency
# ---------------------------------------------------------------------------
# When a team trails, they push forward (higher attack rate).
# When a team leads, they sit back (lower attack rate).
# Multipliers are applied to the base per-slot rate.

_STATE_MULTIPLIER: dict[int, float] = {
    2: 0.60,   # 2+ goal lead — sit back, protect
    1: 0.78,   # 1 goal lead — slightly cautious
    0: 1.00,   # level — balanced
    -1: 1.28,  # 1 goal down — push forward
    -2: 1.50,  # 2+ goals down — all-out attack
}


def _state_mult(goal_diff: int) -> float:
    """Multiplier for a team whose current goal diff is goal_diff (positive = leading)."""
    clamped = max(-2, min(2, goal_diff))
    return _STATE_MULTIPLIER[clamped]


# ---------------------------------------------------------------------------
# Red cards
# ---------------------------------------------------------------------------
# Historical rate in international football: ~0.15 red cards per team per match.
# Timing is skewed toward the second half and late-game tension periods.

TEAM_RED_CARD_RATE = 0.15

_RC_RAW = np.ones(100, dtype=float)
_RC_RAW[50:] *= 1.6     # second half more common
_RC_RAW[70:90] *= 1.4   # 71-90' — peak period for cards
_RC_RAW[45:50] *= 1.2   # first-half stoppage slightly elevated
_RC_RAW[95:] *= 1.5     # second-half stoppage
_RC_PROBS: np.ndarray = _RC_RAW / _RC_RAW.sum()  # per-slot probability fraction

# Per red card, carded team's attack drops; opponent benefits from weaker defence.
_RC_ATTACK_MULT: dict[int, float] = {0: 1.00, 1: 0.65, 2: 0.38}
_RC_OPP_BONUS: dict[int, float]   = {0: 1.00, 1: 1.25, 2: 1.55}


# ---------------------------------------------------------------------------
# In-match injuries
# ---------------------------------------------------------------------------
# ~12% chance per team of a significant mid-match injury (key attacking player).
# Effect: modest attack-rate reduction for the rest of the match.

TEAM_INJURY_PROB = 0.12
INJURY_ATTACK_IMPACT = 0.92  # attack multiplier after injury


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_match(
    home_team: str,
    away_team: str,
    home_xg: float,
    away_xg: float,
    rng: np.random.Generator | None = None,
) -> SimulatedMatch:
    """
    Simulate one match slot-by-slot with time-varying rates, game state dependency,
    red cards, and in-match injuries.
    """
    if rng is None:
        rng = np.random.default_rng()

    home_g = 0
    away_g = 0
    goals: list[GoalEvent] = []
    red_cards: list[RedCardEvent] = []
    injuries: list[InjuryEvent] = []

    rand_goals = rng.random((100, 2))   # goal sampling per slot
    rand_rc = rng.random((100, 2))      # red card sampling per slot
    rand_inj = rng.random(2)            # injury occurrence (home, away)
    rand_inj_slot = rng.integers(0, 100, size=2)  # injury timing

    # Pre-determine whether each team suffers a significant injury and when
    home_injury_slot = int(rand_inj_slot[0]) if rand_inj[0] < TEAM_INJURY_PROB else -1
    away_injury_slot = int(rand_inj_slot[1]) if rand_inj[1] < TEAM_INJURY_PROB else -1

    home_reds = 0
    away_reds = 0
    home_injured = False
    away_injured = False

    for slot in range(100):
        # Trigger pre-determined injury events
        if home_injury_slot == slot:
            home_injured = True
            injuries.append(InjuryEvent(slot, MINUTE_LABELS[slot], "home"))
        if away_injury_slot == slot:
            away_injured = True
            injuries.append(InjuryEvent(slot, MINUTE_LABELS[slot], "away"))

        # Red cards (capped at 2 per team, i.e. minimum 9 men on the pitch)
        rc_prob = TEAM_RED_CARD_RATE * _RC_PROBS[slot]
        if home_reds < 2 and rand_rc[slot, 0] < rc_prob:
            home_reds += 1
            red_cards.append(RedCardEvent(slot, MINUTE_LABELS[slot], "home"))
        if away_reds < 2 and rand_rc[slot, 1] < rc_prob:
            away_reds += 1
            red_cards.append(RedCardEvent(slot, MINUTE_LABELS[slot], "away"))

        diff = home_g - away_g

        # Home attack: weakened by own red cards, boosted against away red cards, injured flag
        home_prob = min(
            home_xg * _SLOT_PROBS[slot]
            * _state_mult(diff)
            * _RC_ATTACK_MULT.get(home_reds, 0.30)
            * _RC_OPP_BONUS.get(away_reds, 1.55)
            * (INJURY_ATTACK_IMPACT if home_injured else 1.0),
            0.99,
        )
        # Away attack: weakened by own red cards, boosted against home red cards, injured flag
        away_prob = min(
            away_xg * _SLOT_PROBS[slot]
            * _state_mult(-diff)
            * _RC_ATTACK_MULT.get(away_reds, 0.30)
            * _RC_OPP_BONUS.get(home_reds, 1.55)
            * (INJURY_ATTACK_IMPACT if away_injured else 1.0),
            0.99,
        )

        if rand_goals[slot, 0] < home_prob:
            goals.append(GoalEvent(slot, MINUTE_LABELS[slot], "home"))
            home_g += 1
        if rand_goals[slot, 1] < away_prob:
            goals.append(GoalEvent(slot, MINUTE_LABELS[slot], "away"))
            away_g += 1

    return SimulatedMatch(home_team, away_team, home_xg, away_xg, goals, red_cards, injuries)


def run_simulations(
    home_team: str,
    away_team: str,
    home_xg: float,
    away_xg: float,
    n: int = 50_000,
    seed: int | None = None,
) -> list[SimulatedMatch]:
    rng = np.random.default_rng(seed)
    return [simulate_match(home_team, away_team, home_xg, away_xg, rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def outcome_probabilities(sims: list[SimulatedMatch]) -> dict[str, float]:
    total = len(sims)
    return {
        "H": sum(1 for s in sims if s.result() == "H") / total,
        "D": sum(1 for s in sims if s.result() == "D") / total,
        "A": sum(1 for s in sims if s.result() == "A") / total,
    }


# Promote draw when P(draw) is genuinely competitive (above 1-in-3) AND
# the gap to the leading team outcome is within this margin.
# Derived from WC 2026 group stage data: threshold=1/3 avoids flipping
# correct near-miss predictions (e.g. Ghana-Panama at p_d=0.328).
SIM_DRAW_PROMOTE_THRESHOLD = 1 / 3   # ~0.333
SIM_DRAW_PROMOTE_MARGIN = 0.05       # 5 percentage points


def sim_predict_result(
    probs: dict[str, float],
    home_xg: float = 0.0,
    away_xg: float = 0.0,
    draw_promote_threshold: float = SIM_DRAW_PROMOTE_THRESHOLD,
    draw_promote_margin: float = SIM_DRAW_PROMOTE_MARGIN,
) -> str:
    """
    Draw-aware result prediction from simulation probabilities.

    Base rule: promote draw when P(draw) > 1/3 and within 5pp of the leader.

    Total xG adjustment: low-scoring games draw more than Poisson alone implies.
    When total xG < 2.5 the threshold and margin are relaxed so near-draw
    games are more likely to be called as draws.
      total_xg < 2.0 → threshold -0.06, margin +0.04
      total_xg 2.0–2.5 → threshold -0.03, margin +0.02
    """
    p_h, p_d, p_a = probs["H"], probs["D"], probs["A"]
    best_non_draw = max(p_h, p_a)
    best_non_draw_label = "H" if p_h >= p_a else "A"

    effective_threshold = draw_promote_threshold
    effective_margin = draw_promote_margin

    total_xg = home_xg + away_xg
    if total_xg > 0:
        if total_xg < 2.0:
            effective_threshold -= 0.06
            effective_margin += 0.04
        elif total_xg < 2.5:
            effective_threshold -= 0.03
            effective_margin += 0.02

    if p_d >= effective_threshold and p_d >= best_non_draw - effective_margin:
        return "D"
    return best_non_draw_label


def scoreline_distribution(sims: list[SimulatedMatch], top_n: int = 12) -> list[tuple[str, float]]:
    counts: dict[str, int] = {}
    for s in sims:
        counts[s.scoreline] = counts.get(s.scoreline, 0) + 1
    total = len(sims)
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    return [(sl, count / total) for sl, count in ranked[:top_n]]


def add_simulation_columns(
    predictions: object,
    predictor: object,
    n_sims: int = 10_000,
    seed: int = 42,
    draw_promote_threshold: float = SIM_DRAW_PROMOTE_THRESHOLD,
    draw_promote_margin: float = SIM_DRAW_PROMOTE_MARGIN,
) -> object:
    """
    Run match simulations for every row in *predictions* and append columns:
      sim_home_xg, sim_away_xg,
      sim_p_home_win, sim_p_draw, sim_p_away_win,
      sim_top_scoreline, sim_top_scoreline_prob,
      sim_predicted_result  (draw-aware: threshold + margin rule applied).
    """
    import pandas as pd  # local import — caller may not have it at module load

    rng_master = np.random.default_rng(seed)

    rows: list[dict] = []
    for row in predictions.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        seed_i = int(rng_master.integers(0, 1_000_000))

        try:
            pred = predictor.predict_match(home, away, neutral=True)  # type: ignore[attr-defined]
            home_xg = pred.expected_home_goals
            away_xg = pred.expected_away_goals
        except Exception:
            rows.append({
                "sim_home_xg": float("nan"),
                "sim_away_xg": float("nan"),
                "sim_p_home_win": float("nan"),
                "sim_p_draw": float("nan"),
                "sim_p_away_win": float("nan"),
                "sim_top_scoreline": "",
                "sim_top_scoreline_prob": float("nan"),
                "sim_predicted_result": "",
            })
            continue

        sims = run_simulations(home, away, home_xg, away_xg, n=n_sims, seed=seed_i)
        probs = outcome_probabilities(sims)
        top = scoreline_distribution(sims, top_n=1)
        top_sl, top_sl_p = top[0] if top else ("", float("nan"))
        predicted = sim_predict_result(
            probs,
            home_xg=home_xg,
            away_xg=away_xg,
            draw_promote_threshold=draw_promote_threshold,
            draw_promote_margin=draw_promote_margin,
        )

        rows.append({
            "sim_home_xg": round(home_xg, 3),
            "sim_away_xg": round(away_xg, 3),
            "sim_p_home_win": round(probs["H"], 4),
            "sim_p_draw": round(probs["D"], 4),
            "sim_p_away_win": round(probs["A"], 4),
            "sim_top_scoreline": top_sl,
            "sim_top_scoreline_prob": round(top_sl_p, 4),
            "sim_predicted_result": predicted,
        })

    sim_df = pd.DataFrame(rows, index=predictions.index)
    return pd.concat([predictions, sim_df], axis=1)


def goal_timing_distribution(sims: list[SimulatedMatch]) -> dict[str, float]:
    """Fraction of all goals scored in each 15-minute band."""
    bands = {"1-15": 0, "16-30": 0, "31-45": 0, "45+": 0,
             "46-60": 0, "61-75": 0, "76-90": 0, "90+": 0}
    total = 0
    for s in sims:
        for g in s.goals:
            total += 1
            idx = g.slot
            if idx < 15:
                bands["1-15"] += 1
            elif idx < 30:
                bands["16-30"] += 1
            elif idx < 45:
                bands["31-45"] += 1
            elif idx < 50:
                bands["45+"] += 1
            elif idx < 65:
                bands["46-60"] += 1
            elif idx < 80:
                bands["61-75"] += 1
            elif idx < 95:
                bands["76-90"] += 1
            else:
                bands["90+"] += 1
    if total == 0:
        return {k: 0.0 for k in bands}
    return {k: v / total for k, v in bands.items()}
