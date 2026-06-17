from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from .config import (
    ADDITIONAL_CLUB_PLAYER_STATS_FILE,
    CLUB_PLAYER_STATS_FILE,
    FIFA_SQUAD_PDF_FILE,
    FIFA_SQUAD_PDF_URL,
    FIFA_SQUAD_PLAYERS_FILE,
    FIFA_TEAM_ALIASES_FILE,
    MANUAL_PROJECTED_LINEUPS_FILE,
    PLAYER_AVAILABILITY_FILE,
    KAGGLE_CLUB_STATS_DATASET,
    PLAYER_ALIASES_FILE,
    TEAM_TACTICS_FILE,
    TEAM_PLAYER_FEATURES_FILE,
)
from .data import normalized_key
from .lineups import DEFAULT_FORMATION, OUT_STATUSES, STARTER_STATUSES, player_lineup_score
from .player_projections import PLAYER_PROJECTION_TEAM_FEATURE_COLUMNS, summarize_team_player_projections

POSITIONS = {"GK", "DF", "MF", "FW"}
DATE_PATTERN = re.compile(r"\d{2}/\d{2}/\d{4}")
CLUB_PATTERN = re.compile(r"^(?P<club>.+?)\s*\((?P<country_code>[A-Z]{3})\)$")
PLAYER_STYLE_COLUMNS = [
    "availability_score",
    "experience_score",
    "attacking_score",
    "creativity_score",
    "shot_volume_score",
    "crossing_score",
    "ball_winning_score",
    "defensive_score",
    "keeper_score",
    "physical_score",
    "discipline_risk",
]
LINEUP_TEAM_FEATURE_COLUMNS = [
    "projected_lineup_score_sum",
    "projected_lineup_score_mean",
    "projected_lineup_club_stats_share",
    "projected_lineup_minutes_sum",
    "projected_lineup_caps_sum",
    "top_11_availability_score",
    "top_11_experience_score",
    "top_11_attacking_score",
    "top_11_creativity_score",
    "top_11_shot_volume_score",
    "top_11_crossing_score",
    "top_11_ball_winning_score",
    "top_11_defensive_score",
    "top_11_keeper_score",
    "top_11_physical_score",
    "top_11_discipline_risk",
    "starting_gk_keeper_score",
    "starting_def_defensive_score",
    "starting_def_physical_score",
    "starting_mf_creativity_score",
    "starting_mf_ball_winning_score",
    "starting_fw_attacking_score",
    "starting_fw_shot_volume_score",
    "bench_depth_score",
    *PLAYER_PROJECTION_TEAM_FEATURE_COLUMNS,
]
TACTIC_TEAM_FEATURE_COLUMNS = [
    "tactics_available",
    "formation_back_line",
    "formation_midfield_line",
    "formation_forward_line",
    "formation_defensive_density",
    "formation_midfield_density",
    "formation_attacking_density",
    "tactic_pressing_intensity",
    "tactic_defensive_line",
    "tactic_tempo",
    "tactic_directness",
    "tactic_possession_style",
    "tactic_width",
    "tactic_set_piece_strength",
    "tactic_transition_speed",
]
AVAILABILITY_TEAM_FEATURE_COLUMNS = [
    "availability_multiplier_sum",
    "availability_multiplier_mean",
    "expected_minutes_share_sum",
    "expected_minutes_share_mean",
    "is_unavailable_sum",
    "is_questionable_sum",
    "is_suspended_sum",
    "is_injured_sum",
]


def download_fifa_squad_pdf(
    destination: Path = FIFA_SQUAD_PDF_FILE,
    url: str = FIFA_SQUAD_PDF_URL,
    force: bool = False,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    urlretrieve(url, destination)
    return destination


def load_fifa_team_aliases(path: Path = FIFA_TEAM_ALIASES_FILE) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path).fillna("")
    aliases = {}
    for row in df.itertuples(index=False):
        fifa_team = str(row.fifa_team).strip()
        project_team = str(row.project_team).strip()
        if fifa_team and project_team:
            aliases[fifa_team] = project_team
    return aliases


def load_player_aliases(path: Path = PLAYER_ALIASES_FILE) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path).fillna("")
    aliases = {}
    for row in df.itertuples(index=False):
        source_name = str(row.source_name).strip()
        canonical_name = str(row.canonical_name).strip()
        if source_name and canonical_name:
            aliases[normalized_key(source_name)] = canonical_name
    return aliases


def parse_team_header(text: str) -> tuple[str, str]:
    for line in text.splitlines():
        match = re.match(r"^(?P<team>.+?)\s+\((?P<code>[A-Z]{3})\)$", line.strip())
        if match:
            return match.group("team").strip(), match.group("code").strip()
    raise ValueError("Could not find team header in FIFA squad page.")


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", "").split()).strip()


def _pick(row: list[object], indexes: list[int]) -> str:
    values = [_clean_cell(row[idx]) for idx in indexes if idx < len(row)]
    return " ".join(value for value in values if value)


def _header_indexes(header: list[object]) -> dict[str, int]:
    indexes = {}
    for idx, value in enumerate(header):
        label = _clean_cell(value).upper()
        if label:
            indexes[label] = idx
    return indexes


def _by_header(row: list[object], indexes: dict[str, int], label: str) -> str:
    idx = indexes.get(label.upper())
    if idx is None or idx >= len(row):
        return ""
    return _clean_cell(row[idx])


def split_club(raw_club: str) -> tuple[str, str]:
    match = CLUB_PATTERN.match(_clean_cell(raw_club))
    if not match:
        return _clean_cell(raw_club), ""
    return match.group("club").strip(), match.group("country_code").strip()


def canonical_player_name(first_names: str, last_names: str, player_name: str) -> str:
    first_names = _clean_cell(first_names)
    last_names = _clean_cell(last_names)
    if first_names or last_names:
        return f"{first_names} {last_names}".strip()
    return _clean_cell(player_name)


def parse_fifa_squad_pdf(
    pdf_path: Path = FIFA_SQUAD_PDF_FILE,
    team_aliases: dict[str, str] | None = None,
    player_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("Install pdfplumber to parse FIFA squad PDFs.") from exc

    team_aliases = team_aliases or {}
    player_aliases = player_aliases or {}
    rows: list[dict[str, object]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            fifa_team, fifa_team_code = parse_team_header(page.extract_text() or "")
            project_team = team_aliases.get(fifa_team, fifa_team)
            tables = page.extract_tables()
            if not tables:
                continue

            table = tables[0]
            if not table:
                continue
            indexes = _header_indexes(table[0])

            for row in table[1:]:
                squad_number = _by_header(row, indexes, "#") or _clean_cell(row[0] if row else "")
                position = _by_header(row, indexes, "POS") or _clean_cell(row[1] if len(row) > 1 else "")
                if not squad_number.isdigit() or position not in POSITIONS:
                    continue

                player_name = _by_header(row, indexes, "PLAYER NAME")
                first_names = _by_header(row, indexes, "FIRST NAME(S)")
                last_names = _by_header(row, indexes, "LAST NAME(S)")
                name_on_shirt = _by_header(row, indexes, "NAME ON SHIRT")
                dob = _by_header(row, indexes, "DOB")
                raw_club = _by_header(row, indexes, "CLUB")
                height_cm = _by_header(row, indexes, "HEIGHT (CM)")
                caps = _by_header(row, indexes, "CAPS")
                goals = _by_header(row, indexes, "GOALS")

                if not DATE_PATTERN.fullmatch(dob):
                    continue

                club, club_country_code = split_club(raw_club)
                canonical_name = canonical_player_name(first_names, last_names, player_name)
                canonical_name = player_aliases.get(normalized_key(canonical_name), canonical_name)

                rows.append(
                    {
                        "team": project_team,
                        "fifa_team": fifa_team,
                        "fifa_team_code": fifa_team_code,
                        "squad_number": int(squad_number),
                        "position": position,
                        "player_name": canonical_name,
                        "fifa_player_name": player_name,
                        "first_names": first_names,
                        "last_names": last_names,
                        "name_on_shirt": name_on_shirt,
                        "date_of_birth": pd.to_datetime(dob, format="%d/%m/%Y"),
                        "club": club,
                        "club_country_code": club_country_code,
                        "height_cm": int(height_cm) if height_cm.isdigit() else np.nan,
                        "international_caps": int(caps) if caps.isdigit() else np.nan,
                        "international_goals": int(goals) if goals.isdigit() else np.nan,
                        "source_page": page_number,
                    }
                )

    if not rows:
        raise ValueError(f"No players parsed from {pdf_path}")

    df = pd.DataFrame(rows)
    df["player_key"] = df["player_name"].map(normalized_key)
    df["club_key"] = df["club"].map(normalized_key)
    df["age_on_2026_06_11"] = (
        (pd.Timestamp("2026-06-11") - df["date_of_birth"]).dt.days / 365.25
    ).round(2)
    return df.sort_values(["team", "squad_number"]).reset_index(drop=True)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif out.index.name is not None:
        out = out.reset_index()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part).strip() for part in column if str(part).strip())
            for column in out.columns
        ]
    else:
        out.columns = [str(column).strip() for column in out.columns]

    normalized_columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", column).strip("_").lower()
        for column in out.columns
    ]
    counts: dict[str, int] = {}
    deduped_columns = []
    for column in normalized_columns:
        counts[column] = counts.get(column, 0) + 1
        if counts[column] == 1:
            deduped_columns.append(column)
        else:
            deduped_columns.append(f"{column}_{counts[column]}")
    out.columns = deduped_columns
    return out


def download_kaggle_club_player_stats(
    destination: Path = CLUB_PLAYER_STATS_FILE,
    dataset: str = KAGGLE_CLUB_STATS_DATASET,
    light: bool = True,
) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub to download Kaggle club player stats.") from exc

    dataset_path = Path(kagglehub.dataset_download(dataset))
    preferred = "players_data_light-2025_2026.csv" if light else "players_data-2025_2026.csv"
    source_path = dataset_path / preferred
    if not source_path.exists():
        csv_files = sorted(dataset_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in Kaggle dataset cache: {dataset_path}")
        source_path = csv_files[0]

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    return destination


def fetch_fbref_player_season_stats(
    leagues: list[str],
    season: str,
    stat_types: list[str],
) -> pd.DataFrame:
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise RuntimeError("Install soccerdata to fetch FBref club player stats.") from exc

    frames = []
    for stat_type in stat_types:
        fbref = sd.FBref(leagues=leagues, seasons=season)
        stats = fbref.read_player_season_stats(stat_type=stat_type)
        flat = flatten_columns(stats)
        flat["stat_type"] = stat_type
        flat["source"] = "FBref"
        flat["source_season"] = season
        frames.append(flat)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_club_player_stats(path: Path = CLUB_PLAYER_STATS_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = flatten_columns(pd.read_csv(path))
    if df.empty:
        return df
    if "player" in df.columns:
        df["player_key"] = df["player"].map(normalized_key)
    elif "player_name" in df.columns:
        df["player_key"] = df["player_name"].map(normalized_key)
    if "team" in df.columns:
        df["club_key"] = df["team"].map(normalized_key)
    elif "squad" in df.columns:
        df["club_key"] = df["squad"].map(normalized_key)
    return df


def load_optional_manual_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return flatten_columns(pd.read_csv(path)).fillna("")


def combine_club_player_stats(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    combined = pd.concat(usable, ignore_index=True, sort=False)
    if "player_key" not in combined.columns:
        if "player" in combined.columns:
            combined["player_key"] = combined["player"].map(normalized_key)
        elif "player_name" in combined.columns:
            combined["player_key"] = combined["player_name"].map(normalized_key)
    if "club_key" not in combined.columns:
        if "team" in combined.columns:
            combined["club_key"] = combined["team"].map(normalized_key)
        elif "squad" in combined.columns:
            combined["club_key"] = combined["squad"].map(normalized_key)
    return combined


def _availability_multiplier_for_status(status: str) -> float:
    status = str(status or "available").strip().casefold()
    if status in {"out", "injured", "suspended", "unavailable"}:
        return 0.0
    if status == "doubtful":
        return 0.35
    if status == "questionable":
        return 0.65
    if status in {"minute_limit", "limited"}:
        return 0.55
    return 1.0


def apply_player_availability(
    players: pd.DataFrame,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = players.copy()
    out["availability_status"] = "available"
    out["availability_multiplier"] = 1.0
    out["expected_minutes_share"] = 1.0
    out["injury_status"] = ""
    out["suspension_status"] = ""

    if availability is not None and not availability.empty:
        table = availability.copy()
        if "player" in table.columns and "player_name" not in table.columns:
            table["player_name"] = table["player"]
        table["team_key"] = table["team"].map(normalized_key) if "team" in table.columns else ""
        table["player_key"] = (
            table["player_name"].map(normalized_key) if "player_name" in table.columns else ""
        )
        out["team_key"] = out["team"].map(normalized_key)

        for row in table.itertuples(index=False):
            mask = out["team_key"].eq(row.team_key) & out["player_key"].eq(row.player_key)
            if not mask.any():
                continue
            status = str(getattr(row, "status", "") or "available").strip().casefold()
            injury_status = str(getattr(row, "injury_status", "") or "").strip()
            suspension_status = str(getattr(row, "suspension_status", "") or "").strip()
            if suspension_status and not status:
                status = "suspended"
            if injury_status and not status:
                status = "injured"
            base_multiplier = _availability_multiplier_for_status(status)
            provided_multiplier = pd.to_numeric(
                pd.Series([getattr(row, "availability_multiplier", "")]),
                errors="coerce",
            ).iloc[0]
            if not pd.isna(provided_multiplier):
                base_multiplier = float(np.clip(provided_multiplier, 0.0, 1.0))
            expected_minutes_share = pd.to_numeric(
                pd.Series([getattr(row, "expected_minutes_share", "")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(expected_minutes_share):
                expected_minutes_share = base_multiplier
            expected_minutes_share = float(np.clip(expected_minutes_share, 0.0, 1.0))

            out.loc[mask, "availability_status"] = status or "available"
            out.loc[mask, "availability_multiplier"] = base_multiplier
            out.loc[mask, "expected_minutes_share"] = expected_minutes_share
            out.loc[mask, "injury_status"] = injury_status
            out.loc[mask, "suspension_status"] = suspension_status

        out = out.drop(columns=["team_key"])

    status = out["availability_status"].astype(str).str.casefold()
    out["is_unavailable"] = status.isin(OUT_STATUSES).astype(float)
    out["is_questionable"] = status.isin({"questionable", "doubtful", "minute_limit", "limited"}).astype(float)
    out["is_suspended"] = (
        status.eq("suspended") | out["suspension_status"].astype(str).str.len().gt(0)
    ).astype(float)
    out["is_injured"] = (
        status.isin({"out", "injured", "doubtful", "questionable"})
        | out["injury_status"].astype(str).str.len().gt(0)
    ).astype(float)
    return out


def apply_manual_projected_lineups(
    players: pd.DataFrame,
    lineups: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = players.copy()
    out["manual_lineup_slot"] = np.nan
    out["manual_lineup_position"] = ""
    out["lineup_status"] = ""
    out["manual_formation"] = ""

    if lineups is None or lineups.empty:
        return out

    table = lineups.copy()
    if "player" in table.columns and "player_name" not in table.columns:
        table["player_name"] = table["player"]
    table["team_key"] = table["team"].map(normalized_key) if "team" in table.columns else ""
    table["player_key"] = (
        table["player_name"].map(normalized_key) if "player_name" in table.columns else ""
    )
    out["team_key"] = out["team"].map(normalized_key)

    for row in table.itertuples(index=False):
        mask = out["team_key"].eq(row.team_key) & out["player_key"].eq(row.player_key)
        if not mask.any():
            continue
        slot = pd.to_numeric(pd.Series([getattr(row, "lineup_slot", "")]), errors="coerce").iloc[0]
        expected_minutes_share = pd.to_numeric(
            pd.Series([getattr(row, "expected_minutes_share", "")]),
            errors="coerce",
        ).iloc[0]
        if not pd.isna(slot):
            out.loc[mask, "manual_lineup_slot"] = float(slot)
        out.loc[mask, "manual_lineup_position"] = str(
            getattr(row, "lineup_position", "") or ""
        ).strip()
        out.loc[mask, "lineup_status"] = str(getattr(row, "lineup_status", "") or "").strip()
        out.loc[mask, "manual_formation"] = str(getattr(row, "formation", "") or "").strip()
        if not pd.isna(expected_minutes_share):
            out.loc[mask, "expected_minutes_share"] = float(np.clip(expected_minutes_share, 0.0, 1.0))

    return out.drop(columns=["team_key"])


def reversed_player_key(value: str) -> str:
    parts = str(value).split()
    if len(parts) < 2:
        return normalized_key(value)
    return normalized_key(" ".join(parts[1:] + parts[:1]))


def _first_token(value: object) -> str:
    parts = str(value).split()
    return parts[0] if parts else ""


def _short_player_key(first_names: object, last_names: object) -> str:
    first = _first_token(first_names)
    last = str(last_names).strip()
    return normalized_key(f"{first} {last}".strip())


def _first_last_token_key(first_names: object, last_names: object) -> str:
    first = _first_token(first_names)
    last = _first_token(last_names)
    return normalized_key(f"{first} {last}".strip())


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _per90(df: pd.DataFrame, column: str, minutes: pd.Series) -> pd.Series:
    values = _numeric(df, column)
    nineties = (minutes / 90.0).clip(lower=1.0)
    return values / nineties


def _cap(series: pd.Series, high: float) -> pd.Series:
    return series.clip(lower=0.0, upper=high)


def add_individual_player_style_scores(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    minutes = _numeric(out, "min")
    starts = _numeric(out, "starts")
    caps = _numeric(out, "international_caps")
    international_goals = _numeric(out, "international_goals")
    height = _numeric(out, "height_cm", 180.0)
    age = _numeric(out, "age_on_2026_06_11", 27.0)

    goals_per90 = _per90(out, "gls", minutes)
    assists_per90 = _per90(out, "ast", minutes)
    shots_per90 = _per90(out, "sh", minutes)
    shots_on_target_per90 = _per90(out, "sot", minutes)
    crosses_per90 = _per90(out, "crs", minutes)
    tackles_won_per90 = _per90(out, "tklw", minutes)
    interceptions_per90 = _per90(out, "int", minutes)
    fouls_drawn_per90 = _per90(out, "fld", minutes)
    saves_per90 = _per90(out, "saves", minutes)
    clean_sheets_per90 = _per90(out, "cs", minutes)
    yellow_cards_per90 = _per90(out, "crdy", minutes)
    red_cards_per90 = _per90(out, "crdr", minutes)

    prime_age_score = 1.0 - ((age - 27.5).abs() / 9.0).clip(upper=1.0)
    out["availability_score"] = (
        _cap(minutes / 2700.0, 1.0) * 0.65
        + _cap(starts / 30.0, 1.0) * 0.35
    )
    out["experience_score"] = np.log1p(caps) + (np.log1p(international_goals) * 0.8)
    out["attacking_score"] = (
        _cap(goals_per90, 1.2) * 2.4
        + _cap(assists_per90, 0.8) * 1.2
        + _cap(shots_on_target_per90, 2.8) * 0.35
        + np.log1p(international_goals) * 0.25
    )
    out["creativity_score"] = (
        _cap(assists_per90, 0.8) * 2.1
        + _cap(crosses_per90, 7.0) * 0.13
        + _cap(fouls_drawn_per90, 4.0) * 0.18
    )
    out["shot_volume_score"] = _cap(shots_per90, 5.0) + _cap(shots_on_target_per90, 2.5) * 0.9
    out["crossing_score"] = _cap(crosses_per90, 8.0)
    out["ball_winning_score"] = _cap(tackles_won_per90, 4.5) + _cap(interceptions_per90, 4.5)
    out["defensive_score"] = (
        _cap(tackles_won_per90, 4.5) * 0.9
        + _cap(interceptions_per90, 4.5) * 1.0
        + _cap(clean_sheets_per90, 0.9) * 0.6
    )
    out["keeper_score"] = (
        _cap(saves_per90, 5.5) * 0.65
        + _cap(clean_sheets_per90, 0.9) * 1.4
        + _cap((height - 180.0) / 18.0, 1.0) * 0.35
    )
    out["physical_score"] = _cap((height - 165.0) / 30.0, 1.0) + prime_age_score
    out["discipline_risk"] = _cap(yellow_cards_per90, 1.2) + _cap(red_cards_per90, 0.25) * 3.0

    position = out["position"].astype(str)
    out.loc[position.eq("GK"), ["attacking_score", "creativity_score", "shot_volume_score"]] *= 0.15
    out.loc[position.eq("FW"), "defensive_score"] *= 0.45
    out.loc[position.eq("DF"), "attacking_score"] *= 0.6
    out.loc[position.eq("MF"), "keeper_score"] = 0.0
    out.loc[position.isin(["DF", "FW"]), "keeper_score"] = 0.0
    contribution_multiplier = (
        _numeric(out, "availability_multiplier", 1.0) * 0.7
        + _numeric(out, "expected_minutes_share", 1.0) * 0.3
    ).clip(lower=0.0, upper=1.0)
    for column in [
        "availability_score",
        "attacking_score",
        "creativity_score",
        "shot_volume_score",
        "crossing_score",
        "ball_winning_score",
        "defensive_score",
        "keeper_score",
        "discipline_risk",
    ]:
        out[column] = out[column] * contribution_multiplier
    return out


def _projected_lineup_players(team_players: pd.DataFrame) -> pd.DataFrame:
    if team_players.empty:
        return pd.DataFrame()

    ranked = team_players.copy()
    if "availability_status" not in ranked.columns:
        ranked["availability_status"] = "available"
    if "availability_multiplier" not in ranked.columns:
        ranked["availability_multiplier"] = 1.0
    if "expected_minutes_share" not in ranked.columns:
        ranked["expected_minutes_share"] = ranked["availability_multiplier"]
    if "lineup_status" not in ranked.columns:
        ranked["lineup_status"] = ""
    ranked["lineup_score"] = ranked.apply(player_lineup_score, axis=1)
    selected_indexes: list[int] = []
    rows = []
    slot = 1

    if "manual_lineup_slot" in ranked.columns:
        manual = ranked[
            ranked["manual_lineup_slot"].notna()
            & ranked["lineup_status"].astype(str).str.casefold().isin(STARTER_STATUSES)
            & ~ranked["availability_status"].astype(str).str.casefold().isin(OUT_STATUSES)
        ].copy()
        if not manual.empty:
            manual["manual_lineup_slot"] = pd.to_numeric(
                manual["manual_lineup_slot"],
                errors="coerce",
            )
            manual = manual.sort_values(["manual_lineup_slot", "lineup_score"], ascending=[True, False])
            for idx, player in manual.head(11).iterrows():
                selected_indexes.append(idx)
                item = player.to_dict()
                item["lineup_slot"] = slot
                item["lineup_position"] = str(player.get("manual_lineup_position") or player.get("position"))
                rows.append(item)
                slot += 1

    for position, count in DEFAULT_FORMATION.items():
        current_count = sum(1 for row in rows if row["lineup_position"] == position)
        remaining_count = max(count - current_count, 0)
        if remaining_count == 0:
            continue
        candidates = (
            ranked[
                ranked["position"].eq(position)
                & ~ranked.index.isin(selected_indexes)
                & ~ranked["availability_status"].astype(str).str.casefold().isin(OUT_STATUSES)
            ]
            .sort_values("lineup_score", ascending=False)
            .head(remaining_count)
        )
        for idx, player in candidates.iterrows():
            selected_indexes.append(idx)
            item = player.to_dict()
            item["lineup_slot"] = slot
            item["lineup_position"] = position
            rows.append(item)
            slot += 1

    if len(rows) < 11:
        remaining = (
            ranked[
                ~ranked.index.isin(selected_indexes)
                & ~ranked["availability_status"].astype(str).str.casefold().isin(OUT_STATUSES)
            ]
            .sort_values("lineup_score", ascending=False)
            .head(11 - len(rows))
        )
        for _, player in remaining.iterrows():
            item = player.to_dict()
            item["lineup_slot"] = slot
            item["lineup_position"] = str(player["position"])
            rows.append(item)
            slot += 1

    return pd.DataFrame(rows)


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return 0.0 if pd.isna(value) else float(value)


def _sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    value = pd.to_numeric(frame[column], errors="coerce").sum()
    return 0.0 if pd.isna(value) else float(value)


def build_lineup_team_style_features(players: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for team, team_players in players.groupby("team", sort=True):
        lineup = _projected_lineup_players(team_players)
        if lineup.empty:
            rows.append({"team": team, **{column: 0.0 for column in LINEUP_TEAM_FEATURE_COLUMNS}})
            continue

        defenders = lineup[lineup["lineup_position"].eq("DF")]
        midfielders = lineup[lineup["lineup_position"].eq("MF")]
        forwards = lineup[lineup["lineup_position"].eq("FW")]
        keepers = lineup[lineup["lineup_position"].eq("GK")]
        bench = (
            team_players.assign(lineup_score=team_players.apply(player_lineup_score, axis=1))
            .sort_values("lineup_score", ascending=False)
            .iloc[11:18]
        )

        rows.append(
            {
                "team": team,
                "projected_lineup_score_sum": _sum(lineup, "lineup_score"),
                "projected_lineup_score_mean": _mean(lineup, "lineup_score"),
                "projected_lineup_club_stats_share": float(lineup["has_club_stats"].fillna(False).mean()),
                "projected_lineup_minutes_sum": _sum(lineup, "min"),
                "projected_lineup_caps_sum": _sum(lineup, "international_caps"),
                "top_11_availability_score": _mean(lineup, "availability_score"),
                "top_11_experience_score": _mean(lineup, "experience_score"),
                "top_11_attacking_score": _mean(lineup, "attacking_score"),
                "top_11_creativity_score": _mean(lineup, "creativity_score"),
                "top_11_shot_volume_score": _mean(lineup, "shot_volume_score"),
                "top_11_crossing_score": _mean(lineup, "crossing_score"),
                "top_11_ball_winning_score": _mean(lineup, "ball_winning_score"),
                "top_11_defensive_score": _mean(lineup, "defensive_score"),
                "top_11_keeper_score": _mean(lineup, "keeper_score"),
                "top_11_physical_score": _mean(lineup, "physical_score"),
                "top_11_discipline_risk": _mean(lineup, "discipline_risk"),
                "starting_gk_keeper_score": _mean(keepers, "keeper_score"),
                "starting_def_defensive_score": _mean(defenders, "defensive_score"),
                "starting_def_physical_score": _mean(defenders, "physical_score"),
                "starting_mf_creativity_score": _mean(midfielders, "creativity_score"),
                "starting_mf_ball_winning_score": _mean(midfielders, "ball_winning_score"),
                "starting_fw_attacking_score": _mean(forwards, "attacking_score"),
                "starting_fw_shot_volume_score": _mean(forwards, "shot_volume_score"),
                "bench_depth_score": _mean(bench, "lineup_score"),
            }
        )

    return pd.DataFrame(rows)


def _parse_formation(value: object) -> tuple[float, float, float]:
    numbers = [
        int(part)
        for part in re.findall(r"\d+", str(value or ""))
        if part.isdigit()
    ]
    if len(numbers) < 2:
        return 0.0, 0.0, 0.0
    defenders = float(numbers[0])
    forwards = float(numbers[-1])
    midfielders = float(sum(numbers[1:-1])) if len(numbers) > 2 else 0.0
    return defenders, midfielders, forwards


def _unit_value(value: object, default: float = 0.5) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return float(np.clip(numeric, 0.0, 1.0))
    text = str(value or "").strip().casefold()
    text_map = {
        "low": 0.25,
        "deep": 0.25,
        "defensive": 0.25,
        "direct": 0.35,
        "balanced": 0.5,
        "medium": 0.5,
        "moderate": 0.5,
        "possession": 0.65,
        "wide": 0.65,
        "high": 0.75,
        "aggressive": 0.8,
    }
    return text_map.get(text, default)


def add_team_tactic_features(
    team_features: pd.DataFrame,
    team_tactics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = team_features.copy()
    for column in TACTIC_TEAM_FEATURE_COLUMNS:
        out[column] = 0.0

    if team_tactics is None or team_tactics.empty:
        return out

    tactics = team_tactics.copy()
    if "team" not in tactics.columns:
        return out

    out["team_key"] = out["team"].map(normalized_key)
    tactics["team_key"] = tactics["team"].map(normalized_key)
    tactic_lookup = {
        row.team_key: row
        for row in tactics.itertuples(index=False)
    }

    for idx, team in out[["team_key"]].itertuples():
        row = tactic_lookup.get(team)
        if row is None:
            continue
        defenders, midfielders, forwards = _parse_formation(getattr(row, "formation", ""))
        out.loc[idx, "tactics_available"] = 1.0
        out.loc[idx, "formation_back_line"] = defenders
        out.loc[idx, "formation_midfield_line"] = midfielders
        out.loc[idx, "formation_forward_line"] = forwards
        out.loc[idx, "formation_defensive_density"] = defenders / 5.0 if defenders else 0.0
        out.loc[idx, "formation_midfield_density"] = midfielders / 5.0 if midfielders else 0.0
        out.loc[idx, "formation_attacking_density"] = forwards / 4.0 if forwards else 0.0
        out.loc[idx, "tactic_pressing_intensity"] = _unit_value(
            getattr(row, "pressing_intensity", ""),
        )
        out.loc[idx, "tactic_defensive_line"] = _unit_value(getattr(row, "defensive_line", ""))
        out.loc[idx, "tactic_tempo"] = _unit_value(getattr(row, "tempo", ""))
        out.loc[idx, "tactic_directness"] = _unit_value(getattr(row, "directness", ""))
        out.loc[idx, "tactic_possession_style"] = _unit_value(
            getattr(row, "possession_style", ""),
        )
        out.loc[idx, "tactic_width"] = _unit_value(getattr(row, "width", ""))
        out.loc[idx, "tactic_set_piece_strength"] = _unit_value(
            getattr(row, "set_piece_strength", ""),
        )
        out.loc[idx, "tactic_transition_speed"] = _unit_value(
            getattr(row, "transition_speed", ""),
        )

    return out.drop(columns=["team_key"])


def build_player_feature_tables(
    squad_players: pd.DataFrame,
    club_stats: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
    manual_lineups: pd.DataFrame | None = None,
    team_tactics: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = squad_players.copy()
    club_stats = club_stats if club_stats is not None else pd.DataFrame()
    players["has_club_stats"] = False
    club_numeric_cols = [
        col
        for col in [
            "mp",
            "starts",
            "min",
            "90s",
            "gls",
            "ast",
            "g_a",
            "g_pk",
            "pk",
            "pkatt",
            "crdy",
            "crdr",
            "sh",
            "sot",
            "crs",
            "tklw",
            "int",
            "fld",
            "ga",
            "saves",
            "cs",
        ]
        if col in club_stats.columns
    ]

    if not club_stats.empty and {"player_key", "club_key"}.issubset(club_stats.columns):
        fifa_name = (
            players["fifa_player_name"]
            if "fifa_player_name" in players.columns
            else players["player_name"]
        )
        first_names = players["first_names"] if "first_names" in players.columns else players["player_name"]
        last_names = players["last_names"] if "last_names" in players.columns else ""
        players["fifa_player_order_key"] = fifa_name.map(normalized_key)
        players["fifa_player_reverse_key"] = fifa_name.map(reversed_player_key)
        players["short_player_key"] = [
            _short_player_key(first, last) for first, last in zip(first_names, last_names)
        ]
        players["first_last_token_key"] = [
            _first_last_token_key(first, last) for first, last in zip(first_names, last_names)
        ]
        players["club_stats_match_key"] = ""

        if club_numeric_cols:
            club_by_player = (
                club_stats.groupby("player_key", as_index=False)[club_numeric_cols]
                .sum(min_count=1)
                .reset_index(drop=True)
            )
            for candidate_key in [
                "player_key",
                "short_player_key",
                "first_last_token_key",
                "fifa_player_reverse_key",
                "fifa_player_order_key",
            ]:
                candidate_matches = players[[candidate_key]].merge(
                    club_by_player,
                    left_on=candidate_key,
                    right_on="player_key",
                    how="left",
                    suffixes=("", "_club"),
                )
                for column in club_numeric_cols:
                    if column not in players.columns:
                        players[column] = np.nan
                    fill_values = candidate_matches[column]
                    newly_matched = players[column].isna() & fill_values.notna()
                    players.loc[newly_matched, column] = fill_values[newly_matched].values
                    players.loc[newly_matched, "club_stats_match_key"] = candidate_key
            players["has_club_stats"] = players[club_numeric_cols].notna().any(axis=1)

    players = apply_player_availability(players, availability)
    players = apply_manual_projected_lineups(players, manual_lineups)
    players = add_individual_player_style_scores(players)

    position_counts = (
        players.pivot_table(
            index="team",
            columns="position",
            values="player_name",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns=lambda col: f"players_{str(col).lower()}")
        .reset_index()
    )

    numeric_aggs = {
        "player_name": "count",
        "has_club_stats": "sum",
        "age_on_2026_06_11": "mean",
        "height_cm": "mean",
        "international_caps": ["sum", "mean"],
        "international_goals": ["sum", "mean"],
        "availability_multiplier": ["sum", "mean"],
        "expected_minutes_share": ["sum", "mean"],
        "is_unavailable": "sum",
        "is_questionable": "sum",
        "is_suspended": "sum",
        "is_injured": "sum",
    }
    optional_numeric = [
        "mp",
        "starts",
        "min",
        "90s",
        "gls",
        "ast",
        "g_a",
        "g_pk",
        "pk",
        "pkatt",
        "crdy",
        "crdr",
        "sh",
        "sot",
        "crs",
        "tklw",
        "int",
        "fld",
        "ga",
        "saves",
        "cs",
    ]
    for column in optional_numeric:
        if column in players.columns:
            numeric_aggs[column] = ["sum", "mean"]

    team_features = players.groupby("team").agg(numeric_aggs)
    team_features.columns = [
        "_".join(str(part) for part in column if part).strip("_")
        for column in team_features.columns
    ]
    team_features = team_features.rename(
        columns={
            "player_name_count": "squad_players",
            "has_club_stats_sum": "club_stats_matched_players",
        }
    ).reset_index()
    team_features = team_features.merge(position_counts, on="team", how="left")

    top_caps = (
        players.sort_values(["team", "international_caps"], ascending=[True, False])
        .groupby("team")
        .head(11)
        .groupby("team")["international_caps"]
        .sum()
        .rename("projected_experience_top_11_caps")
        .reset_index()
    )
    team_features = team_features.merge(top_caps, on="team", how="left")
    lineup_style_features = build_lineup_team_style_features(players)
    team_features = team_features.merge(lineup_style_features, on="team", how="left")
    projection_features = summarize_team_player_projections(players)
    team_features = team_features.merge(projection_features, on="team", how="left")
    for column in LINEUP_TEAM_FEATURE_COLUMNS:
        if column not in team_features.columns:
            team_features[column] = 0.0
        team_features[column] = pd.to_numeric(team_features[column], errors="coerce").fillna(0.0)
    team_features = add_team_tactic_features(team_features, team_tactics)

    return players, team_features.sort_values("team").reset_index(drop=True)


def save_fifa_squad_players(
    input_pdf: Path = FIFA_SQUAD_PDF_FILE,
    output_path: Path = FIFA_SQUAD_PLAYERS_FILE,
) -> Path:
    df = parse_fifa_squad_pdf(
        input_pdf,
        team_aliases=load_fifa_team_aliases(),
        player_aliases=load_player_aliases(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_team_player_features(
    squad_path: Path = FIFA_SQUAD_PLAYERS_FILE,
    club_stats_path: Path = CLUB_PLAYER_STATS_FILE,
    additional_club_stats_path: Path = ADDITIONAL_CLUB_PLAYER_STATS_FILE,
    availability_path: Path = PLAYER_AVAILABILITY_FILE,
    manual_lineups_path: Path = MANUAL_PROJECTED_LINEUPS_FILE,
    team_tactics_path: Path = TEAM_TACTICS_FILE,
    output_path: Path = TEAM_PLAYER_FEATURES_FILE,
) -> tuple[Path, Path]:
    squad_players = pd.read_csv(squad_path, parse_dates=["date_of_birth"])
    club_stats = combine_club_player_stats(
        load_club_player_stats(club_stats_path),
        load_club_player_stats(additional_club_stats_path),
    )
    availability = load_optional_manual_table(availability_path)
    manual_lineups = load_optional_manual_table(manual_lineups_path)
    team_tactics = load_optional_manual_table(team_tactics_path)
    player_features, team_features = build_player_feature_tables(
        squad_players,
        club_stats,
        availability=availability,
        manual_lineups=manual_lineups,
        team_tactics=team_tactics,
    )
    player_path = output_path.parent / "player_features_2026.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_features.to_csv(player_path, index=False)
    team_features.to_csv(output_path, index=False)
    return player_path, output_path
