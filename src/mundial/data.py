"""Ingestion y construccion temporal de features sin fuga de informacion."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mundial.config import CONFIG_DIR, PROCESSED_DIR, RAW_DIR, load_aliases

PLAYER_ATTRIBUTES = ("overall", "pace", "shooting", "defending", "physical")
STATIC_FEATURES = (
    "rank_a", "rank_b", "rank_diff",
    "goals_for_last10_a", "goals_against_last10_a",
    "goals_for_last10_b", "goals_against_last10_b",
    "h2h_wins_a", "h2h_draws", "h2h_wins_b", "h2h_goal_difference",
    "overall_a", "pace_a", "shooting_a", "defending_a", "physical_a",
    "overall_b", "pace_b", "shooting_b", "defending_b", "physical_b",
    "players_imputed_a", "players_imputed_b", "neutral",
)
SEQUENCE_FEATURES = ("goals_for", "goals_against", "won", "drawn", "neutral")


def normalize_team(name: object, aliases: dict[str, str] | None = None) -> str:
    """Normaliza espacios y aplica el catalogo comun de equivalencias."""
    aliases = aliases or load_aliases()
    clean = re.sub(r"\s+", " ", str(name)).strip()
    return aliases.get(clean, clean)


def _find_file(root: Path, patterns: tuple[str, ...]) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No se encontro ninguno de {patterns} dentro de {root}")
    return sorted(candidates, key=lambda path: (len(path.parts), path.name))[0]


def load_results(raw_dir: Path = RAW_DIR, as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    path = _find_file(raw_dir, ("results.csv", "*results*.csv"))
    frame = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"El archivo {path} no tiene las columnas requeridas: {required}")
    aliases = load_aliases()
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["home_team"] = frame["home_team"].map(lambda value: normalize_team(value, aliases))
    frame["away_team"] = frame["away_team"].map(lambda value: normalize_team(value, aliases))
    if "neutral" not in frame:
        frame["neutral"] = False
    if "tournament" not in frame:
        frame["tournament"] = "Unknown"
    frame["neutral"] = frame["neutral"].fillna(False).astype(bool)
    frame["tournament"] = frame["tournament"].fillna("Unknown").astype(str)
    frame = frame.dropna(subset=["date", "home_score", "away_score"])
    frame = frame.loc[frame["date"] >= "2015-01-01"].sort_values("date").reset_index(drop=True)
    if as_of_date is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(as_of_date)].reset_index(drop=True)
    frame["match_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


def append_official_tournament_results(results: pd.DataFrame, state_path: Path) -> pd.DataFrame:
    """Añade resultados aprobados usando solo datos conocidos al inicio de cada partido."""
    from mundial.tournament_state import load_tournament_state

    state = load_tournament_state(state_path, required=True)
    rows = []
    for match in state.finished_matches.values():
        if not match.score_90 or not match.team_a or not match.team_b:
            continue
        rows.append({
            "date": pd.Timestamp(match.kickoff).tz_localize(None),
            "home_team": match.team_a, "away_team": match.team_b,
            "home_score": match.score_90[0], "away_score": match.score_90[1],
            "neutral": True, "tournament": "FIFA World Cup 2026",
            "official_match_id": match.match_id,
        })
    if not rows:
        return results
    official = pd.DataFrame(rows)
    # Exact duplicates are harmless source overlap, but must not train twice.
    combined = pd.concat([results, official], ignore_index=True)
    return combined.drop_duplicates(
        ["date", "home_team", "away_team", "home_score", "away_score"], keep="last"
    ).sort_values("date").reset_index(drop=True).assign(match_id=lambda frame: np.arange(len(frame), dtype=np.int64))


def load_rankings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    aliases = load_aliases()
    ranking_dir = raw_dir / "fifa_rankings"
    paths = sorted(ranking_dir.glob("*.csv")) if ranking_dir.exists() else sorted(raw_dir.rglob("*ranking*.csv"))
    normalized: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        date_column = next((name for name in ("rank_date", "date") if name in frame), None)
        team_column = next((name for name in ("country_full", "team", "country") if name in frame), None)
        rank_column = next((name for name in ("rank", "ranking") if name in frame), None)
        if not all((date_column, team_column, rank_column)):
            continue
        normalized.append(frame[[date_column, team_column, rank_column]].rename(
            columns={date_column: "rank_date", team_column: "team", rank_column: "rank"}
        ))
    if not normalized:
        raise FileNotFoundError("No se encontraron snapshots con fecha, seleccion y ranking")
    result = pd.concat(normalized, ignore_index=True)
    result["rank_date"] = pd.to_datetime(result["rank_date"], errors="coerce")
    result["team"] = result["team"].map(lambda value: normalize_team(value, aliases))
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    return result.dropna().sort_values(["team", "rank_date"]).drop_duplicates(["team", "rank_date"], keep="last")


def build_player_snapshots(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    aliases = load_aliases()
    consolidated = sorted(raw_dir.rglob("male_players.csv"))
    files = consolidated or (sorted(raw_dir.rglob("*players_*.csv")) + sorted(raw_dir.rglob("players_*.csv")))
    files = list(dict.fromkeys(files))
    rows: list[dict[str, object]] = []
    for path in files:
        match = re.search(r"(?:players_|male_players_)(\d{2,4})", path.name)
        header = pd.read_csv(path, nrows=0).columns
        physical_column = "physic" if "physic" in header else "physical"
        desired = [
            column for column in (
                "fifa_version", "nationality_name", "nationality", "overall",
                "pace", "shooting", "defending", physical_column,
            ) if column in header
        ]
        frame = pd.read_csv(path, usecols=desired, low_memory=False)
        nationality = next((column for column in ("nationality_name", "nationality") if column in frame), None)
        if nationality is None or "overall" not in frame:
            continue
        columns = {
            "overall": "overall", "pace": "pace", "shooting": "shooting",
            "defending": "defending", physical_column: "physical",
        }
        available = [column for column in columns if column in frame]
        if "fifa_version" in frame:
            editions = ((2000 + int(version), edition) for version, edition in frame.groupby("fifa_version"))
        elif match:
            raw_year = int(match.group(1))
            editions = ((raw_year if raw_year > 1900 else 2000 + raw_year, frame),)
        else:
            continue
        for year, edition in editions:
            selected = edition[[nationality, *available]].copy()
            selected[nationality] = selected[nationality].map(lambda value: normalize_team(value, aliases))
            for column in available:
                selected[column] = pd.to_numeric(selected[column], errors="coerce")
            for team, players in selected.groupby(nationality, sort=False):
                squad = players.nlargest(23, "overall")
                row: dict[str, object] = {"snapshot_year": year, "team": team, "squad_size": len(squad)}
                for source, target in columns.items():
                    row[target] = float(squad[source].mean()) if source in squad else np.nan
                row["players_imputed"] = float(len(squad) < 23)
                rows.append(row)
    if not rows:
        raise FileNotFoundError("No se encontraron archivos players_XX.csv del dataset FIFA/FC")
    snapshots = pd.DataFrame(rows).sort_values(["snapshot_year", "team"])
    for year, indices in snapshots.groupby("snapshot_year").groups.items():
        for attribute in PLAYER_ATTRIBUTES:
            median = snapshots.loc[indices, attribute].median()
            snapshots.loc[indices, attribute] = snapshots.loc[indices, attribute].fillna(median)
    return snapshots.drop_duplicates(["snapshot_year", "team"], keep="last")


def _ranking_asof(matches: pd.DataFrame, rankings: pd.DataFrame, team_column: str, output: str) -> pd.Series:
    values = pd.Series(np.nan, index=matches.index, dtype=float)
    for team, indices in matches.groupby(team_column).groups.items():
        historical = rankings.loc[rankings["team"] == team, ["rank_date", "rank"]].sort_values("rank_date")
        if historical.empty:
            continue
        left = matches.loc[indices, ["date"]].assign(_original_index=indices).sort_values("date")
        joined = pd.merge_asof(left, historical, left_on="date", right_on="rank_date", direction="backward")
        values.loc[joined["_original_index"].to_numpy()] = joined["rank"].to_numpy()
    values.name = output
    return values


def _rolling_team_history(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["match_id", "date", "home_team", "home_score", "away_score", "neutral"]].rename(
        columns={"home_team": "team", "home_score": "goals_for", "away_score": "goals_against"}
    )
    away = matches[["match_id", "date", "away_team", "away_score", "home_score", "neutral"]].rename(
        columns={"away_team": "team", "away_score": "goals_for", "home_score": "goals_against"}
    )
    history = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "match_id"])
    history["won"] = (history["goals_for"] > history["goals_against"]).astype(float)
    history["drawn"] = (history["goals_for"] == history["goals_against"]).astype(float)
    for column in ("goals_for", "goals_against"):
        history[f"{column}_last10"] = history.groupby("team")[column].transform(
            lambda values: values.shift(1).rolling(10, min_periods=1).mean()
        )
    return history


def _h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    state: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"wins_first": 0, "draws": 0, "wins_second": 0, "gd_first": 0})
    output: list[dict[str, float]] = []
    for row in matches.itertuples(index=False):
        first, second = sorted((row.home_team, row.away_team))
        key = (first, second)
        previous = state[key]
        home_is_first = row.home_team == first
        output.append({
            "h2h_wins_a": previous["wins_first"] if home_is_first else previous["wins_second"],
            "h2h_draws": previous["draws"],
            "h2h_wins_b": previous["wins_second"] if home_is_first else previous["wins_first"],
            "h2h_goal_difference": previous["gd_first"] if home_is_first else -previous["gd_first"],
        })
        goals_first = row.home_score if home_is_first else row.away_score
        goals_second = row.away_score if home_is_first else row.home_score
        if goals_first > goals_second:
            previous["wins_first"] += 1
        elif goals_first < goals_second:
            previous["wins_second"] += 1
        else:
            previous["draws"] += 1
        previous["gd_first"] += goals_first - goals_second
    return pd.DataFrame(output, index=matches.index)


def _attach_players(matches: pd.DataFrame, snapshots: pd.DataFrame, team_column: str, suffix: str) -> pd.DataFrame:
    available_years = sorted(snapshots["snapshot_year"].unique())
    target_year = matches["date"].dt.year.map(lambda year: max((item for item in available_years if item <= year), default=available_years[0]))
    lookup = matches[[team_column]].copy()
    lookup["snapshot_year"] = target_year
    lookup["_index"] = matches.index
    joined = lookup.merge(snapshots, left_on=[team_column, "snapshot_year"], right_on=["team", "snapshot_year"], how="left")
    joined = joined.set_index("_index").reindex(matches.index)
    edition_medians = snapshots.groupby("snapshot_year")[[*PLAYER_ATTRIBUTES]].median()
    for attribute in PLAYER_ATTRIBUTES:
        fallback = target_year.map(edition_medians[attribute])
        matches[f"{attribute}_{suffix}"] = joined[attribute].fillna(fallback).to_numpy()
    matches[f"players_imputed_{suffix}"] = joined["players_imputed"].fillna(1.0).to_numpy()
    return matches


def build_match_dataset(
    results: pd.DataFrame,
    rankings: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    matches = results.copy().sort_values(["date", "match_id"]).reset_index(drop=True)
    history = _rolling_team_history(matches)
    for side, team_column in (("a", "home_team"), ("b", "away_team")):
        side_history = history[["match_id", "team", "goals_for_last10", "goals_against_last10"]]
        side_history = side_history.rename(columns={
            "goals_for_last10": f"goals_for_last10_{side}",
            "goals_against_last10": f"goals_against_last10_{side}",
        })
        matches = matches.merge(side_history, left_on=["match_id", team_column], right_on=["match_id", "team"], how="left").drop(columns="team")
    matches["rank_a"] = _ranking_asof(matches, rankings, "home_team", "rank_a")
    matches["rank_b"] = _ranking_asof(matches, rankings, "away_team", "rank_b")
    matches["rank_diff"] = matches["rank_b"] - matches["rank_a"]
    matches = pd.concat([matches, _h2h_features(matches)], axis=1)
    matches = _attach_players(matches, snapshots, "home_team", "a")
    matches = _attach_players(matches, snapshots, "away_team", "b")
    matches["result"] = np.select(
        [matches["home_score"] > matches["away_score"], matches["home_score"] == matches["away_score"]],
        [0, 1],
        default=2,
    ).astype(np.int8)
    matches["split"] = np.select(
        [
            matches["date"] < "2020-01-01",
            matches["date"] < "2022-11-20",
            (matches["date"] <= "2022-12-18") & matches["tournament"].str.contains("World Cup", case=False, na=False),
        ],
        ["train", "validation", "test"],
        default="production",
    )
    sequences_a, sequences_b = build_sequences(matches, history)
    usable = matches[list(STATIC_FEATURES)].notna().mean(axis=1) >= 0.75
    matches = matches.loc[usable].reset_index(drop=True)
    return matches, sequences_a[usable.to_numpy()], sequences_b[usable.to_numpy()]


def build_sequences(matches: pd.DataFrame, history: pd.DataFrame, lookback: int = 10) -> tuple[np.ndarray, np.ndarray]:
    by_team: dict[str, list[np.ndarray]] = defaultdict(list)
    history_by_match = {(row.match_id, row.team): row for row in history.itertuples(index=False)}
    sequence_a: list[np.ndarray] = []
    sequence_b: list[np.ndarray] = []
    for match in matches.itertuples(index=False):
        for team, target in ((match.home_team, sequence_a), (match.away_team, sequence_b)):
            previous = by_team[team][-lookback:]
            padding = [np.zeros(len(SEQUENCE_FEATURES), dtype=np.float32)] * (lookback - len(previous))
            target.append(np.stack([*padding, *previous]))
        for team in (match.home_team, match.away_team):
            row = history_by_match[(match.match_id, team)]
            by_team[team].append(np.array([row.goals_for, row.goals_against, row.won, row.drawn, float(row.neutral)], dtype=np.float32))
    return np.stack(sequence_a), np.stack(sequence_b)


def save_processed_dataset(
    output_dir: Path = PROCESSED_DIR,
    as_of_date: str | pd.Timestamp | None = None,
    tournament_state_path: Path | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(as_of_date=as_of_date)
    if tournament_state_path is not None:
        results = append_official_tournament_results(results, tournament_state_path)
        if as_of_date is not None:
            results = results.loc[results["date"] <= pd.Timestamp(as_of_date)].reset_index(drop=True)
    matches, seq_a, seq_b = build_match_dataset(results, load_rankings(), build_player_snapshots())
    table_path = output_dir / "matches.parquet"
    sequence_path = output_dir / "sequences.npz"
    matches.to_parquet(table_path, index=False)
    np.savez_compressed(sequence_path, team_a=seq_a, team_b=seq_b, match_id=matches["match_id"].to_numpy())
    metadata = {
        "rows": len(matches), "static_features": list(STATIC_FEATURES),
        "sequence_features": list(SEQUENCE_FEATURES), "lookback": 10,
        "as_of_date": None if as_of_date is None else str(pd.Timestamp(as_of_date).date()),
        "tournament_state": str(tournament_state_path) if tournament_state_path else None,
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return table_path, sequence_path
