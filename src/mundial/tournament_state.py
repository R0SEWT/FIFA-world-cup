"""Estado oficial, validacion y promocion atomica del Mundial 2026."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from mundial.config import ARTIFACTS_DIR, load_aliases, load_groups

FIFA_SOURCE_URL = os.environ.get(
    "MUNDIAL_FIFA_SOURCE_URL",
    "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures",
)
STATE_PATH = ARTIFACTS_DIR / "tournament_state.json"
CANDIDATE_PATH = ARTIFACTS_DIR / "tournament_state.candidate.json"
VALID_STATUSES = {"scheduled", "in_progress", "finished"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("hash", None)
    raw = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class TournamentMatch:
    match_id: str
    phase: str
    kickoff: str
    status: str = "scheduled"
    team_a: str | None = None
    team_b: str | None = None
    group: str | None = None
    score_90: tuple[int, int] | None = None
    score_extra_time: tuple[int, int] | None = None
    penalties: tuple[int, int] | None = None
    winner: str | None = None
    source_a: str | None = None
    source_b: str | None = None

    @property
    def is_finished(self) -> bool:
        return self.status == "finished"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], match_id: str | None = None) -> "TournamentMatch":
        teams = raw.get("teams") if isinstance(raw.get("teams"), Mapping) else {}
        score = raw.get("score") if isinstance(raw.get("score"), Mapping) else {}

        def pair(value: Any) -> tuple[int, int] | None:
            if isinstance(value, Mapping):
                value = [value.get("a", value.get("home")), value.get("b", value.get("away"))]
            if isinstance(value, (list, tuple)) and len(value) == 2 and all(v is not None for v in value):
                return int(value[0]), int(value[1])
            return None

        status_aliases = {
            "scheduled": "scheduled", "not_started": "scheduled", "future": "scheduled",
            "in_progress": "in_progress", "live": "in_progress", "in play": "in_progress",
            "finished": "finished", "completed": "finished", "final": "finished",
        }
        raw_status = str(raw.get("status", "scheduled")).lower().replace("-", "_")
        return cls(
            match_id=str(match_id or raw.get("match_id") or raw.get("id")),
            phase=str(raw.get("phase") or raw.get("stage") or raw.get("round") or ""),
            group=raw.get("group"),
            kickoff=str(raw.get("kickoff") or raw.get("date") or raw.get("start_time") or ""),
            status=status_aliases.get(raw_status, raw_status),
            team_a=raw.get("team_a") or raw.get("home_team") or teams.get("a") or teams.get("home"),
            team_b=raw.get("team_b") or raw.get("away_team") or teams.get("b") or teams.get("away"),
            score_90=pair(raw.get("score_90") or score.get("90") or score.get("regular")),
            score_extra_time=pair(raw.get("score_extra_time") or score.get("extra_time")),
            penalties=pair(raw.get("penalties") or score.get("penalties")),
            winner=raw.get("winner"),
            source_a=raw.get("source_a") or (raw.get("dependencies") or {}).get("a"),
            source_b=raw.get("source_b") or (raw.get("dependencies") or {}).get("b"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TournamentState:
    generated_at: str
    source_url: str
    source_updated_at: str | None
    matches: dict[str, TournamentMatch]
    hash: str = ""
    approved_at: str | None = None
    stale: bool = False
    update_error: str | None = None
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TournamentState":
        source = raw.get("source") if isinstance(raw.get("source"), Mapping) else {}
        match_payload = raw.get("matches", {})
        if isinstance(match_payload, list):
            ids = [str(item.get("match_id") or item.get("id")) for item in match_payload]
            if len(ids) != len(set(ids)):
                raise ValueError("el snapshot contiene identificadores de partido duplicados")
            matches = {key: TournamentMatch.from_dict(item) for key, item in zip(ids, match_payload, strict=True)}
        elif isinstance(match_payload, Mapping):
            matches = {str(key): TournamentMatch.from_dict(value, str(key)) for key, value in match_payload.items()}
        else:
            raise ValueError("matches debe ser una lista o un objeto")
        return cls(
            generated_at=str(raw.get("generated_at") or raw.get("timestamp") or ""),
            source_url=str(raw.get("source_url") or source.get("url") or source.get("name") or ""),
            source_updated_at=raw.get("source_updated_at") or source.get("updated_at"),
            matches=matches,
            hash=str(raw.get("hash") or ""), approved_at=raw.get("approved_at"),
            stale=bool(raw.get("stale", False)), update_error=raw.get("update_error"),
            schema_version=int(raw.get("schema_version", 1)), metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "approved_at": self.approved_at,
            "source": {"name": "FIFA", "url": self.source_url, "updated_at": self.source_updated_at},
            "stale": self.stale,
            "update_error": self.update_error,
            "matches": {key: value.to_dict() for key, value in sorted(self.matches.items(), key=lambda x: _match_number(x[0]))},
            "metadata": self.metadata,
        }
        payload["hash"] = _canonical_hash(payload)
        return payload

    @property
    def finished_matches(self) -> dict[str, TournamentMatch]:
        return {key: match for key, match in self.matches.items() if match.is_finished}

    def match_for_teams(self, team_a: str, team_b: str, *, phase: str | None = None) -> TournamentMatch | None:
        pair = {team_a, team_b}
        for match in self.matches.values():
            if {match.team_a, match.team_b} == pair and (phase is None or match.phase == phase):
                return match
        return None

    def eliminated_teams(self, groups: Mapping[str, list[str]] | None = None) -> set[str]:
        eliminated: set[str] = set()
        for match in self.matches.values():
            if match.is_finished and match.winner and _is_knockout(match.phase):
                loser = match.team_b if match.winner == match.team_a else match.team_a
                if loser:
                    eliminated.add(loser)
        # Group elimination is only definitive when the complete group stage is known.
        group_matches = [m for m in self.matches.values() if _is_group(m.phase)]
        if len(group_matches) == 72 and all(m.is_finished for m in group_matches):
            from mundial.simulation import GROUPS, _PlayedGame, _Stats, _apply_game, _rank_thirds
            rng = __import__("numpy").random.default_rng(0)
            configured = groups or load_groups()
            ranked: dict[str, list[_Stats]] = {}
            thirds = []
            for group in GROUPS:
                stats = {team: _Stats(team) for team in configured[group]}
                games = []
                for match in group_matches:
                    if match.group == group and match.score_90:
                        game = _PlayedGame(match.team_a, match.team_b, *match.score_90)  # type: ignore[arg-type]
                        games.append(game); _apply_game(stats, game)
                from mundial.simulation import _rank_group
                ranked[group] = _rank_group(stats, games, rng)
                thirds.append((group, ranked[group][2]))
            qualified = {row.team for group in GROUPS for row in ranked[group][:2]}
            qualified |= {row.team for _, row in _rank_thirds(thirds, rng)[:8]}
            eliminated |= set(sum((list(v) for v in configured.values()), [])) - qualified
        return eliminated


def _match_number(match_id: str) -> int:
    try:
        return int(match_id.upper().removeprefix("M"))
    except ValueError:
        return 10_000


def _is_group(phase: str) -> bool:
    return phase.strip().lower() in {"group", "group stage", "fase de grupos", "grupos"}


def _is_knockout(phase: str) -> bool:
    return not _is_group(phase)


def _dependency_winner(source: str | None, matches: Mapping[str, TournamentMatch]) -> str | None:
    if not source:
        return None
    words = source.replace("-", " ").split()
    match_id = next((word.upper() for word in words if word.upper().startswith("M") and word[1:].isdigit()), None)
    if match_id and any(token in source.lower() for token in ("winner", "ganador")):
        return matches.get(match_id).winner if matches.get(match_id) else None
    if match_id and any(token in source.lower() for token in ("loser", "perdedor")):
        parent = matches.get(match_id)
        if parent and parent.winner:
            return parent.team_b if parent.winner == parent.team_a else parent.team_a
    return None


def validate_state(
    candidate: TournamentState,
    approved: TournamentState | None = None,
    *,
    now: datetime | None = None,
    require_complete: bool = True,
) -> None:
    """Valida invariantes deportivas y monotonicidad respecto a produccion."""
    errors: list[str] = []
    now = now or datetime.now(timezone.utc)
    if require_complete and (len(candidate.matches) != 104 or set(candidate.matches) != {f"M{i}" for i in range(1, 105)}):
        errors.append("el snapshot debe contener exactamente M1..M104")
    seen_ids: set[str] = set()
    for match_id, match in candidate.matches.items():
        if match_id in seen_ids:
            errors.append(f"resultado duplicado: {match_id}")
        seen_ids.add(match_id)
        if match.status not in VALID_STATUSES:
            errors.append(f"{match_id}: estado desconocido {match.status}")
        try:
            future = bool(match.kickoff) and _parse_time(match.kickoff) > now
        except ValueError:
            errors.append(f"{match_id}: horario invalido")
            future = False
        if future and match.is_finished:
            errors.append(f"{match_id}: un partido futuro no puede estar finalizado")
        if match.is_finished:
            if not match.team_a or not match.team_b or not match.score_90:
                errors.append(f"{match_id}: resultado final incompleto")
                continue
            if _is_group(match.phase):
                expected = match.team_a if match.score_90[0] > match.score_90[1] else match.team_b if match.score_90[1] > match.score_90[0] else None
                if match.winner != expected:
                    errors.append(f"{match_id}: ganador no coincide con marcador a 90 minutos")
            else:
                decisive = match.penalties or match.score_extra_time or match.score_90
                expected = match.team_a if decisive[0] > decisive[1] else match.team_b if decisive[1] > decisive[0] else None
                if expected is None or match.winner != expected:
                    errors.append(f"{match_id}: ganador no coincide con prorroga o penaltis")
        for side, source in ((match.team_a, match.source_a), (match.team_b, match.source_b)):
            expected = _dependency_winner(source, candidate.matches)
            if expected and side != expected:
                errors.append(f"{match_id}: {side} no coincide con dependencia {source}")
    if approved:
        for match_id, old in approved.matches.items():
            new = candidate.matches.get(match_id)
            if new is None:
                errors.append(f"{match_id}: no puede desaparecer")
                continue
            if old.is_finished and new.to_dict() != old.to_dict():
                errors.append(f"{match_id}: un resultado aprobado no puede retroceder ni cambiar")
            if old.status == "in_progress" and new.status == "scheduled":
                errors.append(f"{match_id}: retroceso de estado")
    eliminated: set[str] = set()
    for match in sorted(candidate.matches.values(), key=lambda m: _match_number(m.match_id)):
        if match.match_id != "M103" and (match.team_a in eliminated or match.team_b in eliminated):
            errors.append(f"{match.match_id}: reaparece un equipo eliminado")
        if match.is_finished and match.winner and _is_knockout(match.phase):
            loser = match.team_b if match.winner == match.team_a else match.team_a
            if loser:
                eliminated.add(loser)
    if errors:
        raise ValueError("Estado del torneo invalido:\n- " + "\n- ".join(errors))


def load_tournament_state(path: Path = STATE_PATH, *, required: bool = False) -> TournamentState | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    state = TournamentState.from_dict(raw)
    expected = str(raw.get("hash") or "")
    if expected and expected != _canonical_hash(raw):
        raise ValueError(f"Hash invalido en {path}")
    return state


def write_state(state: TournamentState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_dict()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def approve_candidate(candidate_path: Path = CANDIDATE_PATH, state_path: Path = STATE_PATH) -> TournamentState:
    candidate = load_tournament_state(candidate_path, required=True)
    approved = load_tournament_state(state_path)
    validate_state(candidate, approved)
    candidate.approved_at = utc_now()
    candidate.stale = False
    candidate.update_error = None
    if state_path.exists():
        backup = state_path.with_suffix(f".backup-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
        shutil.copy2(state_path, backup)
    write_state(candidate, state_path)
    return candidate


def parse_fifa_payload(payload: Any, *, source_url: str = FIFA_SOURCE_URL) -> TournamentState:
    """Normaliza JSON de FIFA; admite envoltorios usados por APIs y fixtures."""
    if isinstance(payload, Mapping):
        items: Any = payload.get("matches") or payload.get("results") or payload.get("data")
        if isinstance(items, Mapping):
            items = items.get("matches") or items.get("results") or items.get("items")
        updated = payload.get("updated_at") or payload.get("lastUpdated")
    else:
        items, updated = payload, None
    if not isinstance(items, list):
        raise ValueError("FIFA cambio el formato: no se encontro una lista de partidos")
    aliases = load_aliases()
    matches: dict[str, TournamentMatch] = {}
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        match = TournamentMatch.from_dict(raw)
        if not match.match_id or match.match_id == "None":
            raise ValueError("FIFA devolvio un partido sin identificador oficial")
        if match.match_id in matches:
            raise ValueError(f"FIFA devolvio el resultado duplicado {match.match_id}")
        normalized = asdict(match)
        normalized["team_a"] = aliases.get(match.team_a, match.team_a) if match.team_a else None
        normalized["team_b"] = aliases.get(match.team_b, match.team_b) if match.team_b else None
        normalized["winner"] = aliases.get(match.winner, match.winner) if match.winner else None
        matches[match.match_id] = TournamentMatch(**normalized)
    return TournamentState(utc_now(), source_url, str(updated) if updated else None, matches)


def fetch_fifa_state(url: str = FIFA_SOURCE_URL, *, timeout: int = 30) -> TournamentState:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "mundial-2026-ai/0.2"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_fifa_payload(json.loads(response.read()), source_url=url)


def pending_team_pairs(state: TournamentState) -> set[frozenset[str]]:
    return {
        frozenset((match.team_a, match.team_b))
        for match in state.matches.values()
        if not match.is_finished and match.team_a and match.team_b
    }
