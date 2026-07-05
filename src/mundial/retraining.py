"""Gate temporal para candidatos de modelo al cierre de cada fase."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mundial.tournament_state import TournamentState

PHASES: tuple[tuple[str, range], ...] = (
    ("groups", range(1, 73)),
    ("round_of_32", range(73, 89)),
    ("round_of_16", range(89, 97)),
    ("quarterfinals", range(97, 101)),
    ("semifinals", range(101, 103)),
)


def closed_phases(state: TournamentState) -> list[str]:
    return [
        name for name, numbers in PHASES
        if all(state.matches.get(f"M{number}") and state.matches[f"M{number}"].is_finished for number in numbers)
    ]


def next_retraining_phase(state: TournamentState, manifest: Mapping[str, Any]) -> str | None:
    trained = set(manifest.get("trained_phases", []))
    return next((phase for phase in closed_phases(state) if phase not in trained), None)


def passes_promotion_gate(
    candidate: Mapping[str, float], incumbent: Mapping[str, float], *, max_ece_degradation: float = 0.01
) -> bool:
    required = {"log_loss", "brier", "ece"}
    if not required <= candidate.keys() or not required <= incumbent.keys():
        raise ValueError("El gate requiere log_loss, brier y ece")
    return (
        candidate["log_loss"] < incumbent["log_loss"]
        and candidate["brier"] < incumbent["brier"]
        and candidate["ece"] <= incumbent["ece"] + max_ece_degradation
    )


def record_phase_gate(
    manifest_path: Path,
    *,
    phase: str,
    data_cutoff: str,
    candidate_metrics: Mapping[str, float],
    incumbent_metrics: Mapping[str, float],
    candidate_version: str,
) -> bool:
    """Registra una decision auditable; no entrena ni promociona artefactos por si sola."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    promoted = passes_promotion_gate(candidate_metrics, incumbent_metrics)
    previous_version = manifest.get("model_version") or manifest.get("version")
    entry = {
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": phase, "data_cutoff": data_cutoff,
        "terminal_temporal_fold": phase,
        "candidate_version": candidate_version,
        "candidate_metrics": dict(candidate_metrics), "incumbent_metrics": dict(incumbent_metrics),
        "promoted": promoted, "rollback_version": previous_version,
    }
    manifest.setdefault("phase_gates", []).append(entry)
    manifest.setdefault("trained_phases", []).append(phase)
    if promoted:
        manifest["model_version"] = candidate_version
        manifest["data_cutoff"] = data_cutoff
        manifest["phase"] = phase
        manifest["metrics"] = dict(candidate_metrics)
        manifest["rollback_version"] = previous_version
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=manifest_path.parent, encoding="utf-8", delete=False) as handle:
        json.dump(manifest, handle, indent=2)
        temporary = Path(handle.name)
    temporary.replace(manifest_path)
    return promoted
