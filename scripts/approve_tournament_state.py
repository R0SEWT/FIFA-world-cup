from __future__ import annotations

import argparse
import json
import subprocess
import sys
import dataclasses
from pathlib import Path

from mundial.config import ARTIFACTS_DIR
from mundial.tournament_state import CANDIDATE_PATH, STATE_PATH, approve_candidate, utc_now
from mundial.polymarket import filter_pending_markets, load_snapshot
from mundial.retraining import next_retraining_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida y promueve atomically el candidato FIFA.")
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--output", type=Path, default=STATE_PATH)
    parser.add_argument("--skip-polymarket", action="store_true")
    args = parser.parse_args()
    state = approve_candidate(args.candidate, args.output)
    refresh_error = None
    if not args.skip_polymarket:
        command = [
            sys.executable, str(Path(__file__).with_name("generate_polymarket_snapshot.py")),
            "--output", str(ARTIFACTS_DIR / "polymarket_snapshot.json"),
            "--tournament-state", str(args.output),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode:
            refresh_error = (result.stderr or result.stdout).strip()
            # An old quote for a newly completed match must never leak back
            # into inference, even when Polymarket itself is unavailable.
            snapshot_path = ARTIFACTS_DIR / "polymarket_snapshot.json"
            if snapshot_path.exists():
                retained = filter_pending_markets(load_snapshot(snapshot_path), state)
                old = json.loads(snapshot_path.read_text(encoding="utf-8"))
                old["markets"] = [dataclasses.asdict(market) for market in retained]
                old["tournament_state_hash"] = state.to_dict()["hash"]
                old["refresh_error"] = refresh_error
                snapshot_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    manifest_path = ARTIFACTS_DIR / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    publication = {
        "generated_at": utc_now(), "tournament_state_hash": state.to_dict()["hash"],
        "source": state.source_url, "polymarket_refresh_error": refresh_error,
        "model_retrained": False, "retraining_phase_due": next_retraining_phase(state, manifest),
    }
    (ARTIFACTS_DIR / "simulation_update.json").write_text(json.dumps(publication, indent=2), encoding="utf-8")
    print(f"Estado aprobado: {len(state.finished_matches)} finalizados; hash={publication['tournament_state_hash']}")
    if refresh_error:
        print("Polymarket no disponible; se conserva el predictor DL y el ultimo snapshot.", file=sys.stderr)


if __name__ == "__main__":
    main()
