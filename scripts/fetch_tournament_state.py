from __future__ import annotations

import argparse
import json
from pathlib import Path

from mundial.tournament_state import (
    CANDIDATE_PATH, FIFA_SOURCE_URL, STATE_PATH, fetch_fifa_state,
    load_tournament_state, parse_fifa_payload, utc_now, validate_state, write_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un candidato FIFA sin modificar el estado aprobado.")
    parser.add_argument("--url", default=FIFA_SOURCE_URL)
    parser.add_argument("--input", type=Path, help="JSON descargado (util para auditoria y pruebas)")
    parser.add_argument("--output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--approved", type=Path, default=STATE_PATH)
    parser.add_argument("--allow-partial", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    status_path = args.approved.with_name("tournament_state_update.json")
    try:
        if args.input:
            state = parse_fifa_payload(json.loads(args.input.read_text(encoding="utf-8")), source_url=args.url)
        else:
            state = fetch_fifa_state(args.url)
        approved = load_tournament_state(args.approved)
        validate_state(state, approved, require_complete=not args.allow_partial)
        write_state(state, args.output)
        status_path.write_text(json.dumps({"attempted_at": utc_now(), "ok": True, "error": None}, indent=2), encoding="utf-8")
        print(f"Candidato valido: {len(state.matches)} partidos -> {args.output}")
    except Exception as error:
        args.output.unlink(missing_ok=True)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"attempted_at": utc_now(), "ok": False, "error": str(error)}, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(f"No se modifico produccion: {error}") from error


if __name__ == "__main__":
    main()
