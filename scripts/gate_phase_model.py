from __future__ import annotations

import argparse
import json
from pathlib import Path

from mundial.config import ARTIFACTS_DIR
from mundial.retraining import record_phase_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Aplica el gate temporal de promocion por fase.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--data-cutoff", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--incumbent-metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ARTIFACTS_DIR / "artifact_manifest.json")
    args = parser.parse_args()
    promoted = record_phase_gate(
        args.manifest, phase=args.phase, data_cutoff=args.data_cutoff,
        candidate_version=args.candidate_version,
        candidate_metrics=json.loads(args.candidate_metrics.read_text(encoding="utf-8")),
        incumbent_metrics=json.loads(args.incumbent_metrics.read_text(encoding="utf-8")),
    )
    print("PROMOTED" if promoted else "REJECTED")


if __name__ == "__main__":
    main()
