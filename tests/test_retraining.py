import json

from mundial.retraining import passes_promotion_gate, record_phase_gate


def test_gate_requires_both_scoring_rules_and_calibration():
    incumbent = {"log_loss": 1.0, "brier": 0.7, "ece": 0.05}
    assert passes_promotion_gate({"log_loss": 0.9, "brier": 0.6, "ece": 0.06}, incumbent)
    assert not passes_promotion_gate({"log_loss": 0.9, "brier": 0.8, "ece": 0.04}, incumbent)
    assert not passes_promotion_gate({"log_loss": 0.9, "brier": 0.6, "ece": 0.07}, incumbent)


def test_gate_manifest_records_rollback(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"model_version": "v1"}))
    promoted = record_phase_gate(
        path, phase="groups", data_cutoff="2026-06-27", candidate_version="v2",
        candidate_metrics={"log_loss": 0.9, "brier": 0.6, "ece": 0.05},
        incumbent_metrics={"log_loss": 1.0, "brier": 0.7, "ece": 0.05},
    )
    manifest = json.loads(path.read_text())
    assert promoted and manifest["model_version"] == "v2"
    assert manifest["rollback_version"] == "v1"
