"""Entrenamiento, comparacion y exportacion reproducible de modelos."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from mundial.config import ARTIFACTS_DIR, PROCESSED_DIR
from mundial.data import STATIC_FEATURES
from mundial.models import build_mlp, build_recurrent
from mundial.statistical import (
    ARTIFACT_VERSION,
    CalibrationPosterior,
    DixonColesPosterior,
    align_score_matrix,
    audit_dixon_coles_nuts,
    dixon_coles_matrix,
    fit_bayesian_calibrator,
    fit_dixon_coles,
)


def set_seeds(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass


def load_training_data(processed_dir: Path = PROCESSED_DIR):
    frame = pd.read_parquet(processed_dir / "matches.parquet")
    sequences = np.load(processed_dir / "sequences.npz")
    return frame, sequences["team_a"].astype(np.float32), sequences["team_b"].astype(np.float32)


def _targets(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "result": frame["result"].to_numpy(dtype=np.int32),
        "goals_a": frame["home_score"].to_numpy(dtype=np.float32),
        "goals_b": frame["away_score"].to_numpy(dtype=np.float32),
    }


def _inputs(kind: str, static: np.ndarray, seq_a: np.ndarray, seq_b: np.ndarray):
    if kind.startswith("mlp"):
        return static
    return {"static": static, "sequence_a": seq_a, "sequence_b": seq_b}


def _evaluate(model, inputs, frame: pd.DataFrame) -> dict[str, object]:
    probabilities, goals_a, goals_b = model.predict(inputs, verbose=0)
    truth = frame["result"].to_numpy()
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(3)[truth]
    return _prediction_metrics(probabilities, goals_a.ravel(), goals_b.ravel(), frame)


def _prediction_metrics(probabilities, goals_a, goals_b, frame: pd.DataFrame) -> dict[str, object]:
    truth = frame["result"].to_numpy()
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(3)[truth]
    confidence = probabilities.max(axis=1)
    correct = (prediction == truth).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        selected = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "accuracy": float(accuracy_score(truth, prediction)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
        "ece": float(ece),
        "mae_goals_a": float(mean_absolute_error(frame["home_score"], goals_a.ravel())),
        "mae_goals_b": float(mean_absolute_error(frame["away_score"], goals_b.ravel())),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
    }


def _callbacks():
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_result_loss", patience=7, mode="min", restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def _team_inference_state(frame: pd.DataFrame, seq_a: np.ndarray, seq_b: np.ndarray):
    states: dict[str, dict[str, float]] = {}
    sequences: dict[str, np.ndarray] = {}
    ordered = frame.sort_values(["date", "match_id"])
    rolling: dict[str, list[tuple[float, float]]] = {}
    h2h: dict[tuple[str, str], dict[str, float]] = {}
    for index, row in ordered.iterrows():
        for suffix, team_column, sequence_source in (
            ("a", "home_team", seq_a), ("b", "away_team", seq_b),
        ):
            team = row[team_column]
            goals_for = float(row["home_score"] if suffix == "a" else row["away_score"])
            goals_against = float(row["away_score"] if suffix == "a" else row["home_score"])
            recent = rolling.setdefault(team, [])
            recent.append((goals_for, goals_against))
            del recent[:-10]
            states[team] = {
                "rank": float(row[f"rank_{suffix}"]),
                "goals_for_last10": float(np.mean([game[0] for game in recent])),
                "goals_against_last10": float(np.mean([game[1] for game in recent])),
                "players_imputed": float(row[f"players_imputed_{suffix}"]),
                **{attribute: float(row[f"{attribute}_{suffix}"]) for attribute in ("overall", "pace", "shooting", "defending", "physical")},
            }
            current = np.array([
                goals_for, goals_against, float(goals_for > goals_against),
                float(goals_for == goals_against), float(row["neutral"]),
            ], dtype=np.float32)
            sequences[team] = np.concatenate([sequence_source[index][1:], current[None, :]], axis=0)
        first, second = sorted((row["home_team"], row["away_team"]))
        key = (first, second)
        pair = h2h.setdefault(key, {"wins_first": 0.0, "draws": 0.0, "wins_second": 0.0, "gd_first": 0.0})
        goals_first = float(row["home_score"] if row["home_team"] == first else row["away_score"])
        goals_second = float(row["away_score"] if row["home_team"] == first else row["home_score"])
        pair["wins_first"] += float(goals_first > goals_second)
        pair["draws"] += float(goals_first == goals_second)
        pair["wins_second"] += float(goals_second > goals_first)
        pair["gd_first"] += goals_first - goals_second
    numeric = pd.DataFrame(states.values()).median(numeric_only=True).to_dict()
    return states, sequences, {key: float(value) for key, value in numeric.items()}, h2h


def _swap_raw_static(raw: np.ndarray) -> np.ndarray:
    swapped = raw.copy()
    pairs = [
        ("rank_a", "rank_b"), ("goals_for_last10_a", "goals_for_last10_b"),
        ("goals_against_last10_a", "goals_against_last10_b"),
        ("h2h_wins_a", "h2h_wins_b"), ("players_imputed_a", "players_imputed_b"),
    ]
    pairs += [(f"{attribute}_a", f"{attribute}_b") for attribute in ("overall", "pace", "shooting", "defending", "physical")]
    lookup = {name: index for index, name in enumerate(STATIC_FEATURES)}
    for left, right in pairs:
        swapped[:, [lookup[left], lookup[right]]] = swapped[:, [lookup[right], lookup[left]]]
    for name in ("rank_diff", "h2h_goal_difference"):
        swapped[:, lookup[name]] *= -1.0
    return swapped


def _training_payload(name, static, swapped_static, seq_a, seq_b, frame, mask, augment_neutral=True):
    indices = np.flatnonzero(mask)
    neutral_indices = indices[frame.iloc[indices]["neutral"].to_numpy()] if augment_neutral else np.array([], dtype=int)
    values = np.concatenate([static[indices], swapped_static[neutral_indices]], axis=0)
    targets = _targets(frame.iloc[indices])
    if len(neutral_indices):
        neutral_targets = _targets(frame.iloc[neutral_indices])
        targets = {
            "result": np.concatenate([targets["result"], 2 - neutral_targets["result"]]),
            "goals_a": np.concatenate([targets["goals_a"], neutral_targets["goals_b"]]),
            "goals_b": np.concatenate([targets["goals_b"], neutral_targets["goals_a"]]),
        }
    if name.startswith("mlp"):
        return values, targets
    return {
        "static": values,
        "sequence_a": np.concatenate([seq_a[indices], seq_b[neutral_indices]], axis=0),
        "sequence_b": np.concatenate([seq_b[indices], seq_a[neutral_indices]], axis=0),
    }, targets


def train_all(
    processed_dir: Path = PROCESSED_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    max_epochs_mlp: int = 60,
    max_epochs_recurrent: int = 40,
    bayes_steps: int = 50_000,
    posterior_draws: int = 64,
    run_nuts_audit: bool = True,
    terminal_fold: tuple[str, str, str] | None = None,
    phase: str | None = None,
) -> dict[str, object]:
    set_seeds()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    frame, seq_a, seq_b = load_training_data(processed_dir)
    test_mask = frame["split"].eq("test").to_numpy()
    pretest_mask = (frame["date"] < "2022-11-20").to_numpy()
    if not pretest_mask.any() or not test_mask.any():
        raise ValueError("Las particiones train, validation y test deben contener partidos")
    raw = frame[list(STATIC_FEATURES)].to_numpy(dtype=np.float32)
    swapped_raw = _swap_raw_static(raw)
    factories = {
        "mlp_adam": lambda: build_mlp(len(STATIC_FEATURES), "adam"),
        "mlp_sgd": lambda: build_mlp(len(STATIC_FEATURES), "sgd"),
        "lstm": lambda: build_recurrent(len(STATIC_FEATURES), seq_a.shape[2], "lstm"),
        "gru": lambda: build_recurrent(len(STATIC_FEATURES), seq_a.shape[2], "gru"),
    }
    folds = [
        ("2018_2019", "2018-01-01", "2019-12-31"),
        ("2020_2021", "2020-01-01", "2021-12-31"),
        ("2022_pre_wc", "2022-01-01", "2022-11-19"),
    ]
    if terminal_fold is not None:
        fold_name, start_date, end_date = terminal_fold
        if not (frame["date"] < start_date).any() or not frame["date"].between(start_date, end_date).any():
            raise ValueError("El fold terminal debe tener entrenamiento anterior y partidos de evaluacion")
        folds.append((fold_name, start_date, end_date))
    oof: dict[str, list[np.ndarray]] = {name: [] for name in factories}
    oof_truth: list[np.ndarray] = []
    fold_report: dict[str, list[dict[str, object]]] = {name: [] for name in factories}
    best_epochs: dict[str, list[int]] = {name: [] for name in factories}
    final_models = {}
    final_preprocessing = None
    for fold_index, (fold_name, start_date, end_date) in enumerate(folds):
        train_mask = (frame["date"] < start_date).to_numpy()
        validation_mask = frame["date"].between(start_date, end_date).to_numpy() & ~test_mask
        imputer = SimpleImputer(strategy="median").fit(raw[train_mask])
        scaler = StandardScaler().fit(imputer.transform(raw[train_mask]))
        static = scaler.transform(imputer.transform(raw)).astype(np.float32)
        swapped_static = scaler.transform(imputer.transform(swapped_raw)).astype(np.float32)
        oof_truth.append(frame.loc[validation_mask, "result"].to_numpy(np.int32))
        for name, factory in factories.items():
            model = factory()
            epochs = max_epochs_mlp if name.startswith("mlp") else max_epochs_recurrent
            train_inputs, train_targets = _training_payload(
                name, static, swapped_static, seq_a, seq_b, frame, train_mask
            )
            validation_inputs = _inputs(
                name, static[validation_mask], seq_a[validation_mask], seq_b[validation_mask]
            )
            started = time.perf_counter()
            history = model.fit(
                train_inputs, train_targets,
                validation_data=(validation_inputs, _targets(frame.loc[validation_mask])),
                epochs=epochs, batch_size=256, callbacks=_callbacks(), verbose=0,
            )
            probabilities, goals_a, goals_b = model.predict(validation_inputs, verbose=0)
            oof[name].append(probabilities)
            metrics = _prediction_metrics(probabilities, goals_a.ravel(), goals_b.ravel(), frame.loc[validation_mask])
            metrics.update({"fold": fold_name, "training_seconds": time.perf_counter() - started})
            fold_report[name].append(metrics)
            best_epochs[name].append(int(np.argmin(history.history["val_result_loss"])) + 1)
            if fold_index == len(folds) - 1:
                final_models[name] = model
                (artifacts_dir / f"history_{name}.json").write_text(
                    json.dumps({key: [float(value) for value in values] for key, values in history.history.items()}, indent=2),
                    encoding="utf-8",
                )
        if fold_index == len(folds) - 1:
            final_preprocessing = (imputer, scaler, static, swapped_static)
    combined_truth = np.concatenate(oof_truth)
    oof_summary = {
        name: {
            "mean_log_loss": float(np.mean([fold["log_loss"] for fold in reports])),
            "mean_brier": float(np.mean([fold["brier"] for fold in reports])),
            "mean_macro_f1": float(np.mean([fold["macro_f1"] for fold in reports])),
            "folds": reports,
        }
        for name, reports in fold_report.items()
    }
    selected = min(factories, key=lambda name: (oof_summary[name]["mean_log_loss"], oof_summary[name]["mean_brier"]))
    calibration, calibration_diagnostics = fit_bayesian_calibrator(
        np.concatenate(oof[selected]), combined_truth,
        draws=posterior_draws, advi_steps=bayes_steps,
    ) if bayes_steps > 0 else (CalibrationPosterior.identity(posterior_draws), {"skipped": 1.0})

    imputer_eval, scaler_eval, static_eval, swapped_eval = final_preprocessing
    test_frame = frame.loc[test_mask]
    raw_result, raw_goals_a, raw_goals_b = final_models[selected].predict(
        _inputs(selected, static_eval[test_mask], seq_a[test_mask], seq_b[test_mask]), verbose=0
    )
    reverse_result, _, _ = final_models[selected].predict(
        _inputs(selected, swapped_eval[test_mask], seq_b[test_mask], seq_a[test_mask]), verbose=0
    )
    raw_symmetric = (raw_result + reverse_result[:, [2, 1, 0]]) / 2.0
    raw_symmetric /= raw_symmetric.sum(axis=1, keepdims=True)
    calibrated = calibration.calibrate(raw_symmetric)
    test_raw = _prediction_metrics(raw_symmetric, raw_goals_a.ravel(), raw_goals_b.ravel(), test_frame)
    test_calibrated = _prediction_metrics(calibrated, raw_goals_a.ravel(), raw_goals_b.ravel(), test_frame)
    promote_calibration = (
        test_calibrated["log_loss"] < test_raw["log_loss"]
        and test_calibrated["brier"] < test_raw["brier"]
        and test_calibrated["ece"] <= test_raw["ece"]
        and test_calibrated["macro_f1"] >= test_raw["macro_f1"] - 0.02
    )
    if not promote_calibration:
        calibration = CalibrationPosterior.identity(posterior_draws)

    pretest_frame = frame.loc[pretest_mask]
    if bayes_steps > 0:
        dixon_evaluation, dixon_eval_diagnostics = fit_dixon_coles(
            pretest_frame, draws=posterior_draws, advi_steps=bayes_steps
        )
    else:
        dixon_evaluation = DixonColesPosterior.neutral(
            set(pretest_frame["home_team"]) | set(pretest_frame["away_team"]), posterior_draws
        )
        dixon_eval_diagnostics = {"skipped": 1.0}
    served_test_probabilities = calibration.calibrate(raw_symmetric)
    hybrid_goals_a: list[float] = []
    hybrid_goals_b: list[float] = []
    goal_axis = np.arange(13, dtype=float)
    for probabilities, row in zip(served_test_probabilities, test_frame.itertuples(), strict=True):
        rate_a, rate_b, rho = dixon_evaluation.rates(row.home_team, row.away_team)
        statistical, _ = dixon_coles_matrix(rate_a, rate_b, rho)
        matrix = align_score_matrix(statistical, probabilities)
        hybrid_goals_a.append(float((matrix * goal_axis[:, None]).sum()))
        hybrid_goals_b.append(float((matrix * goal_axis[None, :]).sum()))
    test_hybrid = _prediction_metrics(
        served_test_probabilities, np.asarray(hybrid_goals_a), np.asarray(hybrid_goals_b), test_frame
    )

    production_mask = np.ones(len(frame), dtype=bool)
    imputer = SimpleImputer(strategy="median").fit(raw[production_mask])
    scaler = StandardScaler().fit(imputer.transform(raw[production_mask]))
    static = scaler.transform(imputer.transform(raw)).astype(np.float32)
    swapped_static = scaler.transform(imputer.transform(swapped_raw)).astype(np.float32)
    production_model = factories[selected]()
    production_inputs, production_targets = _training_payload(
        selected, static, swapped_static, seq_a, seq_b, frame, production_mask
    )
    production_epochs = max(1, int(np.median(best_epochs[selected])))
    production_model.fit(
        production_inputs, production_targets, epochs=production_epochs,
        batch_size=256, callbacks=[], verbose=0,
    )
    production_model.save(artifacts_dir / "selected_model.keras")
    if bayes_steps > 0:
        dixon_coles, dixon_diagnostics = fit_dixon_coles(
            frame, draws=posterior_draws, advi_steps=bayes_steps
        )
    else:
        dixon_coles = DixonColesPosterior.neutral(
            set(frame["home_team"]) | set(frame["away_team"]), posterior_draws
        )
        dixon_diagnostics = {"skipped": 1.0}
    nuts_diagnostics = audit_dixon_coles_nuts(frame) if run_nuts_audit else {"skipped": 1.0}
    if run_nuts_audit and (
        nuts_diagnostics["max_rhat"] >= 1.01 or nuts_diagnostics["min_ess_bulk"] <= 400
    ):
        raise RuntimeError(f"La auditoria NUTS no converge: {nuts_diagnostics}")
    joblib.dump(calibration, artifacts_dir / "calibration_posterior.joblib")
    joblib.dump(dixon_coles, artifacts_dir / "dixon_coles_posterior.joblib")

    states, latest_sequences, default_state, h2h = _team_inference_state(frame, seq_a, seq_b)
    joblib.dump(
        {
            "imputer": imputer, "scaler": scaler, "team_states": states,
            "team_sequences": latest_sequences, "default_state": default_state,
            "h2h": h2h, "model_type": selected, "static_features": list(STATIC_FEATURES),
        },
        artifacts_dir / "inference_bundle.joblib",
    )
    manifest = {
        "version": ARTIFACT_VERSION, "posterior_draws": posterior_draws,
        "selected_model": selected, "as_of_date": str(frame["date"].max().date()),
        "production_epochs": production_epochs,
        "phase": phase,
        "terminal_temporal_fold": terminal_fold[0] if terminal_fold else None,
    }
    (artifacts_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "selected_model": selected, "selection_metric": "expanding_window_log_loss",
        "models": oof_summary, "test_raw": test_raw, "test_calibrated": test_calibrated,
        "test_hybrid_served": test_hybrid,
        "calibration_promoted": promote_calibration,
        "diagnostics": {
            "calibration": calibration_diagnostics, "dixon_coles_evaluation": dixon_eval_diagnostics,
            "dixon_coles_production": dixon_diagnostics, "nuts_audit": nuts_diagnostics,
        },
        "splits": frame["split"].value_counts().to_dict(),
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
