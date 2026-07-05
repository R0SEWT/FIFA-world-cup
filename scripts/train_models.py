#!/usr/bin/env python3
"""Entrena MLP Adam/SGD, LSTM y GRU y exporta el ganador."""

import argparse
from pathlib import Path

from mundial.training import train_all
from mundial.config import ARTIFACTS_DIR, PROCESSED_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Usa dos epocas para una prueba de integracion")
    parser.add_argument("--skip-nuts", action="store_true", help="Omite la auditoria NUTS de cuatro cadenas")
    parser.add_argument("--bayes-steps", type=int, default=50_000, help="Iteraciones ADVI para calibracion y Dixon-Coles")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--phase", help="Fase que origina el candidato")
    parser.add_argument("--terminal-fold-start", help="Inicio YYYY-MM-DD de la fase dejada fuera")
    parser.add_argument("--terminal-fold-end", help="Fin YYYY-MM-DD de la fase dejada fuera")
    args = parser.parse_args()
    epochs_mlp, epochs_recurrent = (2, 2) if args.quick else (60, 40)
    bayes_steps = min(args.bayes_steps, 200) if args.quick else args.bayes_steps
    if bool(args.terminal_fold_start) != bool(args.terminal_fold_end):
        parser.error("--terminal-fold-start y --terminal-fold-end deben usarse juntos")
    terminal_fold = (
        (f"terminal_{args.phase or 'phase'}", args.terminal_fold_start, args.terminal_fold_end)
        if args.terminal_fold_start else None
    )
    result = train_all(
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        max_epochs_mlp=epochs_mlp,
        max_epochs_recurrent=epochs_recurrent,
        bayes_steps=bayes_steps,
        run_nuts_audit=not (args.skip_nuts or args.quick),
        terminal_fold=terminal_fold,
        phase=args.phase,
    )
    print(f"Modelo seleccionado: {result['selected_model']}")
