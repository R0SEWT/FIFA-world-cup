#!/usr/bin/env python3
"""Construye el dataset tabular y las ventanas de secuencia."""

import argparse
from pathlib import Path

from mundial.data import save_processed_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-date", help="Incluye solo partidos completados hasta YYYY-MM-DD")
    parser.add_argument("--tournament-state", type=Path, help="Snapshot aprobado para reentrenamiento por fase")
    args = parser.parse_args()
    table, sequences = save_processed_dataset(
        as_of_date=args.as_of_date, tournament_state_path=args.tournament_state
    )
    print(f"Dataset tabular: {table}")
    print(f"Ventanas temporales: {sequences}")
