"""
Calcula error absoluto, error porcentual, MAE y accuracy del sistema.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def cargar_ground_truth(path: str | Path) -> Dict[str, Dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["imagen"]: row for row in reader}


def cargar_resultados(path: str | Path) -> Dict[str, Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["imagen"]: row for row in data}


def evaluar_error(ground_truth_csv: str | Path, resultados_json: str | Path) -> Tuple[List[Dict], float, float]:
    ground_truth = cargar_ground_truth(ground_truth_csv)
    resultados = cargar_resultados(resultados_json)
    errores: List[Dict] = []

    for imagen, row in ground_truth.items():
        if imagen not in resultados:
            continue
        real = float(row["total_real"])
        detectado = float(resultados[imagen]["total_detectado"])
        error_abs = abs(real - detectado)
        error_pct = (error_abs / real) * 100 if real > 0 else 0
        errores.append(
            {
                "imagen": imagen,
                "real": round(real, 2),
                "detectado": round(detectado, 2),
                "error_abs": round(error_abs, 2),
                "error_pct": round(error_pct, 2),
            }
        )

    mae = float(np.mean([e["error_abs"] for e in errores])) if errores else 0.0
    accuracy = float(len([e for e in errores if e["error_abs"] == 0]) / len(errores)) if errores else 0.0
    return errores, mae, accuracy


def imprimir_tabla(errores: List[Dict], mae: float, accuracy: float) -> None:
    print("| Imagen | Real (S/) | Detectado (S/) | Error (S/) | Error % |")
    print("|--------|-----------|----------------|------------|---------|")
    for e in errores:
        print(
            f"| {e['imagen']} | {e['real']:.2f} | {e['detectado']:.2f} | "
            f"{e['error_abs']:.2f} | {e['error_pct']:.2f}% |"
        )
    print()
    print(f"MAE: S/ {mae:.2f}")
    print(f"Accuracy exacta: {accuracy * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua resultados contra ground_truth.csv.")
    parser.add_argument("--ground-truth", default="ground_truth.csv")
    parser.add_argument("--resultados", default="resultados.json")
    parser.add_argument("--salida", default="errores.json")
    args = parser.parse_args()

    errores, mae, accuracy = evaluar_error(args.ground_truth, args.resultados)
    Path(args.salida).write_text(
        json.dumps({"errores": errores, "mae": mae, "accuracy": accuracy}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    imprimir_tabla(errores, mae, accuracy)


if __name__ == "__main__":
    main()
