"""
Procesa el dataset, evalua metricas y exporta resultados.json para index.html.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from detector import COUNT_KEYS, detectar_monedas
from evaluar import evaluar_error


def _image_paths(include_augmented: bool = True) -> List[Path]:
    folders = [Path("photos")]
    if include_augmented and Path("augmented").exists():
        folders.append(Path("augmented"))
    paths: List[Path] = []
    for folder in folders:
        paths.extend(
            p
            for p in sorted(folder.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    return paths


def _ground_truth_map(path: str | Path = "ground_truth.csv") -> Dict[str, Dict]:
    gt_path = Path(path)
    if not gt_path.exists():
        return {}
    with gt_path.open(newline="", encoding="utf-8") as fh:
        return {row["imagen"]: row for row in csv.DictReader(fh)}


def _tipo_imagen(row: Dict, result: Dict) -> str:
    if row.get("origen") == "augmented":
        return "aumentada"
    total_coins = sum(int(row.get(k, 0) or 0) for k in COUNT_KEYS) if row else len(result["monedas"])
    if total_coins <= 1:
        return "individual"
    names = f"{result['imagen']} {row.get('fuente', '') if row else ''}".lower()
    if any(token in names for token in ["blur", "sombra", "sobreexpuesta", "whatsapp"]):
        return "con oclusión"
    return "grupal"


def exportar(
    output_json: str = "resultados.json",
    ground_truth: str = "ground_truth.csv",
    include_augmented: bool = True,
    embedded_js: str = "resultados_embedded.js",
) -> Dict:
    gt = _ground_truth_map(ground_truth)
    resultados = []
    for path in _image_paths(include_augmented=include_augmented):
        result = detectar_monedas(path, output_dir="output")
        row = gt.get(result["imagen"], {})
        real = float(row.get("total_real", 0) or 0)
        result["total_real"] = real
        result["error_abs"] = round(abs(real - result["total_detectado"]), 2) if row else 0
        result["error_pct"] = round((result["error_abs"] / real) * 100, 2) if real > 0 else 0
        result["tipo_imagen"] = _tipo_imagen(row, result)
        resultados.append(result)

    Path(output_json).write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(embedded_js).write_text(
        "window.RESULTADOS_EMBEBIDOS = "
        + json.dumps(resultados, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )

    metricas = {"mae": 0.0, "accuracy": 0.0, "total_imagenes": len(resultados)}
    if Path(ground_truth).exists():
        errores, mae, accuracy = evaluar_error(ground_truth, output_json)
        metricas = {"mae": mae, "accuracy": accuracy, "total_imagenes": len(errores), "errores": errores}
        Path("metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"resultados": resultados, "metricas": metricas}


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera resultados.json para la pagina web.")
    parser.add_argument("--json", default="resultados.json")
    parser.add_argument("--embedded-js", default="resultados_embedded.js")
    parser.add_argument("--ground-truth", default="ground_truth.csv")
    parser.add_argument("--solo-originales", action="store_true", help="No procesa la carpeta augmented.")
    args = parser.parse_args()
    data = exportar(args.json, args.ground_truth, include_augmented=not args.solo_originales, embedded_js=args.embedded_js)
    print(f"{len(data['resultados'])} imagenes exportadas en {args.json}")
    print(f"MAE: S/ {data['metricas'].get('mae', 0):.2f}")
    print(f"Accuracy: {data['metricas'].get('accuracy', 0) * 100:.2f}%")


if __name__ == "__main__":
    main()
