"""
Genera imagenes aumentadas y etiquetas para el dataset de monedas.

Las etiquetas de las monedas individuales salen del nombre del archivo.
Para imagenes grupales sin registro manual se usa una estimacion inicial
del detector, de modo que ground_truth.csv queda completo y editable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import cv2

from detector import COIN_SPECS, COUNT_KEYS, detectar_monedas


TRANSFORMACIONES = [
    ("rot90", lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
    ("rot180", lambda img: cv2.rotate(img, cv2.ROTATE_180)),
    ("rot270", lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ("sombra", lambda img: cv2.convertScaleAbs(img, alpha=0.6, beta=-30)),
    ("sobreexpuesta", lambda img: cv2.convertScaleAbs(img, alpha=1.4, beta=40)),
    ("blur", lambda img: cv2.GaussianBlur(img, (7, 7), 0)),
    ("flip", lambda img: cv2.flip(img, 1)),
]

INDIVIDUALES = {
    "1.jpeg": "10cts",
    "1_a.jpeg": "10cts",
    "5.jpeg": "50cts",
    "5_a.jpeg": "50cts",
    "10.jpeg": "1sol",
    "10_a.jpeg": "1sol",
    "20.jpeg": "2soles",
    "20_a.jpeg": "2soles",
    "50.jpeg": "5soles",
    "50_a.jpeg": "5soles",
}


def conteo_vacio() -> Dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def total_de(conteo: Dict[str, int]) -> float:
    return round(sum(COIN_SPECS[key]["value"] * conteo.get(key, 0) for key in COUNT_KEYS), 2)


def inferir_etiqueta(path: Path) -> Dict[str, int]:
    if path.name in INDIVIDUALES:
        conteo = conteo_vacio()
        conteo[INDIVIDUALES[path.name]] = 1
        return conteo

    # Estimacion inicial para fotos grupales no etiquetadas manualmente.
    # Se deja en CSV para que pueda corregirse si hay conteos reales a mano.
    resultado = detectar_monedas(path, output_dir="output")
    return {key: int(resultado["conteo"].get(key, 0)) for key in COUNT_KEYS}


def escribir_label_txt(path: Path, conteo: Dict[str, int]) -> None:
    lines: List[str] = []
    for key in COUNT_KEYS:
        cantidad = int(conteo.get(key, 0))
        if cantidad:
            lines.append(f"{COIN_SPECS[key]['label']}: {cantidad}")
    lines.append(f"TOTAL REAL: S/ {total_de(conteo):.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generar(input_dir: str = "photos", augmented_dir: str = "augmented", csv_path: str = "ground_truth.csv") -> List[Dict]:
    src_dir = Path(input_dir)
    out_dir = Path(augmented_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p
        for p in sorted(src_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    registros: List[Dict] = []
    etiquetas_por_fuente: Dict[str, Dict[str, int]] = {}
    for path in image_paths:
        conteo = inferir_etiqueta(path)
        etiquetas_por_fuente[path.name] = conteo
        row = {"imagen": path.name, **conteo, "total_real": total_de(conteo), "origen": "photos", "fuente": path.name}
        registros.append(row)

    aug_index = 1
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        conteo = etiquetas_por_fuente[path.name]
        for nombre, transform in TRANSFORMACIONES:
            aug = transform(img)
            aug_name = f"imagen_aug_{aug_index:03d}_{path.stem}_{nombre}.jpg"
            aug_path = out_dir / aug_name
            cv2.imwrite(str(aug_path), aug)
            escribir_label_txt(aug_path.with_suffix(".txt"), conteo)
            registros.append(
                {
                    "imagen": aug_name,
                    **conteo,
                    "total_real": total_de(conteo),
                    "origen": "augmented",
                    "fuente": path.name,
                }
            )
            aug_index += 1

    with Path(csv_path).open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["imagen", *COUNT_KEYS, "total_real", "origen", "fuente"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(registros)

    return registros


def main() -> None:
    parser = argparse.ArgumentParser(description="Aumenta el dataset y genera ground_truth.csv.")
    parser.add_argument("--input", default="photos")
    parser.add_argument("--augmented", default="augmented")
    parser.add_argument("--csv", default="ground_truth.csv")
    args = parser.parse_args()
    registros = generar(args.input, args.augmented, args.csv)
    print(f"Dataset registrado: {len(registros)} filas en {args.csv}")


if __name__ == "__main__":
    main()
