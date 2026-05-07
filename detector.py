"""
Detector clasico de monedas peruanas con OpenCV.

No usa redes neuronales. La escala se calibra con puntos azules HSV
cuando existen; si faltan, usa una referencia cruzada conservadora con
el diametro mediano de las monedas detectadas.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


BLUE_LOWER = np.array([100, 100, 50], dtype=np.uint8)
BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)
MARKER_DIAMETER_MM = 5.0

# Diametros oficiales usados como base de normalizacion. Los cortes se
# colocan entre grupos reales cercanos y se apoyan con HSV para separar
# 50 centimos (dorado) de 2 soles (mas plateado/rosado).
COIN_SPECS = {
    "10cts": {"label": "10 céntimos", "diameter_mm": 15.0, "value": 0.10},
    "50cts": {"label": "50 céntimos", "diameter_mm": 20.0, "value": 0.50},
    "1sol": {"label": "1 sol", "diameter_mm": 23.5, "value": 1.00},
    "2soles": {"label": "2 soles", "diameter_mm": 20.4, "value": 2.00},
    "5soles": {"label": "5 soles", "diameter_mm": 22.0, "value": 5.00},
}

COUNT_KEYS = ["10cts", "50cts", "1sol", "2soles", "5soles"]


@dataclass
class CoinDetection:
    x: int
    y: int
    radius: int
    diameter_px: float
    diameter_mm: float
    key: str
    label: str
    value: float
    hue: float
    saturation: float
    method: str


def _contours_from(mask: np.ndarray) -> List[np.ndarray]:
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return found[0] if len(found) == 2 else found[1]


def detectar_puntos_azules(img: np.ndarray) -> Tuple[Optional[float], List[Dict[str, float]], np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    markers: List[Dict[str, float]] = []
    h, w = img.shape[:2]
    border_margin_x = w * 0.18
    border_margin_y = h * 0.18
    for cnt in _contours_from(mask):
        area = cv2.contourArea(cnt)
        if area < 8 or area > 1400:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        near_border = x < border_margin_x or x > (w - border_margin_x) or y < border_margin_y or y > (h - border_margin_y)
        if not near_border or radius < 2.5 or radius > 18:
            continue
        diameter_px = 2 * radius
        markers.append({"x": x, "y": y, "radius": radius, "diameter_px": diameter_px})

    if not markers:
        return None, [], mask

    diameters = [m["diameter_px"] for m in markers]
    factor = MARKER_DIAMETER_MM / float(np.median(diameters))
    return factor, markers, mask


def _coin_mask_candidates(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    adapt_clean = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kernel, iterations=1)
    return blur, otsu_clean, adapt_clean


def _filtrar_contornos_circulares(mask: np.ndarray, img_shape: Tuple[int, int], min_area_pct: float = 0.0005, max_area_pct: float = 0.10) -> List[Dict[str, float]]:
    h, w = img_shape[:2]
    img_area = h * w
    candidates: List[Dict[str, float]] = []
    for cnt in _contours_from(mask):
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_pct or area > img_area * max_area_pct:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.55:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        min_r = max(12, int(min(h, w) * 0.015))
        max_r = min(120, int(min(h, w) * 0.18))
        if radius < min_r or radius > max_r:
            continue
        candidates.append(
            {
                "x": float(x),
                "y": float(y),
                "radius": float(radius),
                "score": float(circularity),
                "method": "contorno",
                "area": float(area),
            }
        )
    return candidates


def _color_blob_candidates(img: np.ndarray, blue_mask: np.ndarray) -> List[Dict[str, float]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sat_mask = cv2.inRange(hsv[:, :, 1], 35, 255)
    dark_mask = cv2.inRange(gray, 0, 170)
    mask = cv2.bitwise_or(sat_mask, dark_mask)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.dilate(blue_mask, None, iterations=2)))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    candidates = _filtrar_contornos_circulares(mask, img.shape, 0.0005, 0.10)
    for cand in candidates:
        cand["score"] = max(cand["score"], 1.0)
        cand["method"] = "color"
    return candidates


def _detectar_circulos(blur: np.ndarray, img_shape: Tuple[int, int]) -> List[Dict[str, float]]:
    h, w = img_shape[:2]
    min_radius = max(15, int(min(h, w) * 0.018))
    max_radius = min(100, int(min(h, w) * 0.15))
    min_dist = max(25, int(min(h, w) * 0.05))
    
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=55,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    result = []
    for x, y, r in np.round(circles[0, :]).astype(int):
        result.append({"x": float(x), "y": float(y), "radius": float(r), "score": 0.88, "method": "hough"})
    return result


def _merge_candidates(candidates: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    ordered = sorted(candidates, key=lambda c: (c["score"], c["radius"]), reverse=True)
    merged: List[Dict[str, float]] = []
    for cand in ordered:
        duplicate = False
        for kept in merged:
            dist = math.hypot(cand["x"] - kept["x"], cand["y"] - kept["y"])
            if dist < max(kept["radius"], cand["radius"]) * 0.68:
                duplicate = True
                if cand["score"] > kept["score"]:
                    kept.update(cand)
                break
        if not duplicate:
            merged.append(cand)
    return merged


def _roi_stats(img: np.ndarray, x: int, y: int, radius: int) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    x1, x2 = max(0, x - radius), min(w, x + radius)
    y1, y2 = max(0, y - radius), min(h, y + radius)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return roi, 0.0, 0.0

    yy, xx = np.ogrid[y1:y2, x1:x2]
    mask = ((xx - x) ** 2 + (yy - y) ** 2) <= (radius * 0.82) ** 2
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if mask.shape[:2] != hsv.shape[:2] or not np.any(mask):
        return roi, float(np.mean(hsv[:, :, 0])), float(np.mean(hsv[:, :, 1]))
    hue = float(np.mean(hsv[:, :, 0][mask]))
    sat = float(np.mean(hsv[:, :, 1][mask]))
    return roi, hue, sat


def clasificar_moneda(diametro_mm: float, hue: float, sat: float) -> Tuple[str, str, float]:
    if diametro_mm < 17.0:
        key = "10cts"
    elif diametro_mm < 21.0:
        if hue > 12 and sat > 70:
            key = "50cts"
        else:
            key = "2soles"
    elif diametro_mm < 23.0:
        if sat > 60:
            key = "1sol"
        else:
            key = "2soles"
    elif diametro_mm < 25.5:
        if hue > 12 and sat > 50:
            key = "5soles"
        else:
            key = "1sol"
    else:
        if sat < 50:
            key = "1sol"
        else:
            key = "5soles"

    spec = COIN_SPECS[key]
    return key, spec["label"], spec["value"]


def _fallback_factor(candidates: List[Dict[str, float]], img_shape: Tuple[int, int] = None) -> float:
    if not candidates:
        return 1.0
    
    if img_shape:
        h, w = img_shape[:2]
        diag = math.sqrt(h**2 + w**2)
        expected_coin_px = diag * 0.08
        if expected_coin_px > 0:
            return 22.4 / expected_coin_px
    
    diameters = [2 * c["radius"] for c in candidates if c["radius"] > 25]
    if not diameters:
        diameters = [2 * c["radius"] for c in candidates if c["radius"] > 15]
    if not diameters:
        return 1.0
    
    median_d = float(np.median(diameters))
    if median_d < 30 or median_d > 300:
        return 22.4 / max(median_d, 50)
    return 22.4 / median_d


def _filtrar_por_escala(candidates: List[Dict[str, float]], factor_escala: Optional[float]) -> List[Dict[str, float]]:
    if factor_escala is None:
        return candidates
    plausibles = []
    for cand in candidates:
        diameter_mm = 2 * cand["radius"] * factor_escala
        if 12.0 <= diameter_mm <= 32.0:
            plausibles.append(cand)
    return plausibles or candidates


def detectar_monedas(imagen_path: str | Path, output_dir: str | Path = "output") -> Dict:
    image_path = Path(imagen_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    factor_escala, markers, blue_mask = detectar_puntos_azules(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur, otsu_clean, adapt_clean = _coin_mask_candidates(gray)
    
    h, w = img.shape[:2]
    if factor_escala is None:
        diag_px = math.sqrt(h**2 + w**2)
        if w > 0 and h > 0:
            coin_approx_px = min(w, h) * 0.12
            factor_escala = 22.4 / coin_approx_px if coin_approx_px > 0 else 0.15

    candidates = []
    candidates.extend(_detectar_circulos(blur, img.shape))
    candidates.extend(_color_blob_candidates(img, blue_mask))
    candidates.extend(_filtrar_contornos_circulares(otsu_clean, img.shape, 0.0005, 0.08))
    merged = _merge_candidates(candidates)

    # Evita contar los puntos azules como monedas si el marcador quedo grande.
    filtered = []
    for cand in merged:
        is_marker = False
        for marker in markers:
            dist = math.hypot(cand["x"] - marker["x"], cand["y"] - marker["y"])
            if dist < max(cand["radius"], marker["radius"]) * 1.3:
                is_marker = True
                break
        if not is_marker:
            filtered.append(cand)

    filtered = _filtrar_por_escala(filtered, factor_escala)

    if factor_escala is None:
        factor_escala = _fallback_factor(filtered, img.shape)
        filtered = _filtrar_por_escala(filtered, factor_escala)

    detections: List[CoinDetection] = []
    for cand in filtered:
        x, y, radius = int(round(cand["x"])), int(round(cand["y"])), int(round(cand["radius"]))
        roi, hue, sat = _roi_stats(img, x, y, radius)
        diameter_mm = 2 * radius * factor_escala
        key, label, value = clasificar_moneda(diameter_mm, hue, sat)
        detections.append(
            CoinDetection(
                x=x,
                y=y,
                radius=radius,
                diameter_px=float(2 * radius),
                diameter_mm=float(diameter_mm),
                key=key,
                label=label,
                value=float(value),
                hue=float(hue),
                saturation=float(sat),
                method=str(cand["method"]),
            )
        )

    counts = {key: 0 for key in COUNT_KEYS}
    for det in detections:
        counts[det.key] += 1
    total = round(sum(COIN_SPECS[k]["value"] * counts[k] for k in COUNT_KEYS), 2)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    annotated = img.copy()
    for marker in markers:
        cv2.circle(annotated, (int(marker["x"]), int(marker["y"])), int(marker["radius"]), (255, 0, 0), 2)
    for det in detections:
        cv2.circle(annotated, (det.x, det.y), det.radius, (0, 215, 255), 3)
        cv2.circle(annotated, (det.x, det.y), 3, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"{det.label} S/{det.value:.2f}",
            (max(5, det.x - det.radius), max(24, det.y - det.radius - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"TOTAL: S/ {total:.2f}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )
    annotated_name = f"{image_path.stem}_annotated.jpg"
    annotated_rel = str(Path(output_dir) / annotated_name).replace("\\", "/")
    cv2.imwrite(str(output_path / annotated_name), annotated)

    return {
        "imagen": image_path.name,
        "ruta": str(image_path).replace("\\", "/"),
        "anotada": annotated_rel,
        "factor_escala": float(factor_escala),
        "marcadores_azules": markers,
        "conteo": counts,
        "total_detectado": total,
        "monedas": [asdict(det) for det in detections],
    }


def procesar_directorio(input_dir: str | Path = "photos", output_dir: str | Path = "output") -> List[Dict]:
    paths = [
        p
        for p in sorted(Path(input_dir).iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    return [detectar_monedas(path, output_dir=output_dir) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta y clasifica monedas peruanas sin IA.")
    parser.add_argument("imagen", nargs="?", help="Imagen individual a procesar.")
    parser.add_argument("--input", default="photos", help="Carpeta de imagenes si no se pasa una imagen.")
    parser.add_argument("--output", default="output", help="Carpeta para imagenes anotadas.")
    parser.add_argument("--json", default=None, help="Ruta opcional para guardar resultados JSON.")
    args = parser.parse_args()

    if args.imagen:
        resultados = [detectar_monedas(args.imagen, args.output)]
    else:
        resultados = procesar_directorio(args.input, args.output)

    if args.json:
        Path(args.json).write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultados, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
