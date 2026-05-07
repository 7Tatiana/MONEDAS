# Sistema de Conteo y Clasificación de Monedas Peruanas

Proyecto de visión artificial clásica con OpenCV para detectar, clasificar y contar monedas peruanas. No usa TensorFlow, PyTorch ni redes neuronales.

## Estructura

```text
photos/                    Dataset original
augmented/                 Dataset aumentado generado
output/                    Imágenes anotadas por el detector
ground_truth.csv           Etiquetas reales/estimadas
resultados.json            Resultados para la web
resultados_embedded.js     Respaldo para abrir index.html con file://
detector.py                Pipeline principal OpenCV
augmentar_dataset.py       Aumento de datos y etiquetas .txt
evaluar.py                 Métricas de error
exportar_resultados.py     Exporta JSON para index.html
index.html                 Web dinámica en una sola página
```

## Instalación

```bash
pip install opencv-python numpy pandas matplotlib
```

## Ejecución paso a paso

1. Generar dataset aumentado y `ground_truth.csv`:

```bash
python augmentar_dataset.py
```

2. Procesar imágenes y exportar `resultados.json`:

```bash
python exportar_resultados.py
```

Este comando también genera `resultados_embedded.js`, usado como respaldo porque algunos navegadores bloquean la lectura directa de JSON local al abrir `index.html` con `file://`.

3. Evaluar error:

```bash
python evaluar.py
```

4. Abrir `index.html` directamente en el navegador.

## Pipeline técnico

El detector aplica:

1. Detección de puntos azules en HSV con rango `H:100-130, S:100-255, V:50-255`.
2. Cálculo de factor de escala usando el diámetro físico conocido del marcador.
3. Conversión a gris y blur gaussiano.
4. Binarización Otsu y adaptativa.
5. Limpieza morfológica.
6. Detección principal con `cv2.HoughCircles`.
7. Detección complementaria por contornos circulares.
8. Clasificación por diámetro normalizado y color HSV.

Los diámetros usados son: 10 céntimos `17.0 mm`, 50 céntimos `22.0 mm`, 1 sol `25.5 mm`, 2 soles `22.4 mm` y 5 soles `24.0 mm`.

## Nota sobre ground truth

Las 10 imágenes individuales se etiquetan automáticamente por nombre de archivo. Las imágenes grupales no venían con conteo manual en el PDF ni en la carpeta, por eso el script genera una primera versión estimada con el detector. Si tienes conteos reales manuales, edita `ground_truth.csv` y vuelve a ejecutar:

```bash
python exportar_resultados.py
python evaluar.py
```

## Distinción 50 céntimos vs 2 soles

Ambas monedas tienen diámetros muy parecidos. El sistema usa el diámetro para ubicarlas en el grupo de `22-23 mm` y luego separa por color HSV:

- 50 céntimos: tono dorado/amarillo, `H > 15` y saturación alta.
- 2 soles: tono más plateado/rosado o menor saturación.

## Web

La página usa Chart.js desde CDN y JavaScript puro para el uploader. Al subir una imagen, se procesa en canvas con escala de grises, umbral, componentes circulares y clasificación aproximada por diámetro/color.
