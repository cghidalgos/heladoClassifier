# heladoClassifier

Este repositorio contiene dos notebooks principales para el análisis y modelado de datos de obesidad:

## 1. GHS_classifier.ipynb
- Análisis exploratorio, limpieza y preprocesamiento de los datos.
- Modelo de clasificación (CatBoostClassifier) para predecir la clase de obesidad.
- Métricas de clasificación y visualizaciones (matriz de confusión, curva ROC).
- Discusión de resultados y recomendaciones.

## 2. GHS_regression.ipynb
- Flujo similar pero enfocado en regresión.
- Transforma la variable de obesidad a formato numérico (`Obesity_numeric`).
- Modelo de regresión (CatBoostRegressor) para predecir el nivel de obesidad como valor numérico.
- Métricas de regresión y gráficos de comparación real vs predicho.
- Discusión de resultados y sugerencias de mejora.

## Requisitos
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, catboost, kagglehub

## Uso
1. Abre cada notebook en Jupyter o VS Code.
2. Ejecuta las celdas en orden para reproducir el análisis y los modelos.
3. Ajusta rutas de archivos si es necesario para cargar el dataset.

---

Cualquier duda, contactar a Giovanny Hidalgo (cgiohidalgo@gmail.com)