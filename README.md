# heladoClassifier
# Práctica de Entrenamiento de ML con XGBoost

## Autor
**Samuel Mejía Chavarriaga**  
- 3155128625  
- samuel1022007@gmail.com  

---

## Objetivo
Entrenar y evaluar un modelo de Machine Learning con XGBoost para predecir el valor de las propinas, utilizando el dataset *Hotel Tip* de Kaggle.  
El propósito es comprender el flujo completo del proceso de entrenamiento, desde la preparación de los datos hasta la evaluación de resultados.

---

## Descripción de los Datos
Los datos utilizados provienen del dataset publicado por **Jawad Ahmad** en Kaggle:  
[Hotel Tip Dataset](https://www.kaggle.com/datasets/jawad3664/hotal-tip)

- **Cantidad de registros:** 244  
- **Número de columnas:** 7  
- **Tipos de datos:**
  - 1 variable `int64` (numérica)
  - 2 variables `float64`
  - 4 variables categóricas (`object`)
- **Datos nulos:** No se encontraron valores faltantes.

---

## Procesamiento y Limpieza
Se realizó una preparación de datos para dejar únicamente las variables relevantes y transformarlas a formato numérico, ya que **XGBoost solo acepta valores numéricos**.

Conversión de variables categóricas:
| Variable | Conversión |
|-----------|-------------|
| `sex` | Mujer = 0, Hombre = 1 |
| `smoker` | No = 0, Sí = 1 |
| `day` | Jueves = 0, Viernes = 1, Sábado = 2, Domingo = 3 |
| `time` | Almuerzo = 0, Cena = 1 |

---

## Modelo Utilizado: XGBoost
El modelo **XGBoost Regressor** fue empleado para predecir la cantidad de propina con base en las características del cliente y las condiciones del servicio.

Pasos seguidos:
1. Carga y exploración de los datos.
2. Análisis de correlaciones y distribución de variables.
3. Codificación de variables categóricas.
4. División del dataset en entrenamiento y prueba.
5. Entrenamiento del modelo XGBoost.
6. Evaluación mediante métricas de rendimiento.

---

## Estructura de Archivos
| Archivo | Descripción |
|----------|--------------|
| `Practica_XGboost_Samuel.ipynb` | Notebook principal con el desarrollo del modelo. |
| `Practica_XGboost_Samuel.py` | Versión pareada del notebook, generada con Jupytext. |
| `README.md` | Documento descriptivo de la práctica. |

---

## Conclusión
La práctica permitió aplicar el modelo XGBoost para resolver un problema de regresión, consolidando conocimientos sobre:

- Preprocesamiento de datos.
- Codificación de variables categóricas.
- Entrenamiento y evaluación de modelos supervisados.
