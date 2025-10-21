# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="-2Vlqbq-Qj0h"
# ##**Practica de entrenamiento de ML con XGboost**
#
# Mi objetivo es entrenar un modelo de Machine Learning con XGBoosting para predecir propinas.
#
# ###**By**
# - Samuel Mejia Chavarriaga
# - 3155128625
# - samuel1022007@gmail.com
#
# ##**Descripcion de los datos**
# Los datos en este proyecto provienen del dataset "hotel tip" hecho por JAWAD AHMAD publicado en kaggle.
# Este me parecio un dataset interesante y util para lo que busco.
# - https://www.kaggle.com/datasets/jawad3664/hotal-tip

# %% [markdown] id="8rwZUw_LZ1I6"
# # **1. Cargar librerias**

# %% id="oVF5FFmzYypc"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown] id="knQtyjaRaF9X"
# # **2. Cargar el conjunto de datos**

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="_tawZN_CX65K" outputId="39558f43-b724-47b8-d2aa-5ba521c68d70"
df = pd.read_csv('/content/sample_data/tips.csv')
df.head(10)

# %% [markdown] id="3vRCAqGcbphD"
# # **3. Analisis del conjunto de datos**
#
# El data set tiene 244 registros y 7 columnas de las cuales tenemos:
# - 1 int64 // Variables numericas
# - 2 float64 // Variable de conteo
# - 4 objects // Variables categorica
# - No hay datos nulos ni irrelevantes

# %% colab={"base_uri": "https://localhost:8080/"} id="fznZEkbVY98n" outputId="8b0f0ea4-3eb0-4302-af3a-c140bf80c239"
df.info()

# %% [markdown] id="HrkFrQs8cku9"
# # **4. Limpieza y Preparacion del conjunto de datos**
# Organizamos las variables, dejando solo las categorizables y las numericas y desechando las que no usaremos.
# XGBoost requiere que todas las variables sean numericas por lo que convertimos las variables vategoricas en numericas:
# - sex: Mujer=0, Hombre=1
# - smoker: No=0, Sí=1
# - day: Jueves=0, Viernes=1, Sábado=2, Domingo=3
# - time: Almuerzo=0, Cena=1

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="AfI8GqGTbDKX" outputId="e2079201-15b1-4a21-bf19-7ccfaf535c59"
df['sex'] = df['sex'].map({'Female':0,'Male':1})
df['smoker'] = df['smoker'].map({'No':0,'Yes':1})
df['day'] = df['day'].map({'Thur':0,'Fri':1,'Sat':2,'Sun':3})
df['time'] = df['time'].map({'Lunch':0,'Dinner':1})
df.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="PiGeYzOwcqVm" outputId="466ef3cd-503a-48f7-d8a9-f525c28d01b7"
df

# %% colab={"base_uri": "https://localhost:8080/", "height": 853} id="pdZyXxz6iOeC" outputId="655f16a8-2e42-438b-f53a-a52e5c42c6f4"
df.hist(figsize=(10,10))
plt.show()

# %% [markdown] id="gD7v_Zyqf2Ko"
# # Modelo de XGBoost
# #### Separamos las variables en independientes y dependiente
# - X = Todas las variables excepto "tip" ya que este es lo qu8e queremos predecir.
# - Y = la variable "tip" que es lo que buscamos predecir.
# - train_test_split: Dividimos 70% para entrenamiento y 30% para prueba.
#

# %% id="55c7c804"
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
features = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']
target = 'tip'
X = df[features] # Features son todas las columnas excepto 'tip'
y = df[target] # Target es la columna 'tip'

# Dividimos los datos entre sets de entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown] id="m97HBQ9NhsY5"
# #Implememtacion de XgBoost
# - **objective='reg:squarederror'** = Define que estamos trabajando con un problema de regresión (predicción de valores continuos). Este objetivo minimiza el error cuadrático entre predicciones y valores reales.
# - **n_stimators** = Número de árboles de decisión que el modelo construirá secuencialmente. Cada árbol aprende de los errores del anterior, en este caso usamos 100.
# - **learning_rate** = lo ponemos en 0.1 para controlar cuanto aprende el modelo en cada iteracion, entre mas bajo el aprendizaje es mas estable reduciendo el riesgo de sobreajuste.
# - **random_state** = Semilla para reproducibilidad. Garantiza que obtendremos los mismos resultados cada vez que ejecutemos el código. A partir de 42 se considera aleatorio.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="9e2566bc" outputId="5a2030dc-4de6-41f0-f2e1-d0c4d993ed88"
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Definimos el modelo con sus respectivos parametros
xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)

# Entrenamos el modelo con los datos de entrada y etiquetas definidas anteriormente
xgbr.fit(X_train, y_train)

# Hacemos predicciones con el modelo de prueba
y_pred = xgbr.predict(X_test)

y_pred

# %% [markdown] id="FeCYtceX9nfE"
# #**Implementacion de metricas diagnosticas** (Todo esto esta malo debido a que estoy trabajando regresion debido a mi problema por lo cual debo de a aplicar otros metodos de evaluacion)
# Generamos un reporte de clasificacion y las visualizamos
# - **Matriz de confusion (confusion_matrix)** : Evalua la precision de un modelo clasificatorio y visualiza el desempenio en terminos de verdaderos y falsos.
# - **Curva ROC (roc_curve)** : Grafica la relacion entre la tasa de verdaderos positivos y la tasa de falsos positivos a diferentes umbrales de clasificacion.

# %% id="c569d9b2"
# You can ignore this cell as confusion matrix and ROC curve are for classification problems.
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ConfusionMatrix = confusion_matrix(y_test, y_pred)

# disp = ConfusionMatrixDisplay(confusion_matrix=ConfusionMatrix, display_labels=np.array([y_test, y_pred])
# disp.plot()
# plt.show()

# %% [markdown] id="0aab0a08"
# # **Evaluación del modelo**
# Dado que este es un problema de regresión (predecir un valor continuo), utilizaremos métricas de regresión para evaluar el rendimiento del modelo.
#
# - **Mean Squared Error (MSE)**: Mide el promedio de los errores al cuadrado entre las predicciones y los valores reales. Un MSE más bajo indica un mejor ajuste del modelo.
# - **Root Mean Squared Error (RMSE)**: Es la raíz cuadrada del MSE. Tiene la misma unidad que la variable objetivo ('tip'), lo que lo hace más interpretable. Un RMSE más bajo indica un mejor rendimiento del modelo.
# - **Mean Absolute Error (MAE)**: Error absoluto medio.
# - **R² Score**: Coeficiente de determinación (0-1). Indica qué tan bien el modelo explica la variabilidad de los datos. Más cercano a 1 es mejor.
#
# ## **Métricas de Evaluación Explicadas**
#
# ### Mean Squared Error (MSE) - Error Cuadrático Medio
# - **Qué mide**: El promedio de los errores al cuadrado
# - **Cómo interpretarlo**:
#   - Valores más bajos = mejor modelo
#   - Penaliza fuertemente los errores grandes (porque eleva al cuadrado)
#   - En nuestro caso: MSE = 0.8236
# - **Ejemplo práctico**: Si predecimos una propina de 3 pero en realidad fue 4, el error es 1, pero el MSE considera este error como 1² = 1
#
# ### Root Mean Squared Error (RMSE) - Raíz del Error Cuadrático Medio
# - **Qué mide**: La raíz cuadrada del MSE
# - **Cómo interpretarlo**:
#   - Está en las mismas unidades que nuestra variable objetivo (dólares)
#   - Más fácil de interpretar que MSE
#   - En nuestro caso: RMSE = 0.9075
# - **Ejemplo práctico**: En promedio, nuestras predicciones se desvían aproximadamente $0.91 del valor real
#
# ### Mean Absolute Error (MAE) - Error Absoluto Medio
# - **Qué mide**: El promedio de los errores absolutos (sin elevar al cuadrado)
# - **Cómo interpretarlo**:
#   - No penaliza tanto los errores grandes como MSE/RMSE
#   - Más robusto ante valores atípicos
#   - En nuestro caso: MAE = 0.7050
# - **Ejemplo práctico**: En promedio, nos equivocamos por $0.70 en cada predicción
#
# ### R² Score (Coeficiente de Determinación)
# - **Qué mide**: Qué proporción de la variabilidad en los datos explica nuestro modelo
# - **Cómo interpretarlo**:
#   - Rango: 0 a 1 (0% a 100%)
#   - Más cercano a 1 = mejor modelo
#   - En nuestro caso: R² = 0.3752 (37.52%)
# - **Ejemplo práctico**: Nuestro modelo explica el 37.52% de la variación en las propinas. El 62.48% restante se debe a otros factores no capturados por nuestras variables.

# %% colab={"base_uri": "https://localhost:8080/"} id="mMEs3phDDWU8" outputId="68e8bf76-2950-4788-9240-db6335caed9d"
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Calcular métricas de regresión
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar resultados
print("=== Métricas del Modelo ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# %% [markdown] id="_t4cY9E3G_nR"
# ## Interpretación de los Gráficos
#
# ### Gráfico 1: Valores Reales vs Predicciones
# - **Línea roja**: Representa predicciones perfectas (donde predicción = valor real)
# - **Puntos dispersos**: Nuestras predicciones reales
# - **Interpretación**:
#   - Puntos cerca de la línea roja = buenas predicciones
#   - Puntos alejados = predicciones con mayor error
#   - Patrones de dispersión nos muestran dónde el modelo tiene dificultades
#
# ### Gráfico 2: Distribución de Errores
# - **Eje X**: Magnitud del error (positivo o negativo)
# - **Eje Y**: Frecuencia de cada error
# - **Línea roja**: Error = 0 (predicción perfecta)
# - **Interpretación ideal**:
#   - Distribución simétrica alrededor de 0
#   - La mayoría de errores pequeños
#   - Pocos errores grandes
#
# ### Gráfico 3: Gráfico de Residuos
# - **Residuos**: Diferencia entre valor real y predicción
# - **Línea roja horizontal**: Residuo = 0
# - **Interpretación**:
#   - Patrón aleatorio = buen modelo
#   - Patrones sistemáticos = modelo puede mejorarse
#   - Heteroscedasticidad (varianza cambiante) = posible problema

# %% colab={"base_uri": "https://localhost:8080/", "height": 965} id="GPQxOTuaCkku" outputId="23829f14-0dba-4249-8f59-84cf5f5dd612"
import matplotlib.pyplot as plt

# Gráfico: Valores reales vs predicciones
plt.figure(figsize=(5, 3))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs Predicciones')
plt.grid(True, alpha=0.3)
plt.show()

# Gráfico: Distribución de errores
errores = y_test - y_pred
plt.figure(figsize=(5, 3))
plt.hist(errores, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)
plt.show()

# Gráfico: Residuos
plt.figure(figsize=(5, 3))
plt.scatter(y_pred, errores, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown] id="_hDw0J9oHkwr"
# ## **Conclusiones**
#
# 1. **R² de 0.3752** Indica que hay relaciones pero podemos mejorar
# 2. **RMSE de $0.91** Hay un error promedio razonable pero mejorable
#
# **Posibles Mejoras**
# - **Mas datos**
#     - Recopilar mas ejemplos.
#     - Incluir mas variables como hora del dia, validad de servicio o clima.
#   
#
