# Contribución — heladoClassifier

Este documento resume cómo colaborar en el proyecto `heladoClassifier` (notebook principal: `BoneML.ipynb`). Incluye dependencias, flujo de trabajo con ramas, y recomendaciones para trabajar con notebooks en equipo.

## Objetivo del proyecto
El notebook `BoneML.ipynb` guía la construcción y evaluación de un clasificador (scikit-learn). Cubre carga de datos, EDA, limpieza, entrenamiento y evaluación (classification_report, confusion_matrix, curvas ROC).

## Autor
- Samuel — samuel@gmail.com — 19209002

## Dependencias
- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- (opcional) jupytext, nbdime, nbstripout, pre-commit

## Instalación rápida
Se recomienda usar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib jupytext nbdime nbstripout pre-commit
```

## Ejecución del notebook
1. Coloca el CSV de datos en la raíz o ajusta la ruta en `BoneML.ipynb`.
2. Abre el notebook:

```bash
jupyter notebook BoneML.ipynb
```

3. Ejecuta las celdas en orden y sigue las secciones del notebook.

## Flujo de trabajo con ramas
- Cada desarrollador trabaja en su rama: `user/<nombre>` (ej.: `user/samuel`).
- No empujar directamente a `main`. Abrir Pull Requests hacia `main`.
- Nombres alternativos: `feature/<desc>`, `hotfix/<desc>`.

### Pasos básicos
```bash
git clone https://github.com/cghidalgos/heladoClassifier.git
cd heladoClassifier
git fetch --all
git checkout user/samuel   # reemplaza por tu rama
```

Trabaja en tu rama, commit y push:

```bash
# (opcional) Parear con jupytext una vez por máquina:
jupytext --set-formats ipynb,py:percent BoneML.ipynb

git add BoneML.ipynb BoneML.py
git commit -m "feat: descripción corta"
git push origin user/samuel
```

Mantener la rama actualizada con `main`:
```bash
git fetch origin
git checkout user/samuel
git merge origin/main    # o git rebase origin/main
```

Abrir PR (GitHub o gh CLI):
```bash
gh pr create --base main --head user/samuel --title "feat: descripcion" --body "Detalles y cómo probar"
```

## Buenas prácticas con notebooks
- Usa `jupytext` para mantener un archivo `.py` paired (más fácil de mergear).
- Usa `nbstripout` o un hook de `pre-commit` para eliminar outputs antes de commitear.
- Usa `nbdime` para ver y mergear notebooks en caso de conflictos complejos.

## Checklist antes de PR
- PR apunta a `main`.
- Incluye descripción y pasos para probar.
- Paired `.py` incluido si se usa Jupytext.
- Outputs limpiados (nbstripout).

## Archivos adicionales que puedo crear
Si quieres que automatice la configuración, puedo crear y commitear:
- `requirements.txt` con las dependencias listadas
- `.gitattributes` para notebooks
- `.pre-commit-config.yaml` con hooks para nbstripout y jupytext

Dime si quieres que agregue esos archivos y los empuje al repo.
