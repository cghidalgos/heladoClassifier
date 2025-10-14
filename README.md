# heladoClassifier

## Guía rápida de colaboración (para BoneML.ipynb)

Esta guía explica cómo trabajar en equipo con ramas por persona, buenas prácticas para notebooks y comandos listos en zsh.

Resumen rápido
- Cada persona trabaja en su rama: `user/<nombre>` (ej.: `user/samuel`).
- No hacer push directo a `main`. Usar Pull Requests (PR) hacia `main`.
- Parear notebooks con Jupytext para facilitar diffs y merges (`BoneML.ipynb` <-> `BoneML.py`).

Convenciones
- Branch naming: `user/<nombre>` para trabajo individual; `feature/<desc>` para features; `hotfix/<desc>` para arreglos urgentes.

Comandos básicos (zsh)

Clonar y ver ramas remotas:
```bash
git clone https://github.com/cghidalgos/heladoClassifier.git
cd heladoClassifier
git fetch --all
git branch -a
```

Cambiar a tu rama (ej. Samuel):
```bash
git checkout user/samuel
# o si no existe local:
# git checkout -b user/samuel origin/user/samuel
```

Hacer cambios, commitear y pushear:
```bash
git add BoneML.ipynb BoneML.py
git commit -m "feat: descripción corta del cambio"
git push origin user/samuel
```

Actualizar tu rama con `main` antes de abrir PR:
```bash
git fetch origin
git checkout user/samuel
# Merge:
git merge origin/main
# o Rebase (historia más limpia):
git rebase origin/main
# si usas rebase y resuelves conflictos:
git push --force-with-lease origin user/samuel
```

Crear PR (GitHub UI o gh CLI):
```bash
gh pr create --base main --head user/samuel --title "feat: descripción" --body "Detalles y cómo probar"
```

Buenas prácticas con notebooks

Por qué usar Jupytext
- Los `.ipynb` son JSON y generan conflictos difíciles de resolver.
- Jupytext mantiene un `.py` paired con el notebook, lo que facilita diffs y merges en git.

Instalación y uso básico:
```bash
python -m pip install --upgrade pip
pip install jupytext nbdime nbstripout pre-commit

# Parear el notebook (una vez por máquina):
jupytext --set-formats ipynb,py:percent BoneML.ipynb
# Esto crea/actualiza BoneML.py en formato 'percent'
```

Flujo recomendado al trabajar:
- Edita `BoneML.py` en tu editor o edita el `.ipynb` (siempre mantén ambos sincronizados).
- Antes de commitear, limpia outputs con `nbstripout` o el hook de pre-commit.
- Commits pequeños y descriptivos.

Resolver conflictos en notebooks (ejemplo con Jupytext):
```bash
# Si aparece un conflicto en BoneML.ipynb, convierte/asegura el .py:
jupytext --to py:percent BoneML.ipynb
# Edita BoneML.py para resolver conflictos (es texto y más fácil)
git add BoneML.py
git commit -m "fix: resolver conflicto en BoneML via jupytext"
# Sincroniza de vuelta al ipynb
jupytext --sync BoneML.ipynb
git add BoneML.ipynb
git commit -m "sync: BoneML.py -> BoneML.ipynb después de resolver conflicto"
git push origin user/samuel
```

Herramientas útiles
- nbdime: diffs y merges especializados para notebooks.
	- `pip install nbdime` y `nbdime config-git --enable`.
- nbstripout: limpia outputs antes de commitear.
	- `pip install nbstripout` y `nbstripout --install`.

Checklist antes de abrir PR
- PR apunta a `main`.
- Descripción clara y pasos para probar.
- No hay outputs grandes en el `.ipynb` (usar nbstripout).
- Paired `.py` incluido (si usan Jupytext).

Si quieres, puedo añadir automáticamente al repo los siguientes archivos:
- `CONTRIBUTING.md` con esta guía ampliada
- `.gitattributes` para notebooks
- `requirements.txt` con jupytext/nbdime/nbstripout
- `.pre-commit-config.yaml` mínimo

Dime si quieres que los cree (opción: crear archivos y commitear al repo) y lo hago.