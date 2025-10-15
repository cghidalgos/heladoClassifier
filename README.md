# heladoClassifier

## Guía rápida de colaboración (para BoneML.ipynb)

Esta guía explica cómo trabajar en equipo con ramas por persona, buenas prácticas para notebooks.

Resumen rápido
- Cada persona trabaja en su rama: `user/<nombre>` (ej.: `user/samuel`).
- No hacer push directo a `main`. Usar Pull Requests (PR) hacia `main`.
- Parear notebooks con Jupytext para facilitar diffs y merges (`BoneML.ipynb` <-> `BoneML.py`).

Convenciones
- Branch naming: `user/<nombre>` para trabajo individual; `feature/<desc>` para features; `hotfix/<desc>` para arreglos urgentes.

Comandos básicos (zsh)
## Modelo de machine learning 
Clonar y ver ramas remotas:
```bash
git clone https://github.com/cghidalgos/heladoClassifier.git
cd heladoClassifier
git fetch --all
git branch -a
```

Cada uno tiene una rama 
```bash
git branch
```

Cambiar a tu rama (ej. Samuel):
```bash
git checkout user/samuel
```


Hacer cambios, commitear y pushear (hacer el commit y enviar):
```bash
git add BoneML.ipynb BoneML.py
git commit -m "feat: descripción corta del cambio"
git push origin user/samuel
```

<span style="color:red;">cada persona debe crear su propio readme de su proyecto en su rama</span>
