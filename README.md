# TC5061 Proyecto MiBici, Equipo NN

[![CI](https://github.com/neronapier/Proyecto-MiBici/actions/workflows/ci.yml/badge.svg)](https://github.com/neronapier/Proyecto-MiBici/actions/workflows/ci.yml)
[![Pipeline MiBici](https://github.com/neronapier/Proyecto-MiBici/actions/workflows/pipeline.yml/badge.svg)](https://github.com/neronapier/Proyecto-MiBici/actions/workflows/pipeline.yml)
[![Pipeline VPS](https://github.com/neronapier/Proyecto-MiBici/actions/workflows/vps.yml/badge.svg)](https://github.com/neronapier/Proyecto-MiBici/actions/workflows/vps.yml)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto integrador del curso TC5061 Operaciones de Aprendizaje Automático (MLOps), Tec de Monterrey.
Propone y valida una solución de Machine Learning para la operación de MiBici, el sistema de bicicleta
pública del Área Metropolitana de Guadalajara, sobre una base de datos reproducible y versionada.

## Preguntas del proyecto

| # | Pregunta | Tarea | Formulación | Usuario |
|---|----------|-------|-------------|---------|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Salidas por estación y hora del día siguiente, con información hasta el cierre del día anterior. También se reporta el error agregado por municipio. | Operador del sistema |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Riesgo por estación y bloque de 3 h del día siguiente. Riesgo = 1 si \|llegadas - salidas\| >= 0.5 x capacidad en el bloque. | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Grupos de estaciones según su perfil horario de salidas y llegadas, peso del fin de semana, volumen y balance. | Planeación urbana |

P2 no cuenta con inventario por estación. Se supone que el rebalanceo deja cada estación a media capacidad;
si en un bloque pierde o gana al menos la mitad de su capacidad en bicicletas netas, se vaciaría o se llenaría.

El ML Canvas está en [`references/ml_canvas.pdf`](references/ml_canvas.pdf) y la arquitectura en
[`references/arquitectura.pdf`](references/arquitectura.pdf).

![Arquitectura](references/arquitectura.png)

## Datos

- Fuente: [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/), archivos
  `datos_abiertos_AAAA_MM.csv` y `nomenclatura_2026_09.csv`. Los enlaces exactos quedan en `data/manifest.json`.
- Rango: enero de 2025 a septiembre de 2026 (21 meses; septiembre de 2026 es el último mes publicado). Cubre
  2025 completo y lo publicado de 2026, y los meses de prueba tienen su equivalente del año anterior en
  entrenamiento.
- Tamaño: 7,894,097 registros crudos (685 MB), 7,666,077 viajes limpios, 484 estaciones en la nomenclatura
  (470 con actividad en el periodo; 101 de ellas empezaron a operar entre abril y septiembre de 2026).
- Partición temporal, sin mezclar fechas:

| Conjunto | Periodo |
|----------|---------|
| Entrenamiento | 2025-01 a 2026-03 |
| Validación | 2026-04 a 2026-06 |
| Prueba | 2026-07 a 2026-09 |

- Datos personales: los archivos incluyen `Usuario_Id`, `Genero` y `Año_de_nacimiento`. La ingesta los
  elimina antes de guardar los Parquet; ninguna pregunta los necesita. Ningún dato de MiBici se versiona en
  Git ni se registra en MLflow.

## Estructura

```
├── Makefile               <- Comandos: make data, make train, make mlflow-ui, make test
├── README.md
├── pyproject.toml         <- Metadatos del paquete y configuración de ruff y pytest
├── requirements.txt       <- Dependencias con versión fija
├── deploy/vps             <- Pipeline en el VPS (pipeline.sh) y archivos de la cuenta mlflow-ci del servidor
├── data
│   ├── manifest.json      <- Versión de los datos: filas y SHA-256 por archivo (único archivo versionado)
│   ├── raw                <- CSV originales descargados (no se versionan)
│   ├── interim            <- Intermedios (no se versionan)
│   └── processed          <- Viajes y estaciones limpios en Parquet (no se versionan)
├── models                 <- Modelos serializados (no se versionan; se registran en MLflow)
├── notebooks
│   └── 1.0-jpbc-exploracion-mibici.ipynb   <- Solo exploración
├── references             <- ML Canvas y diagrama de arquitectura (PDF, PNG y fuente LaTeX)
├── reports/figures        <- Figuras de la exploración; las de cada run se generan aquí y se registran en MLflow
├── tests                  <- Pruebas de ingesta y de ausencia de fuga temporal
└── mibici                 <- Código del proyecto
    ├── config.py          <- Rutas, fechas, partición, semilla y parámetros (único lugar)
    ├── dataset.py         <- Ingesta reproducible y versionado de datos
    ├── features.py        <- Feature engineering por pregunta
    ├── plots.py           <- Gráficas que se registran como artefactos
    └── modeling
        └── train.py       <- Experimento base de P1, P2 y P3 con MLflow
```

## Cómo se generó la estructura

Primer commit del repositorio, generado con Cookiecutter Data Science 2.3.0:

```bash
pipx install cookiecutter-data-science
ccds   # respuestas: module_name=mibici, python_version_number=3.12, environment_manager=uv,
       # dependency_file=requirements.txt, pydata_packages=none, testing_framework=pytest,
       # linting_and_formatting=ruff, open_source_license=MIT, docs=none, include_code_scaffold=Yes
```

## Instalación

Requiere Python 3.12, [uv](https://docs.astral.sh/uv/) y Git. Desde un clon limpio:

```bash
git clone https://github.com/neronapier/Proyecto-MiBici.git
cd Proyecto-MiBici
make create_environment          # uv venv --python 3.12
source .venv/bin/activate
make requirements                # uv pip install -r requirements.txt
```

Sin `make`: `uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt`.
Con `pip` también funciona: `python3.12 -m venv .venv && pip install -r requirements.txt`.

## Ingesta de datos

```bash
make data      # python -m mibici.dataset ingesta
make verify    # python -m mibici.dataset verificar
```

La ingesta, sin pasos manuales:

1. Lee el portal y obtiene los enlaces de cada mes (las rutas incluyen identificadores que no se pueden
   deducir). El portal rechaza peticiones sin cabeceras de navegador, por eso se envían en `config.py`.
2. Descarga los CSV a `data/raw/` (si ya existen no los vuelve a descargar).
3. Normaliza codificación (en el rango elegido hay meses en UTF-8 y en Latin-1), nombres de columnas y fechas.
   También tolera variantes de años anteriores del portal: encabezados sin comillas, la "ñ" de
   `Año_de_nacimiento` mal codificada (`A}äe`) y fechas `DD/MM/AAAA H:MM` (octubre de 2024), así que el rango
   puede ampliarse cambiando solo `config.py`.
4. Elimina datos personales, nulos, duplicados (758 en septiembre de 2026), duraciones no positivas o
   mayores a 6 h y devoluciones en la misma estación en menos de 60 s.
5. Guarda un Parquet por mes en `data/processed/viajes/` y la nomenclatura en `data/processed/estaciones.parquet`.
6. Escribe `data/manifest.json` con URL, bytes, codificación, filas crudas y limpias, descartes por motivo y
   SHA-256 del CSV y del Parquet de cada archivo.

La versión de los datos (`data_version`, por ejemplo `v1-fc921063bb48`) es un hash del contenido crudo y limpio
y de la versión de la lógica de limpieza; no incluye fechas de ejecución. Correr la ingesta dos veces da la
misma versión. Si MiBici modifica un archivo ya publicado, la ingesta lo advierte y la versión cambia.
`make verify` comprueba los hashes locales contra el manifiesto. Cada run registra la versión en MLflow.

## Experimento base

```bash
make train                       # python -m mibici.modeling.train todos
make mlflow-ui                   # http://127.0.0.1:5000
```

Una sola pregunta o un solo modelo: `python -m mibici.modeling.train p2 --modelo baseline`.

Por defecto MLflow usa `sqlite:///mlflow.db` y guarda los artefactos en `mlartifacts/` (ambos fuera de Git).
Para usar un servidor remoto basta con definir `MLFLOW_TRACKING_URI` (por ejemplo en `.env`).
Los resultados oficiales del equipo están en el servidor de MLflow del VPS, de solo lectura:
https://161.132.54.170 (experimento `mibici-equiponn`, runs con etiqueta `entorno=vps-tc5061`).

| Pregunta | Run | Modelo | Métricas principales |
|----------|-----|--------|----------------------|
| P1 | `p1-baseline-naive-estacional-7d` | Salidas de la misma hora 7 días antes | `test_mae`, `test_rmse`, `test_wape`, `test_wape_municipio` |
| P1 | `p1-hgb-poisson` | HistGradientBoostingRegressor, pérdida Poisson | igual |
| P2 | `p2-baseline-frecuencia-4sem` | Frecuencia de riesgo del mismo bloque en 4 semanas | `test_pr_auc`, `test_recall`, `test_precision`, `test_f1` |
| P2 | `p2-hgb-clasificador` | HistGradientBoostingClassifier | igual |
| P3 | `p3-baseline-kmeans-k4-perfil-salidas` | K-means, k = 4, perfil horario de salidas | `train_silhouette`, `train_davies_bouldin`, `test_ari_estabilidad` |
| P3 | `p3-pca-kmeans-perfil-completo` | Estandarización + PCA + k-means, k por silueta | igual |

En P2 el umbral de decisión se elige en validación (máximo F1) y se aplica sin cambios en prueba.

## Automatización con GitHub Actions

| Workflow | Cuándo corre | Qué hace |
|----------|--------------|----------|
| [`ci.yml`](.github/workflows/ci.yml) | Cada Pull Request y cada push a `main` | `make lint`, `make test`, revisa que no haya datos, modelos, MLflow ni `.env` en Git y, en los Pull Request, que la rama y el título sigan las convenciones |
| [`pipeline.yml`](.github/workflows/pipeline.yml) | Manual (*Run workflow*), push a `main` que toque código, datos o dependencias, y cada lunes | Ingesta (`make data`), verificación de hashes (`make verify`), comprobación de que `data/manifest.json` no cambió y, salvo la corrida semanal, el experimento base (`make train`) |
| [`vps.yml`](.github/workflows/vps.yml) | Manual y push a `main` que toque código, datos, dependencias o `deploy/vps/` | El mismo pipeline dentro del entorno MLOps del VPS; los runs quedan en el servidor de MLflow del curso |

- Los CSV crudos se guardan en la caché de Actions con la versión del manifiesto como llave; los datos nunca
  entran a Git.
- El experimento corre contra un servidor de MLflow dentro del runner. La base (`mlflow.db`), los artefactos
  (`mlartifacts/`) y las figuras se publican como artefacto del workflow (30 días) y el resumen de métricas
  aparece en la página de la corrida.
- Para revisar una corrida en local: descargar el artefacto `mlflow-<version>-<commit>`, descomprimirlo en la
  raíz del repositorio y ejecutar `make mlflow-ui`.
- La corrida semanal funciona como monitoreo de la fuente: falla si MiBici modifica un archivo ya publicado.

### Despliegue en el VPS

`vps.yml` ordena por SSH, con una cuenta restringida (`mlflow-ci`), que el servidor del curso ejecute
[`deploy/vps/pipeline.sh`](deploy/vps/pipeline.sh) para el commit de `main`: lint, pruebas, ingesta, verificación
y experimento base contra su servidor de MLflow (artefactos en MinIO). Los runs llevan la etiqueta
`entorno=vps-tc5061` y se consultan en modo solo lectura en https://161.132.54.170. Detalle y configuración en
[`deploy/vps/README.md`](deploy/vps/README.md).

## Buenas prácticas aplicadas

| Práctica | Cómo se aplica |
|----------|----------------|
| Un experimento y runs con nombre descriptivo | Experimento `mibici-equiponn`; cada run tiene nombre `p<n>-<modelo>` |
| Registrar lo necesario para reproducir | Parámetros `mes_inicio`, `mes_fin`, `train_fin`, `val_fin`, `data_version`, `seed`, modelo e hiperparámetros, variables usadas |
| Ligar el run con el código y la pregunta | Etiquetas `equipo`, `pregunta`, `tipo_modelo`, `git_commit`, `git_branch`, `git_dirty` y `entorno` (local, github-actions o vps-tc5061) |
| Métricas con el conjunto en el nombre | `val_mae`, `test_mae`, `test_pr_auc`, `train_silhouette`, etc. |
| Sin credenciales ni datos personales | La ingesta elimina los datos de usuario; los artefactos son gráficas, el manifiesto y una tabla de estaciones (datos públicos) |
| Notebooks solo para exploración | `notebooks/1.0-jpbc-exploracion-mibici.ipynb`; la lógica vive en `mibici/` |
| Configuración en un solo lugar | Rutas, fechas, partición, semilla y umbrales en `mibici/config.py` |
| Dependencias fijas | `requirements.txt` con `==`; entorno con Python 3.12 |
| Nada de datos, modelos ni MLflow en Git | `.gitignore` excluye `data/*` (salvo el manifiesto), `mlruns/`, `mlartifacts/`, `mlflow.db`, `models/*`, `*.pkl`, `.env` |
| Todo cambio por Pull Request | Rama `main` protegida; ver flujo de trabajo |
| Sin fuga de información del futuro | Rezagos de al menos 24 h, partición temporal y P3 ajustado solo con entrenamiento; probado en `tests/test_features.py` |

## Flujo de trabajo con Git

Se sigue [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):

1. `main` está protegida: no admite push directo y exige un Pull Request con al menos una aprobación.
2. Cada cambio se hace en una rama creada desde `main`.
3. Se abre un Pull Request hacia `main`; otro integrante lo revisa y lo aprueba.
4. Se integra con merge commit y se borra la rama.

Convención de ramas: `<tipo>/<descripcion-corta>`, con tipo `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`
o `perf`.
Ejemplos: `feat/ingesta-datos`, `fix/estaciones-retiradas`, `docs/readme`.

Convención de commits: [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/),
`<tipo>(<alcance>): <descripción>`. Ejemplos: `feat(dataset): ingesta reproducible con manifiesto de version`,
`fix(dataset): municipio de estaciones retiradas`.

Antes de abrir un Pull Request: `make lint` y `make test`. El workflow `CI` repite ambos y valida la rama y el
título; la protección de `main` exige que pase y que otro integrante apruebe.

## Integrantes

- Jordan Piero Borda Colque
- Julián Hazael Martínez Hernández
- Oscar Eduardo Morales Portuondo
- Eduardo Sebastián Vela Alanís

## Referencias

- Dorard, L. (2018, 9 de marzo). *From data to AI with the Machine Learning Canvas (Part I)*. Own Machine Learning.
- Huyen, C. (2022). *Designing machine learning systems*. O'Reilly Media.
- MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. https://www.mibici.net/es/datos-abiertos/
- MLflow. (s.f.). *MLflow Tracking*. https://mlflow.org/docs/latest/ml/tracking/
- Treveil, M., et al. (2020). *Introducing MLOps*. O'Reilly Media.
- Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.
