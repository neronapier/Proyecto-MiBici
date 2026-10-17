#!/bin/bash
# Pipeline completo en el entorno TC5061 del VPS (contenedor runner, red tc5061-net).
# Lo lanza /usr/local/sbin/mlflow-ci-pipeline desde GitHub Actions; también se puede correr a mano.
# Uso: bash deploy/vps/pipeline.sh [completo|datos]
set -euo pipefail
cd "$(dirname "$0")/../.."
MODO=${1:-completo}
export MIBICI_ENTORNO=${MIBICI_ENTORNO:-vps-tc5061}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://mlflow:5000}

echo "::group::Entorno"
[ -x .venv/bin/python ] || python -m venv .venv
.venv/bin/pip install -q --disable-pip-version-check -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"
python --version
echo "::endgroup::"

echo "::group::Lint y pruebas"
ruff format --check
ruff check
python -m pytest -q tests
echo "::endgroup::"

echo "::group::Ingesta y verificación"
python -m mibici.dataset ingesta
python -m mibici.dataset verificar
if ! git diff --exit-code -- data/manifest.json; then
  echo "Los datos publicados cambiaron respecto a data/manifest.json" >&2
  exit 1
fi
echo "::endgroup::"

if [ "$MODO" = datos ]; then
  echo "=== RESUMEN ==="
  echo "### VPS: datos \`$(python -c 'from mibici.dataset import version_datos; print(version_datos())')\` verificados"
  exit 0
fi

inicio_ms=$(($(date +%s) * 1000))
echo "::group::Experimento base en MLflow ($MLFLOW_TRACKING_URI)"
python -m mibici.modeling.train todos
echo "::endgroup::"

echo "=== RESUMEN ==="
python - "$inicio_ms" <<'EOF'
import os
import sys

import mlflow

from mibici.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from mibici.dataset import version_datos

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
ui = os.getenv("MLFLOW_UI_URL", "https://161.132.54.170")
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
runs = mlflow.search_runs(
    [exp.experiment_id],
    filter_string=f"attributes.start_time >= {sys.argv[1]} and tags.entorno = 'vps-tc5061'",
)
print(f"### Experimento en el VPS: [{EXPERIMENT_NAME}]({ui}/#/experiments/{exp.experiment_id})\n")
print(f"Datos `{version_datos()}`, {len(runs)} runs nuevos.\n")
print("| Run | Pregunta | Tipo | test_mae | test_pr_auc | train_silhouette |")
print("|---|---|---|---|---|---|")
cols = ["metrics.test_mae", "metrics.test_pr_auc", "metrics.train_silhouette"]
for _, r in runs.sort_values("tags.mlflow.runName").iterrows():
    vals = ["" if c not in r or r[c] != r[c] else f"{r[c]:.3f}" for c in cols]
    enlace = f"[{r['tags.mlflow.runName']}]({ui}/#/experiments/{exp.experiment_id}/runs/{r['run_id']})"
    print(f"| {enlace} | {r['tags.pregunta']} | {r['tags.tipo_modelo']} | " + " | ".join(vals) + " |")
EOF
