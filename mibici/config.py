"""Configuración central del proyecto: rutas, fechas, semillas y parámetros.

Ningún otro archivo define rutas, fechas ni semillas; todos las importan de aquí.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Variables de entorno opcionales (.env no se versiona)
load_dotenv()

# Equipo y experimento
EQUIPO = "NN"
EXPERIMENT_NAME = f"mibici-equipo{EQUIPO.lower()}"
SEED = 42

# Rutas
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

VIAJES_DIR = PROCESSED_DATA_DIR / "viajes"
ESTACIONES_PATH = PROCESSED_DATA_DIR / "estaciones.parquet"
# Manifiesto de versionado de datos: es el único archivo de data/ que entra a Git
MANIFEST_PATH = DATA_DIR / "manifest.json"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow: por defecto, base SQLite y artefactos locales (ambos fuera de Git)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLARTIFACTS_DIR = PROJ_ROOT / "mlartifacts"

# Fuente de datos
PORTAL_URL = "https://www.mibici.net/es/datos-abiertos/"
BASE_URL = "https://www.mibici.net"
NOMENCLATURA_ARCHIVO = "nomenclatura_2026_09.csv"
# El portal rechaza peticiones sin cabeceras de navegador (Mod_Security)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.8",
}

# Rango de fechas (meses completos, inclusivo) y partición temporal
MES_INICIO = "2024-10"
MES_FIN = "2026-09"
TRAIN_FIN = "2026-03-31 23:59:59"  # train: 2024-10 a 2026-03 (18 meses)
VAL_FIN = "2026-06-30 23:59:59"  # validación: 2026-04 a 2026-06 (3 meses)
# test: 2026-07 a 2026-09 (3 meses)

# Versión de la lógica de limpieza; si cambia, cambia la versión de los datos
INGESTA_VERSION = "1"

# Limpieza de viajes
DURACION_MAX_S = 6 * 3600  # viajes más largos se consideran registros anómalos
DURACION_MIN_MISMA_ESTACION_S = 60  # devolución inmediata en la misma estación

# P1: demanda por estación y hora, pronosticada con un día de anticipación
P1_LAGS_HORAS = [24, 48, 168, 336]
P1_SEMANAS_PERFIL = 4

# P2: riesgo de vaciado o llenado por bloque de horas
P2_HORAS_BLOQUE = 3
P2_UMBRAL_CAPACIDAD = 0.5  # fracción de la capacidad que el flujo neto debe superar

# P3: clustering de estaciones
P3_K_BASELINE = 4
P3_K_CANDIDATOS = [3, 4, 5, 6, 7, 8]

# Si tqdm está instalado, loguru escribe a través de tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
