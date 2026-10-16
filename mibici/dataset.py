"""Ingesta reproducible de MiBici.

Descarga los meses configurados y la nomenclatura de estaciones, normaliza columnas,
codificación y fechas, guarda Parquet y escribe un manifiesto versionado en Git.

Uso:
    python -m mibici.dataset ingesta      # descarga, limpia y genera el manifiesto
    python -m mibici.dataset verificar    # comprueba los archivos locales contra el manifiesto
"""

from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import re
import time
import unicodedata

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from mibici.config import (
    BASE_URL,
    DESCARGA_INTENTOS,
    DESCARGA_MAX_S,
    DURACION_MAX_S,
    DURACION_MIN_MISMA_ESTACION_S,
    ESTACIONES_PATH,
    HTTP_HEADERS,
    INGESTA_VERSION,
    MANIFEST_PATH,
    MES_FIN,
    MES_INICIO,
    NOMENCLATURA_ARCHIVO,
    PORTAL_URL,
    PROJ_ROOT,
    RAW_DATA_DIR,
    VIAJES_DIR,
)

app = typer.Typer(add_completion=False)

# Nombre normalizado -> nombre canónico. Los datos personales se descartan al limpiar.
COLUMNAS_VIAJES = {
    "viaje_id": "viaje_id",
    "usuario_id": "usuario_id",
    "genero": "genero",
    "nacimiento": "anio_nacimiento",  # el nombre original llega con "ñ" mal codificada
    "inicio_del_viaje": "inicio",
    "fin_del_viaje": "fin",
    "origen_id": "origen_id",
    "destino_id": "destino_id",
}
COLUMNAS_PERSONALES = ["usuario_id", "genero", "anio_nacimiento"]
COLUMNAS_ESTACIONES = {
    "id": "estacion_id",
    "name": "nombre",
    "latitude": "lat",
    "longitude": "lon",
    "dpcapacity": "capacidad",
}


def sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def meses_en_rango(inicio: str, fin: str) -> list[str]:
    """Lista de meses 'AAAA_MM' entre inicio y fin (formato 'AAAA-MM'), inclusive."""
    periodos = pd.period_range(inicio, fin, freq="M")
    return [p.strftime("%Y_%m") for p in periodos]


def normalizar_nombre(nombre: str) -> str:
    """Quita comillas, acentos y caracteres extraños; deja minúsculas y guiones bajos."""
    texto = unicodedata.normalize("NFKD", nombre.strip().strip('"'))
    texto = texto.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", texto).strip("_")


def mapear_columnas(columnas: list[str], esquema: dict[str, str]) -> dict[str, str]:
    """Relaciona cada columna original con su nombre canónico por coincidencia parcial."""
    mapa = {}
    for col in columnas:
        norm = normalizar_nombre(col)
        for clave, canonico in esquema.items():
            if norm == clave or (clave == "nacimiento" and clave in norm):
                mapa[col] = canonico
                break
    faltantes = set(esquema.values()) - set(mapa.values())
    if faltantes:
        raise ValueError(f"Columnas no reconocidas; faltan {sorted(faltantes)} en {columnas}")
    return mapa


def decodificar(contenido: bytes) -> tuple[str, str]:
    """Decodifica en UTF-8 y, si falla, en Latin-1 (los archivos alternan entre ambas)."""
    try:
        return contenido.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return contenido.decode("latin-1"), "latin-1"


def enlaces_portal(session: requests.Session) -> dict[str, str]:
    """Lee el portal y devuelve {nombre_archivo: url}; las rutas incluyen ids no predecibles."""
    resp = session.get(PORTAL_URL, timeout=60)
    resp.raise_for_status()
    rutas = re.findall(r'href="(/site/assets/files/\d+/[^"]+\.csv)"', resp.text)
    enlaces = {}
    for ruta in rutas:
        nombre = Path(ruta).name
        # Algunos archivos se publicaron con sufijo (p. ej. "_01-1.csv")
        nombre = re.sub(r"-\d+\.csv$", ".csv", nombre)
        enlaces.setdefault(nombre, BASE_URL + ruta)
    return enlaces


def descargar(session: requests.Session, url: str, destino: Path) -> None:
    """Descarga con límite de tiempo por intento; si la conexión se estanca, reintenta con otra."""
    tmp = destino.with_suffix(".part")
    for intento in range(1, DESCARGA_INTENTOS + 1):
        inicio = time.monotonic()
        try:
            with session.get(url, stream=True, timeout=(15, 60)) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for bloque in resp.iter_content(1 << 20):
                        f.write(bloque)
                        if time.monotonic() - inicio > DESCARGA_MAX_S:
                            raise requests.Timeout(f"más de {DESCARGA_MAX_S} s")
            tmp.rename(destino)
            return
        except requests.RequestException as err:
            logger.warning(f"Fallo al descargar {url} (intento {intento}): {err}")
            session.close()  # descarta conexiones lentas; el siguiente intento abre una nueva
            time.sleep(5 * intento)
    raise RuntimeError(f"No se pudo descargar {url}")


def parsear_fechas(serie: pd.Series) -> pd.Series:
    """Convierte fechas ISO (AAAA-MM-DD HH:MM:SS) o DD/MM/AAAA H:MM, que usan algunos meses."""
    serie = serie.str.strip()
    fechas = pd.to_datetime(serie, format="ISO8601", errors="coerce")
    faltan = fechas.isna()
    if faltan.any():
        fechas[faltan] = pd.to_datetime(serie[faltan], format="%d/%m/%Y %H:%M", errors="coerce")
    return fechas


def limpiar_viajes(texto: str) -> tuple[pd.DataFrame, dict]:
    """Normaliza un CSV mensual de viajes y elimina datos personales y registros inválidos."""
    df = pd.read_csv(io.StringIO(texto), dtype=str)
    df = df.rename(columns=mapear_columnas(list(df.columns), COLUMNAS_VIAJES))
    filas_crudas = len(df)

    # Minimización: se descartan id de usuario, género y año de nacimiento
    df = df.drop(columns=COLUMNAS_PERSONALES)

    for col in ["inicio", "fin"]:
        df[col] = parsear_fechas(df[col])
    for col in ["viaje_id", "origen_id", "destino_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    descartes = {}
    invalidos = df.isna().any(axis=1)
    descartes["nulos_o_formato"] = int(invalidos.sum())
    df = df[~invalidos]

    duplicados = df.duplicated("viaje_id")
    descartes["duplicados"] = int(duplicados.sum())
    df = df[~duplicados]

    df = df.astype({"viaje_id": "int64", "origen_id": "int32", "destino_id": "int32"})
    df["inicio"] = df["inicio"].astype("datetime64[s]")
    df["fin"] = df["fin"].astype("datetime64[s]")
    df["duracion_s"] = (df["fin"] - df["inicio"]).dt.total_seconds().astype("int32")

    no_positiva = df["duracion_s"] <= 0
    muy_larga = df["duracion_s"] > DURACION_MAX_S
    devolucion = (df["origen_id"] == df["destino_id"]) & (
        df["duracion_s"] < DURACION_MIN_MISMA_ESTACION_S
    )
    descartes["duracion_no_positiva"] = int(no_positiva.sum())
    descartes["duracion_mayor_max"] = int(muy_larga.sum())
    descartes["devolucion_inmediata"] = int((devolucion & ~no_positiva).sum())
    df = df[~(no_positiva | muy_larga | devolucion)]

    df = df.sort_values(["inicio", "viaje_id"]).reset_index(drop=True)
    stats = {"filas_crudas": filas_crudas, "filas_limpias": len(df), "descartes": descartes}
    return df, stats


def limpiar_estaciones(texto: str) -> pd.DataFrame:
    """Normaliza la nomenclatura, extrae el municipio del código (p. ej. GDL-001) y marca las
    estaciones retiradas, cuyo nombre empieza con "ex-"."""
    df = pd.read_csv(io.StringIO(texto), dtype=str)
    df = df.rename(columns=mapear_columnas(list(df.columns), COLUMNAS_ESTACIONES))
    df["nombre"] = df["nombre"].str.strip()
    df["estacion_id"] = df["estacion_id"].astype("int32")
    df["lat"] = df["lat"].astype("float64")
    df["lon"] = df["lon"].astype("float64")
    df["capacidad"] = pd.to_numeric(df["capacidad"], errors="coerce").astype("Int32")
    df["municipio"] = df["nombre"].str.extract(r"([A-Z]{3})-\d+")[0].fillna("OTRO")
    df["retirada"] = df["nombre"].str.lower().str.startswith("ex-")
    return df.sort_values("estacion_id").reset_index(drop=True)


def guardar_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def calcular_version(manifiesto: dict) -> str:
    """Hash del contenido crudo y limpio de cada archivo y de la versión de la limpieza.

    No incluye fechas de ejecución: la misma fuente con el mismo código da la misma versión.
    """
    base = {
        "ingesta_version": manifiesto["ingesta_version"],
        "rango": manifiesto["rango"],
        "archivos": [
            (a["archivo"], a["raw_sha256"], a["parquet_sha256"], a["filas_limpias"])
            for a in manifiesto["archivos"]
        ],
        "estaciones": (
            manifiesto["estaciones"]["raw_sha256"],
            manifiesto["estaciones"]["parquet_sha256"],
        ),
    }
    digest = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    return f"v{INGESTA_VERSION}-{digest[:12]}"


def cargar_manifiesto(path: Path = MANIFEST_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def version_datos(path: Path = MANIFEST_PATH) -> str:
    """Versión de los datos para registrar en MLflow."""
    return cargar_manifiesto(path)["version"]


def cargar_viajes(columnas: list[str] | None = None) -> pd.DataFrame:
    """Lee los Parquet de viajes listados en el manifiesto."""
    manifiesto = cargar_manifiesto()
    partes = [
        pd.read_parquet(PROJ_ROOT / a["parquet"], columns=columnas) for a in manifiesto["archivos"]
    ]
    return pd.concat(partes, ignore_index=True)


def cargar_estaciones() -> pd.DataFrame:
    return pd.read_parquet(ESTACIONES_PATH)


def ruta_relativa(path: Path) -> str:
    return path.relative_to(PROJ_ROOT).as_posix()


@app.command()
def ingesta(
    inicio: str = typer.Option(MES_INICIO, help="Primer mes, AAAA-MM"),
    fin: str = typer.Option(MES_FIN, help="Último mes, AAAA-MM"),
    forzar: bool = typer.Option(False, help="Volver a descargar aunque el archivo exista"),
):
    """Descarga, normaliza, guarda en Parquet y escribe el manifiesto de versión."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    previo = cargar_manifiesto() if MANIFEST_PATH.exists() else None
    hash_previo = {a["archivo"]: a["raw_sha256"] for a in (previo or {}).get("archivos", [])}

    session = requests.Session()
    session.headers.update(HTTP_HEADERS)
    enlaces = enlaces_portal(session)

    archivos = []
    for mes in tqdm(meses_en_rango(inicio, fin), desc="Meses"):
        nombre = f"datos_abiertos_{mes}.csv"
        if nombre not in enlaces:
            raise RuntimeError(f"{nombre} no aparece en el portal")
        raw_path = RAW_DATA_DIR / nombre
        if forzar or not raw_path.exists():
            descargar(session, enlaces[nombre], raw_path)
        raw_sha = sha256_archivo(raw_path)
        if nombre in hash_previo and hash_previo[nombre] != raw_sha:
            logger.warning(f"{nombre} cambió respecto al manifiesto: se generará otra versión")

        texto, encoding = decodificar(raw_path.read_bytes())
        df, stats = limpiar_viajes(texto)
        pq_path = VIAJES_DIR / f"viajes_{mes}.parquet"
        guardar_parquet(df, pq_path)
        archivos.append(
            {
                "archivo": nombre,
                "mes": mes.replace("_", "-"),
                "url": enlaces[nombre],
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": raw_sha,
                "encoding": encoding,
                **stats,
                "parquet": ruta_relativa(pq_path),
                "parquet_sha256": sha256_archivo(pq_path),
            }
        )
        logger.info(f"{nombre}: {stats['filas_limpias']:,} viajes limpios ({encoding})")

    nom_path = RAW_DATA_DIR / NOMENCLATURA_ARCHIVO
    if forzar or not nom_path.exists():
        descargar(session, enlaces[NOMENCLATURA_ARCHIVO], nom_path)
    texto, encoding = decodificar(nom_path.read_bytes())
    estaciones = limpiar_estaciones(texto)
    guardar_parquet(estaciones, ESTACIONES_PATH)

    viajes_total = sum(a["filas_limpias"] for a in archivos)
    manifiesto = {
        "ingesta_version": INGESTA_VERSION,
        "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "fuente": PORTAL_URL,
        "rango": {"inicio": inicio, "fin": fin},
        "totales": {
            "meses": len(archivos),
            "viajes_crudos": sum(a["filas_crudas"] for a in archivos),
            "viajes_limpios": viajes_total,
            "estaciones_nomenclatura": len(estaciones),
        },
        "archivos": archivos,
        "estaciones": {
            "archivo": NOMENCLATURA_ARCHIVO,
            "url": enlaces[NOMENCLATURA_ARCHIVO],
            "raw_sha256": sha256_archivo(nom_path),
            "encoding": encoding,
            "filas": len(estaciones),
            "parquet": ruta_relativa(ESTACIONES_PATH),
            "parquet_sha256": sha256_archivo(ESTACIONES_PATH),
        },
    }
    manifiesto["version"] = calcular_version(manifiesto)
    # Si el contenido no cambió se conserva la fecha original para no modificar el archivo en Git
    sin_fecha = {k: v for k, v in manifiesto.items() if k != "generado_utc"}
    if previo and {k: v for k, v in previo.items() if k != "generado_utc"} == sin_fecha:
        manifiesto["generado_utc"] = previo["generado_utc"]
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.success(
        f"Datos versión {manifiesto['version']}: {len(archivos)} meses, "
        f"{viajes_total:,} viajes, {len(estaciones)} estaciones"
    )


@app.command()
def verificar():
    """Comprueba que los archivos locales coinciden con el manifiesto versionado."""
    manifiesto = cargar_manifiesto()
    entradas = manifiesto["archivos"] + [manifiesto["estaciones"]]
    errores = 0
    for a in entradas:
        raw = RAW_DATA_DIR / a["archivo"]
        pq = PROJ_ROOT / a["parquet"]
        for path, esperado in [(raw, a["raw_sha256"]), (pq, a["parquet_sha256"])]:
            if not path.exists():
                logger.error(f"Falta {ruta_relativa(path)}")
                errores += 1
            elif sha256_archivo(path) != esperado:
                logger.error(f"Hash distinto en {ruta_relativa(path)}")
                errores += 1
    if errores:
        raise typer.Exit(code=1)
    logger.success(f"Datos íntegros: versión {manifiesto['version']}")


if __name__ == "__main__":
    app()
