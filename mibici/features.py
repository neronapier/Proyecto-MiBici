"""Feature engineering para P1, P2 y P3.

Regla común: P1 y P2 pronostican el día D con la información disponible al cierre del día D-1.
Por eso todos los rezagos y promedios usan datos con al menos 24 horas de antigüedad respecto a
la hora que se predice. Las variables de calendario se conocen de antemano y no filtran futuro.
"""

import holidays
import numpy as np
import pandas as pd

from mibici.config import (
    MES_FIN,
    MES_INICIO,
    P1_LAGS_HORAS,
    P1_SEMANAS_PERFIL,
    P2_HORAS_BLOQUE,
    P2_SEMANAS_PERFIL,
    P2_UMBRAL_CAPACIDAD,
)

HORAS_DIA = 24
HORAS_SEMANA = 168
# Código fijo por municipio para usarlo como variable categórica
MUNICIPIOS = {"GDL": 0, "ZPN": 1, "TLQ": 2}


# Agregaciones base


def conteos_por_hora(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Salidas y llegadas por estación y hora, con la malla horaria completa (ceros incluidos).

    Base de P1, P2 y P3. La malla de cada estación empieza en su primera hora con actividad para
    no tratar como "cero viajes" las horas en que la estación aún no existía. Solo se conservan
    las estaciones de la nomenclatura, que es donde se conoce la capacidad.
    """
    inicio = pd.Timestamp(MES_INICIO)
    fin = pd.Period(MES_FIN, freq="M").end_time.floor("h")
    validas = set(estaciones["estacion_id"])

    salidas = (
        viajes.assign(hora=viajes["inicio"].dt.floor("h"))
        .groupby(["origen_id", "hora"])
        .size()
        .rename("salidas")
        .rename_axis(["estacion_id", "hora"])
    )
    llegadas = (
        viajes.assign(hora=viajes["fin"].dt.floor("h"))
        .groupby(["destino_id", "hora"])
        .size()
        .rename("llegadas")
        .rename_axis(["estacion_id", "hora"])
    )
    conteos = pd.concat([salidas, llegadas], axis=1).fillna(0).reset_index()
    conteos = conteos[conteos["estacion_id"].isin(validas) & conteos["hora"].between(inicio, fin)]

    horas = pd.date_range(inicio, fin, freq="h")
    ids = np.sort(conteos["estacion_id"].unique())
    malla = pd.MultiIndex.from_product([ids, horas], names=["estacion_id", "hora"])
    df = (
        conteos.set_index(["estacion_id", "hora"])
        .reindex(malla, fill_value=0)
        .reset_index()
        .astype({"estacion_id": "int32", "salidas": "int16", "llegadas": "int16"})
    )
    primera = conteos.groupby("estacion_id")["hora"].min().rename("primera_actividad")
    df = df.merge(primera, on="estacion_id")
    df = df[df["hora"] >= df["primera_actividad"]].drop(columns="primera_actividad")
    return df.sort_values(["estacion_id", "hora"]).reset_index(drop=True)


# Variables de calendario y de estación


def variables_calendario(df: pd.DataFrame, col: str = "hora") -> pd.DataFrame:
    """Hora del día, día de la semana, fin de semana, mes y día festivo nacional.

    P1 y P2: la demanda de bicicletas sigue ciclos diarios (traslados al trabajo o la escuela)
    y semanales, y cae en festivos. Se conocen antes del periodo a predecir.
    """
    t = df[col]
    festivos = holidays.Mexico(years=range(t.dt.year.min(), t.dt.year.max() + 1))
    out = df.copy()
    out["hora_dia"] = t.dt.hour.astype("int8")
    out["dia_semana"] = t.dt.dayofweek.astype("int8")
    out["es_fin_semana"] = (out["dia_semana"] >= 5).astype("int8")
    out["mes"] = t.dt.month.astype("int8")
    out["es_festivo"] = t.dt.date.isin(set(festivos)).astype("int8")
    return out


def variables_estacion(df: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Capacidad, ubicación y municipio de la estación.

    P1 y P2: la capacidad limita cuántas bicicletas pueden salir o llegar, y la ubicación
    distingue zonas de oficinas, universidades o vivienda. Son atributos fijos de la estación.
    """
    cols = ["estacion_id", "capacidad", "lat", "lon", "municipio"]
    out = df.merge(estaciones[cols], on="estacion_id", how="left")
    out["municipio_cod"] = out["municipio"].map(MUNICIPIOS).fillna(len(MUNICIPIOS)).astype("int8")
    out["capacidad"] = out["capacidad"].astype("float32")
    return out


# Rezagos y promedios sin fuga temporal


def variables_rezago(
    df: pd.DataFrame, col: str, lags: list[int], grupo: str = "estacion_id"
) -> pd.DataFrame:
    """Valor de `col` k periodos antes, por estación.

    Supone una malla completa y ordenada por estación y tiempo, de modo que desplazar k filas
    equivale a k periodos. Para no usar información futura, cada k debe ser al menos el
    horizonte de pronóstico (24 horas en P1, 8 bloques de 3 horas en P2).
    """
    out = df.copy()
    g = out.groupby(grupo, observed=True)[col]
    for k in lags:
        out[f"{col}_lag{k}"] = g.shift(k).astype("float32")
    return out


def perfil_semanal(
    df: pd.DataFrame, col: str, semanas: int, periodo: int, grupo: str = "estacion_id"
) -> pd.DataFrame:
    """Promedio de `col` en el mismo periodo (hora o bloque y día) de las últimas `semanas`.

    `periodo` es el número de filas por semana (168 horas en P1, 56 bloques en P2). Resume el
    patrón semanal reciente de la estación sin usar datos posteriores a la semana anterior.
    """
    out = df.copy()
    g = out.groupby(grupo, observed=True)[col]
    rezagos = pd.concat([g.shift(periodo * s) for s in range(1, semanas + 1)], axis=1)
    out[f"{col}_media_{semanas}sem"] = rezagos.mean(axis=1, skipna=True).astype("float32")
    return out


def nivel_reciente(
    df: pd.DataFrame, col: str, ventana: int, desfase: int, grupo: str = "estacion_id"
) -> pd.DataFrame:
    """Promedio móvil de `col` en las `ventana` filas que terminan `desfase` filas antes.

    Con desfase igual al horizonte, mide el nivel de uso más reciente que se conoce al momento
    de pronosticar (por ejemplo, el día anterior completo) y capta tendencias o eventos.
    """
    out = df.copy()
    nombre = f"{col}_media{ventana}_desf{desfase}"
    out[nombre] = (
        out.groupby(grupo, observed=True)[col]
        .transform(lambda s: s.shift(desfase).rolling(ventana, min_periods=1).mean())
        .astype("float32")
    )
    return out


def tasa_historica(df: pd.DataFrame, col: str, desfase: int, grupo: str = "estacion_id"):
    """Media acumulada de `col` por estación hasta `desfase` filas antes.

    P2: frecuencia con la que la estación ha estado en riesgo con los datos conocidos hasta el
    día anterior. Es la tasa base de cada estación y no mira hacia adelante.
    """
    out = df.copy()
    out[f"{col}_tasa_hist"] = (
        out.groupby(grupo, observed=True)[col]
        .transform(lambda s: s.shift(desfase).expanding().mean())
        .astype("float32")
    )
    return out


# P1: demanda por estación y hora


def tabla_p1(conteos: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Tabla de modelado de P1: objetivo `salidas` por estación y hora, pronóstico a un día.

    Variables: calendario, atributos de la estación, salidas y llegadas de 1, 2, 7 y 14 días
    antes en la misma hora, promedio de la misma hora en las últimas 4 semanas y promedio de
    salidas del día anterior.
    """
    df = variables_calendario(conteos)
    df = variables_estacion(df, estaciones)
    df = variables_rezago(df, "salidas", P1_LAGS_HORAS)
    df = variables_rezago(df, "llegadas", [HORAS_DIA, HORAS_SEMANA])
    df = perfil_semanal(df, "salidas", P1_SEMANAS_PERFIL, HORAS_SEMANA)
    df = nivel_reciente(df, "salidas", HORAS_DIA, HORAS_DIA)
    return df


# P2: riesgo de vaciado o llenado por bloque


def conteos_por_bloque(conteos: pd.DataFrame, horas: int = P2_HORAS_BLOQUE) -> pd.DataFrame:
    """Suma salidas y llegadas por bloques de `horas` y calcula el flujo neto.

    Flujo neto = llegadas - salidas. Un valor muy negativo indica que la estación pierde
    bicicletas (riesgo de quedar vacía) y uno muy positivo que se llena de bicicletas (riesgo de
    quedar sin anclajes).
    """
    df = conteos.assign(bloque=conteos["hora"].dt.floor(f"{horas}h"))
    df = df.groupby(["estacion_id", "bloque"], as_index=False)[["salidas", "llegadas"]].sum()
    df["flujo_neto"] = (df["llegadas"] - df["salidas"]).astype("int16")
    return df


def etiquetar_riesgo(
    bloques: pd.DataFrame, estaciones: pd.DataFrame, umbral: float = P2_UMBRAL_CAPACIDAD
) -> pd.DataFrame:
    """Aproxima el riesgo de desabasto a partir del flujo neto y la capacidad.

    Sin inventario, se supone que el rebalanceo deja cada estación a media capacidad. Entonces,
    si en un bloque la estación pierde o gana al menos `umbral` x capacidad bicicletas netas, se
    vaciaría o se llenaría. `riesgo` = 1 si ocurre cualquiera de los dos casos.
    """
    df = bloques.merge(estaciones[["estacion_id", "capacidad"]], on="estacion_id", how="left")
    limite = umbral * df["capacidad"].astype("float32")
    df["riesgo_vaciado"] = (-df["flujo_neto"] >= limite).astype("int8")
    df["riesgo_llenado"] = (df["flujo_neto"] >= limite).astype("int8")
    df["riesgo"] = (df["riesgo_vaciado"] | df["riesgo_llenado"]).astype("int8")
    return df.drop(columns="capacidad")


def tabla_p2(conteos: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Tabla de modelado de P2: objetivo `riesgo` por estación y bloque de 3 horas.

    Variables: calendario del bloque, atributos de la estación, flujo neto, salidas, llegadas y
    riesgo del mismo bloque 1 y 7 días antes, tasa de riesgo de las últimas 4 semanas y tasa
    histórica de la estación, todo conocido al cierre del día anterior. Se excluyen estaciones
    sin capacidad registrada, porque sin ella no se puede definir el umbral.
    """
    por_dia = HORAS_DIA // P2_HORAS_BLOQUE
    por_semana = 7 * por_dia
    estaciones = estaciones[estaciones["capacidad"].fillna(0) > 0]
    conteos = conteos[conteos["estacion_id"].isin(estaciones["estacion_id"])]
    df = etiquetar_riesgo(conteos_por_bloque(conteos), estaciones)
    df = variables_calendario(df, col="bloque")
    df = variables_estacion(df, estaciones)
    for col in ["flujo_neto", "salidas", "llegadas", "riesgo"]:
        df = variables_rezago(df, col, [por_dia, por_semana])
    df = perfil_semanal(df, "riesgo", P2_SEMANAS_PERFIL, por_semana)
    df = perfil_semanal(df, "flujo_neto", P2_SEMANAS_PERFIL, por_semana)
    df = tasa_historica(df, "riesgo", por_dia)
    return df


# P3: perfil de uso de cada estación


def perfil_estaciones(conteos: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Una fila por estación con su patrón de uso en el periodo [inicio, fin].

    - `sal_lab_hHH` y `lle_lab_hHH`: proporción de salidas y llegadas de días hábiles en cada
      hora; distinguen estaciones de origen matutino (vivienda) y de destino (oficinas).
    - `prop_fin_semana`: peso del fin de semana (uso recreativo frente a traslados).
    - `log_viajes_dia`: intensidad de uso (log del promedio diario de salidas).
    - `balance`: (llegadas - salidas) / total; estaciones que suelen vaciarse o llenarse.

    P3 solo usa el periodo de entrenamiento para que los grupos no dependan de datos de prueba.
    """
    df = conteos[conteos["hora"].between(pd.Timestamp(inicio), pd.Timestamp(fin))]
    df = variables_calendario(df)
    habil = df[(df["es_fin_semana"] == 0) & (df["es_festivo"] == 0)]

    perfiles = []
    for col, pref in [("salidas", "sal_lab"), ("llegadas", "lle_lab")]:
        p = habil.pivot_table(
            index="estacion_id", columns="hora_dia", values=col, aggfunc="sum", fill_value=0
        )
        p = p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        p.columns = [f"{pref}_h{h:02d}" for h in p.columns]
        perfiles.append(p)

    g = df.groupby("estacion_id")
    dias = g["hora"].agg(lambda s: s.dt.normalize().nunique())
    salidas = g["salidas"].sum()
    llegadas = g["llegadas"].sum()
    finde = df[df["es_fin_semana"] == 1].groupby("estacion_id")["salidas"].sum()
    resumen = pd.DataFrame(
        {
            "prop_fin_semana": (finde / salidas.replace(0, np.nan)).fillna(0),
            "log_viajes_dia": np.log1p(salidas / dias),
            "balance": ((llegadas - salidas) / (llegadas + salidas).replace(0, np.nan)).fillna(0),
        }
    )
    out = pd.concat(perfiles + [resumen], axis=1).fillna(0)
    return out.reset_index()


def columnas_perfil_horario(perfil: pd.DataFrame, prefijo: str = "sal_lab") -> list[str]:
    """Columnas del perfil horario con el prefijo dado (P3); el baseline agrupa solo con `sal_lab`."""
    return [c for c in perfil.columns if c.startswith(prefijo)]
