"""Pruebas de la normalización de la ingesta."""

from mibici.dataset import decodificar, limpiar_viajes, mapear_columnas, meses_en_rango, parsear_fechas
from mibici.dataset import COLUMNAS_VIAJES

ENCABEZADO = (
    '"Viaje_Id","Usuario_Id","Genero","{anio}_de_nacimiento","Inicio_del_viaje",'
    '"Fin_del_viaje","Origen_Id","Destino_Id"\n'
)
FILAS = (
    '"1","10","M","1990","2025-01-01 08:00:00","2025-01-01 08:20:00","5","7"\n'
    '"2","11","F","1995","2025-01-01 09:00:00","2025-01-01 09:00:30","5","5"\n'
    '"3","12","M","1980","2025-01-01 10:00:00","2025-01-01 09:59:00","5","7"\n'
    '"1","10","M","1990","2025-01-01 08:00:00","2025-01-01 08:20:00","5","7"\n'
)


def test_meses_en_rango():
    assert meses_en_rango("2025-11", "2026-02") == ["2025_11", "2025_12", "2026_01", "2026_02"]


def test_variantes_de_encabezado():
    for anio in ["Año", "AÃ±o", "A}äe", "Ano"]:
        cols = ENCABEZADO.format(anio=anio).strip().split(",")
        mapa = mapear_columnas(cols, COLUMNAS_VIAJES)
        assert set(mapa.values()) == set(COLUMNAS_VIAJES.values())


def test_decodificar_utf8_y_latin1():
    texto = ENCABEZADO.format(anio="Año")
    assert decodificar(texto.encode("utf-8"))[1] == "utf-8"
    assert decodificar(texto.encode("latin-1"))[1] == "latin-1"


def test_limpieza_quita_datos_personales_y_registros_invalidos():
    df, stats = limpiar_viajes(ENCABEZADO.format(anio="Año") + FILAS)
    assert not {"usuario_id", "genero", "anio_nacimiento"} & set(df.columns)
    assert df["viaje_id"].tolist() == [1]
    assert stats["descartes"]["duplicados"] == 1
    assert stats["descartes"]["devolucion_inmediata"] == 1
    assert stats["descartes"]["duracion_no_positiva"] == 1


def test_formatos_de_fecha():
    import pandas as pd

    s = pd.Series(["2025-01-01 08:00:04", "01/10/2024 9:28", "basura"])
    r = parsear_fechas(s)
    assert r[0] == pd.Timestamp("2025-01-01 08:00:04")
    assert r[1] == pd.Timestamp("2024-10-01 09:28")
    assert pd.isna(r[2])
