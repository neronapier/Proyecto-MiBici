"""Pruebas de que las variables no usan información posterior al pronóstico."""

import numpy as np
import pandas as pd

from mibici.features import etiquetar_riesgo, perfil_semanal, variables_rezago


def serie(n=400):
    horas = pd.date_range("2025-01-06", periods=n, freq="h")
    return pd.DataFrame({"estacion_id": 1, "hora": horas, "salidas": np.arange(n)})


def test_rezago_usa_solo_pasado():
    df = variables_rezago(serie(), "salidas", [24])
    assert df["salidas_lag24"].iloc[:24].isna().all()
    assert (df["salidas_lag24"].iloc[24:] == df["salidas"].iloc[:-24].values).all()


def test_perfil_semanal_no_incluye_valor_actual():
    df = perfil_semanal(serie(), "salidas", semanas=2, periodo=168)
    fila = df.iloc[340]
    assert fila["salidas_media_2sem"] == (340 - 168 + 340 - 336) / 2


def test_cambio_futuro_no_altera_variables_pasadas():
    a = variables_rezago(serie(), "salidas", [24, 168])
    b_base = serie()
    b_base.loc[300:, "salidas"] = -1
    b = variables_rezago(b_base, "salidas", [24, 168])
    cols = ["salidas_lag24", "salidas_lag168"]
    pd.testing.assert_frame_equal(a.loc[:323, cols], b.loc[:323, cols])


def test_etiqueta_de_riesgo():
    bloques = pd.DataFrame({"estacion_id": [1, 1, 1], "flujo_neto": [-10, 3, 10]})
    est = pd.DataFrame({"estacion_id": [1], "capacidad": [20]})
    df = etiquetar_riesgo(bloques, est, umbral=0.5)
    assert df["riesgo_vaciado"].tolist() == [1, 0, 0]
    assert df["riesgo_llenado"].tolist() == [0, 0, 1]
    assert df["riesgo"].tolist() == [1, 0, 1]
