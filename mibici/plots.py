"""Gráficas que se registran como artefactos en MLflow."""

from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

COLORES = plt.get_cmap("tab10").colors


def _guardar(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def serie_real_vs_pred(df: pd.DataFrame, estaciones: list[int], path: Path) -> Path:
    """Salidas reales y pronosticadas por hora en la primera semana de prueba."""
    fig, axes = plt.subplots(len(estaciones), 1, figsize=(11, 2.6 * len(estaciones)), sharex=True)
    for ax, est in zip(np.atleast_1d(axes), estaciones):
        d = df[df["estacion_id"] == est]
        ax.plot(d["hora"], d["salidas"], label="Real", color="black", lw=1)
        ax.plot(d["hora"], d["pred"], label="Pronóstico", color=COLORES[0], lw=1.2)
        ax.set_title(f"Estación {est}", fontsize=10, loc="left")
        ax.set_ylabel("Salidas/hora")
    np.atleast_1d(axes)[0].legend(loc="upper right")
    fig.suptitle("P1: salidas reales vs pronóstico (primera semana de prueba)")
    return _guardar(fig, path)


def error_por_hora(df: pd.DataFrame, path: Path) -> Path:
    """MAE por hora del día en el conjunto de prueba."""
    mae = (df["salidas"] - df["pred"]).abs().groupby(df["hora_dia"]).mean()
    media = df.groupby("hora_dia")["salidas"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(mae.index, mae.values, color=COLORES[0], label="MAE")
    ax.plot(media.index, media.values, color="black", marker="o", ms=3, label="Salidas medias")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Viajes por estación")
    ax.set_title("P1: error absoluto medio por hora (prueba)")
    ax.legend()
    return _guardar(fig, path)


def importancia(imp: pd.Series, titulo: str, path: Path) -> Path:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1))
    ax.barh(imp.index, imp.values, color=COLORES[0])
    ax.set_xlabel("Aumento del error al permutar la variable")
    ax.set_title(titulo)
    return _guardar(fig, path)


def curva_pr(y, score, titulo: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    PrecisionRecallDisplay.from_predictions(y, score, ax=ax, name="Modelo")
    ax.axhline(np.mean(y), ls="--", color="gray", label="Tasa de positivos")
    ax.set_title(titulo)
    ax.legend()
    return _guardar(fig, path)


def matriz_confusion(y, pred, titulo: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, ax=ax, display_labels=["Sin riesgo", "Riesgo"], colorbar=False, cmap="Blues"
    )
    ax.set_title(titulo)
    return _guardar(fig, path)


def riesgo_por_bloque(df: pd.DataFrame, path: Path) -> Path:
    """Tasa real y predicha de riesgo por bloque horario en prueba."""
    g = df.groupby(df["bloque"].dt.hour)[["riesgo", "pred"]].mean()
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - 0.2, g["riesgo"], 0.4, label="Real", color="gray")
    ax.bar(x + 0.2, g["pred"], 0.4, label="Predicho", color=COLORES[3])
    ax.set_xticks(x, [f"{h:02d}-{h + 3:02d}" for h in g.index])
    ax.set_ylabel("Proporción de estaciones-bloque")
    ax.set_title("P2: riesgo por bloque horario (prueba)")
    ax.legend()
    return _guardar(fig, path)


def mapa_clusters(est: pd.DataFrame, titulo: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    for c, d in est.groupby("cluster"):
        ax.scatter(d["lon"], d["lat"], s=14, color=COLORES[c % 10], label=f"Grupo {c} ({len(d)})")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    return _guardar(fig, path)


def mapa_clusters_html(est: pd.DataFrame, path: Path) -> Path:
    """Mapa interactivo de estaciones coloreadas por grupo."""
    m = folium.Map(location=[est["lat"].mean(), est["lon"].mean()], zoom_start=13)
    hex_colores = [matplotlib.colors.to_hex(c) for c in COLORES]
    for _, r in est.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=5,
            color=hex_colores[int(r["cluster"]) % 10],
            fill=True,
            fill_opacity=0.9,
            tooltip=f"{r['nombre']} | grupo {int(r['cluster'])}",
        ).add_to(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    return path


def perfiles_clusters(perfil: pd.DataFrame, path: Path) -> Path:
    """Perfil horario medio de salidas y llegadas en días hábiles por grupo."""
    sal = [c for c in perfil.columns if c.startswith("sal_lab")]
    lle = [c for c in perfil.columns if c.startswith("lle_lab")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for c, d in perfil.groupby("cluster"):
        axes[0].plot(range(24), d[sal].mean().values, color=COLORES[c % 10], label=f"Grupo {c}")
        axes[1].plot(range(24), d[lle].mean().values, color=COLORES[c % 10])
    axes[0].set_title("Salidas en días hábiles")
    axes[1].set_title("Llegadas en días hábiles")
    for ax in axes:
        ax.set_xlabel("Hora del día")
    axes[0].set_ylabel("Proporción del día")
    axes[0].legend(fontsize=8)
    fig.suptitle("P3: patrón horario medio por grupo")
    return _guardar(fig, path)


def silueta_por_k(ks: list[int], valores: list[float], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(ks, valores, marker="o", color=COLORES[0])
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Silueta (entrenamiento)")
    ax.set_title("P3: selección de k")
    return _guardar(fig, path)
