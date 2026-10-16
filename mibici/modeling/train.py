"""Experimento base: baseline y modelo candidato para P1, P2 y P3, registrados en MLflow.

Uso:
    python -m mibici.modeling.train todos         # las tres preguntas
    python -m mibici.modeling.train p1 --modelo baseline
"""

from enum import Enum
from pathlib import Path
import subprocess

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici import features as F
from mibici import plots
from mibici.config import (
    EQUIPO,
    EXPERIMENT_NAME,
    FIGURES_DIR,
    INGESTA_VERSION,
    MANIFEST_PATH,
    MES_FIN,
    MES_INICIO,
    MLARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    P1_LAGS_HORAS,
    P2_HORAS_BLOQUE,
    P2_UMBRAL_CAPACIDAD,
    P3_K_BASELINE,
    P3_K_CANDIDATOS,
    P3_PCA_COMPONENTES,
    PROJ_ROOT,
    SEED,
    TRAIN_FIN,
    VAL_FIN,
)
from mibici.dataset import cargar_estaciones, cargar_viajes, version_datos

app = typer.Typer(add_completion=False)


class Modelo(str, Enum):
    baseline = "baseline"
    candidato = "candidato"
    ambos = "ambos"


# Utilidades de MLflow y partición


def git_info() -> dict[str, str]:
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(git("status", "--porcelain", "--untracked-files=no"))).lower(),
    }


def preparar_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        MLARTIFACTS_DIR.mkdir(exist_ok=True)
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=MLARTIFACTS_DIR.as_uri(),
            tags={"equipo": EQUIPO, "proyecto": "MiBici TC5061"},
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def params_comunes() -> dict:
    return {
        "mes_inicio": MES_INICIO,
        "mes_fin": MES_FIN,
        "train_fin": TRAIN_FIN,
        "val_fin": VAL_FIN,
        "data_version": version_datos(),
        "ingesta_version": INGESTA_VERSION,
        "seed": SEED,
    }


def tags_comunes(pregunta: str, tipo: str, descripcion: str) -> dict:
    return {
        "equipo": EQUIPO,
        "pregunta": pregunta,
        "tipo_modelo": tipo,
        **git_info(),
        "mlflow.note.content": descripcion,
    }


def particion(t: pd.Series) -> pd.Series:
    """Partición temporal: train hasta TRAIN_FIN, validación hasta VAL_FIN y el resto prueba."""
    return pd.Series(
        np.select([t <= TRAIN_FIN, t <= VAL_FIN], ["train", "val"], "test"), index=t.index
    )


def cargar_conteos() -> tuple[pd.DataFrame, pd.DataFrame]:
    estaciones = cargar_estaciones()
    viajes = cargar_viajes(["inicio", "fin", "origen_id", "destino_id"])
    return F.conteos_por_hora(viajes, estaciones), estaciones


def figuras_dir(run_name: str) -> Path:
    return FIGURES_DIR / run_name


def registrar_artefactos(paths: list[Path], carpeta: str = "figuras") -> None:
    for p in paths:
        mlflow.log_artifact(str(p), artifact_path=carpeta)
    mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="datos")


# Tipos internos de scikit-learn que skops (formato por defecto de MLflow 3) pide confirmar
TIPOS_CONFIABLES = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]


def registrar_modelo(modelo, ejemplo: pd.DataFrame) -> None:
    mlflow.sklearn.log_model(
        modelo, name="modelo", input_example=ejemplo, skops_trusted_types=TIPOS_CONFIABLES
    )


# P1: regresión de salidas por estación y hora

P1_FEATURES = [
    "hora_dia",
    "dia_semana",
    "es_fin_semana",
    "mes",
    "es_festivo",
    "capacidad",
    "lat",
    "lon",
    "municipio_cod",
    *[f"salidas_lag{k}" for k in P1_LAGS_HORAS],
    "llegadas_lag24",
    "llegadas_lag168",
    "salidas_media_4sem",
    "salidas_media24_desf24",
]


def metricas_regresion(y, pred, prefijo: str) -> dict:
    return {
        f"{prefijo}_mae": mean_absolute_error(y, pred),
        f"{prefijo}_rmse": root_mean_squared_error(y, pred),
        f"{prefijo}_wape": float(np.abs(y - pred).sum() / np.sum(y)),
        f"{prefijo}_r2": r2_score(y, pred),
    }


def wape_municipio(df: pd.DataFrame) -> float:
    """WAPE al agregar estaciones por municipio y hora (pronóstico por zona)."""
    z = df.groupby(["municipio", "hora"], observed=True)[["salidas", "pred"]].sum()
    return float((z["salidas"] - z["pred"]).abs().sum() / z["salidas"].sum())


def correr_p1(conteos, estaciones, modelo: Modelo) -> None:
    df = F.tabla_p1(conteos, estaciones)
    # Las dos primeras semanas de cada estación no tienen todos los rezagos
    df = df.dropna(subset=[f"salidas_lag{max(P1_LAGS_HORAS)}"]).reset_index(drop=True)
    df["split"] = particion(df["hora"])
    sets = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    base = params_comunes() | {
        "objetivo": "salidas por estacion y hora",
        "granularidad": "estacion-hora",
        "horizonte": "dia siguiente (1-24 h), datos hasta el cierre del dia anterior",
        "n_estaciones": df["estacion_id"].nunique(),
        **{f"n_{s}": len(d) for s, d in sets.items()},
    }

    def evaluar(pred_val, pred_test, run_name):
        metr = metricas_regresion(sets["val"]["salidas"], pred_val, "val")
        metr |= metricas_regresion(sets["test"]["salidas"], pred_test, "test")
        test = sets["test"].assign(pred=pred_test)
        metr["test_wape_municipio"] = wape_municipio(test)
        semana = test[test["hora"] < test["hora"].min() + pd.Timedelta(days=7)]
        top = semana.groupby("estacion_id")["salidas"].sum().nlargest(3).index.tolist()
        figs = [
            plots.serie_real_vs_pred(semana, top, figuras_dir(run_name) / "serie_test.png"),
            plots.error_por_hora(test, figuras_dir(run_name) / "mae_por_hora_test.png"),
        ]
        return metr, figs

    if modelo in (Modelo.baseline, Modelo.ambos):
        run_name = "p1-baseline-naive-estacional-7d"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                tags_comunes(
                    "P1",
                    "baseline",
                    "Pronóstico ingenuo estacional: salidas de la misma hora 7 días antes.",
                )
            )
            mlflow.log_params(base | {"modelo": "naive_estacional", "rezago_horas": 168})
            metr, figs = evaluar(
                sets["val"]["salidas_lag168"], sets["test"]["salidas_lag168"], run_name
            )
            mlflow.log_metrics(metr)
            registrar_artefactos(figs)
            logger.info(f"{run_name}: test_mae={metr['test_mae']:.3f}")

    if modelo in (Modelo.candidato, Modelo.ambos):
        run_name = "p1-hgb-poisson"
        hp = {
            "loss": "poisson",
            "learning_rate": 0.1,
            "max_iter": 300,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 100,
            "l2_regularization": 1.0,
        }
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                tags_comunes(
                    "P1",
                    "candidato",
                    "Gradient boosting con pérdida Poisson sobre rezagos y calendario.",
                )
            )
            mlflow.log_params(
                base
                | {"modelo": "HistGradientBoostingRegressor", "features": ",".join(P1_FEATURES)}
                | hp
            )
            reg = HistGradientBoostingRegressor(
                **hp, categorical_features=["municipio_cod"], random_state=SEED
            )
            reg.fit(sets["train"][P1_FEATURES], sets["train"]["salidas"])
            pred_val = reg.predict(sets["val"][P1_FEATURES])
            pred_test = reg.predict(sets["test"][P1_FEATURES])
            metr, figs = evaluar(pred_val, pred_test, run_name)
            muestra = sets["val"].sample(100_000, random_state=SEED)
            imp = permutation_importance(
                reg,
                muestra[P1_FEATURES],
                muestra["salidas"],
                scoring="neg_mean_absolute_error",
                n_repeats=3,
                random_state=SEED,
            )
            imp = pd.Series(imp.importances_mean, index=P1_FEATURES)
            figs.append(
                plots.importancia(
                    imp,
                    "P1: importancia por permutación (val)",
                    figuras_dir(run_name) / "importancia_val.png",
                )
            )
            mlflow.log_metrics(metr)
            registrar_artefactos(figs)
            registrar_modelo(reg, sets["val"][P1_FEATURES].head(5))
            logger.info(f"{run_name}: test_mae={metr['test_mae']:.3f}")


# P2: clasificación de riesgo por estación y bloque de 3 horas

P2_FEATURES = [
    "hora_dia",
    "dia_semana",
    "es_fin_semana",
    "mes",
    "es_festivo",
    "capacidad",
    "lat",
    "lon",
    "municipio_cod",
    "flujo_neto_lag8",
    "flujo_neto_lag56",
    "salidas_lag8",
    "salidas_lag56",
    "llegadas_lag8",
    "llegadas_lag56",
    "riesgo_lag8",
    "riesgo_lag56",
    "riesgo_media_4sem",
    "flujo_neto_media_4sem",
    "riesgo_tasa_hist",
]


def umbral_f1(y, score) -> float:
    """Umbral que maximiza F1 en validación; luego se aplica sin cambios en prueba."""
    prec, rec, umbrales = precision_recall_curve(y, score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(umbrales[np.argmax(f1[:-1])])


def metricas_clasificacion(y, score, umbral: float, prefijo: str) -> dict:
    pred = (score >= umbral).astype(int)
    return {
        f"{prefijo}_pr_auc": average_precision_score(y, score),
        f"{prefijo}_roc_auc": roc_auc_score(y, score),
        f"{prefijo}_f1": f1_score(y, pred),
        f"{prefijo}_precision": precision_score(y, pred, zero_division=0),
        f"{prefijo}_recall": recall_score(y, pred),
        f"{prefijo}_brier": brier_score_loss(y, score),
        f"{prefijo}_tasa_positivos": float(np.mean(y)),
    }


def correr_p2(conteos, estaciones, modelo: Modelo) -> None:
    df = F.tabla_p2(conteos, estaciones)
    # La primera semana de cada estación no tiene el rezago semanal
    df = df.dropna(subset=["riesgo_lag56"]).reset_index(drop=True)
    df["split"] = particion(df["bloque"])
    sets = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    base = params_comunes() | {
        "objetivo": "riesgo de vaciado o llenado",
        "granularidad": f"estacion-bloque de {P2_HORAS_BLOQUE} h",
        "horizonte": "dia siguiente (8 bloques), datos hasta el cierre del dia anterior",
        "umbral_capacidad": P2_UMBRAL_CAPACIDAD,
        "regla_etiqueta": "|llegadas - salidas| >= umbral_capacidad * capacidad en el bloque",
        "n_estaciones": df["estacion_id"].nunique(),
        **{f"n_{s}": len(d) for s, d in sets.items()},
    }

    def evaluar(score_val, score_test, run_name):
        u = umbral_f1(sets["val"]["riesgo"], score_val)
        metr = metricas_clasificacion(sets["val"]["riesgo"], score_val, u, "val")
        metr |= metricas_clasificacion(sets["test"]["riesgo"], score_test, u, "test")
        y = sets["test"]["riesgo"]
        pred = (score_test >= u).astype(int)
        d = figuras_dir(run_name)
        figs = [
            plots.curva_pr(
                y, score_test, "P2: precisión-exhaustividad (prueba)", d / "pr_test.png"
            ),
            plots.matriz_confusion(
                y, pred, "P2: matriz de confusión (prueba)", d / "confusion_test.png"
            ),
            plots.riesgo_por_bloque(sets["test"].assign(pred=pred), d / "riesgo_bloque_test.png"),
        ]
        return u, metr, figs

    if modelo in (Modelo.baseline, Modelo.ambos):
        run_name = "p2-baseline-frecuencia-4sem"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                tags_comunes(
                    "P2",
                    "baseline",
                    "Probabilidad = frecuencia de riesgo en el mismo bloque y día de las 4 "
                    "semanas previas.",
                )
            )
            u, metr, figs = evaluar(
                sets["val"]["riesgo_media_4sem"], sets["test"]["riesgo_media_4sem"], run_name
            )
            mlflow.log_params(base | {"modelo": "frecuencia_historica_4sem", "umbral_decision": u})
            mlflow.log_metrics(metr)
            registrar_artefactos(figs)
            logger.info(f"{run_name}: test_pr_auc={metr['test_pr_auc']:.3f}")

    if modelo in (Modelo.candidato, Modelo.ambos):
        run_name = "p2-hgb-clasificador"
        hp = {
            "learning_rate": 0.1,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 200,
            "l2_regularization": 1.0,
        }
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                tags_comunes(
                    "P2",
                    "candidato",
                    "Gradient boosting de clasificación sobre flujos rezagados y calendario.",
                )
            )
            clf = HistGradientBoostingClassifier(
                **hp, categorical_features=["municipio_cod"], random_state=SEED
            )
            clf.fit(sets["train"][P2_FEATURES], sets["train"]["riesgo"])
            score_val = clf.predict_proba(sets["val"][P2_FEATURES])[:, 1]
            score_test = clf.predict_proba(sets["test"][P2_FEATURES])[:, 1]
            u, metr, figs = evaluar(score_val, score_test, run_name)
            muestra = sets["val"].sample(100_000, random_state=SEED)
            imp = permutation_importance(
                clf,
                muestra[P2_FEATURES],
                muestra["riesgo"],
                scoring="average_precision",
                n_repeats=3,
                random_state=SEED,
            )
            imp = pd.Series(imp.importances_mean, index=P2_FEATURES)
            figs.append(
                plots.importancia(
                    imp,
                    "P2: importancia por permutación (val)",
                    figuras_dir(run_name) / "importancia_val.png",
                )
            )
            mlflow.log_params(
                base
                | {"modelo": "HistGradientBoostingClassifier", "features": ",".join(P2_FEATURES)}
                | hp
                | {"umbral_decision": u}
            )
            mlflow.log_metrics(metr)
            registrar_artefactos(figs)
            registrar_modelo(clf, sets["val"][P2_FEATURES].head(5))
            logger.info(f"{run_name}: test_pr_auc={metr['test_pr_auc']:.3f}")


# P3: clustering de estaciones por patrón de uso


def metricas_clustering(X, labels, prefijo: str) -> dict:
    return {
        f"{prefijo}_silhouette": silhouette_score(X, labels, random_state=SEED),
        f"{prefijo}_davies_bouldin": davies_bouldin_score(X, labels),
        f"{prefijo}_calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def correr_p3(conteos, estaciones, modelo: Modelo) -> None:
    inicio_test = pd.Timestamp(VAL_FIN) + pd.Timedelta(seconds=1)
    perfil = F.perfil_estaciones(conteos, MES_INICIO, TRAIN_FIN)
    perfil_test = F.perfil_estaciones(conteos, str(inicio_test), str(conteos["hora"].max()))
    comunes = perfil["estacion_id"].isin(perfil_test["estacion_id"])
    info = estaciones[["estacion_id", "nombre", "lat", "lon", "municipio"]]
    base = params_comunes() | {
        "objetivo": "grupos de estaciones por patron de uso",
        "granularidad": "estacion",
        "periodo_perfil": f"{MES_INICIO} a {TRAIN_FIN[:10]}",
        "n_estaciones": len(perfil),
    }

    def ajustar(cols, k, reducir):
        """K-means directo o tras estandarizar y reducir con PCA (las 48 proporciones horarias
        son ruidosas; los primeros componentes resumen el patrón de mañana y tarde)."""
        pasos = [StandardScaler(), PCA(P3_PCA_COMPONENTES, random_state=SEED)] if reducir else []
        pipe = make_pipeline(*pasos, KMeans(n_clusters=k, n_init=20, random_state=SEED))
        labels = pipe.fit_predict(perfil[cols])
        X = pipe[:-1].transform(perfil[cols]) if reducir else perfil[cols].to_numpy()
        return pipe, labels, X

    def registrar(pipe, labels, X, cols, run_name, titulo):
        metr = metricas_clustering(X, labels, "train")
        # Estabilidad: grupos del perfil de entrenamiento frente a los del periodo de prueba
        test_ids = perfil_test.set_index("estacion_id").loc[perfil.loc[comunes, "estacion_id"]]
        metr["test_ari_estabilidad"] = adjusted_rand_score(
            labels[comunes.to_numpy()], pipe.predict(test_ids[cols])
        )
        tam = np.bincount(labels)
        metr["train_tamano_min_grupo"] = int(tam.min())
        est = perfil[["estacion_id"]].assign(cluster=labels).merge(info, on="estacion_id")
        d = figuras_dir(run_name)
        figs = [
            plots.mapa_clusters(est, titulo, d / "mapa_grupos.png"),
            plots.mapa_clusters_html(est, d / "mapa_grupos.html"),
            plots.perfiles_clusters(perfil.assign(cluster=labels), d / "perfil_grupos.png"),
        ]
        tabla = d / "estaciones_por_grupo.csv"
        est.to_csv(tabla, index=False)
        figs.append(tabla)
        mlflow.log_metrics(metr)
        registrar_artefactos(figs)
        return metr

    if modelo in (Modelo.baseline, Modelo.ambos):
        run_name = f"p3-baseline-kmeans-k{P3_K_BASELINE}-perfil-salidas"
        cols = F.columnas_perfil_horario(perfil, "sal_lab")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                tags_comunes(
                    "P3",
                    "baseline",
                    "K-means con k fijo sobre el perfil horario de salidas en días hábiles.",
                )
            )
            mlflow.log_params(
                base
                | {"modelo": "KMeans", "n_clusters": P3_K_BASELINE, "n_init": 20}
                | {
                    "features": "perfil horario de salidas (24)",
                    "escalado": False,
                    "pca_componentes": 0,
                }
            )
            pipe, labels, X = ajustar(cols, P3_K_BASELINE, reducir=False)
            metr = registrar(
                pipe, labels, X, cols, run_name, f"P3: {P3_K_BASELINE} grupos (baseline)"
            )
            logger.info(f"{run_name}: train_silhouette={metr['train_silhouette']:.3f}")

    if modelo in (Modelo.candidato, Modelo.ambos):
        run_name = "p3-pca-kmeans-perfil-completo"
        cols = [c for c in perfil.columns if c != "estacion_id"]
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                tags_comunes(
                    "P3",
                    "candidato",
                    "Estandarización, PCA y k-means sobre perfiles de salidas y llegadas, fin "
                    "de semana, volumen y balance; k elegido por silueta.",
                )
            )
            siluetas = []
            for k in P3_K_CANDIDATOS:
                _, labels, X = ajustar(cols, k, reducir=True)
                s = silhouette_score(X, labels, random_state=SEED)
                siluetas.append(s)
                mlflow.log_metric("train_silhouette_por_k", s, step=k)
            k_opt = P3_K_CANDIDATOS[int(np.argmax(siluetas))]
            mlflow.log_params(
                base
                | {"modelo": "KMeans", "n_clusters": k_opt, "n_init": 20}
                | {"k_candidatos": ",".join(map(str, P3_K_CANDIDATOS)), "escalado": True}
                | {"pca_componentes": P3_PCA_COMPONENTES}
                | {"features": f"perfil completo ({len(cols)})"}
            )
            pipe, labels, X = ajustar(cols, k_opt, reducir=True)
            fig_k = plots.silueta_por_k(
                P3_K_CANDIDATOS, siluetas, figuras_dir(run_name) / "silueta_por_k.png"
            )
            metr = registrar(pipe, labels, X, cols, run_name, f"P3: {k_opt} grupos (candidato)")
            mlflow.log_artifact(str(fig_k), artifact_path="figuras")
            registrar_modelo(pipe, perfil[cols].head(5))
            logger.info(f"{run_name}: k={k_opt} train_silhouette={metr['train_silhouette']:.3f}")


# Comandos


@app.command()
def p1(modelo: Modelo = Modelo.ambos):
    """P1: viajes que iniciarán por estación y hora del día siguiente."""
    preparar_mlflow()
    correr_p1(*cargar_conteos(), modelo)


@app.command()
def p2(modelo: Modelo = Modelo.ambos):
    """P2: riesgo de vaciado o llenado por estación y bloque de 3 horas."""
    preparar_mlflow()
    correr_p2(*cargar_conteos(), modelo)


@app.command()
def p3(modelo: Modelo = Modelo.ambos):
    """P3: grupos de estaciones con patrones de uso similares."""
    preparar_mlflow()
    correr_p3(*cargar_conteos(), modelo)


@app.command()
def todos(modelo: Modelo = Modelo.ambos):
    """Ejecuta P1, P2 y P3."""
    preparar_mlflow()
    conteos, estaciones = cargar_conteos()
    for correr in (correr_p1, correr_p2, correr_p3):
        correr(conteos, estaciones, modelo)
    logger.success(f"Experimento '{EXPERIMENT_NAME}' actualizado en {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    app()
