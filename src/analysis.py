"""Análisis de estacionalidad, estabilidad temporal y relación con el Euribor.

Incluye un modelo de regresión con errores autocorrelados (AR(2), ajustado
por mínimos cuadrados generalizados iterativos) para contrastar el efecto de
cada mes controlando por el nivel del Euribor y una tendencia temporal, y una
partición óptima por mínimos cuadrados (con selección del número de tramos
por BIC) para detectar cambios de tendencia en el Euribor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import GLSAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

from data_processing import MESES

EURIBOR_LAGS = range(0, 7)  # euribor_l0 .. euribor_l6


def monthly_deviation_from_annual_mean(viviendas: pd.DataFrame) -> pd.DataFrame:
    """Diferencia de cada mes frente a la media de su propio año, con IC 95%.

    Compara cada mes con la media de su año para no dejar que los años con
    tipos muy altos o muy bajos distorsionen la comparación entre meses.
    Solo usa años completos (12 observaciones).
    """
    anios_completos = viviendas.groupby("anio").size()
    anios_completos = anios_completos[anios_completos == 12].index

    datos = viviendas[viviendas["anio"].isin(anios_completos)].copy()
    datos["diferencia_anual"] = datos.groupby("anio")["tipo_hipotecario"].transform(
        lambda x: x - x.mean()
    )

    resumen = (
        datos.groupby(["mes_num", "mes"], observed=True)["diferencia_anual"]
        .agg(n_anios="count", diferencia_media="mean", desviacion="std")
        .reset_index()
    )
    resumen["error"] = resumen["desviacion"] / np.sqrt(resumen["n_anios"])
    resumen["limite"] = stats.t.ppf(0.975, resumen["n_anios"] - 1) * resumen["error"]
    resumen["inferior"] = resumen["diferencia_media"] - resumen["limite"]
    resumen["superior"] = resumen["diferencia_media"] + resumen["limite"]
    return resumen.sort_values("mes_num").reset_index(drop=True)


def _build_month_effect_design(serie: pd.DataFrame, euribor: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de diseño: índice temporal, lags 0-6 de Euribor y
    variables de mes con codificación de suma a cero: 11 columnas para
    Ene..Nov, y Dic queda implícito como -1 en todas ellas, de modo que su
    efecto se recupera como -suma del resto y los doce efectos estimados
    suman cero.
    """
    datos = serie.sort_values("fecha").merge(euribor[["fecha", "euribor_12m"]], on="fecha", how="left")
    datos = datos.reset_index(drop=True)
    datos["indice"] = np.arange(1, len(datos) + 1)
    datos["mes_num"] = datos["fecha"].dt.month
    for k in EURIBOR_LAGS:
        datos[f"euribor_l{k}"] = datos["euribor_12m"].shift(k)

    lag_cols = [f"euribor_l{k}" for k in EURIBOR_LAGS]
    datos = datos.dropna(subset=["tipo_hipotecario"] + lag_cols).reset_index(drop=True)

    for m in range(1, 12):
        datos[f"mes{m}"] = np.where(
            datos["mes_num"] == m, 1, np.where(datos["mes_num"] == 12, -1, 0)
        )
    return datos


def glsar_month_effect_test(serie: pd.DataFrame, euribor: pd.DataFrame, ar_order: int = 2):
    """Contraste conjunto del efecto mes, controlando por el nivel del Euribor
    (lags 0-6) y una tendencia lineal, con errores AR(ar_order).

    Se estima un AR(ar_order) sobre los residuos (``statsmodels.GLSAR``) y se
    compara, mediante un test F sobre los datos ya "blanqueados" por ese
    AR(ar_order), el modelo con variables de mes frente al modelo sin ellas.

    Devuelve (tabla_efectos, p_valor_conjunto, p_valor_ljung_box, n_obs).
    """
    datos = _build_month_effect_design(serie, euribor)
    lag_cols = [f"euribor_l{k}" for k in EURIBOR_LAGS]
    base_cols = ["indice"] + lag_cols
    month_cols = [f"mes{m}" for m in range(1, 12)]

    y = datos["tipo_hipotecario"].to_numpy()
    x_full = sm.add_constant(datos[base_cols + month_cols]).to_numpy()
    x_reduced = sm.add_constant(datos[base_cols]).to_numpy()

    # Se estima rho (AR(2)) a partir del modelo completo y se reutiliza en
    # ambos modelos para que el test F compare sobre la misma transformación.
    model_full_iter = GLSAR(y, x_full, rho=ar_order)
    res_full_iter = model_full_iter.iterative_fit(maxiter=50)
    rho = model_full_iter.rho

    res_full = GLSAR(y, x_full, rho=rho).fit()
    res_reduced = GLSAR(y, x_reduced, rho=rho).fit()

    df_num = res_reduced.df_resid - res_full.df_resid
    f_stat = ((res_reduced.ssr - res_full.ssr) / df_num) / (res_full.ssr / res_full.df_resid)
    p_global = float(stats.f.sf(f_stat, df_num, res_full.df_resid))

    # Ljung-Box sobre los residuos ya blanqueados (whitened) por el AR(2),
    # no sobre los residuos en la escala original (que seguirían muy
    # autocorrelacionados y darían un resultado engañoso).
    wresid = model_full_iter.wendog - model_full_iter.wexog @ res_full_iter.params
    lb = acorr_ljungbox(wresid, lags=[24], model_df=ar_order, return_df=True)
    p_ljung_box = float(lb["lb_pvalue"].iloc[0])

    n_base = len(base_cols)
    month_param_idx = list(range(1 + n_base, 1 + n_base + 11))
    coefs = res_full.params[month_param_idx]
    cov_sub = res_full.cov_params()[np.ix_(month_param_idx, month_param_idx)]

    efecto = list(coefs) + [-float(np.sum(coefs))]
    error = list(np.sqrt(np.diag(cov_sub))) + [float(np.sqrt(cov_sub.sum()))]
    tabla = pd.DataFrame({"mes": MESES, "efecto": efecto, "error": error})
    tabla["inferior"] = tabla["efecto"] - 1.96 * tabla["error"]
    tabla["superior"] = tabla["efecto"] + 1.96 * tabla["error"]

    return tabla, p_global, p_ljung_box, len(datos)


def holm_corrected_month_table(tabla_efectos: pd.DataFrame) -> pd.DataFrame:
    """Añade p-valores individuales y ajustados (Holm) a la tabla de efectos mensuales."""
    tabla = tabla_efectos.copy()
    tabla["p_valor"] = 2 * stats.norm.sf(np.abs(tabla["efecto"] / tabla["error"]))
    _, p_adj, _, _ = multipletests(tabla["p_valor"], method="holm")
    tabla["p_ajustado"] = p_adj
    tabla["resultado"] = np.select(
        [
            (tabla["efecto"] < 0) & (tabla["p_ajustado"] < 0.05),
            (tabla["efecto"] > 0) & (tabla["p_ajustado"] < 0.05),
        ],
        ["Tipo inferior", "Tipo superior"],
        default="Sin diferencia fiable",
    )
    return tabla


def rolling_window_test(
    hipotecas: pd.DataFrame,
    euribor: pd.DataFrame,
    naturaleza_finca: str = "Viviendas",
    tipo_interes: str = "Total",
    window_years: int = 8,
) -> pd.DataFrame:
    """Repite el contraste conjunto de mes en ventanas móviles de N años, para
    comprobar si el resultado depende de la fecha de corte elegida.
    """
    serie = hipotecas[
        (hipotecas["naturaleza_finca"] == naturaleza_finca) & (hipotecas["tipo_interes"] == tipo_interes)
    ][["fecha", "tipo_hipotecario"]].copy()
    serie["anio"] = serie["fecha"].dt.year

    primer_anio = int(serie["anio"].min())
    ultimo_anio = int(serie["anio"].max())
    filas = []
    for inicio in range(primer_anio, ultimo_anio - (window_years - 1)):
        fin = inicio + (window_years - 1)
        ventana = serie[(serie["anio"] >= inicio) & (serie["anio"] <= fin)][["fecha", "tipo_hipotecario"]]
        _, p_valor, _, _ = glsar_month_effect_test(ventana, euribor)
        filas.append({"inicio": inicio, "fin": fin, "p_valor": p_valor})
    return pd.DataFrame(filas)


def fixed_vs_variable_test(hipotecas: pd.DataFrame, euribor: pd.DataFrame) -> pd.DataFrame:
    """Repite el contraste conjunto de mes por separado para Total, Fijo y Variable."""
    filas = []
    for tipo in ["Total", "Fijo", "Variable"]:
        serie = hipotecas[
            (hipotecas["naturaleza_finca"] == "Viviendas") & (hipotecas["tipo_interes"] == tipo)
        ][["fecha", "tipo_hipotecario"]]
        _, p_valor, _, _ = glsar_month_effect_test(serie, euribor)
        filas.append({"Tipo": tipo, "p_valor": round(p_valor, 3)})
    tabla = pd.DataFrame(filas)
    tabla["Diferencias mensuales"] = np.where(tabla["p_valor"] < 0.05, "Sí", "No")
    return tabla


def _segment_ssr(y: np.ndarray, x: np.ndarray, i: int, j: int) -> float:
    """SSR de una regresión lineal simple y ~ 1 + x en el tramo [i, j)."""
    yy, xx = y[i:j], x[i:j]
    design = np.column_stack([np.ones(len(xx)), xx])
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    resid = yy - design @ beta
    return float(np.sum(resid**2))


def detect_breakpoints(
    serie: pd.DataFrame,
    value_col: str,
    max_breaks: int = 5,
    min_segment_length: int = 24,
) -> tuple[list[int], pd.DataFrame]:
    """Divide una serie en tramos de tendencia lineal homogénea.

    Para cada número de rupturas de 0 a ``max_breaks`` calcula, por
    programación dinámica, la partición que minimiza la suma de cuadrados de
    los residuos (partición óptima de Bai-Perron); después elige el número
    de rupturas que minimiza el BIC.

    Devuelve los índices (posición en la serie) de las rupturas elegidas y
    una tabla resumen por etapa.
    """
    y = serie[value_col].to_numpy()
    x = np.arange(1, len(y) + 1, dtype=float)
    n = len(y)
    h = min_segment_length
    max_segments = max_breaks + 1

    cost: dict[tuple[int, int], float] = {}
    for i in range(0, n):
        for j in range(i + h, n + 1):
            cost[(i, j)] = _segment_ssr(y, x, i, j)

    dp: list[dict[int, tuple[float, list[int]]]] = [dict() for _ in range(max_segments + 1)]
    for j in range(h, n + 1):
        if (0, j) in cost:
            dp[1][j] = (cost[(0, j)], [])

    for m in range(2, max_segments + 1):
        for j in range(m * h, n + 1):
            best = None
            for t in range((m - 1) * h, j - h + 1):
                if t in dp[m - 1] and (t, j) in cost:
                    val = dp[m - 1][t][0] + cost[(t, j)]
                    if best is None or val < best[0]:
                        best = (val, dp[m - 1][t][1] + [t])
            if best is not None:
                dp[m][j] = best

    bic_by_m = {}
    for m in range(1, max_segments + 1):
        if n in dp[m]:
            ssr, cuts = dp[m][n]
            k = 2 * m  # intercepto + pendiente por tramo
            bic = n * np.log(ssr / n) + k * np.log(n)
            bic_by_m[m] = (ssr, bic, cuts)

    best_m = min(bic_by_m, key=lambda m: bic_by_m[m][1])
    _, _, cuts = bic_by_m[best_m]

    limites = [0] + cuts + [n]
    filas = []
    for etapa, (ini, fin) in enumerate(zip(limites[:-1], limites[1:]), start=1):
        tramo = serie.iloc[ini:fin]
        filas.append(
            {
                "Etapa": etapa,
                "Inicio": tramo["fecha"].iloc[0],
                "Fin": tramo["fecha"].iloc[-1],
                f"{value_col}_inicial": round(float(tramo[value_col].iloc[0]), 2),
                f"{value_col}_final": round(float(tramo[value_col].iloc[-1]), 2),
                "Cambio": round(float(tramo[value_col].iloc[-1] - tramo[value_col].iloc[0]), 2),
            }
        )
    etapas = pd.DataFrame(filas)
    return cuts, etapas


def assign_stage(fecha: pd.Series, etapas: pd.DataFrame) -> pd.Series:
    """Asigna a cada fecha el número de etapa (1..N) según las fechas de inicio de `etapas`."""
    inicios = etapas["Inicio"].to_numpy()
    idx = np.searchsorted(inicios, fecha.to_numpy(), side="right")
    return pd.Series(idx, index=fecha.index)


def stage_relationship(
    serie: pd.DataFrame,
    euribor: pd.DataFrame,
    etapas: pd.DataFrame,
    value_col: str = "tipo_hipotecario",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Correlación y sensibilidad entre el cambio mensual del Euribor y el
    cambio mensual de `value_col`, calculadas por separado dentro de cada
    etapa (y, opcionalmente, por grupo adicional como el producto fijo/variable).
    """
    datos = serie.merge(euribor[["fecha", "euribor_12m"]], on="fecha", how="inner")
    datos["etapa"] = assign_stage(datos["fecha"], etapas)

    group_cols = group_cols or []
    sort_cols = group_cols + ["fecha"]
    datos = datos.sort_values(sort_cols)
    grp = datos.groupby(group_cols + ["etapa"]) if group_cols else datos.groupby("etapa")
    datos["cambio_valor"] = grp[value_col].diff()
    datos["cambio_euribor"] = grp["euribor_12m"].diff()

    datos = datos.dropna(subset=["cambio_valor", "cambio_euribor"])
    resumen_group_cols = group_cols + ["etapa"]
    resumen = datos.groupby(resumen_group_cols).apply(
        lambda g: pd.Series(
            {
                "Observaciones": len(g),
                "Correlacion": g["cambio_euribor"].corr(g["cambio_valor"]),
                "Sensibilidad": g["cambio_euribor"].cov(g["cambio_valor"]) / g["cambio_euribor"].var(),
            }
        ),
        include_groups=False,
    ).reset_index()
    resumen["Correlacion"] = resumen["Correlacion"].round(2)
    resumen["Sensibilidad"] = resumen["Sensibilidad"].round(2)
    return resumen
