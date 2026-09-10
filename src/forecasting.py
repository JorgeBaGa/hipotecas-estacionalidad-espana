"""Previsión del tipo hipotecario: modelos, backtesting y previsión vigente.

Cuatro modelos (valor reciente, ETS, ARIMA, ARIMAX con Euribor), validación
temporal con orígenes móviles y selección del modelo con menor error fuera
de muestra por horizonte.

Los modelos ETS y ARIMA se seleccionan automáticamente (por AICc) dentro de
una búsqueda acotada sobre sus especificaciones habituales (tipo de error y
tendencia para ETS; orden (p, d, q) para ARIMA).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

Z80 = stats.norm.ppf(0.9)
Z95 = stats.norm.ppf(0.975)

MODEL_LABELS = {
    "naive": "Valor reciente",
    "ets": "ETS sin estacionalidad",
    "arima": "ARIMA",
    "arimax_euribor": "ARIMAX con Euribor",
}


def add_months(fecha: pd.Timestamp, months: int) -> pd.Timestamp:
    return (pd.Timestamp(fecha) + pd.DateOffset(months=months)).replace(day=1)


def _as_monthly_series(fechas: pd.Series, valores: pd.Series) -> pd.Series:
    serie = pd.Series(valores.to_numpy(), index=pd.DatetimeIndex(fechas.to_numpy()))
    serie.index.freq = "MS"
    return serie


def _forecast_frame(origin_date, model, mean, lower_80, upper_80, lower_95, upper_95) -> pd.DataFrame:
    horizon = len(mean)
    return pd.DataFrame(
        {
            "origin_date": origin_date,
            "horizon_months": np.arange(1, horizon + 1),
            "target_date": [add_months(origin_date, h) for h in range(1, horizon + 1)],
            "model": model,
            "forecast": np.asarray(mean),
            "lower_80": np.asarray(lower_80),
            "upper_80": np.asarray(upper_80),
            "lower_95": np.asarray(lower_95),
            "upper_95": np.asarray(upper_95),
        }
    )


def forecast_naive(mortgage_train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Mantiene el último valor observado (paseo aleatorio sin deriva)."""
    y = mortgage_train["tipo_hipotecario"].to_numpy()
    origin_date = mortgage_train["fecha"].iloc[-1]
    last_value = y[-1]
    sigma = np.std(np.diff(y), ddof=1)
    h = np.arange(1, horizon + 1)
    se = sigma * np.sqrt(h)
    mean = np.repeat(last_value, horizon)
    return _forecast_frame(
        origin_date, "naive", mean, mean - Z80 * se, mean + Z80 * se, mean - Z95 * se, mean + Z95 * se
    )


def _fit_best_ets(y: pd.Series):
    """Búsqueda acotada de un modelo de suavizado exponencial (tipo de error
    x tipo de tendencia, sin componente estacional) por AICc."""
    best = None
    for error in ["add", "mul"]:
        for trend, damped in [(None, False), ("add", False), ("add", True)]:
            try:
                model = ETSModel(y, error=error, trend=trend, damped_trend=damped, seasonal=None)
                res = model.fit(disp=False)
                k = res.params.shape[0]
                n = len(y)
                aicc = res.aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
                if best is None or aicc < best[0]:
                    best = (aicc, res)
            except Exception:
                continue
    if best is None:
        raise RuntimeError("No se pudo ajustar ningun modelo ETS.")
    return best[1]


def forecast_ets(mortgage_train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    origin_date = mortgage_train["fecha"].iloc[-1]
    y = _as_monthly_series(mortgage_train["fecha"], mortgage_train["tipo_hipotecario"])
    res = _fit_best_ets(y)
    pred = res.get_prediction(start=len(y), end=len(y) + horizon - 1)
    sf80 = pred.summary_frame(alpha=0.2)
    sf95 = pred.summary_frame(alpha=0.05)
    return _forecast_frame(
        origin_date, "ets", sf80["mean"], sf80["pi_lower"], sf80["pi_upper"], sf95["pi_lower"], sf95["pi_upper"]
    )


def _auto_arima(y, exog=None, max_p=2, max_d=1, max_q=2):
    """Búsqueda acotada de orden (p, d, q) por AICc, sin componente estacional."""
    best = None
    for d in range(0, max_d + 1):
        trend = "c" if d == 0 else "t"
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                if p == 0 and q == 0 and d == 0:
                    continue
                try:
                    model = SARIMAX(
                        y, exog=exog, order=(p, d, q), trend=trend,
                        enforce_stationarity=False, enforce_invertibility=False,
                    )
                    res = model.fit(disp=False, maxiter=200, method="lbfgs")
                    if not res.mle_retvals.get("converged", True):
                        continue
                    if best is None or res.aicc < best[0]:
                        best = (res.aicc, (p, d, q), res)
                except Exception:
                    continue
    if best is None:
        raise RuntimeError("No se pudo ajustar ningun modelo ARIMA.")
    return best[2]


def forecast_arima(mortgage_train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    origin_date = mortgage_train["fecha"].iloc[-1]
    y = _as_monthly_series(mortgage_train["fecha"], mortgage_train["tipo_hipotecario"])
    res = _auto_arima(y)
    fc = res.get_forecast(steps=horizon)
    sf80 = fc.summary_frame(alpha=0.2)
    sf95 = fc.summary_frame(alpha=0.05)
    return _forecast_frame(
        origin_date, "arima", sf80["mean"], sf80["mean_ci_lower"], sf80["mean_ci_upper"],
        sf95["mean_ci_lower"], sf95["mean_ci_upper"],
    )


def future_euribor_path(
    euribor: pd.DataFrame,
    origin_date: pd.Timestamp,
    horizon: int,
    publication_lead_months: int = 2,
) -> pd.Series:
    """Euribor disponible para el ARIMAX en cada mes objetivo del horizonte.

    Reproduce el desfase real de publicación: en la fecha en que se emite la
    previsión (``origin_date``), el Euribor solo está confirmado hasta
    ``origin_date + publication_lead_months`` (igual que ocurre con el dato
    hipotecario del INE, que se publica con retraso). Para los meses del
    horizonte más allá de ese límite, el Euribor se proyecta con un ARIMA
    auxiliar en vez de usar el valor real futuro, para no introducir
    información que no habría estado disponible en la fecha de origen.
    """
    available_cutoff = min(add_months(origin_date, publication_lead_months), euribor["fecha"].max())
    available = euribor[euribor["fecha"] <= available_cutoff].sort_values("fecha")

    target_dates = [add_months(origin_date, h) for h in range(1, horizon + 1)]
    known = available.set_index("fecha")["euribor_12m"]
    path = pd.Series(index=target_dates, dtype=float)
    for date in target_dates:
        if date in known.index:
            path.loc[date] = known.loc[date]

    missing_dates = [d for d in target_dates if pd.isna(path.loc[d])]
    if missing_dates:
        last_available = available["fecha"].max()
        months_needed = (
            (missing_dates[-1].year - last_available.year) * 12
            + (missing_dates[-1].month - last_available.month)
        )
        y_euribor = _as_monthly_series(available["fecha"], available["euribor_12m"])
        res = _auto_arima(y_euribor)
        projected = res.get_forecast(steps=months_needed).summary_frame(alpha=0.2)["mean"]
        projected.index = [add_months(last_available, h) for h in range(1, months_needed + 1)]
        for date in missing_dates:
            path.loc[date] = projected.loc[date]

    assert path.notna().all(), "El calendario de Euribor futuro tiene huecos."
    return path


def forecast_arimax(mortgage_train: pd.DataFrame, euribor: pd.DataFrame, horizon: int, publication_lead_months: int = 2) -> pd.DataFrame:
    origin_date = mortgage_train["fecha"].iloc[-1]
    training = mortgage_train.merge(euribor[["fecha", "euribor_12m"]], on="fecha", how="inner").sort_values("fecha")
    y = _as_monthly_series(training["fecha"], training["tipo_hipotecario"])
    exog_train = training["euribor_12m"].to_numpy().reshape(-1, 1)

    future_euribor = future_euribor_path(euribor, origin_date, horizon, publication_lead_months)
    exog_future = future_euribor.to_numpy().reshape(-1, 1)

    res = _auto_arima(y, exog=exog_train)
    fc = res.get_forecast(steps=horizon, exog=exog_future)
    sf80 = fc.summary_frame(alpha=0.2)
    sf95 = fc.summary_frame(alpha=0.05)
    return _forecast_frame(
        origin_date, "arimax_euribor", sf80["mean"], sf80["mean_ci_lower"], sf80["mean_ci_upper"],
        sf95["mean_ci_lower"], sf95["mean_ci_upper"],
    )


def forecast_model(model: str, mortgage_train: pd.DataFrame, euribor: pd.DataFrame, horizon: int, publication_lead_months: int = 2) -> pd.DataFrame:
    if model == "naive":
        return forecast_naive(mortgage_train, horizon)
    if model == "ets":
        return forecast_ets(mortgage_train, horizon)
    if model == "arima":
        return forecast_arima(mortgage_train, horizon)
    if model == "arimax_euribor":
        return forecast_arimax(mortgage_train, euribor, horizon, publication_lead_months)
    raise ValueError(f"Modelo desconocido: {model}")


def run_rolling_backtest(
    mortgage: pd.DataFrame,
    euribor: pd.DataFrame,
    models: list[str] = ("naive", "ets", "arima", "arimax_euribor"),
    horizons: list[int] = (1, 3, 6),
    origins: int = 60,
    minimum_training: int = 120,
    publication_lead_months: int = 2,
    verbose: bool = False,
) -> pd.DataFrame:
    """Validación temporal con orígenes móviles: en cada fecha histórica, el
    modelo solo utiliza datos que habrían estado disponibles en ese momento.
    """
    max_horizon = max(horizons)
    last_origin = len(mortgage) - max_horizon
    first_origin = max(minimum_training, last_origin - origins + 1)

    observed = mortgage.set_index("fecha")["tipo_hipotecario"]
    rows = []
    for origin_index in range(first_origin, last_origin + 1):
        training = mortgage.iloc[:origin_index]
        if verbose:
            print(f"origen {origin_index}/{last_origin} ({training['fecha'].iloc[-1].date()})")
        for model in models:
            try:
                predicted = forecast_model(model, training, euribor, max_horizon, publication_lead_months)
            except Exception as exc:
                if verbose:
                    print(f"  {model} fallo: {exc}")
                continue
            predicted = predicted[predicted["horizon_months"].isin(horizons)].copy()
            predicted["observed"] = predicted["target_date"].map(observed)
            rows.append(predicted)

    backtest = pd.concat(rows, ignore_index=True)
    backtest = backtest.dropna(subset=["observed"]).copy()
    backtest["error"] = backtest["observed"] - backtest["forecast"]
    backtest["abs_error"] = backtest["error"].abs()
    backtest["squared_error"] = backtest["error"] ** 2
    backtest["covered_80"] = (backtest["observed"] >= backtest["lower_80"]) & (backtest["observed"] <= backtest["upper_80"])
    backtest["covered_95"] = (backtest["observed"] >= backtest["lower_95"]) & (backtest["observed"] <= backtest["upper_95"])
    return backtest


def summarise_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    evaluation = (
        backtest.groupby(["model", "horizon_months"])
        .agg(
            observations=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
            coverage_80=("covered_80", "mean"),
            coverage_95=("covered_95", "mean"),
        )
        .reset_index()
    )
    evaluation = evaluation.sort_values(["horizon_months", "mae"])
    evaluation["selected"] = evaluation.groupby("horizon_months").cumcount() == 0
    return evaluation.reset_index(drop=True)


def build_current_forecasts(
    mortgage: pd.DataFrame,
    euribor: pd.DataFrame,
    evaluation: pd.DataFrame,
    models: list[str] = ("naive", "ets", "arima", "arimax_euribor"),
    horizons: list[int] = (1, 3, 6),
    publication_lead_months: int = 2,
) -> pd.DataFrame:
    all_forecasts = pd.concat(
        [forecast_model(model, mortgage, euribor, max(horizons), publication_lead_months) for model in models],
        ignore_index=True,
    )
    all_forecasts = all_forecasts[all_forecasts["horizon_months"].isin(horizons)].copy()

    selected_models = evaluation[evaluation["selected"]][["horizon_months", "model"]].rename(
        columns={"model": "selected_model"}
    )
    all_forecasts = all_forecasts.merge(selected_models, on="horizon_months", how="left")
    all_forecasts["selected"] = all_forecasts["model"] == all_forecasts["selected_model"]
    return all_forecasts.drop(columns="selected_model")


def update_forecast_history(current_forecasts: pd.DataFrame, mortgage: pd.DataFrame, history_path) -> pd.DataFrame:
    """Añade la previsión vigente al histórico sin sobrescribir las anteriores.

    Cuando ya hay datos observados para una previsión antigua, recalcula su
    error; nunca sustituye el valor originalmente pronosticado.
    """
    history_columns = [
        "issue_date", "origin_date", "target_date", "horizon_months", "model",
        "forecast", "lower_80", "upper_80", "lower_95", "upper_95",
        "observed", "error", "abs_error",
    ]
    observed_values = mortgage.set_index("fecha")["tipo_hipotecario"]

    if history_path.exists():
        history = pd.read_csv(history_path, parse_dates=["issue_date", "origin_date", "target_date"])
        history = history.drop(columns=["observed", "error", "abs_error"], errors="ignore")
        history["observed"] = history["target_date"].map(observed_values)
        history["error"] = history["observed"] - history["forecast"]
        history["abs_error"] = history["error"].abs()
    else:
        history = pd.DataFrame(columns=history_columns)

    new_entries = current_forecasts[current_forecasts["selected"]].copy()
    new_entries["issue_date"] = pd.Timestamp.today().normalize()
    new_entries["observed"] = new_entries["target_date"].map(observed_values)
    new_entries["error"] = new_entries["observed"] - new_entries["forecast"]
    new_entries["abs_error"] = new_entries["error"].abs()
    new_entries = new_entries[history_columns]

    history = pd.concat([history, new_entries], ignore_index=True)
    history = history.sort_values(["origin_date", "horizon_months", "issue_date"])
    history = history.drop_duplicates(subset=["origin_date", "horizon_months", "model"], keep="last")
    return history[history_columns].reset_index(drop=True)


@dataclass
class ForecastRun:
    backtest: pd.DataFrame
    evaluation: pd.DataFrame
    current_forecasts: pd.DataFrame
    history: pd.DataFrame
    metadata: dict


def run_full_forecast_pipeline(
    mortgage: pd.DataFrame,
    euribor: pd.DataFrame,
    history_path,
    horizons: list[int] = (1, 3, 6),
    origins: int = 60,
    minimum_training: int = 120,
    publication_lead_months: int = 2,
    verbose: bool = False,
) -> ForecastRun:
    backtest = run_rolling_backtest(
        mortgage, euribor, horizons=horizons, origins=origins,
        minimum_training=minimum_training, publication_lead_months=publication_lead_months, verbose=verbose,
    )
    evaluation = summarise_backtest(backtest)
    current_forecasts = build_current_forecasts(
        mortgage, euribor, evaluation, horizons=horizons, publication_lead_months=publication_lead_months
    )
    history = update_forecast_history(current_forecasts, mortgage, history_path)

    metadata = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "mortgage_latest": str(mortgage["fecha"].max().date()),
        "euribor_latest": str(euribor["fecha"].max().date()),
        "publication_lead_months": publication_lead_months,
        "backtest_first_origin": str(backtest["origin_date"].min().date()),
        "backtest_last_origin": str(backtest["origin_date"].max().date()),
        "backtest_origins": int(backtest["origin_date"].nunique()),
        "horizons": list(horizons),
        "selected_models": evaluation[evaluation["selected"]][["horizon_months", "model"]].to_dict("records"),
    }
    return ForecastRun(backtest, evaluation, current_forecasts, history, metadata)
