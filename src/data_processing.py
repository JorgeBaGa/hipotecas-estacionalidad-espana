"""Carga y preparación de las series de tipos hipotecarios y Euribor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MESES_LARGOS = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def fecha_mes_es(fecha) -> str:
    """Formatea una fecha como 'mes de año' en español, p.ej. 'mayo de 2026'."""
    fecha = pd.Timestamp(fecha)
    return f"{MESES_LARGOS[fecha.month - 1]} de {fecha.year}"


def check_data_quality(processed_dir: Path = PROCESSED_DIR) -> None:
    """Lanza un error si la última validación de datos no pasó."""
    report_path = processed_dir / "data_quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError(f"La validación de datos no ha pasado: {report.get('errors')}")


def load_raw_tables(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga las tablas procesadas de hipotecas (formato largo) y Euribor."""
    hipotecas = pd.read_csv(processed_dir / "tipo_hipotecario_ine_long.csv", parse_dates=["fecha"])
    euribor = pd.read_csv(processed_dir / "euribor_12m_bde.csv", parse_dates=["fecha"])
    euribor = euribor.sort_values("fecha").reset_index(drop=True)
    return hipotecas, euribor


def mortgage_series(
    hipotecas: pd.DataFrame,
    naturaleza_finca: str = "Viviendas",
    tipo_interes: str = "Total",
) -> pd.DataFrame:
    """Filtra la tabla larga del INE a una serie mensual única con columnas de calendario."""
    serie = hipotecas[
        (hipotecas["naturaleza_finca"] == naturaleza_finca) & (hipotecas["tipo_interes"] == tipo_interes)
    ][["fecha", "tipo_hipotecario"]].sort_values("fecha").reset_index(drop=True)
    serie["anio"] = serie["fecha"].dt.year
    serie["mes_num"] = serie["fecha"].dt.month
    serie["mes"] = pd.Categorical([MESES[m - 1] for m in serie["mes_num"]], categories=MESES, ordered=True)
    return serie
