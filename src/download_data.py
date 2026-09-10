"""Descarga y limpieza de las series oficiales de tipos hipotecarios y Euribor.

Fuentes:
- INE, tabla 24457: tipo de interés medio al inicio de las hipotecas constituidas.
- Banco de España: serie D_1NBAF472 (Euribor a 12 meses).
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

INE_MORTGAGE_RATE_URL = "https://www.ine.es/jaxiT3/files/t/csv_bdsc/24457.csv"
BDE_MONTHLY_INTEREST_CSV_URL = (
    "https://www.bde.es/webbe/es/estadisticas/compartido/datos/csv/cal_umv_mfi001.csv"
)
BDE_LEGACY_BE1901_CSV_URL = (
    "https://www.bde.es/webbe/es/estadisticas/compartido/datos/csv/be1901.csv"
)
BDE_EURIBOR_SERIES_CODE = "D_1NBAF472"

MONTH_MAP = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DIC": 12,
}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers={"User-Agent": "hipotecas-estacionalidad/1.0"}, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def normalize_text(value: str) -> str:
    value = str(value).replace("﻿", "").replace("ï»¿", "")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(value.lower().strip().split())


def find_column(normalized_columns: dict[str, str], expected: str) -> str:
    try:
        return normalized_columns[expected]
    except KeyError as exc:
        available = sorted(normalized_columns.values())
        raise ValueError(f"No se encontro la columna esperada '{expected}'. Columnas: {available}") from exc


def parse_ine_mortgage_rates(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="latin1")
    normalized_columns = {normalize_text(col): col for col in df.columns}
    tipo_col = find_column(normalized_columns, "tipo de interes")
    naturaleza_col = find_column(normalized_columns, "naturaleza de la finca")
    periodo_col = find_column(normalized_columns, "periodo")
    total_col = find_column(normalized_columns, "total")

    out = df.rename(
        columns={
            tipo_col: "tipo_interes",
            naturaleza_col: "naturaleza_finca",
            periodo_col: "periodo",
            total_col: "tipo_hipotecario",
        }
    ).copy()
    out["tipo_hipotecario"] = (
        out["tipo_hipotecario"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )
    out["fecha"] = pd.to_datetime(out["periodo"].str.replace("M", "-", regex=False))
    out = out[
        ["fecha", "periodo", "naturaleza_finca", "tipo_interes", "tipo_hipotecario"]
    ].sort_values(["fecha", "naturaleza_finca", "tipo_interes"])
    return out.reset_index(drop=True)


def parse_bde_wide_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        sep=",",
        encoding="latin1",
        skiprows=[1, 2, 3, 4, 5],
        header=0,
        quotechar='"',
    )
    raw.rename(columns={raw.columns[0]: "fecha_raw"}, inplace=True)
    if BDE_EURIBOR_SERIES_CODE not in raw.columns:
        raise ValueError(f"No se encontro {BDE_EURIBOR_SERIES_CODE} en {path.name}")

    out = raw[["fecha_raw", BDE_EURIBOR_SERIES_CODE]].copy()
    out.columns = ["fecha_raw", "euribor_12m"]
    out = out[out["fecha_raw"].astype(str).str.strip().str[:3].isin(MONTH_MAP)].copy()
    out["euribor_12m"] = pd.to_numeric(out["euribor_12m"], errors="coerce")
    out = out.dropna(subset=["euribor_12m"])
    out["month_token"] = out["fecha_raw"].str.strip().str[:3]
    out["year"] = out["fecha_raw"].str.strip().str[-4:].astype(int)
    out["month"] = out["month_token"].map(MONTH_MAP)
    out["fecha"] = pd.to_datetime(
        out[["year", "month"]].assign(day=1).rename(columns={"year": "year", "month": "month"})
    )
    return out[["fecha", "euribor_12m"]].sort_values("fecha").reset_index(drop=True)


def main(download: bool = True) -> dict[str, str]:
    ensure_dirs()
    ine_raw = RAW_DIR / "tipo_hipotecario_ine_24457.csv"
    bde_raw = RAW_DIR / "euribor_12m_bde.csv"

    if download:
        save_download(INE_MORTGAGE_RATE_URL, ine_raw)
        try:
            save_download(BDE_MONTHLY_INTEREST_CSV_URL, bde_raw)
        except requests.RequestException:
            save_download(BDE_LEGACY_BE1901_CSV_URL, bde_raw)

    ine_clean = parse_ine_mortgage_rates(ine_raw)
    try:
        euribor_clean = parse_bde_wide_csv(bde_raw)
        bde_source_used = BDE_MONTHLY_INTEREST_CSV_URL
    except ValueError:
        if not download:
            raise
        save_download(BDE_LEGACY_BE1901_CSV_URL, bde_raw)
        euribor_clean = parse_bde_wide_csv(bde_raw)
        bde_source_used = BDE_LEGACY_BE1901_CSV_URL

    ine_out = PROCESSED_DIR / "tipo_hipotecario_ine_long.csv"
    euribor_out = PROCESSED_DIR / "euribor_12m_bde.csv"
    ine_clean.to_csv(ine_out, index=False)
    euribor_clean.to_csv(euribor_out, index=False)

    manifest = {
        "ine_source": INE_MORTGAGE_RATE_URL,
        "bde_source": bde_source_used,
        "bde_series": BDE_EURIBOR_SERIES_CODE,
        "ine_rows": str(len(ine_clean)),
        "euribor_rows": str(len(euribor_clean)),
        "ine_latest": ine_clean["fecha"].max().strftime("%Y-%m-%d"),
        "euribor_latest": euribor_clean["fecha"].max().strftime("%Y-%m-%d"),
    }
    (PROCESSED_DIR / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Usa los archivos raw ya existentes y solo regenera processed.",
    )
    args = parser.parse_args()
    result = main(download=not args.no_download)
    print(json.dumps(result, indent=2, ensure_ascii=False))
