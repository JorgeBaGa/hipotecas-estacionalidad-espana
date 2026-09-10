"""Validación de esquema y calidad de los datos procesados."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

INE_FILE = PROCESSED_DIR / "tipo_hipotecario_ine_long.csv"
EURIBOR_FILE = PROCESSED_DIR / "euribor_12m_bde.csv"
REPORT_FILE = PROCESSED_DIR / "data_quality_report.json"


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_ine(errors: list[str]) -> dict[str, object]:
    df = pd.read_csv(INE_FILE, parse_dates=["fecha"])
    expected_columns = [
        "fecha",
        "periodo",
        "naturaleza_finca",
        "tipo_interes",
        "tipo_hipotecario",
    ]
    check(list(df.columns) == expected_columns, f"INE schema inesperado: {list(df.columns)}", errors)
    check(not df.empty, "INE esta vacio", errors)
    check(df["fecha"].notna().all(), "INE contiene fechas nulas", errors)
    check(df["tipo_hipotecario"].notna().all(), "INE contiene tipos hipotecarios nulos", errors)
    check((df["tipo_hipotecario"] >= 0).all(), "INE contiene tipos hipotecarios negativos", errors)

    duplicate_keys = df.duplicated(["fecha", "naturaleza_finca", "tipo_interes"]).sum()
    check(duplicate_keys == 0, f"INE tiene claves duplicadas: {duplicate_keys}", errors)

    rows_per_month = df.groupby("fecha").size()
    check(rows_per_month.min() == rows_per_month.max(), "INE no tiene el mismo numero de filas por mes", errors)

    return {
        "rows": int(len(df)),
        "unique_months": int(df["fecha"].nunique()),
        "start": df["fecha"].min().strftime("%Y-%m-%d"),
        "end": df["fecha"].max().strftime("%Y-%m-%d"),
        "rows_per_month_min": int(rows_per_month.min()),
        "rows_per_month_max": int(rows_per_month.max()),
        "naturaleza_finca": sorted(df["naturaleza_finca"].dropna().unique().tolist()),
        "tipo_interes": sorted(df["tipo_interes"].dropna().unique().tolist()),
    }


def validate_euribor(errors: list[str]) -> dict[str, object]:
    df = pd.read_csv(EURIBOR_FILE, parse_dates=["fecha"])
    expected_columns = ["fecha", "euribor_12m"]
    check(list(df.columns) == expected_columns, f"Euribor schema inesperado: {list(df.columns)}", errors)
    check(not df.empty, "Euribor esta vacio", errors)
    check(df["fecha"].notna().all(), "Euribor contiene fechas nulas", errors)
    check(df["euribor_12m"].notna().all(), "Euribor contiene valores nulos", errors)

    duplicate_dates = df["fecha"].duplicated().sum()
    check(duplicate_dates == 0, f"Euribor tiene fechas duplicadas: {duplicate_dates}", errors)

    return {
        "rows": int(len(df)),
        "unique_months": int(df["fecha"].nunique()),
        "start": df["fecha"].min().strftime("%Y-%m-%d"),
        "end": df["fecha"].max().strftime("%Y-%m-%d"),
        "min_value": float(df["euribor_12m"].min()),
        "max_value": float(df["euribor_12m"].max()),
    }


def main() -> dict[str, object]:
    errors: list[str] = []
    check(INE_FILE.exists(), f"No existe {INE_FILE}", errors)
    check(EURIBOR_FILE.exists(), f"No existe {EURIBOR_FILE}", errors)

    report: dict[str, object] = {"status": "failed", "errors": errors}
    if not errors:
        report["ine"] = validate_ine(errors)
        report["euribor"] = validate_euribor(errors)

    report["status"] = "passed" if not errors else "failed"
    report["errors"] = errors
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if errors:
        raise SystemExit("\n".join(errors))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
