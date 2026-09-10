"""Recalcula el backtest, la previsión vigente y el histórico de previsiones.

Se ejecuta tras ``download_data.py`` y ``validate_data.py``; usa
exclusivamente los datos ya procesados en ``data/processed``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import data_processing as dp
import forecasting as fc

TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
HORIZONS = [1, 3, 6]
PUBLICATION_LEAD_MONTHS = 2


def main(verbose: bool = True) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    dp.check_data_quality()

    hipotecas, euribor = dp.load_raw_tables()
    mortgage = dp.mortgage_series(hipotecas)[["fecha", "tipo_hipotecario"]]

    run = fc.run_full_forecast_pipeline(
        mortgage,
        euribor,
        history_path=TABLES_DIR / "forecast_history.csv",
        horizons=HORIZONS,
        origins=60,
        minimum_training=120,
        publication_lead_months=PUBLICATION_LEAD_MONTHS,
        verbose=verbose,
    )

    run.backtest.to_csv(TABLES_DIR / "forecast_backtest_detail.csv", index=False)
    run.evaluation.to_csv(TABLES_DIR / "forecast_evaluation.csv", index=False)
    run.current_forecasts.to_csv(TABLES_DIR / "current_forecasts.csv", index=False)
    run.history.to_csv(TABLES_DIR / "forecast_history.csv", index=False)
    (TABLES_DIR / "forecast_metadata.json").write_text(
        json.dumps(run.metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(run.evaluation.to_string(index=False))
    print(run.current_forecasts[run.current_forecasts["selected"]].to_string(index=False))


if __name__ == "__main__":
    main()
