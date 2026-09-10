# Fuentes y esquemas de datos

Este documento describe las fuentes, los archivos derivados y sus controles.

## INE

- Fuente: `https://www.ine.es/jaxiT3/files/t/csv_bdsc/24457.csv`
- Archivo raw: `data/raw/tipo_hipotecario_ine_24457.csv`
- Archivo procesado: `data/processed/tipo_hipotecario_ine_long.csv`

Columnas procesadas:

- `fecha`: primer día del mes en formato ISO.
- `periodo`: periodo original del INE, por ejemplo `2025M12`.
- `naturaleza_finca`: categoría del INE.
- `tipo_interes`: total, fijo o variable.
- `tipo_hipotecario`: porcentaje convertido a número decimal con punto.

## Banco de España

- Fuente principal: `https://www.bde.es/webbe/es/estadisticas/compartido/datos/csv/cal_umv_mfi001.csv`
- Fallback: `https://www.bde.es/webbe/es/estadisticas/compartido/datos/csv/be1901.csv`
- Serie usada: `D_1NBAF472`
- Archivo raw: `data/raw/euribor_12m_bde.csv`
- Archivo procesado: `data/processed/euribor_12m_bde.csv`

Columnas procesadas:

- `fecha`: primer día del mes en formato ISO.
- `euribor_12m`: valor mensual de la serie `D_1NBAF472`.

## Validación

El script `src/validate_data.py` comprueba:

- existencia de archivos procesados;
- esquema esperado;
- fechas nulas;
- valores nulos;
- duplicados por clave;
- número constante de filas INE por mes;
- rango temporal disponible.

El reporte se guarda en `data/processed/data_quality_report.json`.

## Previsiones derivadas

El script `src/update_forecasts.py` utiliza exclusivamente los archivos
procesados anteriores y genera:

- `forecast_backtest_detail.csv`: predicciones históricas de origen móvil;
- `forecast_evaluation.csv`: MAE, RMSE y cobertura por modelo y horizonte;
- `current_forecasts.csv`: previsiones vigentes de todos los modelos;
- `forecast_history.csv`: registro acumulado de las previsiones seleccionadas;
- `forecast_metadata.json`: fechas de datos y parámetros de la ejecución.

La validación compara un último valor, ETS, ARIMA y ARIMAX con Euríbor. Para
simular la disponibilidad de información, el ARIMAX puede utilizar hasta dos
meses de Euríbor posteriores al último dato hipotecario, coherente con el
retraso habitual de publicación del INE.
