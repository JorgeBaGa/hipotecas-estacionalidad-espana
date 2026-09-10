# Estacionalidad y previsión del tipo hipotecario en España

Análisis de datos sobre el mercado hipotecario español: evolución histórica
de los tipos de interés, relación con el Euríbor, posible estacionalidad
mensual y previsión a corto plazo con validación fuera de muestra.

Proyecto de análisis de datos desarrollado originalmente como soporte para el
seguimiento del mercado hipotecario en
[calculatucuota.es](https://calculatucuota.es), una web de simulación de
hipotecas. Este repositorio recoge esa parte de análisis de datos de forma
independiente; no es el código de la plataforma web.

El cuaderno principal, [`notebooks/hipotecas_analysis.ipynb`](notebooks/hipotecas_analysis.ipynb),
es el documento a abrir primero: contiene toda la narrativa, los gráficos y
las conclusiones.

## Qué analiza

- Evolución histórica de los tipos hipotecarios y del Euríbor a 12 meses.
- Si existe un mes del año sistemáticamente más barato para contratar una
  hipoteca (estacionalidad).
- Si ese patrón, de existir, es estable en el tiempo.
- Diferencias entre hipotecas a tipo fijo y a tipo variable.
- Previsión del tipo hipotecario medio a 1, 3 y 6 meses, con validación
  temporal (backtesting) de los modelos utilizados.

## Fuentes de datos

- **INE**, tabla [24457](https://www.ine.es/jaxiT3/files/t/csv_bdsc/24457.csv):
  tipo de interés medio al inicio de las hipotecas constituidas sobre
  fincas, por naturaleza de la finca y tipo de interés (fijo/variable/total).
- **Banco de España**, catálogo de tipos de interés mensuales: serie
  `D_1NBAF472` (Euríbor a 12 meses).

Ambas fuentes se descargan directamente de sus URLs oficiales con
`src/download_data.py`. El detalle de columnas y controles de calidad está en
[DATA_SOURCES.md](DATA_SOURCES.md).

## Estructura del repositorio

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── DATA_SOURCES.md
├── data/
│   ├── raw/                     # CSV descargados tal cual de INE y BdE
│   └── processed/                # tablas limpias listas para analizar
├── notebooks/
│   └── hipotecas_analysis.ipynb  # cuaderno principal: análisis y previsión
├── src/
│   ├── download_data.py          # descarga y limpieza de datos oficiales
│   ├── validate_data.py          # validación de esquema y calidad
│   ├── data_processing.py        # carga y preparación de las series
│   ├── analysis.py                # estacionalidad, rupturas, relación con Euríbor
│   ├── forecasting.py             # modelos, backtesting y previsión
│   └── update_forecasts.py        # orquesta el backtest y guarda resultados
├── outputs/
│   ├── figures/                   # gráficos exportados desde el cuaderno
│   └── tables/                    # evaluación de modelos, previsión e historial
└── .github/workflows/
    └── update-data.yml            # actualización mensual automática
```

## Metodología

1. **Extracción**: descarga de los CSV oficiales del INE y del Banco de
   España (`src/download_data.py`), con reintento a una fuente alternativa
   del Banco de España si la principal cambia de formato o no está
   disponible.
2. **Limpieza y validación**: normalización de columnas y tipos, y
   comprobación automática de esquema, fechas, valores nulos, duplicados y
   consistencia del número de filas por mes (`src/validate_data.py`). El
   resultado se guarda en `data/processed/data_quality_report.json`.
3. **Análisis exploratorio y estacionalidad** (`src/analysis.py`):
   comparación mensual dentro de cada año, un modelo de regresión con
   errores autocorrelacionados (AR(2)) que controla por el nivel del Euríbor
   de los últimos siete meses, repetición del contraste en ventanas móviles
   de ocho años y por tipo de producto (fijo/variable), y corrección de Holm
   al comparar los doce meses a la vez.
4. **Historia del mercado**: detección de rupturas en la tendencia del
   Euríbor (partición óptima por mínimos cuadrados con selección del número
   de tramos por BIC) para dividir la serie en etapas, y comparación en cada
   una de la correlación y sensibilidad entre el cambio mensual del Euríbor
   y el del tipo hipotecario.
5. **Previsión** (`src/forecasting.py`): cuatro modelos alternativos (último
   valor, ETS, ARIMA y ARIMAX con Euríbor) a 1, 3 y 6 meses, con selección
   automática de la especificación de cada modelo por AICc.
6. **Validación (backtesting)**: validación temporal con 60 orígenes
   móviles: en cada fecha histórica, el modelo solo utiliza los datos
   disponibles hasta ese momento. El ARIMAX simula el retraso real de
   publicación del Euríbor frente al dato hipotecario (hasta dos meses) en
   vez de usar el valor futuro real, para evitar *look-ahead bias*. El
   modelo con menor error fuera de muestra (MAE) se selecciona para la
   previsión vigente en cada horizonte.
7. **Seguimiento de acierto**: cada ejecución añade la previsión vigente a
   `outputs/tables/forecast_history.csv` sin sobrescribir las anteriores;
   cuando el INE publica el dato real, se calcula el error de la previsión ya
   emitida.

No es un análisis econométrico causal: el objetivo es un análisis aplicado y
reproducible para el seguimiento del mercado.

## Reproducibilidad

Requisitos: Python 3.10+.

```bash
git clone <url-del-repositorio>
cd hipotecas-python-github

python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Descargar y validar los datos oficiales:

```bash
python src/download_data.py
python src/validate_data.py
```

Recalcular el backtest y la previsión vigente (tarda varios minutos: ajusta
~250 modelos ETS/ARIMA/ARIMAX sobre 60 orígenes históricos):

```bash
python src/update_forecasts.py
```

Abrir y ejecutar el cuaderno principal:

```bash
jupyter notebook notebooks/hipotecas_analysis.ipynb
```

El cuaderno lee los datos de `data/processed/` y las previsiones ya
calculadas en `outputs/tables/`; no vuelve a ejecutar el backtest en cada
ejecución (por eso es un paso aparte). Con **Kernel → Restart & Run All**
se reproducen todos los gráficos y tablas de principio a fin en segundos.

Si se prefiere no descargar datos nuevos, `data/raw/` y `data/processed/` ya
incluyen una copia funcional, y `outputs/tables/` ya incluye una previsión
vigente calculada.

## Automatización

`.github/workflows/update-data.yml` se ejecuta el último domingo de cada mes
(o manualmente): descarga y valida las fuentes oficiales, recalcula el
backtest y las previsiones, actualiza el historial de acierto sin borrar
previsiones anteriores, vuelve a ejecutar el cuaderno de principio a fin y
hace commit de los datos, tablas y cuaderno actualizados. Si una fuente
cambia de formato o falla una validación, el flujo se detiene sin publicar
resultados a medias.

## Tecnologías

- **Python**: pandas, NumPy.
- **statsmodels / scipy**: modelos de regresión con errores AR(2), ETS,
  ARIMA/ARIMAX, contrastes estadísticos y corrección de Holm.
- **matplotlib**: todos los gráficos del cuaderno.
- **requests**: descarga de las fuentes oficiales.
- **Jupyter**: cuaderno principal.
- **GitHub Actions**: automatización mensual.

## Licencia

MIT. Ver [LICENSE](LICENSE).
