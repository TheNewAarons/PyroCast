# Backlog (hallazgos medios y bajos de la revisión independiente)

> **Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.**

Lo que dejó abierto `docs/review.md` (2026-10-01): ningún crítico ni alto; lo de abajo no cambia las conclusiones publicadas pero conviene resolverlo, en orden de prioridad. Cada ítem dice dónde está, por qué importa y qué hacer.

## Medios

**M1 — Remuestreo con submuestreo y doble interpolación.** `ingestion/dem/pipeline.py:99` baja el DEM de 30 m a 250 m con `bilinear` (usa 4 píxeles de ~70 por celda: aliasing que ensucia elevación y, luego, pendiente); `ingestion/worldcover/pipeline.py:98` toma un solo píxel de 10 m (`nearest`) como el combustible de una celda de 250 m (hay 625); `features/vegetation/ndvi.py:131` repite el patrón con NDVI. Además, cada capa se remuestrea **dos veces** (a la grilla de 250 m en la ingesta y luego a la grilla del evento en `features/dataset/pipeline.py:180-216`, con medio píxel de desfase): un suavizado extra. *Qué hacer*: `Resampling.average` para continuas (DEM, NDVI) y `Resampling.mode` para `fuel_type` al bajar de resolución; remuestrear una sola vez desde la fuente a la grilla del evento. Requiere re-ingerir y reentrenar.

**M2 — NDVI del mismo mes que el evento (mirada hacia adelante).** `features/dataset/pipeline.py:92` (`_nearest_month_path`) y `:213` eligen el composite mensual *más cercano*, que para un incendio de enero es el de enero, mediana que incluye imágenes posteriores al inicio del fuego. En principio, las cicatrices de quema podrían filtrar la etiqueta a un canal de entrada. **Evidencia empírica en contra** (`bench/results/ndvi_fix_audit.json`): en 15 eventos el NDVI medio de las celdas quemadas es *mayor* que el de las no quemadas (13 de 15; diferencia media +0.05), o sea que se ve el tipo de vegetación que arde, no una cicatriz. Aun así, la política correcta es usar solo composites estrictamente anteriores al inicio del evento. *Qué hacer*: cambiar la política (con test) e ingerir el mes previo de cada evento (consume créditos de Copernicus: pedir permiso).

**M3 — Persistencia en PostGIS sin ejercitar.** Tres tests se saltan sin una base real (`shared/tests/test_db_schema.py:58`, `features/tests/test_dataset_db.py:27`, `models/tests/test_evaluation_db.py:36`); `features/dataset/db.py` tiene 31 % de cobertura. Nadie ha corrido ese código contra PostGIS en esta máquina. *Qué hacer*: un job de CI con el servicio `postgis` de `docker-compose.yml` que corra esos tests.

**M4 — Ramas del backtest sin test.** `models/cli.py:200-248` (modelos `blend` y `stacking` del CLI) tiene 68 % de cobertura; solo se ejercitaron corriendo el CLI a mano. *Qué hacer*: un test con un checkpoint de fixture, como el de `test_report.py`.

**M5 — Sesgo de exposición.** `models/deep/calibration.py:218` (`CalibratedUNet.predict`): el U-Net se entrena y se calibra con un paso (estado real del día anterior) pero se evalúa encadenando sus propias predicciones continuas. *Qué hacer*: entrenar con rollouts cortos (scheduled sampling) y calibrar sobre rollouts.

**M6 — Una sola semilla.** Un solo entrenamiento con semilla 42 por modelo; con n=2 eventos de test, la variación entre semillas probablemente supera las diferencias que muestra `docs/results.md`. *Qué hacer*: >= 5 semillas, reportar media y dispersión.

**M7 — El U-Net no se puede servir.** El checkpoint y su calibrador son `.pt` ignorados por git y `serving/api/model_registry.py:29` solo sabe cargar el autómata celular. Es el motivo operativo por el que el default sigue siendo el CA aunque el U-Net rinda mejor en los datos actuales. *Qué hacer*: publicar checkpoint + calibrador como asset de un release con su huella sha256 (`models/deep/calibration.py` ya la calcula), y agregar el cargador.

**M8 — `nan_to_num(0)` sobre cualquier canal.** `models/deep/train.py:111-112` (y la carga de val en `models/cli.py`, `models/evaluation/report_artifacts.py`) reemplaza NaN por 0 en todos los canales; un clima faltante entraría como 0 K. Hoy casi no ocurre (un NDVI residual y `fuel_type` de un evento de val), pero `features/dataset/pipeline.py:209,218` escribe NaN sin error cuando falta clima o NDVI. *Qué hacer*: fallar si el NaN supera un umbral por canal, y rellenar con un valor neutro por canal (no 0 universal).

**M9 — Selección de eventos y tamaño de muestra.** 15 eventos elegidos por umbral de detecciones (>= 40) más uno manual; 2 de test. Más temporadas y un val independiente del calibrador son lo que haría concluyentes las comparaciones.

**M10 — Viento de 10 m en el autómata celular.** `models/cellular_automata/rules.py` usa el viento de ERA5 a 10 m tal cual como "viento de propagación" (el viento a media llama suele ser una fracción menor) y sus parámetros son heurísticos sin calibrar. *Qué hacer*: calibrar `base_spread_prob`, `slope_coefficient` y `wind_coefficient` contra trayectorias multi-día reales (hoy `calibrate.py` solo soporta un paso sintético).

## Bajos

**L1 — Vectores de viento sin rotar.** `features/weather/derive.py:104` reproyecta `u10`/`v10` como si fueran escalares; en UTM 19S, a lon -72.5, la convergencia de cuadrícula es ~2°. Efecto pequeño frente a la incertidumbre del propio viento.

**L2 — `/tmp` en el Makefile.** `Makefile:94-95` escribe el requirements de `pip-audit` en `/tmp/...`; usar un archivo temporal (`mktemp`).

**L3 — Sin CSP.** `serving/api/main.py`: la API pone `nosniff`, `X-Frame-Options`, `Referrer-Policy` y `Permissions-Policy`, pero no `Content-Security-Policy` (no se pudo verificar contra un navegador real; una CSP mal calibrada rompería el mapa).

**L4 — Días UTC.** `ingestion/era5/aggregate.py` agrega ERA5-Land por día UTC y FIRMS usa `acq_date` UTC; Chile está en UTC-3/-4, así que un "día" de clima y uno de fuego no coinciden exactamente con el día local.

**L5 — `docs/limitations.md` es larguísimo** (~70 viñetas). Funciona como registro; falta un resumen de una página con las 10 que más importan.

**L6 — `make demo` llama a FIRMS con una clave de relleno** (para `/active-fires`): inofensivo (responde "Invalid MAP_KEY" y el mapa lo muestra), pero es una llamada de red en una demo "sin credenciales".

**L7 — Interfaz sin prueba en navegador.** Tests de servidor + sintaxis del JS; falta un test end-to-end con un navegador real (clic, predicción, deslizador).

**L8 — Imagen Docker pesada y sin construir aquí.** `serving` depende de `models` (arrastra `torch`) aunque el modelo servido no lo use; separar el autómata celular de `models/deep/` aliviaría la imagen. El job `docker` de CI es lo que la valida.

## Hecho en esta revisión (referencia)

C1 split por grupos · C2 NDVI · H1 normalización · H2 carga igual en entrenamiento y evaluación · H3 sin día 0 · H4 objetivo acumulado · aviso de investigación en cada doc (con test) · el backtest sobrevive a una base de datos caída con un aviso claro · `docs/dataset-card.md`/`backtest-2026.md`/`limitations.md`/`decisions.md` sincronizados.
