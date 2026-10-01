# Backtest 2025-2026: autómata celular vs. U-Net vs. ensamble contra incendios reales

*Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.*

Este documento reporta el resultado de evaluar y comparar el autómata celular (P7) y el U-Net (P9-P11) contra un conjunto de incendios reales de la temporada 2025-2026 en Biobío, Ñuble y La Araucanía, construidos con el pipeline completo de ingesta y features (P1-P6) sobre datos reales de esas fechas -- no de fixture.

**Resultado honesto por adelantado (P13, ensamble)**: el blend CA+U-Net (peso del U-Net 0.4, elegido en val) **no supera al mejor modelo individual en todas las métricas** -- mejora IoU/Dice sobre el CA pero pierde Brier/ECE contra el U-Net. El stacking logístico sí gana en las 4 métricas, pero se ajustó sobre 2 eventos de val que el calibrador del U-Net ya vio, y se evalúa con n=2 -- **no se considera evidencia suficiente** para promoverlo. **Ningún ensamble es el default de `serving/`**; ver sección 8.

**Resultado honesto por adelantado (P12)**: en 1 de los 2 eventos de test, el U-Net calibrado **no supera** al autómata celular en IoU/Dice (los pierde por un margen amplio). Ver la sección "Comparación" y su hipótesis. Con solo 2 eventos de test, esta comparación **no es estadísticamente robusta** -- ver "Limitaciones de esta evaluación".

## 1. Credenciales y datos reales usados

Se confirmó al inicio de esta tarea que no había credenciales configuradas (`shared.config.get_settings()` fallaba nombrando las 5 variables faltantes). El usuario proveyó credenciales reales de NASA FIRMS, Copernicus CDS (ERA5-Land) y Copernicus Data Space (Sentinel-2); se verificaron con llamadas reales y no destructivas antes de usarlas. Nunca se comitearon (`.env`, en `.gitignore`).

## 2. Eventos reales usados

Fuente: detecciones activas de fuego NASA FIRMS, sensor VIIRS (satélite Suomi NPP), producto **VIIRS_SNPP_SP** (Standard Processing / archivo científico -- `VIIRS_SNPP_NRT` no cubre fechas de más de ~7 meses de antigüedad). Rango descargado: 2025-11-01 a 2026-03-31, bbox de `shared.config.study_area_bbox` (Biobío + Ñuble + Araucanía). Total: **8083 detecciones reales**, clusterizadas (`features.fire_state.clustering`, `spatial_eps_m`/`temporal_eps` por defecto) en **1441 eventos**.

**Criterio de selección** (documentado, no arbitrario): la inmensa mayoría de los 1441 eventos son ruido de 1-5 detecciones (un solo píxel VIIRS aislado). Se seleccionaron los eventos con **≥40 detecciones**, más **1 evento de diciembre incluido a mano** (`4199024687`, 25 detecciones) para no perder diversidad estacional -- sin él, la muestra habría cubierto solo noviembre, enero y marzo. Total: **15 eventos reales**.

| event_id | detecciones | fecha (real, sin padding) | bbox aprox. (WGS84, oeste,sur,este,norte) | split |
|---|---|---|---|---|
| 1277049523 | 1918 | 2026-01-17 a 2026-01-23 | -73.025,-36.931,-72.863,-36.637 | train |
| 3051657321 | 450 | 2026-01-18 a 2026-01-22 | -72.682,-36.811,-72.589,-36.725 | train |
| 2030469600 | 369 | 2026-01-18 a 2026-01-23 | -72.662,-37.393,-72.603,-37.242 | train |
| 2582836092 | 207 | 2026-01-18 a 2026-01-21 | -72.605,-38.018,-72.540,-37.944 | **test** |
| 12676775 | 159 | 2026-01-16 a 2026-01-20 | -72.613,-36.586,-72.557,-36.514 | val |
| 2728181583 | 143 | 2026-01-14 a 2026-01-18 | -72.646,-36.621,-72.607,-36.572 | val |
| 1262509675 | 105 | 2026-01-18 a 2026-01-20 | -72.430,-36.832,-72.387,-36.767 | train |
| 1013976330 | 96 | 2025-11-15 a 2025-11-23 | -71.170,-38.085,-71.151,-38.066 | train |
| 2320813586 | 77 | 2026-03-26 a 2026-03-29 | -72.200,-37.867,-72.175,-37.832 | train |
| 1214084367 | 75 | 2026-01-18 a 2026-01-21 | -72.568,-37.554,-72.532,-37.532 | train |
| 203187374 | 66 | 2026-01-15 a 2026-01-20 | -72.604,-36.623,-72.571,-36.602 | **test** |
| 775785671 | 50 | 2026-01-19 a 2026-01-20 | -73.056,-38.658,-73.006,-38.642 | train |
| 1482774057 | 47 | 2026-01-19 a 2026-01-21 | -72.625,-37.506,-72.595,-37.479 | train |
| 2970931921 | 42 | 2026-01-18 a 2026-01-27 | -71.294,-37.637,-71.262,-37.623 | train |
| 4199024687 | 25 | 2025-12-17 a 2025-12-18 | -72.204,-36.648,-72.184,-36.631 | train |

`event_id` 1277049523 (1918 detecciones, 17-23 de enero) es, por lejos, el incendio dominante de la muestra -- corresponde al complejo de incendios de mediados de enero de 2026 en Biobío, el más severo mencionado en `CLAUDE.md`.

**Split train/val/test**: `features.dataset.split.split_events`, por evento (nunca por píxel ni por día, evita fuga de datos), semilla 42, 70/15/15 → **train=11, val=2, test=2**. El split es determinista (depende solo del conjunto de IDs y la semilla), reproducido en este documento arriba.

**Pipeline P1-P6 real ejecutado para cada evento**: DEM Copernicus GLO-30 + pendiente/orientación (bbox completo, una sola vez), ESA WorldCover (bbox completo, una sola vez), ERA5-Land diario (por evento, ventana `[fecha_inicio - 5 días, fecha_fin]` -- ver `padded_days_for_event`), Sentinel-2 L2A mensual + NDVI (por evento, bbox chico centrado en cada evento -- el bbox completo de la zona de estudio excede el límite de píxeles de un job síncrono de openEO, ver `docs/limitations.md`).

## 3. Bugs reales encontrados y corregidos durante esta tarea

Ejecutar el pipeline contra datos reales (por primera vez en todo este proyecto) expuso 6 bugs/gaps reales que ningún fixture de test había disparado:

1. **FIRMS `acq_time` pierde ceros a la izquierda** (Area API lo serializa numérico: `"517"` = 05:17 UTC, no 51:7) -- `ingestion/firms/parser.py` y `features/dataset/firms_loader.py` (duplicado a propósito, `features` nunca depende de `ingestion`). Commits `b374700`, `862cc4f`.
2. **Sentinel-2 openEO adivina el formato de salida desde la extensión del archivo** -- rompía el patrón de escritura atómica vía `.part`. Commit `2c622d6`.
3. **`pyrocast-features build-dataset` no tenía forma de acotar a un subconjunto de eventos** -- procesar el rango de fechas completo de la temporada clusteriza en 1441 eventos, la mayoría ruido. Se agregó `--event-ids`. Commit `ea8b493`.
4. **Ventanas de ERA5 mal dimensionadas**: no se tuvo en cuenta que `padded_days_for_event` antepone 5 días de padding antes del inicio real de cada evento -- la primera vez que se corrió el entrenamiento, la pérdida fue `NaN` en todas las épocas. Se re-ingirió ERA5 con ventanas correctas.
5. **ERA5-Land solo cubre tierra**: eventos cercanos a la costa producen NaN parcial en clima incluso en días con datos disponibles (interpolación bilineal tierra/océano). Se rellena con el promedio regional del archivo fuente completo -- `features/dataset/pipeline.py`, commit `71bc3e3`, documentado en `docs/limitations.md`.
6. **NaN residual minúsculo** (<0.1% de celdas, bordes de cobertura de WorldCover/Sentinel-2) se reemplaza por 0 en `ChileFinetuneDataset` -- commit `f9ba428`.
7. **Hallazgo más importante para la validez de este backtest**: el día 0 de cada tensor de evento (por el padding de 5 días) **no tiene ningún píxel en llamas** -- pero tanto `CalibratedUNet.predict` como `CellularAutomatonModel.predict` asumen que el día 0 es el "ancla conocida" (el estado actual del fuego del que hay que propagar). Sin corregir esto, ambos modelos partían de "nada ardiendo" y no podían anticipar la ignición real, midiendo algo que ninguno de los dos modelos está diseñado a hacer. Se recorta cada evento de test a su primer día con fuego antes de evaluar -- `models/cli.py::_trim_to_first_fire_day`, commit `358f3d4`.

Ninguno de estos 7 hallazgos se ocultó ni se "arregló para que el número se vea mejor" sin dejar rastro: cada uno tiene su commit, su test que reproduce el fallo real, y una entrada en `docs/limitations.md` cuando aplica.

## 4. Modelos evaluados

- **Autómata celular (P7)**: `CellularAutomatonModel`, parámetros **HEURÍSTICOS POR DEFECTO** (`base_spread_prob=0.3`, `slope_coefficient=4.0`, `wind_coefficient=0.2`), **NO calibrados contra incendios reales de Chile**. `calibrate.py` (el mecanismo de calibración de P7) solo soporta muestras sintéticas de un único paso simulado (`_score_sample`, `n_days=1` hardcodeado) -- extenderlo a trayectorias multi-día reales de `features/dataset/` es una limitación preexistente y documentada (`docs/limitations.md`), fuera de alcance de esta tarea. **Este es un punto importante de honestidad**: pese a que el enunciado de esta tarea habla de "el autómata celular calibrado (P7)", el autómata efectivamente evaluado aquí NO está calibrado -- la infraestructura de calibración de P7 existe pero no alcanza a producir un candidato ajustado contra estos eventos reales.
- **U-Net (P9-P11)**: `SmallUNet`, entrenado **desde cero** (sin preentrenamiento en NDWS/Kaggle -- decisión explícita del usuario, sin esas credenciales disponibles) directamente sobre los **11 eventos reales de train**, 22 épocas con early stopping (`pyrocast-train finetune`), calibrado con regresión isotónica sobre los **2 eventos reales de val** (`pyrocast-calibrate run --chile-val`, nueva opción agregada en esta tarea, commit `97475fc`).

  Entrenamiento: `runs/finetune_2026_v2/history.csv`, pérdida focal baja de 0.019 a ~0.012 (train) y se estabiliza en ~0.016 (val) -- convergencia real, sin NaN ni inf.

  Calibración (`--chile-val`, 29268 muestras): Brier 0.0764 → 0.0382, ECE 0.1621 → 0.0000. El ECE exactamente 0.0000 post-calibración es un artefacto **estructural** ya documentado en `docs/calibration.md`/`docs/limitations.md` (calibrar y evaluar sobre el MISMO split infla el ECE hacia 0, no evidencia de calibración perfecta) -- no específico de esta corrida.

  **El U-Net vio en total 11 eventos reales para entrenar** -- órdenes de magnitud menos que el dataset público de referencia (NDWS: 18.545 chips). Cualquier resultado débil debe leerse a la luz de este tamaño de entrenamiento, no como evidencia de que la arquitectura sea inadecuada.

## 5. Backtest: métricas + intervalos de confianza (bootstrap, 1000 remuestreos, 95%)

Los modelos se evaluaron contra los **mismos 2 eventos de test** (`203187374`, `2582836092`), con las mismas 4 métricas (`models/evaluation/metrics.py`): Brier, ECE, IoU, Dice.

### Por evento

| event_id | modelo | IoU | Dice | Brier | ECE |
|---|---|---|---|---|---|
| 203187374 | CA (sin calibrar) | 0.351 | 0.520 | 0.082 | 0.082 |
| 203187374 | U-Net (calibrado) | **0.442** | 0.613 | **0.044** | **0.035** |
| 203187374 | Blend CA+U-Net (w=0.4) | 0.384 | 0.555 | 0.058 | 0.058 |
| 203187374 | Stacking logístico | 0.462 | 0.632 | 0.061 | 0.055 |
| 2582836092 | CA (sin calibrar) | **0.288** | **0.447** | 0.103 | 0.100 |
| 2582836092 | U-Net (calibrado) | 0.055 | 0.104 | 0.096 | 0.094 |
| 2582836092 | Blend CA+U-Net (w=0.4) | 0.284 | 0.442 | 0.092 | 0.090 |
| 2582836092 | Stacking logístico | 0.564 | 0.722 | 0.056 | 0.020 |

(Negritas: mejor valor entre CA y U-Net, comparación original de P12; las filas de ensamble se agregaron sin recalcular negritas.)

### Agregado (bootstrap sobre los 2 eventos de test)

| métrica | CA (sin calibrar) | U-Net (calibrado) | Ensamble: blend (w=0.4) | Ensamble: stacking |
|---|---|---|---|---|
| IoU ↑ | 0.319 [0.288, 0.351] | 0.249 [0.055, 0.442] | 0.334 [0.284, 0.384] | 0.513 [0.462, 0.564] |
| Dice ↑ | 0.483 [0.447, 0.520] | 0.359 [0.104, 0.613] | 0.499 [0.442, 0.555] | 0.677 [0.632, 0.722] |
| Brier ↓ | 0.092 [0.082, 0.103] | 0.070 [0.044, 0.096] | 0.075 [0.058, 0.092] | 0.058 [0.056, 0.061] |
| ECE ↓ | 0.091 [0.082, 0.100] | 0.064 [0.035, 0.094] | 0.074 [0.058, 0.090] | 0.038 [0.020, 0.055] |

Resultados crudos, con el comando exacto y el commit de git que los produjo: `bench/results/baseline.json` (CA), `bench/results/unet.json` (U-Net), `bench/results/blend.json` y `bench/results/stacking.json` (ensambles, commit `d19511a`). Mismo código de backtest (`run_backtest`), mismos 2 eventos de test, misma semilla, mismos 1000 remuestreos.

## 6. Comparación honesta: dónde el U-Net NO supera al autómata celular

En el evento `203187374`, el U-Net supera al autómata celular en las 4 métricas, con margen claro.

En el evento `2582836092`, **el U-Net pierde marcadamente contra el autómata celular en IoU (0.055 vs. 0.288) y Dice (0.104 vs. 0.447)** -- menos de un quinto del desempeño del autómata celular en solape espacial. En Brier/ECE el U-Net sigue levemente mejor (0.096 vs. 0.103; 0.094 vs. 0.100), pero la diferencia es pequeña comparada con la brecha en IoU/Dice. Esto arrastra el AGREGADO de IoU y Dice por debajo del autómata celular (0.249 vs. 0.319; 0.359 vs. 0.483) -- en esas dos métricas, en promedio sobre los 2 eventos de test, **el U-Net no supera al autómata celular**.

**Hipótesis** (documentadas, no una sola causa confirmada -- con 11 eventos de train y 2 de test no hay forma de aislar la causa exacta con confianza estadística):

1. **Tamaño de entrenamiento extremo**: 11 eventos reales es una fracción ínfima de lo que la literatura usa (NDWS: 18.545 chips). Un solo evento de train con características idiosincráticas (tipo de combustible, patrón de viento) puede dominar lo que el modelo aprende, y el modelo puede no generalizar bien a un evento de test cuyas condiciones locales estén sub-representadas en esos 11 eventos.
2. **Sin preentrenamiento**: por la decisión explícita de no usar NDWS (sin credenciales de Kaggle), el U-Net nunca vio patrones generales de propagación de fuego más allá de estos 11 eventos chilenos -- a diferencia de un U-Net preentrenado, que traería un prior aprendido de miles de incendios reales antes de siquiera ver Chile.
3. **Autómata celular como modelo puramente físico, no aprendido**: al no depender de datos de entrenamiento, el autómata celular aplica las mismas reglas heurísticas en cualquier ubicación -- puede ser menos preciso en promedio, pero no sufre la misma brecha de generalización que un modelo aprendido con una muestra de entrenamiento tan chica.
4. **Tamaño de muestra de test**: con solo 2 eventos de test, un solo evento "difícil" domina completamente el agregado -- esto podría ser varianza de muestra, no una deficiencia sistemática del U-Net; una muestra de test más grande podría mostrar un panorama distinto.

No se ajustó nada de lo anterior para que el número final se viera mejor -- este resultado se reporta tal cual.

## 7. Limitaciones de esta evaluación (además de las ya documentadas en `docs/limitations.md`)

- **n=2 eventos de test**: el split 70/15/15 sobre 15 eventos totales deja solo 2 eventos de test (y 2 de val). Los intervalos de confianza bootstrap son anchos y, sobre todo para el U-Net, prácticamente abarcan de "mediocre" a "bueno" (IoU: 0.055 a 0.442) -- esta NO es una comparación estadísticamente robusta. Los resultados son ilustrativos del pipeline funcionando de punta a punta con datos reales, no una conclusión definitiva sobre qué modelo es mejor.
- **Autómata celular sin calibrar** (ver sección 4) -- la comparación no es "U-Net calibrado vs. CA calibrado" como pedía el enunciado original, sino "U-Net calibrado vs. CA con parámetros heurísticos por defecto".
- **U-Net entrenado sin preentrenamiento y con 11 eventos reales** -- no representativo de lo que un U-Net con preentrenamiento en un dataset público lograría.
- **ERA5-Land interpolado geométricamente** de 9 km a 250 m (limitación de diseño ya documentada) + relleno de NaN costero con el promedio regional (hallazgo #5 de la sección 3) -- el clima "local" de un evento muy costero es, en la práctica, un promedio regional, no una medición específica de ese punto.
- **Selección de eventos por umbral de detecciones (≥40)**, no aleatoria -- sesga la muestra hacia incendios más grandes/mejor detectados por VIIRS; no representa la cola larga de incendios pequeños de la temporada.

## 8. Ensamble CA + U-Net (P13) y modelo por defecto en `serving/`

Implementación: `models/deep/ensemble.py` (`BlendEnsemble`, `StackingEnsemble`, `select_blend_weight`), expuesta como `pyrocast-models backtest --model blend|stacking`. Ambos implementan `FireSpreadModel`, así que corren por el MISMO `run_backtest`.

- **Blend**: `(1-w)·CA + w·U-Net`. `w` se elige por Brier medio sobre los 2 eventos de **val** (grilla 0.0-1.0 paso 0.1), nunca sobre test -> **w=0.4**.
- **Stacking**: regresión logística sobre `[logit(p_CA), logit(p_UNet)]`, ajustada sobre val. Coeficientes: CA 0.08, U-Net 1.75, intercepto 4.49 (`bench/results/stacking.json`). Es decir, ignora casi por completo al CA y re-escala la salida del U-Net.
- Los NaN residuales de val (99 celdas de `fuel_type` del evento `12676775`) se reemplazan por 0 al ajustar, igual que `ChileFinetuneDataset` al entrenar.

**Lectura honesta** (criterio de aceptación: ¿mejora sobre el mejor individual?):

1. **Blend: NO mejora sobre el mejor individual de forma consistente.** IoU/Dice suben respecto del CA (0.334 vs 0.319; 0.499 vs 0.483) y mucho respecto del U-Net, pero **en Brier (0.075 vs 0.070) y ECE (0.074 vs 0.064) el U-Net solo es mejor**. Es un compromiso entre los dos, no un dominio. Los intervalos de CA y blend se solapan por completo en IoU/Dice.
2. **Stacking: gana en las 4 métricas, pero no se acepta como evidencia.** (a) Se ajusta sobre **2 eventos de val**, y el calibrador isotónico del U-Net ya se ajustó sobre esos mismos 2 eventos, así que las salidas del U-Net en val son optimistas (fuga documentada en `docs/calibration.md`). (b) Evaluación con **n=2 eventos de test**: el intervalo bootstrap estrecho del stacking (IoU [0.462, 0.564]) refleja que ambos eventos dieron valores parecidos, no precisión. (c) El intercepto grande (4.49) indica que gran parte de la ganancia puede ser re-escalado de probabilidades (y de la base de "ya ardió", acumulada) y no información nueva; con 2 eventos no se puede separar. (d) Es el único modelo cuyo resultado cambia tan radicalmente el evento `2582836092` (IoU 0.055 -> 0.564 vs. el U-Net), lo que pide replicación antes de creerlo. **No se hizo validación cruzada por evento** porque 2 eventos de val no la permiten.
3. Con una muestra mayor de eventos (más temporadas) y un split val distinto del usado para calibrar el U-Net, el stacking podría confirmarse o desaparecer. No hay forma de saberlo con estos datos.

**Decisión de modelo por defecto en `serving/`: autómata celular (`CellularAutomatonModel`, parámetros por defecto).** Ningún ensamble queda como opción por defecto.

Por qué el CA y no el U-Net: (i) mejor IoU/Dice agregado (0.319 vs 0.249) y **sin el fallo catastrófico** del U-Net en el evento `2582836092` (IoU 0.288 vs 0.055); (ii) sin checkpoint ni calibrador que versionar y sin dependencia de torch en el camino de servido; (iii) comportamiento explicable por reglas. Costo asumido: peor Brier/ECE que el U-Net (probabilidades menos calibradas) -- cualquier consumidor de `serving/` debe tratar sus probabilidades como un puntaje relativo, no como una probabilidad calibrada. Hoy `serving/` solo expone `/healthz` (no hay endpoint de predicción), así que esta decisión es la que **deberá implementar** el futuro endpoint; no hay código de servido que cambiar todavía. Se revisa cuando exista una evaluación con más eventos. Registrado en `docs/decisions.md`.

## 9. Reproducibilidad

Comandos exactos y commit de git usados, embebidos en cada resultado (`bench/results/baseline.json`, `unet.json`, `blend.json`, `stacking.json`, campos `"command"` y `"git_commit"`):

```
pyrocast-models backtest --n-bootstrap 1000 --seed 42
pyrocast-models backtest --model unet --checkpoint runs/finetune_2026_v2/best.pt --n-bootstrap 1000 --seed 42
pyrocast-models backtest --model blend --checkpoint runs/finetune_2026_v2/best.pt --n-bootstrap 1000 --seed 42
pyrocast-models backtest --model stacking --checkpoint runs/finetune_2026_v2/best.pt --n-bootstrap 1000 --seed 42
```

Para reproducir de punta a punta con las mismas credenciales: ingerir FIRMS/DEM/WorldCover/ERA5/Sentinel-2 para las fechas y bboxes de la sección 2, `pyrocast-features build-dataset --event-ids <ids>` por evento, `pyrocast-train finetune`, `pyrocast-calibrate run --chile-val`, y los dos comandos de arriba.
