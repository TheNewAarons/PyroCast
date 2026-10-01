# Backtest 2025-2026: autómata celular vs. U-Net vs. ensamble contra incendios reales

*Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.*

Este documento reporta el resultado de evaluar y comparar el autómata celular (P7) y el U-Net (P9-P11) contra un conjunto de incendios reales de la temporada 2025-2026 en Biobío, Ñuble y La Araucanía, construidos con el pipeline completo de ingesta y features (P1-P6) sobre datos reales de esas fechas -- no de fixture.



**IMPORTANTE -- este documento fue reescrito tras una revisión independiente (`docs/review.md`).** La primera versión tenía defectos de datos, entrenamiento y evaluación que cambiaron las conclusiones:

1. **NDVI incorrecto en todos los tensores (C2)**: el offset BOA de Sentinel-2 se restaba por segunda vez (los composites de CDSE ya lo traen aplicado), dando NDVI de hasta ~10 (mediana ~1.8). Corregido, con máscara de píxeles oscuros y un chequeo de rango [-1, 1]; los tensores locales se repararon (`scripts/migrate_ndvi_offset.py`).
2. **Fuga espacio-temporal entre splits (C1)**: el evento de test `203187374` tenía la extensión de fuego pegada (0 km, mismos días) a un evento de val y a 1.4 km de otro -- los tres son el mismo complejo partido por el clustering. El calibrador, el peso del blend y el stacking se ajustan sobre val, así que el test no era retenido. Se rehizo el split por **grupos de eventos acoplados** (<= 10 km, <= 3 días).
3. **Desajuste objetivo de entrenamiento vs. evaluación (H4)**: el U-Net se entrenaba para predecir el fuego *activo* del día siguiente y se evaluaba (como el autómata celular) contra el área quemada *acumulada*; en el rollout no propagaba nada. Ahora se entrena con área quemada acumulada.
4. **Entradas sin normalizar (H1)** y **entrenamiento con días de padding sin fuego (H2)**: se agregó normalización fija y se recortan los eventos de entrenamiento igual que en la evaluación.
5. **El día 0 (ancla conocida) entraba en las métricas (H3)**: ahora se excluye por defecto.

Todo (datos, split, entrenamiento, calibración, backtest, ensambles, reporte) se volvió a correr. Los resultados anteriores se conservan en la sección 10, rotulados como reemplazados.

**Resultado honesto por adelantado**: con las correcciones, el U-Net calibrado supera al autómata celular sin calibrar en IoU/Dice en los dos eventos de test (por poco en uno, por mucho en el otro) y también en los dos de val; el blend elegido en val tiene el mejor Brier/ECE con IoU/Dice iguales a los del U-Net; el stacking no mejora. **Aun así, nada de esto es estadísticamente concluyente**: hay 2 eventos de test (el bootstrap solo remuestrea dos valores por modelo: su IC es el rango entre ambos eventos, y que dos IC no se solapen no es una prueba estadística), es una sola corrida con una semilla, y val (2 eventos) calibró al U-Net y eligió el peso del blend. Los números exactos, intervalos, mapas y análisis de fallas están en [`docs/results.md`](results.md) (generado desde `bench/results/`); este documento ya no los copia a mano para que no se desincronicen.

## 1. Credenciales y datos reales usados

Se confirmó al inicio de esta tarea que no había credenciales configuradas (`shared.config.get_settings()` fallaba nombrando las 5 variables faltantes). El usuario proveyó credenciales reales de NASA FIRMS, Copernicus CDS (ERA5-Land) y Copernicus Data Space (Sentinel-2); se verificaron con llamadas reales y no destructivas antes de usarlas. Nunca se comitearon (`.env`, en `.gitignore`).

## 2. Eventos reales usados

Fuente: detecciones activas de fuego NASA FIRMS, sensor VIIRS (satélite Suomi NPP), producto **VIIRS_SNPP_SP** (Standard Processing / archivo científico -- `VIIRS_SNPP_NRT` no cubre fechas de más de ~7 meses de antigüedad). Rango descargado: 2025-11-01 a 2026-03-31, bbox de `shared.config.study_area_bbox` (Biobío + Ñuble + Araucanía). Total: **8083 detecciones reales**, clusterizadas (`features.fire_state.clustering`, `spatial_eps_m`/`temporal_eps` por defecto) en **1441 eventos**.

**Criterio de selección** (documentado, no arbitrario): la inmensa mayoría de los 1441 eventos son ruido de 1-5 detecciones (un solo píxel VIIRS aislado). Se seleccionaron los eventos con **≥40 detecciones**, más **1 evento de diciembre incluido a mano** (`4199024687`, 25 detecciones) para no perder diversidad estacional -- sin él, la muestra habría cubierto solo noviembre, enero y marzo. Total: **15 eventos reales**.

| event_id | detecciones | fecha (real, sin padding) | bbox aprox. (WGS84, oeste,sur,este,norte) | split |
|---|---|---|---|---|
| 1277049523 | 1918 | 2026-01-17 a 2026-01-23 | -73.025,-36.931,-72.863,-36.637 | **val** |
| 3051657321 | 450 | 2026-01-18 a 2026-01-22 | -72.682,-36.811,-72.589,-36.725 | train |
| 2030469600 | 369 | 2026-01-18 a 2026-01-23 | -72.662,-37.393,-72.603,-37.242 | train |
| 2582836092 | 207 | 2026-01-18 a 2026-01-21 | -72.605,-38.018,-72.540,-37.944 | **test** |
| 12676775 | 159 | 2026-01-16 a 2026-01-20 | -72.613,-36.586,-72.557,-36.514 | train |
| 2728181583 | 143 | 2026-01-14 a 2026-01-18 | -72.646,-36.621,-72.607,-36.572 | train |
| 1262509675 | 105 | 2026-01-18 a 2026-01-20 | -72.430,-36.832,-72.387,-36.767 | train |
| 1013976330 | 96 | 2025-11-15 a 2025-11-23 | -71.170,-38.085,-71.151,-38.066 | **test** |
| 2320813586 | 77 | 2026-03-26 a 2026-03-29 | -72.200,-37.867,-72.175,-37.832 | train |
| 1214084367 | 75 | 2026-01-18 a 2026-01-21 | -72.568,-37.554,-72.532,-37.532 | train |
| 203187374 | 66 | 2026-01-15 a 2026-01-20 | -72.604,-36.623,-72.571,-36.602 | train |
| 775785671 | 50 | 2026-01-19 a 2026-01-20 | -73.056,-38.658,-73.006,-38.642 | train |
| 1482774057 | 47 | 2026-01-19 a 2026-01-21 | -72.625,-37.506,-72.595,-37.479 | train |
| 2970931921 | 42 | 2026-01-18 a 2026-01-27 | -71.294,-37.637,-71.262,-37.623 | **val** |
| 4199024687 | 25 | 2025-12-17 a 2025-12-18 | -72.204,-36.648,-72.184,-36.631 | train |

`event_id` 1277049523 (1918 detecciones, 17-23 de enero) es, por lejos, el incendio dominante de la muestra (hoy en val) -- corresponde al complejo de incendios de mediados de enero de 2026 en Biobío, el más severo mencionado en `CLAUDE.md`.

**Split train/val/test**: `features.dataset.split.split_events_grouped` (`pyrocast-features resplit`), por **grupos de eventos acoplados** -- eventos a <= 10 km (~una celda de ERA5-Land) y <= 3 días entre sí van siempre al mismo split; ver `data/processed/dataset/split_groups.json` y `docs/review.md` C1. Semilla 42 (la de por defecto, sin probar otras: elegir la semilla viendo resultados sería un grado de libertad del investigador) -> **train=11, val=2, test=2**. Consecuencia a notar: el evento dominante de la muestra (`1277049523`, 1918 detecciones) cayó en **val**, no en train, así que el U-Net no lo vio al entrenar.

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

- **Autómata celular (P7)**: `CellularAutomatonModel`, parámetros **HEURÍSTICOS POR DEFECTO** (`base_spread_prob=0.3`, `slope_coefficient=4.0`, `wind_coefficient=0.2`), **NO calibrados contra incendios reales de Chile**. `calibrate.py` solo soporta muestras sintéticas de un único paso simulado; extenderlo a trayectorias multi-día reales es una limitación preexistente (`docs/limitations.md`). El enunciado hablaba del "autómata celular calibrado (P7)": el evaluado **no** lo está.
- **U-Net (P9-P11)**: `SmallUNet`, entrenado **desde cero** (sin preentrenamiento en NDWS/Kaggle: sin esas credenciales) sobre los **11 eventos de train** del split sin fuga (`pyrocast-train finetune`, semilla 42, early stopping; entradas con normalización fija `v1`, objetivo = área quemada acumulada, eventos recortados a su primer día con fuego), calibrado con regresión isotónica sobre los **2 eventos de val** (`pyrocast-calibrate run --chile-val`). El historial de entrenamiento está en `runs/finetune_2026_v6/history.csv` (no versionado). El ECE ~0 post-calibración sobre val es estructural (se ajusta y evalúa sobre lo mismo, `docs/calibration.md`); `docs/results.md` sección 3 reporta además la calibración fuera de muestra sobre test.

**El U-Net vio solo 11 eventos reales**, órdenes de magnitud menos que NDWS (18.545 chips): cualquier resultado débil debe leerse a la luz de ese tamaño, no como evidencia de que la arquitectura sea inadecuada.

## 5. Backtest: métricas + intervalos de confianza

Todas las métricas (Brier, ECE, IoU, Dice; media y IC 95 % bootstrap de 1000 remuestreos sobre los valores por evento) y su desglose por evento están en [`docs/results.md`](results.md) secciones 1-2, generadas desde `bench/results/*.json`; los resultados crudos, con comando exacto y commit, en `bench/results/baseline.json` (CA), `unet.json`, `blend.json` y `stacking.json`. Los 4 modelos corren por el mismo `run_backtest`, sobre los mismos 2 eventos de test, con la misma semilla.

Convenciones heredadas: predicción y verdad **acumuladas** (`models/evaluation/backtest.py`), cada evento recortado a su primer día con fuego, umbral 0.5. Las métricas **excluyen el día 0** (el ancla conocida, que coincide con la verdad por construcción; `exclude_anchor_day=True`, hallazgo H3 del review): antes, un modelo que no predecía ninguna propagación igual obtenía un IoU > 0 por el ancla.

## 6. Comparación honesta

Con los datos, el split y el entrenamiento corregidos, el U-Net calibrado propaga el fuego y su IoU/Dice agregado es claramente mayor que el del autómata celular sin calibrar; además supera al CA en los cuatro eventos retenidos (2 de test, 2 de val; `docs/results.md` sección 2.3 y el análisis de la sección 5). En Brier/ECE el U-Net también queda mejor. Con n=2 el IC bootstrap es solo el rango entre los dos eventos (en IoU/Dice los del U-Net y los del CA no se solapan, pero eso no es una prueba estadística), y es una sola semilla: **es una señal consistente, no una demostración**.

**Cómo llegamos acá (honestidad sobre el camino)**: con el split original (con fuga, NDVI incorrecto, objetivo de entrenamiento activo) el U-Net "ganaba" en 1 de 2 eventos; al quitar la fuga y *antes* de corregir el objetivo de entrenamiento, el U-Net no predecía ninguna propagación (IoU ~0); al corregir el objetivo volvió a ser competitivo, y al corregir el NDVI mejoró más. Es decir, defectos distintos habían hecho parecer al U-Net mejor y peor de lo que es. Ver `docs/review.md`.

**Dónde sigue fallando el autómata celular** (hipótesis, no verificadas): subpredice los incendios grandes (`2582836092` en test, `1277049523` en val) porque con sus parámetros por defecto el frente avanza como mucho ~1 celda por día; el análisis por descriptores de evento está en `docs/results.md` sección 5.

## 7. Limitaciones de esta evaluación (además de `docs/limitations.md`)

- **n=2 eventos de test**: los IC bootstrap no sostienen afirmaciones de superioridad.
- **Autómata celular sin calibrar** (sección 4).
- **U-Net entrenado sin preentrenamiento, con 11 eventos**, sin evaluar la sensibilidad a la semilla (una sola corrida con semilla 42: con n=2 la variación entre semillas probablemente supera las diferencias entre modelos).
- **El split por grupos desplazó el evento dominante a val**: el U-Net no vio el mayor incendio de la temporada al entrenar (consecuencia de la semilla 42, no elegida).
- **Entrenamiento de un paso, evaluación en rollout**: el U-Net se entrena para predecir el día siguiente con el estado real del día anterior y se evalúa encadenando sus propias predicciones (sesgo de exposición); el calibrador isotónico también se ajusta a un paso.
- **ERA5-Land interpolado** de 9 km a 250 m + relleno de NaN costero con el promedio regional (hallazgo #5 de la sección 3).
- **Selección de eventos por umbral de detecciones (>= 40)**: sesga hacia incendios grandes/bien detectados.
- **Val sigue siendo el conjunto de ajuste** del calibrador, el peso del blend y el stacking.
- **Todos los eventos comparten el mismo episodio meteorológico de enero de 2026**: el split controla la fuga espacial/temporal directa, no la correlación climática regional.

## 8. Ensamble CA + U-Net (P13) y modelo por defecto en `serving/`

Implementación: `models/deep/ensemble.py`, `pyrocast-models backtest --model blend|stacking`. **Blend**: `(1-w)·CA + w·U-Net`, `w` elegido por Brier medio sobre los eventos de **val** (nunca test); el valor está en `bench/results/blend.json` (`config.weight_unet`). **Stacking**: regresión logística sobre `[logit(p_CA), logit(p_UNet)]` ajustada sobre val (coeficientes en `bench/results/stacking.json`).

Lectura honesta (criterio de aceptación: ¿mejora sobre el mejor individual?): ver `docs/results.md` sección 1. El **blend** mejora Brier/ECE respecto del mejor individual, con IoU/Dice prácticamente iguales: es una mejora de calibración, pequeña, y con n=2 y un peso elegido sobre solo 2 eventos de val no se puede distinguir de ruido; **no es una mejora consistente en todas las métricas, así que no es el default**. El **stacking** no mejora (con el split sin fuga ya no gana; en la versión con fuga ganaba, sección 10).

**Decisión de modelo por defecto en `serving/`: autómata celular (`CellularAutomatonModel`, parámetros por defecto)** -- decisión **provisional**, y ya no por falta de desempeño del U-Net sino por razones operativas y de evidencia:

1. **Operativo (bloqueante)**: el U-Net y su calibrador son archivos `.pt` no versionados (`runs/`, `*.pt` en `.gitignore`) y `serving/` no tiene un cargador de U-Net: no podría arrancar desde un checkout limpio ni reproducir la predicción. El CA no tiene artefactos.
2. **Evidencia insuficiente para promoverlo**: n=2 eventos de test (4 retenidos contando val, que está contaminado), una sola semilla, IC bootstrap sin valor inferencial con n=2, sin validación cruzada.
3. **Riesgo asimétrico**: las probabilidades del U-Net dependen de un calibrador ajustado sobre 2 eventos.

Costo asumido: el servido usa el modelo que los datos indican como el peor de los individuales, con probabilidades sin calibrar (puntaje relativo). **Para cambiar el default** (en `docs/backlog.md`): versionar checkpoint + calibrador (p. ej. como asset de un release, con su huella sha256) y agregar un cargador en `serving/api/model_registry.py`; repetir con varias semillas; más eventos con un val independiente.

## 9. Reproducibilidad

Comandos exactos y commit de git usados, embebidos en cada resultado (`bench/results/baseline.json`, `unet.json`, `blend.json`, `stacking.json`, campos `"command"` y `"git_commit"`):

```
pyrocast-models backtest --n-bootstrap 1000 --seed 42
pyrocast-models backtest --model unet --checkpoint runs/finetune_2026_v6/best.pt --n-bootstrap 1000 --seed 42
pyrocast-models backtest --model blend --checkpoint runs/finetune_2026_v6/best.pt --n-bootstrap 1000 --seed 42
pyrocast-models backtest --model stacking --checkpoint runs/finetune_2026_v6/best.pt --n-bootstrap 1000 --seed 42
```

Para reproducir de punta a punta con las mismas credenciales: ingerir FIRMS/DEM/WorldCover/ERA5/Sentinel-2 para las fechas y bboxes de la sección 2, `pyrocast-features build-dataset --event-ids <ids>` por evento, `pyrocast-features resplit` (split por grupos), `pyrocast-train finetune --run-dir runs/finetune_2026_v6 --seed 42`, `pyrocast-calibrate run --checkpoint runs/finetune_2026_v3/best.pt --chile-val`, y los comandos de arriba.

## 10. Resultados anteriores con el split con fuga (REEMPLAZADOS)

Se conservan por transparencia (CLAUDE.md: no esconder ni suavizar resultados). **No usar**: el split tenía fuga val/test (ver arriba). Archivados en `bench/results/archive/2026-10-01_split_with_leakage/` (JSON, `splits.json` de entonces). Agregado sobre los 2 eventos de test de entonces (`203187374`, `2582836092`), media [IC 95 %]:

| métrica | CA | U-Net | blend (w=0.4) | stacking |
|---|---|---|---|---|
| IoU | 0.319 [0.288, 0.351] | 0.249 [0.055, 0.442] | 0.334 [0.284, 0.384] | 0.513 [0.462, 0.564] |
| Dice | 0.483 [0.447, 0.520] | 0.359 [0.104, 0.613] | 0.499 [0.442, 0.555] | 0.677 [0.632, 0.722] |
| Brier | 0.092 [0.082, 0.103] | 0.070 [0.044, 0.096] | 0.075 [0.058, 0.092] | 0.058 [0.056, 0.061] |
| ECE | 0.091 [0.082, 0.100] | 0.064 [0.035, 0.094] | 0.074 [0.058, 0.090] | 0.038 [0.020, 0.055] |

Con ese split (y NDVI incorrecto, objetivo de entrenamiento activo, día 0 incluido) el U-Net parecía ganar en un evento y el stacking en todo. Parte de esa ventaja venía de la fuga (el evento de test `203187374` estaba pegado a dos eventos de val) y las métricas incluían el día 0.
