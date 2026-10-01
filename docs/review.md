# Revisión independiente del proyecto (2026-10-01)

> **Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.**

Revisión crítica del repositorio completo, hecha con la actitud de un revisor externo: se buscó activamente lo que podía estar mal, no lo que confirmaba lo ya documentado. **Resultado: 2 hallazgos críticos y 4 altos, todos corregidos en esta revisión; ninguno crítico queda abierto.** Los hallazgos medios y bajos están en [`docs/backlog.md`](backlog.md).

**Los resultados publicados cambiaron.** Los cuatro defectos de datos/entrenamiento/evaluación (C1, C2, H1-H4) alteraban las conclusiones del backtest: el U-Net parecía mejor y peor de lo que es según la etapa. Todo se reentrenó y reevaluó; los resultados anteriores se conservan en `bench/results/archive/2026-10-01_split_with_leakage/` y en `docs/backtest-2026.md` sección 10.

## 1. Método

- Suite completa, lint, typecheck, `pip-audit` y cobertura por paquete.
- Lectura del código buscando: fallas de red que produzcan un número en vez de un error, inconsistencias de unidades entre módulos, reproyecciones dobles o faltantes, fuga de datos entre splits y credenciales o rutas absolutas hardcodeadas. Además de leer, se **midió contra los datos reales** (extensiones y fechas de los eventos, rangos de cada canal del tensor, composites crudos de Sentinel-2).
- Verificación de cada afirmación del README y de `docs/results.md` contra `bench/results/` y el código que las produjo (se volvieron a correr los backtests: mismos números).
- Presencia del aviso de "herramienta de investigación" donde `CLAUDE.md` lo exige.

## 2. Hallazgos

| ID | Severidad | Hallazgo | Dónde | Estado |
|---|---|---|---|---|
| C1 | **Crítico** | Fuga espacio-temporal entre val y test | `features/dataset/split.py:31` (`split_events` por evento suelto), `features/cli.py:109` | **Corregido** |
| C2 | **Crítico** | NDVI incorrecto en todos los tensores (offset BOA restado dos veces) | `features/vegetation/ndvi.py:48,64` | **Corregido** |
| H1 | Alto | Entradas del U-Net sin normalizar (escalas de 1e-3 a 1e3; `fuel_type` como número ordinal) | `models/deep/unet.py:105`, `models/deep/normalization.py` | **Corregido** |
| H2 | Alto | Entrenamiento con días de padding sin fuga que la evaluación recortaba | `models/deep/train.py:242` | **Corregido** |
| H3 | Alto | El día 0 (ancla conocida) entraba en las métricas del backtest | `models/evaluation/backtest.py:108,126` | **Corregido** |
| H4 | Alto | Desajuste entre el objetivo de entrenamiento (fuego activo) y la evaluación (área quemada acumulada) | `models/deep/train.py:68`, `models/deep/calibration.py:245` | **Corregido** |
| M1-M8, L1-L8 | Medio / Bajo | Ver `docs/backlog.md` | — | Abiertos (backlog) |

### C1 — Fuga espacio-temporal entre splits (crítico)

**Qué.** El split era "por evento", pero el clustering FIRMS puede partir un mismo complejo de incendios en varios eventos contiguos, que el reparto aleatorio mandaba a splits distintos. Medido sobre los tensores reales (extensión de las celdas con fuego y días con fuego): el evento de **test** `203187374` tenía la extensión de fuego a **0 km** (pegada) del evento de **val** `2728181583` y a 1.4 km de `12676775`, con días de fuego solapados. Los tres son el mismo complejo.

**Por qué es crítico.** Val no es solo "validación": ajusta el calibrador isotónico, elige el peso del blend, ajusta el stacking y decide el early stopping. Así, tres de las cuatro filas del backtest se ajustaron sobre datos casi idénticos a los de test; el test no era retenido, y el ensamble "stacking gana en todo" era en parte esa contaminación.

**Corrección.** `split_events_grouped` agrupa eventos acoplados (<= 10 km, <= 3 días: ~una celda de ERA5-Land, que comparten clima de entrada, y un día más que el `temporal_eps` del clustering) y reparte grupos, nunca un grupo entre dos splits; exige >= 3 grupos independientes y falla ruidosamente si no (no degrada en silencio). `find_split_leakage` audita cualquier split y el reporte la muestra (`docs/results.md` sección 7, "Auditoría de fuga entre splits"; hoy: 0 pares acoplados en splits distintos). `pyrocast-features resplit` rehízo el split del dataset ya construido (guardó el anterior en `splits.previous.json` y las fugas en `split_groups.json`). Tests: `features/tests/test_dataset_split.py`.

**Lo que NO resuelve** (documentado en `limitations.md`): casi todos los eventos son del mismo episodio meteorológico de enero de 2026, así que queda correlación climática regional; y con la semilla 42 el evento dominante (`1277049523`) cayó en val.

### C2 — NDVI incorrecto (crítico)

**Qué.** `compute_ndvi_masked` restaba un offset BOA de -1000 a los DN, suponiendo que el composite lo traía. Los composites reales de openEO/CDSE **ya lo traen aplicado** (los 15 composites descargados tienen DN mínimos entre -98 y 36 (9 de 15 con valores negativos), imposibles con un +1000 presente; medianas de RED entre 267 y 1012 y de NIR entre 2316 y 3159). Restar -1000 de nuevo duplicaba la corrección: el NDVI de los tensores tenía mediana ~1.8 y máximos de hasta ~10 en una magnitud acotada a [-1, 1]. Era un caso de "inconsistencia de unidades entre módulos" que ningún test de fixture podía ver (los tests usaban DN inventados que *asumían* el offset), y la "corrección" de una revisión anterior (que *agregó* el -1000) era el error.

**Cómo se confirmó.** Se reprodujo **exactamente** (diferencia máxima 0.0) el NDVI guardado de los 15 eventos aplicando el offset duplicado a los composites crudos: la causa es esa, no otra.

**Corrección.** Offset por defecto 0.0 (sigue siendo parámetro), píxeles oscuros (RED o NIR <= 0, o suma < 100 DN: agua, sombra) como nodata, y un chequeo que **rechaza** cualquier NDVI fuera de [-1, 1] en vez de escribirlo. Los tensores locales se repararon sin descargar nada (`scripts/migrate_ndvi_offset.py`, que identifica el composite de cada evento reproduciendo el valor viejo y se niega a migrar a ciegas); auditoría en `bench/results/ndvi_fix_audit.json`: NDVI corregido con mediana 0.4-0.7 y rango [-0.55, 0.93].

**Impacto.** Solo el U-Net usa el canal NDVI (el autómata celular y `serving/` no). Es plausible que parte del desempeño inicial del U-Net estuviera afectado; tras corregirlo mejoró.

### H1 — Entradas sin normalizar

El U-Net recibía el tensor crudo: elevación en cientos de metros, temperatura ~290 K, precipitación ~1e-3 m, y `fuel_type` como código de clase (1..99) tratado como número ordinal, con `batch_size=1` y sin estadísticas de entrada. Se agregó una normalización **fija** (constantes físicas, no estadísticos ajustados a los datos: nada que filtrar entre splits; `fuel_type` pasa a su flamabilidad). El modo se guarda en el checkpoint (`input_norm`); los checkpoints anteriores se cargan como antes.

### H2 — Entrenamiento y evaluación sobre distintos eventos

La evaluación recortaba cada evento a su primer día con fuego (el padding de 5 días no tiene fuego), pero el entrenamiento usaba los eventos completos, incluyendo el par "sin fuego → aparece fuego". Ahora ambos usan la misma carga (`models/events.py`).

### H3 — El día 0 en las métricas

El día 0 es el estado conocido y coincide con la verdad por construcción; incluirlo daba IoU > 0 a un modelo que no predecía **ninguna** propagación (el U-Net "malo" de una etapa intermedia obtenía 0.055 solo por el ancla). `run_backtest` lo excluye por defecto (`exclude_anchor_day=True`) y la confiabilidad del reporte también.

### H4 — Objetivo de entrenamiento vs. evaluación (el más sutil)

El backtest evalúa **área quemada acumulada** ("¿ha ardido esta celda alguna vez hasta el día d?"), igual que predice el autómata celular. Pero el U-Net se entrenaba para predecir el fuego **activo** del día siguiente, que desaparece entre pasadas del satélite: aprendía "el fuego se apaga" y en el rollout del backtest no propagaba nada (0 celdas con probabilidad >= 0.5, IoU ~0). Ahora se entrena con área quemada acumulada como entrada y objetivo, y `CalibratedUNet` fuerza que la probabilidad acumulada no decrezca.

## 3. Cómo cambiaron los resultados, etapa por etapa

IoU medio en test (n=2). Las etapas 0 y 4 están en `bench/results/` (la 0, en `archive/`); las intermedias se copian de los registros de esta sesión y **no se archivaron como JSON**.

| etapa | cambio | CA | U-Net | blend | stacking |
|---|---|---|---|---|---|
| 0 | original (split con fuga, NDVI incorrecto, objetivo activo, día 0 incluido) | 0.319 | 0.249 | 0.334 | 0.513 |
| 1 | + split por grupos (C1) | 0.398 | 0.033 | 0.033 | 0.486 |
| 2 | + normalización, entrenamiento recortado, sin día 0 (H1-H3) | 0.380 | 0.000 | 0.000 | 0.513 |
| 3 | + objetivo acumulado (H4) | 0.380 | 0.450 | 0.454 | 0.388 |
| 4 | + NDVI correcto (C2) — **actual** | 0.380 | 0.563 | 0.562 | 0.527 |

Lectura: el U-Net pasó de "no propaga" a "supera al autómata celular en los 4 eventos retenidos"; el stacking, que ganaba con la fuga, ya no. **Sigue sin ser concluyente**: n=2 eventos de test, una semilla, val contaminado por el calibrador. La decisión de modelo por defecto en `serving/` (autómata celular) se mantiene como **provisional**, ahora por razones operativas (el checkpoint no está versionado y `serving/` no tiene cargador) y de evidencia, no por mal desempeño del U-Net (`docs/backtest-2026.md` sección 8).

Moraleja metodológica: cada defecto movía el resultado en una dirección distinta; sin corregirlos todos, cualquier conclusión sobre "qué modelo es mejor" habría sido accidental.

## 4. Verificaciones

| Verificación | Resultado |
|---|---|
| `make test` | 497 passed, 3 skipped (los 3 requieren un PostgreSQL real: `shared/tests/test_db_schema.py:58`, `features/tests/test_dataset_db.py:27`, `models/tests/test_evaluation_db.py:36`) |
| `make lint` (ruff) | limpio |
| `make typecheck` (`mypy --strict` en shared, features, serving) | limpio, 42 archivos |
| `make audit` (`pip-audit` sobre `uv.lock`) | sin vulnerabilidades conocidas |
| Cobertura de `src/` (líneas) | shared 97 %, ingestion 96 %, features 96 %, models 95 %, serving 97 % |
| Backtests re-ejecutados al final | idénticos a los publicados (solo cambia el commit registrado) |

Cobertura baja a notar (backlog M3/M4): `features/dataset/db.py` 31 % y `models/cli.py` 68 % (las ramas de ensamble del backtest), más el código de PostGIS nunca ejercitado contra una base real.

**Fallas silenciosas → número**: no se encontró ninguna en la ruta de servido (todo dato faltante es un error 422/503 explícito). En el entrenamiento, un `nan_to_num` a 0 se aplica a cualquier canal (backlog M8); medido, solo afecta un NDVI residual de un evento de train y el `fuel_type` del evento de val `12676775`. Las descargas de DEM/WorldCover toleraban cualquier error como si fuera un 404 (ya corregido en el endurecimiento previo).

**Unidades**: viento en m/s (ERA5 u10/v10, NDWS desde `vs`/`th` con la convención "desde" bien invertida), temperatura en K, humedad relativa en % (0-100) en ERA5 y en NDWS, precipitación en m (NDWS `pr` mm→m), elevación en m, resolución en m: consistentes. La única inconsistencia real fue C2.

**Reproyecciones**: ninguna faltante. Hay remuestreos de calidad discutible y una doble interpolación (backlog M1).

**Credenciales y rutas**: un test escanea los archivos versionados (claves hex/UUID/AWS/asignaciones) y pasa; `.env` no está versionado ni en el historial de git; no hay rutas absolutas hardcodeadas (solo `/tmp` en el target `audit` del Makefile, backlog L2).

**README y `docs/results.md`**: `docs/results.md` y el bloque de resultados del README se generan desde `bench/results/`; se verificó que sus números coinciden con los JSON y que cambian si cambian los JSON. Las cifras del incendio (>34.000 ha, 21 fallecidos, >4.100 viviendas) coinciden con el reporte de la ONU citado. Las afirmaciones del README sobre `make demo`, `make run-ca`, `make calibrate`, `make report` y el modo `production` de la API se ejecutaron o están cubiertas por tests. Se corrigieron referencias desactualizadas (`finetune_2026_v2/v3`, valores del blend, el texto de `docs/dataset-card.md` sobre el split).

**Aviso de investigación**: presente en el README (primeras líneas), en cada `docs/*.md` (se agregó a 9 que no lo tenían; un test lo exige), en `docs/results.md`/`.html`, en la descripción de OpenAPI, en cada respuesta de `/predict` y `/active-fires` (`research_tool: true` + `disclaimer`) y como encabezado permanente del mapa web.

## 5. Qué no se pudo verificar

- **Ningún flujo en un navegador real**: la extensión de Chrome no conectó; el mapa se probó por tests de servidor y contra la API real, no con clics.
- **La imagen Docker**: no se construyó (el job de CI la construye).
- **La persistencia en PostGIS**: no hay una base disponible aquí.
- **Errores reales de cuota de CDS/openEO**: se cubrieron con fakes.
- **Variación entre semillas** de entrenamiento: una sola corrida (backlog M6).
