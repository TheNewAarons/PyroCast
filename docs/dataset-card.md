# Dataset card: tensores espaciotemporales por evento de incendio

> **Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.**

## Qué es

Cada evento de incendio (`features/fire_state/clustering.py`) se
ensambla en UN tensor 4D `(día, canal, alto, ancho)`, persistido como un
array Zarr en `data/processed/dataset/event_NNNN.zarr` (`NNNN` = id de
evento, un hash determinista del contenido del evento — no un contador,
ver `docs/decisions.md`). El tensor lleva coords `x`/`y` (centros de
píxel en el CRS del evento) y atributos `crs`, `transform`,
`resolution_m`, `event_id` — un Zarr abierto por sí solo, sin ningún
otro archivo, ya dice a qué ubicación real y a qué evento corresponde
cada píxel. Los metadatos del evento (bbox recortado en WGS84, fecha de
inicio/fin, id) se persisten además en la tabla `fire_event` de PostGIS,
enlazados por la columna `firms_event_id` (upsert: re-correr
`build-dataset` para el mismo evento actualiza su fila, no la duplica).

## Canales (orden fijo, `features.dataset.assemble.CHANNEL_ORDER`)

| # | Canal | Tipo | Unidad/rango | Fuente |
|---|---|---|---|---|
| 0 | `elevation` | estático | m | Copernicus DEM GLO-30 |
| 1 | `slope_deg` | estático | grados [0,90] | `features/terrain` (Horn 1981) |
| 2 | `aspect_deg` | estático | grados [0,360), -1=plano | `features/terrain` |
| 3 | `wind_u` | dinámico (diario) | m/s | ERA5-Land `u10` |
| 4 | `wind_v` | dinámico (diario) | m/s | ERA5-Land `v10` |
| 5 | `temperature` | dinámico (diario) | K | ERA5-Land `t2m` |
| 6 | `relative_humidity` | dinámico (diario) | % [0,100] | derivado (Magnus-Tetens) |
| 7 | `precipitation` | dinámico (diario) | m | ERA5-Land `tp` |
| 8 | `ndvi` | dinámico (mensual, repetido por día) | [-1,1] | Sentinel-2 L2A, composite mensual más cercano |
| 9 | `fuel_type` | estático | código entero (ver `ingestion/worldcover/fuel_type.py`) | ESA WorldCover |
| 10 | `fire_mask` | dinámico (diario) | {0.0, 1.0} | `features/fire_state` (variable objetivo/histórica) |

Los canales estáticos se repiten idénticos en cada día del tensor — es
una decisión de forma deliberada (un solo array 4D uniforme, no un dict
de arrays de dimensión mixta), no una afirmación de que la elevación
cambia día a día.

**`fire_mask` es extensión ACTIVA de fuego por día, no superficie
quemada acumulada** — cada máscara diaria es independiente de las
anteriores (una celda que ardió ayer y no hoy vuelve a `0.0`). Reconstruir
la superficie quemada acumulada (unión de las máscaras hasta la fecha) es
responsabilidad de quien consuma este dataset, no de este canal. Ver
`docs/fire-events.md`.

## Alineación temporal

Cada evento cubre desde `event.start_date - 5 días` (contexto previo al
modelo, antes de la primera detección) hasta `event.end_date` (la
última detección del evento) — `DEFAULT_PRE_EVENT_PADDING_DAYS = 5` en
`features/dataset/pipeline.py`, una heurística sin calibrar contra
incendios reales de Chile (igual que los defaults de
`features/fire_state`, ver `docs/fire-events.md`).

Un canal de clima para un día sin archivo ya procesado en disco se
rellena con `NaN` **por campo, no todo-o-nada**: si de los 5 campos de
clima de un día solo falta `precipitation`, los otros 4 (`wind_u`,
`wind_v`, `temperature`, `relative_humidity`) se resamplean igual — antes
de la revisión final del 2026-09-27, un solo campo faltante descartaba
los 5 a NaN. NDVI usa el composite mensual más cercano dentro de un
máximo de 3 meses de distancia (`max_month_distance` en
`_nearest_month_path`) — más allá de eso, NaN en vez de presentar un
composite de años de antigüedad como si fuera vigente. **Precondición
operativa**: para que el padding tenga datos reales, ingerir un rango de
fechas que empiece al menos 5 días antes del primer evento de interés,
no solo las fechas del evento.

**Días de padding y el rango de fechas pedido al CLI**: si
`pyrocast-features build-dataset --start/--end` recorta el historial de
detecciones ANTES de que el incendio realmente empezara (p. ej.
`--start` cae después del inicio real del fuego), los días de padding
calculados desde `event.start_date` (la primera detección DENTRO del
rango pedido, no la primera detección real) llevan `fire_mask=0.0` —
un falso negativo en la variable objetivo para días en que el fuego
probablemente ya ardía. Pedir un rango que empiece bien antes del
período de interés reduce este riesgo; no está validado
automáticamente.

## Alineación espacial: grilla por evento, no la grilla de estudio completa

Cada evento usa su PROPIA `WorkGrid` (`features/grid/`), construida a
partir del bbox de sus detecciones más un buffer de CONTEXTO
(`DEFAULT_CONTEXT_BUFFER_M=2000m`) — deliberadamente DISTINTO del radio
de la máscara de fuego (`DEFAULT_BUFFER_M=375m` de `features/fire_state`,
ver `docs/fire-events.md` para la corrección de esa cifra: es ~3.1x el
área nominal de un píxel VIIRS, deliberado, no calibrado). Antes de la
revisión final del 2026-09-27 se usaba el MISMO buffer (375m) para
ambos propósitos: un evento de una sola detección terminaba con un
tensor de 4x4 píxeles, casi sin margen alrededor del fuego para que un
modelo de propagación tuviera hacia dónde propagar. `context_buffer_m`
tampoco está calibrado — es una heurística elegida para producir
tensores de un tamaño razonable, no un valor derivado de la velocidad de
propagación esperada. No es la grilla de todo el área de estudio: el
"bbox recortado" persistido en `fire_event.bbox` es exactamente ese bbox
por evento (WGS84, `west,south,east,north` — no el CRS proyectado del
evento, corregido en la misma revisión), y el tensor cubre solo esa
área — mucho más chico que Biobío+Ñuble+Araucanía completos.

Cada canal se resamplea (`features/dataset/resample.py`) a esa grilla
desde la salida YA PROCESADA de su pipeline de origen (DEM/terreno vía
`pyrocast-ingest dem`, clima vía `pyrocast-ingest era5`, NDVI vía
`pyrocast-ingest sentinel2`, tipo de combustible vía `pyrocast-ingest
worldcover`) — bilineal para los continuos, nearest para `fuel_type`
(categórico) y para `aspect_deg` (magnitud CIRCULAR, no categórica —
interpolar bilinealmente 358° y 2° da ~180°, un error de 180 grados;
corregido en la revisión final del 2026-09-27, `ingestion/worldcover`
usa el mismo criterio nearest por la razón categórica, distinta pero con
la misma solución).

## Persistencia y metadatos

- Zarr: `data/processed/dataset/event_NNNN.zarr`, un Dataset con una
  única variable `fire_event_tensor`, dims `(day, channel, y, x)`, coords
  `day` (ISO-8601 string) y `channel` (`CHANNEL_ORDER`).
- PostGIS: una fila en `fire_event` por evento — `bbox` (texto
  `west,south,east,north`), `start_date`/`end_date`, `source="firms_cluster"`,
  `geom` (rectángulo del bbox, SRID 4326).
- Split: `data/processed/dataset/splits.json` — `{"train": [...ids],
  "val": [...ids], "test": [...ids]}`, `features/dataset/split.py`,
  semilla fija (`seed=42` por defecto), 70/15/15, **por evento completo**
  (nunca por píxel ni por día dentro de un evento — evita fuga de datos:
  días consecutivos del mismo incendio son casi idénticos, entrenar con
  un día y evaluar con el siguiente del MISMO evento mediría
  memorización, no generalización). Con menos de 3 eventos, todo va a
  `train` (documentado, no un `val`/`test` vacío por accidente); a partir
  de 3, cada split recibe al menos 1 evento — el 70/15/15 es exacto solo
  asintóticamente, no para conteos chicos.
- **Reproducibilidad de `random.Random(seed).shuffle`**: garantizada
  dentro de una misma versión de Python (`requires-python
  ">=3.12,<3.13"` en la raíz del workspace) — Python no garantiza que el
  algoritmo de `random` sea estable ENTRE versiones mayores. Si esa
  restricción se relaja en el futuro, un split ya guardado podría dejar
  de ser reproducible con una versión de Python distinta a la que lo
  generó.

## Cuántos eventos hay, distribución, desbalance

**Sin datos reales todavía**: este dataset-card se escribe junto con la
implementación, antes de correr `pyrocast-features build-dataset` contra
el historial real de detecciones FIRMS de Chile 2025-2026. No hay
todavía una cuenta real de eventos, su distribución temporal/geográfica,
ni evidencia empírica de desbalance (pocos eventos grandes vs. muchos
chicos) — completar esta sección con las cifras reales es tarea de
`models/evaluation/` (backtesting, sin implementar) o de una corrida real
de este pipeline, no de este bootstrap. Documentado explícitamente en vez
de inventar cifras (ver el mandato de honestidad de CLAUDE.md).

## Limitaciones conocidas (ver también `docs/limitations.md`)

- Convención de un único DEM/estudio de área a la vez
  (`features/dataset/pipeline.py::_single_file`) — un segundo estudio de
  área simultáneo rompería la resolución de `elevation_path`.
- Ningún canal está anclado a la grilla canónica de TODO el proyecto
  (`features/grid/` con el bbox de estudio completo) — cada evento tiene
  su propia grilla recortada; comparar tensores de dos eventos distintos
  píxel a píxel no es válido sin un paso de reproyección adicional.
- `features/dataset/` no re-ingiere nada: si el rango pedido no fue
  previamente cubierto por `pyrocast-ingest`, los canales de ese rango
  quedan en `NaN`, no en un error explícito — un dataset con muchos NaN
  puede pasar desapercibido si no se inspecciona.
- El encadenamiento del clustering de eventos no tiene límite
  temporal/espacial acotado por evento (ver `docs/fire-events.md`) — un
  evento inusualmente largo o disperso producirá un tensor
  proporcionalmente grande o con muchos días de padding/hueco.
- **`pyrocast-features build-dataset` no es transaccional entre
  eventos**: si el evento N de M falla (p. ej. la convención de un único
  DEM se viola, o falta un raster requerido), los eventos `1..N-1` ya
  quedan con su Zarr escrito y su fila de PostGIS insertada/actualizada,
  pero `splits.json` (escrito solo al final) no existe todavía. Volver a
  correr el comando reintenta desde el principio — el upsert por
  `firms_event_id` hace que esto sea seguro para PostGIS (no duplica
  filas), y `save_event_to_zarr` sobrescribe (`mode="w"`), pero no hay
  manejo explícito de fallos parciales ni un resumen de qué eventos
  fallaron. Encontrado en la revisión final del 2026-09-27.
- `features/dataset/firms_loader.py` deduplica detecciones por
  `(fecha/hora, coordenadas redondeadas a 6 decimales, satélite)` —
  necesario porque re-ingerir el mismo rango con FIRMS (o cubrir el
  mismo período con dos satélites VIIRS distintos, p. ej. NOAA-20 y
  SNPP) puede escribir la misma detección física más de una vez en el
  parquet crudo. La deduplicación asume que dos detecciones reales
  distintas casi nunca coinciden en los tres campos a la vez — no
  verificado contra el historial real de FIRMS de Chile.
- Los canales ESTÁTICOS (`elevation`, `slope_deg`, `aspect_deg`,
  `fuel_type`) también pueden quedar en `NaN` donde la grilla del evento
  se extiende más allá de la cobertura del raster de origen — no solo
  los canales dinámicos por fecha. Esto es más probable para eventos
  cerca del borde del área de estudio.
