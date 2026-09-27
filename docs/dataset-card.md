# Dataset card: tensores espaciotemporales por evento de incendio

## Qué es

Cada evento de incendio (`features/fire_state/clustering.py`) se
ensambla en UN tensor 4D `(día, canal, alto, ancho)`, persistido como un
array Zarr en `data/processed/dataset/event_NNNN.zarr` (`NNNN` = id de
evento, un hash determinista del contenido del evento — no un contador,
ver `docs/decisions.md`). Los metadatos del evento (bbox recortado, fecha
de inicio/fin, id) se persisten además en la tabla `fire_event` de
PostGIS.

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

Un canal de clima o NDVI para un día sin archivo ya procesado en disco
(p. ej. si el rango ingerido con `pyrocast-ingest` no cubre los días de
padding, solo los días del evento en sí) se rellena con `NaN` — nunca se
fabrica un valor. **Precondición operativa**: para que el padding tenga
datos reales, ingerir un rango de fechas que empiece al menos 5 días
antes del primer evento de interés, no solo las fechas del evento.

## Alineación espacial: grilla por evento, no la grilla de estudio completa

Cada evento usa su PROPIA `WorkGrid` (`features/grid/`), construida a
partir del bbox de sus detecciones + un buffer (`DEFAULT_BUFFER_M=375m`,
radio — ver `docs/fire-events.md` para la corrección de esta cifra: es
~3.1x el área nominal de un píxel VIIRS, deliberado, no calibrado) — no
la grilla de todo el área de estudio. El "bbox recortado" persistido en
`fire_event.bbox` es exactamente ese bbox por evento (WGS84,
`west,south,east,north`), y el tensor cubre solo esa área — mucho más
chico que Biobío+Ñuble+Araucanía completos.

Cada canal se resamplea (`features/dataset/resample.py`) a esa grilla
desde la salida YA PROCESADA de su pipeline de origen (DEM/terreno vía
`pyrocast-ingest dem`, clima vía `pyrocast-ingest era5`, NDVI vía
`pyrocast-ingest sentinel2`, tipo de combustible vía `pyrocast-ingest
worldcover`) — bilineal para los continuos, nearest para `fuel_type`
(categórico, mismo criterio que en `ingestion/worldcover`).

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
  memorización, no generalización).

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
