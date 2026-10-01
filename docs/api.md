# API de PyroCast (`serving/api/`)

*Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.*

FastAPI. Levantar con `make serve` (127.0.0.1:8000; `/docs` solo existe con `ENVIRONMENT=development`, que `make serve` y `make demo` fijan; en el modo `production` por defecto no hay `/docs`, `/redoc` ni `/openapi.json`). Requiere `.env` (la app falla al iniciar con `ConfigurationError` si faltan variables) y los datos procesados en `DATA_PROCESSED_DIR` (`make ingest-terrain`, `make ingest-weather`, `make ingest-vegetation`).

Las respuestas de `/predict` y `/active-fires` son GeoJSON `FeatureCollection` válidos con campos extra ("foreign members", RFC 7946 §6.1), así que Leaflet puede usarlos con `L.geoJSON` directamente. Todas incluyen:

| campo | valor |
|---|---|
| `research_tool` | `true` |
| `limitations` | `"docs/limitations.md"` |
| `disclaimer` | el aviso de herramienta de investigación |

## `GET /` — mapa web

Página Jinja2 (`serving/web/templates/index.html`) con Leaflet vía CDN (con SRI) y JS propio sin build step (`serving/web/static/app.js`, `app.css`, servidos en `/static`). Con `make serve`, abrir <http://localhost:8000/>.

- Mapa centrado en el área de estudio, base OpenStreetMap, detecciones activas de `/active-fires` (últimos 2 días).
- Clic en el mapa = punto de ignición; fecha y horizonte en el panel; "Predecir" llama a `POST /predict`.
- Celdas coloreadas por probabilidad acumulada (5 tramos, leyenda en el panel; < 5 % sin color), deslizador para moverse entre días.
- El aviso de herramienta de investigación es un encabezado fijo del layout, sin botón de cierre.
- Los errores del backend (`error.code` / `error.message`, o validación de FastAPI) se muestran en un banner `role="alert"`; ante un error se conserva la predicción anterior, nunca se dibuja una nueva. Las detecciones activas son opcionales: si fallan, solo se avisa en el panel.
- Las URLs y el rango de clima procesado los inyecta el HTML en atributos `data-*` de `#app`; la fecha por defecto es el primer día con clima.
- Texto del backend siempre con `textContent` (sin HTML). Sin JavaScript solo se ve el aviso y el mensaje `noscript`.

## `GET /healthz`

`{"status": "ok"}`. No toca modelo ni datos.

## `POST /predict`

Cuerpo:

```json
{"lat": -36.61, "lon": -72.59, "date": "2026-01-15", "horizon_days": 3}
```

| campo | regla |
|---|---|
| `lat`, `lon` | WGS84, dentro del bbox de estudio (`shared.config.study_area_bbox`) |
| `date` | día de la ignición (día 0, estado conocido). Debe haber clima procesado para `date` y los `horizon_days` siguientes |
| `horizon_days` | 1 a 7 (por defecto 3) |

Qué hace: ajusta el punto a la celda de 250 m que lo contiene (se usa el centro de la celda), arma el tensor de features con el mismo pipeline de `features/dataset/` (terreno y tipo de combustible de capas estáticas ya procesadas; clima del día buscado por fecha; NDVI del mes más cercano, máx. 3 meses), marca solo esa celda como fuego en el día 0 y ejecuta el modelo por defecto.

Respuesta 200 (resumen):

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {"type": "Polygon", "coordinates": [[[lon, lat], "..."]]},
    "properties": {"row": 3, "col": 7, "probability_by_day": [1.0, 1.0, 1.0]}
  }],
  "research_tool": true, "limitations": "docs/limitations.md", "disclaimer": "...",
  "model": {"name": "cellular_automata", "calibrated": false},
  "request": {"lat": -36.61, "lon": -72.59, "snapped_lat": -36.6101, "snapped_lon": -72.5902,
              "date": "2026-01-15", "horizon_days": 3},
  "grid": {"crs": "EPSG:32719", "resolution_m": 250.0, "rows": 17, "cols": 19, "bbox_wgs84": [...]},
  "days": [{"day": 1, "date": "2026-01-16"}, "..."],
  "warnings": ["ndvi: sin datos en ...; el modelo cellular_automata no usa este canal, ..."],
  "cached": false
}
```

- `probability_by_day[i]` corresponde a `days[i]`. Es **acumulada**: "¿ha ardido esta celda alguna vez hasta ese día?" (misma convención que el backtest, `docs/backtest-2026.md`). El día 0 (ignición) no se devuelve.
- `calibrated: false`: el autómata celular no está calibrado; las probabilidades son un puntaje relativo, no una probabilidad calibrada (`docs/limitations.md`).
- `warnings`: canales sin datos que el modelo **no** usa. Nunca se ocultan.
- `cached: true` si se sirvió de caché (ver abajo).
- Solo se devuelve la grilla en GeoJSON; la opción de URL a un raster no está implementada.

### Errores

Todos tienen la forma `{"error": {"code", "message", "details"}}`. Nunca se devuelve una predicción construida con datos faltantes.

| HTTP | `code` | cuándo |
|---|---|---|
| 422 | `outside_study_area` | el punto cae fuera del bbox de estudio |
| 422 | `weather_unavailable` | falta clima requerido por el modelo para `date` o algún día del horizonte. `details.missing_days` y `details.available_range` lo dicen. ERA5-Land es reanálisis histórico, **no un pronóstico**: una fecha reciente o futura normalmente no tiene datos |
| 422 | `terrain_coverage_unavailable` | el punto está dentro del bbox pero en el borde o fuera de la cobertura de los rasters procesados (celdas sin elevación o combustible) |
| 503 | `static_layers_unavailable` | faltan DEM/pendiente/orientación/combustible (`make ingest-terrain`, `make ingest-vegetation`) |
| 422 | (sin `code`; formato estándar de FastAPI) | cuerpo inválido (`horizon_days` fuera de 1..7, latitud inválida, fecha mal formada) |

Los clima "bajo demanda" **no** se piden a CDS desde la API (una descarga de ERA5-Land tarda minutos y requiere credenciales): si falta, el error indica qué ingerir.

### Caché

Clave: `(celda de la grilla, fecha, horizonte, modelo)`, LRU de 256 entradas en memoria del proceso. Dos puntos de la misma celda de 250 m comparten resultado. Solo se cachean respuestas exitosas. Sin expiración: la predicción depende únicamente de rasters en disco; si se re-ingieren datos hay que reiniciar la API.

## `GET /active-fires?days=2`

Detecciones activas recientes de NASA FIRMS (VIIRS SNPP NRT) dentro del bbox de trabajo, como puntos GeoJSON (`properties`: `detected_at`, `frp`, `confidence`, `satellite`, `instrument`). `days` de 1 a 5 (límite del Area API). Solo contexto para el mapa; no alimenta `/predict`.

Caché de 10 minutos por `days` (FIRMS limita 5000 transacciones / 10 min por clave). Si FIRMS falla responde `502` con `code: "firms_unavailable"`, nunca una lista vacía (sería indistinguible de "no hay incendios").

## Modelo servido

`cellular_automata` con parámetros por defecto, cargado **una vez** al iniciar la app (`serving/api/model_registry.py`). Decisión y motivos: `docs/backtest-2026.md` sección 8 y `docs/decisions.md`. Los ensambles y el U-Net no se sirven.
