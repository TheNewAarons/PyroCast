# features/dataset/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble, per fire event, a spatiotemporal tensor (day, channel,
height, width) from already-processed feature layers (terrain, weather,
vegetation, fuel-type, fire state), persist it as Zarr with metadata in
PostGIS, compute a reproducible per-event train/val/test split, and wire a
new `pyrocast-features build-dataset` CLI command that runs the whole
assembly for a date range.

**Architecture:** `features/dataset/` does NOT re-run ingestion (DEM/ERA5/
Sentinel-2/WorldCover/FIRMS downloads) — it assumes `pyrocast-ingest
dem/era5/sentinel2/worldcover` and `firms` already ran for (at least) the
requested date range, and reads their already-materialized output files by
a fixed filename convention (documented in `docs/dataset-card.md`). For
each fire event (built via the already-implemented
`features.fire_state.clustering.build_fire_events`), it builds a
per-event `WorkGrid` (via the already-implemented `features.grid.grid.
build_grid`, called with an event-specific padded bbox instead of the
whole study area), resamples every channel source onto that grid, calls
the already-implemented `features.fire_state.rasterize.build_fire_state`
for the fire-mask channel, stacks everything into one `xarray.DataArray`
with a fixed channel order, writes it to Zarr, and inserts event metadata
into the existing `shared.db.schema.fire_event` table. `features/dataset/`
has zero dependency on the `ingestion` package (see Global Constraints) —
it reads FIRMS' raw parquet format directly with a small reader of its
own, duplicating ~20 lines rather than importing `ingestion.firms.storage`
and inverting the established `ingestion` → `features` dependency
direction.

Two small prerequisite fixes to already-shipped code are needed before
assembly can work at all (Tasks 1–2): today's NDVI output file always
overwrites the same `ndvi.tif` name regardless of which month was
requested (so "nearest available month" has nothing to choose from), and
today's weather derivation only saves wind speed/direction, never the raw
u/v components the tensor needs.

**Tech Stack:** `numpy`, `rasterio` (generic `reproject`-to-`WorkGrid`
helper), `xarray`+`zarr` (the tensor + its persistence), `pyproj`
(event-bbox padding in projected meters), `pyarrow` (reading FIRMS' raw
parquet directly — new `features` dependency), `sqlalchemy` (writing
`shared.db.schema.fire_event` — new direct `features` dependency,
already available transitively via `shared`), `typer` (new `features`
CLI — new direct dependency, already available transitively via
`ingestion` but `features` never had its own CLI before).

**Spec:** the user's request (quoted below), governed by `/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md`.

```
Implementa features/dataset/.

1. Para cada evento de incendio (P5), ensambla un tensor espaciotemporal
   (día, canal, alto, ancho) con estos canales: elevación, pendiente,
   orientación (estáticos), viento (u, v), temperatura, humedad relativa,
   precipitación (dinámicos, uno por día), NDVI (dinámico, del composite
   mensual más cercano), tipo de combustible (estático), máscara de fuego
   del día (variable objetivo/histórica).
2. Alineación temporal: cada evento cubre desde unos días antes de la
   primera detección hasta el final del evento, para dar contexto previo
   al modelo.
3. Persiste cada evento como un array Zarr en data/processed/dataset/, con
   sus metadatos (bbox recortado, fechas, id de evento) en la tabla
   `fire_event` de PostGIS (P0).
4. Split reproducible train/val/test POR EVENTO (nunca por píxel ni por
   día dentro de un mismo evento, para evitar fuga de datos), con semilla
   fija, documentado en docs/dataset-card.md junto con: cuántos eventos
   hay en total, su distribución temporal y geográfica, y cualquier
   desbalance relevante (por ejemplo, pocos eventos grandes frente a
   muchos pequeños).
5. CLI: `pyrocast-features build-dataset` que corre todo el pipeline de
   features (P2-P6) de punta a punta para un rango de fechas dado.

Tests: ensamblado de un evento sintético pequeño con canales de fixture,
verificación de que las dimensiones y el orden de canales son
consistentes, verificación de que el split por evento no deja el mismo
evento en dos splits a la vez, reproducibilidad del split con la misma
semilla.
Criterios de aceptación: make build-dataset corre sobre datos de fixture y
produce al menos un evento en Zarr; docs/dataset-card.md completo. Cierra
la etapa 2.
```

## Global Constraints

- Python 3.12, `mypy --strict` on `features/src`, `ruff check .` clean
  repo-wide, tests via `pytest`, no real network/credentialed calls in
  any test. A real-Postgres test (Task 6) is `@pytest.mark.skipif`-guarded
  on the `POSTGRES_*`/other required env vars being set — same pattern as
  the existing `shared/tests/test_db_schema.py`.
- **`features/dataset/` never imports from `ingestion`.** The established
  dependency direction in this repo is `ingestion` → `features` (an
  accepted, documented exception used by `ingestion/dem/cli.py` and
  `ingestion/era5/cli.py`); the reverse has never been accepted. FIRMS'
  raw parquet format is read directly by a small reader in
  `features/dataset/firms_loader.py` rather than importing
  `ingestion.firms.storage` — a deliberate small duplication, documented
  in `docs/decisions.md` (Task 10).
- **Filename conventions this plan reads from (all already-established,
  by other modules' own CLIs), verified against the actual source before
  writing this plan:**
  - `settings.data_processed_dir/"dem"/*.tif` — exactly one expected
    (single study-area config; `ingestion/dem/pipeline.py:33` hashes
    bbox+resolution+crs into the filename, so a second file only appears
    if the study area itself changes — out of scope here, documented as
    a known limitation in `docs/limitations.md`, Task 10).
  - `settings.data_processed_dir/"terrain"/"slope_deg.tif"` and
    `"aspect_deg.tif"` — fixed names (`features/terrain/slope_aspect.py`).
  - `settings.data_processed_dir/"vegetation"/"fuel_type.tif"` — fixed
    name (`ingestion/worldcover/cli.py`).
  - `settings.data_processed_dir/"vegetation"/"ndvi_YYYY-MM.tif"` — one
    per ingested month, **after Task 1's fix** (today it's a single
    `ndvi.tif` that a second `pyrocast-ingest sentinel2` call for a
    different month silently overwrites — a real bug this plan must fix
    to make "nearest available month" meaningful at all).
  - `settings.data_processed_dir/"weather"/f"{field}_{date_iso}.tif"` —
    already date-stamped (`features/weather/derive.py:161`), no fix
    needed; `field` gains two new values (`wind_u`, `wind_v`) in Task 2.
- **New `features` package dependencies** (add in Task 3, with a
  `docs/decisions.md` note in Task 10): `pyarrow>=17.0` (FIRMS parquet
  reader), `sqlalchemy>=2.0` (writing `fire_event` directly — declared
  even though already transitively present via `shared`, matching this
  session's "declare what you import directly" convention), `typer>=0.12`
  (new `features` CLI).
- Static channels (`elevation`, `slope_deg`, `aspect_deg`, `fuel_type`)
  are broadcast across every day in the tensor so the whole tensor has one
  uniform `(day, channel, y, x)` shape — never a dict of mixed-dimension
  arrays. This is a deliberate reading of "un tensor... con estos
  canales", documented in `docs/dataset-card.md` (Task 10).
- `DEFAULT_PRE_EVENT_PADDING_DAYS = 5` (features/dataset/pipeline.py) —
  "unos días antes" has no literature-derived value in the spec; 5 is a
  heuristic default, configurable per call, documented as such (same
  honesty bar as `features/fire_state`'s `DEFAULT_SPATIAL_EPS_M`/
  `DEFAULT_TEMPORAL_EPS`).
- A weather/NDVI channel for a day with no matching processed file on
  disk (e.g. the padding days fall before the earliest date the user ran
  `pyrocast-ingest era5`/`sentinel2` for) is filled with `NaN`, never
  fabricated — documented as an operational precondition in
  `docs/dataset-card.md` (Task 10): ingest a date range that already
  covers the padding, not just the raw event dates.

## Review Focus

- **An event whose padded window (`event.start_date - 5 days`) crosses a
  month boundary for NDVI lookup**: the nearest-available-month search
  must not accidentally prefer a later month over an earlier one at a
  tie, and must return `None` cleanly (never crash) when zero NDVI files
  exist yet — covered by Task 7's dedicated `_nearest_month_path` tests.
- **A fire event with only ONE detection** (a real, valid, already-tested
  case from `features/fire_state`): `build_dataset_for_event` must not
  divide by zero or crash computing the event's bbox (a single point has
  zero spatial extent before buffering) — covered by Task 7's integration
  test using a single-detection `FireEvent`.
- **Split with a very small number of events** (a real early-project
  state: a handful of events total, not thousands): rounding
  `train_frac`/`val_frac` fractions of a tiny `n` must still assign every
  event to exactly one split and never crash on an empty remainder —
  covered by Task 5's dedicated small-`n` test.
- **The CLI producing writes are all real, even though every collaborator
  is faked**: the Task 8 CLI smoke test must assert against files that
  actually exist on disk after the call (a real Zarr directory, a real
  `splits.json`) — not just an exit code of 0 — mirroring the plan's own
  "Review Focus" lesson from the `ingestion/sentinel2+worldcover` review
  ("tests that only check 'something happened' can hide the wrong
  behavior").
- **`resample_to_grid` fed a categorical raster (`fuel_type.tif`) with the
  default `Resampling.bilinear`**: nothing in the type system stops a
  caller from doing this by mistake — covered by Task 4's dedicated test
  proving `Resampling.nearest` on `fuel_type` never fabricates
  intermediate class codes, mirroring the identical guard already proven
  in `ingestion/worldcover/pipeline.py`.

---

## Task 1: Fix Sentinel-2 CLI — month-stamped NDVI filenames

**Files:**
- Modify: `ingestion/src/ingestion/sentinel2/cli.py`
- Modify: `ingestion/tests/test_sentinel2_cli.py`

**Interfaces:**
- Consumes: `features.vegetation.ndvi.compute_and_save_vegetation` (unchanged signature).
- Produces: `settings.data_processed_dir/"vegetation"/f"ndvi_{year:04d}-{month:02d}.tif"` on disk — Task 7 consumes this exact filename pattern via a glob (`ndvi_*.tif`).

- [ ] **Step 1: Write the failing test**

In `ingestion/tests/test_sentinel2_cli.py`, replace the final assertions of
`test_sentinel2_cli_wires_settings_and_produces_ndvi` (the `assert "NDVI"
in result.output` line and everything after it) with:

```python
    result = runner.invoke(app, ["sentinel2", "--year", "2026", "--month", "1"])
    assert result.exit_code == 0, result.output
    assert "NDVI" in result.output

    from shared.config import get_settings as reread_settings

    settings = reread_settings()
    month_stamped = settings.data_processed_dir / "vegetation" / "ndvi_2026-01.tif"
    assert month_stamped.exists()
    assert not (settings.data_processed_dir / "vegetation" / "ndvi.tif").exists()
    get_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --package ingestion pytest ingestion/tests/test_sentinel2_cli.py -v`
Expected: FAIL — `ndvi_2026-01.tif` does not exist (the old code always
wrote `ndvi.tif` and never renamed it), `assert month_stamped.exists()` fails.

- [ ] **Step 3: Write minimal implementation**

In `ingestion/src/ingestion/sentinel2/cli.py`, replace:

```python
    ndvi_path = compute_and_save_vegetation(
        composite_path,
        settings.data_processed_dir / "vegetation",
        target_crs=settings.crs,
        target_resolution_m=settings.spatial_resolution_m,
    )
    typer.echo(f"NDVI: {ndvi_path}")
```

with:

```python
    ndvi_path = compute_and_save_vegetation(
        composite_path,
        settings.data_processed_dir / "vegetation",
        target_crs=settings.crs,
        target_resolution_m=settings.spatial_resolution_m,
    )
    # compute_and_save_vegetation siempre escribe "ndvi.tif" -- sin este
    # renombrado, una segunda invocación para otro mes pisaría el NDVI del
    # mes anterior, dejando siempre un único archivo en disco. El nombre
    # con mes es lo que permite a features/dataset/ elegir "el composite
    # mensual más cercano" entre varios meses ya ingeridos.
    month_stamped_path = ndvi_path.parent / f"ndvi_{year:04d}-{month:02d}.tif"
    ndvi_path.replace(month_stamped_path)
    typer.echo(f"NDVI: {month_stamped_path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --package ingestion pytest ingestion/tests/test_sentinel2_cli.py -v`
Expected: PASS

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict shared/src features/src && uv run ruff check ingestion/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add ingestion/src/ingestion/sentinel2/cli.py ingestion/tests/test_sentinel2_cli.py
git commit -m "fix: month-stamp NDVI output filename so multiple months coexist"
```

---

## Task 2: Add wind_u/wind_v channels to weather derivation

**Files:**
- Modify: `features/src/features/weather/derive.py`
- Modify: `features/tests/test_weather_derive.py`

**Interfaces:**
- Produces: `compute_and_save_weather(...)`'s returned dict gains two new
  keys, `"wind_u"` and `"wind_v"` (same `dict[str, Path]` per-date shape as
  the existing 5 keys). Task 7 consumes these two new keys by name.

- [ ] **Step 1: Write the failing test**

In `features/tests/test_weather_derive.py`, update the existing exact-set
assertion and add a new dedicated test:

```python
    assert set(paths) == {
        "wind_speed", "wind_direction", "wind_u", "wind_v",
        "relative_humidity", "temperature", "precipitation"
    }
```

(replace the old 5-key `assert set(paths) == {...}` line in
`test_compute_and_save_weather_writes_geotiffs_at_target_resolution` with
the 7-key version above), and append a new test to the file:

```python
def test_compute_and_save_weather_wind_components_match_source_u10_v10(tmp_path):
    lat = np.array([-36.0, -37.0, -38.0, -39.0])
    lon = np.array([-74.0, -73.0, -72.0, -71.0])
    daily_nc = tmp_path / "daily.nc"
    _write_synthetic_daily_nc(daily_nc, lat, lon)  # fixture: u10=2.0, v10=3.0 uniforme

    output_dir = tmp_path / "weather"
    paths = compute_and_save_weather(
        daily_nc, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )

    with rasterio.open(paths["wind_u"]["2026-01-15"]) as ds:
        u = ds.read(1)
    with rasterio.open(paths["wind_v"]["2026-01-15"]) as ds:
        v = ds.read(1)
    # campo uniforme en el origen -> uniforme tras reproyección bilineal
    assert np.allclose(u[~np.isnan(u)], 2.0, atol=1e-3)
    assert np.allclose(v[~np.isnan(v)], 3.0, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package features pytest features/tests/test_weather_derive.py -v`
Expected: FAIL — `test_compute_and_save_weather_writes_geotiffs_at_target_resolution`
fails on the set-equality assert (missing `wind_u`/`wind_v`); the new test
fails with `KeyError: 'wind_u'`.

- [ ] **Step 3: Write minimal implementation**

In `features/src/features/weather/derive.py`, in `compute_and_save_weather`:

```python
    paths: dict[str, dict[str, Path]] = {
        "wind_speed": {}, "wind_direction": {}, "wind_u": {}, "wind_v": {},
        "relative_humidity": {}, "temperature": {}, "precipitation": {},
    }
```

and:

```python
            day_fields = {
                "wind_speed": speed,
                "wind_direction": direction,
                "wind_u": fields["u10"],
                "wind_v": fields["v10"],
                "relative_humidity": rh,
                "temperature": fields["t2m"],
                "precipitation": fields["tp"],
            }
```

(both are the only two edits — `fields["u10"]`/`fields["v10"]` are already
computed a few lines above, from the already-latitude-normalized `fields`
dict; `_reproject_field`/`_write_geotiff` need no changes since they're
already generic over any 2-D field.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package features pytest features/tests/test_weather_derive.py -v`
Expected: `6 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add features/src/features/weather/derive.py features/tests/test_weather_derive.py
git commit -m "feat: expose raw wind_u/wind_v components from ERA5-Land derivation"
```

---

## Task 3: `features/dataset/firms_loader.py` — read FIRMS raw parquet directly

**Files:**
- Create: `features/src/features/dataset/firms_loader.py`
- Modify: `features/src/features/dataset/__init__.py`
- Modify: `features/pyproject.toml` (add `pyarrow>=17.0`, `sqlalchemy>=2.0`, `typer>=0.12`)
- Test: `features/tests/test_dataset_firms_loader.py`

**Interfaces:**
- Consumes: the on-disk parquet format written by
  `ingestion.firms.storage.save_raw_response` — string-typed columns
  named exactly as FIRMS' Area API CSV (`latitude`, `longitude`,
  `acq_date`, `acq_time`, `confidence`, `satellite`, `instrument`, `frp`),
  partitioned under `<base_dir>/firms/download_date=.../*.parquet` —
  verified directly against the real function's output before writing
  this plan (`{'latitude': '-37.5', 'longitude': '-72.3', 'acq_date':
  '2026-01-15', 'acq_time': '0130', 'confidence': 'n', 'satellite': 'N',
  'instrument': 'VIIRS', 'frp': '12.3'}`).
- Produces: `load_firms_detections(base_dir: Path, start: date, end: date)
  -> list[FireDetection]`. Task 9 (CLI) consumes this directly.

- [ ] **Step 1: Write the failing tests**

Create `features/tests/test_dataset_firms_loader.py`:

```python
"""Tests del lector de detecciones FIRMS crudas (parquet), sin depender
de `ingestion` -- ver docs/decisions.md."""
import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from features.dataset.firms_loader import load_firms_detections


def _write_fixture_parquet(base_dir: Path, rows: list[dict], filename: str) -> None:
    partition_dir = base_dir / "firms" / "download_date=2026-01-20"
    partition_dir.mkdir(parents=True, exist_ok=True)
    columns = {name: [row[name] for row in rows] for name in rows[0]}
    table = pa.table({name: pa.array(values) for name, values in columns.items()})
    pq.write_table(table, partition_dir / filename)


def test_load_firms_detections_filters_by_date_range(tmp_path):
    _write_fixture_parquet(
        tmp_path,
        [
            {
                "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2026-01-10",
                "acq_time": "0130", "confidence": "n", "satellite": "N",
                "instrument": "VIIRS", "frp": "12.3",
            },
            {
                "latitude": "-37.6", "longitude": "-72.4", "acq_date": "2026-01-15",
                "acq_time": "0140", "confidence": "h", "satellite": "N",
                "instrument": "VIIRS", "frp": "8.0",
            },
            {
                "latitude": "-37.7", "longitude": "-72.5", "acq_date": "2026-02-01",
                "acq_time": "0150", "confidence": "n", "satellite": "N",
                "instrument": "VIIRS", "frp": "5.0",
            },
        ],
        "chunk1.parquet",
    )
    detections = load_firms_detections(tmp_path, dt.date(2026, 1, 12), dt.date(2026, 1, 20))
    assert len(detections) == 1
    assert detections[0].latitude == -37.6
    assert detections[0].detected_at == dt.datetime(2026, 1, 15, 1, 40, tzinfo=dt.UTC)


def test_load_firms_detections_preserves_leading_zero_in_acq_time(tmp_path):
    _write_fixture_parquet(
        tmp_path,
        [{
            "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2026-01-15",
            "acq_time": "0005", "confidence": "n", "satellite": "N",
            "instrument": "VIIRS", "frp": "1.0",
        }],
        "chunk1.parquet",
    )
    detections = load_firms_detections(tmp_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert detections[0].detected_at.hour == 0
    assert detections[0].detected_at.minute == 5


def test_load_firms_detections_handles_missing_frp(tmp_path):
    _write_fixture_parquet(
        tmp_path,
        [{
            "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2026-01-15",
            "acq_time": "0130", "confidence": "n", "satellite": "N",
            "instrument": "VIIRS", "frp": "",
        }],
        "chunk1.parquet",
    )
    detections = load_firms_detections(tmp_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert detections[0].frp is None


def test_load_firms_detections_returns_empty_list_when_no_files(tmp_path):
    assert load_firms_detections(tmp_path, dt.date(2026, 1, 1), dt.date(2026, 1, 31)) == []


def test_load_firms_detections_reads_across_multiple_parquet_files(tmp_path):
    _write_fixture_parquet(
        tmp_path,
        [{
            "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2026-01-15",
            "acq_time": "0130", "confidence": "n", "satellite": "N",
            "instrument": "VIIRS", "frp": "1.0",
        }],
        "chunk1.parquet",
    )
    _write_fixture_parquet(
        tmp_path,
        [{
            "latitude": "-38.5", "longitude": "-73.3", "acq_date": "2026-01-16",
            "acq_time": "0230", "confidence": "h", "satellite": "N",
            "instrument": "VIIRS", "frp": "2.0",
        }],
        "chunk2.parquet",
    )
    detections = load_firms_detections(tmp_path, dt.date(2026, 1, 1), dt.date(2026, 1, 31))
    assert len(detections) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package features pytest features/tests/test_dataset_firms_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.dataset.firms_loader'`

- [ ] **Step 3: Add `pyarrow`, `sqlalchemy`, `typer` to `features/pyproject.toml`**

In `features/pyproject.toml`, add to `dependencies`:

```toml
    "pyarrow>=17.0",
    "sqlalchemy>=2.0",
    "typer>=0.12",
```

(placed after `"pyproj>=3.6",`). Run `uv sync --all-packages` — **never**
a scoped `uv sync`/`uv sync --package X` in this workspace: it silently
prunes the shared `.venv` down to just that scope, dropping the root
`dev` dependency-group (pytest/mypy/ruff) and every other member package
(discovered and fixed during the `features/grid`+`features/fire_state`
plan's Task 1 — see that plan's ledger).

- [ ] **Step 4: Write minimal implementation**

Create `features/src/features/dataset/firms_loader.py`:

```python
"""Lee las detecciones FIRMS crudas directamente desde el parquet que
`ingestion.firms.storage.save_raw_response` ya escribe -- sin importar
`ingestion` (ver docs/decisions.md: `features` nunca depende de
`ingestion`, es al revés). Duplica ~20 líneas de lectura de un formato de
almacenamiento simple (columnas string, mismo esquema que el CSV del Area
API de FIRMS) en vez de invertir esa dependencia."""
import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq

from shared.schemas import FireDetection


def _parse_detected_at(acq_date: str, acq_time: str) -> dt.datetime:
    # acq_time viene como "HHMM" sin separador, con cero a la izquierda
    # (p. ej. "0005" = 00:05 UTC) -- NO tratar como entero, se pierde el
    # cero inicial. (Mismo cuidado que ingestion/firms/parser.py.)
    hour = int(acq_time[:2])
    minute = int(acq_time[2:])
    date = dt.date.fromisoformat(acq_date)
    return dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=dt.UTC)


def load_firms_detections(
    base_dir: Path, start: dt.date, end: dt.date
) -> list[FireDetection]:
    firms_dir = base_dir / "firms"
    if not firms_dir.exists():
        return []

    detections: list[FireDetection] = []
    for parquet_path in sorted(firms_dir.glob("**/*.parquet")):
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            acq_date = dt.date.fromisoformat(row["acq_date"])
            if not (start <= acq_date <= end):
                continue
            frp_raw = (row.get("frp") or "").strip()
            detections.append(
                FireDetection(
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    detected_at=_parse_detected_at(row["acq_date"], row["acq_time"]),
                    frp=float(frp_raw) if frp_raw else None,
                    confidence=row["confidence"],
                    satellite=row["satellite"],
                    instrument=row["instrument"],
                )
            )
    return detections
```

Update `features/src/features/dataset/__init__.py`:

```python
"""Módulo de features: ensamblado del dataset espaciotemporal por evento
de incendio (P6) -- tensor (día, canal, alto, ancho), persistencia Zarr +
metadatos PostGIS, split reproducible por evento.

Ver `docs/dataset-card.md` para la definición completa de canales,
convención de rutas de entrada, y estadísticas del dataset.
"""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --package features pytest features/tests/test_dataset_firms_loader.py -v`
Expected: `5 passed`

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean

- [ ] **Step 7: Commit**

```bash
git add features/src/features/dataset features/tests/test_dataset_firms_loader.py features/pyproject.toml uv.lock
git commit -m "feat: add FIRMS raw-parquet loader (features/dataset, no ingestion dependency)"
```

---

## Task 4: `features/dataset/resample.py` + `channels.py`/`assemble.py` — tensor assembly

**Files:**
- Create: `features/src/features/dataset/resample.py`
- Create: `features/src/features/dataset/assemble.py`
- Test: `features/tests/test_dataset_resample.py`
- Test: `features/tests/test_dataset_assemble.py`

**Interfaces:**
- Consumes: `features.grid.grid.WorkGrid`.
- Produces: `resample_to_grid(source_path: Path, grid: WorkGrid,
  resampling: Resampling = Resampling.bilinear) -> np.ndarray`;
  `CHANNEL_ORDER: tuple[str, ...]` (exactly `("elevation", "slope_deg",
  "aspect_deg", "wind_u", "wind_v", "temperature", "relative_humidity",
  "precipitation", "ndvi", "fuel_type", "fire_mask")`); `EventChannels`
  (frozen dataclass: `days: tuple[date, ...]`, `static: dict[str,
  np.ndarray]`, `dynamic: dict[str, dict[date, np.ndarray]]`);
  `assemble_event_tensor(channels: EventChannels) -> xr.DataArray` (dims
  `("day", "channel", "y", "x")`); `save_event_to_zarr(tensor:
  xr.DataArray, output_dir: Path, event_id: int) -> Path`. Task 7 consumes
  all of these directly.

- [ ] **Step 1: Write the failing tests**

Create `features/tests/test_dataset_resample.py`:

```python
"""Tests de resampleo genérico de un raster a una WorkGrid arbitraria."""
import numpy as np
import rasterio
from features.dataset.resample import resample_to_grid
from features.grid.grid import build_grid
from rasterio.transform import from_origin
from rasterio.warp import Resampling


def _write_tif(path, data, transform, crs, nodata):
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype=str(data.dtype), crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_resample_to_grid_bilinear_uniform_field_stays_uniform(tmp_path):
    src_path = tmp_path / "src.tif"
    transform = from_origin(190000, 5791000, 100, 100)
    data = np.full((50, 50), 42.0, dtype="float32")
    _write_tif(src_path, data, transform, "EPSG:32719", nodata=-9999.0)

    grid = build_grid((-72.51, -38.01, -72.49, -37.99), "EPSG:32719", 250.0)
    result = resample_to_grid(src_path, grid, Resampling.bilinear)
    assert result.shape == (grid.height, grid.width)
    finite = result[~np.isnan(result)]
    assert finite.size > 0
    assert np.allclose(finite, 42.0, atol=1e-3)


def test_resample_to_grid_nearest_never_fabricates_class_codes(tmp_path):
    src_path = tmp_path / "classes.tif"
    transform = from_origin(190000, 5791000, 100, 100)
    data = np.full((50, 50), 10, dtype="uint8")
    data[:, 25:] = 80  # dos clases reales en bloques, como en WorldCover
    _write_tif(src_path, data, transform, "EPSG:32719", nodata=0.0)

    grid = build_grid((-72.51, -38.01, -72.49, -37.99), "EPSG:32719", 250.0)
    result = resample_to_grid(src_path, grid, Resampling.nearest)
    present = set(np.unique(result[~np.isnan(result)]))
    assert present <= {10.0, 80.0}


def test_resample_to_grid_propagates_source_nodata(tmp_path):
    src_path = tmp_path / "with_nodata.tif"
    transform = from_origin(190000, 5791000, 100, 100)
    data = np.full((50, 50), 5.0, dtype="float32")
    data[:10, :10] = -9999.0
    _write_tif(src_path, data, transform, "EPSG:32719", nodata=-9999.0)

    grid = build_grid((-72.51, -38.01, -72.49, -37.99), "EPSG:32719", 250.0)
    result = resample_to_grid(src_path, grid, Resampling.bilinear)
    assert np.any(np.isnan(result))  # el hueco de origen no se fabrica como 5.0
```

Create `features/tests/test_dataset_assemble.py`:

```python
"""Tests de ensamblado del tensor (día, canal, alto, ancho) y su
persistencia en Zarr -- canales sintéticos de fixture."""
import datetime as dt

import numpy as np
import xarray as xr
from features.dataset.assemble import (
    CHANNEL_ORDER,
    EventChannels,
    assemble_event_tensor,
    save_event_to_zarr,
)


def _fixture_channels() -> EventChannels:
    height, width = 4, 4
    days = (dt.date(2026, 1, 1), dt.date(2026, 1, 2), dt.date(2026, 1, 3))
    static = {
        "elevation": np.full((height, width), 100.0, dtype="float32"),
        "slope_deg": np.full((height, width), 5.0, dtype="float32"),
        "aspect_deg": np.full((height, width), 180.0, dtype="float32"),
        "fuel_type": np.full((height, width), 3.0, dtype="float32"),
    }
    dynamic = {
        name: {day: np.full((height, width), float(i), dtype="float32") for i, day in enumerate(days)}
        for name in ("wind_u", "wind_v", "temperature", "relative_humidity", "precipitation", "ndvi")
    }
    dynamic["fire_mask"] = {
        days[0]: np.zeros((height, width), dtype="float32"),
        days[1]: np.zeros((height, width), dtype="float32"),
        days[2]: np.ones((height, width), dtype="float32"),
    }
    return EventChannels(days=days, static=static, dynamic=dynamic)


def test_assemble_event_tensor_has_expected_shape_and_channel_order():
    tensor = assemble_event_tensor(_fixture_channels())
    assert isinstance(tensor, xr.DataArray)
    assert tensor.dims == ("day", "channel", "y", "x")
    assert tensor.shape == (3, len(CHANNEL_ORDER), 4, 4)
    assert list(tensor.coords["channel"].values) == list(CHANNEL_ORDER)
    assert list(tensor.coords["day"].values) == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_assemble_event_tensor_static_channels_repeat_identically_across_days():
    tensor = assemble_event_tensor(_fixture_channels())
    elevation_idx = CHANNEL_ORDER.index("elevation")
    elevation_across_days = tensor.values[:, elevation_idx, :, :]
    assert np.all(elevation_across_days == 100.0)


def test_assemble_event_tensor_dynamic_channels_vary_by_day():
    tensor = assemble_event_tensor(_fixture_channels())
    wind_u_idx = CHANNEL_ORDER.index("wind_u")
    assert np.all(tensor.values[0, wind_u_idx, :, :] == 0.0)
    assert np.all(tensor.values[1, wind_u_idx, :, :] == 1.0)
    assert np.all(tensor.values[2, wind_u_idx, :, :] == 2.0)


def test_assemble_event_tensor_fire_mask_channel_matches_input():
    tensor = assemble_event_tensor(_fixture_channels())
    fire_mask_idx = CHANNEL_ORDER.index("fire_mask")
    assert np.all(tensor.values[0, fire_mask_idx, :, :] == 0.0)
    assert np.all(tensor.values[2, fire_mask_idx, :, :] == 1.0)


def test_save_event_to_zarr_roundtrip_preserves_values(tmp_path):
    tensor = assemble_event_tensor(_fixture_channels())
    zarr_path = save_event_to_zarr(tensor, tmp_path, event_id=7)
    assert zarr_path.exists()
    assert zarr_path.name == "event_0007.zarr"

    reopened = xr.open_zarr(zarr_path)
    reopened_tensor = reopened["fire_event_tensor"]
    assert reopened_tensor.shape == tensor.shape
    assert np.allclose(reopened_tensor.values, tensor.values)
    assert list(reopened_tensor.coords["channel"].values) == list(CHANNEL_ORDER)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package features pytest features/tests/test_dataset_resample.py features/tests/test_dataset_assemble.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.dataset.resample'` (and `.assemble`)

- [ ] **Step 3: Write minimal implementation**

Create `features/src/features/dataset/resample.py`:

```python
"""Resampleo genérico de un raster de una sola banda a una `WorkGrid`
arbitraria -- usado para cada canal del tensor de evento (terreno, clima,
vegetación, tipo de combustible), sea cual sea su CRS/resolución/bounds
de origen. Reutiliza el mismo patrón de reproyección (nodata explícito
end-to-end) que `ingestion/dem`, `ingestion/worldcover` y
`features/weather/derive.py` ya usan cada uno por su cuenta -- este
módulo es lo que finalmente pone `features/grid/` a trabajar: en vez de
migrar cada pipeline de ingesta a la grilla canónica (todavía sin hacer,
ver docs/decisions.md), se resamplea la SALIDA ya procesada de cada uno
sobre la `WorkGrid` del evento aquí."""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from features.grid.grid import WorkGrid


def resample_to_grid(
    source_path: Path, grid: WorkGrid, resampling: Resampling = Resampling.bilinear
) -> np.ndarray:
    with rasterio.open(source_path) as src:
        src_nodata = src.nodata
        dst_nodata = src_nodata if src_nodata is not None else float("nan")
        dst_array = np.full((grid.height, grid.width), dst_nodata, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
    if dst_nodata is not None and not (isinstance(dst_nodata, float) and np.isnan(dst_nodata)):
        return np.where(dst_array == dst_nodata, np.nan, dst_array)
    return dst_array
```

Create `features/src/features/dataset/assemble.py`:

```python
"""Ensamblado del tensor espaciotemporal (día, canal, alto, ancho) de un
evento de incendio, y su persistencia como Zarr.

Orden de canales FIJO (`CHANNEL_ORDER`) -- el mismo para todo evento,
todo el tiempo: estáticos (elevación, pendiente, orientación, tipo de
combustible) se repiten idénticos en cada día del tensor; dinámicos
(viento u/v, temperatura, humedad relativa, precipitación, NDVI, máscara
de fuego) varían día a día. Esto es una decisión deliberada de forma: un
solo array 4D uniforme, no un dict de arrays de dimensión mixta -- ver
docs/dataset-card.md.
"""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

CHANNEL_ORDER: tuple[str, ...] = (
    "elevation", "slope_deg", "aspect_deg",
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
    "ndvi", "fuel_type", "fire_mask",
)


@dataclass(frozen=True)
class EventChannels:
    days: tuple[dt.date, ...]
    static: dict[str, np.ndarray]
    dynamic: dict[str, dict[dt.date, np.ndarray]]


def assemble_event_tensor(channels: EventChannels) -> xr.DataArray:
    n_days = len(channels.days)
    height, width = next(iter(channels.static.values())).shape
    data = np.empty((n_days, len(CHANNEL_ORDER), height, width), dtype="float32")
    for c_idx, name in enumerate(CHANNEL_ORDER):
        if name in channels.static:
            data[:, c_idx, :, :] = channels.static[name][np.newaxis, :, :]
        else:
            for d_idx, day in enumerate(channels.days):
                data[d_idx, c_idx, :, :] = channels.dynamic[name][day]
    return xr.DataArray(
        data,
        dims=("day", "channel", "y", "x"),
        coords={
            "day": [day.isoformat() for day in channels.days],
            "channel": list(CHANNEL_ORDER),
        },
        name="fire_event_tensor",
    )


def save_event_to_zarr(tensor: xr.DataArray, output_dir: Path, event_id: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"event_{event_id:04d}.zarr"
    tensor.to_dataset().to_zarr(path, mode="w")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package features pytest features/tests/test_dataset_resample.py features/tests/test_dataset_assemble.py -v`
Expected: `8 passed` (3 resample + 5 assemble)

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add features/src/features/dataset/resample.py features/src/features/dataset/assemble.py features/tests/test_dataset_resample.py features/tests/test_dataset_assemble.py
git commit -m "feat: add generic grid resampling and event-tensor assembly (features/dataset)"
```

---

## Task 5: `features/dataset/split.py` — reproducible per-event split

**Files:**
- Create: `features/src/features/dataset/split.py`
- Test: `features/tests/test_dataset_split.py`

**Interfaces:**
- Produces: `split_events(event_ids: list[int], seed: int = 42,
  train_frac: float = 0.7, val_frac: float = 0.15) -> dict[str,
  list[int]]` (keys exactly `"train"`, `"val"`, `"test"`). Task 9 (CLI)
  consumes this directly.

- [ ] **Step 1: Write the failing tests**

Create `features/tests/test_dataset_split.py`:

```python
"""Tests del split train/val/test reproducible POR EVENTO."""
from features.dataset.split import split_events


def test_split_events_no_event_appears_in_two_splits():
    event_ids = list(range(20))
    splits = split_events(event_ids, seed=1)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(event_ids)


def test_split_events_is_reproducible_with_the_same_seed():
    event_ids = list(range(50))
    first = split_events(event_ids, seed=7)
    second = split_events(event_ids, seed=7)
    assert first == second


def test_split_events_handles_small_event_count_without_crashing():
    event_ids = [101, 102, 103]
    splits = split_events(event_ids, seed=1)
    all_assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_assigned) == event_ids


def test_split_events_input_order_does_not_change_the_result():
    # el split depende del contenido del conjunto de ids, no del orden en
    # que la lista de entrada los trae (los ids se ordenan antes de
    # mezclar con la semilla) -- dos llamadas con los mismos ids en
    # distinto orden de entrada deben dar el mismo split.
    forward = split_events([1, 2, 3, 4, 5], seed=3)
    shuffled_input = split_events([5, 3, 1, 4, 2], seed=3)
    assert forward == shuffled_input
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package features pytest features/tests/test_dataset_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.dataset.split'`

- [ ] **Step 3: Write minimal implementation**

Create `features/src/features/dataset/split.py`:

```python
"""Split train/val/test reproducible, POR EVENTO -- nunca por píxel ni
por día dentro de un mismo evento (evitaría fuga de datos: días
consecutivos del mismo incendio son casi idénticos, y modelo entrenado
con un día y evaluado con el día siguiente del MISMO evento mediría
memorización, no generalización)."""
import random

DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_VAL_FRAC = 0.15
# el resto (0.15 por defecto) va a test.


def split_events(
    event_ids: list[int],
    seed: int = 42,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> dict[str, list[int]]:
    # ordenar antes de mezclar: el resultado depende solo del CONJUNTO de
    # ids y de la semilla, nunca del orden en que la lista de entrada los
    # trae.
    ids_sorted = sorted(event_ids)
    rng = random.Random(seed)
    shuffled = ids_sorted[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package features pytest features/tests/test_dataset_split.py -v`
Expected: `4 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add features/src/features/dataset/split.py features/tests/test_dataset_split.py
git commit -m "feat: add reproducible per-event train/val/test split (features/dataset)"
```

---

## Task 6: `features/dataset/db.py` — persist event metadata to PostGIS

**Files:**
- Create: `features/src/features/dataset/db.py`
- Test: `features/tests/test_dataset_db.py`

**Interfaces:**
- Consumes: `shared.db.schema.fire_event` (columns: `bbox: str`,
  `start_date: date`, `end_date: date | None`, `source: str`, `geom:
  Geometry`).
- Produces: `persist_fire_event_metadata(engine: Engine, event_id: int,
  bbox_cut: tuple[float, float, float, float], start_date: date,
  end_date: date, source: str = "firms_cluster") -> int` (returns the
  `fire_event.id` primary key of the inserted row). Task 7/9 consume this
  directly (dependency-injected — Task 9's CLI test never calls the real
  one).

- [ ] **Step 1: Write the failing tests**

Create `features/tests/test_dataset_db.py`:

```python
"""Tests de persistencia de metadatos de evento en `fire_event`
(PostGIS). El insert real solo se ejercita contra un Postgres real
(guardado con skipif, mismo patrón que shared/tests/test_db_schema.py) --
lo demás se verifica sin red/DB."""
import datetime as dt
import os

import pytest
from features.dataset.db import persist_fire_event_metadata
from shared.config import Settings
from sqlalchemy import create_engine, select, text
from shared.db.schema import fire_event, metadata


def _live_settings() -> Settings | None:
    required = [
        "FIRMS_MAP_KEY", "CDS_API_URL", "CDS_API_KEY",
        "COPERNICUS_DATASPACE_CLIENT_ID", "COPERNICUS_DATASPACE_CLIENT_SECRET",
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
        "POSTGRES_USER", "POSTGRES_PASSWORD",
    ]
    if not all(os.getenv(k) for k in required):
        return None
    return Settings()


@pytest.mark.skipif(_live_settings() is None, reason="requiere POSTGRES_* de un contenedor real")
def test_persist_fire_event_metadata_roundtrip_against_real_postgis():
    settings = _live_settings()
    assert settings is not None
    engine = create_engine(settings.postgres_dsn)
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        metadata.create_all(conn)

        event_id = persist_fire_event_metadata(
            engine=conn,  # Connection implementa el mismo Protocol que Engine para .execute
            event_id=1,
            bbox_cut=(-72.9, -38.9, -72.1, -38.1),
            start_date=dt.date(2026, 1, 15),
            end_date=dt.date(2026, 1, 20),
        )
        stored_bbox = conn.execute(
            select(fire_event.c.bbox).where(fire_event.c.id == event_id)
        ).scalar_one()
        assert stored_bbox == "-72.9,-38.9,-72.1,-38.1"
    finally:
        trans.rollback()
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --package features pytest features/tests/test_dataset_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.dataset.db'`
(the test collects and fails at import time even though it's
`skipif`-guarded — skip evaluation happens after import, so a missing
module is still a real collection error to fix).

- [ ] **Step 3: Write minimal implementation**

Create `features/src/features/dataset/db.py`:

```python
"""Persiste metadatos de un evento de incendio en la tabla `fire_event`
de PostGIS (`shared.db.schema`) -- bbox recortado del evento, fechas, y
un `source` fijo identificando que viene del clustering de FIRMS (no de
un incendio catalogado por CONAF/SENAPRED, que usaría otro `source`)."""
import datetime as dt

from sqlalchemy import Connection, Engine, insert

from shared.db.schema import fire_event


def persist_fire_event_metadata(
    engine: Engine | Connection,
    event_id: int,
    bbox_cut: tuple[float, float, float, float],
    start_date: dt.date,
    end_date: dt.date,
    source: str = "firms_cluster",
) -> int:
    west, south, east, north = bbox_cut
    wkt = (
        f"SRID=4326;POLYGON(({west} {south}, {west} {north}, "
        f"{east} {north}, {east} {south}, {west} {south}))"
    )
    values = {
        "bbox": f"{west},{south},{east},{north}",
        "start_date": start_date,
        "end_date": end_date,
        "source": source,
        "geom": wkt,
    }
    stmt = insert(fire_event).values(**values).returning(fire_event.c.id)
    if isinstance(engine, Engine):
        with engine.begin() as conn:
            row_id: int = conn.execute(stmt).scalar_one()
            return row_id
    row_id = engine.execute(stmt).scalar_one()
    return row_id
```

Update `features/src/features/dataset/__init__.py` docstring is unchanged
(already written in Task 3).

- [ ] **Step 4: Run test to verify it passes (or skips cleanly)**

Run: `uv run --package features pytest features/tests/test_dataset_db.py -v`
Expected: `1 skipped` (no `POSTGRES_*` etc. in this environment) — this
IS the expected passing state; if a real Postgres container with those
env vars is available, expect `1 passed` instead.

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add features/src/features/dataset/db.py features/tests/test_dataset_db.py
git commit -m "feat: persist fire-event metadata to PostGIS (features/dataset)"
```

---

## Task 7: `features/dataset/pipeline.py` — per-event orchestration

**Files:**
- Create: `features/src/features/dataset/pipeline.py`
- Test: `features/tests/test_dataset_pipeline.py`

**Interfaces:**
- Consumes: `features.fire_state.clustering.FireEvent`,
  `features.fire_state.rasterize.build_fire_state`,
  `features.grid.grid.build_grid`, `features.dataset.resample.resample_to_grid`,
  `features.dataset.assemble.{EventChannels, assemble_event_tensor,
  CHANNEL_ORDER}`.
- Produces: `EventSources` (frozen dataclass: `elevation_path: Path`,
  `slope_path: Path`, `aspect_path: Path`, `fuel_type_path: Path`,
  `ndvi_paths_by_month: dict[str, Path]`, `weather_paths_by_day:
  dict[date, dict[str, Path]]`); `resolve_event_sources(days: list[date],
  settings: Settings) -> EventSources`; `build_dataset_for_event(event:
  FireEvent, sources: EventSources, resolution_m: float, crs: str,
  pre_event_padding_days: int = DEFAULT_PRE_EVENT_PADDING_DAYS, fire_buffer_m:
  float = DEFAULT_BUFFER_M) -> tuple[xr.DataArray, tuple[float,float,float,float]]`
  (tensor, event bbox in WGS84). Task 9 (CLI) consumes both
  `resolve_event_sources` and `build_dataset_for_event`.

- [ ] **Step 1: Write the failing tests**

Create `features/tests/test_dataset_pipeline.py`:

```python
"""Tests de la orquestación por evento: bbox recortado, ventana de
padding previo, resolución de fuentes por convención de archivos, y
ensamblado end-to-end con rasters de fixture."""
import datetime as dt
from pathlib import Path

import numpy as np
import rasterio
from features.dataset.assemble import CHANNEL_ORDER
from features.dataset.pipeline import (
    DEFAULT_PRE_EVENT_PADDING_DAYS,
    EventSources,
    _event_bbox_wgs84,
    _nearest_month_path,
    build_dataset_for_event,
    resolve_event_sources,
)
from features.fire_state.clustering import FireEvent
from rasterio.transform import from_origin
from shared.schemas import FireDetection


def _det(lat: float, lon: float, at: dt.datetime) -> FireDetection:
    return FireDetection(
        latitude=lat, longitude=lon, detected_at=at, frp=1.0,
        confidence="n", satellite="N", instrument="VIIRS",
    )


def test_event_bbox_wgs84_pads_around_single_detection_without_crashing():
    # una detección sola tiene extensión espacial CERO antes del buffer --
    # no debe dividir por cero ni producir un bbox degenerado.
    at = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    event = FireEvent(event_id=0, detections=(_det(-38.0, -72.5, at),))
    west, south, east, north = _event_bbox_wgs84(event.detections, "EPSG:32719", buffer_m=1000.0)
    assert west < east
    assert south < north
    assert west < -72.5 < east
    assert south < -38.0 < north


def test_nearest_month_path_picks_closest_available_month():
    paths = {"2025-11": Path("nov.tif"), "2026-02": Path("feb.tif")}
    # "2026-01" está a 2 meses de nov (2025-11) y 1 mes de feb (2026-02)
    assert _nearest_month_path(paths, "2026-01") == Path("feb.tif")


def test_nearest_month_path_returns_none_when_no_months_available():
    assert _nearest_month_path({}, "2026-01") is None


def _write_tif(path: Path, value: float, size: int, transform, crs: str, nodata=-9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((size, size), value, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_resolve_event_sources_finds_files_by_convention(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    transform = from_origin(190000, 5791000, 250, 250)
    _write_tif(processed / "dem" / "dem_abc123.tif", 100.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "terrain" / "slope_deg.tif", 5.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "terrain" / "aspect_deg.tif", 180.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "vegetation" / "fuel_type.tif", 3.0, 20, transform, "EPSG:32719")
    _write_tif(processed / "vegetation" / "ndvi_2026-01.tif", 0.5, 20, transform, "EPSG:32719")
    day = dt.date(2026, 1, 15)
    for field in ("wind_u", "wind_v", "temperature", "relative_humidity", "precipitation"):
        _write_tif(processed / "weather" / f"{field}_{day.isoformat()}.tif", 1.0, 20, transform, "EPSG:32719")

    from shared.config import Settings

    settings = Settings(
        firms_map_key="x", cds_api_url="x", cds_api_key="x",
        copernicus_dataspace_client_id="x", copernicus_dataspace_client_secret="x",
        postgres_host="x", postgres_port=5432, postgres_db="x", postgres_user="x",
        postgres_password="x", data_processed_dir=processed,
    )
    sources = resolve_event_sources([day], settings)
    assert sources.elevation_path == processed / "dem" / "dem_abc123.tif"
    assert sources.slope_path.name == "slope_deg.tif"
    assert sources.ndvi_paths_by_month == {"2026-01": processed / "vegetation" / "ndvi_2026-01.tif"}
    assert day in sources.weather_paths_by_day
    assert set(sources.weather_paths_by_day[day]) == {
        "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation"
    }


def test_build_dataset_for_event_pads_before_first_detection_and_fills_fire_mask(tmp_path):
    processed = tmp_path / "processed"
    resolution_m = 250.0
    crs = "EPSG:32719"

    day1 = dt.datetime(2026, 1, 10, 12, 0, tzinfo=dt.UTC)
    day2 = dt.datetime(2026, 1, 11, 12, 0, tzinfo=dt.UTC)
    event = FireEvent(
        event_id=3,
        detections=(_det(-38.0, -72.5, day1), _det(-38.0, -72.5, day2)),
    )
    event_bbox = _event_bbox_wgs84(event.detections, crs, buffer_m=1500.0)
    grid = _grid_for_test(event_bbox, crs, resolution_m)
    transform, size = grid.transform, max(grid.height, grid.width)

    _write_tif(processed / "dem" / "dem_x.tif", 100.0, size, transform, crs)
    _write_tif(processed / "terrain" / "slope_deg.tif", 5.0, size, transform, crs)
    _write_tif(processed / "terrain" / "aspect_deg.tif", 180.0, size, transform, crs)
    _write_tif(processed / "vegetation" / "fuel_type.tif", 3.0, size, transform, crs)
    _write_tif(processed / "vegetation" / "ndvi_2026-01.tif", 0.5, size, transform, crs)

    padded_start = event.start_date - dt.timedelta(days=DEFAULT_PRE_EVENT_PADDING_DAYS)
    all_days = [padded_start + dt.timedelta(days=i) for i in range((event.end_date - padded_start).days + 1)]
    for day in all_days:
        for field in ("wind_u", "wind_v", "temperature", "relative_humidity", "precipitation"):
            _write_tif(
                processed / "weather" / f"{field}_{day.isoformat()}.tif", 1.0, size, transform, crs
            )

    from shared.config import Settings

    settings = Settings(
        firms_map_key="x", cds_api_url="x", cds_api_key="x",
        copernicus_dataspace_client_id="x", copernicus_dataspace_client_secret="x",
        postgres_host="x", postgres_port=5432, postgres_db="x", postgres_user="x",
        postgres_password="x", data_processed_dir=processed,
    )
    sources = resolve_event_sources(all_days, settings)
    tensor, bbox_cut = build_dataset_for_event(event, sources, resolution_m, crs)

    assert tensor.dims == ("day", "channel", "y", "x")
    assert list(tensor.coords["channel"].values) == list(CHANNEL_ORDER)
    assert len(tensor.coords["day"]) == len(all_days)

    fire_mask_idx = CHANNEL_ORDER.index("fire_mask")
    first_day_iso = padded_start.isoformat()
    day_index = list(tensor.coords["day"].values).index(first_day_iso)
    # día de padding (antes de la primera detección real) -- sin fuego.
    assert np.all(tensor.values[day_index, fire_mask_idx, :, :] == 0.0)


def _grid_for_test(bbox, crs, resolution_m):
    from features.grid.grid import build_grid

    return build_grid(bbox, crs, resolution_m)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package features pytest features/tests/test_dataset_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.dataset.pipeline'`

- [ ] **Step 3: Write minimal implementation**

Create `features/src/features/dataset/pipeline.py`:

```python
"""Orquesta, POR EVENTO: bbox recortado (detecciones + buffer) -> grilla
propia del evento -> resampleo de cada fuente ya procesada -> máscara de
fuego diaria -> tensor. No descarga ni procesa nada de cero: asume que
`pyrocast-ingest dem/era5/sentinel2/worldcover` ya corrieron para (al
menos) el rango de fechas del evento MÁS su padding previo -- ver
docs/dataset-card.md."""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer
from rasterio.warp import Resampling

from features.dataset.assemble import EventChannels, assemble_event_tensor
from features.dataset.resample import resample_to_grid
from features.fire_state.clustering import FireEvent
from features.fire_state.rasterize import DEFAULT_BUFFER_M, build_fire_state
from features.grid.grid import build_grid
from shared.config import Settings

DEFAULT_PRE_EVENT_PADDING_DAYS = 5
# "unos días antes" en el enunciado no da un valor -- heurística sin
# calibrar contra incendios reales, igual que los defaults de
# features/fire_state (ver docs/dataset-card.md).

_WEATHER_FIELDS: tuple[str, ...] = (
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
)


@dataclass(frozen=True)
class EventSources:
    elevation_path: Path
    slope_path: Path
    aspect_path: Path
    fuel_type_path: Path
    ndvi_paths_by_month: dict[str, Path]
    weather_paths_by_day: dict[dt.date, dict[str, Path]]


def _event_bbox_wgs84(
    detections: tuple, crs: str, buffer_m: float
) -> tuple[float, float, float, float]:
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs: list[float] = []
    ys: list[float] = []
    for detection in detections:
        x, y = to_crs.transform(detection.longitude, detection.latitude)
        xs.append(x)
        ys.append(y)
    west, south = min(xs) - buffer_m, min(ys) - buffer_m
    east, north = max(xs) + buffer_m, max(ys) + buffer_m
    lon_w, lat_s = to_wgs84.transform(west, south)
    lon_e, lat_n = to_wgs84.transform(east, north)
    return (lon_w, lat_s, lon_e, lat_n)


def _nearest_month_path(paths_by_month: dict[str, Path], target_month: str) -> Path | None:
    if not paths_by_month:
        return None

    def month_index(key: str) -> int:
        year_str, month_str = key.split("-")
        return int(year_str) * 12 + int(month_str)

    target_idx = month_index(target_month)
    closest_key = min(paths_by_month, key=lambda k: abs(month_index(k) - target_idx))
    return paths_by_month[closest_key]


def _single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Se esperaba exactamente 1 archivo en {directory} que matcheara "
            f"{pattern!r}, se encontraron {len(matches)}. Convención de un "
            f"único estudio de área a la vez -- ver docs/limitations.md."
        )
    return matches[0]


def resolve_event_sources(days: list[dt.date], settings: Settings) -> EventSources:
    processed = settings.data_processed_dir
    ndvi_paths_by_month = {
        p.stem.removeprefix("ndvi_"): p
        for p in (processed / "vegetation").glob("ndvi_*.tif")
    }
    weather_paths_by_day: dict[dt.date, dict[str, Path]] = {}
    for day in days:
        iso = day.isoformat()
        candidate = {
            field: processed / "weather" / f"{field}_{iso}.tif" for field in _WEATHER_FIELDS
        }
        if all(path.exists() for path in candidate.values()):
            weather_paths_by_day[day] = candidate

    return EventSources(
        elevation_path=_single_file(processed / "dem", "*.tif"),
        slope_path=processed / "terrain" / "slope_deg.tif",
        aspect_path=processed / "terrain" / "aspect_deg.tif",
        fuel_type_path=processed / "vegetation" / "fuel_type.tif",
        ndvi_paths_by_month=ndvi_paths_by_month,
        weather_paths_by_day=weather_paths_by_day,
    )


def build_dataset_for_event(
    event: FireEvent,
    sources: EventSources,
    resolution_m: float,
    crs: str,
    pre_event_padding_days: int = DEFAULT_PRE_EVENT_PADDING_DAYS,
    fire_buffer_m: float = DEFAULT_BUFFER_M,
) -> tuple[xr.DataArray, tuple[float, float, float, float]]:
    padded_start = event.start_date - dt.timedelta(days=pre_event_padding_days)
    total_days = (event.end_date - padded_start).days + 1
    days = tuple(padded_start + dt.timedelta(days=i) for i in range(total_days))

    event_bbox = _event_bbox_wgs84(event.detections, crs, buffer_m=fire_buffer_m)
    grid = build_grid(event_bbox, crs, resolution_m)

    static = {
        "elevation": resample_to_grid(sources.elevation_path, grid, Resampling.bilinear),
        "slope_deg": resample_to_grid(sources.slope_path, grid, Resampling.bilinear),
        "aspect_deg": resample_to_grid(sources.aspect_path, grid, Resampling.bilinear),
        "fuel_type": resample_to_grid(sources.fuel_type_path, grid, Resampling.nearest),
    }

    fire_masks = build_fire_state(event, grid, buffer_m=fire_buffer_m)

    dynamic: dict[str, dict[dt.date, np.ndarray]] = {field: {} for field in _WEATHER_FIELDS}
    dynamic["ndvi"] = {}
    dynamic["fire_mask"] = {}
    for day in days:
        weather_for_day = sources.weather_paths_by_day.get(day)
        for field in _WEATHER_FIELDS:
            path = weather_for_day[field] if weather_for_day else None
            dynamic[field][day] = (
                resample_to_grid(path, grid, Resampling.bilinear)
                if path is not None
                else _nan_array(grid.height, grid.width)
            )
        month_key = f"{day.year:04d}-{day.month:02d}"
        ndvi_path = _nearest_month_path(sources.ndvi_paths_by_month, month_key)
        dynamic["ndvi"][day] = (
            resample_to_grid(ndvi_path, grid, Resampling.bilinear)
            if ndvi_path is not None
            else _nan_array(grid.height, grid.width)
        )
        fire_mask = fire_masks.get(day)
        dynamic["fire_mask"][day] = (
            fire_mask.astype("float32") if fire_mask is not None else _zeros_array(grid.height, grid.width)
        )

    channels = EventChannels(days=days, static=static, dynamic=dynamic)
    tensor = assemble_event_tensor(channels)
    return tensor, grid.bounds


def _nan_array(height: int, width: int) -> np.ndarray:
    return np.full((height, width), np.nan, dtype="float32")


def _zeros_array(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype="float32")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package features pytest features/tests/test_dataset_pipeline.py -v`
Expected: `5 passed`

If `test_build_dataset_for_event_pads_before_first_detection_and_fills_fire_mask`
fails because the fixture rasters don't fully cover the event grid's
bounds after reprojection (leaving `NaN` where the test expects `0.0` for
`fire_mask` specifically) — note `fire_mask` never goes through
`resample_to_grid` (it's produced directly at the event grid's own
resolution/transform by `build_fire_state`, no reprojection involved), so
this failure mode cannot apply to that specific assertion; if it applies
to a different assertion, re-derive the fixture raster's required extent
(`size = max(grid.height, grid.width)` combined with `transform =
grid.transform` should always be big enough since the fixture is
generated directly from the SAME grid the code under test builds) and fix
the fixture, not the implementation.

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean (this pattern — a precisely-typed `dict[str,
dict[dt.date, np.ndarray]]` built incrementally, `np` imported at module
level — was pre-verified against `mypy --strict` before this plan was
written).

- [ ] **Step 6: Commit**

```bash
git add features/src/features/dataset/pipeline.py features/tests/test_dataset_pipeline.py
git commit -m "feat: add per-event dataset orchestration (features/dataset/pipeline.py)"
```

---

## Task 8: `pyrocast-features build-dataset` CLI + Makefile target

**Files:**
- Create: `features/src/features/cli.py`
- Modify: `features/pyproject.toml` (add `[project.scripts]`)
- Modify: `Makefile`
- Test: `features/tests/test_dataset_cli.py`

**Interfaces:**
- Consumes: `features.dataset.firms_loader.load_firms_detections`,
  `features.fire_state.clustering.build_fire_events`,
  `features.dataset.pipeline.{resolve_event_sources, build_dataset_for_event}`,
  `features.dataset.assemble.save_event_to_zarr`,
  `features.dataset.split.split_events`,
  `features.dataset.db.persist_fire_event_metadata`.
- Produces: the `pyrocast-features` console script with a `build-dataset`
  command; `data/processed/dataset/event_NNNN.zarr` per event;
  `data/processed/dataset/splits.json`.

- [ ] **Step 1: Write the failing test**

Create `features/tests/test_dataset_cli.py`:

```python
"""Test de humo del CLI `pyrocast-features build-dataset`: sin red, sin
Postgres real -- cada colaborador externo está monkeypatcheado con datos
de fixture, pero el ensamblado, el Zarr y el split corren de verdad."""
import datetime as dt
import json

import numpy as np
import rasterio
from features.cli import app
from rasterio.transform import from_origin
from shared.schemas import FireDetection
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}


def _det(lat: float, lon: float, at: dt.datetime) -> FireDetection:
    return FireDetection(
        latitude=lat, longitude=lon, detected_at=at, frp=1.0,
        confidence="n", satellite="N", instrument="VIIRS",
    )


def test_build_dataset_cli_produces_at_least_one_zarr_event(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    at = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    fixture_detections = [_det(-38.0, -72.5, at)]
    monkeypatch.setattr(
        "features.cli.load_firms_detections", lambda base_dir, start, end: fixture_detections
    )

    def fake_resolve_event_sources(days, settings_arg):
        processed = settings_arg.data_processed_dir
        size = 20
        transform = from_origin(190000, 5791000, 250, 250)
        for subdir, name in [
            ("dem", "dem_x.tif"), ("terrain", "slope_deg.tif"), ("terrain", "aspect_deg.tif"),
            ("vegetation", "fuel_type.tif"), ("vegetation", "ndvi_2026-01.tif"),
        ]:
            path = processed / subdir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                path, "w", driver="GTiff", height=size, width=size, count=1,
                dtype="float32", crs="EPSG:32719", transform=transform, nodata=-9999.0,
            ) as dst:
                dst.write(np.ones((size, size), dtype="float32"), 1)

        from features.dataset.pipeline import EventSources

        return EventSources(
            elevation_path=processed / "dem" / "dem_x.tif",
            slope_path=processed / "terrain" / "slope_deg.tif",
            aspect_path=processed / "terrain" / "aspect_deg.tif",
            fuel_type_path=processed / "vegetation" / "fuel_type.tif",
            ndvi_paths_by_month={"2026-01": processed / "vegetation" / "ndvi_2026-01.tif"},
            weather_paths_by_day={},
        )

    monkeypatch.setattr("features.cli.resolve_event_sources", fake_resolve_event_sources)
    monkeypatch.setattr(
        "features.cli.persist_fire_event_metadata",
        lambda **kwargs: 1,
    )

    result = runner.invoke(app, ["build-dataset", "--start", "2026-01-10", "--end", "2026-01-15"])
    assert result.exit_code == 0, result.output

    dataset_dir = settings.data_processed_dir / "dataset"
    zarr_dirs = list(dataset_dir.glob("event_*.zarr"))
    assert len(zarr_dirs) >= 1

    splits_path = dataset_dir / "splits.json"
    assert splits_path.exists()
    splits = json.loads(splits_path.read_text())
    assert set(splits) == {"train", "val", "test"}
    get_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --package features pytest features/tests/test_dataset_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.cli'`

- [ ] **Step 3: Add `[project.scripts]` to `features/pyproject.toml`**

```toml
[project.scripts]
pyrocast-features = "features.cli:app"
```

(placed after the `dependencies` list, before `[tool.uv.sources]`).

- [ ] **Step 4: Write minimal implementation**

Create `features/src/features/cli.py`:

```python
"""Punto de entrada del CLI de features: `pyrocast-features`."""
import datetime as dt
import json

import typer
from sqlalchemy import create_engine

from features.dataset.assemble import save_event_to_zarr
from features.dataset.db import persist_fire_event_metadata
from features.dataset.firms_loader import load_firms_detections
from features.dataset.pipeline import build_dataset_for_event, resolve_event_sources
from features.dataset.split import split_events
from features.fire_state.clustering import build_fire_events
from shared.config import get_settings

app = typer.Typer()


@app.callback()
def _callback() -> None:
    """CLI de features de PyroCast."""


def build_dataset(
    start: dt.datetime = typer.Option(
        ..., formats=["%Y-%m-%d"], help="Fecha de inicio (YYYY-MM-DD)"
    ),
    end: dt.datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Fecha de fin (YYYY-MM-DD)"),
) -> None:
    """Ensambla el dataset espaciotemporal por evento (P2-P6): clusteriza
    detecciones FIRMS ya ingeridas en eventos, arma el tensor de cada uno
    a partir de las capas ya procesadas (DEM/ERA5-Land/Sentinel-2/
    WorldCover -- deben haberse ingerido antes con `pyrocast-ingest`, ver
    docs/dataset-card.md), lo persiste en Zarr, guarda sus metadatos en
    PostGIS, y calcula el split train/val/test reproducible."""
    settings = get_settings()

    detections = load_firms_detections(settings.data_raw_dir, start.date(), end.date())
    events = build_fire_events(detections)
    if not events:
        typer.echo("No se encontraron eventos de incendio en el rango pedido.")
        raise typer.Exit(code=0)

    output_dir = settings.data_processed_dir / "dataset"
    engine = create_engine(settings.postgres_dsn)

    event_ids: list[int] = []
    for event in events:
        padded_start = event.start_date - dt.timedelta(days=5)
        days = [
            padded_start + dt.timedelta(days=i)
            for i in range((event.end_date - padded_start).days + 1)
        ]
        sources = resolve_event_sources(days, settings)
        tensor, bbox_cut = build_dataset_for_event(
            event, sources, settings.spatial_resolution_m, settings.crs
        )
        zarr_path = save_event_to_zarr(tensor, output_dir, event.event_id)
        persist_fire_event_metadata(
            engine=engine,
            event_id=event.event_id,
            bbox_cut=bbox_cut,
            start_date=event.start_date,
            end_date=event.end_date,
        )
        event_ids.append(event.event_id)
        typer.echo(f"Evento {event.event_id}: {zarr_path}")

    splits = split_events(event_ids)
    (output_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    typer.echo(
        f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )


app.command("build-dataset")(build_dataset)
```

Update `Makefile`'s `build-dataset` target:

```makefile
build-dataset:
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then \
		echo "uso: make build-dataset START=YYYY-MM-DD END=YYYY-MM-DD"; \
		exit 1; \
	fi
	uv run --package features pyrocast-features build-dataset --start $(START) --end $(END)
```

(replace the old `@echo "pendiente: features/dataset aún no implementado"`
line, and add `build-dataset` to the `.PHONY` list at the top of the
Makefile if not already present — it already is, from the bootstrap.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run --package features pytest features/tests/test_dataset_cli.py -v`
Expected: PASS. `uv sync --all-packages` first if `pyrocast-features` the
console script itself needs to be (re)installed into the venv's entry
points — the `CliRunner` test imports `features.cli:app` directly and
does not need the console script installed to pass, but a REAL manual
verification of the installed script (Step 7 below) does.

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean

- [ ] **Step 7: Manually verify the installed console script**

Run:
```bash
uv sync --all-packages
uv run --package features pyrocast-features --help
uv run --package features pyrocast-features build-dataset --help
```
Expected: both show a working Typer help screen (`build-dataset` listed
as a command; `--start`/`--end` listed as required options).

- [ ] **Step 8: Commit**

```bash
git add features/src/features/cli.py features/pyproject.toml features/tests/test_dataset_cli.py Makefile uv.lock
git commit -m "feat: add pyrocast-features CLI with build-dataset command"
```

---

## Task 9: Documentation — `docs/dataset-card.md`, `docs/decisions.md`, `docs/limitations.md`

**Files:**
- Create: `docs/dataset-card.md`
- Modify: `docs/decisions.md`
- Modify: `docs/limitations.md`

**Interfaces:**
- Consumes: the final shipped behavior from Tasks 1–8 — no code
  interface, this task only writes prose that must match the shipped
  code exactly.

- [ ] **Step 1: Write `docs/dataset-card.md`**

```markdown
# Dataset card: tensores espaciotemporales por evento de incendio

## Qué es

Cada evento de incendio (`features/fire_state/clustering.py`) se
ensambla en UN tensor 4D `(día, canal, alto, ancho)`, persistido como un
array Zarr en `data/processed/dataset/event_NNNN.zarr` (`NNNN` = id de
evento, 4 dígitos con cero a la izquierda). Los metadatos del evento
(bbox recortado, fecha de inicio/fin, id) se persisten además en la
tabla `fire_event` de PostGIS.

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
`features/fire_state/rasterize.py`) — no la grilla de todo el área de
estudio. El "bbox recortado" persistido en `fire_event.bbox` es
exactamente ese bbox por evento (WGS84, `west,south,east,north`), y el
tensor cubre solo esa área — mucho más chico que Biobío+Ñuble+Araucanía
completos.

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
```

- [ ] **Step 2: Append to `docs/decisions.md`**

```markdown
## `features/dataset/` no depende de `ingestion`

`features/dataset/firms_loader.py` re-implementa un lector mínimo del
parquet crudo de FIRMS (mismo formato que `ingestion/firms/storage.py`
ya escribe) en vez de importar `ingestion.firms.storage` directamente.
La dirección de dependencia establecida en este proyecto es `ingestion`
→ `features` (excepción ya aceptada y documentada para
`ingestion/dem/cli.py` y `ingestion/era5/cli.py`); invertirla para que
`features` dependiera de `ingestion` habría sido la primera vez que esa
regla se rompe en sentido contrario. Duplicar ~20 líneas de lectura de un
formato de almacenamiento simple es un costo menor que esa inversión.

## `pyrocast-features build-dataset` no re-ingiere P1-P4

El enunciado pide un CLI que corra "todo el pipeline de features (P2-P6)
de punta a punta" — se interpretó como el pipeline de FEATURES (terreno,
clima, vegetación, grilla+eventos, ensamblado), no como re-disparar las
descargas crudas de ingesta (DEM/ERA5-Land/Sentinel-2/WorldCover/FIRMS,
que ya tienen sus propios comandos `pyrocast-ingest`). `build-dataset`
asume que esos comandos ya corrieron para el rango de fechas pedido (más
el padding previo al evento) y lee sus salidas ya procesadas por
convención de nombre de archivo — documentado explícitamente como
precondición operativa en `docs/dataset-card.md`, no una limitación
oculta.

## Corrección de nombre de archivo NDVI (`ingestion/sentinel2/cli.py`)

Se encontró, al diseñar la búsqueda de "composite mensual más cercano"
de NDVI, que `compute_and_save_vegetation` siempre escribe `ndvi.tif` —
un segundo mes ingerido sobrescribía silenciosamente el NDVI del mes
anterior, dejando como máximo UN mes disponible en disco en cualquier
momento. Corregido en `ingestion/sentinel2/cli.py` (no en
`features/vegetation/ndvi.py`, que no necesita saber qué mes calculó):
el CLI renombra el resultado a `ndvi_YYYY-MM.tif` inmediatamente después
de escribirlo. Encontrado y corregido durante la implementación de
`features/dataset/`, no en una revisión final posterior.

## `features/grid/` finalmente en uso real (`features/dataset/resample.py`)

`features/grid/build_grid` ya existía pero ningún pipeline lo usaba
todavía (ver la entrada anterior sobre su estado de adopción). En vez de
migrar los cuatro pipelines de ingesta existentes a la grilla canónica
del área de estudio completa, `features/dataset/` construye una `WorkGrid`
POR EVENTO (bbox recortado a las detecciones + buffer, no el bbox de
estudio completo) y resamplea la salida YA PROCESADA de cada fuente sobre
esa grilla (`features/dataset/resample.py::resample_to_grid`) — logra la
alineación píxel-a-píxel que el consumidor real necesita sin tocar los
cuatro pipelines existentes. Migrar esos pipelines a compartir una única
grilla de estudio completa (en vez de que cada evento tenga la suya) sigue
diferido, y ahora es estrictamente una optimización/limpieza, no un
bloqueante funcional.
```

- [ ] **Step 3: Append to `docs/limitations.md`**

```markdown
- **`features/dataset/`: un único DEM/estudio de área asumido**:
  `resolve_event_sources` espera exactamente un archivo bajo
  `data_processed_dir/dem/` (convención de nombre con hash de
  bbox+resolución+CRS, un único estudio de área configurado a la vez) —
  levanta un error claro si encuentra 0 o más de 1, pero no soporta
  múltiples estudios de área simultáneos.
- **`features/dataset/`: canales sin cobertura se rellenan con NaN, sin
  error explícito**: si el rango de fechas ingerido con `pyrocast-ingest`
  no cubre los días de padding previos a un evento (o el mes de NDVI más
  cercano no existe en absoluto), el canal correspondiente queda en NaN
  para esos días — un dataset con muchos NaN no falla ruidosamente, hay
  que inspeccionarlo. Ver `docs/dataset-card.md`.
- **`features/dataset/`: cada evento tiene su propia grilla, no la grilla
  de estudio completa**: dos tensores de eventos distintos no son
  comparables píxel a píxel sin un paso de reproyección adicional — ver
  `docs/dataset-card.md`.
```

- [ ] **Step 4: Run the full verification sweep**

Run:
```bash
env -i PATH="$PATH" HOME="$HOME" uv run --package features pytest features/tests -v
env -i PATH="$PATH" HOME="$HOME" uv run --package ingestion pytest ingestion/tests -v
uv run ruff check .
uv run mypy --strict shared/src features/src
```
Expected: all green/clean.

- [ ] **Step 5: Commit**

```bash
git add docs/dataset-card.md docs/decisions.md docs/limitations.md
git commit -m "docs: add dataset-card.md, document features/dataset design decisions"
```
