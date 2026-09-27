# features/grid/ + features/fire_state/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the canonical work grid (`features/grid/`) and the fire-event
reconstruction pipeline (`features/fire_state/`) — clustering FIRMS
detections into spatiotemporal events and rasterizing each event to a daily
binary fire mask on the grid.

**Architecture:** `features/grid/grid.py` defines `WorkGrid` (CRS + Affine
transform + width/height), built deterministically from a bbox (WGS84) + CRS
+ resolution by reprojecting the bbox and snapping it outward to whole
pixels — no dependency on any source raster's own mosaic bounds, unlike the
existing DEM/ERA5/Sentinel-2/WorldCover pipelines (that gap is the known,
already-documented limitation this module starts closing; migrating the
existing pipelines onto it is out of scope here, see Global Constraints).
`features/fire_state/clustering.py` groups `FireDetection` records into
`FireEvent`s with a union-find over a spatial+temporal neighbor relation
(chaining, like DBSCAN with `min_samples=1`). `features/fire_state/rasterize.py`
turns one `FireEvent` into a `dict[date, np.ndarray]` binary mask cube: a
fixed-radius buffer around each detection for days with real detections,
linear interpolation of the binary indicator (equivalent to unioning the
nearest anchor masks) for gap days inside the event's date range.

**Tech Stack:** `numpy`, `rasterio` (`features.rasterize`, `warp.transform_bounds`,
`transform.from_origin`), `xarray`/`rioxarray` (grid template), `shapely`
(point buffering), `pyproj` (WGS84→CRS reprojection of detections) — all
already features-package dependencies except `pyproj`, added explicitly in
Task 1 (see Global Constraints).

**Spec:** the user's request (quoted below), governed by `/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md`.

```
Implementa features/grid/ y features/fire_state/.

1. features/grid/: define la grilla de trabajo (bbox, CRS EPSG:32719,
   resolución) como un único objeto reutilizable (por ejemplo con
   xarray/rioxarray), del que todos los módulos de ingesta y features
   dependan para reproyectar. Debe poder recrearse de forma determinista a
   partir de shared/config.py (misma configuración → mismos límites de
   grilla, mismo número de celdas).
2. features/fire_state/: a partir de las detecciones de FIRMS (P1), define
   "evento de incendio" como un clúster espaciotemporal de detecciones (usa
   DBSCAN u otro método simple sobre coordenadas + tiempo, con parámetros
   configurables y documentados). Para cada evento, rasteriza las
   detecciones diarias en una máscara binaria de fuego sobre la grilla
   común.
3. Documenta explícitamente que esta reconstrucción es una simplificación
   frente al kriging usado en la literatura (WildfireCube): aquí se usa un
   buffer espacial simple alrededor de cada detección más interpolación
   temporal lineal entre días con detección, y se anota como limitación
   conocida en docs/limitations.md.
4. Tests: clustering con casos sintéticos conocidos (dos incendios
   claramente separados en espacio/tiempo deben quedar en dos eventos
   distintos; detecciones contiguas en el mismo incendio deben unirse),
   rasterización verificada contra un caso de coordenadas conocidas,
   determinismo de la grilla (dos ejecuciones con la misma config producen
   el mismo objeto de grilla).

Criterios de aceptación: mypy --strict y ruff limpios; tests verdes;
docs/fire-events.md explicando la definición de "evento" y sus parámetros.
```

## Global Constraints

- Python 3.12, `mypy --strict` on `features/src` (already enforced by
  `make typecheck`), `ruff check .` clean repo-wide, tests via `pytest`, no
  real network calls in any test.
- `pyproj` is added as an explicit direct dependency of `features`
  (`features/pyproject.toml`) — it was already installed transitively via
  `rasterio`, but `fire_state/rasterize.py` imports it directly, so it must
  be declared, not relied on as undeclared transitive magic. Documented in
  `docs/decisions.md` (Task 4) — this is a declaration of an already-present
  library, not a new dependency footprint.
- Clustering uses a **hand-written union-find**, not `scikit-learn`'s
  `DBSCAN` — the spec explicitly allows "DBSCAN u otro método simple".
  `scikit-learn` is a real project dependency (used later by
  `models/evaluation` for isotonic calibration per CLAUDE.md) but is not a
  `features`-package dependency today, and pulling it in for one ~30-line
  algorithm is not justified when detection volumes for this project (a few
  thousand per season in the study area) make an O(n²) union-find entirely
  adequate. Documented in `docs/decisions.md` (Task 4).
- Neither module gets a CLI command or Makefile target in this plan.
  `features/terrain` and `features/weather` (existing precedent) also have
  none — they're invoked as a computation step from inside
  `ingestion/dem/cli.py`/`ingestion/era5/cli.py` right after that source's
  own fetch. `features/fire_state` has no equivalent single-fetch moment to
  hook into: clustering needs the *accumulated* detection history across
  potentially many `ingest-firms` runs, not one chunk's response. Wiring it
  into a real pipeline belongs to `features/dataset/` (already stubbed as
  "pendiente" in the Makefile) — document this in `docs/decisions.md`
  (Task 4) so it reads as a stated plan, not an oversight.
- `features/fire_state` does not persist anything to disk in this plan
  (no Zarr/GeoTIFF writer). The spec's acceptance criteria only ask for the
  in-memory clustering + rasterization behavior and its tests/docs;
  inventing a persistence schema now (format, path layout, chunking) would
  be built without the `features/dataset/` consumer that will actually
  dictate those requirements. Documented as a deferred decision in
  `docs/decisions.md` (Task 4).
- `WorkGrid` is intentionally NOT wired into the four existing ingestion
  pipelines (DEM/ERA5-Land/Sentinel-2/WorldCover) in this plan — each still
  reprojects independently from its own mosaic's bounds, a limitation
  already documented in `docs/decisions.md`'s "Cierre de Etapa 1" section
  and `docs/limitations.md`. This plan makes the canonical grid *exist and
  be correct*; migrating existing pipelines onto it is separate,
  higher-risk work for a future request.
- `fill_temporal_gaps` (Task 3) never extrapolates: a gap day with no
  anchor detection day on one side (before the event's first detection, or
  after its last) is left as an all-`False` mask, never guessed from a
  single-sided neighbor.

## Review Focus

- **A detection list clustered with the default temporal_eps spanning a
  year boundary (Dec 31 → Jan 1)**: `dt.timedelta` arithmetic on
  timezone-aware `datetime` must not silently misbehave across a UTC day
  rollover — covered by using real `datetime` subtraction (never manual
  date-string parsing) throughout `clustering.py`.
- **A `FireEvent` with a single detection (cluster of size 1)**: must not
  crash `start_date`/`end_date` (both should just equal that single day) or
  `rasterize_daily_masks`/`fill_temporal_gaps` (a 1-day event has no gap
  days to fill at all) — covered by Task 2's clustering tests implicitly
  producing singleton events, and Task 3's rasterize test using a
  single-detection event.
- **`build_grid` called with a bbox whose reprojected extent is already an
  exact multiple of the resolution** (no snapping needed): `floor`/`ceil`
  must not add a spurious extra row/column of padding in that exact case —
  covered by a dedicated Task 1 test.
- **`fill_temporal_gaps` where the SAME pixel is `True` in both the
  previous and next anchor day**: the linear interpolation must still
  resolve to `True` for that pixel on the gap day (not some fractional
  value in limbo) — covered explicitly by Task 3's gap-fill test using a
  pixel present in both anchors alongside ones present in only one.
- **Two detections at the exact same instant and the exact same
  coordinates** (a duplicate row in the FIRMS CSV, which happens with real
  overlapping satellite swaths): `_haversine_m` must return `0.0` without a
  `math.asin` domain error from floating-point overshoot past `1.0` —
  covered by a dedicated Task 2 test.

---

## Task 1: `features/grid/` — canonical WorkGrid

**Files:**
- Create: `features/src/features/grid/grid.py`
- Modify: `features/src/features/grid/__init__.py`
- Test: `features/tests/test_grid.py`
- Modify: `features/pyproject.toml` (add `pyproj>=3.6` — used by Task 3, declared here since Task 1 lands first and the dependency is project-wide for the `features` package)

**Interfaces:**
- Produces: `WorkGrid` (frozen dataclass: `crs: str`, `transform: Affine`,
  `width: int`, `height: int`, `resolution_m: float`; property
  `bounds -> tuple[float, float, float, float]`), `build_grid(bbox:
  tuple[float,float,float,float], crs: str, resolution_m: float) ->
  WorkGrid`, `build_grid_from_settings(settings: Settings) -> WorkGrid`,
  `grid_template(grid: WorkGrid, fill_value: float = 0.0, dtype: str =
  "float32") -> xr.DataArray`. Task 3 consumes `WorkGrid.transform`,
  `.width`, `.height`, `.crs`, `.resolution_m` directly.

- [ ] **Step 1: Write the failing tests**

Create `features/tests/test_grid.py`:

```python
"""Tests de la grilla de trabajo canónica: determinismo y snapping."""
import math

import numpy as np
import xarray as xr
from features.grid.grid import WorkGrid, build_grid, build_grid_from_settings, grid_template

BBOX = (-73.7, -39.3, -71.0, -36.5)  # west, south, east, north (WGS84)
CRS = "EPSG:32719"
RESOLUTION_M = 250.0


def test_build_grid_is_deterministic_same_inputs_same_grid():
    grid_a = build_grid(BBOX, CRS, RESOLUTION_M)
    grid_b = build_grid(BBOX, CRS, RESOLUTION_M)
    assert grid_a == grid_b


def test_build_grid_produces_whole_pixel_extent():
    grid = build_grid(BBOX, CRS, RESOLUTION_M)
    west, south, east, north = grid.bounds
    assert math.isclose((east - west) / RESOLUTION_M, grid.width, rel_tol=0, abs_tol=1e-9)
    assert math.isclose((north - south) / RESOLUTION_M, grid.height, rel_tol=0, abs_tol=1e-9)
    assert (east - west) % RESOLUTION_M == 0.0
    assert (north - south) % RESOLUTION_M == 0.0


def test_build_grid_exact_multiple_bbox_adds_no_spurious_padding():
    # Un bbox cuyo extent reproyectado ya cae en un múltiplo exacto de la
    # resolución no debe ganar una fila/columna extra por floor/ceil.
    resolution_m = 1000.0
    grid_a = build_grid((-72.0, -38.0, -71.0, -37.0), CRS, resolution_m)
    # mismo bbox pedido dos veces -> mismas dimensiones (determinismo),
    # y el ancho/alto en metros debe ser un múltiplo entero exacto de la
    # resolución sin celdas de sobra más allá de lo que exige el snapping
    # hacia afuera del propio extent reproyectado (no exactamente
    # cuadrado por la proyección, pero SIEMPRE múltiplo entero).
    grid_b = build_grid((-72.0, -38.0, -71.0, -37.0), CRS, resolution_m)
    assert grid_a == grid_b
    west, south, east, north = grid_a.bounds
    assert (east - west) % resolution_m == 0.0
    assert (north - south) % resolution_m == 0.0


def test_build_grid_from_settings_matches_build_grid(monkeypatch):
    required_env = {
        "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
        "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
        "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
        "POSTGRES_PASSWORD": "x",
    }
    for key, value in required_env.items():
        monkeypatch.setenv(key, value)

    from shared.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    expected = build_grid(settings.study_area_bbox, settings.crs, float(settings.spatial_resolution_m))
    actual = build_grid_from_settings(settings)
    assert actual == expected
    get_settings.cache_clear()


def test_grid_template_has_expected_shape_crs_and_fill_value():
    grid = build_grid(BBOX, CRS, RESOLUTION_M)
    template = grid_template(grid, fill_value=7.0, dtype="float32")
    assert isinstance(template, xr.DataArray)
    assert template.shape == (grid.height, grid.width)
    assert template.rio.crs.to_string() == CRS
    assert np.all(template.values == 7.0)
    # las coords x/y deben caer dentro de los bounds de la grilla, no en
    # los bordes exactos (son centros de píxel, no esquinas).
    west, south, east, north = grid.bounds
    assert west < float(template.x.min()) < east
    assert south < float(template.y.min()) < north
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package features pytest features/tests/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.grid.grid'`

- [ ] **Step 3: Add `pyproj` to `features/pyproject.toml`**

In `features/pyproject.toml`, add `"pyproj>=3.6",` to the `dependencies`
list (next to `"shapely>=2.0",`). Run `uv sync` (or just proceed — `uv run`
resolves it automatically the next time it's invoked) so the lockfile
picks it up.

- [ ] **Step 4: Write minimal implementation**

Create `features/src/features/grid/grid.py`:

```python
"""Grilla de trabajo canónica de PyroCast: un único objeto (`WorkGrid`),
derivable de forma determinista desde `shared/config.py`.

Determinismo: misma bbox (WGS84) + CRS + resolución -> siempre la misma
`WorkGrid` (mismos bounds, mismo ancho/alto). El bbox se reproyecta al CRS
de destino y luego se "snapea" hacia afuera al múltiplo de la resolución
más cercano (floor para el borde oeste/sur, ceil para el este/norte) —
esto es lo que garantiza un número entero de celdas y un origen fijo
reproducible, en vez de depender de los bounds reproyectados exactos (que
en general no caen en un múltiplo exacto de la resolución).

Estado actual (ver docs/decisions.md): los módulos de ingesta existentes
(DEM, ERA5-Land, Sentinel-2, WorldCover) todavía reproyectan cada uno de
forma independiente a partir de los bounds de su propio mosaico, NO desde
esta grilla — migrarlos queda para `features/dataset/` (sin implementar).
Este módulo hace que la grilla canónica exista y sea correcta; no fuerza
todavía su uso en los pipelines existentes.
"""
import math
from dataclasses import dataclass

import numpy as np
import rasterio.transform
import rioxarray  # noqa: F401  (registra el accessor .rio en xarray)
import xarray as xr
from affine import Affine
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from shared.config import Settings


@dataclass(frozen=True)
class WorkGrid:
    crs: str
    transform: Affine
    width: int
    height: int
    resolution_m: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        west, south, east, north = rasterio.transform.array_bounds(
            self.height, self.width, self.transform
        )
        return (float(west), float(south), float(east), float(north))


def build_grid(
    bbox: tuple[float, float, float, float], crs: str, resolution_m: float
) -> WorkGrid:
    """bbox = (west, south, east, north) en WGS84 (EPSG:4326)."""
    west, south, east, north = transform_bounds("EPSG:4326", crs, *bbox)

    snapped_west = math.floor(west / resolution_m) * resolution_m
    snapped_south = math.floor(south / resolution_m) * resolution_m
    snapped_east = math.ceil(east / resolution_m) * resolution_m
    snapped_north = math.ceil(north / resolution_m) * resolution_m

    width = round((snapped_east - snapped_west) / resolution_m)
    height = round((snapped_north - snapped_south) / resolution_m)
    transform = from_origin(snapped_west, snapped_north, resolution_m, resolution_m)

    return WorkGrid(
        crs=crs, transform=transform, width=width, height=height, resolution_m=resolution_m
    )


def build_grid_from_settings(settings: Settings) -> WorkGrid:
    return build_grid(settings.study_area_bbox, settings.crs, float(settings.spatial_resolution_m))


def grid_template(
    grid: WorkGrid, fill_value: float = 0.0, dtype: str = "float32"
) -> xr.DataArray:
    """DataArray con las coords x/y y CRS/transform de `grid`, relleno con
    `fill_value` — template reutilizable (p. ej. `xr.full_like(grid_template(grid), otro_valor)`)."""
    xs = grid.transform.c + grid.transform.a * (np.arange(grid.width) + 0.5)
    ys = grid.transform.f + grid.transform.e * (np.arange(grid.height) + 0.5)
    data = np.full((grid.height, grid.width), fill_value, dtype=dtype)
    da = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=("y", "x"), name="grid_template")
    da = da.rio.write_crs(grid.crs)
    # anotación explícita: rioxarray no está completamente tipado, y sin
    # esto mypy --strict ve `Any` devuelto por `.rio.write_transform` y
    # se queja de "Returning Any from function declared to return DataArray".
    result: xr.DataArray = da.rio.write_transform(grid.transform)
    return result
```

Update `features/src/features/grid/__init__.py`:

```python
"""Módulo de features: grilla de trabajo canónica (`WorkGrid`).

Ver `features/grid/grid.py` para la definición completa y
`docs/decisions.md` para el estado de adopción por los módulos existentes.
"""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --package features pytest features/tests/test_grid.py -v`
Expected: `5 passed`

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean

- [ ] **Step 7: Commit**

```bash
git add features/src/features/grid features/tests/test_grid.py features/pyproject.toml uv.lock
git commit -m "feat: add canonical WorkGrid (features/grid)"
```

---

## Task 2: `features/fire_state/clustering.py` — spatiotemporal event clustering

**Files:**
- Create: `features/src/features/fire_state/clustering.py`
- Modify: `features/src/features/fire_state/__init__.py`
- Test: `features/tests/test_fire_state_clustering.py`

**Interfaces:**
- Consumes: `shared.schemas.FireDetection` (fields: `latitude: float`,
  `longitude: float`, `detected_at: datetime`, `frp: float | None`,
  `confidence: str`, `satellite: str`, `instrument: str`).
- Produces: `DEFAULT_SPATIAL_EPS_M: float`, `DEFAULT_TEMPORAL_EPS:
  timedelta`, `cluster_detections(detections: list[FireDetection],
  spatial_eps_m: float = DEFAULT_SPATIAL_EPS_M, temporal_eps: timedelta =
  DEFAULT_TEMPORAL_EPS) -> list[int]` (one label per input detection, same
  order, never `-1`), `FireEvent` (frozen dataclass: `event_id: int`,
  `detections: tuple[FireDetection, ...]`; properties `start_date -> date`,
  `end_date -> date`), `build_fire_events(detections: list[FireDetection],
  spatial_eps_m: float = DEFAULT_SPATIAL_EPS_M, temporal_eps: timedelta =
  DEFAULT_TEMPORAL_EPS) -> list[FireEvent]`. Task 3 consumes `FireEvent`
  directly (`.detections`, `.start_date`, `.end_date`).

- [ ] **Step 1: Write the failing tests**

Create `features/tests/test_fire_state_clustering.py`:

```python
"""Tests de clustering espaciotemporal de detecciones FIRMS en eventos."""
import datetime as dt

import pytest
from features.fire_state.clustering import (
    DEFAULT_SPATIAL_EPS_M,
    DEFAULT_TEMPORAL_EPS,
    FireEvent,
    build_fire_events,
    cluster_detections,
)
from pyproj import Transformer
from shared.schemas import FireDetection

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)


def _det(lat: float, lon: float, at: dt.datetime) -> FireDetection:
    return FireDetection(
        latitude=lat, longitude=lon, detected_at=at, frp=1.0,
        confidence="n", satellite="N", instrument="VIIRS",
    )


def _det_offset_m(base: FireDetection, dx: float, dy: float, at: dt.datetime) -> FireDetection:
    x, y = _TO_UTM.transform(base.longitude, base.latitude)
    lon, lat = _TO_WGS84.transform(x + dx, y + dy)
    return _det(lat, lon, at)


def test_two_fires_separated_in_space_and_time_are_distinct_events():
    base = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    fire_a = [_det(-37.0, -72.0, base), _det(-37.001, -72.001, base + dt.timedelta(hours=6))]
    fire_b = [_det(-39.0, -71.0, base), _det(-39.001, -71.001, base + dt.timedelta(hours=6))]
    labels = cluster_detections(fire_a + fire_b)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_contiguous_detections_chain_into_one_event_even_if_endpoints_alone_would_not_merge():
    # day1 y day3 quedan a ~1400 m entre sí (por encima de spatial_eps
    # =750 m), pero day2 está a ~700 m de ambos -- deben unirse los 3 vía
    # transitividad de union-find, no solo comparación directa por pares.
    base = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    day1 = _det(-37.0, -72.0, base)
    day2 = _det_offset_m(day1, dx=700.0, dy=0.0, at=base + dt.timedelta(days=1))
    day3 = _det_offset_m(day1, dx=1400.0, dy=0.0, at=base + dt.timedelta(days=2))
    labels = cluster_detections([day1, day2, day3], spatial_eps_m=750.0)
    assert labels[0] == labels[1] == labels[2]


def test_detections_beyond_temporal_eps_are_distinct_events_even_if_colocated():
    base = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.UTC)
    same_spot_early = _det(-37.0, -72.0, base)
    same_spot_late = _det(-37.0, -72.0, base + dt.timedelta(days=30))
    labels = cluster_detections(
        [same_spot_early, same_spot_late], temporal_eps=dt.timedelta(days=2)
    )
    assert labels[0] != labels[1]


def test_duplicate_detection_same_instant_same_coordinates_does_not_raise():
    at = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.UTC)
    det = _det(-37.0, -72.0, at)
    labels = cluster_detections([det, det])
    assert labels[0] == labels[1]


def test_clustering_handles_new_year_boundary_correctly():
    # dt.timedelta sobre datetimes timezone-aware maneja el rollover de
    # año de forma nativa -- este test lo pin-ea explícitamente en vez de
    # confiar en que "debería funcionar".
    dec_31 = dt.datetime(2025, 12, 31, 23, 0, tzinfo=dt.UTC)
    jan_1 = dt.datetime(2026, 1, 1, 1, 0, tzinfo=dt.UTC)  # 2 horas después
    labels = cluster_detections(
        [_det(-37.0, -72.0, dec_31), _det(-37.001, -72.001, jan_1)],
        temporal_eps=dt.timedelta(days=2),
    )
    assert labels[0] == labels[1]


def test_defaults_are_the_documented_values():
    assert DEFAULT_SPATIAL_EPS_M == 750.0
    assert DEFAULT_TEMPORAL_EPS == dt.timedelta(days=2)


def test_build_fire_events_groups_detections_and_exposes_date_range():
    base = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    fire_a = [_det(-37.0, -72.0, base), _det(-37.001, -72.001, base + dt.timedelta(days=1))]
    fire_b = [_det(-39.0, -71.0, base)]
    events = build_fire_events(fire_a + fire_b)
    assert len(events) == 2
    assert all(isinstance(e, FireEvent) for e in events)
    sizes = sorted(len(e.detections) for e in events)
    assert sizes == [1, 2]
    two_detection_event = next(e for e in events if len(e.detections) == 2)
    assert two_detection_event.start_date == base.date()
    assert two_detection_event.end_date == (base + dt.timedelta(days=1)).date()


def test_single_detection_event_start_and_end_date_are_the_same_day():
    at = dt.datetime(2026, 3, 1, 8, 0, tzinfo=dt.UTC)
    events = build_fire_events([_det(-37.0, -72.0, at)])
    assert len(events) == 1
    assert events[0].start_date == events[0].end_date == at.date()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package features pytest features/tests/test_fire_state_clustering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.fire_state.clustering'`

- [ ] **Step 3: Write minimal implementation**

Create `features/src/features/fire_state/clustering.py`:

```python
"""Definición de "evento de incendio" como clúster espaciotemporal de
detecciones FIRMS — ver docs/fire-events.md para la justificación completa
de parámetros y la comparación con kriging (WildfireCube).

Algoritmo: unión de conjuntos (union-find) sobre una relación de vecindad
— "las detecciones A y B pertenecen al mismo evento si están a lo sumo
`spatial_eps_m` de distancia (haversine) Y a lo sumo `temporal_eps` de
diferencia temporal" — equivalente a DBSCAN con `min_samples=1` y una
métrica precomputada que combina distancia espacial y temporal, pero
implementado directamente en vez de agregar `scikit-learn` como
dependencia nueva de `features` (ver docs/decisions.md): para el volumen
de detecciones de este proyecto (unos pocos miles por temporada en el
área de estudio) un O(n²) de unión de conjuntos es más que suficiente.

Es intencionalmente transitivo/por cadena ("chaining"): si A-B están
cerca y B-C están cerca, A y C quedan en el mismo evento aunque A-C por
sí solas no lo estén — esto modela un incendio que se mueve/crece de
forma continua en el tiempo, no una bola fija alrededor de un punto.
"""
import datetime as dt
import math
from dataclasses import dataclass

from shared.schemas import FireDetection

_EARTH_RADIUS_M = 6_371_000.0

DEFAULT_SPATIAL_EPS_M = 750.0
# ~2x el tamaño de píxel nominal de VIIRS (375 m, CLAUDE.md) -- une
# detecciones contiguas del mismo incendio sin fusionar focos separados
# por más de un par de píxeles VIIRS.
DEFAULT_TEMPORAL_EPS = dt.timedelta(days=2)
# VIIRS revisita el área ~1 vez/día; 2 días tolera un día de nubosidad
# perdido sin fusionar incendios de episodios distintos.


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    # min(1.0, ...) evita un ValueError de asin por overshoot de punto
    # flotante cuando a es minúsculamente > 1.0 (dos detecciones idénticas).
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b


def cluster_detections(
    detections: list[FireDetection],
    spatial_eps_m: float = DEFAULT_SPATIAL_EPS_M,
    temporal_eps: dt.timedelta = DEFAULT_TEMPORAL_EPS,
) -> list[int]:
    """labels[i] = id de evento para detections[i] — nunca -1: toda
    detección pertenece a algún evento, aunque sea de un solo elemento."""
    n = len(detections)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            time_diff = abs(detections[i].detected_at - detections[j].detected_at)
            if time_diff > temporal_eps:
                continue
            distance_m = _haversine_m(
                detections[i].latitude, detections[i].longitude,
                detections[j].latitude, detections[j].longitude,
            )
            if distance_m <= spatial_eps_m:
                uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    # Re-etiquetar raíces (arbitrarias) a ids consecutivos 0..k-1 en orden
    # de primera aparición -- resultado determinista, no depende del valor
    # interno de cada raíz de union-find.
    label_by_root: dict[int, int] = {}
    labels: list[int] = []
    for root in roots:
        if root not in label_by_root:
            label_by_root[root] = len(label_by_root)
        labels.append(label_by_root[root])
    return labels


@dataclass(frozen=True)
class FireEvent:
    event_id: int
    detections: tuple[FireDetection, ...]

    @property
    def start_date(self) -> dt.date:
        # anotación explícita de la lista intermedia: sin ella, mypy
        # --strict infiere `Any` para el resultado de `min()` sobre el
        # generador y se queja de "Returning Any from function declared
        # to return date".
        dates: list[dt.date] = [d.detected_at.date() for d in self.detections]
        return min(dates)

    @property
    def end_date(self) -> dt.date:
        dates: list[dt.date] = [d.detected_at.date() for d in self.detections]
        return max(dates)


def build_fire_events(
    detections: list[FireDetection],
    spatial_eps_m: float = DEFAULT_SPATIAL_EPS_M,
    temporal_eps: dt.timedelta = DEFAULT_TEMPORAL_EPS,
) -> list[FireEvent]:
    labels = cluster_detections(detections, spatial_eps_m, temporal_eps)
    grouped: dict[int, list[FireDetection]] = {}
    for label, detection in zip(labels, detections, strict=True):
        grouped.setdefault(label, []).append(detection)
    return [
        FireEvent(event_id=label, detections=tuple(dets))
        for label, dets in sorted(grouped.items())
    ]
```

Update `features/src/features/fire_state/__init__.py`:

```python
"""Módulo de features: reconstrucción de eventos de incendio a partir de
detecciones FIRMS (clustering espaciotemporal + rasterización).

Ver `docs/fire-events.md` para la definición completa de "evento" y sus
parámetros, y `docs/limitations.md` para la comparación con el kriging de
la literatura (WildfireCube).
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package features pytest features/tests/test_fire_state_clustering.py -v`
Expected: `9 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add features/src/features/fire_state/clustering.py features/src/features/fire_state/__init__.py features/tests/test_fire_state_clustering.py
git commit -m "feat: add spatiotemporal fire-event clustering (features/fire_state)"
```

---

## Task 3: `features/fire_state/rasterize.py` — daily binary fire masks

**Files:**
- Create: `features/src/features/fire_state/rasterize.py`
- Test: `features/tests/test_fire_state_rasterize.py`

**Interfaces:**
- Consumes: `features.grid.grid.WorkGrid` (`.transform`, `.width`,
  `.height`, `.crs`, `.resolution_m`), `features.fire_state.clustering.FireEvent`
  (`.detections`, `.start_date`, `.end_date`).
- Produces: `DEFAULT_BUFFER_M: float`, `rasterize_daily_masks(event:
  FireEvent, grid: WorkGrid, buffer_m: float = DEFAULT_BUFFER_M) ->
  dict[date, np.ndarray]` (one boolean `(height, width)` array per day WITH
  a real detection), `fill_temporal_gaps(masks_by_day: dict[date,
  np.ndarray], start_date: date, end_date: date) -> dict[date, np.ndarray]`
  (one boolean array for every day in `[start_date, end_date]` inclusive),
  `build_fire_state(event: FireEvent, grid: WorkGrid, buffer_m: float =
  DEFAULT_BUFFER_M) -> dict[date, np.ndarray]` (composes the two).

- [ ] **Step 1: Write the failing tests**

Create `features/tests/test_fire_state_rasterize.py`:

```python
"""Tests de rasterización de eventos de incendio: buffer espacial +
interpolación temporal lineal (unión de máscaras ancla) sobre días sin
detección dentro del rango del evento."""
import datetime as dt

import numpy as np
from features.fire_state.clustering import FireEvent
from features.fire_state.rasterize import (
    build_fire_state,
    fill_temporal_gaps,
    rasterize_daily_masks,
)
from features.grid.grid import WorkGrid
from rasterio.transform import from_origin
from shared.schemas import FireDetection

# Grilla de 20x20 @ 250 m en EPSG:32719, con origen elegido a mano para
# que el punto (lat=-38.0, lon=-72.5) caiga en el píxel (10, 10) --
# verificado independientemente con pyproj antes de escribir el test
# (ver docs/superpowers/plans/2026-09-27-features-grid-fire-state.md).
_GRID = WorkGrid(
    crs="EPSG:32719", transform=from_origin(190000, 5791000, 250, 250),
    width=20, height=20, resolution_m=250.0,
)


def _det(lat: float, lon: float, at: dt.datetime) -> FireDetection:
    return FireDetection(
        latitude=lat, longitude=lon, detected_at=at, frp=1.0,
        confidence="n", satellite="N", instrument="VIIRS",
    )


def test_rasterize_daily_masks_known_coordinate_lands_on_expected_pixel():
    at = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    event = FireEvent(event_id=0, detections=(_det(-38.0, -72.5, at),))
    masks = rasterize_daily_masks(event, _GRID, buffer_m=300.0)
    assert list(masks.keys()) == [at.date()]
    mask = masks[at.date()]
    assert mask.shape == (20, 20)
    assert mask.dtype == np.bool_
    assert mask[10, 10]  # centro esperado -- verificado independientemente
    assert not mask[0, 0]  # esquina, a >3000 m del punto -- fuera del buffer


def test_fill_temporal_gaps_interpolates_as_union_of_neighboring_anchors():
    shape = (20, 20)
    day1 = dt.date(2026, 1, 1)
    day2 = dt.date(2026, 1, 2)  # día sin detección propia -- se rellena
    day3 = dt.date(2026, 1, 3)

    mask_day1 = np.zeros(shape, dtype=bool)
    mask_day1[5, 5] = True
    mask_day1[10, 10] = True  # presente en AMBAS anclas
    mask_day3 = np.zeros(shape, dtype=bool)
    mask_day3[8, 8] = True
    mask_day3[10, 10] = True  # presente en AMBAS anclas

    filled = fill_temporal_gaps({day1: mask_day1, day3: mask_day3}, day1, day3)

    assert set(filled.keys()) == {day1, day2, day3}
    assert filled[day1][5, 5] and not filled[day1][8, 8]
    assert filled[day3][8, 8] and not filled[day3][5, 5]
    # día intermedio: unión de ambas anclas (interpolación lineal del
    # indicador 0/1 nunca cruza 0 salvo que ambas anclas sean 0) -- un
    # píxel presente en AMBAS anclas debe resolver a True, no quedar en
    # un valor fraccionario sin umbralizar.
    assert filled[day2][10, 10]
    assert filled[day2][5, 5]
    assert filled[day2][8, 8]
    assert not filled[day2][0, 0]


def test_fill_temporal_gaps_never_extrapolates_past_a_single_sided_anchor():
    shape = (5, 5)
    day1 = dt.date(2026, 1, 5)  # única ancla, a mitad del rango pedido
    only_mask = np.zeros(shape, dtype=bool)
    only_mask[2, 2] = True

    filled = fill_temporal_gaps(
        {day1: only_mask}, dt.date(2026, 1, 3), dt.date(2026, 1, 7)
    )

    assert filled[day1][2, 2]
    for day in (dt.date(2026, 1, 3), dt.date(2026, 1, 4)):
        assert not np.any(filled[day])  # antes de la única ancla -- vacío
    for day in (dt.date(2026, 1, 6), dt.date(2026, 1, 7)):
        assert not np.any(filled[day])  # después de la única ancla -- vacío


def test_build_fire_state_combines_rasterize_and_fill_across_a_gap_day():
    day1 = dt.datetime(2026, 2, 1, 12, 0, tzinfo=dt.UTC)
    day3 = dt.datetime(2026, 2, 3, 12, 0, tzinfo=dt.UTC)  # deja el 2 sin detección
    event = FireEvent(
        event_id=0,
        detections=(_det(-38.0, -72.5, day1), _det(-38.0, -72.5, day3)),
    )
    masks = build_fire_state(event, _GRID, buffer_m=300.0)
    assert set(masks.keys()) == {day1.date(), dt.date(2026, 2, 2), day3.date()}
    assert masks[dt.date(2026, 2, 2)][10, 10]  # relleno del día intermedio
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package features pytest features/tests/test_fire_state_rasterize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.fire_state.rasterize'`

- [ ] **Step 3: Write minimal implementation**

Create `features/src/features/fire_state/rasterize.py`:

```python
"""Rasterización de un `FireEvent` a una máscara binaria diaria sobre la
grilla de trabajo — ver docs/fire-events.md.

*** SIMPLIFICACIÓN EXPLÍCITA frente a la literatura: WildfireCube
reconstruye la superficie quemada mediante kriging espaciotemporal sobre
las detecciones. Este proyecto usa en cambio (a) un buffer espacial FIJO
alrededor de cada detección puntual (nunca una interpolación
geoestadística) y (b) interpolación temporal LINEAL del indicador binario
entre días con detección para rellenar días sin observación dentro del
rango del evento — nunca extrapola fuera de ese rango. Ver
docs/limitations.md. ***
"""
import datetime as dt

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import Point

from features.fire_state.clustering import FireEvent
from features.grid.grid import WorkGrid

DEFAULT_BUFFER_M = 375.0  # tamaño de píxel nominal VIIRS (CLAUDE.md)


def rasterize_daily_masks(
    event: FireEvent, grid: WorkGrid, buffer_m: float = DEFAULT_BUFFER_M
) -> dict[dt.date, np.ndarray]:
    """Una máscara booleana (grid.height, grid.width) por cada día CON
    detección en el evento — un círculo de radio `buffer_m` (metros, en
    `grid.crs`) alrededor de cada detección de ese día."""
    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    geoms_by_day: dict[dt.date, list[Point]] = {}
    for detection in event.detections:
        day = detection.detected_at.date()
        x, y = transformer.transform(detection.longitude, detection.latitude)
        geoms_by_day.setdefault(day, []).append(Point(x, y).buffer(buffer_m))

    masks: dict[dt.date, np.ndarray] = {}
    for day, geoms in geoms_by_day.items():
        raw = rasterize(
            [(geom, 1) for geom in geoms],
            out_shape=(grid.height, grid.width),
            transform=grid.transform,
            fill=0,
            dtype="uint8",
        )
        masks[day] = raw.astype(bool)
    return masks


def fill_temporal_gaps(
    masks_by_day: dict[dt.date, np.ndarray], start_date: dt.date, end_date: dt.date
) -> dict[dt.date, np.ndarray]:
    """Para cada día en `[start_date, end_date]` sin máscara propia,
    interpola LINEALMENTE el indicador binario entre el día ancla anterior
    y el siguiente con detección, marcando como "fuego" todo lo que el
    valor interpolado sea > 0 — equivalente a la unión de las máscaras
    ancla anterior/siguiente para ese píxel (un 0/1 interpolado
    linealmente entre dos anclas solo da 0 si AMBAS anclas son 0). Un día
    sin ancla en algún lado (antes de la primera detección del evento, o
    después de la última) se deja vacío — nunca se extrapola."""
    anchor_days = sorted(masks_by_day)
    if not anchor_days:
        return {}
    shape = next(iter(masks_by_day.values())).shape

    filled: dict[dt.date, np.ndarray] = {}
    total_days = (end_date - start_date).days + 1
    for offset in range(total_days):
        day = start_date + dt.timedelta(days=offset)
        if day in masks_by_day:
            filled[day] = masks_by_day[day]
            continue

        earlier = [a for a in anchor_days if a < day]
        later = [a for a in anchor_days if a > day]
        if not earlier or not later:
            filled[day] = np.zeros(shape, dtype=bool)
            continue

        prev_day, next_day = earlier[-1], later[0]
        prev_mask, next_mask = masks_by_day[prev_day], masks_by_day[next_day]
        span_days = (next_day - prev_day).days
        weight = (day - prev_day).days / span_days
        interpolated = (1 - weight) * prev_mask.astype(np.float64) + weight * next_mask.astype(
            np.float64
        )
        filled[day] = interpolated > 0.0
    return filled


def build_fire_state(
    event: FireEvent, grid: WorkGrid, buffer_m: float = DEFAULT_BUFFER_M
) -> dict[dt.date, np.ndarray]:
    daily_masks = rasterize_daily_masks(event, grid, buffer_m)
    return fill_temporal_gaps(daily_masks, event.start_date, event.end_date)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package features pytest features/tests/test_fire_state_rasterize.py -v`
Expected: `4 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict features/src && uv run ruff check features/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add features/src/features/fire_state/rasterize.py features/tests/test_fire_state_rasterize.py
git commit -m "feat: add fire-event rasterization with temporal gap-fill (features/fire_state)"
```

---

## Task 4: Documentation — `docs/fire-events.md`, `docs/limitations.md`, `docs/decisions.md`

**Files:**
- Create: `docs/fire-events.md`
- Modify: `docs/limitations.md`
- Modify: `docs/decisions.md`

**Interfaces:**
- Consumes: the final parameter values and behavior from Tasks 1–3
  (`DEFAULT_SPATIAL_EPS_M`, `DEFAULT_TEMPORAL_EPS`, `DEFAULT_BUFFER_M`, the
  snapping algorithm, the gap-fill semantics) — no code interface, this
  task only writes prose that must match the shipped code exactly.

- [ ] **Step 1: Write `docs/fire-events.md`**

```markdown
# Definición de "evento de incendio"

`features/fire_state/` reconstruye, a partir de detecciones activas de
FIRMS (`shared.schemas.FireDetection`), qué detecciones pertenecen al
mismo incendio y qué superficie ocupó ese incendio día a día.

## 1. Clustering espaciotemporal (`features/fire_state/clustering.py`)

Un **evento** es un clúster de detecciones conectadas por una relación de
vecindad simple: dos detecciones A y B son "vecinas" si están a lo sumo
`spatial_eps_m` de distancia (haversine, sobre la esfera terrestre) Y a lo
sumo `temporal_eps` de diferencia temporal. El clustering final es la
componente conexa de esta relación (unión de conjuntos/union-find) — **no**
comparación directa por pares: si A-B son vecinas y B-C son vecinas, A y C
quedan en el mismo evento aunque A-C por sí solas no lo sean. Esto modela
un incendio que se mueve o crece de forma continua en el tiempo (un
"gusano" de detecciones encadenadas), no una bola fija alrededor de un
punto de origen.

Es equivalente a ejecutar DBSCAN con `min_samples=1` sobre una métrica
precomputada que combina distancia espacial y temporal — implementado a
mano en vez de agregar `scikit-learn` como dependencia de `features` (ver
`docs/decisions.md`).

**Parámetros (configurables por llamada, con default documentado):**

| Parámetro | Default | Por qué |
|---|---|---|
| `spatial_eps_m` | 750 m | ~2x el tamaño de píxel nominal de VIIRS (375 m) — une detecciones contiguas del mismo incendio sin fusionar focos separados por más de un par de píxeles VIIRS. |
| `temporal_eps` | 2 días | VIIRS revisita el área ~1 vez/día; 2 días tolera un día de nubosidad perdido sin fusionar episodios de incendio distintos y no relacionados. |

Ninguno de los dos valores viene de una calibración contra incendios
reales de Chile — son heurísticas razonables basadas en la resolución
nominal del sensor, documentadas como tales. `models/evaluation/`
(backtesting, sin implementar) es el lugar donde debería, a futuro,
ajustarse contra eventos reales.

## 2. Rasterización (`features/fire_state/rasterize.py`)

Para cada evento, se produce una máscara binaria (`True`/`False` por
celda de la grilla de trabajo — `features/grid/`) por cada día entre la
primera y la última detección del evento (`event.start_date` a
`event.end_date`, inclusive):

1. **Días con detección real**: cada detección se reproyecta a `grid.crs`
   y se le aplica un buffer circular fijo de radio `buffer_m` (default:
   375 m, el tamaño de píxel nominal de VIIRS). La unión de todos los
   círculos de detecciones de ese día, rasterizada sobre la grilla, es la
   máscara de ese día.
2. **Días sin detección dentro del rango del evento** ("días de hueco",
   p. ej. por nubosidad): se interpola LINEALMENTE el indicador binario
   (0/1) de cada celda entre el día ancla anterior y el siguiente con
   detección real. Como ambos extremos son 0 o 1, el valor interpolado
   solo puede ser 0 si AMBAS anclas son 0 en esa celda — en la práctica,
   esto equivale a tomar la **unión** de las máscaras de las dos anclas
   más cercanas para ese día de hueco.
3. **Nunca se extrapola**: un día de hueco sin ancla en alguno de los dos
   lados (fuera del rango `[start_date, end_date]` del evento) queda
   fuera de la salida — no existe fuera de ese rango — y dentro del
   rango pero sin ancla en un lado (imposible dado que start/end_date
   son por definición días con detección) no ocurre por construcción.

## 3. Simplificación explícita frente a la literatura

WildfireCube (el paper de referencia de este proyecto, ver CLAUDE.md)
reconstruye la superficie quemada mediante **kriging espaciotemporal**
sobre las detecciones — un método geoestadístico que estima la
incertidumbre espacial de la interpolación y puede producir un frente de
fuego suavizado y físicamente más plausible que una unión de círculos.

PyroCast usa en cambio:
- Un **buffer espacial fijo** (no kriging) alrededor de cada detección
  puntual — ignora completamente la incertidumbre de geolocalización real
  del sensor (que varía con el ángulo de barrido) y no captura la forma
  real del frente de fuego entre detecciones cercanas.
- **Interpolación temporal lineal** del indicador binario (equivalente a
  unión de máscaras ancla) en vez de una interpolación espaciotemporal
  conjunta — un incendio que se apaga y luego se reactiva en un lugar
  distinto dentro de la misma ventana `temporal_eps` se rellena como si
  hubiera seguido ardiendo continuamente en ambos lugares durante el
  hueco, lo cual sobreestima la superficie quemada en ese caso.

Esta es una simplificación deliberada de una sola persona, documentada
también en `docs/limitations.md`, no un método validado contra
reconstrucciones reales de perímetros de incendio en Chile.
```

- [ ] **Step 2: Append to `docs/limitations.md`**

Add before the closing "## Herramienta de investigación" section:

```markdown
- **Reconstrucción de eventos de incendio: buffer + interpolación lineal,
  no kriging**: `features/fire_state/` reconstruye la superficie quemada
  diaria de un evento con un buffer espacial fijo alrededor de cada
  detección FIRMS más interpolación temporal lineal (equivalente a unión
  de máscaras) entre días con detección — WildfireCube (paper de
  referencia) usa kriging espaciotemporal, que estima incertidumbre
  espacial y produce una reconstrucción más plausible físicamente. Un
  incendio que se apaga y se reactiva en otro punto dentro de la misma
  ventana `temporal_eps` (2 días por defecto) se rellena como si hubiera
  seguido ardiendo en ambos lugares durante el hueco, sobreestimando la
  superficie quemada en ese caso. Ver `docs/fire-events.md`.
- **Parámetros de clustering de eventos sin calibrar contra incendios
  reales**: `spatial_eps_m=750m` y `temporal_eps=2 días`
  (`features/fire_state/clustering.py`) son heurísticas basadas en la
  resolución nominal de VIIRS (375 m, revisita diaria), no un ajuste
  contra el historial real de incendios de Chile — ese ajuste
  corresponde a `models/evaluation/` (backtesting), sin implementar.
```

- [ ] **Step 3: Append to `docs/decisions.md`**

```markdown
## `features/grid/`: snapping determinista en vez de bounds reproyectados exactos

`build_grid` reproyecta el bbox de estudio (WGS84) al CRS de destino y
"snapea" el resultado hacia afuera al múltiplo de la resolución más
cercano (`floor` para el borde oeste/sur, `ceil` para el este/norte) en
vez de usar los bounds reproyectados exactos. Esto es lo que garantiza
que "misma configuración → mismos límites de grilla, mismo número de
celdas" sea trivialmente cierto (ancho/alto son siempre un entero exacto,
nunca dependen de redondeos acumulados de una transformación de GDAL) —
el costo es que la grilla resultante cubre un área ligeramente mayor que
el bbox pedido (hasta una celda de más por lado), nunca menor.

**Estado de adopción, explícito**: los cuatro módulos de ingesta ya
implementados (DEM, ERA5-Land, Sentinel-2, WorldCover) NO usan todavía
esta grilla — cada uno sigue reproyectando de forma independiente a
partir de los bounds de su propio mosaico (`calculate_default_transform`
sobre el raster ya descargado), como ya documentaba la sección "Cierre de
Etapa 1" más arriba. `features/grid/` hace que la grilla canónica exista
y sea correcta; migrar los pipelines existentes queda para
`features/dataset/` (todavía sin implementar), que es el consumidor que
en la práctica necesita alineación píxel-a-píxel entre todas las capas.

## `features/fire_state/`: union-find propio en vez de `scikit-learn` (DBSCAN)

El enunciado permite "DBSCAN u otro método simple". Se implementó un
union-find de ~30 líneas sobre una relación de vecindad
espacial+temporal, en vez de agregar `scikit-learn` como dependencia
nueva de `features` — `scikit-learn` ya es parte del stack del proyecto
(`models/evaluation`, calibración isotónica, según CLAUDE.md) pero no es
hoy una dependencia de `features`, y el volumen de detecciones de este
proyecto (algunos miles por temporada en el área de estudio) hace que un
O(n²) directo sea más que suficiente — no vale la pena la dependencia
pesada por un algoritmo tan chico. Si el volumen de detecciones creciera
significativamente (p. ej. multi-país, multi-temporada), esto debería
revisarse.

## `features/fire_state/`: sin CLI/Makefile todavía

A diferencia de FIRMS/DEM/ERA5-Land/Sentinel-2/WorldCover, este módulo no
tiene comando de CLI ni target de Makefile. `features/terrain` y
`features/weather` tampoco lo tienen — se invocan como un paso de cómputo
desde dentro de `ingestion/dem/cli.py`/`ingestion/era5/cli.py`,
inmediatamente después de la propia descarga de esa fuente.
`features/fire_state` no tiene un punto de enganche equivalente: el
clustering necesita el HISTORIAL acumulado de detecciones (potencialmente
de muchas corridas de `ingest-firms` a lo largo de varios días), no la
respuesta de una sola descarga. Conectarlo a un pipeline real es tarea de
`features/dataset/` (ya marcado como "pendiente" en el Makefile) — no es
un descuido, es una decisión explícita de no inventar ese punto de
enganche antes de que exista el consumidor real.

## `features/fire_state/`: sin persistencia a disco todavía

Este módulo devuelve `dict[date, np.ndarray]` en memoria — no escribe
Zarr/GeoTIFF. Los criterios de aceptación del enunciado piden clustering
+ rasterización + tests + docs, no un esquema de persistencia; inventar
un formato/layout de archivos ahora, sin que `features/dataset/` (el
consumidor real que en la práctica dictará esos requisitos: ¿un Zarr por
evento? ¿un cubo único por toda el área de estudio?) exista todavía,
sería adivinar. Documentado aquí para que quede como decisión explícita,
no como una omisión.

## `pyproj` declarado explícitamente en `features/pyproject.toml`

`features/fire_state/rasterize.py` importa `pyproj` directamente (para
reproyectar detecciones WGS84 al CRS de la grilla). Ya estaba instalado
de forma transitiva vía `rasterio`, pero depender de eso sin declararlo
es frágil — se agrega como dependencia directa del paquete `features`.
```

- [ ] **Step 4: Run the full verification sweep**

Run:
```bash
env -i PATH="$PATH" HOME="$HOME" uv run --package features pytest features/tests -v
uv run ruff check .
uv run mypy --strict shared/src features/src
```
Expected: all green/clean (features test count: 23 existing + 5 (grid) + 9
(clustering) + 4 (rasterize) = 41 passed).

- [ ] **Step 5: Commit**

```bash
git add docs/fire-events.md docs/limitations.md docs/decisions.md
git commit -m "docs: document features/grid and features/fire_state (fire-events.md, limitations, decisions)"
```
