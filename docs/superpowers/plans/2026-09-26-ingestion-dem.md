# Ingestion DEM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ingestion/dem/` end to end: Copernicus DEM GLO-30 tile download (AWS public bucket, no credentials), mosaicking + reprojection to EPSG:32719 with bilinear resampling, bbox+resolution-hash caching, and `features/terrain/` slope/aspect computation (Horn's method) written to `data/processed/terrain/` — all fixture/mock-tested, zero real network calls.

**Architecture:** `ingestion/dem/tiles.py` computes which 1°×1° Copernicus DEM tiles cover a bbox (pure math, no I/O). `ingestion/dem/client.py` downloads one tile over plain HTTPS from the public `copernicus-dem-30m` S3 bucket (no AWS SDK, no credentials — it's a public bucket served over HTTPS like any static file host). `ingestion/dem/cache.py` computes a cache key from `(bbox, resolution_m, crs)`. `ingestion/dem/pipeline.py` orchestrates: check the bbox-hash cache → download any missing tiles (each tile is itself cached by its own deterministic name, since tiles are global/reusable across bboxes) → `rasterio.merge.merge` the tiles in their native CRS (EPSG:4326) → `rasterio.warp.reproject` the mosaic to EPSG:32719 at the configured resolution with `Resampling.bilinear` → write the result to the bbox-hash-named cache file. `features/terrain/slope_aspect.py` computes slope (degrees) and aspect (compass bearing, 0–360, -1 for flat) from the reprojected DEM using Horn's method (the same 3×3-kernel algorithm used by GDAL's `gdaldem` and ESRI), and writes both as single-band GeoTIFFs.

**Tech Stack:** Python 3.12, `requests` (plain HTTPS GET, no AWS SDK needed — the bucket is public), `rasterio` + `numpy` (already CLAUDE.md-approved, newly declared on `ingestion` alongside their existing `features` declaration), `responses` (HTTP mocking, already an `ingestion` dev dependency), `pytest`. No new dependencies beyond what CLAUDE.md already lists.

**Spec:** User's message in this conversation (5 numbered tasks + acceptance criteria) plus `/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md` (Copernicus DEM row: "Elevación, pendiente, orientación | OpenTopography API o bucket AWS de Copernicus"; EPSG:32719 for planar calcs is already CLAUDE.md's stated CRS). Both travel with this plan.

**Copernicus DEM GLO-30 access — verified against live sources (2026-09-26), not assumed from memory:**
- Two documented options exist: OpenTopography's Global DEM API (`https://portal.opentopography.org/apidocs/#/Public/getGlobalDem`, dataset code `COP30`) requires a free API key, and its exact rate/area limits are not published anywhere findable; or the AWS Open Data Registry bucket (https://registry.opendata.aws/copernicus-dem/), which is **fully public over plain HTTPS with no credentials at all**.
- Verified directly against the live bucket in this session:
  - `curl -sI https://copernicus-dem-30m.s3.amazonaws.com/tileList.txt` → `HTTP/1.1 200 OK`, no auth headers needed.
  - Tile naming, confirmed against the bucket's own `tileList.txt` for our study area's latitude band: `Copernicus_DSM_COG_10_S37_00_W072_00_DEM`, `Copernicus_DSM_COG_10_S38_00_W072_00_DEM`, `Copernicus_DSM_COG_10_S39_00_W073_00_DEM` (the `10` is the tile-format code for 1.0 arc-second/GLO-30 spacing, not meters or arc-seconds literally — GLO-90 tiles use `30` for 3.0 arc-seconds).
  - Full download URL, confirmed with `curl -sI` returning `200 OK` / `Content-Type: image/tiff`:
    `https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_{10}_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM/Copernicus_DSM_COG_{10}_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM.tif`
  - Each tile covers exactly a 1°×1° cell named by its **south-west corner** (e.g. `S37_00_W072_00` covers `[-37, -36) × [-72, -71)`), native CRS EPSG:4326, native resolution ~30 m (1 arc-second).
- **This plan chooses the AWS bucket over OpenTopography**: zero credentials (no new `.env` variable, no `shared.config` field, no API-key rate limit to manage), no new dependency (a plain `requests.get`, not `boto3`), and the tile grid is simple, deterministic math. Documented as a decision in Task 5 — see Global Constraints for what NOT to add because of this choice.

## Global Constraints

- No AWS credentials, no `boto3`, no new `.env` variable for DEM — the chosen access method is a public HTTPS bucket.
- `rasterio`/`numpy` are already CLAUDE.md-approved dependencies (currently declared only on `features`); this plan adds them to `ingestion`'s own `pyproject.toml` too, since `ingestion/dem/` imports them directly — this is a normal per-package dependency declaration, not a new external dependency, so it does **not** need a `docs/decisions.md` entry (unlike a genuinely new package would).
- Zero real network calls in tests — every HTTP GET to the DEM bucket is mocked via `responses`, exactly like the FIRMS task.
- Slope/aspect must be computed on the **reprojected, planar** (EPSG:32719, meters) raster, never on the raw EPSG:4326 (degrees) DEM — computing terrain derivatives on a lat/lon grid without correcting for cell-size-in-meters varying by latitude would be a real correctness bug (out of scope to even build that path).
- `mypy --strict` and `ruff` must be clean on `ingestion/src/ingestion/dem` (one-off command, same pattern as the FIRMS task — `ingestion` isn't in the Makefile's `typecheck` scope) and `ruff`/`mypy --strict` must stay clean on `features/src/features/terrain` (this one **is** already inside the Makefile's `typecheck` scope, so `make typecheck` itself must keep passing).
- Identifiers in English; docstrings/comments/docs in Spanish (CLAUDE.md convention).
- Nodata handling in the slope/aspect kernel is intentionally minimal for this bootstrap (edge-padding only, no nodata-aware masking) — this is a real, documented limitation (Task 6's `docs/data-sources.md` entry), not silently swept under the rug.

## Review Focus

- **Tile-grid math at bbox edges**: a bbox whose north/east edge lands exactly on an integer degree (e.g. `north=-36.0`) must not silently drop the tile that covers that boundary — off-by-one in the `floor()`-based range is the most likely bug here, and it's invisible unless a test puts a boundary exactly on an integer.
- **Southern/western hemisphere sign handling in tile naming**: our whole study area is south (`S`) and west (`W`) of the prime meridian/equator — a naive `f"{lat:02d}"` on a negative float would emit something like `S-37_00` instead of `S37_00`; this must be tested with our actual bbox's negative coordinates, not a northern-hemisphere example.
- **Cache short-circuit actually skips the network**, not just "returns the same result" — a test that only checks the returned path is identical across two calls would pass even if the implementation silently re-downloads every time; the test must assert the download function was called zero additional times on the second call.
- **Reprojection uses bilinear, not the rasterio default (nearest)** — trivially easy to typo/forget the `resampling=` kwarg, and a test that only checks output shape/CRS (not resampling method) would not catch a silent fallback to nearest-neighbor, which would visibly show up as blocky, un-interpolated elevation steps rather than a smooth gradient — the test should assert on a value that only bilinear (not nearest) produces correctly for a known synthetic gradient.
- **Aspect's flat-terrain and wrap-around edge cases**: a perfectly flat synthetic DEM must report a defined "no aspect" sentinel (not a `NaN`-propagating garbage value from `atan2(0, 0)`, which is technically `0.0` in IEEE 754 and would be silently indistinguishable from "due north" if not special-cased), and the 0°/360° wrap-around (e.g. a slope facing just west of due north) must land in the correct half of the circle, not off by 180° from a sign error in the `atan2` argument order — a formula this easy to get backwards needs a test with a slope facing an unambiguous, non-cardinal direction, not just N/E/S/W.

---

## File Structure

```
ingestion/
├── pyproject.toml                      # + rasterio, numpy (already CLAUDE.md-approved elsewhere)
├── src/ingestion/dem/
│   ├── __init__.py                     # (exists, stub)
│   ├── tiles.py                        # new: tiles_for_bbox, tile_key, tile_url
│   ├── client.py                       # new: download_tile, DemDownloadError
│   ├── cache.py                        # new: cache_key_for
│   └── pipeline.py                     # new: build_dem
└── tests/
    ├── fixtures/
    │   └── (synthetic tiles built at test time via rasterio, not committed binaries)
    ├── test_dem_tiles.py                # new
    ├── test_dem_client.py               # new
    └── test_dem_pipeline.py             # new

features/
├── src/features/terrain/
│   ├── __init__.py                     # (exists, stub)
│   └── slope_aspect.py                 # new: compute_slope_aspect, compute_and_save_terrain
└── tests/
    └── test_terrain_slope_aspect.py     # new

docs/
├── data-sources.md                     # + DEM section
└── decisions.md                        # + AWS-vs-OpenTopography choice
```

---

### Task 1: `ingestion/dem/tiles.py` — bbox → tile grid (pure math, no I/O)

**Files:**
- Create: `ingestion/src/ingestion/dem/tiles.py`
- Test: `ingestion/tests/test_dem_tiles.py`

**Interfaces:**
- Produces: `ingestion.dem.tiles.tile_key(lat: int, lon: int) -> str` (`lat`/`lon` are the tile's SW-corner integer degrees, e.g. `tile_key(-37, -72) == "Copernicus_DSM_COG_10_S37_00_W072_00_DEM"`), `ingestion.dem.tiles.tile_url(key: str) -> str`, `ingestion.dem.tiles.tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]` (returns `(lat, lon)` SW-corner pairs, not keys — Task 2/3 build keys/URLs from these).

- [ ] **Step 1: Write the failing tests**

`ingestion/tests/test_dem_tiles.py`:

```python
"""Tests de la grilla de tiles Copernicus DEM: solo matemática, sin red."""
from ingestion.dem.tiles import tile_key, tile_url, tiles_for_bbox


def test_tile_key_south_west_hemisphere():
    assert tile_key(-37, -72) == "Copernicus_DSM_COG_10_S37_00_W072_00_DEM"


def test_tile_key_north_east_hemisphere():
    assert tile_key(5, 10) == "Copernicus_DSM_COG_10_N05_00_E010_00_DEM"


def test_tile_url_builds_full_tif_path():
    url = tile_url("Copernicus_DSM_COG_10_S37_00_W072_00_DEM")
    assert url == (
        "https://copernicus-dem-30m.s3.amazonaws.com/"
        "Copernicus_DSM_COG_10_S37_00_W072_00_DEM/"
        "Copernicus_DSM_COG_10_S37_00_W072_00_DEM.tif"
    )


def test_tiles_for_bbox_covers_study_area_example():
    # west, south, east, north — Biobío/Ñuble/Araucanía-shaped bbox
    bbox = (-73.7, -39.3, -71.0, -36.5)
    tiles = tiles_for_bbox(bbox)
    assert (-40, -74) in tiles  # SW-most tile
    assert (-37, -71) in tiles  # NE-most tile
    assert len(tiles) == (40 - 37 + 1) * (74 - 71 + 1)  # 4 lat bands x 4 lon bands = 16


def test_tiles_for_bbox_includes_tile_when_edge_lands_on_integer_degree():
    # north edge exactly on -36.0: must still include the S36 tile whose
    # southern edge IS that boundary, and must NOT silently stop at S37.
    bbox = (-72.0, -37.0, -71.0, -36.0)
    tiles = tiles_for_bbox(bbox)
    assert (-37, -72) in tiles
    assert (-36, -72) in tiles


def test_tiles_for_bbox_single_tile():
    bbox = (-72.6, -37.6, -72.1, -37.1)
    assert tiles_for_bbox(bbox) == [(-38, -73)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_dem_tiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.dem.tiles'`

- [ ] **Step 3: Write `ingestion/src/ingestion/dem/tiles.py`**

```python
"""Grilla de tiles de Copernicus DEM GLO-30 (bucket público de AWS).

Cada tile cubre exactamente 1°x1°, nombrado por su esquina suroeste.
Verificado contra el bucket real (2026-09-26): ver docs/data-sources.md
y el encabezado de docs/superpowers/plans/2026-09-26-ingestion-dem.md
para la evidencia (curl contra tileList.txt y una URL de tile real).
"""
import math

_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
_TILE_FORMAT_CODE = "10"  # 1.0 arco-segundo == GLO-30 (~30 m)


def tile_key(lat: int, lon: int) -> str:
    """lat/lon son la esquina SO del tile, en grados enteros."""
    ns = "S" if lat < 0 else "N"
    ew = "W" if lon < 0 else "E"
    return (
        f"Copernicus_DSM_COG_{_TILE_FORMAT_CODE}_{ns}{abs(lat):02d}_00_"
        f"{ew}{abs(lon):03d}_00_DEM"
    )


def tile_url(key: str) -> str:
    return f"{_BASE_URL}/{key}/{key}.tif"


def tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """bbox = (west, south, east, north). Devuelve pares (lat, lon) de la
    esquina SO de cada tile 1x1 que intersecta el bbox."""
    west, south, east, north = bbox
    lat_start = math.floor(south)
    lat_end = math.floor(north)
    lon_start = math.floor(west)
    lon_end = math.floor(east)
    return [
        (lat, lon)
        for lat in range(lat_start, lat_end + 1)
        for lon in range(lon_start, lon_end + 1)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_dem_tiles.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/src/ingestion/dem/tiles.py ingestion/tests/test_dem_tiles.py
git commit -m "feat(ingestion): add Copernicus DEM tile grid math"
```

---

### Task 2: `ingestion/dem/client.py` — download one tile (mocked HTTP)

**Files:**
- Modify: `ingestion/pyproject.toml` (+ `rasterio`, `numpy` — needed starting this task, since the test fixture builds a real tiny GeoTIFF with rasterio)
- Create: `ingestion/src/ingestion/dem/client.py`
- Test: `ingestion/tests/test_dem_client.py`

**Interfaces:**
- Consumes: `ingestion.dem.tiles.tile_url` (Task 1).
- Produces: `ingestion.dem.client.DemDownloadError(RuntimeError)`, `ingestion.dem.client.download_tile(key: str, dest_path: Path, session: requests.Session | None = None) -> Path` (single attempt — no retry/backoff was asked for this source, unlike FIRMS; raises `DemDownloadError` on any non-200; writes the response bytes to `dest_path` and returns it; if `dest_path` already exists, returns immediately without making a request — this is the tile-level cache Task 3's pipeline relies on).

- [ ] **Step 1: Add `rasterio`/`numpy` to `ingestion/pyproject.toml`**

```toml
dependencies = [
    "shared",
    "requests>=2.32",
    "cdsapi>=0.7",
    "typer>=0.12",
    "pyarrow>=17.0",
    "rasterio>=1.3",
    "numpy>=2.0",
]
```

- [ ] **Step 2: Write the failing tests**

`ingestion/tests/test_dem_client.py`:

```python
"""Tests del cliente de descarga DEM: cero red real (usa `responses`)."""
import responses
from ingestion.dem.client import DemDownloadError, download_tile
from ingestion.dem.tiles import tile_url
import pytest

KEY = "Copernicus_DSM_COG_10_S37_00_W072_00_DEM"


@responses.activate
def test_download_tile_writes_bytes_to_dest_path(tmp_path):
    responses.add(responses.GET, tile_url(KEY), body=b"fake-tif-bytes", status=200)
    dest = tmp_path / f"{KEY}.tif"
    result = download_tile(KEY, dest)
    assert result == dest
    assert dest.read_bytes() == b"fake-tif-bytes"


@responses.activate
def test_download_tile_raises_on_404():
    with pytest.raises(DemDownloadError, match="404"):
        download_tile(KEY, __import__("pathlib").Path("/tmp/unused.tif"))
    # no response registered at all -> ConnectionError from `responses`,
    # which is itself proof no real network call happened; register a
    # 404 explicitly instead so the assertion is about our error message:


@responses.activate
def test_download_tile_raises_dem_download_error_on_404(tmp_path):
    responses.add(responses.GET, tile_url(KEY), status=404)
    with pytest.raises(DemDownloadError, match="404"):
        download_tile(KEY, tmp_path / f"{KEY}.tif")


@responses.activate
def test_download_tile_skips_request_when_dest_already_exists(tmp_path):
    dest = tmp_path / f"{KEY}.tif"
    dest.write_bytes(b"already-here")
    # deliberately no responses.add(...) — a real request would raise
    # responses.ConnectionError, proving the cache hit skipped the network
    result = download_tile(KEY, dest)
    assert result == dest
    assert dest.read_bytes() == b"already-here"
```

Note: `test_download_tile_raises_on_404` above is redundant with `test_download_tile_raises_dem_download_error_on_404` and was left in mid-draft — **remove the first one** before running; it calls `download_tile` with no mock registered at all, which raises `responses.exceptions.ConnectionError`, not `DemDownloadError`, and would fail for the wrong reason. Keep only `test_download_tile_raises_dem_download_error_on_404`.

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv sync --all-packages --group dev --reinstall-package ingestion && uv run --package ingestion pytest ingestion/tests/test_dem_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.dem.client'`

- [ ] **Step 3: Write `ingestion/src/ingestion/dem/client.py`**

```python
"""Descarga de un tile de Copernicus DEM desde el bucket público de AWS.

Bucket público sobre HTTPS plano — sin credenciales AWS, sin boto3. A
diferencia de FIRMS, esta fuente no tiene un límite de tasa documentado
para descargas de archivos estáticos, así que no se implementa
retry/backoff aquí (YAGNI); un fallo se reporta de inmediato.
"""
from pathlib import Path

import requests
from ingestion.dem.tiles import tile_url


class DemDownloadError(RuntimeError):
    """Fallo al descargar un tile de Copernicus DEM."""


def download_tile(key: str, dest_path: Path, session: requests.Session | None = None) -> Path:
    if dest_path.exists():
        return dest_path

    sess = session or requests.Session()
    response = sess.get(tile_url(key), timeout=60)
    if response.status_code != 200:
        raise DemDownloadError(
            f"Error {response.status_code} descargando tile {key}: {tile_url(key)}"
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(response.content)
    return dest_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_dem_client.py -v`
Expected: 3 passed (after removing the redundant test per the note in Step 2)

- [ ] **Step 5: Commit**

```bash
git add ingestion/pyproject.toml ingestion/src/ingestion/dem/client.py ingestion/tests/test_dem_client.py uv.lock
git commit -m "feat(ingestion): add DEM tile downloader (public AWS bucket, mocked tests)"
```

---

### Task 3: `ingestion/dem/cache.py` + `pipeline.py` — mosaic, reproject (bilinear), bbox-hash cache

**Files:**
- Create: `ingestion/src/ingestion/dem/cache.py`
- Create: `ingestion/src/ingestion/dem/pipeline.py`
- Test: `ingestion/tests/test_dem_pipeline.py`

**Interfaces:**
- Consumes: `tiles_for_bbox`, `tile_key` (Task 1), `download_tile` (Task 2).
- Produces: `ingestion.dem.cache.cache_key_for(bbox: tuple[float, float, float, float], resolution_m: int, crs: str) -> str` (16-hex-char sha256 prefix), `ingestion.dem.pipeline.build_dem(bbox: tuple[float, float, float, float], resolution_m: int, crs: str, raw_tiles_dir: Path, cache_dir: Path, download_fn: Callable[[str, Path], Path] = download_tile) -> Path` — returns the path to the cached, reprojected, merged GeoTIFF. `download_fn` is injected so tests can substitute a fake downloader that copies pre-built local fixture tiles instead of hitting `responses`-mocked HTTP for every call (keeps the pipeline test fast and focused on the merge/reproject logic, not the HTTP layer already covered by Task 2's tests).

- [ ] **Step 1: Write `ingestion/src/ingestion/dem/cache.py`** (no test-first needed in isolation — it's exercised by Task 3's own pipeline tests below; a single pure hash function doesn't carry independent risk worth a dedicated red/green cycle, but IS covered by the pipeline's cache-hit test)

```python
"""Clave de cache para un DEM procesado: hash de (bbox, resolución, CRS)."""
import hashlib


def cache_key_for(bbox: tuple[float, float, float, float], resolution_m: int, crs: str) -> str:
    payload = f"{bbox}|{resolution_m}|{crs}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 2: Write the failing pipeline tests**

`ingestion/tests/test_dem_pipeline.py`:

```python
"""Tests del pipeline DEM: mosaico + reproyección bilinear + cache, sin red."""
from pathlib import Path

import numpy as np
import rasterio
from ingestion.dem.pipeline import build_dem
from rasterio.transform import from_origin

BBOX = (-72.6, -37.6, -72.1, -37.1)  # single tile: (-38, -73)


def _write_synthetic_tile(path: Path, sw_lat: int, sw_lon: int, value_at_sw: float) -> None:
    """Tile sintético 4x4 en EPSG:4326, 1 grado de lado (0.25 deg/pixel),
    con un gradiente lineal simple (rampa este-oeste) para poder verificar
    que la reproyección bilineal interpola en vez de escalonar (nearest)."""
    size = 4
    transform = from_origin(sw_lon, sw_lat + 1, 0.25, 0.25)
    data = np.tile(value_at_sw + np.arange(size, dtype="float32") * 10.0, (size, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-32767.0,
    ) as dst:
        dst.write(data, 1)


def test_build_dem_reprojects_to_target_crs_and_resolution(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"

    def fake_download(key: str, dest_path: Path) -> Path:
        if not dest_path.exists():
            _write_synthetic_tile(dest_path, sw_lat=-38, sw_lon=-73, value_at_sw=100.0)
        return dest_path

    result_path = build_dem(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    assert result_path.exists()
    with rasterio.open(result_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
        assert ds.res == (250.0, 250.0)
        arr = ds.read(1)
    # Bilinear must produce interpolated (non-integer-multiple, varied)
    # values, not the small set of exact source values nearest-neighbor
    # would just copy verbatim.
    unique_vals = {round(float(v), 3) for v in arr.flatten() if v > 0}
    assert len(unique_vals) > 4  # more distinct values than the 4 source columns


def test_build_dem_cache_hit_skips_download_entirely(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    calls = {"n": 0}

    def fake_download(key: str, dest_path: Path) -> Path:
        calls["n"] += 1
        if not dest_path.exists():
            _write_synthetic_tile(dest_path, sw_lat=-38, sw_lon=-73, value_at_sw=100.0)
        return dest_path

    first = build_dem(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    calls_after_first = calls["n"]
    assert calls_after_first > 0

    second = build_dem(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    assert second == first
    assert calls["n"] == calls_after_first  # zero additional download calls
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_dem_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.dem.pipeline'`

- [ ] **Step 3: Write `ingestion/src/ingestion/dem/pipeline.py`**

```python
"""Orquesta: cache -> descarga de tiles faltantes -> mosaico -> reproyección
bilineal a la grilla de trabajo del proyecto (ver shared/config.py)."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
from ingestion.dem.cache import cache_key_for
from ingestion.dem.client import download_tile
from ingestion.dem.tiles import tile_key, tiles_for_bbox
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject


def build_dem(
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    crs: str,
    raw_tiles_dir: Path,
    cache_dir: Path,
    download_fn: Callable[[str, Path], Path] = download_tile,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"dem_{cache_key_for(bbox, resolution_m, crs)}.tif"
    if cache_path.exists():
        return cache_path

    raw_tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    for lat, lon in tiles_for_bbox(bbox):
        key = tile_key(lat, lon)
        dest = raw_tiles_dir / f"{key}.tif"
        tile_paths.append(download_fn(key, dest))

    sources = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_array, mosaic_transform = merge(sources)
        src_crs = sources[0].crs
        src_nodata = sources[0].nodata
    finally:
        for src in sources:
            src.close()

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        crs,
        mosaic_array.shape[-1],
        mosaic_array.shape[-2],
        *rasterio.transform.array_bounds(
            mosaic_array.shape[-2], mosaic_array.shape[-1], mosaic_transform
        ),
        resolution=(resolution_m, resolution_m),
    )

    dst_array = np.empty((mosaic_array.shape[0], dst_height, dst_width), dtype=mosaic_array.dtype)
    reproject(
        source=mosaic_array,
        destination=dst_array,
        src_transform=mosaic_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=crs,
        dst_nodata=src_nodata,
        resampling=Resampling.bilinear,
    )

    with rasterio.open(
        cache_path, "w", driver="GTiff",
        height=dst_height, width=dst_width, count=dst_array.shape[0],
        dtype=dst_array.dtype, crs=crs, transform=dst_transform, nodata=src_nodata,
    ) as dst:
        dst.write(dst_array)

    return cache_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_dem_pipeline.py -v`
Expected: 2 passed

If `ds.res` doesn't come back as exactly `(250.0, 250.0)` (rasterio's `calculate_default_transform` can return a resolution slightly different from the requested one due to its bounds-alignment algorithm), relax that assertion to `pytest.approx` with a small tolerance and ledger the ruling — do not change the `resolution_m` argument's meaning to fix a test that was checking too strictly.

- [ ] **Step 5: Run the full `ingestion` suite together**

Run: `uv run --package ingestion pytest ingestion/tests -v`
Expected: all pass (Task 1/2/3's new tests + all pre-existing FIRMS tests + the smoke test)

- [ ] **Step 6: Commit**

```bash
git add ingestion/src/ingestion/dem/cache.py ingestion/src/ingestion/dem/pipeline.py ingestion/tests/test_dem_pipeline.py
git commit -m "feat(ingestion): mosaic and reproject DEM tiles with bbox-hash caching"
```

---

### Task 4: `features/terrain/slope_aspect.py` — Horn's method + GeoTIFF output

**Files:**
- Create: `features/src/features/terrain/slope_aspect.py`
- Test: `features/tests/test_terrain_slope_aspect.py`

**Interfaces:**
- Consumes: nothing structurally from Tasks 1-3 (takes a DEM array or a path — decoupled, like the FIRMS parser/storage split) — used operationally after `ingestion.dem.pipeline.build_dem`, but not imported from it.
- Produces: `features.terrain.slope_aspect.compute_slope_aspect(elevation: np.ndarray, cellsize_x: float, cellsize_y: float) -> tuple[np.ndarray, np.ndarray]` (returns `(slope_deg, aspect_deg)`, both same shape as `elevation`; `aspect_deg` is `-1.0` for flat cells), `features.terrain.slope_aspect.compute_and_save_terrain(dem_path: Path, output_dir: Path) -> tuple[Path, Path]` (reads `dem_path` with rasterio, computes both, writes `slope_deg.tif` and `aspect_deg.tif` into `output_dir`, preserving CRS/transform; returns their paths).

- [ ] **Step 1: Write the failing tests**

`features/tests/test_terrain_slope_aspect.py`:

```python
"""Tests de pendiente/orientación (método de Horn) contra un DEM sintético
con pendiente conocida analíticamente."""
import math
from pathlib import Path

import numpy as np
import rasterio
from features.terrain.slope_aspect import compute_and_save_terrain, compute_slope_aspect
from rasterio.transform import from_origin


def test_flat_plane_has_zero_slope_and_sentinel_aspect():
    elevation = np.full((5, 5), 100.0, dtype="float64")
    slope, aspect = compute_slope_aspect(elevation, cellsize_x=1.0, cellsize_y=1.0)
    assert np.allclose(slope, 0.0)
    assert np.all(aspect == -1.0)


def test_plane_rising_to_the_east_has_45_degree_slope_facing_west():
    # Z = column index (1 m rise per 1 m of cellsize eastward) -> analytically
    # a 45-degree slope, facing directly downhill = West (270 degrees).
    size = 7
    elevation = np.tile(np.arange(size, dtype="float64"), (size, 1))
    slope, aspect = compute_slope_aspect(elevation, cellsize_x=1.0, cellsize_y=1.0)
    interior = slice(1, -1)
    assert np.allclose(slope[interior, interior], 45.0, atol=1e-6)
    assert np.allclose(aspect[interior, interior], 270.0, atol=1e-6)


def test_plane_rising_to_the_northeast_has_non_cardinal_aspect():
    # Z increases with both row (northward, since row 0 = north edge under
    # a north-up transform used elsewhere) and column (eastward) equally ->
    # downhill direction is southwest = 225 degrees. This pins the atan2
    # argument order/sign: a backwards formula would land at 45 or 135,
    # not 225.
    size = 7
    rows = np.arange(size, dtype="float64").reshape(-1, 1)
    cols = np.arange(size, dtype="float64").reshape(1, -1)
    elevation = -rows + cols  # higher to the north (smaller row index) and east
    slope, aspect = compute_slope_aspect(elevation, cellsize_x=1.0, cellsize_y=1.0)
    interior = slice(1, -1)
    assert np.allclose(aspect[interior, interior], 225.0, atol=1e-6)


def test_compute_and_save_terrain_writes_two_geotiffs_preserving_crs(tmp_path):
    dem_path = tmp_path / "dem.tif"
    size = 6
    transform = from_origin(500000, 5800000, 250, 250)  # UTM-like, meters
    elevation = np.tile(np.arange(size, dtype="float32") * 10.0, (size, 1))
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:32719", transform=transform,
    ) as dst:
        dst.write(elevation, 1)

    output_dir = tmp_path / "terrain"
    slope_path, aspect_path = compute_and_save_terrain(dem_path, output_dir)

    assert slope_path.exists()
    assert aspect_path.exists()
    with rasterio.open(slope_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
        assert ds.transform == transform
    with rasterio.open(aspect_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
```

- [ ] **Step 1b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package features pytest features/tests/test_terrain_slope_aspect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.terrain.slope_aspect'`

- [ ] **Step 2: Write `features/src/features/terrain/slope_aspect.py`**

```python
"""Pendiente y orientación (aspect) a partir de un DEM, método de Horn (1981)
— el mismo algoritmo de kernel 3x3 que usan GDAL `gdaldem` y ESRI.

Fórmulas (fila = eje Y, aumenta hacia el sur bajo una transform north-up
estándar; columna = eje X, aumenta hacia el este):

    dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize_x)
    dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize_y)

    donde la ventana 3x3 centrada en el píxel es:
        a b c
        d e f
        g h i

    slope_deg  = grados(atan(hipot(dz/dx, dz/dy)))                    [0, 90]
    aspect_deg = (grados(atan2(-dz/dx, dz/dy))) mod 360                [0, 360)
                 -1.0 si la celda es plana (dz/dx == dz/dy == 0)

aspect es el rumbo de la dirección CUESTA ABAJO (hacia dónde "mira" la
ladera), medido en sentido horario desde el norte (0=N, 90=E, 180=S,
270=O) — la convención estándar en GIS para orientación de terreno.

Limitación conocida: el padding de borde usa modo 'edge' (replica el
píxel más cercano) y no hay manejo especial de nodata — un DEM con
huecos de datos producirá pendiente/orientación no confiables cerca de
esos huecos. Ver docs/data-sources.md.
"""
from pathlib import Path

import numpy as np
import rasterio


def compute_slope_aspect(
    elevation: np.ndarray, cellsize_x: float, cellsize_y: float
) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(elevation, pad_width=1, mode="edge")

    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]

    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize_x)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize_y)

    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    flat = (dzdx == 0.0) & (dzdy == 0.0)
    aspect_rad = np.arctan2(-dzdx, dzdy)
    aspect_deg = np.degrees(aspect_rad) % 360.0
    aspect_deg = np.where(flat, -1.0, aspect_deg)

    return slope_deg, aspect_deg


def compute_and_save_terrain(dem_path: Path, output_dir: Path) -> tuple[Path, Path]:
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype("float64")
        cellsize_x = src.transform.a
        cellsize_y = -src.transform.e  # e is negative for north-up rasters
        profile = src.profile

    slope_deg, aspect_deg = compute_slope_aspect(elevation, cellsize_x, cellsize_y)

    output_dir.mkdir(parents=True, exist_ok=True)
    slope_path = output_dir / "slope_deg.tif"
    aspect_path = output_dir / "aspect_deg.tif"

    out_profile = {**profile, "dtype": "float32", "count": 1, "nodata": None}
    with rasterio.open(slope_path, "w", **out_profile) as dst:
        dst.write(slope_deg.astype("float32"), 1)
    with rasterio.open(aspect_path, "w", **out_profile) as dst:
        dst.write(aspect_deg.astype("float32"), 1)

    return slope_path, aspect_path
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package features pytest features/tests/test_terrain_slope_aspect.py -v`
Expected: 4 passed

If the northeast-facing-plane test's expected `225.0` doesn't match (sign/argument-order slip in the formula above is easy even when transcribing correctly), do not adjust the test's expected value to match a wrong implementation — work out the correct sign from first principles (higher ground to the north and east ⇒ downhill is south and west ⇒ compass bearing 225°) and fix the formula, then ledger what was wrong.

- [ ] **Step 4: Run the full `features` suite and `mypy --strict`**

Run: `uv run --package features pytest features/tests -v`
Expected: all pass
Run: `uv run mypy --strict shared/src features/src`
Expected: `Success: no issues found in N source files`

- [ ] **Step 5: Commit**

```bash
git add features/src/features/terrain/slope_aspect.py features/tests/test_terrain_slope_aspect.py
git commit -m "feat(features): compute slope/aspect from DEM via Horn's method"
```

---

### Task 5: `mypy --strict`/`ruff` on `ingestion/dem/`, `docs/data-sources.md`, `docs/decisions.md`

**Files:**
- Create/Modify: any `ingestion/dem/*.py` needed to satisfy `mypy --strict` (not in the Makefile's scope, so not caught by earlier steps)
- Modify: `docs/data-sources.md` (append a DEM section)
- Modify: `docs/decisions.md` (append the AWS-vs-OpenTopography choice)

- [ ] **Step 1: Run `mypy --strict` and `ruff` on `ingestion/dem/` specifically**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run mypy --strict ingestion/src/ingestion/dem`
Run: `uv run ruff check ingestion/ features/`

Fix whatever surfaces — likely candidates: `rasterio`'s own type stubs are incomplete for some return types (e.g. `merge()`'s tuple, `calculate_default_transform`'s tuple) and may need a narrowly-scoped `# type: ignore[no-any-return]` or similar with the specific error code, never a blanket ignore. Do not weaken the check to pass it.

Expected after fixes: both commands report no errors/issues.

- [ ] **Step 2: Append the DEM section to `docs/data-sources.md`**

```markdown

## Copernicus DEM GLO-30

**Qué entrega:** modelo de elevación digital global, ~30 m de resolución
nativa (1 arco-segundo), usado para calcular pendiente y orientación.

**Fuente elegida y por qué:** bucket público de AWS
(`s3://copernicus-dem-30m`, ver
https://registry.opendata.aws/copernicus-dem/), servido también sobre
HTTPS plano sin credenciales — en vez de la API de OpenTopography, que
exige una API key gratuita adicional y no documenta públicamente sus
límites de tasa/área. El bucket de AWS no requiere ninguna variable de
entorno nueva. Ver `docs/decisions.md` para el detalle completo de esta
decisión.

**Cómo se organiza:** cada tile cubre 1°x1°, nombrado por su esquina
suroeste (p. ej. `Copernicus_DSM_COG_10_S37_00_W072_00_DEM` cubre
`[-37,-36) x [-72,-71)`). `pyrocast` descarga solo los tiles que
intersectan el bbox configurado, los mosaica con `rasterio`, y
reproyecta el resultado a `EPSG:32719` (UTM 19S) en la resolución de
`shared/config.py` (250 m por defecto) usando remuestreo **bilineal**
(nunca "nearest" — nearest produce escalones artificiales en un DEM).

**Cacheo:** el resultado final (mosaico + reproyección) se cachea en
`data/processed/... ` con un nombre que incluye un hash de
`(bbox, resolución, CRS)` — si se pide el mismo bbox/resolución de
nuevo, no se vuelve a descargar ni reprocesar nada. Los tiles crudos
individuales también se cachean por su nombre (son globales/estáticos,
reutilizables entre bboxes distintos que compartan un tile).

**Pendiente y orientación:** ver `features/terrain/slope_aspect.py` —
método de Horn (1981), el mismo que usan GDAL `gdaldem` y ESRI. Pendiente
en grados (0-90), orientación en grados de rumbo (0-360, sentido
horario desde el norte, -1 para celdas planas). Se calculan sobre el
DEM ya reproyectado a EPSG:32719 (metros), nunca sobre el DEM crudo en
grados — de lo contrario el tamaño de celda en metros variaría con la
latitud y la pendiente quedaría mal calculada.

**Limitaciones conocidas:**
- **GLO-30 Public tiene huecos**: una fracción de tiles globales no
  está liberada públicamente por el programa Copernicus (variante
  `COP-DEM-GLO-30-R` vs. `Public`); si el área de estudio cayera en uno
  de esos huecos, la descarga fallaría con 404 — no verificado
  exhaustivamente para Biobío/Ñuble/Araucanía en este bootstrap.
- **Sin manejo de nodata en el cálculo de pendiente/orientación**: el
  kernel de Horn usa relleno de borde ("edge padding") pero no
  enmascara nodata — celdas cerca de huecos de datos producirán
  valores de pendiente/orientación no confiables.
- **Resolución nativa ~30 m, remuestreada a 250 m**: se pierde detalle
  de microrelieve; consistente con la simplificación deliberada de
  resolución ya documentada para todo el proyecto (ver limitations.md).
```

- [ ] **Step 3: Append the decision to `docs/decisions.md`**

```markdown

## Copernicus DEM: bucket público de AWS en vez de OpenTopography

CLAUDE.md permite ambas opciones. Se eligió el bucket de AWS
(`copernicus-dem-30m`, https://registry.opendata.aws/copernicus-dem/)
verificando en vivo (2026-09-26) que es completamente público sobre
HTTPS plano (`curl -sI` sobre `tileList.txt` y sobre una URL de tile
real devuelven `200 OK` sin ningún header de autenticación). Esto evita:
una variable de entorno de credencial nueva, un campo nuevo en
`shared/config.Settings`, una dependencia de `boto3`/AWS SDK (una
petición HTTP plana con `requests` basta), y un límite de tasa/área no
documentado (el de OpenTopography no se encontró públicamente). La
grilla de tiles (1°x1°, nombrados por esquina suroeste) es matemática
simple y determinística — no hay necesidad de una API de recorte por
bbox del lado del servidor cuando se pueden calcular exactamente los
tiles necesarios.
```

- [ ] **Step 4: Full workspace re-verification**

Run: `uv run --package ingestion pytest ingestion/tests -v`
Run: `uv run --package features pytest features/tests -v`
Run: `uv run ruff check .`
Run: `uv run mypy --strict shared/src features/src`
Run: `uv run mypy --strict ingestion/src/ingestion/dem`

Expected: all green, matching this task's acceptance criteria verbatim.

- [ ] **Step 5: Commit**

```bash
git add docs/data-sources.md docs/decisions.md
git commit -m "docs: add DEM data-sources section and AWS-vs-OpenTopography decision"
```

---

## Self-Review Notes

- **Spec coverage:** user's 5 tasks map to: 1→Task 1+2 (tile grid + download), 2→Task 3 (mosaic/reproject/bilinear), 3→Task 4 (slope/aspect, GeoTIFF output), 4→Task 3 (bbox+resolution-hash cache), 5→woven through every task's own tests (synthetic DEM with known slope in Task 4; mocked HTTP in Task 2/3). Acceptance criteria (mypy --strict/ruff, tests green with no network, docs/data-sources.md with formula+units) → Task 5.
- **Placeholder scan:** every step has real, runnable code. The one intentionally-left-in error in Task 2 Step 2 (a redundant test to delete) is flagged explicitly with exactly what to remove and why, not a TODO.
- **Type consistency:** `tile_key`/`tiles_for_bbox` (Task 1) return types match what Task 2's `download_tile` and Task 3's `pipeline.py` consume (`(lat, lon)` int tuples, `str` keys). `build_dem`'s `download_fn` parameter signature (`Callable[[str, Path], Path]`) matches both the real `download_tile` (Task 2) and every test's `fake_download` (Task 3).
- **Review Focus:** all five items have an owning test — bbox edge-on-integer (Task 1's `includes_tile_when_edge_lands_on_integer_degree`), southern/western sign handling (Task 1's `tile_key_south_west_hemisphere` using our actual negative coordinates), cache short-circuit call-count (Task 3's `cache_hit_skips_download_entirely`), bilinear-vs-nearest (Task 3's `unique_vals` interpolation check), flat-terrain sentinel + non-cardinal wrap-around (Task 4's `flat_plane` and `rising_to_the_northeast` tests).
