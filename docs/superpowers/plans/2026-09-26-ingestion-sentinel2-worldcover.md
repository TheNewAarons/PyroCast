# Ingestion Sentinel-2 + WorldCover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ingestion/sentinel2/` (openEO-based least-cloud monthly Sentinel-2 L2A composite), `features/vegetation/` (NDVI + SCL cloud masking + reprojection), `ingestion/worldcover/` (ESA WorldCover tile download/mosaic/categorical-resample + fuel-type mapping), wire both into the CLI/Makefile, and close out Stage 1 (P1–P4) with a homogeneous 4-source `docs/data-sources.md` and a `docs/decisions.md` reprojection/resampling summary. Zero real network/credentials in tests.

**Architecture:** `ingestion/sentinel2/client.py` wraps `openeo.connect(...).authenticate_oidc_client_credentials(...)`, builds a process graph (`load_collection` → SCL-based `.mask()` → `.reduce_dimension(..., "median")` temporal composite) and downloads a 3-band GeoTIFF (B04, B08, SCL, in that fixed order — an explicit contract with `features/vegetation`). `features/vegetation/ndvi.py` re-applies an SCL cloud mask locally (defense in depth, and the only way to unit-test masking without mocking openEO's server-side graph), computes NDVI, and reprojects to the 250 m EPSG:32719 grid with **bilinear** resampling (NDVI is continuous, like temperature). `ingestion/worldcover/` mirrors `ingestion/dem/`'s exact shape (tile grid math, atomic per-tile download, bbox-hash cache, mosaic+reproject) but applies every lesson from `ingestion/dem/`'s final review up front: half-open tile bounds using `ceil(x)-1` from the start, atomic `.part`+`os.replace()` downloads from the start, forced/explicit nodata from the start, and a wired CLI command from the start — plus **`Resampling.nearest`, never bilinear**, because WorldCover pixel values are categorical class codes, not continuous measurements. `ingestion/worldcover/fuel_type.py` holds the WorldCover-class → fuel-type lookup table the user's Task 3 explicitly places under `worldcover/`, not `features/`.

**Tech Stack:** Python 3.12, `openeo` (new — see Global Constraints), `rasterio`/`numpy` (already approved, declared on `ingestion` and `features` from prior plans), `requests` (WorldCover tile downloads, already declared), `pytest`. No new HTTP-mocking library needed for Sentinel-2 (like `cdsapi`, `openeo.Connection` is mocked at the Python-object level via constructor injection, not the HTTP layer); WorldCover downloads reuse the `responses`-mocking pattern already proven in `ingestion/dem/`.

**Spec:** User's message in this conversation (5 numbered tasks + acceptance criteria) plus `/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md` (Sentinel-2/WorldCover rows of the data-sources table). Both travel with this plan.

**openEO / Copernicus Data Space Ecosystem — verified against live sources (2026-09-26), not assumed from memory:**
- Docs: https://documentation.dataspace.copernicus.eu/APIs/openEO/Python_Client/Python.html, https://documentation.dataspace.copernicus.eu/APIs/openEO/authentication/client_credentials.html (fetched directly in this session).
- Connect: `openeo.connect("openeofed.dataspace.copernicus.eu")`. Auth: `connection.authenticate_oidc_client_credentials(client_id=..., client_secret=...)` — matches `shared.config.Settings.copernicus_dataspace_client_id`/`copernicus_dataspace_client_secret` **exactly**, already present in `shared/config.py` from the bootstrap. No new `.env` variable, no new config field.
- `load_collection("SENTINEL2_L2A", spatial_extent={"west":..,"south":..,"east":..,"north":..}, temporal_extent=[start,end], bands=[...], max_cloud_cover=...)` — `spatial_extent` is a **named-key dict** matching `shared.config.Settings.study_area_bbox`'s `(west, south, east, north)` order directly, no axis reordering needed (unlike CDS's ERA5 `[N,W,S,E]` from the prior plan — this source has no such gotcha).
- Cloud masking pattern (from the docs' own example, adapted): `datacube.band("SCL")`, build a boolean mask, `mask.resample_cube_spatial(datacube)`, `datacube.mask(mask_resampled)`. This plan uses the standard SCL cloud/shadow class set `{3, 8, 9, 10}` (cloud shadows, cloud medium probability, cloud high probability, thin cirrus) rather than the docs' own narrow single-class example (which masked everything except class 4 for a vegetation-only tutorial) — that example doesn't match "mask clouds," it masks "keep only vegetation," which is not what this task asks for.
- **openEO chosen over `sentinelhub-py`**: `sentinelhub-py`'s CDSE config additionally requires `sh_base_url` and `sh_token_url` fields (verified via its own docs) that `shared/config.py` doesn't have and would need to grow; `openeo`'s client-credentials auth needs only the client ID/secret pair already in `shared/config.py`. Documented as a decision in Task 5.

**ESA WorldCover — verified against the live public bucket (2026-09-26), not assumed from memory:**
- Public bucket, no credentials: `curl -sI` against a real tile URL and an `S3 ListObjectsV2` query against `https://esa-worldcover.s3.eu-central-1.amazonaws.com/?list-type=2&prefix=...` both returned real results with no auth headers.
- Tiles are **3°×3°**, EPSG:4326, named `ESA_WorldCover_10m_2021_v200_{NS}{lat:02d}{EW}{lon:03d}_Map.tif` (confirmed real keys: `..._S39W072_Map.tif`, `..._N00E006_Map.tif` — no separators between the lat/lon parts, unlike Copernicus DEM's tile names) at `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{filename}`. Same **SW-corner, half-open** convention as Copernicus DEM tiles (confirmed by reading a real tile's transform: `S39W072`'s northwest corner sits at `(-72.00, -36.00)`, i.e. the tile covers `[-39,-36) × [-72,-69)`).
- A real tile's profile (read via `/vsicurl/`, no download): `nodata=0.0` (**properly declared**, unlike Copernicus DEM's `nodata=None`), `dtype=uint8`, `crs=EPSG:4326`, 36000×36000 px.
- Class legend (11 classes, confirmed via `https://collections.sentinel-hub.com/worldcover/readme.html`): `10`=Tree cover, `20`=Shrubland, `30`=Grassland, `40`=Cropland, `50`=Built-up, `60`=Bare/sparse vegetation, `70`=Snow and ice, `80`=Permanent water bodies, `90`=Herbaceous wetland, `95`=Mangroves, `100`=Moss and lichen.

## Global Constraints

- New dependency: `openeo` (on `ingestion`) — not in CLAUDE.md's explicit list. Justify in `docs/decisions.md` (Task 5): it's the client library CLAUDE.md itself names as an option ("OpenEO o sentinelhub-py"), so declaring it is expected, not a surprise addition — but it's still new and gets the paragraph.
- `openeo.Connection` (and the object it returns from `.load_collection()`/`.mask()`/`.reduce_dimension()`/`.download()`) is never imported or instantiated for real in any test — every test injects a fake connection via constructor-parameter dependency injection, the same pattern as `Era5Client`'s `client_factory` and `ingestion/dem`'s `download_fn`.
- WorldCover tile math uses **`ceil(x) - 1`** for the upper (north/east) bound from the start — this plan does not repeat `ingestion/dem`'s original mistake of using `floor()` for both bounds (that bug was found and fixed in the DEM final review; the lesson transfers directly since both sources use the identical SW-corner half-open tile convention, just at a different tile size — 3° here vs. 1° for DEM).
- WorldCover's real tiles **do** declare `nodata=0.0` (verified above) — still force it explicitly end-to-end in the mosaic/reproject/write pipeline rather than trusting `sources[0].nodata` blindly, for the same defense-in-depth reason `ingestion/dem` now does (a future WorldCover version or a substitute tile could omit it).
- WorldCover reprojection/resampling **must use `Resampling.nearest`**, never bilinear — the pixel values are categorical class codes; averaging two adjacent codes during interpolation would fabricate a class that doesn't exist (e.g. blending `10` Tree cover and `80` Water into `45`, a meaningless value). This is the one Review Focus item most likely to be gotten backwards by habit, since every other raster pipeline in this project (DEM, ERA5) correctly uses bilinear.
- NDVI reprojection uses **bilinear** (it's continuous, like DEM elevation and ERA5-Land fields) — the opposite resampling choice from WorldCover, in the same plan, is deliberate and must be visible in both the code comments and `docs/data-sources.md`.
- The Sentinel-2 composite's 3-band GeoTIFF ordering (B04, B08, SCL) is an implicit contract between `ingestion/sentinel2/client.py` (producer) and `features/vegetation/ndvi.py` (consumer) — document it in both places, not just one.
- Identifiers in English; docstrings/comments/docs in Spanish (CLAUDE.md convention).
- Both new ingestion modules get a `pyrocast-ingest <name>` CLI command and a real `Makefile` target from the start (Task 6) — the DEM review found this missing was a real, user-visible gap; this plan doesn't wait for a review to add it.

## Review Focus

- **WorldCover tile grid upper bound**: a bbox edge landing exactly on a multiple of 3 must use the `ceil(x)-1` fix, tested directly against the actual half-open tile semantics — not re-derived from scratch and re-broken.
- **Resampling method per source, in the same plan**: WorldCover must be `Resampling.nearest` (categorical — must literally verify no intermediate/fabricated class codes appear after reprojection, the same way the DEM plan proved bilinear interpolation with a "more than N unique values" assertion, but inverted: prove nearest by asserting the *only* values present are ones that existed in the source). NDVI must be `Resampling.bilinear`. Getting these backwards is easy since they're adjacent in the same session's muscle memory.
- **SCL cloud-class masking correctness**: a synthetic SCL fixture must contain both a clearly-cloud pixel (class 8 or 9) and a clearly-clear pixel (class 4, vegetation, or 6, bare soil) side by side, and the test must assert the cloud pixel's NDVI is masked out (nodata) while the clear pixel's NDVI is computed normally — a test that only checks "some masking happened" without pinning per-pixel behavior could hide an inverted mask (masking the wrong pixels).
- **NDVI formula edge case**: a pixel where `NIR + RED == 0` (both bands zero — a genuine nodata/black-fill pixel, not a masked-cloud pixel) must not produce a `ZeroDivisionError` or a silent `inf`/`nan` that then poisons a downstream reprojection average; must resolve to a defined nodata value.
- **Fuel-type mapping must not silently invent a fuel type for an unmapped WorldCover class** — the lookup must be a closed set matching exactly the 11 documented classes, and an out-of-range/unexpected pixel value (e.g. from a corrupted tile or an unanticipated future WorldCover version) must map to an explicit "unknown" sentinel, not silently fall through to whatever a `dict.get(..., default)` happens to default to without that default being a deliberate, documented choice.

---

## File Structure

```
ingestion/
├── pyproject.toml                      # + openeo
├── src/ingestion/
│   ├── cli.py                          # + sentinel2, worldcover commands
│   ├── sentinel2/
│   │   ├── __init__.py                 # (exists, stub)
│   │   ├── client.py                   # new: Sentinel2Client
│   │   ├── cache.py                    # new: cache_key_for
│   │   ├── pipeline.py                 # new: fetch_sentinel2
│   │   └── cli.py                      # new: `sentinel2` command
│   └── worldcover/
│       ├── __init__.py                 # (exists, stub)
│       ├── tiles.py                    # new: tile grid (3°, ceil()-1 from the start)
│       ├── client.py                   # new: download_tile (atomic from the start)
│       ├── cache.py                    # new: cache_key_for
│       ├── fuel_type.py                # new: CLASS_TO_FUEL_TYPE mapping
│       ├── pipeline.py                 # new: build_worldcover
│       └── cli.py                      # new: `worldcover` command
└── tests/
    ├── test_sentinel2_client.py
    ├── test_sentinel2_pipeline.py
    ├── test_worldcover_tiles.py
    ├── test_worldcover_client.py
    ├── test_worldcover_pipeline.py
    ├── test_worldcover_fuel_type.py
    ├── test_sentinel2_cli.py
    └── test_worldcover_cli.py

features/
├── src/features/vegetation/
│   ├── __init__.py                     # (exists, stub)
│   └── ndvi.py                         # new: compute_ndvi, mask_clouds, compute_and_save_vegetation
└── tests/
    └── test_vegetation_ndvi.py

Makefile                                 # ingest-vegetation wired
docs/
├── data-sources.md                     # rewritten: 4 sources, homogeneous format
└── decisions.md                        # + openEO choice, + P1-P4 reprojection/resampling summary
```

---

### Task 1: `ingestion/sentinel2/client.py` — openEO client (mocked)

**Files:**
- Modify: `ingestion/pyproject.toml` (+ `openeo`)
- Create: `ingestion/src/ingestion/sentinel2/client.py`
- Test: `ingestion/tests/test_sentinel2_client.py`

**Interfaces:**
- Produces: `ingestion.sentinel2.client.SENTINEL2_BANDS = ("B04", "B08", "SCL")`, `ingestion.sentinel2.client.CLOUD_SCL_CLASSES = frozenset({3, 8, 9, 10})`, `ingestion.sentinel2.client.Sentinel2Client(client_id: str, client_secret: str, connect_fn: Callable[[str], Any] = openeo.connect, backend_url: str = "openeofed.dataspace.copernicus.eu")` with method `fetch_monthly_composite(bbox: tuple[float,float,float,float], year: int, month: int, target: Path, max_cloud_cover: int = 85) -> Path`.

- [ ] **Step 1: Add `openeo` to `ingestion/pyproject.toml`**

```toml
dependencies = [
    "shared",
    "features",
    "requests>=2.32",
    "cdsapi>=0.7",
    "typer>=0.12",
    "pyarrow>=17.0",
    "rasterio>=1.3",
    "numpy>=2.0",
    "xarray>=2024.7",
    "h5netcdf>=1.3",
    "h5py>=3.11",
    "openeo>=0.31",
]
```

- [ ] **Step 2: Write the failing tests**

`ingestion/tests/test_sentinel2_client.py`:

```python
"""Tests del cliente Sentinel-2: openeo.Connection mockeado por completo."""
from pathlib import Path

from ingestion.sentinel2.client import CLOUD_SCL_CLASSES, SENTINEL2_BANDS, Sentinel2Client

BBOX = (-73.7, -39.3, -71.0, -36.5)  # west, south, east, north


class _FakeDataCube:
    def __init__(self, log: list, name: str = "load_collection"):
        self._log = log
        self._name = name

    def band(self, name: str) -> "_FakeDataCube":
        self._log.append(("band", name))
        return _FakeDataCube(self._log, f"band:{name}")

    def __ne__(self, other) -> "_FakeDataCube":  # scl_band != cloud_class
        self._log.append(("ne", other))
        return _FakeDataCube(self._log, "mask")

    def __and__(self, other) -> "_FakeDataCube":
        self._log.append(("and", "combine"))
        return _FakeDataCube(self._log, "mask")

    def resample_cube_spatial(self, target: "_FakeDataCube") -> "_FakeDataCube":
        self._log.append(("resample_cube_spatial",))
        return self

    def mask(self, mask_cube: "_FakeDataCube") -> "_FakeDataCube":
        self._log.append(("mask",))
        return self

    def reduce_dimension(self, dimension: str, reducer: str) -> "_FakeDataCube":
        self._log.append(("reduce_dimension", dimension, reducer))
        return self

    def download(self, target: str) -> None:
        self._log.append(("download", target))
        Path(target).write_bytes(b"fake-composite-bytes")


class _FakeConnection:
    def __init__(self):
        self.log: list = []
        self.authenticated_with: tuple | None = None

    def authenticate_oidc_client_credentials(self, client_id: str, client_secret: str) -> None:
        self.authenticated_with = (client_id, client_secret)

    def load_collection(self, collection_id, spatial_extent, temporal_extent, bands, max_cloud_cover):
        self.log.append(
            ("load_collection", collection_id, spatial_extent, temporal_extent, bands, max_cloud_cover)
        )
        return _FakeDataCube(self.log)


def test_fetch_monthly_composite_authenticates_with_client_credentials(tmp_path):
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="my-id", client_secret="my-secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2026, month=1, target=tmp_path / "out.tif")
    assert fake_connection.authenticated_with == ("my-id", "my-secret")


def test_fetch_monthly_composite_uses_named_spatial_extent_no_axis_reorder(tmp_path):
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2026, month=1, target=tmp_path / "out.tif")
    call = next(c for c in fake_connection.log if c[0] == "load_collection")
    _, collection_id, spatial_extent, temporal_extent, bands, _ = call
    assert collection_id == "SENTINEL2_L2A"
    assert spatial_extent == {"west": -73.7, "south": -39.3, "east": -71.0, "north": -36.5}
    assert temporal_extent == ["2026-01-01", "2026-01-31"]
    assert bands == list(SENTINEL2_BANDS)


def test_fetch_monthly_composite_downloads_to_target(tmp_path):
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    target = tmp_path / "out.tif"
    result = client.fetch_monthly_composite(BBOX, year=2026, month=2, target=target)
    assert result == target
    assert target.read_bytes() == b"fake-composite-bytes"


def test_february_month_end_is_28_not_30():
    # un mes de 28/29/30/31 días construido con calendar, no un +30 fijo
    fake_connection = _FakeConnection()
    client = Sentinel2Client(
        client_id="id", client_secret="secret", connect_fn=lambda url: fake_connection
    )
    client.fetch_monthly_composite(BBOX, year=2025, month=2, target=Path("/tmp/unused.tif"))
    call = next(c for c in fake_connection.log if c[0] == "load_collection")
    assert call[3] == ["2025-02-01", "2025-02-28"]  # 2025 no es bisiesto


def test_cloud_scl_classes_are_the_documented_set():
    assert CLOUD_SCL_CLASSES == frozenset({3, 8, 9, 10})
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv sync --all-packages --group dev --reinstall-package ingestion && uv run --package ingestion pytest ingestion/tests/test_sentinel2_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.sentinel2.client'`

- [ ] **Step 3: Write `ingestion/src/ingestion/sentinel2/client.py`**

```python
"""Cliente Sentinel-2 L2A vía openEO (Copernicus Data Space Ecosystem).

Verificado contra la documentación vigente (2026-09-26):
https://documentation.dataspace.copernicus.eu/APIs/openEO/ — elegido
sobre sentinelhub-py porque su autenticación por client credentials usa
solo client_id/client_secret (ya en shared/config.py), sin campos de
configuración adicionales (sentinelhub-py necesita además sh_base_url y
sh_token_url). Ver docs/decisions.md.

Composición mensual de menor nubosidad: se carga SENTINEL2_L2A con las
bandas B04 (rojo), B08 (NIR) y SCL (Scene Classification), se enmascaran
las clases de nube/sombra de SCL, y se reduce sobre el tiempo con la
mediana — el resultado se descarga como GeoTIFF de 3 bandas en ESE ORDEN
(B04, B08, SCL): es un contrato implícito con
features/vegetation/ndvi.py, que lee las bandas por posición.
"""
import calendar
import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openeo

SENTINEL2_BANDS: tuple[str, ...] = ("B04", "B08", "SCL")
# Sombra de nube, nube prob. media, nube prob. alta, cirros delgados.
CLOUD_SCL_CLASSES: frozenset[int] = frozenset({3, 8, 9, 10})


class Sentinel2Client:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        connect_fn: Callable[[str], Any] = openeo.connect,
        backend_url: str = "openeofed.dataspace.copernicus.eu",
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._connection = connect_fn(backend_url)
        self._connection.authenticate_oidc_client_credentials(
            client_id=client_id, client_secret=client_secret
        )

    def fetch_monthly_composite(
        self,
        bbox: tuple[float, float, float, float],
        year: int,
        month: int,
        target: Path,
        max_cloud_cover: int = 85,
    ) -> Path:
        west, south, east, north = bbox
        last_day = calendar.monthrange(year, month)[1]
        start = dt.date(year, month, 1).isoformat()
        end = dt.date(year, month, last_day).isoformat()

        datacube = self._connection.load_collection(
            "SENTINEL2_L2A",
            spatial_extent={"west": west, "south": south, "east": east, "north": north},
            temporal_extent=[start, end],
            bands=list(SENTINEL2_BANDS),
            max_cloud_cover=max_cloud_cover,
        )

        scl_band = datacube.band("SCL")
        cloud_mask = None
        for cloud_class in CLOUD_SCL_CLASSES:
            class_mask = scl_band != cloud_class
            cloud_mask = class_mask if cloud_mask is None else (cloud_mask & class_mask)
        mask_resampled = cloud_mask.resample_cube_spatial(datacube)
        masked = datacube.mask(mask_resampled)

        composite = masked.reduce_dimension(dimension="t", reducer="median")
        composite.download(str(target))
        return target
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_sentinel2_client.py -v`
Expected: 5 passed

Note: the `_FakeDataCube.__ne__`/`__and__` overloads in the test double are a simplified stand-in for openEO's real `DataCube` boolean-algebra overloads — they only need to support chaining, not compute anything, since the fake `.mask()`/`.download()` never touch pixel data. If `cloud_mask & class_mask`'s repeated-`&`-folding doesn't match how the real `openeo.rest.datacube.DataCube` overloads `__and__` (verify by checking the actual openEO Python client source if this feels uncertain), that's a real-world fidelity gap worth a one-line ledger note — it does not block this task, since the whole point of Task 1 is the request-building and auth wiring, not the exact boolean-mask expression tree.

- [ ] **Step 5: Commit**

```bash
git add ingestion/pyproject.toml ingestion/src/ingestion/sentinel2/client.py ingestion/tests/test_sentinel2_client.py uv.lock
git commit -m "feat(ingestion): add Sentinel-2 L2A client via openEO (mocked tests)"
```

---

### Task 2: `ingestion/sentinel2/cache.py` + `pipeline.py` — bbox+year-month cache

**Files:**
- Create: `ingestion/src/ingestion/sentinel2/cache.py`
- Create: `ingestion/src/ingestion/sentinel2/pipeline.py`
- Test: `ingestion/tests/test_sentinel2_pipeline.py`

**Interfaces:**
- Consumes: `Sentinel2Client.fetch_monthly_composite` (Task 1).
- Produces: `ingestion.sentinel2.cache.cache_key_for(bbox, year, month) -> str`, `ingestion.sentinel2.pipeline.fetch_sentinel2(bbox, year, month, client, cache_dir) -> Path`.

- [ ] **Step 1: Write `ingestion/src/ingestion/sentinel2/cache.py`**

```python
"""Clave de cache para una composición mensual de Sentinel-2: hash de
(bbox, año, mes)."""
import hashlib


def cache_key_for(bbox: tuple[float, float, float, float], year: int, month: int) -> str:
    payload = f"{bbox}|{year:04d}-{month:02d}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 2: Write the failing pipeline tests**

`ingestion/tests/test_sentinel2_pipeline.py`:

```python
"""Tests del pipeline Sentinel-2: cache + orquestación, openEO mockeado."""
from pathlib import Path

from ingestion.sentinel2.pipeline import fetch_sentinel2

BBOX = (-73.7, -39.3, -71.0, -36.5)


class _FakeSentinel2Client:
    def __init__(self):
        self.calls: list = []

    def fetch_monthly_composite(self, bbox, year, month, target, **kwargs):
        self.calls.append(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-composite")
        return target


def test_fetch_sentinel2_produces_composite(tmp_path):
    client = _FakeSentinel2Client()
    result = fetch_sentinel2(
        bbox=BBOX, year=2026, month=1, client=client, cache_dir=tmp_path / "cache"
    )
    assert result.exists()
    assert len(client.calls) == 1


def test_fetch_sentinel2_cache_hit_skips_fetch_entirely(tmp_path):
    client = _FakeSentinel2Client()
    first = fetch_sentinel2(
        bbox=BBOX, year=2026, month=1, client=client, cache_dir=tmp_path / "cache"
    )
    second = fetch_sentinel2(
        bbox=BBOX, year=2026, month=1, client=client, cache_dir=tmp_path / "cache"
    )
    assert second == first
    assert len(client.calls) == 1  # no segunda llamada


def test_fetch_sentinel2_different_month_is_a_cache_miss(tmp_path):
    client = _FakeSentinel2Client()
    fetch_sentinel2(bbox=BBOX, year=2026, month=1, client=client, cache_dir=tmp_path / "cache")
    fetch_sentinel2(bbox=BBOX, year=2026, month=2, client=client, cache_dir=tmp_path / "cache")
    assert len(client.calls) == 2
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_sentinel2_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.sentinel2.pipeline'`

- [ ] **Step 3: Write `ingestion/src/ingestion/sentinel2/pipeline.py`**

```python
"""Orquesta: cache -> composición mensual Sentinel-2 vía Sentinel2Client."""
from pathlib import Path
from typing import Any

from ingestion.sentinel2.cache import cache_key_for


def fetch_sentinel2(
    bbox: tuple[float, float, float, float],
    year: int,
    month: int,
    client: Any,
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key_for(bbox, year, month)
    target = cache_dir / f"sentinel2_{key}.tif"
    if target.exists():
        return target
    return client.fetch_monthly_composite(bbox, year, month, target)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_sentinel2_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/src/ingestion/sentinel2/cache.py ingestion/src/ingestion/sentinel2/pipeline.py ingestion/tests/test_sentinel2_pipeline.py
git commit -m "feat(ingestion): cache Sentinel-2 monthly composites by bbox+year-month"
```

---

### Task 3: `features/vegetation/ndvi.py` — NDVI, SCL masking, reprojection

**Files:**
- Create: `features/src/features/vegetation/ndvi.py`
- Test: `features/tests/test_vegetation_ndvi.py`

**Interfaces:**
- Consumes: nothing structurally from Tasks 1-2 (takes a composite GeoTIFF path — decoupled, matching `features/terrain`/`features/weather`'s precedent).
- Produces: `features.vegetation.ndvi.compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray` (nodata `-9999.0` where `red+nir == 0`), `features.vegetation.ndvi.mask_clouds(band: np.ndarray, scl: np.ndarray, cloud_classes: frozenset[int]) -> np.ndarray` (sets `band` to `np.nan` where `scl` is in `cloud_classes`), `features.vegetation.ndvi.compute_and_save_vegetation(composite_path: Path, output_dir: Path, target_crs: str, target_resolution_m: int) -> Path` (reads bands 1/2/3 = B04/B08/SCL by position, masks clouds, computes NDVI, reprojects bilinear, writes one GeoTIFF).

- [ ] **Step 1: Write the failing tests**

`features/tests/test_vegetation_ndvi.py`:

```python
"""Tests de NDVI, enmascarado de nubes (SCL) y remuestreo, sin red."""
import numpy as np
import pytest
import rasterio
from features.vegetation.ndvi import compute_and_save_vegetation, compute_ndvi, mask_clouds
from rasterio.transform import from_origin


def test_compute_ndvi_known_value():
    # NIR=0.8, RED=0.2 -> NDVI = (0.8-0.2)/(0.8+0.2) = 0.6
    red = np.array([[0.2]])
    nir = np.array([[0.8]])
    ndvi = compute_ndvi(red, nir)
    assert ndvi[0, 0] == pytest.approx(0.6)


def test_compute_ndvi_handles_zero_denominator_without_raising():
    red = np.array([[0.0]])
    nir = np.array([[0.0]])
    ndvi = compute_ndvi(red, nir)
    assert ndvi[0, 0] == -9999.0  # nodata explícito, no NaN/inf propagando


def test_mask_clouds_masks_only_cloud_pixels():
    # fila 0: nube (SCL=9); fila 1: claro (SCL=4, vegetación)
    band = np.array([[0.5], [0.7]])
    scl = np.array([[9], [4]])
    masked = mask_clouds(band, scl, cloud_classes=frozenset({3, 8, 9, 10}))
    assert np.isnan(masked[0, 0])
    assert masked[1, 0] == pytest.approx(0.7)


def test_compute_and_save_vegetation_masks_cloud_pixel_before_ndvi(tmp_path):
    composite_path = tmp_path / "composite.tif"
    size = 4
    transform = from_origin(-72.0, -37.0, 0.0001, 0.0001)
    # banda 1=B04 (red), 2=B08 (nir), 3=SCL
    red = np.full((size, size), 0.2, dtype="float32")
    nir = np.full((size, size), 0.8, dtype="float32")
    scl = np.full((size, size), 4, dtype="float32")  # todo claro (vegetación)
    scl[0, 0] = 9  # una celda nublada
    with rasterio.open(
        composite_path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(red, 1)
        dst.write(nir, 2)
        dst.write(scl, 3)

    output_dir = tmp_path / "vegetation"
    ndvi_path = compute_and_save_vegetation(
        composite_path, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )

    assert ndvi_path.exists()
    with rasterio.open(ndvi_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
        assert ds.nodata is not None
        arr = ds.read(1)
    # la celda nublada original ya no existe 1:1 tras reproyectar/remuestrear,
    # pero el resultado no debe tener NaN sin marcar como nodata: cualquier
    # NaN residual del enmascarado debe haberse convertido a nodata antes
    # de escribir el GeoTIFF (GDAL no soporta NaN como nodata de forma
    # confiable en todos los drivers/dtypes).
    assert not np.any(np.isnan(arr))
```

- [ ] **Step 1b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package features pytest features/tests/test_vegetation_ndvi.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.vegetation.ndvi'`

- [ ] **Step 2: Write `features/src/features/vegetation/ndvi.py`**

```python
"""NDVI (Normalized Difference Vegetation Index) desde Sentinel-2 L2A, con
enmascarado de nubes por SCL y remuestreo a la grilla de trabajo.

Fórmula: NDVI = (NIR - RED) / (NIR + RED)               (adimensional, [-1, 1])

*** NDVI es un proxy del estado/vigor de la vegetación (verdor,
actividad fotosintética), NO una medición directa de humedad de
combustible. Vegetación con NDVI alto puede seguir teniendo bajo
contenido de humedad si está fenológicamente senescente o bajo estrés
hídrico no visible en el verdor foliar — no usar NDVI como sustituto de
una medición real de humedad de combustible. ***

Enmascarado de nubes: usa la banda SCL (Scene Classification Layer) de
Sentinel-2 — las clases {3,8,9,10} (sombra de nube, nube prob.
media/alta, cirros delgados) se tratan como sin dato. Ver
ingestion/sentinel2/client.py (mismo conjunto de clases, aplicado
también server-side vía openEO; esto es una segunda pasada local, no
redundante: permite testear el enmascarado sin mockear openEO).

Remuestreo: **bilineal** — NDVI es una magnitud continua, igual que la
elevación del DEM o los campos de ERA5-Land (a diferencia de WorldCover,
que es categórico y usa nearest).
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

_NDVI_NODATA = -9999.0


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denominator = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / denominator
    return np.where(denominator == 0, _NDVI_NODATA, ndvi)


def mask_clouds(band: np.ndarray, scl: np.ndarray, cloud_classes: frozenset[int]) -> np.ndarray:
    cloud_mask = np.isin(scl, list(cloud_classes))
    return np.where(cloud_mask, np.nan, band)


def compute_and_save_vegetation(
    composite_path: Path, output_dir: Path, target_crs: str, target_resolution_m: int
) -> Path:
    from ingestion.sentinel2.client import CLOUD_SCL_CLASSES  # contrato compartido

    with rasterio.open(composite_path) as src:
        red = src.read(1).astype("float64")
        nir = src.read(2).astype("float64")
        scl = src.read(3).astype("float64")
        src_crs = src.crs
        src_transform = src.transform
        height, width = src.height, src.width

    red_masked = mask_clouds(red, scl, CLOUD_SCL_CLASSES)
    nir_masked = mask_clouds(nir, scl, CLOUD_SCL_CLASSES)
    ndvi = compute_ndvi(red_masked, nir_masked)
    # Cualquier NaN que haya sobrevivido (celda nublada -> NaN en red/nir ->
    # NaN se propaga a través de la resta/suma en compute_ndvi, ya que
    # denominator == 0 no captura "denominator is NaN") se convierte a
    # nodata explícito antes de escribir — un NaN sin marcar sobreviviría
    # a la reproyección bilineal y contaminaría celdas vecinas al
    # promediar.
    ndvi = np.where(np.isnan(ndvi), _NDVI_NODATA, ndvi).astype("float32")

    west, south, east, north = rasterio.transform.array_bounds(height, width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, width, height, west, south, east, north,
        resolution=(target_resolution_m, target_resolution_m),
    )
    dst_array = np.full((dst_height, dst_width), _NDVI_NODATA, dtype="float32")
    reproject(
        source=ndvi,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=_NDVI_NODATA,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        dst_nodata=_NDVI_NODATA,
        resampling=Resampling.bilinear,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ndvi_path = output_dir / "ndvi.tif"
    with rasterio.open(
        ndvi_path, "w", driver="GTiff", height=dst_height, width=dst_width, count=1,
        dtype="float32", crs=target_crs, transform=dst_transform, nodata=_NDVI_NODATA,
    ) as dst:
        dst.write(dst_array, 1)

    return ndvi_path
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package features pytest features/tests/test_vegetation_ndvi.py -v`
Expected: 4 passed

If `test_compute_and_save_vegetation_masks_cloud_pixel_before_ndvi` fails because `compute_and_save_vegetation` importing `ingestion.sentinel2.client` creates a circular/layering problem (features importing from ingestion — the *opposite* direction of the already-accepted `ingestion -> features` exception from the DEM plan, and a new, different-direction dependency features would need on ingestion), do not silently add `ingestion` as a `features` dependency to make an import error disappear. Instead: duplicate the tiny `CLOUD_SCL_CLASSES` constant in `features/vegetation/ndvi.py` itself (it's a 6-word frozenset, not worth a cross-package dependency), remove the `from ingestion.sentinel2.client import ...` line, and ledger the ruling. Update `ingestion/sentinel2/client.py`'s docstring to note both copies must stay in sync (or, if there's time, note this as a follow-up: extracting the constant to `shared/` would remove the duplication, but is out of scope for this plan).

- [ ] **Step 4: Run the full `features` suite and `mypy --strict`**

Run: `uv run --package features pytest features/tests -v`
Expected: all pass
Run: `uv run mypy --strict shared/src features/src`
Expected: `Success: no issues found in N source files`

- [ ] **Step 5: Commit**

```bash
git add features/src/features/vegetation/ndvi.py features/tests/test_vegetation_ndvi.py
git commit -m "feat(features): compute NDVI from Sentinel-2 with SCL cloud masking"
```

---

### Task 4: `ingestion/worldcover/` — tile grid, atomic download, mosaic+nearest-reproject, fuel-type mapping

**Files:**
- Create: `ingestion/src/ingestion/worldcover/tiles.py`
- Create: `ingestion/src/ingestion/worldcover/client.py`
- Create: `ingestion/src/ingestion/worldcover/cache.py`
- Create: `ingestion/src/ingestion/worldcover/fuel_type.py`
- Create: `ingestion/src/ingestion/worldcover/pipeline.py`
- Test: `ingestion/tests/test_worldcover_tiles.py`, `test_worldcover_client.py`, `test_worldcover_pipeline.py`, `test_worldcover_fuel_type.py`

**Interfaces:**
- Produces: `ingestion.worldcover.tiles.tile_key(lat: int, lon: int, version: str = "v200", year: str = "2021") -> str`, `tile_url(key: str) -> str`, `tiles_for_bbox(bbox) -> list[tuple[int, int]]` (3° grid, `ceil(x)-1` upper bound); `ingestion.worldcover.client.download_tile(key, dest_path, session=None) -> Path` (atomic, same shape as `ingestion.dem.client.download_tile`), `WorldCoverDownloadError(RuntimeError)`; `ingestion.worldcover.cache.cache_key_for(bbox, resolution_m, crs, version) -> str`; `ingestion.worldcover.fuel_type.CLASS_TO_FUEL_TYPE: dict[int, int]`, `FUEL_TYPE_LABELS: dict[int, str]`, `map_worldcover_to_fuel_type(classes: np.ndarray) -> np.ndarray` (unmapped codes → `FUEL_TYPE_UNKNOWN`); `ingestion.worldcover.pipeline.build_worldcover(bbox, resolution_m, crs, raw_tiles_dir, cache_dir, version="v200", year="2021", download_fn=download_tile) -> Path`.

- [ ] **Step 1: Write `ingestion/src/ingestion/worldcover/tiles.py`**

```python
"""Grilla de tiles de ESA WorldCover (bucket público de AWS).

Cada tile cubre 3°x3°, nombrado por su esquina suroeste — mismo esquema
semiabierto que Copernicus DEM (ver ingestion/dem/tiles.py), verificado
contra el bucket real (2026-09-26): S39W072 cubre [-39,-36) x [-72,-69).
Usa ceil(x)-1 para los límites superiores desde el principio — la
revisión final de ingestion/dem encontró y corrigió este mismo bug
usando floor() para ambos límites; se aplica la lección aquí de entrada.
"""
import math

_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
_TILE_SIZE_DEG = 3


def tile_key(lat: int, lon: int, version: str = "v200", year: str = "2021") -> str:
    ns = "S" if lat < 0 else "N"
    ew = "W" if lon < 0 else "E"
    return f"ESA_WorldCover_10m_{year}_{version}_{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map"


def tile_url(key: str, version: str = "v200", year: str = "2021") -> str:
    return f"{_BASE_URL}/{version}/{year}/map/{key}.tif"


def tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """bbox = (west, south, east, north). Tiles son múltiplos de 3°."""
    west, south, east, north = bbox
    lat_start = math.floor(south / _TILE_SIZE_DEG) * _TILE_SIZE_DEG
    lat_end = (math.ceil(north / _TILE_SIZE_DEG) - 1) * _TILE_SIZE_DEG
    lon_start = math.floor(west / _TILE_SIZE_DEG) * _TILE_SIZE_DEG
    lon_end = (math.ceil(east / _TILE_SIZE_DEG) - 1) * _TILE_SIZE_DEG
    return [
        (lat, lon)
        for lat in range(lat_start, lat_end + 1, _TILE_SIZE_DEG)
        for lon in range(lon_start, lon_end + 1, _TILE_SIZE_DEG)
    ]
```

- [ ] **Step 2: Write the failing tests for `tiles.py`**

`ingestion/tests/test_worldcover_tiles.py`:

```python
"""Tests de la grilla de tiles WorldCover: solo matemática, sin red."""
from ingestion.worldcover.tiles import tile_key, tile_url, tiles_for_bbox


def test_tile_key_matches_real_bucket_naming():
    assert tile_key(-39, -72) == "ESA_WorldCover_10m_2021_v200_S39W072_Map"


def test_tile_key_north_east_hemisphere():
    assert tile_key(0, 6) == "ESA_WorldCover_10m_2021_v200_N00E006_Map"


def test_tile_url_matches_real_bucket_path():
    url = tile_url("ESA_WorldCover_10m_2021_v200_S39W072_Map")
    assert url == (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        "v200/2021/map/ESA_WorldCover_10m_2021_v200_S39W072_Map.tif"
    )


def test_tiles_for_bbox_study_area_uses_3_degree_multiples():
    bbox = (-73.7, -39.3, -71.0, -36.5)
    tiles = tiles_for_bbox(bbox)
    assert (-42, -75) in tiles  # SW-most 3-degree tile
    assert (-39, -72) in tiles  # NE-most 3-degree tile
    for lat, lon in tiles:
        assert lat % 3 == 0
        assert lon % 3 == 0


def test_tiles_for_bbox_excludes_tile_when_edge_lands_exactly_on_its_start():
    # borde exactamente en un múltiplo de 3: el tile que arranca ahí no
    # se solapa (cubre [N, N+3)) — mismo caso que ingestion/dem, corregido
    # de entrada aquí.
    bbox = (-72.0, -39.0, -69.0, -36.0)
    assert tiles_for_bbox(bbox) == [(-39, -72)]


def test_tiles_for_bbox_single_tile():
    bbox = (-72.9, -38.9, -72.1, -38.1)
    assert tiles_for_bbox(bbox) == [(-39, -75)]
```

- [ ] **Step 2b: Run tests, verify they fail**

Run: `uv run --package ingestion pytest ingestion/tests/test_worldcover_tiles.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 2c: Run tests again after Step 1's file exists, verify they pass**

Run: `uv run --package ingestion pytest ingestion/tests/test_worldcover_tiles.py -v`
Expected: 6 passed

If `test_tiles_for_bbox_single_tile`'s expected tile doesn't match (double-check by hand: `west=-72.9 -> floor(-72.9/3)*3 = floor(-24.3)*3 = -25*3 = -75`; `east=-72.1 -> (ceil(-72.1/3)-1)*3 = (ceil(-24.033)-1)*3 = (-24-1)*3 = -75`; `south=-38.9 -> floor(-38.9/3)*3 = floor(-12.966)*3 = -13*3 = -39`; `north=-38.1 -> (ceil(-38.1/3)-1)*3 = (ceil(-12.7)-1)*3 = (-12-1)*3 = -39`; single tile `(-39,-75)` — matches), do not adjust the expected value without re-deriving it by hand first; a mismatch here is much more likely to be a real bug than in the DEM plan's 1°-grid case, since this arithmetic has an extra `* 3` step that's easy to place wrong.

- [ ] **Step 3: Write `ingestion/src/ingestion/worldcover/client.py`**

```python
"""Descarga de un tile de ESA WorldCover desde el bucket público de AWS.

Mismo patrón que ingestion/dem/client.py: bucket público sobre HTTPS
plano, sin retry/backoff (no hay límite de tasa documentado para
archivos estáticos), escritura atómica (.part + os.replace) desde el
principio — la revisión final de ingestion/dem encontró y corrigió la
ausencia de esto; se aplica la lección aquí de entrada.
"""
import os
from pathlib import Path

import requests
from ingestion.worldcover.tiles import tile_url


class WorldCoverDownloadError(RuntimeError):
    """Fallo al descargar un tile de ESA WorldCover."""


def download_tile(key: str, dest_path: Path, session: requests.Session | None = None) -> Path:
    if dest_path.exists():
        return dest_path

    sess = session or requests.Session()
    response = sess.get(tile_url(key), timeout=60)
    if response.status_code != 200:
        raise WorldCoverDownloadError(
            f"Error {response.status_code} descargando tile {key}: {tile_url(key)}"
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    tmp_path.write_bytes(response.content)
    os.replace(tmp_path, dest_path)
    return dest_path
```

- [ ] **Step 4: Write the failing tests for `client.py`**

`ingestion/tests/test_worldcover_client.py`:

```python
"""Tests del cliente de descarga WorldCover: cero red real (usa `responses`)."""
import pytest
import responses
from ingestion.worldcover.client import WorldCoverDownloadError, download_tile
from ingestion.worldcover.tiles import tile_url

KEY = "ESA_WorldCover_10m_2021_v200_S39W072_Map"


@responses.activate
def test_download_tile_writes_bytes_to_dest_path(tmp_path):
    responses.add(responses.GET, tile_url(KEY), body=b"fake-tif-bytes", status=200)
    dest = tmp_path / f"{KEY}.tif"
    result = download_tile(KEY, dest)
    assert result == dest
    assert dest.read_bytes() == b"fake-tif-bytes"


@responses.activate
def test_download_tile_raises_on_404(tmp_path):
    responses.add(responses.GET, tile_url(KEY), status=404)
    with pytest.raises(WorldCoverDownloadError, match="404"):
        download_tile(KEY, tmp_path / f"{KEY}.tif")


@responses.activate
def test_download_tile_skips_request_when_dest_already_exists(tmp_path):
    dest = tmp_path / f"{KEY}.tif"
    dest.write_bytes(b"already-here")
    result = download_tile(KEY, dest)  # sin response registrado -> probaría red real si se llamara
    assert result == dest
    assert dest.read_bytes() == b"already-here"


@responses.activate
def test_download_tile_leaves_no_stray_part_file_after_success(tmp_path):
    responses.add(responses.GET, tile_url(KEY), body=b"fake-tif-bytes", status=200)
    dest = tmp_path / f"{KEY}.tif"
    download_tile(KEY, dest)
    assert not dest.with_suffix(dest.suffix + ".part").exists()
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run --package ingestion pytest ingestion/tests/test_worldcover_client.py -v`
Expected: 4 passed

- [ ] **Step 6: Write `ingestion/src/ingestion/worldcover/cache.py`**

```python
"""Clave de cache para un mosaico WorldCover procesado: hash de (bbox,
resolución, CRS, versión)."""
import hashlib


def cache_key_for(
    bbox: tuple[float, float, float, float], resolution_m: int, crs: str, version: str
) -> str:
    payload = f"{bbox}|{resolution_m}|{crs}|{version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 7: Write `ingestion/src/ingestion/worldcover/fuel_type.py`**

```python
"""Mapeo heurístico de clase de cobertura ESA WorldCover a "tipo de
combustible" simplificado — usado por el autómata celular y el U-Net
como proxy grosero de carga/tipo de combustible, NO un mapa de
combustibles forestales validado en terreno (p. ej. no distingue los
sistemas Fireline/Behave/Scott-Burgan).

*** LIMITACIÓN EXPLÍCITA: WorldCover no distingue bosque nativo de
plantación forestal (ambos caen en la clase 10 "Tree cover") — una
distinción crítica para el comportamiento del fuego en Biobío/Ñuble/
Araucanía (plantaciones de Pinus radiata y Eucalyptus se comportan de
forma muy distinta a bosque nativo de Nothofagus). Separar ambos
requeriría una fuente adicional (p. ej. catastro de CONAF), no
integrada en este proyecto. ***
"""
import numpy as np

FUEL_PASTIZAL = 1
FUEL_MATORRAL = 2
FUEL_BOSQUE = 3
FUEL_CULTIVO = 4
FUEL_HUMEDAL = 5
FUEL_URBANO_NO_COMBUSTIBLE = 90
FUEL_SUELO_DESNUDO_NO_COMBUSTIBLE = 91
FUEL_AGUA_NO_COMBUSTIBLE = 92
FUEL_NIEVE_HIELO_NO_COMBUSTIBLE = 93
FUEL_TYPE_UNKNOWN = 99

FUEL_TYPE_LABELS: dict[int, str] = {
    FUEL_PASTIZAL: "pastizal",
    FUEL_MATORRAL: "matorral",
    FUEL_BOSQUE: "bosque (nativo o plantación — no distinguible)",
    FUEL_CULTIVO: "cultivo",
    FUEL_HUMEDAL: "humedal",
    FUEL_URBANO_NO_COMBUSTIBLE: "urbano/no combustible",
    FUEL_SUELO_DESNUDO_NO_COMBUSTIBLE: "suelo desnudo/no combustible",
    FUEL_AGUA_NO_COMBUSTIBLE: "agua/no combustible",
    FUEL_NIEVE_HIELO_NO_COMBUSTIBLE: "nieve o hielo/no combustible",
    FUEL_TYPE_UNKNOWN: "desconocido (clase WorldCover no mapeada)",
}

# Las 11 clases documentadas de ESA WorldCover 2021 v200, verificadas
# contra https://collections.sentinel-hub.com/worldcover/readme.html
CLASS_TO_FUEL_TYPE: dict[int, int] = {
    10: FUEL_BOSQUE,                        # Tree cover
    20: FUEL_MATORRAL,                       # Shrubland
    30: FUEL_PASTIZAL,                       # Grassland
    40: FUEL_CULTIVO,                        # Cropland
    50: FUEL_URBANO_NO_COMBUSTIBLE,           # Built-up
    60: FUEL_SUELO_DESNUDO_NO_COMBUSTIBLE,    # Bare / sparse vegetation
    70: FUEL_NIEVE_HIELO_NO_COMBUSTIBLE,      # Snow and ice
    80: FUEL_AGUA_NO_COMBUSTIBLE,             # Permanent water bodies
    90: FUEL_HUMEDAL,                         # Herbaceous wetland
    95: FUEL_HUMEDAL,                         # Mangroves
    100: FUEL_NIEVE_HIELO_NO_COMBUSTIBLE,     # Moss and lichen (proxy: vegetación rala no combustible)
}


def map_worldcover_to_fuel_type(classes: np.ndarray) -> np.ndarray:
    result = np.full(classes.shape, FUEL_TYPE_UNKNOWN, dtype="int32")
    for wc_class, fuel_type in CLASS_TO_FUEL_TYPE.items():
        result[classes == wc_class] = fuel_type
    return result
```

- [ ] **Step 8: Write the failing tests for `fuel_type.py`**

`ingestion/tests/test_worldcover_fuel_type.py`:

```python
"""Tests del mapeo WorldCover -> tipo de combustible."""
import numpy as np
from ingestion.worldcover.fuel_type import (
    CLASS_TO_FUEL_TYPE,
    FUEL_BOSQUE,
    FUEL_PASTIZAL,
    FUEL_TYPE_UNKNOWN,
    map_worldcover_to_fuel_type,
)


def test_all_11_documented_classes_are_mapped():
    assert set(CLASS_TO_FUEL_TYPE.keys()) == {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}


def test_map_worldcover_to_fuel_type_known_classes():
    classes = np.array([[10, 30]])
    result = map_worldcover_to_fuel_type(classes)
    assert result[0, 0] == FUEL_BOSQUE
    assert result[0, 1] == FUEL_PASTIZAL


def test_map_worldcover_to_fuel_type_unmapped_class_is_explicit_unknown():
    # p. ej. un código corrupto o de una versión futura de WorldCover
    classes = np.array([[255]])
    result = map_worldcover_to_fuel_type(classes)
    assert result[0, 0] == FUEL_TYPE_UNKNOWN
```

- [ ] **Step 9: Run tests, verify they pass**

Run: `uv run --package ingestion pytest ingestion/tests/test_worldcover_fuel_type.py -v`
Expected: 3 passed

- [ ] **Step 10: Write `ingestion/src/ingestion/worldcover/pipeline.py`**

```python
"""Orquesta: cache -> descarga de tiles WorldCover faltantes -> mosaico ->
reproyección con remuestreo NEAREST (categórico — nunca bilineal)."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
from ingestion.worldcover.cache import cache_key_for
from ingestion.worldcover.client import WorldCoverDownloadError, download_tile
from ingestion.worldcover.tiles import tile_key, tiles_for_bbox
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

_DEFAULT_NODATA = 0.0  # WorldCover ya declara nodata=0 en sus tiles reales;
# se fuerza explícitamente igual que en ingestion/dem, por si un tile
# individual (o una versión futura) no lo trajera.


def build_worldcover(
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    crs: str,
    raw_tiles_dir: Path,
    cache_dir: Path,
    version: str = "v200",
    year: str = "2021",
    download_fn: Callable[[str, Path], Path] = download_tile,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"worldcover_{cache_key_for(bbox, resolution_m, crs, version)}.tif"
    if cache_path.exists():
        return cache_path

    raw_tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    requested_tiles = tiles_for_bbox(bbox)
    for lat, lon in requested_tiles:
        key = tile_key(lat, lon, version=version, year=year)
        dest = raw_tiles_dir / f"{key}.tif"
        try:
            tile_paths.append(download_fn(key, dest))
        except WorldCoverDownloadError:
            continue

    if not tile_paths:
        raise WorldCoverDownloadError(
            f"Ninguno de los {len(requested_tiles)} tile(s) requeridos para "
            f"este bbox pudo descargarse."
        )

    sources = [rasterio.open(p) for p in tile_paths]
    try:
        src_nodata = sources[0].nodata
        effective_nodata = src_nodata if src_nodata is not None else _DEFAULT_NODATA
        mosaic_array, mosaic_transform = merge(sources, nodata=effective_nodata)
        src_crs = sources[0].crs
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

    dst_array = np.full(
        (mosaic_array.shape[0], dst_height, dst_width), effective_nodata, dtype=mosaic_array.dtype
    )
    reproject(
        source=mosaic_array,
        destination=dst_array,
        src_transform=mosaic_transform,
        src_crs=src_crs,
        src_nodata=effective_nodata,
        dst_transform=dst_transform,
        dst_crs=crs,
        dst_nodata=effective_nodata,
        # CATEGÓRICO: nunca bilineal. Interpolar códigos de clase
        # fabricaría clases inexistentes (p. ej. Tree cover=10 mezclado
        # con Water=80 daría 45, que no es ninguna clase real).
        resampling=Resampling.nearest,
    )

    with rasterio.open(
        cache_path, "w", driver="GTiff",
        height=dst_height, width=dst_width, count=dst_array.shape[0],
        dtype=dst_array.dtype, crs=crs, transform=dst_transform, nodata=effective_nodata,
    ) as dst:
        dst.write(dst_array)

    return cache_path
```

- [ ] **Step 11: Write the failing pipeline tests**

`ingestion/tests/test_worldcover_pipeline.py`:

```python
"""Tests del pipeline WorldCover: mosaico + reproyección NEAREST + cache."""
from pathlib import Path

import numpy as np
import rasterio
from ingestion.worldcover.pipeline import build_worldcover
from rasterio.transform import from_origin

BBOX = (-72.9, -38.9, -72.1, -38.1)  # single tile: (-39, -75)


def _write_synthetic_tile(path: Path, sw_lat: int, sw_lon: int) -> None:
    """Tile sintético 4x4 en EPSG:4326, 3 grados de lado, con dos clases
    WorldCover reales en bloques (10=Tree cover, 80=Water) para poder
    verificar que el remuestreo nearest no fabrica clases intermedias."""
    size = 4
    transform = from_origin(sw_lon, sw_lat + 3, 0.75, 0.75)
    data = np.full((size, size), 10, dtype="uint8")
    data[:, size // 2 :] = 80  # mitad este = agua
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0.0,
    ) as dst:
        dst.write(data, 1)


def test_build_worldcover_nearest_resample_never_invents_classes(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"

    def fake_download(key: str, dest_path: Path) -> Path:
        if not dest_path.exists():
            _write_synthetic_tile(dest_path, sw_lat=-39, sw_lon=-75)
        return dest_path

    result_path = build_worldcover(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    with rasterio.open(result_path) as ds:
        assert ds.crs.to_string() == "EPSG:32719"
        arr = ds.read(1)
    present_values = set(np.unique(arr))
    # Solo deben aparecer los valores de origen (10, 80) y el nodata
    # forzado — nunca un valor fabricado por interpolación entre ambos.
    assert present_values <= {10, 80, 0}


def test_build_worldcover_cache_hit_skips_download_entirely(tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    calls = {"n": 0}

    def fake_download(key: str, dest_path: Path) -> Path:
        calls["n"] += 1
        if not dest_path.exists():
            _write_synthetic_tile(dest_path, sw_lat=-39, sw_lon=-75)
        return dest_path

    first = build_worldcover(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    calls_after_first = calls["n"]
    second = build_worldcover(
        bbox=BBOX, resolution_m=250, crs="EPSG:32719",
        raw_tiles_dir=raw_dir, cache_dir=cache_dir, download_fn=fake_download,
    )
    assert second == first
    assert calls["n"] == calls_after_first
```

- [ ] **Step 12: Run tests, verify they pass**

Run: `uv run --package ingestion pytest ingestion/tests/test_worldcover_pipeline.py -v`
Expected: 2 passed

- [ ] **Step 13: Run the full `ingestion` suite together**

Run: `uv run --package ingestion pytest ingestion/tests -v`
Expected: all pass

- [ ] **Step 14: Commit**

```bash
git add ingestion/src/ingestion/worldcover ingestion/tests/test_worldcover_tiles.py ingestion/tests/test_worldcover_client.py ingestion/tests/test_worldcover_pipeline.py ingestion/tests/test_worldcover_fuel_type.py
git commit -m "feat(ingestion): add WorldCover tile download, nearest-resample mosaic, fuel-type mapping"
```

---

### Task 5: `mypy`/`ruff` on `ingestion/sentinel2`/`ingestion/worldcover`, homogeneous `docs/data-sources.md`, closing `docs/decisions.md`

**Files:**
- Rewrite: `docs/data-sources.md` (all 4 sources in one homogeneous format)
- Modify: `docs/decisions.md` (openEO choice + P1-P4 reprojection/resampling closing summary)
- Possibly modify: any `ingestion/sentinel2/*.py`/`ingestion/worldcover/*.py` needed for `mypy --strict`

- [ ] **Step 1: Run `mypy --strict` and `ruff` on both new modules and the whole repo**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run mypy --strict ingestion/src/ingestion/sentinel2 ingestion/src/ingestion/worldcover`
Run: `uv run ruff check .`

Fix whatever surfaces (likely candidate: `openeo` ships no/partial type stubs — a narrow `# type: ignore[import-untyped]` on the `import openeo` line, only if mypy actually flags it). Do not weaken the check to pass it.

- [ ] **Step 2: Rewrite `docs/data-sources.md` with all 4 sources in one homogeneous format**

Each source gets exactly these subsections, in this order: **Qué entrega**, **Resolución nativa**, **Cómo se obtiene** (credencial/registro), **Cómo se organiza / cacheo**, **Reproyección y remuestreo**, **Limitaciones conocidas**. Preserve every fact already written for FIRMS/DEM/ERA5-Land verbatim (do not re-verify or soften any of it — this is a reformat for homogeneity, not a rewrite of content) and add Sentinel-2/WorldCover in the same shape:

```markdown
# Fuentes de datos

## NASA FIRMS

**Qué entrega:** detecciones activas de fuego casi en tiempo real
(VIIRS 375 m por defecto en este proyecto). El Area API también acepta
MODIS 1 km vía `--sensor` (`MODIS_NRT`/`MODIS_SP`); LANDSAT
(`LANDSAT_NRT`) **es exclusivo de EE.UU./Canadá** y no sirve para Chile
pese a estar en la lista de `SOURCE` válidos. Solo los sensores
`VIIRS_*` están probados end-to-end en este proyecto — `MODIS_*` usa
una escala de `confidence` numérica distinta que no se ha ejercitado
con datos reales.

**Resolución nativa:** 375 m (VIIRS).

**Cómo se obtiene:**
1. Ir a https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Registrar un correo — el MAP_KEY llega por email.
3. Ponerlo en `.env` como `FIRMS_MAP_KEY=...` (ver `.env.example`).

**Cómo se organiza / cacheo:** `DAY_RANGE` máximo por consulta: 5 días
(`pyrocast-ingest firms` divide rangos más largos automáticamente).
Límite de uso: 5000 transacciones / 10 min por MAP_KEY. Persistencia
cruda en Parquet particionado por fecha de descarga
(`ingestion/firms/storage.py`); no hay cache de "no repetir la misma
consulta" (cada consulta es un rango de fechas distinto por diseño).

**Reproyección y remuestreo:** ninguno — los puntos de detección se
normalizan a `shared.schemas.FireDetection` (lat/lon en WGS84) sin
reproyectar; la reproyección a la grilla de trabajo es tarea de
`features/fire_state/` (aún no implementado).

**Limitaciones conocidas:**
- Resolución 375 m — no resuelve ignición puntual con más precisión.
- Falsos positivos por reflejo solar ("sun glint") sobre agua.
- Falsos negativos bajo cobertura de nubes/humo denso.
- Frecuencia de paso limitada (~1-4 pases/día).
- `confidence` no es comparable entre sensores (categórico en VIIRS,
  numérico en MODIS) — se guarda tal cual, sin unificar escala.
- El comportamiento documentado ante errores (MAP_KEY inválido) es
  débil — el cliente trata 429/5xx y errores de transporte como
  reintentables con backoff, y cualquier 200 que no sea CSV real como
  error no reintentable.

## Copernicus DEM GLO-30

**Qué entrega:** modelo de elevación digital global, usado para
calcular pendiente y orientación (`features/terrain/slope_aspect.py`,
método de Horn 1981 — mismo algoritmo que GDAL `gdaldem` y ESRI):

```
dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize_x)
dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize_y)
slope_deg  = grados(atan(hipot(dz/dx, dz/dy)))          # [0, 90]
aspect_deg = (grados(atan2(-dz/dx, dz/dy))) mod 360      # [0, 360), -1 si es plana
```
(ventana 3x3 `a b c / d e f / g h i`; pendiente en grados, orientación
en grados de rumbo horario desde el norte).

**Resolución nativa:** ~30 m (1 arco-segundo).

**Cómo se obtiene:** bucket público de AWS
(`s3://copernicus-dem-30m`), sin credenciales — HTTPS plano. Elegido
sobre la API de OpenTopography (exige API key adicional, límites de
tasa no documentados públicamente); ver `docs/decisions.md`.

**Cómo se organiza / cacheo:** tiles de 1°x1°, nombrados por esquina
suroeste (semiabierto: `[lat,lat+1) x [lon,lon+1)`). Un tile faltante
(hueco de GLO-30 Public, u oceánico) no aborta el mosaico completo — se
tolera y queda marcado con nodata. Resultado final (mosaico +
reproyección) cacheado por hash de `(bbox, resolución, CRS)`; tiles
crudos cacheados por su nombre (reutilizables entre bboxes).

**Reproyección y remuestreo:** reproyectado a `EPSG:32719` (UTM 19S) en
la resolución de `shared/config.py` (250 m por defecto) con remuestreo
**bilineal** (nunca nearest — produciría escalones artificiales en una
magnitud continua como elevación).

**Limitaciones conocidas:**
- GLO-30 Public tiene huecos de cobertura; tiles oceánicos genuinamente
  no existen (verificado: varios 404 reales cerca del área de estudio).
- Los tiles reales declaran `nodata=None` — se fuerza un nodata propio
  (`-32767.0` DEM, `-9999.0` pendiente/orientación) en todo el pipeline;
  sin esto, huecos se rellenarían con `0.0` sin marcar (terreno
  fabricado a nivel del mar).
- Celdas vecinas a un hueco de datos no son confiables (el kernel de
  Horn 3x3 sigue usando el hueco).
- Salida no recortada al bbox exacto ni anclada a una grilla canónica
  compartida entre fuentes — diferido a `features/grid/` (sin
  implementar).
- Resolución nativa ~30 m, remuestreada a 250 m — se pierde detalle de
  microrelieve.

## ERA5-Land (Copernicus CDS)

**Qué entrega:** reanálisis de viento, temperatura, humedad y
precipitación, agregado a diario por este proyecto (media para
temperatura/punto de rocío/viento, suma para precipitación —
`ingestion/era5/aggregate.py`). Se pide el dataset horario crudo
(`reanalysis-era5-land`), no el derivado de estadísticas diarias de
CDS, porque este último excluye variables acumuladas (incluida
precipitación total).

**Resolución nativa:** ~9 km.

**Cómo se obtiene:**
1. Crear cuenta en https://cds.climate.copernicus.eu/
2. Copiar el "Personal Access Token" del perfil.
3. Ponerlo en `.env` como `CDS_API_KEY=...` y
   `CDS_API_URL=https://cds.climate.copernicus.eu/api`. Este proyecto
   pasa `url`/`key` directo al constructor de `cdsapi.Client` — nunca
   escribe `~/.cdsapirc`.

**Cómo se organiza / cacheo:** las solicitudes a CDS son asíncronas
(encoladas) — `ingestion/era5/client.py` implementa su propio polling
con timeout configurable (por defecto 1 hora); `cdsapi` en su modo por
defecto no tiene límite de espera total (verificado en su código
fuente). Puede tardar minutos u horas según la carga del servicio.
Cacheo por hash de `(rango de fechas, variables solicitadas)` — el bbox
no forma parte de la clave (se asume el bbox de estudio fijo del
proyecto).

**Reproyección y remuestreo:** reproyectado a `EPSG:32719` en la
resolución de trabajo con remuestreo **bilineal** (magnitudes
continuas). `features/weather/derive.py` también deriva velocidad/
dirección del viento desde u/v y humedad relativa aproximada desde
temperatura/punto de rocío (Magnus-Tetens, coeficientes de Alduchov &
Eskridge 1996 — válida -40°C a 50°C, error máximo documentado ±0.4%
RH).

**Limitaciones conocidas:**
- **Downscaling por interpolación, no física**: ~9 km a 250 m es
  puramente geométrico — no introduce detalle real de sub-grilla.
- Humedad relativa es una aproximación, no una medición real.
- La semántica exacta de "DATE = primer día del rango" en el Area
  API de FIRMS y el comportamiento de `year`/`month`/`day` de CDS están
  verificados contra documentación, no contra una llamada real
  autenticada (este entorno no tiene credenciales reales) — ver
  `docs/decisions.md`.

## Sentinel-2 L2A (Copernicus Data Space Ecosystem, vía openEO)

**Qué entrega:** composición mensual de menor nubosidad (mediana
temporal tras enmascarar nubes por SCL) de las bandas B04 (rojo), B08
(NIR) y SCL (Scene Classification), usada para calcular NDVI —
`features/vegetation/ndvi.py`:

```
NDVI = (NIR - RED) / (NIR + RED)          # adimensional, [-1, 1]
```

**NDVI es un proxy del estado/vigor de la vegetación (verdor,
actividad fotosintética), NO una medición directa de humedad de
combustible** — vegetación con NDVI alto puede tener bajo contenido de
humedad real si está senescente o bajo estrés hídrico no visible en el
verdor foliar.

**Resolución nativa:** 10 m (B04/B08), 20 m (SCL, remuestreada a 10 m
por el propio proceso de openEO al combinar bandas).

**Cómo se obtiene:**
1. Crear cuenta en https://dataspace.copernicus.eu/
2. Registrar un cliente OAuth en el dashboard de Sentinel Hub Services
   (ver enlace desde el perfil de Copernicus Data Space Ecosystem).
3. Poner `COPERNICUS_DATASPACE_CLIENT_ID`/`COPERNICUS_DATASPACE_CLIENT_SECRET`
   en `.env` (ver `.env.example`) — mismas variables que ya existían en
   `shared/config.py` desde el bootstrap del proyecto.

**Cómo se organiza / cacheo:** cliente `openeo` (elegido sobre
`sentinelhub-py` — ver `docs/decisions.md`), autenticado por client
credentials. Enmascarado de nubes con SCL, clases `{3,8,9,10}` (sombra
de nube, nube prob. media/alta, cirros delgados), aplicado tanto en el
proceso openEO (server-side) como localmente en
`features/vegetation/ndvi.py` (defensa en profundidad, y la única forma
de testear el enmascarado sin mockear todo el grafo de openEO).
Cacheado por hash de `(bbox, año, mes)`.

**Reproyección y remuestreo:** NDVI reproyectado a `EPSG:32719` en la
resolución de trabajo con remuestreo **bilineal** (magnitud continua,
igual que DEM y ERA5-Land).

**Limitaciones conocidas:**
- El enmascarado de nubes por SCL no es perfecto — nubes delgadas,
  sombras difusas o bordes de nube pueden no clasificarse correctamente
  en el producto L2A de origen.
- Una composición mensual por mediana puede seguir mostrando artefactos
  si un mes completo tiene cobertura de nubes muy alta (pocas o ninguna
  observación clara) — no hay una verificación automática de "cobertura
  mínima de píxeles válidos" en este bootstrap.
- No verificado contra una llamada real autenticada a Copernicus Data
  Space Ecosystem en este entorno (sin credenciales reales disponibles)
  — el grafo openEO y el mockeo de `openeo.Connection` están verificados
  contra la documentación y tests unitarios, no contra un pedido real.

## ESA WorldCover

**Qué entrega:** mapa de cobertura de suelo global, usado como proxy de
tipo de combustible mediante una tabla de mapeo heurística
(`ingestion/worldcover/fuel_type.py`) — pastizal, matorral, bosque,
cultivo, humedal, y clases no combustibles (urbano, agua, suelo
desnudo, nieve/hielo).

**Resolución nativa:** 10 m.

**Cómo se obtiene:** bucket público de AWS
(`s3://esa-worldcover`), sin credenciales — HTTPS plano, mismo patrón
que Copernicus DEM.

**Cómo se organiza / cacheo:** tiles de 3°x3°, nombrados por esquina
suroeste, mismo esquema semiabierto que Copernicus DEM. Resultado final
cacheado por hash de `(bbox, resolución, CRS, versión)`.

**Reproyección y remuestreo:** reproyectado a `EPSG:32719` con
remuestreo **nearest (nunca bilineal)** — los valores son códigos de
clase categóricos; interpolar produciría clases inexistentes (p. ej.
promediar Tree cover=10 con Water=80 daría 45, que no es ninguna clase
real).

**Limitaciones conocidas:**
- **WorldCover no distingue bosque nativo de plantación forestal**
  (ambos caen en la clase 10 "Tree cover") — una distinción crítica
  para el comportamiento del fuego en la zona de estudio (plantaciones
  de Pinus/Eucalyptus vs. bosque nativo de Nothofagus). Separarlos
  requeriría una fuente adicional (p. ej. catastro de CONAF), no
  integrada.
- La tabla de mapeo a tipo de combustible es una simplificación
  heurística de una persona, no un sistema de combustibles validado en
  terreno (Fireline/Behave/Scott-Burgan) — ver el docstring de
  `fuel_type.py`.
- Producto de un único año (2021); no captura cambios de uso de suelo
  posteriores (p. ej. cosecha de plantaciones, incendios previos que ya
  cambiaron la cobertura).
```

- [ ] **Step 3: Append the closing decisions to `docs/decisions.md`**

```markdown

## Sentinel-2: openEO en vez de sentinelhub-py

Ambos acceden a Copernicus Data Space Ecosystem. Se eligió `openeo`
porque su autenticación por client credentials
(`authenticate_oidc_client_credentials(client_id, client_secret)`) usa
exactamente los dos campos que `shared/config.py` ya tenía desde el
bootstrap (`copernicus_dataspace_client_id`/`_client_secret`) — cero
variables de entorno o campos de configuración nuevos. `sentinelhub-py`
necesita además `sh_base_url` y `sh_token_url` (verificado en su propia
documentación de configuración), que habrían requerido ampliar
`shared/config.py`. La API de alto nivel de openEO (`load_collection` +
`.mask()` + `.reduce_dimension()` + `.download()`) también expresa el
flujo pedido (cargar, enmascarar nubes, componer, descargar) sin
necesidad de escribir un evalscript custom, que sí sería necesario con
la Process API de Sentinel Hub.

## Cierre de Etapa 1 (P1-P4): decisiones de reproyección y remuestreo

Resumen de las decisiones tomadas en los cuatro módulos de ingesta
implementados hasta ahora (FIRMS, DEM, ERA5-Land, Sentinel-2,
WorldCover):

| Fuente | CRS destino | Resolución | Remuestreo | Por qué |
|---|---|---|---|---|
| Copernicus DEM | EPSG:32719 | 250 m | Bilineal | Elevación es continua; nearest produce escalones artificiales. |
| ERA5-Land | EPSG:32719 | 250 m | Bilineal | Viento/temperatura/humedad/precipitación son continuos. |
| Sentinel-2 (NDVI) | EPSG:32719 | 250 m | Bilineal | NDVI es una magnitud continua derivada de reflectancia. |
| ESA WorldCover | EPSG:32719 | 250 m | **Nearest** | Códigos de clase categóricos — interpolar fabricaría clases inexistentes. |
| NASA FIRMS | (sin reproyectar) | — | — | Puntos de detección en WGS84; la reproyección a grilla es tarea de `features/fire_state/` (pendiente). |

Decisión transversal: **todas** las fuentes rasterizadas comparten el
mismo CRS de destino (`EPSG:32719`, UTM 19S) y la misma resolución
nominal (250 m, configurable en `shared/config.py`), pero **cada una
reproyecta de forma independiente** — no hay todavía una grilla
canónica compartida que garantice alineación píxel-a-píxel exacta
entre capas (mismo origen, mismo ancho/alto). Esto es una limitación
conocida y diferida a `features/grid/` (todavía sin implementar, ver
`docs/limitations.md`): hasta que exista, dos capas de este proyecto
con el mismo CRS/resolución nominal pueden tener orígenes de píxel
ligeramente distintos, y superponerlas exactamente requiere un
remuestreo adicional de alineación en el consumidor (p. ej.
`features/dataset/`, también pendiente).
```

- [ ] **Step 4: Full workspace re-verification**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package ingestion pytest ingestion/tests -v`
Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package features pytest features/tests -v`
Run: `uv run ruff check .`
Run: `uv run mypy --strict shared/src features/src`
Run: `uv run mypy --strict ingestion/src/ingestion/sentinel2 ingestion/src/ingestion/worldcover`

Expected: all green, matching this task's acceptance criteria verbatim, zero env vars.

- [ ] **Step 5: Commit**

```bash
git add docs/data-sources.md docs/decisions.md
git commit -m "docs: homogenize data-sources.md across 4 sources, close Stage 1 reprojection decisions"
```

---

### Task 6: CLI + Makefile wiring for `sentinel2` and `worldcover`

**Files:**
- Create: `ingestion/src/ingestion/sentinel2/cli.py`
- Create: `ingestion/src/ingestion/worldcover/cli.py`
- Modify: `ingestion/src/ingestion/cli.py` (register both commands)
- Modify: `Makefile` (`ingest-vegetation` target)
- Test: `ingestion/tests/test_sentinel2_cli.py`, `ingestion/tests/test_worldcover_cli.py`

**Interfaces:**
- Consumes: `fetch_sentinel2` (Task 2), `compute_and_save_vegetation` (Task 3), `build_worldcover` + `map_worldcover_to_fuel_type` (Task 4), `shared.config.get_settings`.

- [ ] **Step 1: Write `ingestion/src/ingestion/sentinel2/cli.py`**

```python
"""Comando `sentinel2` del CLI de ingesta: composición mensual + NDVI."""
import typer
from features.vegetation.ndvi import compute_and_save_vegetation
from shared.config import get_settings

from ingestion.sentinel2.client import Sentinel2Client
from ingestion.sentinel2.pipeline import fetch_sentinel2


def sentinel2(
    year: int = typer.Option(..., help="Año (YYYY)"),
    month: int = typer.Option(..., min=1, max=12, help="Mes (1-12)"),
) -> None:
    """Descarga una composición mensual de menor nubosidad de Sentinel-2
    L2A y calcula NDVI reproyectado a la grilla de trabajo."""
    settings = get_settings()
    client = Sentinel2Client(
        client_id=settings.copernicus_dataspace_client_id,
        client_secret=settings.copernicus_dataspace_client_secret,
    )
    composite_path = fetch_sentinel2(
        bbox=settings.study_area_bbox,
        year=year,
        month=month,
        client=client,
        cache_dir=settings.data_raw_dir / "sentinel2",
    )
    typer.echo(f"Composición Sentinel-2: {composite_path}")

    ndvi_path = compute_and_save_vegetation(
        composite_path,
        settings.data_processed_dir / "vegetation",
        target_crs=settings.crs,
        target_resolution_m=settings.spatial_resolution_m,
    )
    typer.echo(f"NDVI: {ndvi_path}")
```

- [ ] **Step 2: Write `ingestion/src/ingestion/worldcover/cli.py`**

```python
"""Comando `worldcover` del CLI de ingesta: descarga+mosaico+tipo de
combustible."""
import typer
from shared.config import get_settings

from ingestion.worldcover.fuel_type import map_worldcover_to_fuel_type
from ingestion.worldcover.pipeline import build_worldcover


def worldcover() -> None:
    """Descarga ESA WorldCover para el bbox de estudio, lo mosaica y
    reproyecta (nearest), y calcula el tipo de combustible simplificado."""
    import rasterio

    settings = get_settings()
    worldcover_path = build_worldcover(
        bbox=settings.study_area_bbox,
        resolution_m=settings.spatial_resolution_m,
        crs=settings.crs,
        raw_tiles_dir=settings.data_raw_dir / "worldcover",
        cache_dir=settings.data_processed_dir / "worldcover",
    )
    typer.echo(f"WorldCover: {worldcover_path}")

    with rasterio.open(worldcover_path) as src:
        classes = src.read(1)
        profile = src.profile

    fuel_type = map_worldcover_to_fuel_type(classes)
    output_dir = settings.data_processed_dir / "vegetation"
    output_dir.mkdir(parents=True, exist_ok=True)
    fuel_type_path = output_dir / "fuel_type.tif"
    fuel_profile = {**profile, "dtype": "int32", "nodata": None}
    with rasterio.open(fuel_type_path, "w", **fuel_profile) as dst:
        dst.write(fuel_type.astype("int32"), 1)
    typer.echo(f"Tipo de combustible: {fuel_type_path}")
```

- [ ] **Step 3: Register both commands in `ingestion/src/ingestion/cli.py`**

Add alongside the existing `firms`/`dem` registrations:

```python
from ingestion.sentinel2.cli import sentinel2 as sentinel2_command
from ingestion.worldcover.cli import worldcover as worldcover_command
# ...
app.command("sentinel2")(sentinel2_command)
app.command("worldcover")(worldcover_command)
```

- [ ] **Step 4: Write the CLI tests**

`ingestion/tests/test_sentinel2_cli.py` (monkeypatches `Sentinel2Client`, `fetch_sentinel2`, `compute_and_save_vegetation` at module level, same pattern as `test_dem_cli.py`):

```python
"""Test de humo del CLI de ingesta Sentinel-2: sin red, sin descargas reales."""
from pathlib import Path

from ingestion.cli import app
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}


def test_sentinel2_cli_wires_settings_and_produces_ndvi(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()

    monkeypatch.setattr("ingestion.sentinel2.cli.Sentinel2Client", lambda **kwargs: object())

    def fake_fetch_sentinel2(bbox, year, month, client, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "composite.tif"
        path.write_bytes(b"fake")
        return path

    def fake_compute_and_save_vegetation(composite_path, output_dir, target_crs, target_resolution_m):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ndvi.tif"
        path.write_bytes(b"fake-ndvi")
        return path

    monkeypatch.setattr("ingestion.sentinel2.cli.fetch_sentinel2", fake_fetch_sentinel2)
    monkeypatch.setattr(
        "ingestion.sentinel2.cli.compute_and_save_vegetation", fake_compute_and_save_vegetation
    )

    result = runner.invoke(app, ["sentinel2", "--year", "2026", "--month", "1"])
    assert result.exit_code == 0, result.output
    assert "NDVI" in result.output
    get_settings.cache_clear()
```

`ingestion/tests/test_worldcover_cli.py` (same pattern, monkeypatching `build_worldcover` and using a small real GeoTIFF fixture so `rasterio.open` in the CLI itself has something real to read):

```python
"""Test de humo del CLI de ingesta WorldCover: sin red, sin descargas reales."""
import numpy as np
import rasterio
from ingestion.cli import app
from rasterio.transform import from_origin
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}


def test_worldcover_cli_wires_settings_and_produces_fuel_type(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()

    def fake_build_worldcover(bbox, resolution_m, crs, raw_tiles_dir, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "worldcover.tif"
        transform = from_origin(500000, 5800000, 250, 250)
        data = np.full((4, 4), 10, dtype="uint8")
        with rasterio.open(
            path, "w", driver="GTiff", height=4, width=4, count=1,
            dtype="uint8", crs=crs, transform=transform, nodata=0.0,
        ) as dst:
            dst.write(data, 1)
        return path

    monkeypatch.setattr("ingestion.worldcover.cli.build_worldcover", fake_build_worldcover)

    result = runner.invoke(app, ["worldcover"])
    assert result.exit_code == 0, result.output
    assert "Tipo de combustible" in result.output
    get_settings.cache_clear()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_sentinel2_cli.py ingestion/tests/test_worldcover_cli.py -v`
Expected: 2 passed

- [ ] **Step 6: Wire the Makefile**

```makefile
ingest-vegetation:
	uv run --package ingestion pyrocast-ingest sentinel2 --year $(YEAR) --month $(MONTH)
	uv run --package ingestion pyrocast-ingest worldcover
```

If `YEAR`/`MONTH` aren't set, follow the same usage-message pattern `ingest-firms` already uses (`Makefile` — check for `$(if ...)` / missing-var guard already established there) rather than inventing a new convention.

- [ ] **Step 7: Verify the real installed console script**

Run: `uv run --package ingestion pyrocast-ingest --help` — must list `sentinel2` and `worldcover` alongside `firms`/`dem`.
Run: `uv run --package ingestion pyrocast-ingest sentinel2 --help` and `... worldcover --help`.

- [ ] **Step 8: Full final re-verification and commit**

Run: `env -i PATH="$PATH" HOME="$HOME" make test && make lint && make typecheck && uv run mypy --strict ingestion/src/ingestion/sentinel2 ingestion/src/ingestion/worldcover`

```bash
git add ingestion/src/ingestion/sentinel2/cli.py ingestion/src/ingestion/worldcover/cli.py ingestion/src/ingestion/cli.py Makefile ingestion/tests/test_sentinel2_cli.py ingestion/tests/test_worldcover_cli.py
git commit -m "feat(ingestion): wire sentinel2 and worldcover CLI commands and Makefile target"
```

---

## Self-Review Notes

- **Spec coverage:** user's 5 tasks map to: 1→Task 1+2, 2→Task 3, 3→Task 4, 4→Tasks 2/4 (per-source cache keys), 5→every task's own synthetic-fixture tests (NDVI known value + SCL mask fixture in Task 3, categorical nearest-resample verification in Task 4). Acceptance criteria (mypy/ruff, tests green, homogeneous 4-source docs/data-sources.md, closing docs/decisions.md summary) → Task 5. CLI/Makefile wiring (Task 6) wasn't separately requested by name but is required for the modules to be reachable at all — added proactively per the DEM review's lesson, not left for a future review to catch.
- **Placeholder scan:** every step has real, runnable code. The two "verify by hand before accepting a test failure" notes (Task 4 Step 2c's tile-math arithmetic, Task 3 Step 3's cross-package import fallback) name the exact expected values/fallback, not a vague TODO.
- **Type consistency:** `tile_key`/`tiles_for_bbox` (Task 4) mirror `ingestion.dem.tiles`'s exact shape but at 3° instead of 1°. `Sentinel2Client.fetch_monthly_composite`'s signature (Task 1) matches Task 2's `fetch_sentinel2` call and Task 6's CLI call. `build_worldcover`'s signature mirrors `ingestion.dem.pipeline.build_dem`'s exactly (same parameter names/order), so the CLI wiring pattern transfers directly.
- **Review Focus:** all five items have an owning test — WorldCover tile upper bound (Task 4's `excludes_tile_when_edge_lands_exactly_on_its_start`, ported from the DEM lesson), resampling method per source (Task 4's `nearest_resample_never_invents_classes` for WorldCover, Task 3's bilinear reprojection implicit in `compute_and_save_vegetation`'s own correctness — the DEM plan's "more unique values" bilinear-proof pattern isn't repeated verbatim here since NDVI's continuity is already proven by Task 3 Step 1's exact-value test, and the *inverse* claim — WorldCover nearest *not* inventing values — is the one that needs its own proof), SCL masking correctness (Task 3's `mask_clouds_masks_only_cloud_pixels`, pinned per-pixel not just "some masking happened"), NDVI zero-denominator (Task 3's `handles_zero_denominator_without_raising`), fuel-type unmapped-class handling (Task 4's `unmapped_class_is_explicit_unknown`).
