# Ingestion ERA5-Land Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ingestion/era5/` end to end: a `cdsapi`-based client for ERA5-Land (hourly, native resolution) with our own submit/poll/timeout wrapper around CDS's asynchronous job queue, hourly→daily aggregation, date-range+variable caching, and `features/weather/` deriving wind speed/direction and approximate relative humidity, reprojected/resampled from ERA5-Land's ~9 km grid to the project's 250 m EPSG:32719 grid — all fixture/mock-tested, zero real network or credentials in tests.

**Architecture:** `ingestion/era5/client.py` wraps `cdsapi.Client` (injected via a factory so tests never import the real `cdsapi.Client`), builds a `reanalysis-era5-land` request (hourly, 5 variables, bbox→CDS `[N,W,S,E]` area), and implements our own bounded poll loop (`submit_and_wait`) instead of trusting the library's built-in `wait_until_complete=True` mode — which has **no overall timeout** (verified by reading `cdsapi`'s actual source in this session: its internal `while True` loop only bounds *per-HTTP-request* timeouts via `timeout=`, never the total wait). `ingestion/era5/aggregate.py` opens the downloaded hourly NetCDF with `xarray`, clips it to the exact requested date range (CDS's year/month/day list request shape can return a superset of days across a month boundary — this clip is what keeps the *output* correct regardless), and resamples to daily (mean for temperature/dewpoint/wind, sum for precipitation). `ingestion/era5/cache.py` + `pipeline.py` key the cached daily NetCDF by `(start, end, variables)` and skip the CDS request entirely on a cache hit. `features/weather/derive.py` computes wind speed/direction from u/v and approximate relative humidity from temperature/dewpoint (Magnus-Tetens/Alduchov-Eskridge formula, documented margin of error), then reprojects each 2D field from ERA5-Land's native ~0.1° lat/lon grid to EPSG:32719 at 250 m via `rasterio.warp.reproject` with bilinear resampling — explicitly commented and documented as **interpolation-based downscaling, not physical modeling**.

**Tech Stack:** Python 3.12, `cdsapi` (already CLAUDE.md-approved), `xarray` (already CLAUDE.md-approved, newly declared on `ingestion` alongside its existing `features` declaration), `rasterio`/`numpy` (already declared on both), `h5netcdf` (new — see Global Constraints), `pytest`. No `responses`/`vcrpy` needed here: `cdsapi.Client` is mocked at the Python object level (constructor injection), not at the HTTP layer, since `cdsapi` itself owns all HTTP calls internally and the user's instruction is explicit: "Mockea cdsapi.Client por completo."

**Spec:** User's message in this conversation (5 numbered tasks + acceptance criteria) plus `/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md` (ERA5-Land row: "Viento, temperatura, humedad, precipitación | cdsapi, requiere cuenta gratuita"; the ERA5-Land ~9 km → 250 m downscaling limitation CLAUDE.md already flags as "debe mencionarse en resultados"). Both travel with this plan.

**cdsapi / CDS API — verified against live sources (2026-09-26), not assumed from memory:**
- Docs: https://cds.climate.copernicus.eu/how-to-api, https://github.com/ecmwf/cdsapi (fetched the actual `cdsapi/api.py` source in this session, not just the README).
- Config: `~/.cdsapirc` with `url: https://cds.climate.copernicus.eu/api` and `key: <PERSONAL-ACCESS-TOKEN>` — but `cdsapi.Client(url=..., key=...)` also accepts both directly as constructor kwargs (confirmed in source: falls back to `CDSAPI_URL`/`CDSAPI_KEY` env vars, then the rc file, only if not passed). **This plan passes `url`/`key` directly from `shared.config.Settings` and never writes a `.cdsapirc` file** — simpler, no filesystem side effect, and works identically.
- `Client.__init__` (confirmed from source): `timeout=60` (per-HTTP-request, not overall), `sleep_max=120`, `retry_max=500`, `wait_until_complete=True` (default). When `wait_until_complete=True`, `retrieve()` polls internally with `sleep *= 1.5` capped at `sleep_max`, in a bare `while True` — **there is no parameter that bounds the total wait time**. This plan sets `wait_until_complete=False` and implements its own bounded loop instead, which is what actually satisfies the user's "timeout configurable" requirement.
- With `wait_until_complete=False`, `retrieve(name, request)` (no `target`) returns a job handle. Two real implementations exist depending on which token format a user's CDS account issues (verified via source of both `ecmwf/cdsapi` and `ecmwf/ecmwf-datastores-client`, which `cdsapi.Client.__new__` dispatches to based on whether the key contains a `:`): the classic `cdsapi.api.Result` (`.reply["state"]` ∈ `{queued,running,completed,failed}`, `.update()`, `.download(target)`), or the modern `datastores.Remote` (`.status` property ∈ `{accepted,running,successful,failed,rejected,dismissed,deleted}`, `.update()`, `.download(target)`). **This plan duck-types both** rather than assuming one — see Task 1.
- Dataset choice: `reanalysis-era5-land` (raw hourly), **not** `derived-era5-land-daily-statistics` (CDS's server-side daily aggregation) — because that derived dataset's own documentation states it **omits accumulated variables, explicitly including total precipitation**, which this task requires. Using the raw hourly dataset and aggregating ourselves (mean for instantaneous fields, sum for precipitation) is the only way to get all 5 requested variables as daily aggregates. Documented as a decision in Task 5.
- Request shape (confirmed via a real-world example script and forum threads, since the CDS site's own "API request" code snippet is client-side-rendered and not fetchable statically): `variable` (list of strings — `2m_temperature`, `2m_dewpoint_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `total_precipitation`), `year`/`month`/`day` (lists of zero-padded strings), `time` (list of `"HH:00"` strings), `area` = **`[North, West, South, East]`** (verified — a different axis order than the `west,south,east,north` convention used by the FIRMS/DEM tasks in this same project; easy to get backwards), `data_format: "netcdf"`, `download_format: "unarchived"` (without this, ERA5-Land returns a zip even for a single file — confirmed via an ECMWF forum thread titled exactly about this surprise).

## Global Constraints

- No `.cdsapirc` file is ever written by this code — credentials flow only through `shared.config.Settings` → `cdsapi.Client(url=..., key=...)` constructor kwargs.
- `cdsapi.Client` is never imported or instantiated for real in any test — every test injects a fake client via constructor-parameter dependency injection (the same pattern as `download_fn` in `ingestion/dem/`).
- New dependency: `h5netcdf` (on both `ingestion` and `features`) — `xarray` needs an engine capable of real NetCDF4/HDF5 files (what CDS actually delivers); the only NetCDF backend already resolved in this workspace is `scipy`'s, which is NetCDF3-classic-only and cannot read real ERA5 downloads. `h5netcdf` is a pure-wheel, lighter alternative to the `netCDF4` package (which wraps the full netCDF-C + HDF5 C libraries). Justify in `docs/decisions.md` (Task 5).
- `xarray` is already CLAUDE.md-approved and already declared on `features`; this plan adds it to `ingestion` too (same non-decision as `rasterio`/`numpy` in the DEM plan — a normal per-package dependency declaration, not a new external dependency).
- The CDS area axis order is `[North, West, South, East]` — **not** the `west,south,east,north` order `shared.config.Settings.study_area_bbox` and the FIRMS/DEM code use. The conversion must be explicit and tested, not implicit.
- Wind/humidity derivation and spatial reprojection are two separable concerns — this plan does not conflate them the same way it kept `ingestion/dem`'s downloader/mosaicker/cacher and `features/terrain`'s slope/aspect decoupled.
- The downscaling-by-interpolation limitation (ERA5-Land ~9 km native → 250 m) must be stated in three places, per the user's explicit instruction: a code comment in `features/weather/derive.py`, `docs/limitations.md`, and `docs/data-sources.md`.
- Identifiers in English; docstrings/comments/docs in Spanish (CLAUDE.md convention).

## Review Focus

- **CDS area axis order** (`[N,W,S,E]`, not `[W,S,E,N]`): the single most likely silent-wrong-answer bug in this whole plan — a swapped order would request the wrong geography without any error, since all four values are just numbers to the API. Must be tested against `shared.config.Settings.study_area_bbox`'s actual `(west, south, east, north)` tuple, not a symmetric/degenerate example where a swap wouldn't be detectable.
- **The bounded poll loop's three exit conditions** (success, failure, timeout) must each be independently tested with a fake client that reports the corresponding state sequence — a test that only exercises the success path could hide a bug where a `failed` state is silently treated as "still running" (infinite-loop risk) or where the timeout check is off (e.g. checked before the first poll, so a slow-but-eventually-successful request always times out).
- **Duck-typing both `Result`-shaped (`reply["state"]`) and `Remote`-shaped (`.status` property) fake clients**: a test suite that only ever constructs one shape of fake would not catch a regression that breaks compatibility with the other real implementation — both shapes need their own test.
- **Hourly→daily aggregation must use `mean` for temperature/dewpoint/wind and `sum` for precipitation** — using `mean` for precipitation (an easy copy-paste mistake) would silently produce a daily precipitation *rate* instead of a daily *total*, off by roughly a factor of 24, with no error or warning.
- **The date-range clip before aggregating** (Global Constraints: CDS's month/day list request can overfetch across a month boundary) must actually be tested with a fixture whose hourly NetCDF contains timestamps *outside* the requested `[start, end]` window — a test built only from exactly-in-range timestamps would never exercise the clip and could hide it being a no-op.

---

## File Structure

```
ingestion/
├── pyproject.toml                      # + xarray, h5netcdf
├── src/ingestion/era5/
│   ├── __init__.py                     # (exists, stub)
│   ├── client.py                       # new: Era5Client, submit_and_wait, area conversion
│   ├── aggregate.py                    # new: aggregate_hourly_to_daily
│   ├── cache.py                        # new: cache_key_for
│   └── pipeline.py                     # new: fetch_daily_era5 (orchestration)
└── tests/
    ├── test_era5_client.py             # new
    ├── test_era5_aggregate.py          # new
    └── test_era5_pipeline.py           # new

features/
├── pyproject.toml                      # + h5netcdf
├── src/features/weather/
│   ├── __init__.py                     # (exists, stub)
│   └── derive.py                       # new: wind, RH, compute_and_save_weather
└── tests/
    └── test_weather_derive.py          # new

docs/
├── data-sources.md                     # + ERA5-Land section
├── limitations.md                      # + downscaling entry
└── decisions.md                        # + dataset choice, h5netcdf, polling design
```

---

### Task 1: `ingestion/era5/client.py` — request builder + bounded poll/timeout wrapper

**Files:**
- Modify: `ingestion/pyproject.toml` (+ `xarray`, `h5netcdf`)
- Create: `ingestion/src/ingestion/era5/client.py`
- Test: `ingestion/tests/test_era5_client.py`

**Interfaces:**
- Produces: `ingestion.era5.client.ERA5_VARIABLES` (tuple of the 5 variable names), `ingestion.era5.client.build_request(bbox, start, end, variables=ERA5_VARIABLES) -> dict` (pure function — CDS area conversion + year/month/day/time lists), `ingestion.era5.client.Era5RequestFailedError(RuntimeError)`, `ingestion.era5.client.Era5RequestTimeoutError(RuntimeError)`, `ingestion.era5.client.Era5Client(url: str, key: str, client_factory: Callable[..., Any] = cdsapi.Client)` with method `download_hourly(dataset: str, request: dict, target: Path, poll_interval_seconds: float = 10.0, timeout_seconds: float = 3600.0, sleep_fn: Callable[[float], None] = time.sleep, monotonic_fn: Callable[[], float] = time.monotonic) -> Path`.

- [ ] **Step 1: Add `xarray`/`h5netcdf` to `ingestion/pyproject.toml`**

```toml
dependencies = [
    "shared",
    "requests>=2.32",
    "cdsapi>=0.7",
    "typer>=0.12",
    "pyarrow>=17.0",
    "rasterio>=1.3",
    "numpy>=2.0",
    "xarray>=2024.7",
    "h5netcdf>=1.3",
]
```

- [ ] **Step 2: Write the failing tests**

`ingestion/tests/test_era5_client.py`:

```python
"""Tests del cliente ERA5: cdsapi.Client mockeado por completo, sin red."""
import datetime as dt
from pathlib import Path

import pytest
from ingestion.era5.client import (
    ERA5_VARIABLES,
    Era5Client,
    Era5RequestFailedError,
    Era5RequestTimeoutError,
    build_request,
)

BBOX = (-73.7, -39.3, -71.0, -36.5)  # west, south, east, north


def test_build_request_converts_bbox_to_cds_north_west_south_east_order():
    request = build_request(BBOX, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert request["area"] == [-36.5, -73.7, -39.3, -71.0]  # N, W, S, E


def test_build_request_includes_all_five_variables_by_default():
    request = build_request(BBOX, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert set(request["variable"]) == set(ERA5_VARIABLES)
    assert len(ERA5_VARIABLES) == 5


def test_build_request_uses_unarchived_netcdf_format():
    request = build_request(BBOX, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert request["data_format"] == "netcdf"
    assert request["download_format"] == "unarchived"


def test_build_request_spans_year_month_day_across_a_range():
    request = build_request(BBOX, dt.date(2026, 1, 30), dt.date(2026, 2, 2))
    assert request["year"] == ["2026"]
    assert request["month"] == ["01", "02"]
    assert set(request["day"]) == {"30", "31", "01", "02"}
    assert request["time"] == [f"{h:02d}:00" for h in range(24)]


class _FakeResultShapedRemote:
    """Simula cdsapi.api.Result: reply['state'] + update() + download()."""

    def __init__(self, states: list[str]):
        self._states = list(states)
        self.reply = {"state": self._states[0]}
        self.downloaded_to: str | None = None

    def update(self) -> None:
        if len(self._states) > 1:
            self._states.pop(0)
        self.reply = {"state": self._states[0]}

    def download(self, target: str) -> str:
        self.downloaded_to = target
        Path(target).write_bytes(b"fake-netcdf-bytes")
        return target


class _FakeRemoteShapedRemote:
    """Simula datastores.Remote: propiedad .status + update() + download()."""

    def __init__(self, states: list[str]):
        self._states = list(states)
        self.downloaded_to: str | None = None

    @property
    def status(self) -> str:
        return self._states[0]

    def update(self) -> None:
        if len(self._states) > 1:
            self._states.pop(0)

    def download(self, target: str) -> str:
        self.downloaded_to = target
        Path(target).write_bytes(b"fake-netcdf-bytes")
        return target


class _FakeCdsapiClient:
    def __init__(self, remote):
        self._remote = remote
        self.retrieve_calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, key: str, wait_until_complete: bool) -> "_FakeCdsapiClient":
        assert wait_until_complete is False
        return self

    def retrieve(self, dataset: str, request: dict):
        self.retrieve_calls.append((dataset, request))
        return self._remote


def test_download_hourly_succeeds_with_result_shaped_remote(tmp_path):
    remote = _FakeResultShapedRemote(["queued", "running", "completed"])
    factory = _FakeCdsapiClient(remote)
    client = Era5Client(url="https://x", key="k", client_factory=factory)
    target = tmp_path / "out.nc"
    result = client.download_hourly(
        "reanalysis-era5-land", {}, target, poll_interval_seconds=0, sleep_fn=lambda _s: None
    )
    assert result == target
    assert target.read_bytes() == b"fake-netcdf-bytes"


def test_download_hourly_succeeds_with_remote_shaped_remote(tmp_path):
    remote = _FakeRemoteShapedRemote(["accepted", "running", "successful"])
    factory = _FakeCdsapiClient(remote)
    client = Era5Client(url="https://x", key="k", client_factory=factory)
    target = tmp_path / "out.nc"
    result = client.download_hourly(
        "reanalysis-era5-land", {}, target, poll_interval_seconds=0, sleep_fn=lambda _s: None
    )
    assert result == target


def test_download_hourly_raises_on_failed_state_not_infinite_loop(tmp_path):
    remote = _FakeResultShapedRemote(["queued", "running", "failed"])
    factory = _FakeCdsapiClient(remote)
    client = Era5Client(url="https://x", key="k", client_factory=factory)
    with pytest.raises(Era5RequestFailedError, match="failed"):
        client.download_hourly(
            "reanalysis-era5-land", {}, tmp_path / "out.nc",
            poll_interval_seconds=0, sleep_fn=lambda _s: None,
        )


def test_download_hourly_raises_on_timeout_not_treated_as_success(tmp_path):
    remote = _FakeResultShapedRemote(["queued", "running", "running", "running"])
    factory = _FakeCdsapiClient(remote)
    client = Era5Client(url="https://x", key="k", client_factory=factory)
    ticks = iter([0.0, 1.0, 2.0, 100.0])  # jumps past timeout on the 4th check
    with pytest.raises(Era5RequestTimeoutError, match="running"):
        client.download_hourly(
            "reanalysis-era5-land", {}, tmp_path / "out.nc",
            poll_interval_seconds=0, timeout_seconds=10.0,
            sleep_fn=lambda _s: None, monotonic_fn=lambda: next(ticks),
        )
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv sync --all-packages --group dev --reinstall-package ingestion && uv run --package ingestion pytest ingestion/tests/test_era5_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.era5.client'`

- [ ] **Step 3: Write `ingestion/src/ingestion/era5/client.py`**

```python
"""Cliente ERA5-Land: construcción de la solicitud a cdsapi y manejo propio
de la cola asíncrona de CDS (envío, polling con timeout configurable,
descarga cuando está lista).

Verificado contra el código fuente real de cdsapi (2026-09-26): con
wait_until_complete=True (el default de la librería), el polling interno
de cdsapi NO tiene límite de tiempo total — solo bounded por
timeout/retry_max de cada petición HTTP individual, en un `while True`
sin salida por tiempo. Por eso este cliente usa wait_until_complete=False
y su propio bucle acotado.

Puede tardar minutos u horas según la carga del servicio CDS — el
timeout_seconds de download_hourly() es la única protección contra una
espera indefinida en un pipeline automatizado.

Ambas familias de objeto "remote" que cdsapi puede devolver (según el
formato del token del usuario) se soportan por duck typing:
- cdsapi.api.Result (clásico): remote.reply["state"] en
  {queued,running,completed,failed}
- ecmwf.datastores Remote (moderno): remote.status en
  {accepted,running,successful,failed,rejected,dismissed,deleted}
"""
import datetime as dt
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cdsapi

ERA5_VARIABLES: tuple[str, ...] = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
)

_SUCCESS_STATES = {"completed", "successful"}
_FAILURE_STATES = {"failed", "rejected", "dismissed", "deleted"}


class Era5RequestFailedError(RuntimeError):
    """La solicitud a CDS terminó en un estado de fallo."""


class Era5RequestTimeoutError(RuntimeError):
    """La solicitud a CDS no completó dentro del timeout configurado."""


def build_request(
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    variables: tuple[str, ...] = ERA5_VARIABLES,
) -> dict[str, Any]:
    if end < start:
        raise ValueError(f"end ({end}) es anterior a start ({start})")
    west, south, east, north = bbox
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += dt.timedelta(days=1)
    years = sorted({d.strftime("%Y") for d in dates})
    months = sorted({d.strftime("%m") for d in dates})
    days = sorted({d.strftime("%d") for d in dates})
    return {
        "variable": list(variables),
        "year": years,
        "month": months,
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        # CDS usa [Norte, Oeste, Sur, Este] — distinto del orden
        # west,south,east,north usado en ingestion/firms y ingestion/dem.
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _state_of(remote: Any) -> str:
    reply = getattr(remote, "reply", None)
    if reply is not None:
        return str(reply["state"])
    return str(remote.status)


class Era5Client:
    def __init__(
        self,
        url: str,
        key: str,
        client_factory: Callable[..., Any] = cdsapi.Client,
    ) -> None:
        self._client = client_factory(url=url, key=key, wait_until_complete=False)

    def download_hourly(
        self,
        dataset: str,
        request: dict[str, Any],
        target: Path,
        poll_interval_seconds: float = 10.0,
        timeout_seconds: float = 3600.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> Path:
        remote = self._client.retrieve(dataset, request)
        start = monotonic_fn()

        while True:
            state = _state_of(remote)

            if state in _SUCCESS_STATES:
                remote.download(str(target))
                return target

            if state in _FAILURE_STATES:
                raise Era5RequestFailedError(
                    f"Solicitud CDS terminó en estado {state!r}"
                )

            if monotonic_fn() - start > timeout_seconds:
                raise Era5RequestTimeoutError(
                    f"Solicitud CDS no completó en {timeout_seconds}s "
                    f"(último estado: {state!r})"
                )

            sleep_fn(poll_interval_seconds)
            remote.update()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_era5_client.py -v`
Expected: 9 passed

If the timeout test's `monotonic_fn` call-count assumption doesn't line up (the loop calls `monotonic_fn()` once per iteration for the timeout check, plus once at start — count them from the actual code path, not from a guess), fix the test's `ticks` iterator to match, ledger the correction, and do not change `download_hourly`'s timeout-check placement to make a miscounted test pass.

- [ ] **Step 5: Commit**

```bash
git add ingestion/pyproject.toml ingestion/src/ingestion/era5/client.py ingestion/tests/test_era5_client.py uv.lock
git commit -m "feat(ingestion): add ERA5-Land client with bounded poll/timeout wrapper"
```

---

### Task 2: `ingestion/era5/aggregate.py` — hourly → daily (mean/sum), date-range clip

**Files:**
- Create: `ingestion/src/ingestion/era5/aggregate.py`
- Test: `ingestion/tests/test_era5_aggregate.py`

**Interfaces:**
- Consumes: nothing from Task 1 (decoupled — takes an `xr.Dataset` or a NetCDF path, not an `Era5Client`).
- Produces: `ingestion.era5.aggregate.aggregate_hourly_to_daily(hourly_nc_path: Path, start: dt.date, end: dt.date, output_path: Path) -> Path`.

- [ ] **Step 1: Write the failing tests**

`ingestion/tests/test_era5_aggregate.py`:

```python
"""Tests de agregación horaria -> diaria de ERA5-Land, sin red."""
import datetime as dt

import numpy as np
import xarray as xr
from ingestion.era5.aggregate import aggregate_hourly_to_daily


def _write_synthetic_hourly_nc(path, start: dt.datetime, n_hours: int) -> None:
    times = [start + dt.timedelta(hours=h) for h in range(n_hours)]
    lat = np.array([-37.0, -37.1])
    lon = np.array([-72.0, -71.9])
    shape = (n_hours, len(lat), len(lon))
    # t2m sube 1 K cada hora desde 280 K; tp es 0.001 m constante por hora
    # (para que la suma diaria dé un total predecible: 24 * 0.001 = 0.024 m).
    t2m = 280.0 + np.arange(n_hours, dtype="float64").reshape(-1, 1, 1) * np.ones(shape)
    d2m = t2m - 5.0
    u10 = np.full(shape, 2.0)
    v10 = np.full(shape, 3.0)
    tp = np.full(shape, 0.001)
    ds = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), t2m),
            "d2m": (("time", "latitude", "longitude"), d2m),
            "u10": (("time", "latitude", "longitude"), u10),
            "v10": (("time", "latitude", "longitude"), v10),
            "tp": (("time", "latitude", "longitude"), tp),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path, engine="h5netcdf")


def test_aggregate_hourly_to_daily_uses_mean_for_temperature(tmp_path):
    hourly_path = tmp_path / "hourly.nc"
    _write_synthetic_hourly_nc(hourly_path, dt.datetime(2026, 1, 15, 0), n_hours=24)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 1
        # media de 280..303 (24 valores, paso 1) = 291.5
        assert float(ds["t2m"].isel(time=0, latitude=0, longitude=0)) == pytest_approx(291.5)


def test_aggregate_hourly_to_daily_uses_sum_for_precipitation(tmp_path):
    hourly_path = tmp_path / "hourly.nc"
    _write_synthetic_hourly_nc(hourly_path, dt.datetime(2026, 1, 15, 0), n_hours=24)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        total = float(ds["tp"].isel(time=0, latitude=0, longitude=0))
        assert total == pytest_approx(0.024)  # 24 horas x 0.001 m, NO el promedio (0.001)


def test_aggregate_hourly_to_daily_clips_dates_outside_requested_range(tmp_path):
    # 3 dias de datos horarios (14, 15, 16 de enero), pero solo se pide el 15
    # -- simula el sobre-fetch documentado de CDS por listas year/month/day.
    hourly_path = tmp_path / "hourly.nc"
    _write_synthetic_hourly_nc(hourly_path, dt.datetime(2026, 1, 14, 0), n_hours=72)
    output_path = tmp_path / "daily.nc"

    result_path = aggregate_hourly_to_daily(
        hourly_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15), output_path
    )

    with xr.open_dataset(result_path, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 1
        assert str(ds["time"].values[0])[:10] == "2026-01-15"


import pytest  # noqa: E402

pytest_approx = pytest.approx
```

- [ ] **Step 1b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_era5_aggregate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.era5.aggregate'`

- [ ] **Step 2: Write `ingestion/src/ingestion/era5/aggregate.py`**

```python
"""Agrega ERA5-Land horario a diario: media para temperatura/punto de
rocío/viento, SUMA para precipitación total (es un campo acumulado, no
una tasa instantánea — promediarlo daría un valor ~24x menor al total
real del día). Recorta primero al rango [start, end] exacto, porque una
solicitud CDS construida con listas year/month/day puede devolver días
de más cuando el rango cruza un límite de mes (ver
ingestion/era5/client.py y docs/decisions.md)."""
import datetime as dt
from pathlib import Path

import xarray as xr

_SUM_VARIABLES = {"tp"}


def aggregate_hourly_to_daily(
    hourly_nc_path: Path, start: dt.date, end: dt.date, output_path: Path
) -> Path:
    with xr.open_dataset(hourly_nc_path, engine="h5netcdf") as ds:
        clipped = ds.sel(
            time=slice(
                dt.datetime.combine(start, dt.time.min),
                dt.datetime.combine(end, dt.time.max),
            )
        )

        sum_vars = [v for v in clipped.data_vars if v in _SUM_VARIABLES]
        mean_vars = [v for v in clipped.data_vars if v not in _SUM_VARIABLES]

        daily_mean = clipped[mean_vars].resample(time="1D").mean() if mean_vars else None
        daily_sum = clipped[sum_vars].resample(time="1D").sum() if sum_vars else None

        if daily_mean is not None and daily_sum is not None:
            daily = xr.merge([daily_mean, daily_sum])
        else:
            daily = daily_mean if daily_mean is not None else daily_sum

        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_netcdf(output_path, engine="h5netcdf")

    return output_path
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_era5_aggregate.py -v`
Expected: 3 passed

- [ ] **Step 4: Commit**

```bash
git add ingestion/src/ingestion/era5/aggregate.py ingestion/tests/test_era5_aggregate.py
git commit -m "feat(ingestion): aggregate ERA5-Land hourly data to daily mean/sum"
```

---

### Task 3: `ingestion/era5/cache.py` + `pipeline.py` — orchestration, date+variables cache

**Files:**
- Create: `ingestion/src/ingestion/era5/cache.py`
- Create: `ingestion/src/ingestion/era5/pipeline.py`
- Test: `ingestion/tests/test_era5_pipeline.py`

**Interfaces:**
- Consumes: `Era5Client.download_hourly`, `build_request`, `ERA5_VARIABLES` (Task 1); `aggregate_hourly_to_daily` (Task 2).
- Produces: `ingestion.era5.cache.cache_key_for(start: dt.date, end: dt.date, variables: tuple[str, ...]) -> str`, `ingestion.era5.pipeline.fetch_daily_era5(bbox, start, end, variables, era5_client, raw_dir, cache_dir, timeout_seconds=3600.0) -> Path` — returns the path to the cached daily NetCDF, skipping the CDS request+aggregation entirely on a cache hit (keyed only on `(start, end, variables)`, per the user's explicit spec — bbox is assumed to be the project's fixed study area for this cache, noted as a scope decision).

- [ ] **Step 1: Write `ingestion/src/ingestion/era5/cache.py`**

```python
"""Clave de cache para un producto ERA5-Land diario ya agregado: hash de
(rango de fechas, variables solicitadas) — bbox no forma parte de la
clave (se asume el bbox de estudio del proyecto, fijo por defecto; ver
docs/decisions.md)."""
import datetime as dt
import hashlib


def cache_key_for(start: dt.date, end: dt.date, variables: tuple[str, ...]) -> str:
    payload = f"{start.isoformat()}|{end.isoformat()}|{','.join(sorted(variables))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 2: Write the failing pipeline tests**

`ingestion/tests/test_era5_pipeline.py`:

```python
"""Tests del pipeline ERA5-Land: cache + orquestación, cdsapi mockeado."""
import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from ingestion.era5.client import ERA5_VARIABLES
from ingestion.era5.pipeline import fetch_daily_era5

BBOX = (-73.7, -39.3, -71.0, -36.5)


class _FakeEra5Client:
    def __init__(self):
        self.download_calls: list[Path] = []

    def download_hourly(self, dataset, request, target, **kwargs):
        self.download_calls.append(target)
        lat = np.array([-37.0, -37.1])
        lon = np.array([-72.0, -71.9])
        times = [dt.datetime(2026, 1, 15, h) for h in range(24)]
        shape = (24, 2, 2)
        ds = xr.Dataset(
            {
                "t2m": (("time", "latitude", "longitude"), np.full(shape, 290.0)),
                "d2m": (("time", "latitude", "longitude"), np.full(shape, 285.0)),
                "u10": (("time", "latitude", "longitude"), np.full(shape, 1.0)),
                "v10": (("time", "latitude", "longitude"), np.full(shape, 1.0)),
                "tp": (("time", "latitude", "longitude"), np.full(shape, 0.001)),
            },
            coords={"time": times, "latitude": lat, "longitude": lon},
        )
        ds.to_netcdf(target, engine="h5netcdf")
        return target


def test_fetch_daily_era5_produces_daily_netcdf(tmp_path):
    client = _FakeEra5Client()
    result = fetch_daily_era5(
        bbox=BBOX, start=dt.date(2026, 1, 15), end=dt.date(2026, 1, 15),
        variables=ERA5_VARIABLES, era5_client=client,
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
    )
    assert result.exists()
    with xr.open_dataset(result, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 1


def test_fetch_daily_era5_cache_hit_skips_download_entirely(tmp_path):
    client = _FakeEra5Client()
    first = fetch_daily_era5(
        bbox=BBOX, start=dt.date(2026, 1, 15), end=dt.date(2026, 1, 15),
        variables=ERA5_VARIABLES, era5_client=client,
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
    )
    calls_after_first = len(client.download_calls)
    assert calls_after_first > 0

    second = fetch_daily_era5(
        bbox=BBOX, start=dt.date(2026, 1, 15), end=dt.date(2026, 1, 15),
        variables=ERA5_VARIABLES, era5_client=client,
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
    )
    assert second == first
    assert len(client.download_calls) == calls_after_first
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_era5_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.era5.pipeline'`

- [ ] **Step 3: Write `ingestion/src/ingestion/era5/pipeline.py`**

```python
"""Orquesta: cache -> solicitud+descarga a CDS -> agregación diaria."""
import datetime as dt
from pathlib import Path
from typing import Any

from ingestion.era5.aggregate import aggregate_hourly_to_daily
from ingestion.era5.cache import cache_key_for
from ingestion.era5.client import ERA5_VARIABLES, build_request


def fetch_daily_era5(
    bbox: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    era5_client: Any,
    raw_dir: Path,
    cache_dir: Path,
    variables: tuple[str, ...] = ERA5_VARIABLES,
    timeout_seconds: float = 3600.0,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key_for(start, end, variables)
    daily_path = cache_dir / f"era5_daily_{key}.nc"
    if daily_path.exists():
        return daily_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = raw_dir / f"era5_hourly_{key}.nc"
    request = build_request(bbox, start, end, variables=variables)
    era5_client.download_hourly(
        "reanalysis-era5-land", request, hourly_path, timeout_seconds=timeout_seconds
    )

    return aggregate_hourly_to_daily(hourly_path, start, end, daily_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_era5_pipeline.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full `ingestion` suite together**

Run: `uv run --package ingestion pytest ingestion/tests -v`
Expected: all pass (existing FIRMS/DEM tests + new ERA5 tests + smoke test)

- [ ] **Step 6: Commit**

```bash
git add ingestion/src/ingestion/era5/cache.py ingestion/src/ingestion/era5/pipeline.py ingestion/tests/test_era5_pipeline.py
git commit -m "feat(ingestion): orchestrate ERA5-Land fetch with date+variables caching"
```

---

### Task 4: `features/weather/derive.py` — wind, RH, reprojection to 250 m

**Files:**
- Modify: `features/pyproject.toml` (+ `h5netcdf`)
- Create: `features/src/features/weather/derive.py`
- Test: `features/tests/test_weather_derive.py`

**Interfaces:**
- Consumes: nothing structurally from `ingestion/era5/` (decoupled — takes a daily NetCDF path, same pattern as `features/terrain` taking a DEM path).
- Produces: `features.weather.derive.wind_speed_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (speed m/s, direction — meteorological "from" convention, degrees, 0–360), `features.weather.derive.relative_humidity_approx(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray` (percent, 0–100), `features.weather.derive.compute_and_save_weather(daily_nc_path: Path, output_dir: Path, target_crs: str, target_resolution_m: int) -> dict[str, Path]` (derives all fields, reprojects each to `target_crs`/`target_resolution_m` with bilinear resampling, writes one GeoTIFF per field, returns `{"wind_speed": path, "wind_direction": path, "relative_humidity": path, "temperature": path, "precipitation": path}`).

- [ ] **Step 1: Add `h5netcdf` to `features/pyproject.toml`**

```toml
dependencies = [
    "shared",
    "rasterio>=1.3",
    "xarray>=2024.7",
    "rioxarray>=0.17",
    "geopandas>=1.0",
    "shapely>=2.0",
    "zarr>=2.18",
    "numpy>=2.0",
    "h5netcdf>=1.3",
]
```

- [ ] **Step 2: Write the failing tests**

`features/tests/test_weather_derive.py`:

```python
"""Tests de derivación de viento/humedad y remuestreo, sin red."""
import datetime as dt

import numpy as np
import pytest
import rasterio
import xarray as xr
from features.weather.derive import (
    compute_and_save_weather,
    relative_humidity_approx,
    wind_speed_direction,
)


def test_wind_speed_direction_northerly_wind():
    # Viento soplando DESDE el norte hacia el sur: u=0 (sin componente
    # este-oeste), v=-1 (moviéndose hacia el sur) -> dirección 0/360 (N).
    speed, direction = wind_speed_direction(np.array([0.0]), np.array([-1.0]))
    assert speed[0] == pytest.approx(1.0)
    assert direction[0] == pytest.approx(0.0, abs=1e-6) or direction[0] == pytest.approx(360.0)


def test_wind_speed_direction_westerly_wind():
    # Soplando DESDE el oeste hacia el este: u=+1, v=0 -> dirección 270 (O).
    speed, direction = wind_speed_direction(np.array([1.0]), np.array([0.0]))
    assert speed[0] == pytest.approx(1.0)
    assert direction[0] == pytest.approx(270.0, abs=1e-6)


def test_relative_humidity_is_100_percent_when_dewpoint_equals_temperature():
    temp_k = np.array([293.15])  # 20 C
    rh = relative_humidity_approx(temp_k, temp_k.copy())
    assert rh[0] == pytest.approx(100.0, abs=0.5)


def test_relative_humidity_decreases_as_dewpoint_spread_widens():
    temp_k = np.full(3, 293.15)
    dewpoints_k = np.array([293.15, 288.15, 273.15])  # spread creciente
    rh = relative_humidity_approx(temp_k, dewpoints_k)
    assert rh[0] > rh[1] > rh[2]
    assert 0.0 <= rh[2] < rh[0] <= 100.5


def _write_synthetic_daily_nc(path, lat, lon) -> None:
    shape = (1, len(lat), len(lon))
    ds = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), np.full(shape, 293.15)),
            "d2m": (("time", "latitude", "longitude"), np.full(shape, 283.15)),
            "u10": (("time", "latitude", "longitude"), np.full(shape, 2.0)),
            "v10": (("time", "latitude", "longitude"), np.full(shape, 3.0)),
            "tp": (("time", "latitude", "longitude"), np.full(shape, 0.01)),
        },
        coords={"time": [dt.datetime(2026, 1, 15)], "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path, engine="h5netcdf")


def test_compute_and_save_weather_writes_geotiffs_at_target_resolution(tmp_path):
    lat = np.array([-36.0, -37.0, -38.0, -39.0])  # descendente, típico ERA5
    lon = np.array([-74.0, -73.0, -72.0, -71.0])
    daily_nc = tmp_path / "daily.nc"
    _write_synthetic_daily_nc(daily_nc, lat, lon)

    output_dir = tmp_path / "weather"
    paths = compute_and_save_weather(
        daily_nc, output_dir, target_crs="EPSG:32719", target_resolution_m=250
    )

    assert set(paths) == {
        "wind_speed", "wind_direction", "relative_humidity", "temperature", "precipitation"
    }
    for path in paths.values():
        assert path.exists()
        with rasterio.open(path) as ds:
            assert ds.crs.to_string() == "EPSG:32719"
            assert ds.res == pytest.approx((250.0, 250.0), abs=1.0)
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv sync --all-packages --group dev --reinstall-package features && uv run --package features pytest features/tests/test_weather_derive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.weather.derive'`

- [ ] **Step 3: Write `features/src/features/weather/derive.py`**

```python
"""Deriva viento (velocidad/dirección) y humedad relativa aproximada desde
ERA5-Land, y remuestrea de ~9 km nativos a la grilla de trabajo del
proyecto (250 m por defecto, EPSG:32719).

Fórmulas:

Velocidad del viento (m/s):
    speed = hipot(u, v)

Dirección del viento (convención meteorológica: rumbo desde donde SOPLA
el viento, no hacia dónde va; grados, sentido horario desde el norte):
    direction = (grados(atan2(u, v)) + 180) mod 360

Humedad relativa aproximada (%) — fórmula de Magnus-Tetens con los
coeficientes de Alduchov & Eskridge (1996), la variante mejorada más
usada en meteorología operativa:
    RH = 100 * exp(17.625*Td / (243.04+Td)) / exp(17.625*T / (243.04+T))
    (T, Td en grados Celsius)
Válida entre -40°C y 50°C, con un error máximo documentado de ±0.4% RH
en ese rango (Alduchov & Eskridge, J. Appl. Meteor., 1996). Es una
aproximación, no una medición: no reemplaza humedad relativa observada.

*** LIMITACIÓN DE DISEÑO, NO UN DETALLE MENOR ***
El remuestreo de ~9 km (grilla nativa de ERA5-Land) a 250 m es
downscaling por INTERPOLACIÓN (bilineal), no una modelación física de
procesos de sub-grilla. No introduce información real a esa escala; solo
suaviza la transición entre celdas de 9 km. Ver docs/limitations.md.
"""
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject


def wind_speed_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(u, v)) + 180.0) % 360.0
    return speed, direction


def relative_humidity_approx(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    numerator = np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    denominator = np.exp((17.625 * temp_c) / (243.04 + temp_c))
    return 100.0 * numerator / denominator


def _source_transform(lat: np.ndarray, lon: np.ndarray) -> rasterio.Affine:
    xres = abs(float(lon[1] - lon[0]))
    yres = abs(float(lat[1] - lat[0]))
    west = float(lon.min()) - xres / 2
    north = float(lat.max()) + yres / 2
    return from_origin(west, north, xres, yres)


def _reproject_field(
    data: np.ndarray,
    src_transform: rasterio.Affine,
    target_crs: str,
    target_resolution_m: int,
) -> tuple[np.ndarray, rasterio.Affine, int, int]:
    height, width = data.shape
    west, south, east, north = rasterio.transform.array_bounds(height, width, src_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        "EPSG:4326", target_crs, width, height, west, south, east, north,
        resolution=(target_resolution_m, target_resolution_m),
    )
    dst_array = np.empty((dst_height, dst_width), dtype="float32")
    reproject(
        source=data.astype("float32"),
        destination=dst_array,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )
    return dst_array, dst_transform, dst_width, dst_height


def _write_geotiff(
    path: Path, data: np.ndarray, transform: rasterio.Affine, crs: str
) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def compute_and_save_weather(
    daily_nc_path: Path, output_dir: Path, target_crs: str, target_resolution_m: int
) -> dict[str, Path]:
    with xr.open_dataset(daily_nc_path, engine="h5netcdf") as ds:
        day = ds.isel(time=0)
        lat = day["latitude"].values
        lon = day["longitude"].values
        u = day["u10"].values
        v = day["v10"].values
        temp_k = day["t2m"].values
        dewpoint_k = day["d2m"].values
        precip = day["tp"].values

    speed, direction = wind_speed_direction(u, v)
    rh = relative_humidity_approx(temp_k, dewpoint_k)

    src_transform = _source_transform(lat, lon)
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = {
        "wind_speed": speed,
        "wind_direction": direction,
        "relative_humidity": rh,
        "temperature": temp_k,
        "precipitation": precip,
    }
    paths: dict[str, Path] = {}
    for name, field in fields.items():
        reprojected, transform, _, _ = _reproject_field(
            field, src_transform, target_crs, target_resolution_m
        )
        paths[name] = _write_geotiff(
            output_dir / f"{name}.tif", reprojected, transform, target_crs
        )
    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package features pytest features/tests/test_weather_derive.py -v`
Expected: 5 passed

If the northerly-wind test's `direction[0] == 0.0 or 360.0` disjunction is unnecessary (the `% 360.0` in the implementation should always normalize to `[0, 360)`, i.e. exactly `0.0`, never `360.0`), simplify the test assertion once confirmed — this is a test-clarity fix, not an implementation change.

- [ ] **Step 5: Run the full `features` suite and `mypy --strict`**

Run: `uv run --package features pytest features/tests -v`
Expected: all pass
Run: `uv run mypy --strict shared/src features/src`
Expected: `Success: no issues found in N source files`

- [ ] **Step 6: Commit**

```bash
git add features/pyproject.toml features/src/features/weather/derive.py features/tests/test_weather_derive.py uv.lock
git commit -m "feat(features): derive wind/humidity from ERA5-Land and resample to 250m"
```

---

### Task 5: `mypy`/`ruff` on `ingestion/era5/`, docs (data-sources, limitations, decisions)

**Files:**
- Modify: `docs/data-sources.md` (append ERA5-Land section)
- Modify: `docs/limitations.md` (append downscaling entry)
- Modify: `docs/decisions.md` (append dataset choice, h5netcdf, polling design)
- Possibly modify: any `ingestion/era5/*.py` needed to satisfy `mypy --strict`

- [ ] **Step 1: Run `mypy`/`ruff` on `ingestion/era5/` and the whole repo**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run mypy --strict ingestion/src/ingestion/era5`
Run: `uv run ruff check .`

Fix whatever surfaces — likely candidate: `cdsapi` ships no type stubs, so `import cdsapi` and `cdsapi.Client` may need a narrow `# type: ignore[import-untyped]` on the import line (check the actual error first; do not add an ignore that isn't needed). Do not weaken the check to pass it.

Expected after fixes: both commands report no errors/issues.

- [ ] **Step 2: Append the ERA5-Land section to `docs/data-sources.md`**

```markdown

## ERA5-Land (Copernicus CDS)

**Qué entrega:** reanálisis de viento, temperatura, humedad y
precipitación, resolución nativa ~9 km, agregado a diario por este
proyecto (media para temperatura/punto de rocío/viento, suma para
precipitación).

**Cómo obtener la API key (gratuita):**
1. Crear cuenta en https://cds.climate.copernicus.eu/
2. Ir a tu perfil y copiar el "Personal Access Token".
3. Ponerlo en `.env` como `CDS_API_KEY=...` y `CDS_API_URL=https://cds.climate.copernicus.eu/api`
   (ver `.env.example`). Este proyecto pasa `url`/`key` directo al
   constructor de `cdsapi.Client` — nunca escribe `~/.cdsapirc`.

**Naturaleza asíncrona de las solicitudes (importante):** CDS encola
cada solicitud; puede tardar **minutos u horas** según la carga del
servicio, no segundos. `ingestion/era5/client.py` implementa su propio
polling con timeout configurable (`timeout_seconds`, por defecto 1
hora) — la propia librería `cdsapi`, en su modo por defecto, no tiene
ningún límite de espera total (se verificó leyendo su código fuente).

**Por qué `reanalysis-era5-land` (horario) y no
`derived-era5-land-daily-statistics`:** el dataset de estadísticas
diarias de CDS **omite variables acumuladas, incluyendo precipitación
total** — inútil para este proyecto, que la requiere. Se pide el
dataset horario crudo y se agrega a diario en `ingestion/era5/aggregate.py`.

**Downscaling — limitación central, no un detalle:** ERA5-Land tiene
~9 km de resolución nativa. Se reproyecta y remuestrea a la grilla de
250 m del proyecto mediante interpolación **bilineal**
(`features/weather/derive.py`). **Esto es downscaling por
interpolación, no una modelación física de procesos de sub-grilla** —
no introduce detalle real a esa escala, solo suaviza la transición
entre celdas de 9 km. Ver `docs/limitations.md`.

**Humedad relativa:** aproximada desde temperatura y punto de rocío con
la fórmula de Magnus-Tetens (coeficientes de Alduchov & Eskridge, 1996):
`RH = 100 * exp(17.625*Td/(243.04+Td)) / exp(17.625*T/(243.04+T))` (T,
Td en °C). Válida entre -40°C y 50°C, error máximo documentado ±0.4% RH
en ese rango — es una aproximación, no una medición real de humedad.
```

- [ ] **Step 3: Append the downscaling entry to `docs/limitations.md`**

```markdown
- **Downscaling de ERA5-Land por interpolación (no física)**: el
  remuestreo de ~9 km a 250 m en `features/weather/derive.py` usa
  interpolación bilineal — es una operación puramente geométrica, no
  una modelación de procesos atmosféricos de sub-grilla. Toda variable
  meteorológica derivada de ERA5-Land a 250 m hereda este límite: la
  variabilidad real por debajo de ~9 km simplemente no está en los
  datos de origen.
```

- [ ] **Step 4: Append the decisions entries to `docs/decisions.md`**

```markdown

## ERA5-Land: `reanalysis-era5-land` horario + agregación propia, en vez de `derived-era5-land-daily-statistics`

El dataset de estadísticas diarias post-procesadas de CDS excluye
variables acumuladas (incluida precipitación total) — verificado en su
propia documentación. Como este proyecto necesita precipitación diaria,
se pide el dataset horario crudo (`reanalysis-era5-land`) y se agrega a
diario en `ingestion/era5/aggregate.py` (media para temperatura/punto de
rocío/viento, suma para precipitación). Costo: una solicitud CDS más
pesada (24 pasos horarios en vez de un producto ya diario); beneficio:
control total y correcto sobre la agregación, sin depender de qué
variables decida excluir el dataset derivado.

## `wait_until_complete=False` + polling propio, en vez del modo por defecto de cdsapi

Se leyó el código fuente de `cdsapi` (2026-09-26): con
`wait_until_complete=True` (el default), el polling interno no tiene
límite de tiempo total — solo timeouts por petición HTTP individual, en
un `while True` sin salida por tiempo. Para cumplir el requisito de
"timeout configurable", este proyecto usa `wait_until_complete=False` y
un bucle propio en `ingestion/era5/client.py` con `timeout_seconds`
explícito. El bucle soporta por duck typing ambas formas de objeto
"remote" que `cdsapi` puede devolver según el formato del token del
usuario (clásico `reply["state"]` vs. moderno `.status`), verificado
leyendo el código fuente de ambas rutas (`ecmwf/cdsapi` y
`ecmwf/ecmwf-datastores-client`).

## `h5netcdf` (en `ingestion` y `features`)

`xarray` (ya aprobado por CLAUDE.md) necesita un motor capaz de leer
NetCDF4/HDF5 real — lo que efectivamente entrega CDS. El único backend
NetCDF ya resuelto transitivamente en el workspace es el de `scipy`,
que solo soporta NetCDF3 clásico y no puede leer una descarga real de
ERA5-Land. `h5netcdf` es la alternativa de wheel puro (vía `h5py`),
más liviana que `netCDF4` (que empaqueta las librerías C completas de
netCDF-C + HDF5).
```

- [ ] **Step 5: Full workspace re-verification**

Run: `uv run --package ingestion pytest ingestion/tests -v`
Run: `uv run --package features pytest features/tests -v`
Run: `uv run ruff check .`
Run: `uv run mypy --strict shared/src features/src`
Run: `uv run mypy --strict ingestion/src/ingestion/era5`

Expected: all green, matching this task's acceptance criteria verbatim, with zero env vars/credentials set (`env -i PATH="$PATH" HOME="$HOME" ...`).

- [ ] **Step 6: Commit**

```bash
git add docs/data-sources.md docs/limitations.md docs/decisions.md
git commit -m "docs: add ERA5-Land data-sources section, downscaling limitation, decisions"
```

---

## Self-Review Notes

- **Spec coverage:** user's 5 tasks map to: 1→Task 1 (client+request builder), 2→Task 1 (bounded poll/timeout), 3→Task 4 (wind/RH derivation + reprojection), 4→Task 3 (date+variables cache), 5→woven through every task's own tests (synthetic NetCDF fixtures built with `xarray` in Task 2/3/4, `cdsapi.Client` fully mocked via constructor injection in Task 1/3). Acceptance criteria (mypy/ruff clean, tests green with no network/credentials, docs/data-sources.md with API key how-to + downscaling limitation) → Task 5.
- **Placeholder scan:** every step has real, runnable code. The two "if this assumption is wrong, fix the test and ledger it" notes (Task 1 Step 4, Task 4 Step 4) name exactly what to check, mirroring the pattern that worked in the FIRMS/DEM plans.
- **Type consistency:** `Era5Client.download_hourly`'s signature (Task 1) matches exactly how Task 3's `pipeline.py` calls it (`era5_client.download_hourly(dataset, request, target, timeout_seconds=...)`). `aggregate_hourly_to_daily`'s signature (Task 2) matches Task 3's call. `build_request`'s output dict keys match what a real `cdsapi.Client.retrieve()` call expects (per the verified request shape in the plan header).
- **Review Focus:** all five items have an owning test — CDS area axis order (Task 1's `converts_bbox_to_cds_north_west_south_east_order`, using the actual asymmetric study-area bbox, not a degenerate symmetric example), the three poll-loop exit conditions (Task 1's three separate `download_hourly` tests: success, failure, timeout), both remote object shapes (Task 1's `_FakeResultShapedRemote`/`_FakeRemoteShapedRemote`), mean-vs-sum aggregation (Task 2's two separate assertions on the same synthetic fixture), and the date-range clip (Task 2's `clips_dates_outside_requested_range` test using a fixture with 72 hours spanning 3 days when only 1 day is requested).
