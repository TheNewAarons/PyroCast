# Ingestion FIRMS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ingestion/firms/` end to end: a NASA FIRMS Area API client (retry + backoff + rate limiting), raw Parquet persistence partitioned by download date, a normalizer into a `shared.schemas.FireDetection` pydantic model, a `pyrocast-ingest firms` typer CLI, fixture-based tests (zero real network calls), and `docs/data-sources.md`.

**Architecture:** `ingestion/firms/client.py` wraps the documented FIRMS Area CSV endpoint (`https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{DATE}`), chunking any `[start, end]` date range into ≤5-day windows (the API's hard `DAY_RANGE` ceiling) and returning raw CSV text per chunk. `ingestion/firms/storage.py` writes each chunk's raw CSV rows to Parquet under `data/raw/firms/download_date=YYYY-MM-DD/`, with a sidecar `.meta.json`. `ingestion/firms/parser.py` turns raw CSV text into `shared.schemas.FireDetection` instances. `ingestion/firms/cli.py` (via `ingestion/cli.py`, the `pyrocast-ingest` console script) wires client → storage → parser → summary print for a date range, defaulting the bbox to `shared.config.get_settings().study_area_bbox`.

**Tech Stack:** Python 3.12, `requests` (already an `ingestion` dependency), `pyarrow` (new — Parquet writer, no pandas needed), `pydantic` (new explicit dep on `shared`, already transitive via `pydantic-settings`), `typer`, `responses` (already an `ingestion` dev dependency, used for the fixture-based HTTP tests), `pytest`.

**Spec:** User's message in this conversation (5 numbered tasks + acceptance criteria) plus `/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md` (FIRMS row of the data-sources table: "NASA FIRMS | Detecciones activas de fuego (VIIRS, 375 m) | API REST, requiere MAP_KEY gratuito"). Both travel with this plan.

**FIRMS Area API — verified against current NASA documentation (2026-09-26), not assumed from memory:**
- Docs: https://firms.modaps.eosdis.nasa.gov/api/area/ and https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html
- URL: `GET https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE]/[DATE]` — `[DATE]` is optional (omitting it returns the most recent available data).
- `AREA_COORDINATES` = `west,south,east,north` (matches `shared.config.Settings.study_area_bbox`'s `(min_lon, min_lat, max_lon, max_lat)` tuple order exactly — no reordering needed).
- `DAY_RANGE`: integer **1–5**. This is a hard ceiling, not a suggestion — the client MUST chunk any longer request.
- `DATE`, when given, is **the first day of the range**: the response covers `[DATE, DATE + DAY_RANGE - 1]` inclusive.
- `SOURCE` (sensor) values include `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `VIIRS_NOAA21_NRT`, `MODIS_NRT`, `LANDSAT_NRT` (US/Canada only), plus `*_SP` (standard/reprocessed) variants. CLAUDE.md's data-sources table names "VIIRS, 375 m" without picking a satellite, so this plan defaults to `VIIRS_SNPP_NRT` and makes it overridable — documented as a decision in Task 6.
- Rate limit: **5000 transactions / 10-minute window per MAP_KEY** (a usage-check endpoint `mapserver/mapkey_status/?MAP_KEY=...` exists but this plan does not call it — see Global Constraints).
- CSV columns actually returned for a VIIRS source (verified from the Academy tutorial's example output): `latitude, longitude, bright_ti4, scan, track, acq_date, acq_time, satellite, instrument, confidence, version, bright_ti5, frp, daynight`. `acq_time` is `HHMM` UTC with no separator (e.g. `"0512"`).
- Documented error behavior is thin and inconsistent across sources: NASA's own tutorial only shows a bare `try/except` around `requests.get`, with no documented status codes for a bad `MAP_KEY` or malformed request — some third-party summaries claim plain-text bodies like `"Invalid MAP_KEY"` come back with HTTP 200, not a 4xx. **This plan does not trust undocumented status-code claims** — it treats any HTTP 429 or 5xx as retryable, and separately validates that a 200 response body is either genuinely empty or starts with the expected CSV header; anything else (a one-line text error body, HTML, etc.) is raised immediately as a non-retryable `FirmsApiError`, never silently parsed as zero detections.

## Global Constraints

- Every module in `ingestion/firms/` ships with `mypy --strict`-clean, `ruff`-clean code (Global Constraint carried over from the bootstrap: `mypy --strict` runs on `shared/` and `features/` only per CLAUDE.md, but this task's own acceptance criteria additionally requires `mypy --strict` and `ruff` clean **specifically on `ingestion/firms/`** — run `mypy --strict ingestion/src/ingestion/firms` as an extra one-off check in Task 6, it does not change the Makefile's `typecheck` target scope).
- Zero real network calls in tests. All HTTP is mocked via the `responses` library (already a dev dependency of `ingestion`).
- New dependencies (`pyarrow` on `ingestion`, explicit `pydantic` on `shared`) get a paragraph in `docs/decisions.md` per CLAUDE.md's "No agregues dependencias sin justificarlas" rule.
- `FIRMS_MAP_KEY` continues to come only from `shared.config.get_settings()` / environment — never hardcoded, never passed as a CLI default.
- The default bbox is `shared.config.get_settings().study_area_bbox`, overridable per-call (client method parameter) and per CLI invocation (`--bbox`).
- Raw persistence must save the response **as-is** (the exact columns FIRMS returns, not the normalized schema) — the normalized `FireDetection` model is a separate, additional artifact, not a replacement for the raw file.
- `docs/data-sources.md` must not be replaced/summarized-away if it already documents other sources later — this task only adds a FIRMS section (the file doesn't exist yet, so Task 6 creates it, but is written so future ingestion tasks append their own `## <Source>` sections rather than rewriting the file).
- Identifiers in English; docstrings/comments/docs in Spanish (CLAUDE.md convention, matches the rest of the repo).

## Review Focus

- **Multi-day request spanning the 5-day ceiling**: a user asking for `--start 2026-01-01 --end 2026-01-12` (12 days) must get complete, non-overlapping coverage via multiple ≤5-day chunk requests — off-by-one here silently drops or duplicates a day of detections.
- **A 200 response whose body is not real CSV** (bad key / malformed request per the ambiguous docs above): must raise `FirmsApiError`, not be parsed as "zero fires" or crash with an unhandled `csv` module exception.
- **A genuinely empty result** (valid CSV header, zero data rows — a real "no fire detected" outcome, not an error): must produce a zero-length Parquet file and an empty `list[FireDetection]`, and the CLI must print a normal summary, not crash or treat it as a failure.
- **429/5xx mid-multi-chunk-request**: the retry/backoff must apply per-chunk-request, and a chunk that exhausts its retries must fail loudly (raise), not silently skip that day range while returning partial success for the rest.
- **`acq_time` without a separator** (`"0512"` not `"05:12"`) parsed into a `datetime`: a naive `datetime.strptime` format string mismatch here would raise on every real row, which unit tests using a hand-rolled fixture (rather than the real recorded fixture) could easily fail to catch — the fixture CSV must use real-shaped values (4-digit zero-padded `acq_time`, including a value like `"0005"` to catch bugs from treating it as an int and losing the leading zero).

---

## File Structure

```
shared/
├── pyproject.toml                          # + explicit `pydantic` dependency
├── src/shared/schemas.py                   # new: FireDetection pydantic model
└── tests/test_schemas.py                   # new: smoke test for the model

ingestion/
├── pyproject.toml                          # + `pyarrow` dependency, + [project.scripts]
├── src/ingestion/
│   ├── cli.py                              # new: root Typer app, registers `firms`
│   └── firms/
│       ├── __init__.py                     # (exists, currently a stub docstring)
│       ├── client.py                       # new: FirmsClient (HTTP, retry, rate limit, chunking)
│       ├── storage.py                      # new: save_raw_response (Parquet + .meta.json)
│       ├── parser.py                       # new: parse_csv_to_detections
│       └── cli.py                          # new: `firms` typer command
└── tests/
    ├── fixtures/
    │   ├── firms_area_ok.csv               # new: recorded-shape fixture, normal response
    │   └── firms_area_error.txt            # new: recorded-shape fixture, error body
    ├── test_firms_client.py                # new
    ├── test_firms_storage.py               # new
    ├── test_firms_parser.py                # new
    └── test_firms_cli.py                   # new

docs/
└── data-sources.md                          # new
```

---

### Task 1: `shared.schemas.FireDetection` + explicit `pydantic` dependency

**Files:**
- Modify: `shared/pyproject.toml`
- Create: `shared/src/shared/schemas.py`
- Test: `shared/tests/test_schemas.py`

**Interfaces:**
- Produces: `shared.schemas.FireDetection` — a `pydantic.BaseModel` with fields `latitude: float`, `longitude: float`, `detected_at: datetime` (UTC, combines the source's acquisition date + time), `frp: float | None` (Fire Radiative Power in MW; `None` when the source omits it), `confidence: str` (kept as the raw label — VIIRS uses `"l"/"n"/"h"` or `"low"/"nominal"/"high"` depending on product version, MODIS uses a numeric-string percentage; normalizing across sensors to one type would lose information, so this plan keeps it a string verbatim), `satellite: str`, `instrument: str`.

- [ ] **Step 1: Add explicit `pydantic` dependency to `shared`**

Edit `shared/pyproject.toml`'s `dependencies` list:

```toml
dependencies = [
    "pydantic>=2.4",
    "pydantic-settings>=2.4",
    "sqlalchemy>=2.0",
    "geoalchemy2>=0.15",
    "psycopg[binary]>=3.2",
]
```

(`pydantic` was already resolved transitively through `pydantic-settings`, and `shared/config.py` already imports it directly — this just makes the dependency graph honest.)

- [ ] **Step 2: Write the failing test**

`shared/tests/test_schemas.py`:

```python
"""Tests de shared.schemas: el modelo normalizado de detecciones de fuego."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from shared.schemas import FireDetection


def test_fire_detection_accepts_valid_fields():
    detection = FireDetection(
        latitude=-37.4689,
        longitude=-72.3524,
        detected_at=datetime(2026, 1, 15, 5, 12, tzinfo=timezone.utc),
        frp=12.3,
        confidence="n",
        satellite="N",
        instrument="VIIRS",
    )
    assert detection.latitude == -37.4689
    assert detection.frp == 12.3
    assert detection.confidence == "n"


def test_fire_detection_allows_frp_none():
    detection = FireDetection(
        latitude=-37.4689,
        longitude=-72.3524,
        detected_at=datetime(2026, 1, 15, 5, 12, tzinfo=timezone.utc),
        frp=None,
        confidence="low",
        satellite="Terra",
        instrument="MODIS",
    )
    assert detection.frp is None


def test_fire_detection_rejects_out_of_range_latitude():
    with pytest.raises(ValidationError):
        FireDetection(
            latitude=95.0,
            longitude=-72.3524,
            detected_at=datetime(2026, 1, 15, 5, 12, tzinfo=timezone.utc),
            frp=None,
            confidence="n",
            satellite="N",
            instrument="VIIRS",
        )
```

- [ ] **Step 2b: Run test to verify it fails**

Run: `cd shared && uv run pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.schemas'`

- [ ] **Step 3: Write `shared/src/shared/schemas.py`**

```python
"""Esquemas normalizados compartidos entre módulos de ingesta y features.

FireDetection es el resultado de normalizar la respuesta cruda de
cualquier fuente de detecciones activas de fuego (hoy: NASA FIRMS) a un
esquema común, independiente del formato CSV/columnas propias de cada
sensor.
"""
from datetime import datetime

from pydantic import BaseModel, Field


class FireDetection(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    detected_at: datetime
    frp: float | None = None
    # Se mantiene como texto tal cual lo entrega la fuente: VIIRS usa
    # categorías ("l"/"n"/"h" o "low"/"nominal"/"high" según versión del
    # producto), MODIS usa un porcentaje numérico como string. Forzar un
    # tipo único perdería información.
    confidence: str
    satellite: str
    instrument: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd shared && uv run pytest tests/test_schemas.py -v`
Expected: 3 passed

- [ ] **Step 5: `uv sync` to pick up the new explicit dependency, then full shared suite + mypy --strict**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv sync --all-packages --group dev --reinstall-package shared`
Run: `uv run --package shared pytest shared/tests -v`
Expected: all shared tests pass (existing + 3 new)
Run: `uv run mypy --strict shared/src features/src`
Expected: `Success: no issues found in N source files`

- [ ] **Step 6: Commit**

```bash
git add shared/pyproject.toml shared/src/shared/schemas.py shared/tests/test_schemas.py uv.lock
git commit -m "feat(shared): add FireDetection normalized schema"
```

---

### Task 2: `FirmsClient` — HTTP, chunking, retry/backoff, rate limiting

**Files:**
- Modify: `ingestion/pyproject.toml` (no new runtime deps needed here — `requests` and dev-only `responses`/`vcrpy` already present; `pyarrow` is added in Task 3)
- Create: `ingestion/src/ingestion/firms/client.py`
- Create: `ingestion/tests/fixtures/firms_area_ok.csv`
- Create: `ingestion/tests/fixtures/firms_area_error.txt`
- Test: `ingestion/tests/test_firms_client.py`

**Interfaces:**
- Consumes: nothing from other tasks (this is the leaf-most new module).
- Produces: `ingestion.firms.client.FirmsClient(map_key: str, session: requests.Session | None = None, max_retries: int = 4, backoff_base_seconds: float = 1.0, min_request_interval_seconds: float = 0.25, sleep_fn: Callable[[float], None] = time.sleep, monotonic_fn: Callable[[], float] = time.monotonic)`, method `fetch_area_csv(bbox: tuple[float, float, float, float], sensor: str, day_range: int, date: dt.date | None = None) -> str` (single request, `day_range` 1–5, raises `ValueError` if out of range), method `fetch_range(bbox: tuple[float, float, float, float], sensor: str, start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date, str]]` (chunks `[start, end]` into ≤5-day windows, returns a list of `(chunk_start, chunk_end, raw_csv_text)` — Task 4's storage/parser consume `raw_csv_text` per chunk, Task 4/5's orchestration consumes `chunk_start`/`chunk_end` for the metadata sidecar). Exception types: `ingestion.firms.client.FirmsApiError(RuntimeError)` (non-retryable — bad key/malformed response), raised also when retries are exhausted on a retryable error (chained via `raise ... from ...`).

- [ ] **Step 1: Record the two fixtures**

`ingestion/tests/fixtures/firms_area_ok.csv` (a realistic-shaped VIIRS response — note the zero-padded 4-digit `acq_time` and the mix of day/night and confidence labels, per Review Focus):

```csv
latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
-37.4689,-72.3524,335.2,0.42,0.39,2026-01-15,0005,N,VIIRS,n,2.0NRT,289.1,12.3,N
-37.5012,-72.4108,310.7,0.41,0.38,2026-01-15,1742,N,VIIRS,h,2.0NRT,285.4,45.8,D
-37.4801,-72.3899,301.9,0.44,0.40,2026-01-16,0312,N,VIIRS,l,2.0NRT,280.2,3.1,N
```

`ingestion/tests/fixtures/firms_area_error.txt` (shaped like the plain-text error body the ambiguous docs describe — deliberately NOT valid CSV, to exercise the "reject a 200 that isn't real CSV" path):

```
Invalid MAP_KEY
```

- [ ] **Step 2: Write the failing tests**

`ingestion/tests/test_firms_client.py`:

```python
"""Tests del cliente FIRMS: cero llamadas de red reales (usa `responses`)."""
import datetime as dt
from pathlib import Path

import pytest
import responses
from ingestion.firms.client import FirmsApiError, FirmsClient

FIXTURES = Path(__file__).parent / "fixtures"
OK_CSV = (FIXTURES / "firms_area_ok.csv").read_text()
ERROR_BODY = (FIXTURES / "firms_area_error.txt").read_text()

BBOX = (-73.7, -39.3, -71.0, -36.5)  # west, south, east, north


def _url(map_key: str = "test-key", sensor: str = "VIIRS_SNPP_NRT", day_range: int = 5,
          date: str | None = "2026-01-15") -> str:
    coords = f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"
    base = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{sensor}/{coords}/{day_range}"
    return f"{base}/{date}" if date else base


@responses.activate
def test_fetch_area_csv_returns_raw_text_on_success():
    responses.add(responses.GET, _url(), body=OK_CSV, status=200)
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    result = client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15))
    assert result == OK_CSV


@responses.activate
def test_fetch_area_csv_rejects_day_range_above_five():
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    with pytest.raises(ValueError, match="day_range"):
        client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=6, date=dt.date(2026, 1, 15))


@responses.activate
def test_fetch_area_csv_retries_on_429_then_succeeds():
    responses.add(responses.GET, _url(), status=429)
    responses.add(responses.GET, _url(), body=OK_CSV, status=200)
    sleeps: list[float] = []
    client = FirmsClient(map_key="test-key", max_retries=3, sleep_fn=sleeps.append)
    result = client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15))
    assert result == OK_CSV
    assert len(sleeps) == 1  # one retry happened, one backoff sleep


@responses.activate
def test_fetch_area_csv_retries_on_500_then_raises_after_exhausting_retries():
    for _ in range(4):
        responses.add(responses.GET, _url(), status=500)
    client = FirmsClient(map_key="test-key", max_retries=3, sleep_fn=lambda _seconds: None)
    with pytest.raises(FirmsApiError, match="500"):
        client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15))


@responses.activate
def test_fetch_area_csv_raises_on_error_body_without_retrying():
    responses.add(responses.GET, _url(), body=ERROR_BODY, status=200)
    calls = {"n": 0}

    def _counting_sleep(_seconds: float) -> None:
        calls["n"] += 1

    client = FirmsClient(map_key="test-key", max_retries=3, sleep_fn=_counting_sleep)
    with pytest.raises(FirmsApiError, match="Invalid MAP_KEY"):
        client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15))
    assert calls["n"] == 0  # no retry for a bad-key body — retrying can't fix it


@responses.activate
def test_fetch_area_csv_rate_limits_between_requests():
    responses.add(responses.GET, _url(date="2026-01-15"), body=OK_CSV, status=200)
    responses.add(responses.GET, _url(date="2026-01-16"), body=OK_CSV, status=200)
    ticks = iter([0.0, 0.05, 0.05])  # third call is the post-sleep re-check
    monotonic = lambda: next(ticks, 10.0)  # noqa: E731
    sleeps: list[float] = []
    client = FirmsClient(
        map_key="test-key",
        min_request_interval_seconds=0.2,
        sleep_fn=sleeps.append,
        monotonic_fn=monotonic,
    )
    client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15))
    client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 16))
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.15, abs=0.01)  # 0.2 - (0.05 - 0.0)


def test_fetch_range_chunks_into_windows_of_at_most_five_days():
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    chunks = client._chunk_date_range(dt.date(2026, 1, 1), dt.date(2026, 1, 12))
    assert chunks == [
        (dt.date(2026, 1, 1), dt.date(2026, 1, 5)),
        (dt.date(2026, 1, 6), dt.date(2026, 1, 10)),
        (dt.date(2026, 1, 11), dt.date(2026, 1, 12)),
    ]


def test_fetch_range_chunks_single_day():
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    chunks = client._chunk_date_range(dt.date(2026, 1, 1), dt.date(2026, 1, 1))
    assert chunks == [(dt.date(2026, 1, 1), dt.date(2026, 1, 1))]


@responses.activate
def test_fetch_range_makes_one_request_per_chunk_and_returns_all():
    responses.add(responses.GET, _url(date="2026-01-01", day_range=5), body=OK_CSV, status=200)
    responses.add(responses.GET, _url(date="2026-01-06", day_range=2), body=OK_CSV, status=200)
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    results = client.fetch_range(BBOX, sensor="VIIRS_SNPP_NRT", start=dt.date(2026, 1, 1), end=dt.date(2026, 1, 7))
    assert [(s, e) for s, e, _ in results] == [
        (dt.date(2026, 1, 1), dt.date(2026, 1, 5)),
        (dt.date(2026, 1, 6), dt.date(2026, 1, 7)),
    ]
    assert all(raw == OK_CSV for _, _, raw in results)
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_firms_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.firms.client'`

- [ ] **Step 3: Write `ingestion/src/ingestion/firms/client.py`**

```python
"""Cliente para el Area API de NASA FIRMS (detecciones activas de fuego).

Referencia verificada contra la documentación vigente (2026-09-26):
https://firms.modaps.eosdis.nasa.gov/api/area/ — no asumida de memoria.

URL: https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/
     [west,south,east,north]/[DAY_RANGE]/[DATE]
- DAY_RANGE: entero 1..5 (límite duro de la API).
- DATE (opcional): primer día del rango devuelto; el rango es
  [DATE, DATE + DAY_RANGE - 1].
- Límite de la API: 5000 transacciones / ventana de 10 minutos por
  MAP_KEY.

El comportamiento documentado ante errores es débil: no hay códigos de
estado documentados oficialmente para un MAP_KEY inválido o una
solicitud malformada. Este cliente no confía en suposiciones no
verificadas: trata 429/5xx como reintentables, y valida que un cuerpo
200 sea CSV real (encabezado esperado o vacío) antes de intentar
parsearlo — cualquier otra cosa (texto de error de una línea, HTML,
etc.) se levanta de inmediato como FirmsApiError, sin reintentar.
"""
import datetime as dt
import time
from collections.abc import Callable

import requests

_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
_EXPECTED_HEADER_PREFIX = "latitude,longitude"
_MIN_DAY_RANGE = 1
_MAX_DAY_RANGE = 5
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class FirmsApiError(RuntimeError):
    """Error no reintentable del Area API de FIRMS (clave inválida,
    cuerpo de respuesta que no es CSV, o reintentos agotados)."""


class FirmsClient:
    def __init__(
        self,
        map_key: str,
        session: requests.Session | None = None,
        max_retries: int = 4,
        backoff_base_seconds: float = 1.0,
        min_request_interval_seconds: float = 0.25,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._map_key = map_key
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._min_request_interval_seconds = min_request_interval_seconds
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        self._last_request_at: float | None = None

    def _rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self._monotonic_fn() - self._last_request_at
        remaining = self._min_request_interval_seconds - elapsed
        if remaining > 0:
            self._sleep_fn(remaining)

    def fetch_area_csv(
        self,
        bbox: tuple[float, float, float, float],
        sensor: str,
        day_range: int,
        date: dt.date | None = None,
    ) -> str:
        if not (_MIN_DAY_RANGE <= day_range <= _MAX_DAY_RANGE):
            raise ValueError(
                f"day_range debe estar entre {_MIN_DAY_RANGE} y {_MAX_DAY_RANGE} "
                f"(límite del Area API de FIRMS), recibido: {day_range}"
            )
        west, south, east, north = bbox
        coords = f"{west},{south},{east},{north}"
        url = f"{_BASE_URL}/{self._map_key}/{sensor}/{coords}/{day_range}"
        if date is not None:
            url = f"{url}/{date.isoformat()}"

        attempt = 0
        while True:
            self._rate_limit()
            self._last_request_at = self._monotonic_fn()
            response = self._session.get(url, timeout=30)

            if response.status_code == 200:
                body = response.text
                if body == "" or body.lstrip().lower().startswith(_EXPECTED_HEADER_PREFIX):
                    return body
                raise FirmsApiError(
                    f"Respuesta 200 de FIRMS no parece CSV válido: {body[:200]!r}"
                )

            if response.status_code in _RETRYABLE_STATUSES and attempt < self._max_retries:
                self._sleep_fn(self._backoff_base_seconds * (2**attempt))
                attempt += 1
                continue

            raise FirmsApiError(
                f"FIRMS Area API devolvió estado {response.status_code} tras "
                f"{attempt} reintento(s): {response.text[:200]!r}"
            )

    def _chunk_date_range(self, start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
        if end < start:
            raise ValueError(f"end ({end}) es anterior a start ({start})")
        chunks: list[tuple[dt.date, dt.date]] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + dt.timedelta(days=_MAX_DAY_RANGE - 1), end)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + dt.timedelta(days=1)
        return chunks

    def fetch_range(
        self,
        bbox: tuple[float, float, float, float],
        sensor: str,
        start: dt.date,
        end: dt.date,
    ) -> list[tuple[dt.date, dt.date, str]]:
        results: list[tuple[dt.date, dt.date, str]] = []
        for chunk_start, chunk_end in self._chunk_date_range(start, end):
            day_range = (chunk_end - chunk_start).days + 1
            raw = self.fetch_area_csv(bbox, sensor=sensor, day_range=day_range, date=chunk_start)
            results.append((chunk_start, chunk_end, raw))
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv sync --all-packages --group dev --reinstall-package ingestion && uv run --package ingestion pytest ingestion/tests/test_firms_client.py -v`
Expected: 9 passed

If `test_fetch_area_csv_rate_limits_between_requests` is flaky because of the `monotonic_fn` iterator's call-count assumptions, fix the test's mock (not the implementation) to match exactly how many times `_rate_limit` calls `monotonic_fn` per `fetch_area_csv` invocation (once, at the top of `_rate_limit`, plus once when stamping `_last_request_at` — i.e. 2 calls per request); ledger the correction as a ruling if the exact count differs from what's written above.

- [ ] **Step 5: Commit**

```bash
git add ingestion/tests/fixtures ingestion/tests/test_firms_client.py ingestion/src/ingestion/firms/client.py
git commit -m "feat(ingestion): add FirmsClient with retry, backoff and rate limiting"
```

---

### Task 3: Raw persistence — `storage.py` (Parquet + metadata sidecar)

**Files:**
- Modify: `ingestion/pyproject.toml` (+ `pyarrow` dependency)
- Create: `ingestion/src/ingestion/firms/storage.py`
- Test: `ingestion/tests/test_firms_storage.py`

**Interfaces:**
- Consumes: nothing structurally from Task 2 (storage takes raw CSV text as a plain string — decoupled from the client on purpose, so storage tests don't need `responses`).
- Produces: `ingestion.firms.storage.save_raw_response(raw_csv_text: str, query_start: dt.date, query_end: dt.date, bbox: tuple[float, float, float, float], sensor: str, downloaded_at: dt.datetime, base_dir: Path) -> Path` — returns the path to the written `.parquet` file (the `.meta.json` sidecar sits next to it, same stem). Task 5 (CLI orchestration) calls this once per chunk from `FirmsClient.fetch_range`.

- [ ] **Step 1: Add `pyarrow` to `ingestion/pyproject.toml`**

```toml
dependencies = [
    "shared",
    "requests>=2.32",
    "cdsapi>=0.7",
    "typer>=0.12",
    "pyarrow>=17.0",
]
```

- [ ] **Step 2: Write the failing tests**

`ingestion/tests/test_firms_storage.py`:

```python
"""Tests de persistencia cruda de FIRMS: Parquet + metadatos, sin red."""
import datetime as dt
import json

import pyarrow.parquet as pq
from ingestion.firms.storage import save_raw_response

RAW_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "-37.4689,-72.3524,335.2,0.42,0.39,2026-01-15,0005,N,VIIRS,n,2.0NRT,289.1,12.3,N\n"
)
BBOX = (-73.7, -39.3, -71.0, -36.5)


def test_save_raw_response_writes_parquet_partitioned_by_download_date(tmp_path):
    downloaded_at = dt.datetime(2026, 1, 20, 10, 30, tzinfo=dt.timezone.utc)
    parquet_path = save_raw_response(
        raw_csv_text=RAW_CSV,
        query_start=dt.date(2026, 1, 15),
        query_end=dt.date(2026, 1, 15),
        bbox=BBOX,
        sensor="VIIRS_SNPP_NRT",
        downloaded_at=downloaded_at,
        base_dir=tmp_path,
    )
    assert parquet_path.exists()
    assert parquet_path.parent == tmp_path / "firms" / "download_date=2026-01-20"

    table = pq.read_table(parquet_path)
    assert table.num_rows == 1
    assert "latitude" in table.column_names
    assert table.column("latitude")[0].as_py() == -37.4689


def test_save_raw_response_writes_metadata_sidecar(tmp_path):
    downloaded_at = dt.datetime(2026, 1, 20, 10, 30, tzinfo=dt.timezone.utc)
    parquet_path = save_raw_response(
        raw_csv_text=RAW_CSV,
        query_start=dt.date(2026, 1, 15),
        query_end=dt.date(2026, 1, 16),
        bbox=BBOX,
        sensor="VIIRS_SNPP_NRT",
        downloaded_at=downloaded_at,
        base_dir=tmp_path,
    )
    meta_path = parquet_path.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["query_start"] == "2026-01-15"
    assert meta["query_end"] == "2026-01-16"
    assert meta["bbox"] == list(BBOX)
    assert meta["sensor"] == "VIIRS_SNPP_NRT"
    assert meta["row_count"] == 1


def test_save_raw_response_handles_empty_result_without_raising(tmp_path):
    empty_csv = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
    )
    downloaded_at = dt.datetime(2026, 1, 20, 10, 30, tzinfo=dt.timezone.utc)
    parquet_path = save_raw_response(
        raw_csv_text=empty_csv,
        query_start=dt.date(2026, 1, 15),
        query_end=dt.date(2026, 1, 15),
        bbox=BBOX,
        sensor="VIIRS_SNPP_NRT",
        downloaded_at=downloaded_at,
        base_dir=tmp_path,
    )
    table = pq.read_table(parquet_path)
    assert table.num_rows == 0
    meta = json.loads(parquet_path.with_suffix(".meta.json").read_text())
    assert meta["row_count"] == 0
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_firms_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.firms.storage'`

- [ ] **Step 3: Write `ingestion/src/ingestion/firms/storage.py`**

```python
"""Persistencia cruda de respuestas del Area API de FIRMS.

Cada respuesta (ya sea de una consulta completa o de un chunk de ≤5
días) se guarda tal cual — mismas columnas que entrega la API, sin
normalizar — en Parquet, particionado por fecha de DESCARGA (no de
detección), junto a un .meta.json con los datos de la consulta.
"""
import csv
import datetime as dt
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def save_raw_response(
    raw_csv_text: str,
    query_start: dt.date,
    query_end: dt.date,
    bbox: tuple[float, float, float, float],
    sensor: str,
    downloaded_at: dt.datetime,
    base_dir: Path,
) -> Path:
    reader = csv.DictReader(io.StringIO(raw_csv_text))
    fieldnames = reader.fieldnames or []
    rows = list(reader)

    columns: dict[str, list[str]] = {name: [] for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            columns[name].append(row[name])
    table = pa.table({name: pa.array(values) for name, values in columns.items()}) if fieldnames else pa.table({})

    partition_dir = base_dir / "firms" / f"download_date={downloaded_at.date().isoformat()}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{query_start.isoformat()}_{query_end.isoformat()}_{sensor}"
    parquet_path = partition_dir / f"{stem}.parquet"
    pq.write_table(table, parquet_path)

    meta = {
        "query_start": query_start.isoformat(),
        "query_end": query_end.isoformat(),
        "bbox": list(bbox),
        "sensor": sensor,
        "downloaded_at": downloaded_at.isoformat(),
        "row_count": len(rows),
    }
    parquet_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    return parquet_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv sync --all-packages --group dev --reinstall-package ingestion && uv run --package ingestion pytest ingestion/tests/test_firms_storage.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/pyproject.toml ingestion/src/ingestion/firms/storage.py ingestion/tests/test_firms_storage.py uv.lock
git commit -m "feat(ingestion): persist raw FIRMS responses to partitioned Parquet"
```

---

### Task 4: Parser — raw CSV → `list[FireDetection]`

**Files:**
- Create: `ingestion/src/ingestion/firms/parser.py`
- Test: `ingestion/tests/test_firms_parser.py`

**Interfaces:**
- Consumes: `shared.schemas.FireDetection` (Task 1).
- Produces: `ingestion.firms.parser.parse_csv_to_detections(raw_csv_text: str) -> list[FireDetection]`. Task 5's CLI orchestration calls this once per chunk, after `storage.save_raw_response` for that same chunk.

- [ ] **Step 1: Write the failing tests**

`ingestion/tests/test_firms_parser.py`:

```python
"""Tests del parser FIRMS → shared.schemas.FireDetection."""
import datetime as dt
from pathlib import Path

from ingestion.firms.parser import parse_csv_to_detections

FIXTURES = Path(__file__).parent / "fixtures"
OK_CSV = (FIXTURES / "firms_area_ok.csv").read_text()


def test_parse_csv_to_detections_normalizes_all_rows():
    detections = parse_csv_to_detections(OK_CSV)
    assert len(detections) == 3
    first = detections[0]
    assert first.latitude == -37.4689
    assert first.longitude == -72.3524
    assert first.frp == 12.3
    assert first.confidence == "n"
    assert first.satellite == "N"
    assert first.instrument == "VIIRS"


def test_parse_csv_to_detections_combines_date_and_zero_padded_time_utc():
    detections = parse_csv_to_detections(OK_CSV)
    # acq_date=2026-01-15, acq_time="0005" -> 00:05 UTC, not 05:00 and not
    # dropped-leading-zero "5" minutes.
    assert detections[0].detected_at == dt.datetime(2026, 1, 15, 0, 5, tzinfo=dt.timezone.utc)


def test_parse_csv_to_detections_handles_empty_result_without_raising():
    empty_csv = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
    )
    assert parse_csv_to_detections(empty_csv) == []
```

- [ ] **Step 1b: Run tests to verify they fail**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_firms_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.firms.parser'`

- [ ] **Step 2: Write `ingestion/src/ingestion/firms/parser.py`**

```python
"""Normaliza el CSV crudo del Area API de FIRMS a shared.schemas.FireDetection."""
import csv
import datetime as dt
import io

from shared.schemas import FireDetection


def _parse_detected_at(acq_date: str, acq_time: str) -> dt.datetime:
    # acq_time viene como "HHMM" sin separador, con cero a la izquierda
    # (p. ej. "0005" = 00:05 UTC) — NO tratar como entero, se pierde el
    # cero inicial.
    hour = int(acq_time[:2])
    minute = int(acq_time[2:])
    date = dt.date.fromisoformat(acq_date)
    return dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=dt.timezone.utc)


def parse_csv_to_detections(raw_csv_text: str) -> list[FireDetection]:
    reader = csv.DictReader(io.StringIO(raw_csv_text))
    detections: list[FireDetection] = []
    for row in reader:
        frp_raw = row.get("frp", "").strip()
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

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_firms_parser.py -v`
Expected: 3 passed

- [ ] **Step 4: Commit**

```bash
git add ingestion/src/ingestion/firms/parser.py ingestion/tests/test_firms_parser.py
git commit -m "feat(ingestion): parse raw FIRMS CSV into FireDetection records"
```

---

### Task 5: CLI — `pyrocast-ingest firms --start DATE --end DATE`

**Files:**
- Modify: `ingestion/pyproject.toml` (+ `[project.scripts]`)
- Create: `ingestion/src/ingestion/cli.py`
- Create: `ingestion/src/ingestion/firms/cli.py`
- Test: `ingestion/tests/test_firms_cli.py`

**Interfaces:**
- Consumes: `FirmsClient.fetch_range` (Task 2), `storage.save_raw_response` (Task 3), `parser.parse_csv_to_detections` (Task 4), `shared.config.get_settings` (existing — `study_area_bbox`, `firms_map_key`, `data_raw_dir`).
- Produces: console script `pyrocast-ingest`, invoked as `pyrocast-ingest firms --start 2026-01-15 --end 2026-01-16 [--bbox west,south,east,north] [--sensor VIIRS_SNPP_NRT]`.

- [ ] **Step 1: Add the console script entry point**

Edit `ingestion/pyproject.toml`, add:

```toml
[project.scripts]
pyrocast-ingest = "ingestion.cli:app"
```

- [ ] **Step 2: Write the failing test**

`ingestion/tests/test_firms_cli.py`:

```python
"""Test de humo del CLI de ingesta FIRMS: cero red real."""
import re
from pathlib import Path

import responses
from ingestion.cli import app
from typer.testing import CliRunner

FIXTURES = Path(__file__).parent / "fixtures"
OK_CSV = (FIXTURES / "firms_area_ok.csv").read_text()

runner = CliRunner()


@responses.activate
def test_firms_cli_downloads_and_prints_summary(tmp_path, monkeypatch):
    for key, value in {
        "FIRMS_MAP_KEY": "test-key",
        "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
        "CDS_API_KEY": "x",
        "COPERNICUS_DATASPACE_CLIENT_ID": "x",
        "COPERNICUS_DATASPACE_CLIENT_SECRET": "x",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "pyrocast",
        "POSTGRES_USER": "pyrocast",
        "POSTGRES_PASSWORD": "x",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    responses.add(
        responses.GET,
        re.compile(r"https://firms\.modaps\.eosdis\.nasa\.gov/api/area/csv/.*"),
        body=OK_CSV,
        status=200,
    )

    result = runner.invoke(
        app,
        [
            "firms",
            "--start",
            "2026-01-15",
            "--end",
            "2026-01-15",
            "--bbox",
            "-73.7,-39.3,-71.0,-36.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "3" in result.output  # 3 detections in the fixture
    saved = list((tmp_path / "data" / "raw" / "firms").glob("**/*.parquet"))
    assert len(saved) == 1
```

- [ ] **Step 2b: Run test to verify it fails**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run --package ingestion pytest ingestion/tests/test_firms_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.cli'`

- [ ] **Step 3: Write `ingestion/src/ingestion/firms/cli.py`**

```python
"""Comando `firms` del CLI de ingesta: descarga, persiste y normaliza."""
import datetime as dt

import typer
from ingestion.firms.client import FirmsClient
from ingestion.firms.parser import parse_csv_to_detections
from ingestion.firms.storage import save_raw_response
from shared.config import get_settings

app = typer.Typer()


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("bbox debe tener 4 valores: west,south,east,north")
    return (parts[0], parts[1], parts[2], parts[3])


@app.command()
def firms(
    start: dt.datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Fecha de inicio (YYYY-MM-DD)"),
    end: dt.datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Fecha de fin (YYYY-MM-DD)"),
    bbox: str | None = typer.Option(
        None, help="west,south,east,north — por defecto, el bbox de shared.config"
    ),
    sensor: str = typer.Option("VIIRS_SNPP_NRT", help="SOURCE del Area API de FIRMS"),
) -> None:
    settings = get_settings()
    area = _parse_bbox(bbox) if bbox else settings.study_area_bbox

    client = FirmsClient(map_key=settings.firms_map_key)
    chunks = client.fetch_range(area, sensor=sensor, start=start.date(), end=end.date())

    total_detections = 0
    for chunk_start, chunk_end, raw_csv in chunks:
        save_raw_response(
            raw_csv_text=raw_csv,
            query_start=chunk_start,
            query_end=chunk_end,
            bbox=area,
            sensor=sensor,
            downloaded_at=dt.datetime.now(dt.timezone.utc),
            base_dir=settings.data_raw_dir,
        )
        detections = parse_csv_to_detections(raw_csv)
        total_detections += len(detections)
        typer.echo(f"{chunk_start}..{chunk_end}: {len(detections)} detecciones")

    typer.echo(f"Total: {total_detections} detecciones en {len(chunks)} consulta(s)")
```

- [ ] **Step 4: Write `ingestion/src/ingestion/cli.py`**

```python
"""Punto de entrada del CLI de ingesta: `pyrocast-ingest`."""
import typer
from ingestion.firms.cli import app as firms_app

app = typer.Typer()
app.add_typer(firms_app, name="firms")
```

Note: `ingestion/firms/cli.py`'s `@app.command()` with no explicit name, mounted via `add_typer(firms_app, name="firms")`, yields `pyrocast-ingest firms --start ... --end ...` (Typer treats a single-command sub-app as directly invokable without a nested command name) — if this is wrong when actually run (Typer instead demands `pyrocast-ingest firms firms --start ...`), the fix is `app.command(name="firms")` directly on the root `app` in `ingestion/cli.py` calling into a plain function imported from `ingestion.firms.cli`, not a nested Typer — ledger this as a ruling if the brief's assumption doesn't hold, and adjust both `cli.py` files accordingly; the test in Step 2 either passes as-is (confirming the nesting works) or fails with a Click "no such command" message (confirming it doesn't), so Step 2b/5 is the actual verification, not this note.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv sync --all-packages --group dev --reinstall-package ingestion && uv run --package ingestion pytest ingestion/tests/test_firms_cli.py -v`
Expected: 1 passed. If it fails on command nesting (see the note in Step 4), apply that fix, rerun, and ledger the ruling.

- [ ] **Step 6: Run the full `ingestion` suite together**

Run: `uv run --package ingestion pytest ingestion/tests -v`
Expected: all pass (existing smoke test + all new firms tests)

- [ ] **Step 7: Commit**

```bash
git add ingestion/pyproject.toml ingestion/src/ingestion/cli.py ingestion/src/ingestion/firms/cli.py ingestion/tests/test_firms_cli.py uv.lock
git commit -m "feat(ingestion): add pyrocast-ingest firms CLI"
```

---

### Task 6: `mypy --strict` / `ruff` on `ingestion/firms/`, `docs/data-sources.md`, `docs/decisions.md`

**Files:**
- Create: `docs/data-sources.md`
- Modify: `docs/decisions.md`
- Possibly modify: any `ingestion/firms/*.py` file, to satisfy `mypy --strict` (this task's acceptance criteria requires it specifically for `ingestion/firms/`, which is not in the Makefile's `typecheck` scope, so issues here weren't caught by Tasks 1–5's verification steps)

- [ ] **Step 1: Run `mypy --strict` and `ruff` on `ingestion/firms/` specifically**

Run: `cd /Users/aarons/Documents/Projects/FullPy/PyroCast && uv run mypy --strict ingestion/src/ingestion/firms ingestion/src/ingestion/cli.py`
Run: `uv run ruff check ingestion/`

Fix whatever surfaces (likely candidates given the code above: missing return-type annotations are already present throughout, but `--strict` may flag the `pa.table(...) if fieldnames else pa.table({})` ternary's inferred type, or untyped `responses`/`typer` third-party stubs — add `# type: ignore[...]` with the specific error code only where a real stub gap exists, never blanket-ignore). Do not weaken the check to make it pass — fix the actual code or add a precise, justified ignore.

Expected after fixes: both commands report no errors/issues.

- [ ] **Step 2: Write `docs/data-sources.md`**

```markdown
# Fuentes de datos

## NASA FIRMS

**Qué entrega:** detecciones activas de fuego casi en tiempo real
(VIIRS 375 m por defecto en este proyecto; también soporta MODIS 1 km y
LANDSAT, ver `--sensor`).

**Cómo obtener el MAP_KEY (gratuito):**
1. Ir a https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Registrar un correo — el MAP_KEY llega por email.
3. Ponerlo en `.env` como `FIRMS_MAP_KEY=...` (ver `.env.example`).

**Límites de la API (verificados contra la documentación vigente):**
- `DAY_RANGE` máximo por consulta: **5 días**. `pyrocast-ingest firms`
  divide automáticamente rangos más largos en múltiples consultas de
  ≤5 días.
- Límite de uso: **5000 transacciones / ventana de 10 minutos** por
  MAP_KEY (hay un endpoint `mapserver/mapkey_status/?MAP_KEY=...` para
  consultar el uso actual; este proyecto no lo llama todavía — el
  cliente aplica un rate limit local conservador entre requests en su
  lugar).
- El comportamiento documentado ante errores (MAP_KEY inválido,
  solicitud malformada) es débil — no hay códigos de estado oficiales
  documentados. El cliente (`ingestion/firms/client.py`) reintenta
  429/5xx con backoff exponencial, y trata cualquier respuesta 200 que
  no sea CSV real como un error no reintentable.

**Limitaciones conocidas de la fuente (no del cliente):**
- **Resolución 375 m** (VIIRS) — no puede resolver ignición puntual con
  precisión menor a eso; múltiples focos cercanos pueden fusionarse en
  una sola detección o viceversa.
- **Falsos positivos por reflejo solar** ("sun glint") sobre cuerpos de
  agua y superficies reflectantes, especialmente en ángulos de
  observación bajos — puede producir detecciones espurias cerca de
  lagos/embalses/costa que no son incendios reales.
- **Falsos negativos por cobertura de nubes/humo denso** — el sensor no
  detecta a través de nubes; un incendio activo bajo una columna de
  humo densa puede no aparecer en un pase satelital.
- **Frecuencia de paso limitada**: los satélites polares (VIIRS/MODIS)
  pasan sobre cualquier punto ~1-4 veces al día — un incendio que se
  inicia y extingue entre pasadas puede no quedar registrado.
- **`confidence` no es comparable entre sensores**: VIIRS usa categorías
  (`l`/`n`/`h` o `low`/`nominal`/`high` según versión de producto),
  MODIS usa un porcentaje numérico. Este proyecto guarda el valor tal
  cual (`shared.schemas.FireDetection.confidence: str`), sin intentar
  unificar la escala.
```

- [ ] **Step 3: Add the new-dependency justifications to `docs/decisions.md`**

Append:

```markdown
## Dependencias añadidas en la implementación de ingestion/firms/

- **`pyarrow`** (en `ingestion`): necesaria para escribir Parquet crudo
  (Task 2 del pedido de ingestion/firms). Se eligió `pyarrow` puro
  (parseo manual del CSV con el módulo estándar `csv` + `pyarrow.Table`)
  en vez de `pandas` + un engine de Parquet, para no añadir una
  dependencia pesada que `ingestion` no necesita para nada más — el
  parseo real a un esquema tipado ocurre después, en
  `ingestion/firms/parser.py`, usando `shared.schemas.FireDetection`.
- **`pydantic`** (en `shared`, ahora explícito): ya estaba resuelto de
  forma transitiva vía `pydantic-settings`, y `shared/config.py` ya lo
  importaba directamente; esto solo hace explícita una dependencia que
  ya existía en la práctica, para `shared/schemas.py`.
```

- [ ] **Step 4: Full workspace re-verification**

Run: `uv run --package shared pytest shared/tests -v`
Run: `uv run --package ingestion pytest ingestion/tests -v`
Run: `uv run ruff check .`
Run: `uv run mypy --strict shared/src features/src`
Run: `uv run mypy --strict ingestion/src/ingestion/firms ingestion/src/ingestion/cli.py`

Expected: all green, matching this task's acceptance criteria verbatim.

- [ ] **Step 5: Commit**

```bash
git add docs/data-sources.md docs/decisions.md
git commit -m "docs: add FIRMS data-sources section and dependency justifications"
```

---

## Self-Review Notes

- **Spec coverage:** user's 5 numbered tasks map onto: 1→Task 2 (client), 2→Task 3 (storage), 3→Task 1+4 (schema+parser), 4→Task 5 (CLI), 5→Tasks 2-5 (fixtures + retry/empty-response tests threaded through each task's own test file, per TDD, rather than one giant end-of-plan test file). Acceptance criteria (mypy --strict/ruff on `ingestion/firms/`, tests green with no network, `docs/data-sources.md`) → Task 6.
- **Placeholder scan:** every step has runnable code; the two "if this assumption is wrong, ledger a ruling and adjust" notes (Task 2 Step 4, Task 5 Step 4) are flagged uncertainties about undocumented API/library behavior, not missing implementation — each names exactly what to check and what the fallback fix is.
- **Type consistency:** `FirmsClient.fetch_area_csv`/`fetch_range` signatures (Task 2) are used identically in Task 5's CLI; `save_raw_response`'s parameter names (Task 3) match the CLI's call; `parse_csv_to_detections` (Task 4) returns `list[FireDetection]` matching Task 1's model construction signature (`latitude`, `longitude`, `detected_at`, `frp`, `confidence`, `satellite`, `instrument` — same names throughout).
- **Review Focus:** all five items map to an owning test — multi-day chunking (Task 2's `_chunk_date_range`/`fetch_range` tests), non-CSV 200 body (Task 2's `error_body` test), genuine empty result (Task 3 and Task 4's `empty` tests), retry-exhaustion-raises (Task 2's 500 test), `acq_time` zero-padding (Task 4's `combines_date_and_zero_padded_time` test using `"0005"` specifically).
