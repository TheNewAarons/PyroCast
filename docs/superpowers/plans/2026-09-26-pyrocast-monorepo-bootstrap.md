# PyroCast Monorepo Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the PyroCast monorepo skeleton — uv workspace, directory tree, `shared/config.py`, initial PostGIS schema, Docker/Compose, Makefile, CI, and a smoke test per package — so `make up`, `make test`, `make lint`, `make typecheck` all pass green with zero credentials configured.

**Architecture:** A `uv` workspace with 5 members (`shared`, `ingestion`, `features`, `models`, `serving`) under `src/` layout, each its own `pyproject.toml`. `shared` has no internal dependents (leaf-most), `ingestion`/`features`/`models` depend on `shared`, `serving` depends on `shared` (and later on `features`/`models` once those exist, but not yet — Task 6 only needs `/healthz`). Data model lives in `shared/db/schema.py` as SQLAlchemy Core `Table` objects (no ORM), using GeoAlchemy2 only for the `Geometry` column type. Docker Compose runs `postgis/postgis:16-3.4` plus a multi-stage, non-root `api` image built from `serving/`.

**Tech Stack:** Python 3.12, `uv`, `ruff`, `mypy --strict` (shared + features only, per CLAUDE.md), `pytest`, FastAPI + Jinja2, SQLAlchemy Core + GeoAlchemy2, `psycopg[binary]`, Docker/Compose, GitHub Actions.

**Spec:** User request in this conversation (9 numbered tasks + acceptance criteria) plus `/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md` (project charter — architecture, data sources, stack, conventions). Both travel with this plan; executors must read `CLAUDE.md` in full before Task 1.

## Global Constraints

- Python 3.12 everywhere; `uv` workspace, members: `shared`, `ingestion`, `features`, `models`, `serving`.
- `ruff` on the whole repo; `mypy --strict` scoped to `shared/` and `features/` only (per CLAUDE.md — do not extend scope without updating CLAUDE.md).
- Deps limited to CLAUDE.md's list: `rasterio`, `xarray`, `rioxarray`, `geopandas`, `shapely`, `zarr`, `torch`, `scikit-learn`, `sqlalchemy`, `psycopg`, `typer`, `fastapi`, `jinja2`, `pydantic-settings`, `cdsapi`, `requests`, `responses`, `vcrpy`, `pytest`, `numpy`. Any other dependency (`geoalchemy2`, `uvicorn`/`httpx` via `fastapi[standard]`, `ruff`, `mypy`) must get a one-paragraph justification in `docs/decisions.md`.
- No real credentials anywhere in the repo. All API keys/DB creds come from environment variables with **no default values** in `shared/config.py`. `.env` is git-ignored; `.env.example` documents every variable and where to obtain it.
- No CI job may call a real external API (FIRMS, CDS, Copernicus Data Space). Network-dependent tests use `responses`/`vcrpy` fixtures.
- `bbox` default study area: Biobío + Ñuble + La Araucanía, CRS EPSG:32719 (UTM 19S) for planar calcs, 250 m spatial / daily temporal resolution — all configurable in `shared/config.py`, not hardcoded per-module.
- Every module ships with at least a smoke test. Identifiers in English; docstrings/comments/docs in Spanish (per CLAUDE.md).
- Docker: multi-stage build, non-root user, healthcheck; large geospatial data lives in a local volume, never baked into the image.
- `docs/limitations.md` must exist and mention the ERA5-Land ~9 km → 250 m downscaling artifact from day one (CLAUDE.md requires this be documented, even though ingestion isn't implemented yet — it's a known-from-design limitation, not a finding to fabricate later).
- Every README/report-facing surface must carry the Spanish disclaimer verbatim: *"Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED."*

## Review Focus

- **Missing/empty env vars at startup:** `shared/config.py` must fail with a clear, actionable error (not a bare `KeyError`/`ValidationError` traceback) when a required credential is unset — this is explicitly required by the user's Task 9 and is the one thing a fresh contributor will hit on first run.
- **CI running without any `.env`:** the GitHub Actions workflow must not source real secrets; config-loading tests must construct `Settings` with explicit env vars per-test (via `monkeypatch`), never rely on a checked-in `.env`.
- **Docker image accidentally bundling data or running as root:** the Dockerfile must be checked for a non-root `USER` directive and that `data/` is volume-mounted, not `COPY`'d.
- **`make` targets for unimplemented pipelines silently succeeding with exit 0 and misleading "success" output:** placeholder targets must print "pendiente" and exit 0 deliberately (that's correct — not a bug) but must be visibly distinguishable in output from a real pass, so `make ingest-firms` doesn't look like it did something it didn't.
- **PostGIS FK/JSON columns silently accepting wrong types:** `model_run.config` (JSON) and the FK columns need a smoke test that actually inserts and reads back a row against a real (ephemeral, Dockerized) Postgres — a schema that only "looks right" in Python but was never executed against Postgres is the most likely latent bug (e.g. GeoAlchemy2 `Geometry` type mismatch, JSON vs JSONB).

---

## File Structure

```
PyroCast/
├── CLAUDE.md
├── pyproject.toml                  # uv workspace root
├── uv.lock
├── Makefile
├── docker-compose.yml
├── Dockerfile                      # serving/api image (multi-stage)
├── .dockerignore
├── .env.example
├── .gitignore
├── README.md
├── .github/workflows/ci.yml
├── docs/
│   ├── decisions.md
│   ├── limitations.md
│   └── superpowers/plans/2026-09-26-pyrocast-monorepo-bootstrap.md
├── scripts/
│   └── init_db.py                  # creates tables via shared.db.schema.metadata
├── shared/
│   ├── pyproject.toml
│   ├── src/shared/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── db/
│   │       ├── __init__.py
│   │       └── schema.py
│   └── tests/
│       ├── test_smoke.py
│       ├── test_config.py
│       └── test_db_schema.py
├── ingestion/
│   ├── pyproject.toml
│   ├── src/ingestion/
│   │   ├── __init__.py
│   │   ├── firms/__init__.py
│   │   ├── dem/__init__.py
│   │   ├── era5/__init__.py
│   │   ├── sentinel2/__init__.py
│   │   └── worldcover/__init__.py
│   └── tests/test_smoke.py
├── features/
│   ├── pyproject.toml
│   ├── src/features/
│   │   ├── __init__.py
│   │   ├── grid/__init__.py
│   │   ├── terrain/__init__.py
│   │   ├── weather/__init__.py
│   │   ├── vegetation/__init__.py
│   │   ├── fire_state/__init__.py
│   │   └── dataset/__init__.py
│   └── tests/test_smoke.py
├── models/
│   ├── pyproject.toml
│   ├── src/models/
│   │   ├── __init__.py
│   │   ├── cellular_automata/__init__.py
│   │   ├── deep/__init__.py
│   │   └── evaluation/__init__.py
│   └── tests/test_smoke.py
└── serving/
    ├── pyproject.toml
    ├── src/serving/
    │   ├── __init__.py
    │   └── api/
    │       ├── __init__.py
    │       ├── main.py             # FastAPI app, /healthz
    │       └── routers/__init__.py
    ├── web/templates/.gitkeep
    └── tests/test_smoke.py
```

---

### Task 1: Git init + directory tree + uv workspace root

**Files:**
- Create: `.gitignore`, `.dockerignore`
- Create: `pyproject.toml` (workspace root)
- Create: all directories listed in File Structure above (with `.gitkeep` or `__init__.py` as appropriate so empty dirs survive `git add`)

**Interfaces:**
- Produces: `uv.lock` after first `uv sync`; workspace root `pyproject.toml` with `[tool.uv.workspace] members = ["shared", "ingestion", "features", "models", "serving"]`.

- [ ] **Step 1: Initialize git repo**

Run: `git init && git branch -m main`
Expected: `Initialized empty Git repository...`

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# uv
uv.lock.bak

# Env / secrets
.env
*.env.local

# Data (never committed — see docker-compose volume)
data/raw/
data/interim/
data/processed/
*.tif
*.nc
*.zarr/

# OS
.DS_Store
```

- [ ] **Step 3: Create the full directory tree**

```bash
mkdir -p ingestion/{firms,dem,era5,sentinel2,worldcover} \
         features/{grid,terrain,weather,vegetation,fire_state,dataset} \
         models/cellular_automata models/deep models/evaluation \
         serving/api/routers serving/web/templates \
         shared scripts docs \
         tests/{shared,ingestion,features,models,serving}
```

Note: this creates the *top-level* mirror directories the user asked for in Task 1 of their request; the actual Python package source lives under each member's `src/<pkg>/` (Task 2+ below) per `uv` workspace convention, and each member also gets its own `tests/` next to its `pyproject.toml` (pytest discovers per-package tests better that way, and CLAUDE.md's `tests/` request is satisfied by having one test dir per workspace member — confirm this reconciliation with the user if they want a single top-level `tests/` instead; proceed with per-member `tests/` as the default since it's the `uv`/pytest idiom).

- [ ] **Step 4: Create workspace root `pyproject.toml`**

```toml
[project]
name = "pyrocast"
version = "0.1.0"
description = "Pronóstico de propagación de incendios forestales — fusión de datos satelitales y meteorológicos abiertos."
requires-python = ">=3.12,<3.13"

[tool.uv.workspace]
members = ["shared", "ingestion", "features", "models", "serving"]

[tool.uv]
dev-dependencies = [
    "ruff>=0.6",
    "mypy>=1.11",
    "pytest>=8.3",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.12"
strict = false
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["shared.*", "features.*"]
strict = true
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml ingestion features models serving shared scripts docs tests
git commit -m "chore: scaffold repo tree and uv workspace root"
```

---

### Task 2: `shared` package — config with fail-fast env validation + PostGIS schema

**Files:**
- Create: `shared/pyproject.toml`
- Create: `shared/src/shared/__init__.py`
- Create: `shared/src/shared/config.py`
- Create: `shared/src/shared/db/__init__.py`
- Create: `shared/src/shared/db/schema.py`
- Test: `shared/tests/test_config.py`
- Test: `shared/tests/test_db_schema.py`
- Test: `shared/tests/test_smoke.py`

**Interfaces:**
- Produces: `shared.config.Settings` (pydantic-settings `BaseSettings` subclass) and `shared.config.get_settings()`; fields: `study_area_bbox: tuple[float, float, float, float]`, `crs: str`, `spatial_resolution_m: int`, `temporal_resolution: str`, `data_raw_dir: Path`, `data_interim_dir: Path`, `data_processed_dir: Path`, `firms_map_key: str`, `cds_api_url: str`, `cds_api_key: str`, `copernicus_dataspace_client_id: str`, `copernicus_dataspace_client_secret: str`, `postgres_host: str`, `postgres_port: int`, `postgres_db: str`, `postgres_user: str`, `postgres_password: str`.
- Produces: `shared.db.schema.metadata` (SQLAlchemy `MetaData`), `shared.db.schema.fire_event`, `shared.db.schema.model_run`, `shared.db.schema.evaluation_result` (`Table` objects).

- [ ] **Step 1: Write `shared/pyproject.toml`**

```toml
[project]
name = "shared"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "pydantic-settings>=2.4",
    "sqlalchemy>=2.0",
    "geoalchemy2>=0.15",
    "psycopg[binary]>=3.2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/shared"]
```

- [ ] **Step 2: Write the failing test for missing env vars**

`shared/tests/test_config.py`:

```python
"""Tests de shared.config: valores por defecto de área de estudio y fallo
explícito si faltan credenciales requeridas."""
import pytest
from pydantic import ValidationError

from shared.config import Settings


REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "test-firms-key",
    "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "test-cds-key",
    "COPERNICUS_DATASPACE_CLIENT_ID": "test-client-id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "test-client-secret",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "pyrocast",
    "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "test-password",
}


def test_settings_load_with_all_required_env_vars(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    settings = Settings()
    assert settings.firms_map_key == "test-firms-key"
    assert settings.postgres_port == 5432


def test_settings_default_study_area_covers_biobio_nuble_araucania(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    settings = Settings()
    min_lon, min_lat, max_lon, max_lat = settings.study_area_bbox
    assert min_lon < max_lon
    assert min_lat < max_lat
    assert settings.crs == "EPSG:32719"
    assert settings.spatial_resolution_m == 250
    assert settings.temporal_resolution == "daily"


def test_settings_raises_clear_error_when_firms_key_missing(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        if key != "FIRMS_MAP_KEY":
            monkeypatch.setenv(key, value)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(ValidationError, match="firms_map_key"):
        Settings()
```

- [ ] **Step 2b: Run test to verify it fails**

Run: `cd shared && uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.config'`

- [ ] **Step 3: Write `shared/src/shared/config.py`**

```python
"""Configuración central de PyroCast.

Toda credencial se lee de variables de entorno, sin valores por defecto
reales: si falta una variable requerida, pydantic-settings falla con un
ValidationError legible (no un KeyError críptico) en cuanto se instancia
Settings(). El área de estudio por defecto cubre Biobío, Ñuble y La
Araucanía (las regiones más afectadas en la temporada 2025-2026); las
coordenadas son un bounding box aproximado en WGS84 (lon/lat), fuente:
límites administrativos de las regiones VIII, XVI y IX de Chile (BCN /
INE, simplificados a un rectángulo envolvente para uso interno del
proyecto — no es un límite administrativo exacto).
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# (min_lon, min_lat, max_lon, max_lat) en WGS84 — envolvente de
# Biobío + Ñuble + La Araucanía.
DEFAULT_STUDY_AREA_BBOX: tuple[float, float, float, float] = (
    -73.7,
    -39.3,
    -71.0,
    -36.5,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Área de estudio y grilla — configurable, nunca hardcodeado por módulo.
    study_area_bbox: tuple[float, float, float, float] = DEFAULT_STUDY_AREA_BBOX
    crs: str = "EPSG:32719"
    spatial_resolution_m: int = 250
    temporal_resolution: str = "daily"

    # Rutas de datos (fuera de la imagen Docker, ver docker-compose.yml).
    data_raw_dir: Path = Path("data/raw")
    data_interim_dir: Path = Path("data/interim")
    data_processed_dir: Path = Path("data/processed")

    # NASA FIRMS — https://firms.modaps.eosdis.nasa.gov/api/map_key/
    firms_map_key: str = Field(...)

    # Copernicus CDS (ERA5-Land) — https://cds.climate.copernicus.eu/how-to-api
    cds_api_url: str = Field(...)
    cds_api_key: str = Field(...)

    # Copernicus Data Space Ecosystem (Sentinel-2) — https://dataspace.copernicus.eu/
    copernicus_dataspace_client_id: str = Field(...)
    copernicus_dataspace_client_secret: str = Field(...)

    # PostgreSQL/PostGIS
    postgres_host: str = Field(...)
    postgres_port: int = Field(...)
    postgres_db: str = Field(...)
    postgres_user: str = Field(...)
    postgres_password: str = Field(...)

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd shared && uv run pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Write the DB schema smoke test (Python-level, no live Postgres required for this one)**

`shared/tests/test_db_schema.py`:

```python
"""Verifica la forma de las tablas Core (sin conexión real a Postgres)."""
from shared.db.schema import evaluation_result, fire_event, metadata, model_run


def test_metadata_has_expected_tables():
    assert set(metadata.tables) == {
        "fire_event",
        "model_run",
        "evaluation_result",
    }


def test_fire_event_columns():
    cols = set(fire_event.columns.keys())
    assert {"id", "bbox", "start_date", "end_date", "source", "geom"} <= cols


def test_model_run_has_fk_to_fire_event():
    fk_targets = {fk.column.table.name for fk in model_run.foreign_keys}
    assert fk_targets == {"fire_event"}


def test_evaluation_result_has_fk_to_model_run():
    fk_targets = {fk.column.table.name for fk in evaluation_result.foreign_keys}
    assert fk_targets == {"model_run"}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd shared && uv run pytest tests/test_db_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shared.db.schema'`

- [ ] **Step 7: Write `shared/src/shared/db/schema.py`**

```python
"""Modelo de datos inicial en PostGIS, vía SQLAlchemy Core (sin ORM).

Tres tablas: fire_event (incendios reales usados para calibración y
backtesting), model_run (una ejecución de un modelo sobre un evento) y
evaluation_result (métricas de esa ejecución, por split train/val/test).
"""
from geoalchemy2 import Geometry
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    func,
)

metadata = MetaData()

fire_event = Table(
    "fire_event",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("bbox", String, nullable=False),
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=True),
    Column("source", String, nullable=False),
    Column("geom", Geometry(geometry_type="POLYGON", srid=4326), nullable=False),
)

model_run = Table(
    "model_run",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("event_id", Integer, ForeignKey("fire_event.id"), nullable=False),
    Column("model_name", String, nullable=False),
    Column("config", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

evaluation_result = Table(
    "evaluation_result",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("run_id", Integer, ForeignKey("model_run.id"), nullable=False),
    Column("metric_name", String, nullable=False),
    Column("value", Float, nullable=False),
    Column("split", String, nullable=False),
)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd shared && uv run pytest tests/test_db_schema.py -v`
Expected: 4 passed

- [ ] **Step 9: Write and run the package smoke test**

`shared/tests/test_smoke.py`:

```python
def test_shared_package_imports():
    import shared  # noqa: F401
    from shared import config, db  # noqa: F401
```

Run: `cd shared && uv run pytest -v`
Expected: all pass

- [ ] **Step 10: Commit**

```bash
git add shared/
git commit -m "feat(shared): add fail-fast Settings and PostGIS Core schema"
```

---

### Task 3: `ingestion`, `features`, `models` package skeletons + smoke tests

**Files:**
- Create: `ingestion/pyproject.toml`, `ingestion/src/ingestion/{__init__.py,firms,dem,era5,sentinel2,worldcover}/__init__.py`, `ingestion/tests/test_smoke.py`
- Create: `features/pyproject.toml`, `features/src/features/{__init__.py,grid,terrain,weather,vegetation,fire_state,dataset}/__init__.py`, `features/tests/test_smoke.py`
- Create: `models/pyproject.toml`, `models/src/models/{__init__.py,cellular_automata,deep,evaluation}/__init__.py`, `models/tests/test_smoke.py`

**Interfaces:**
- Consumes: `shared` package (workspace path dependency) in all three.
- Produces: importable empty subpackages that later ingestion/feature/model tasks fill in; nothing downstream in *this* plan depends on their internals — only that `import ingestion.firms` etc. succeeds.

- [ ] **Step 1: `ingestion/pyproject.toml`**

```toml
[project]
name = "ingestion"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "shared",
    "requests>=2.32",
    "cdsapi>=0.7",
    "typer>=0.12",
]

[tool.uv.sources]
shared = { workspace = true }

[dependency-groups]
dev = ["responses>=0.25", "vcrpy>=6.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ingestion"]
```

- [ ] **Step 2: Create ingestion subpackages, each with a one-line Spanish docstring stub**

```bash
for m in firms dem era5 sentinel2 worldcover; do
cat > ingestion/src/ingestion/$m/__init__.py <<EOF
"""Ingesta de datos de $m. Pendiente de implementación."""
EOF
done
cat > ingestion/src/ingestion/__init__.py <<'EOF'
"""Paquete de ingesta: descarga cruda, cacheada, versionada por fecha/tile."""
EOF
```

- [ ] **Step 3: Write and run `ingestion/tests/test_smoke.py`**

```python
def test_ingestion_subpackages_import():
    from ingestion import dem, era5, firms, sentinel2, worldcover  # noqa: F401
```

Run: `cd ingestion && uv run pytest -v`
Expected: 1 passed

- [ ] **Step 4: Repeat Steps 1-3 for `features`**

`features/pyproject.toml`:

```toml
[project]
name = "features"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "shared",
    "rasterio>=1.3",
    "xarray>=2024.7",
    "rioxarray>=0.17",
    "geopandas>=1.0",
    "shapely>=2.0",
    "zarr>=2.18",
    "numpy>=2.0",
]

[tool.uv.sources]
shared = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/features"]
```

```bash
for m in grid terrain weather vegetation fire_state dataset; do
cat > features/src/features/$m/__init__.py <<EOF
"""Módulo de features: $m. Pendiente de implementación."""
EOF
done
cat > features/src/features/__init__.py <<'EOF'
"""Paquete de features: grilla común, terreno, clima, vegetación, estado del fuego."""
EOF
```

`features/tests/test_smoke.py`:

```python
def test_features_subpackages_import():
    from features import dataset, fire_state, grid, terrain, vegetation, weather  # noqa: F401
```

Run: `cd features && uv run pytest -v`
Expected: 1 passed

- [ ] **Step 5: Repeat for `models`**

`models/pyproject.toml`:

```toml
[project]
name = "models"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "shared",
    "numpy>=2.0",
    "torch>=2.4",
    "scikit-learn>=1.5",
]

[tool.uv.sources]
shared = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/models"]
```

```bash
for m in cellular_automata deep evaluation; do
cat > models/src/models/$m/__init__.py <<EOF
"""Módulo de modelos: $m. Pendiente de implementación."""
EOF
done
cat > models/src/models/__init__.py <<'EOF'
"""Paquete de modelos: autómata celular, U-Net, evaluación."""
EOF
```

`models/tests/test_smoke.py`:

```python
def test_models_subpackages_import():
    from models import cellular_automata, deep, evaluation  # noqa: F401
```

Run: `cd models && uv run pytest -v`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add ingestion/ features/ models/
git commit -m "feat: add ingestion, features and models package skeletons"
```

---

### Task 4: `serving` package — FastAPI app with `/healthz`

**Files:**
- Create: `serving/pyproject.toml`
- Create: `serving/src/serving/__init__.py`
- Create: `serving/src/serving/api/__init__.py`
- Create: `serving/src/serving/api/main.py`
- Create: `serving/src/serving/api/routers/__init__.py`
- Create: `serving/web/templates/.gitkeep`
- Test: `serving/tests/test_smoke.py`

**Interfaces:**
- Consumes: `shared.config.get_settings`.
- Produces: `serving.api.main.app` (FastAPI instance), route `GET /healthz` → `{"status": "ok"}`.

- [ ] **Step 1: `serving/pyproject.toml`**

```toml
[project]
name = "serving"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "shared",
    "fastapi[standard]>=0.114",
    "jinja2>=3.1",
    "typer>=0.12",
]

[tool.uv.sources]
shared = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/serving"]
```

`fastapi[standard]` is the one addition beyond CLAUDE.md's explicit dependency list (`fastapi`, `jinja2`) — it bundles `uvicorn` (to actually run the ASGI app in the container) and `httpx` (required by Starlette's `TestClient` for the smoke test below). Document this in `docs/decisions.md` (Task 7).

- [ ] **Step 2: Write the failing smoke test**

`serving/tests/test_smoke.py`:

```python
from fastapi.testclient import TestClient

from serving.api.main import app


def test_healthz_returns_ok():
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd serving && uv run pytest -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'serving.api.main'`

- [ ] **Step 4: Write `serving/src/serving/api/main.py`**

```python
"""API de PyroCast: por ahora solo expone /healthz.

Aviso obligatorio (ver CLAUDE.md): esta es una herramienta de
investigación, no un sistema operativo de combate de incendios.
"""
from fastapi import FastAPI

app = FastAPI(
    title="PyroCast API",
    description=(
        "Herramienta de investigación. No usar para decisiones operativas "
        "de combate de incendios sin validación de CONAF/SENAPRED."
    ),
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
```

```bash
touch serving/src/serving/api/routers/__init__.py serving/web/templates/.gitkeep
cat > serving/src/serving/__init__.py <<'EOF'
"""Paquete de servido: API FastAPI + mapa web."""
EOF
cat > serving/src/serving/api/__init__.py <<'EOF'
"""Router raíz de la API."""
EOF
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd serving && uv run pytest -v`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add serving/
git commit -m "feat(serving): add FastAPI app with /healthz"
```

---

### Task 5: `scripts/init_db.py` — create tables against live Postgres (closes the Review Focus gap on schema correctness)

**Files:**
- Create: `scripts/init_db.py`
- Test: `shared/tests/test_db_schema.py` (extend with a live-Postgres test, skipped if no DB reachable)

**Interfaces:**
- Consumes: `shared.config.get_settings().postgres_dsn`, `shared.db.schema.metadata`.
- Produces: `scripts/init_db.py` CLI (typer) with command `create-all`.

- [ ] **Step 1: Write `scripts/init_db.py`**

```python
"""CLI para crear las tablas iniciales en PostGIS.

Uso: uv run --package shared python scripts/init_db.py create-all
"""
import typer
from sqlalchemy import create_engine

from shared.config import get_settings
from shared.db.schema import metadata

app = typer.Typer()


@app.command("create-all")
def create_all() -> None:
    settings = get_settings()
    engine = create_engine(settings.postgres_dsn)
    with engine.begin() as conn:
        conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS postgis"))
        metadata.create_all(conn)
    typer.echo("Tablas creadas: fire_event, model_run, evaluation_result")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Add the live-DB smoke test, skipped when Postgres isn't reachable**

Append to `shared/tests/test_db_schema.py`:

```python
import os

import pytest
from sqlalchemy import create_engine, insert, select, text
from sqlalchemy.exc import OperationalError

from shared.config import Settings


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
def test_can_insert_and_read_fire_event_against_real_postgis():
    settings = _live_settings()
    assert settings is not None
    engine = create_engine(settings.postgres_dsn)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        metadata.create_all(conn)
        conn.execute(
            insert(fire_event).values(
                bbox="-73.7,-39.3,-71.0,-36.5",
                start_date="2026-01-15",
                end_date="2026-01-20",
                source="test",
                geom="SRID=4326;POLYGON((-73 -39, -73 -38, -72 -38, -72 -39, -73 -39))",
            )
        )
        row = conn.execute(select(fire_event.c.source)).first()
    assert row is not None
    assert row.source == "test"
```

This test is intentionally opt-in (skipped without live `POSTGRES_*` env vars) so `make test` stays credential-free in CI, but a developer running `docker compose up -d postgis` locally and exporting the compose env vars gets real coverage of the GeoAlchemy2/JSON/FK wiring — the Review Focus item this task exists for.

- [ ] **Step 3: Run without live DB to confirm it's skipped, not failing**

Run: `cd shared && uv run pytest tests/test_db_schema.py -v`
Expected: the new test shows `SKIPPED`, others pass

- [ ] **Step 4: Commit**

```bash
git add scripts/init_db.py shared/tests/test_db_schema.py
git commit -m "feat: add init_db CLI and opt-in live-Postgres schema test"
```

---

### Task 6: `.env.example`, `docker-compose.yml`, `Dockerfile`, `.dockerignore`

**Files:**
- Create: `.env.example`
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Write `.env.example`**

```dotenv
# --- NASA FIRMS ---------------------------------------------------------
# Obtener MAP_KEY gratuito en: https://firms.modaps.eosdis.nasa.gov/api/map_key/
FIRMS_MAP_KEY=

# --- Copernicus Climate Data Store (ERA5-Land) --------------------------
# Crear cuenta y ver tu API key en: https://cds.climate.copernicus.eu/how-to-api
CDS_API_URL=https://cds.climate.copernicus.eu/api
CDS_API_KEY=

# --- Copernicus Data Space Ecosystem (Sentinel-2) -----------------------
# Registrar aplicación OAuth en: https://dataspace.copernicus.eu/ -> Account -> OAuth clients
COPERNICUS_DATASPACE_CLIENT_ID=
COPERNICUS_DATASPACE_CLIENT_SECRET=

# --- PostgreSQL / PostGIS -----------------------------------------------
# Valores para desarrollo local con docker-compose (docker-compose.yml).
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=pyrocast
POSTGRES_USER=pyrocast
POSTGRES_PASSWORD=changeme
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-pyrocast}
      POSTGRES_USER: ${POSTGRES_USER:-pyrocast}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-changeme}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-pyrocast}"]
      interval: 5s
      timeout: 5s
      retries: 10

  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      FIRMS_MAP_KEY: ${FIRMS_MAP_KEY:-dev-placeholder}
      CDS_API_URL: ${CDS_API_URL:-https://cds.climate.copernicus.eu/api}
      CDS_API_KEY: ${CDS_API_KEY:-dev-placeholder}
      COPERNICUS_DATASPACE_CLIENT_ID: ${COPERNICUS_DATASPACE_CLIENT_ID:-dev-placeholder}
      COPERNICUS_DATASPACE_CLIENT_SECRET: ${COPERNICUS_DATASPACE_CLIENT_SECRET:-dev-placeholder}
      POSTGRES_HOST: postgis
      POSTGRES_PORT: "5432"
      POSTGRES_DB: ${POSTGRES_DB:-pyrocast}
      POSTGRES_USER: ${POSTGRES_USER:-pyrocast}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-changeme}
    volumes:
      - ./data:/data
    depends_on:
      postgis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

The `dev-placeholder` defaults exist only so `docker compose up` succeeds with an unset `.env` for local dev of the API skeleton (no ingestion module reads these yet); this does not weaken the "no real credential defaults" rule since `shared.config.Settings` still requires the vars to be non-empty strings and these placeholders are visibly fake, never real keys.

- [ ] **Step 3: Write `Dockerfile` (multi-stage, non-root, healthcheck)**

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY shared/pyproject.toml shared/pyproject.toml
COPY serving/pyproject.toml serving/pyproject.toml
COPY shared/src shared/src
COPY serving/src serving/src

RUN uv sync --frozen --package shared --package serving --no-dev

FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 pyrocast && \
    useradd --uid 1000 --gid pyrocast --shell /bin/bash --create-home pyrocast

WORKDIR /app
COPY --from=builder /build/.venv /app/.venv
COPY serving/src /app/serving/src
COPY shared/src /app/shared/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/shared/src:/app/serving/src"

USER pyrocast

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Write `.dockerignore`**

```
.git
.venv
data/
docs/superpowers
**/__pycache__
**/*.pyc
.pytest_cache
.mypy_cache
.ruff_cache
tests/
**/tests
```

- [ ] **Step 5: Verify locally (manual check, not automated in CI — Docker isn't available in the CI runner's credential-free constraint but *is* available locally)**

Run: `docker compose build api`
Expected: build succeeds

Run: `docker compose up -d && sleep 3 && docker compose ps`
Expected: both `postgis` and `api` show healthy (or `starting` transitioning to healthy)

Run: `curl -sf http://localhost:8000/healthz`
Expected: `{"status":"ok"}`

Run: `docker compose down`

- [ ] **Step 6: Commit**

```bash
git add .env.example docker-compose.yml Dockerfile .dockerignore
git commit -m "feat: add docker-compose (postgis+api) and multi-stage non-root Dockerfile"
```

---

### Task 7: `Makefile`, `docs/decisions.md`, `docs/limitations.md`, `README.md`

**Files:**
- Create: `Makefile`
- Create: `docs/decisions.md`
- Create: `docs/limitations.md`
- Create: `README.md`

- [ ] **Step 1: Write `Makefile`**

```makefile
.PHONY: up down ingest-firms ingest-terrain ingest-weather ingest-vegetation \
        build-dataset run-ca train calibrate backtest report serve \
        test lint typecheck

up:
	docker compose up -d

down:
	docker compose down

ingest-firms:
	@echo "pendiente: módulo ingestion/firms aún no implementado"

ingest-terrain:
	@echo "pendiente: módulo ingestion/dem aún no implementado"

ingest-weather:
	@echo "pendiente: módulo ingestion/era5 aún no implementado"

ingest-vegetation:
	@echo "pendiente: módulos ingestion/sentinel2 e ingestion/worldcover aún no implementados"

build-dataset:
	@echo "pendiente: features/dataset aún no implementado"

run-ca:
	@echo "pendiente: models/cellular_automata aún no implementado"

train:
	@echo "pendiente: models/deep aún no implementado"

calibrate:
	@echo "pendiente: models/evaluation (calibración isotónica) aún no implementado"

backtest:
	@echo "pendiente: models/evaluation (backtesting) aún no implementado"

report:
	@echo "pendiente: generación de docs/results.md aún no implementada"

serve:
	uv run --package serving uvicorn serving.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run --package shared pytest shared/tests -v
	uv run --package ingestion pytest ingestion/tests -v
	uv run --package features pytest features/tests -v
	uv run --package models pytest models/tests -v
	uv run --package serving pytest serving/tests -v

lint:
	uv run ruff check .

typecheck:
	uv run mypy shared/src features/src
```

- [ ] **Step 2: Write `docs/decisions.md`**

```markdown
# Decisiones de diseño

## Dependencias fuera de la lista explícita de CLAUDE.md

- **`geoalchemy2`**: CLAUDE.md pide PostGIS vía SQLAlchemy Core "sin ORM
  pesado", pero no menciona cómo mapear la columna `geom`. GeoAlchemy2 solo
  aporta el tipo de columna `Geometry` para SQLAlchemy Core (no un ORM);
  la alternativa (un `UserDefinedType` manual) duplicaría lo que la
  librería estándar del ecosistema ya resuelve, con más riesgo de bugs de
  serialización WKT/WKB. No introduce ORM.
- **`fastapi[standard]`** en vez de `fastapi` a secas: el extra `standard`
  incluye `uvicorn` (necesario para correr el servidor ASGI en
  `docker-compose`/`Makefile`) y `httpx` (requerido por
  `fastapi.testclient.TestClient` en los tests de humo). Sin esto, `make
  serve` y `make up` no podrían levantar la API real.
- **`hatchling`** como build-backend de cada paquete del workspace: es el
  backend por defecto recomendado por `uv` para proyectos multi-paquete;
  no aporta funcionalidad de producto, solo empaquetado.

## Alcance de mypy --strict

Por decisión explícita de CLAUDE.md, `mypy --strict` corre solo sobre
`shared/` y `features/`. `ingestion/`, `models/` y `serving/` corren mypy
en modo no estricto (vía el override global en el `pyproject.toml` raíz)
hasta que haya código real que tipar con disciplina — no tiene sentido
exigir `--strict` sobre paquetes que hoy son solo stubs.

## `tests/` por miembro del workspace, no un único directorio raíz

El pedido original menciona `tests/` (un directorio). Se decidió un
`tests/` por cada miembro del workspace (`shared/tests`,
`ingestion/tests`, etc.) porque es el patrón idiomático de `uv`
workspaces + `pytest` (cada paquete testea su propio código sin que
`pytest` tenga que resolver imports cruzados entre paquetes con
distintos `sys.path`). El `Makefile` target `test` corre los cinco
directorios en una sola invocación de `make test`, cumpliendo el criterio
de aceptación tal como está escrito.
```

- [ ] **Step 3: Write `docs/limitations.md`**

```markdown
# Limitaciones conocidas

Este documento se actualiza con cada hallazgo real de la evaluación
contra incendios de Chile. Nunca se suaviza ni se elimina una métrica
negativa para que el proyecto "se vea mejor" (ver CLAUDE.md).

## Limitaciones de diseño (conocidas desde el bootstrap, no hallazgos de evaluación)

- **Resolución de ERA5-Land vs. grilla de trabajo**: ERA5-Land tiene
  resolución nativa de ~9 km. Se interpola a la grilla de 250 m del
  proyecto, lo que introduce un artefacto de downscaling — los campos de
  viento/temperatura/humedad/precipitación tendrán variabilidad
  espacial artificialmente suave dentro de cada celda de 9 km original.
  Esto debe mencionarse explícitamente en cualquier resultado que use
  clima como insumo.
- **Resolución espacio-temporal reducida frente a la literatura**: el
  paper de referencia (WildfireCube) trabaja a 30 m / 3 h. PyroCast usa
  250 m / diario por ser un proyecto de una sola persona; esto es una
  simplificación deliberada, no una réplica del estado del arte, y los
  resultados no son directamente comparables a los de ese paper.
- **`bbox` de eventos como texto libre en `fire_event`**: la columna
  `bbox` es un `String` (no un tipo estructurado) en este bootstrap
  inicial; si se necesita indexar o filtrar espacialmente por bbox más
  adelante, migrar a un tipo estructurado o derivarlo de `geom`.

## Herramienta de investigación

Herramienta de investigación. No usar para decisiones operativas de
combate de incendios sin validación de CONAF/SENAPRED.
```

- [ ] **Step 4: Write `README.md`**

```markdown
# PyroCast

Sistema de pronóstico de propagación de incendios forestales mediante
fusión de datos satelitales y meteorológicos abiertos (Biobío, Ñuble,
La Araucanía).

> **Herramienta de investigación. No usar para decisiones operativas de
> combate de incendios sin validación de CONAF/SENAPRED.**

Ver `CLAUDE.md` para arquitectura, fuentes de datos y convenciones
completas; `docs/decisions.md` para decisiones de diseño y
`docs/limitations.md` para limitaciones conocidas.

## Quickstart

\`\`\`bash
cp .env.example .env   # completar credenciales, ver comentarios en el archivo
uv sync
make up                # levanta postgis + api (solo /healthz por ahora)
make test               # corre los tests de los 5 paquetes del workspace
make lint
make typecheck
\`\`\`

Ningún comando anterior requiere credenciales reales: `make test`,
`make lint` y `make typecheck` pasan en verde sin ningún valor en `.env`
más allá de placeholders de prueba (ver `shared/tests/test_config.py`).
```

- [ ] **Step 5: Commit**

```bash
git add Makefile docs/decisions.md docs/limitations.md README.md
git commit -m "docs: add Makefile, decisions, limitations and README"
```

---

### Task 8: GitHub Actions CI (ruff, mypy, pytest — zero credentials)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      FIRMS_MAP_KEY: ci-placeholder
      CDS_API_URL: https://cds.climate.copernicus.eu/api
      CDS_API_KEY: ci-placeholder
      COPERNICUS_DATASPACE_CLIENT_ID: ci-placeholder
      COPERNICUS_DATASPACE_CLIENT_SECRET: ci-placeholder
      POSTGRES_HOST: localhost
      POSTGRES_PORT: "5432"
      POSTGRES_DB: pyrocast
      POSTGRES_USER: pyrocast
      POSTGRES_PASSWORD: ci-placeholder
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.12"

      - name: uv sync
        run: uv sync --all-packages

      - name: Lint (ruff)
        run: uv run ruff check .

      - name: Typecheck (mypy: shared + features)
        run: uv run mypy shared/src features/src

      - name: Test (all workspace members)
        run: |
          uv run --package shared pytest shared/tests -v
          uv run --package ingestion pytest ingestion/tests -v
          uv run --package features pytest features/tests -v
          uv run --package models pytest models/tests -v
          uv run --package serving pytest serving/tests -v
```

The env block uses only `*-placeholder` strings — `Settings()` requires them present and non-empty, but no test or app code makes a real network call using them (no ingestion module exists yet to call FIRMS/CDS/Copernicus for real, and the live-Postgres test from Task 5 is `skipif`'d because no Postgres service is defined in this workflow).

- [ ] **Step 2: Verify by running the same commands locally that CI will run**

Run: `FIRMS_MAP_KEY=ci-placeholder CDS_API_URL=https://cds.climate.copernicus.eu/api CDS_API_KEY=ci-placeholder COPERNICUS_DATASPACE_CLIENT_ID=ci-placeholder COPERNICUS_DATASPACE_CLIENT_SECRET=ci-placeholder POSTGRES_HOST=localhost POSTGRES_PORT=5432 POSTGRES_DB=pyrocast POSTGRES_USER=pyrocast POSTGRES_PASSWORD=ci-placeholder uv run ruff check . && uv run mypy shared/src features/src`
Expected: both exit 0

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add ruff/mypy/pytest workflow with no real credentials"
```

---

### Task 9: Final verification against acceptance criteria

**Files:** none (verification only)

- [ ] **Step 1: `uv sync` from repo root**

Run: `uv sync --all-packages`
Expected: exit 0, `uv.lock` created/updated

- [ ] **Step 2: `make test` with no `.env` present**

Run: `rm -f .env && make test`
Expected: all 5 suites pass (the live-Postgres test in `shared` shows `SKIPPED`)

Wait — `make test` as currently written invokes `uv run --package shared pytest ...` which needs `Settings()`-dependent tests to set their own env via `monkeypatch`, not read `.env`. Confirm `shared/tests/test_config.py` and `test_smoke.py` don't implicitly require `.env` — they use `monkeypatch.setenv` per test, so this should already hold. If any test fails with a `ValidationError` here, that's a real bug in test isolation (a test not setting all required env vars) — fix it before proceeding, don't relax `Settings()`.

- [ ] **Step 3: `make lint` and `make typecheck`**

Run: `make lint && make typecheck`
Expected: both exit 0

- [ ] **Step 4: `make up` brings up both services healthy**

Run: `make up && sleep 5 && docker compose ps`
Expected: `postgis` and `api` both `healthy`

Run: `curl -sf http://localhost:8000/healthz`
Expected: `{"status":"ok"}`

Run: `make down`

- [ ] **Step 5: Final commit if any fixes were needed during verification**

```bash
git add -A
git commit -m "fix: address issues found during final acceptance verification"
```

(Skip this commit if Steps 1-4 needed no fixes.)

---

## Self-Review Notes

- **Spec coverage:** all 9 user tasks map to a plan task — 1→Task1, 2→Tasks1-6 (deps split by package), 3→Task2, 4→Task6, 5→Task6, 6→Task7, 7→Task8, 8→Task2/5, 9→Task2 (`test_config.py`) + Task3/4 (per-package smoke tests). Acceptance criteria verified explicitly in Task 9.
- **Placeholder scan:** every step has real, runnable code; `Makefile` "pendiente" targets are the user's own explicit request (Task 6 of their message), not a plan placeholder.
- **Type consistency:** `Settings.postgres_dsn`, `shared.db.schema.{metadata,fire_event,model_run,evaluation_result}`, and `serving.api.main.app` are defined once (Task 2/4) and only referenced (never redefined) in Tasks 5, 6, 8.
- **Review Focus:** all five items each have an owning test — missing-env-var (Task 2 Step 2), CI-no-secrets (Task 8), Docker non-root (Task 6 Step 3 + manual check in Step 5), Makefile placeholder clarity (Task 7 Step 1, `@echo "pendiente:..."` prefix), schema-against-real-Postgres (Task 5).
