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

## `scripts/init_db.py` corre con `--package serving`, no `--package shared`

`shared` depende solo de lo que necesita como librería (pydantic-settings,
sqlalchemy, geoalchemy2, psycopg) y no de `typer`, para no forzar una
dependencia de CLI en un paquete que también importan `ingestion`,
`features` y `models`. `serving` ya depende de `shared` y de `typer`, así
que el script de inicialización de base de datos se ejecuta en su
entorno.

## Entorno de este bootstrap no tenía Docker instalado

Este primer commit del monorepo se hizo en un entorno sin el binario
`docker` disponible. El `Dockerfile` y `docker-compose.yml` se validaron
por inspección estática y simulando el paso `uv sync --frozen --package
shared --package serving --no-dev` fuera de Docker (confirmando que
resuelve e instala correctamente sin los manifiestos de `ingestion`,
`features` y `models`, y que la app FastAPI resultante responde
`/healthz`), pero **no se corrió `docker compose up` real**. Un
desarrollador con Docker instalado debe validar `make up` y el healthcheck
del contenedor antes de asumir que el criterio de aceptación
correspondiente está probado end-to-end.
