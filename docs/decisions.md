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
`shared/` y `features/`. Esto se implementa pasando `--strict` en la
línea de comandos (`Makefile` target `typecheck`, y el mismo comando en
`.github/workflows/ci.yml`), apuntando solo a `shared/src features/src`
— no vía `[[tool.mypy.overrides]] strict = true` en `pyproject.toml`,
porque `strict` no es una opción real por-módulo en mypy: un override
que la fija en `true` para `shared.*`/`features.*` termina activando el
flag globalmente para toda invocación de `mypy` en el repo (verificado:
un módulo fuera de esos paquetes empezaba a fallar con `no-untyped-def`
hasta que se quitó el override). `ingestion/`, `models/` y `serving/`
así quedan realmente sin `--strict` hasta que haya código real que
tipar con disciplina.

Nota de proceso: el `pyproject.toml` también declara
`plugins = ["pydantic.mypy"]` bajo `[tool.mypy]` — sin él, `mypy
--strict` falla con `call-arg` sobre `Settings()` en
`shared/config.py`, porque no reconoce que una subclase de
`BaseSettings` de pydantic-settings puede construirse sin argumentos
(los campos requeridos vienen de variables de entorno en tiempo de
ejecución, no del constructor). El plugin le enseña eso a mypy.

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
`docker` disponible. El `Dockerfile` y `docker-compose.yml` se
revisaron por inspección estática, y se simuló fuera de Docker el paso
`uv sync --frozen --package shared --package serving --no-dev` en un
directorio aparte que no tenía los manifiestos de `ingestion`,
`features` ni `models` (confirmando que resuelve e instala igual) y se
verificó que la app FastAPI resultante responde `/healthz` **a través
del `TestClient` en ese entorno virtual copiado, no dentro de una
imagen construida** — eso sigue sin probarse. **No se corrió `docker
compose up` real, ni se construyó la imagen con `docker build`.** Un
desarrollador con Docker instalado debe validar `make up` y el
healthcheck del contenedor antes de asumir que el criterio de
aceptación correspondiente está probado end-to-end.

## Dependencias añadidas en la implementación de ingestion/firms/

- **`pyarrow`** (en `ingestion`): necesaria para escribir Parquet crudo
  (persistencia de respuestas de FIRMS tal cual, sin normalizar). Se
  eligió `pyarrow` puro (parseo manual del CSV con el módulo estándar
  `csv` + `pyarrow.Table`) en vez de `pandas` + un engine de Parquet,
  para no añadir una dependencia pesada que `ingestion` no necesita para
  nada más — el parseo real a un esquema tipado ocurre después, en
  `ingestion/firms/parser.py`, usando `shared.schemas.FireDetection`.
- **`pydantic`** (en `shared`, ahora explícito): ya estaba resuelto de
  forma transitiva vía `pydantic-settings`, y `shared/config.py` ya lo
  importaba directamente; esto solo hace explícita una dependencia que
  ya existía en la práctica, para `shared/schemas.py`.

## `per-file-ignores` de ruff para B008 en archivos `cli.py`

`typer` requiere `typer.Option(...)`/`typer.Argument(...)` como valor
por defecto de los parámetros de una función de comando — es cómo Typer
introspecciona las opciones del CLI. La regla `B008` de `flake8-bugbear`
(que ruff trae vía el grupo `B`) existe para atrapar el error clásico de
"llamada a función como valor por defecto mutable", pero aquí es el uso
previsto por el framework, no un bug. Se ignora `B008` específicamente
para `**/cli.py` en vez de deshabilitar la regla globalmente.

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

`rasterio`/`numpy` ya estaban aprobados por CLAUDE.md y ya declarados en
`features`; se agregaron también a `ingestion` (mosaico/reproyección de
tiles ocurre ahí) — declaración de dependencia normal, no una
dependencia externa nueva.

## `ingestion` depende de `features` (invierte la dirección habitual del pipeline)

`Makefile`'s `ingest-terrain` target está documentado en CLAUDE.md como
"DEM + derivados" — un solo comando que entrega tanto el DEM reproyectado
como pendiente/orientación. Como CLAUDE.md pide `data/processed/terrain/`
como destino final (no un DEM crudo), `ingestion/dem/cli.py` importa
`features.terrain.slope_aspect.compute_and_save_terrain` directamente, y
`ingestion/pyproject.toml` ahora depende de `features`. Esto invierte la
dirección `ingestion -> features` del diagrama de arquitectura de
CLAUDE.md. Es una dependencia acotada a un solo archivo de CLI
(`ingestion/dem/cli.py`); `ingestion/dem/pipeline.py` (la lógica real de
descarga/mosaico/reproyección) sigue sin depender de `features` en
absoluto. Alternativa considerada y descartada: un script separado fuera
del workspace de `ingestion`/`features` — más consistente con la capa,
pero rompe la promesa de "un solo comando" que el Makefile ya
publicitaba desde el bootstrap.

## DEM: manejo de nodata forzado, tile-grid semiabierta, descargas atómicas, tiles tolerantes a 404

Hallazgos de la revisión final de `ingestion/dem/` (2026-09-26), todos
corregidos en el mismo commit:
- Los tiles reales de Copernicus DEM declaran `nodata=None` (verificado
  contra el bucket real). Sin un valor propio, cualquier hueco del
  mosaico se rellenaba con `0.0` sin marcar, y `features/terrain` lo
  interpretaba como terreno real a nivel del mar, fabricando pendientes
  de hasta ~66° sobre relieve en realidad plano. Se fuerza un nodata
  propio (`-32767.0` DEM, `-9999.0` pendiente/orientación) en todo el
  pipeline.
- `tiles_for_bbox` usaba `floor()` también para el límite superior
  (norte/este), pero cada tile cubre `[lat, lat+1)` — semiabierto. Un
  borde del bbox exactamente sobre un entero incluía de más un tile de
  cobertura cero, que en la práctica falla con 404 si cae en el océano
  (bboxes de números redondos son comunes). Corregido a `ceil(x) - 1`
  para los límites superiores.
- Un tile individual faltante (hueco de cobertura de GLO-30 Public, o
  tile oceánico) ya no aborta todo el build — se tolera, queda como
  hueco marcado con nodata, y solo se falla si ningún tile del bbox pudo
  descargarse.
- La descarga de un tile ahora escribe a un archivo `.part` temporal y
  usa `os.replace()` al completar — una descarga interrumpida (los tiles
  pesan ~40 MB) ya no deja un `.tif` truncado que se trataría como
  cache-hit válido para siempre.
- `compute_and_save_terrain` ahora rechaza explícitamente un DEM en CRS
  geográfico o con transform rotada — antes calculaba pendientes
  fabricadas (una pendiente real de 45° se calculaba como ~90°) sin
  ningún error.

## ERA5-Land: `reanalysis-era5-land` horario + agregación propia, en vez de `derived-era5-land-daily-statistics`

El dataset de estadísticas diarias post-procesadas de CDS excluye
variables acumuladas (incluida precipitación total) — verificado en su
propia documentación. Como este proyecto necesita precipitación diaria,
se pide el dataset horario crudo (`reanalysis-era5-land`) y se agrega a
diario en `ingestion/era5/aggregate.py`: media para temperatura/punto de
rocío/viento, y para precipitación **el valor de la muestra de (día+1)
a las 00 UTC** (no una suma de las 24 muestras horarias — ver la
corrección de la revisión final, más abajo). Costo: una solicitud CDS
más pesada (24 pasos horarios en vez de un producto ya diario), y un
día adicional de solicitud (para poder leer el acumulado del último día
pedido); beneficio: control total y correcto sobre la agregación, sin
depender de qué variables decida excluir el dataset derivado.

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

## `h5netcdf` + `h5py` (en `ingestion` y `features`)

`xarray` (ya aprobado por CLAUDE.md) necesita un motor capaz de leer
NetCDF4/HDF5 real — lo que efectivamente entrega CDS. El único backend
NetCDF ya resuelto transitivamente en el workspace es el de `scipy`,
que solo soporta NetCDF3 clásico y no puede leer una descarga real de
ERA5-Land. `h5netcdf` es la alternativa de wheel puro (vía `h5py`), más
liviana que `netCDF4` (que empaqueta las librerías C completas de
netCDF-C + HDF5). Durante la implementación se descubrió que
`h5netcdf` por sí solo no basta: no declara `h5py` como dependencia dura
(falla con `ImportError: No module named 'h5py'` al primer uso real),
así que `h5py` se agregó explícitamente también.

## ERA5-Land: la clave de cache no incluye el bbox

`ingestion/era5/cache.py:cache_key_for` clavea solo por
`(rango de fechas, variables solicitadas)` — no por bbox. Es una
decisión de alcance, no un descuido: este proyecto tiene un único bbox
de estudio fijo por defecto (`shared.config.Settings.study_area_bbox`),
a diferencia de `ingestion/dem` y `ingestion/worldcover`, donde pedir
bboxes distintos dentro de la misma corrida es plausible. Costo si se
llegara a cambiar el bbox de estudio y reutilizar el mismo rango de
fechas: el daily NetCDF cacheado de la geografía anterior se
reutilizaría silenciosamente para la nueva — hoy no hay ninguna
verificación que lo detecte. Si en algún momento se necesita soportar
múltiples bboxes de ERA5-Land en la misma corrida, agregar el bbox a la
clave elimina el riesgo por completo.

## `openeo` (en `ingestion`)

CLAUDE.md ya nombra "openEO o sentinelhub-py" como las dos opciones
para acceder a Sentinel-2 vía Copernicus Data Space Ecosystem — no es
una dependencia sorpresa, pero al no estar en la lista explícita de
dependencias aprobadas del bootstrap original, se deja constancia:
`openeo` es el cliente Python oficial del proyecto openEO, necesario
para construir y ejecutar el grafo de procesamiento (carga + máscara de
nubes + composición temporal + descarga) contra el backend federado de
Copernicus Data Space Ecosystem. Ver la siguiente sección para por qué
se prefirió sobre `sentinelhub-py`.

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

## `CLOUD_SCL_CLASSES` duplicado entre `ingestion/sentinel2` y `features/vegetation` (SUPERSEDIDO)

Decisión original (durante la implementación del plan): duplicar el
frozenset de 4 clases SCL en ambos módulos para que
`features/vegetation/ndvi.py` pudiera re-aplicar el enmascarado
localmente como "defensa en profundidad", evitando invertir la
dependencia features->ingestion.

**Superseded en la revisión final** (ver hallazgos C3/C1 de la revisión
del branch `ingestion/sentinel2+worldcover`, 2026-09-26): la única forma
de que `ndvi.py` tuviera SCL disponible para re-enmascarar localmente
era conservar esa banda en el composite descargado — pero SCL es un
código categórico, y `reduce_dimension(dimension="t", reducer="median")`
aplicado a esa banda fabrica clases inexistentes (mediana de
observaciones `[4, 8]` = `6.0`, "Bare soil", que no ocurrió en ninguna
observación real). La "defensa en profundidad" quedaba operando sobre
datos ya corrompidos por el propio composite, y encima enmascaraba con
la lógica invertida (ver hallazgo C1). Se corrigió eliminando SCL del
composite descargado (se usa solo para construir la máscara
server-side, con la dirección de comparación corregida: `==`/`|`, "true
= es nube", no `!=`/`&`) — `ingestion/sentinel2/client.py` es ahora la
ÚNICA fuente de verdad para el enmascarado de nubes, y
`CLOUD_SCL_CLASSES` ya no existe en `features/vegetation/ndvi.py`. La
regla de dirección de dependencia (`ingestion` puede depender de
`features`, no al revés) sigue vigente y no cambió; lo que cambió es
que ya no hay una segunda copia que mantener sincronizada, porque ya no
hay una segunda pasada de enmascarado que hacer.

## Cierre de Etapa 1 (P1-P4): decisiones de reproyección y remuestreo

Resumen de las decisiones tomadas en los cinco módulos de ingesta
implementados hasta ahora (FIRMS, DEM, ERA5-Land, Sentinel-2,
WorldCover):

| Fuente | CRS destino | Resolución | Remuestreo | Por qué |
|---|---|---|---|---|
| Copernicus DEM | EPSG:32719 | 250 m | Bilineal | Elevación es continua; nearest produce escalones artificiales. |
| ERA5-Land | EPSG:32719 | 250 m | Bilineal | Viento/temperatura/humedad/precipitación son continuos. |
| Sentinel-2 (NDVI) | EPSG:32719 | 250 m | Bilineal | NDVI es una magnitud continua derivada de reflectancia. |
| ESA WorldCover | EPSG:32719 | 250 m | **Nearest** | Códigos de clase categóricos — interpolar fabricaría clases inexistentes. |
| NASA FIRMS | (sin reproyectar) | — | — | Puntos de detección en WGS84; `features/fire_state/` los clusteriza en eventos y rasteriza cada uno sobre su propia grilla (por evento, no la de estudio completo — ver `docs/fire-events.md`). |

Decisión transversal: **todas** las fuentes rasterizadas comparten el
mismo CRS de destino (`EPSG:32719`, UTM 19S) y la misma resolución
nominal (250 m, configurable en `shared/config.py`), pero **cada una
reproyecta de forma independiente** — no hay todavía una grilla
canónica compartida que garantice alineación píxel-a-píxel exacta
entre capas (mismo origen, mismo ancho/alto). `features/grid/` ya define
esa grilla canónica de forma determinista, pero estos cuatro pipelines de
ingesta todavía no reproyectan contra ella (ver `docs/limitations.md`):
hasta que lo hagan, dos capas de este proyecto
con el mismo CRS/resolución nominal pueden tener orígenes de píxel
ligeramente distintos, y superponerlas exactamente requiere un
remuestreo adicional de alineación en el consumidor (p. ej.
`features/dataset/`, también pendiente).

Patrón repetido de errores encontrados en revisiones finales sucesivas
(FIRMS, DEM, ERA5-Land) que este cierre deja documentado para el
próximo módulo de ingesta: (1) nodata sin forzar explícitamente permite
que huecos del mosaico se lean como `0.0` fabricado — DEM y WorldCover
ya lo corrigen desde el diseño inicial; (2) grillas de tiles con
esquema semiabierto necesitan `ceil(x)-1` para el límite superior, no
`floor()` — DEM lo corrigió en revisión, WorldCover lo aplicó desde el
principio; (3) un módulo sin comando de CLI ni target de Makefile real
es, en la práctica, código muerto que ningún test de integración
ejercita — FIRMS/DEM/ERA5-Land lo corrigieron en revisión,
Sentinel-2/WorldCover lo incluyen desde el principio (ver Task 6 del
plan de este módulo).

## Revisión final de `ingestion/sentinel2`+`ingestion/worldcover`+`features/vegetation`: hallazgos y correcciones (2026-09-26)

La revisión de branch completo encontró 3 hallazgos Críticos y 9
Importantes. Fueron corregidos en un único fix pass, cada uno con test
RED→GREEN propio (ver `ingestion/tests/` y `features/tests/`); los 9
Minor quedan diferidos sin corregir (ver `docs/limitations.md` y la
lista de abajo).

**Críticos corregidos:**
- **Máscara de nubes invertida** — `ingestion/sentinel2/client.py`
  construía `cloud_mask` como "true = está claro" (`!=`/`&`) y openEO
  `mask(mask_cube)` reemplaza por nodata donde el mask es `true` — el
  composite resultante conservaba nube y descartaba lo claro. Corregido
  a `==`/`|` ("true = es nube").
- **NDVI no leía el nodata declarado por la fuente** — un sentinel de
  nodata entero (p. ej. `-32768`) se leía como reflectancia real,
  fabricando un NDVI plausible que además contaminaba celdas vecinas al
  reproyectar con bilineal. Corregido: `compute_ndvi_masked` construye
  una máscara de validez desde `src.nodata` (y NaN) antes de calcular
  el cociente.
- **Mediana temporal sobre SCL (categórico)** — fabricaba clases de
  nube inexistentes, que luego alimentaban un enmascarado local ya
  corrompido. Corregido eliminando SCL del composite final (ver sección
  "SUPERSEDIDO" más arriba) — el composite descargado ahora tiene 2
  bandas (B04, B08), no 3.

**Importantes corregidos:**
- `temporal_extent` de openEO es exclusivo en su límite superior —
  usar el último día del mes perdía ese día completo. Corregido: límite
  = primer día del mes siguiente.
- Descarga de Sentinel-2 no era atómica — una descarga interrumpida
  dejaba un archivo parcial que el chequeo de cache trataba como válido
  para siempre. Corregido: `.part` + `os.replace`, mismo patrón que
  WorldCover.
- `version`/`year` de WorldCover no estaban en la clave de cache
  (colisión silenciosa entre años) y `download_tile` ignoraba los
  valores del llamador al construir la URL (garantizaba 404).
  Corregido: ambos parámetros ahora en la clave de cache y threaded
  hasta `tile_url`.
- Mosaico de WorldCover materializaba la unión completa de tiles de
  3°x3° en vez de recortar al bbox pedido. Corregido con
  `merge(sources, bounds=bbox, ...)`.
- `fuel_type.tif` se escribía con `nodata=None`, perdiendo la
  distinción entre "combustible desconocido" y "sin dato" para
  cualquier lector downstream. Corregido: `nodata=FUEL_TYPE_UNKNOWN`.
- Offset radiométrico BOA (`BOA_ADD_OFFSET=-1000`, processing baseline
  04.00+) no se aplicaba ni se documentaba — un error de NDVI de hasta
  0.25 en una magnitud acotada a [-1,1]. Corregido: constante nombrada
  aplicada explícitamente en `compute_ndvi_masked`, documentada en
  `features/vegetation/ndvi.py` y `docs/data-sources.md`.
- `docs/limitations.md` no tenía ninguna entrada de Sentinel-2/
  WorldCover. Corregido: backfill de la limitación de escala/memoria y
  del supuesto urbano/no-combustible.
- El supuesto "Built-up = no combustible" no estaba documentado en
  ningún lado con la fuerza de la limitación bosque/plantación.
  Corregido: bloque `***` en `fuel_type.py` + entrada en
  `docs/data-sources.md` y `docs/limitations.md`.
- Los tests del cliente Sentinel-2 no podían detectar una máscara
  invertida (registraban las llamadas al grafo pero nunca las
  aseveraban). Corregido: nuevo test que verifica los operadores
  exactos (`==`/`|`) y el conjunto de clases usado.

**Ruling explícito — escala/memoria a 10 m no resuelta completamente
(Importante, parcialmente diferido):** el hallazgo de la revisión
(`I7`) identificó que `compute_and_save_vegetation` y el mosaico de
WorldCover procesan arrays completos en memoria sin ventaneo, y que
`composite.download()` es una descarga openEO síncrona sin batch job.
Se corrigió la parte contenida y de bajo riesgo — WorldCover ahora
recorta con `merge(..., bounds=bbox)` en vez de mosaiquear tiles
enteros, y NDVI usa `float32` en vez de `float64` — pero el rediseño
completo (lectura/escritura por ventanas para NDVI a 10 m sobre el bbox
real, y conversión de la descarga Sentinel-2 a batch job) es un trabajo
de alcance comparable a un plan nuevo, no un fix puntual. Se documenta
honestamente como limitación conocida en `docs/data-sources.md` en vez
de forzarlo dentro de este fix pass — costo si esta decisión es
incorrecta: el pipeline de vegetación falla (OOM o rechazo de CDSE) al
correr contra el bbox de estudio real hasta que se haga ese trabajo.

**Minor deferidos (no corregidos, ver revisión completa para detalle):**
mapeo de clase 100 (Moss/lichen) a la etiqueta "nieve o hielo" en vez
de "suelo desnudo"; falta de comentario de contrato de orden de bandas
en el lado consumidor (parcialmente resuelto como efecto colateral del
fix de C3, que redujo el contrato a 2 bandas); bbox degenerado/puntual
produce 0 tiles con mensaje confuso; `int32` para códigos de
combustible que caben en `uint8`; descarga de tiles sin streaming;
`max_cloud_cover` no está en la clave de cache de Sentinel-2; fixture
de integración NDVI original con solapamiento casi nulo (ya
reemplazada en el fix pass por fixtures más grandes).

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
y sea correcta; migrar esos cuatro pipelines a compartir esta grilla
sigue sin hacerse.

**Corrección (revisión final, 2026-09-27)**: la primera versión de esta
nota afirmaba, en tiempo pasado, que `features/dataset/` "se implementó a
continuación de este módulo" y que `features/fire_state` "quedó conectado
a `pyrocast-features build-dataset` en cuanto ese módulo existió" —
ambas afirmaciones eran falsas en el momento en que se escribieron
(`features/dataset/__init__.py` decía literalmente "Pendiente de
implementación", no existía ningún `pyrocast-features`, y `make
build-dataset` seguía siendo un stub). Eran una intención declarada como
hecho consumado, no una descripción de lo que existía. `features/dataset/`
sí se implementó, pero en una solicitud posterior y separada del usuario
dentro de esta misma sesión, no como parte de este módulo — ver el
commit que agrega `features/dataset/` y `pyrocast-features` para el
estado real.

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

## `features/fire_state/`: sin CLI/Makefile propio (se conecta desde `features/dataset/`)

A diferencia de FIRMS/DEM/ERA5-Land/Sentinel-2/WorldCover, este módulo no
tiene comando de CLI ni target de Makefile propio. `features/terrain` y
`features/weather` tampoco lo tienen — se invocan como un paso de cómputo
desde dentro de `ingestion/dem/cli.py`/`ingestion/era5/cli.py`,
inmediatamente después de la propia descarga de esa fuente.
`features/fire_state` no tenía un punto de enganche equivalente en el
momento de implementarse: el clustering necesita el HISTORIAL acumulado
de detecciones, no la respuesta de una sola descarga. Quedó sin CLI
propio hasta que `features/dataset/` (solicitud separada y posterior
dentro de esta misma sesión) agregó `pyrocast-features build-dataset`,
que sí lo invoca — ver `docs/dataset-card.md` y el commit que agrega ese
módulo.

## `pyproj` declarado explícitamente en `features/pyproject.toml`

`features/fire_state/rasterize.py` importa `pyproj` directamente (para
reproyectar detecciones WGS84 al CRS de la grilla). Ya estaba instalado
de forma transitiva vía `rasterio`, pero depender de eso sin declararlo
es frágil — se agrega como dependencia directa del paquete `features`.

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

## `expose raw wind_u`/`wind_v` en `features/weather/derive.py`

`compute_and_save_weather` guardaba solo velocidad/dirección de viento
derivadas — el tensor de `features/dataset/` pide explícitamente los
componentes u/v crudos, no derivados, así que se agregaron como dos
claves nuevas del dict retornado (`wind_u`, `wind_v`), sin quitar
`wind_speed`/`wind_direction` (ambas representaciones tienen consumidores
distintos: el autómata celular probablemente querrá velocidad/dirección
para el término de alineación de viento, el tensor de dataset quiere
componentes crudos para no imponerle a un modelo de deep learning una
descomposición polar innecesaria).

## `event_id` estable derivado del contenido, no de la posición en la lista

`features/fire_state/clustering.py::build_fire_events` originalmente
asignaba `event_id` por `enumerate()` sobre los clusters ordenados —
determinista dado un orden de entrada fijo, pero el orden de entrada real
(detecciones leídas de un `glob()` de archivos parquet) no está
garantizado entre corridas. Encontrado en la revisión final de
`features/grid`+`fire_state` (2026-09-27): el mismo incendio físico podía
recibir un `event_id` distinto en dos corridas con las mismas detecciones
en distinto orden, lo cual invalida silenciosamente el split
train/val/test guardado por id y el nombre del archivo Zarr de un evento
ya persistido. Corregido: `event_id` es ahora un hash determinista del
contenido del evento (fechas + coordenadas de sus detecciones, ordenadas
antes de hashear) — estable sin importar el orden de entrada. Una
colisión de hash entre dos eventos físicos distintos es posible en
principio (espacio de 32 bits) pero, al volumen de detecciones de este
proyecto, del mismo orden de riesgo que una colisión de hash corto de
git; no se agregó detección de colisión.

## Revisión final de `features/dataset/`: hallazgos y correcciones (2026-09-27)

La revisión de branch completo encontró 2 hallazgos Críticos y 12
Importantes. Se corrigieron ambos Críticos y 9 de los 12 Importantes en
un único fix pass, cada uno con test RED→GREEN propio.

**Críticos corregidos:**
- **`build_dataset_for_event` devolvía `grid.bounds` (EPSG:32719,
  metros) como si fuera el bbox WGS84** — se persistía en PostGIS
  etiquetado `SRID=4326` con una "latitud" de millones de metros, un
  polígono geométricamente sin sentido que PostGIS acepta sin quejarse
  (`Geometry`, a diferencia de `geography`, no valida rango). Corregido:
  se devuelve `event_bbox` (ya calculado en WGS84) en vez de
  `grid.bounds`.
- **`aspect_deg` se resampleaba con `Resampling.bilinear`** — es una
  magnitud CIRCULAR (0-360° más el sentinel -1 para plano), no continua
  en el sentido que bilineal asume: interpolar 358° y 2° (ambos "casi
  norte") da ~180° (sur), un error de 180 grados en la variable que
  alimenta el término de viento/pendiente del autómata celular. Medido:
  100% de una ladera norte homogénea salía "sur" tras el resampleo.
  Corregido a `Resampling.nearest` (mismo criterio que `fuel_type`, pero
  por una razón distinta — circular, no categórica).

**Importantes corregidos:**
- El mismo `DEFAULT_BUFFER_M` (radio de la máscara de fuego, 375 m) se
  usaba también para el buffer espacial del bbox del evento — un evento
  de una detección terminaba con un tensor de 4x4 píxeles. Separado en
  dos parámetros: `fire_buffer_m` (máscara) y `context_buffer_m`
  (contexto espacial del tensor, default 2000 m).
- `firms_loader.py` no deduplicaba detecciones — re-ingerir el mismo
  rango, o cubrir el mismo período con dos satélites VIIRS, duplicaba la
  detección física en el parquet crudo, cambiando el `event_id` estable
  (el hash de contenido retiene duplicados) y defeando el propósito del
  fix de `event_id` de la revisión anterior. Corregido: dedup por
  `(fecha/hora, lat/lon redondeados a 6 decimales, satélite)`.
- `split_events` podía dejar `val` o `test` completamente vacíos para
  3-5 eventos, contradiciendo el 70/15/15 documentado. Corregido:
  garantiza al menos 1 evento por split a partir de n=3; con menos de 3,
  todo va a `train` explícitamente.
- `_nearest_month_path` resolvía empates de distancia según el orden de
  iteración del dict (a su vez dependiente de un `glob()` sin orden
  garantizado) — no determinista en la práctica. Corregido: `sorted()`
  antes de buscar el mínimo, el mes más antiguo gana la empatada siempre.
- `_nearest_month_path` no tenía distancia máxima — un composite NDVI de
  años de antigüedad se presentaba como si fuera el estado vigente de la
  vegetación. Corregido: `max_month_distance` (default 3 meses en el
  llamador), más allá de eso NaN explícito.
- El tensor Zarr no llevaba coords `x`/`y` ni atributos de CRS/transform/
  `event_id` — no había forma de recuperar la ubicación real de un
  píxel, ni de qué evento era, a partir solo del Zarr. Corregido:
  `assemble_event_tensor` ahora recibe la `WorkGrid` del evento y su
  `event_id`, y los escribe como coords/atributos.
- `features/cli.py` recalculaba la ventana de padding con un `5`
  hardcodeado en vez de usar `DEFAULT_PRE_EVENT_PADDING_DAYS` — podían
  desincronizarse. Corregido: `padded_days_for_event()`, único lugar que
  calcula esa ventana, usado tanto por el CLI como por
  `build_dataset_for_event`.
- Un día de clima con UN solo campo faltante (p. ej. `precipitation`) se
  trataba como "sin clima ese día", descartando las otras 4 capas reales
  a NaN. Corregido: `resolve_event_sources` conserva los campos que sí
  existen por día, en vez de todo-o-nada.
- `fire_event` no tenía columna para enlazar con el `event_id` de
  `features/fire_state` (hash de contenido, potencialmente > 2^31 —
  desborda `Integer` de Postgres). Agregada `firms_event_id`
  (`BigInteger`, `UNIQUE`, `nullable=True` — un evento catalogado por
  CONAF/SENAPRED no tiene este id). `persist_fire_event_metadata` ahora
  hace upsert (`INSERT ... ON CONFLICT (firms_event_id) DO UPDATE`) en
  vez de un `insert` liso — re-correr `build-dataset` para el mismo
  evento actualiza su fila en vez de duplicarla.
- Varias aserciones de test no discriminaban entre una implementación
  correcta y una rota (`present <= {...}` en vez de `==`, ausencia de
  aserciones sobre `bbox_cut`/`fire_mask==1`/apertura real del Zarr en
  el smoke test del CLI). Fortalecidas junto con cada fix de arriba.

**Ruling explícito — partial-failure no transaccional (Importante,
diferido):** `pyrocast-features build-dataset` no envuelve el loop sobre
eventos en manejo de excepciones; si un evento falla a mitad de camino,
los anteriores quedan con Zarr escrito y fila de PostGIS
insertada/actualizada, pero `splits.json` (escrito al final) no existe.
El upsert por `firms_event_id` (arriba) hace que re-correr el comando
sea seguro para PostGIS, y `save_event_to_zarr` sobrescribe — pero no
hay un resumen de qué eventos fallaron ni un intento de continuar tras
un fallo puntual. Se documenta honestamente en `docs/dataset-card.md` en
vez de forzar un rediseño transaccional completo dentro de este fix
pass — costo si esta decisión es incorrecta: una corrida larga que falla
a mitad de camino requiere inspeccionar manualmente qué eventos
terminaron de procesarse.

**Minor deferidos (no corregidos):** `fuel_type.tif` con `nodata=99`
(código real de "desconocido") convierte esas celdas a NaN en vez de
distinguir "desconocido real" de "sin dato"; `event_{id:04d}.zarr` con
id-hash de más de 4 dígitos hace que el `:04d` sea vestigial (el
nombrado sigue siendo correcto, solo el padding es cosmético); un
`ndvi_*.tif` con nombre de mes malformado lanza `ValueError` sin
capturar; `resolve_event_sources` re-glob-ea el DEM en cada evento del
loop (ineficiente, no incorrecto); el CLI no valida `--end >= --start`
(a diferencia de `ingestion/firms/cli.py`); falta el aviso de
"herramienta de investigación" en `dataset-card.md` (consistente con
`fire-events.md`/`data-sources.md`, no una regresión nueva).

## `models/cellular_automata/`: tabla de flammability duplicada desde `ingestion/worldcover/fuel_type.py`

`models/cellular_automata/rules.py::DEFAULT_FUEL_FLAMMABILITY` usa los
mismos códigos enteros de tipo de combustible simplificado que
`ingestion/worldcover/fuel_type.py` ya define, sin importar ese módulo
directamente — importar desde `ingestion` invertiría la dirección de
dependencia establecida (`ingestion` → `features`/`models`, nunca al
revés), exactamente el mismo caso ya resuelto para `CLOUD_SCL_CLASSES`
entre `ingestion/sentinel2` y `features/vegetation`. Si los códigos de
`ingestion/worldcover/fuel_type.py` cambiaran, esta tabla quedaría
desincronizada silenciosamente — mantenerlas en sync es manual.

## `models/evaluation/metrics.py` implementado directamente en su ubicación final de P8

El enunciado pidió las métricas de evaluación (IoU, Brier score) "aunque
aún no exista el módulo completo" de `models/evaluation/`, con la
instrucción explícita de moverlas ahí en P8 "sin duplicar código". En vez
de crearlas dentro de `models/cellular_automata/` y planear un movimiento
futuro, se crearon DIRECTAMENTE en `models/evaluation/metrics.py` desde
el principio — cuando P8 (calibración isotónica, backtesting) se
implemente, reutiliza estas mismas funciones sin ningún movimiento de
archivo ni duplicación.

## `models/src` no se agrega a `make typecheck` en este plan

CLAUDE.md especifica `mypy --strict` en `shared/` y `features/`
únicamente; el Makefile's `typecheck` target refleja eso. Este plan
verifica `mypy --strict models/src/models/cellular_automata` (y
`models/cli.py`) tarea por tarea, y pasa limpio, pero no modifica el
Makefile para agregar todo `models/src` a `make typecheck` — los stubs
pre-existentes `models/deep/__init__.py` no fueron verificados contra
`--strict` y podrían no pasar. Ampliar la cobertura de `mypy --strict` a
todo `models/` queda como una decisión separada, más grande, para cuando
`models/deep`/`models/evaluation` completo se implementen.

## `typer` agregado a `models/pyproject.toml`

`models/cli.py` (el nuevo comando `pyrocast-models run-ca`) necesita
`typer` — no estaba entre las dependencias de `models` (que hasta ahora
era una librería pura, sin CLI propio). Se agrega como dependencia
directa, mismo patrón que `ingestion`/`features`.

## `_score_sample` de `calibrate.py`: comparación de un solo paso

`TrainingSample` solo carga `observed_final_mask` (el estado final
relevante para entrenar), no una trayectoria diaria completa —
`_score_sample` simula exactamente UN día desde `initial_burning` del
propio sample y compara ese resultado directamente contra
`observed_final_mask`. Esto solo tiene sentido literal cuando la
"máscara final" del sample es alcanzable en un solo paso simulado desde
su condición inicial declarada — cierto para los tests de este módulo
(construidos así a propósito), pero un sample real de un evento
multi-día de `features/fire_state` necesitaría o bien su propio
`initial_burning` por día (el estado al INICIO del día que se está
puntuando), o que la calibración corra la trayectoria completa
multi-día y compare la probabilidad del ÚLTIMO día contra la máscara
observada del último día. Documentado como limitación conocida en
`docs/limitations.md` — no se rediseñó dentro de este plan porque no
hay todavía datos de entrenamiento reales (de `features/dataset/`) que
forzaran la forma correcta de `TrainingSample` para el caso multi-día.

## Revisión final de `models/cellular_automata/`: hallazgos y correcciones (2026-09-27)

0 hallazgos Críticos (el pre-verificado exhaustivo de la fórmula/geometría
antes de escribir el plan se sostuvo: las 8 cifras exactas verificadas a
mano reprodujeron byte a byte contra el código final). 5 Importantes,
todos corregidos:

- `_build_params` ignoraba en silencio cualquier clave de `param_grid`
  no reconocida (`.get(key, default)`) — un typo en el nombre de un
  parámetro evaluaba los defaults N veces sin avisar, devolviendo un
  resultado que parece una calibración exitosa. Corregido: validación
  explícita de claves con `ValueError` nombrando las desconocidas.
- Dos tests se llamaban "...recovers_the_true_base_spread_prob..." pero
  el grid search con IoU y un `TrainingSample` de un solo paso NO
  recupera el valor verdadero — solo distingue de qué lado del umbral
  0.5 cae cada candidato (verificado: candidatos `[0.5, 0.9]` con
  verdad=0.9 elige 0.5). Renombrados para describir lo que realmente
  prueban, con un test nuevo que documenta explícitamente la
  función-escalón del IoU en este caso.
- Un comentario en `_score_sample` afirmaba que `n_days=1` "se
  re-derivaba desde el propio estado observado" — falso, es un literal
  fijo. Corregido el comentario para describir el comportamiento real
  (incluida la consecuencia de que `seed` no tiene ningún efecto con
  `n_days=1`).
- `simulate_fire_spread` no validaba `elevation`/`wind_u`/`wind_v` —
  un NaN (el mismo patrón que `features/dataset/resample.py` produce
  para huecos de cobertura) se propagaba en silencio y apagaba la
  simulación sin aviso. Corregido: `ValueError` explícito al detectar
  valores no finitos, antes de que se propaguen.
- `np.clip(p_dir, 0.0, 1.0)` satura a spread cierto por encima de ~22°
  de pendiente o ~8 m/s de viento alineado — condiciones reales de
  incendios chilenos (viento Puelche), no solo casos extremos. Ningún
  test ejercitaba ese régimen. Documentado con la tabla de valores
  exactos en `docs/cellular-automata.md` y `docs/limitations.md` — no
  se cambió el comportamiento (recortar a 1.0 sigue siendo la decisión
  correcta para mantener `p_dir` como una probabilidad válida), solo se
  hizo explícito que el modelo deja de ser probabilístico ahí.

**Minor deferidos (no corregidos):** `models/cellular_automata/__init__.py`
sigue con el docstring "Pendiente de implementación" (desactualizado,
cosmético); el test de "más vecinos en llamas" no fija el valor exacto
`1-(1-p)^4`, solo la desigualdad; `grid_search_calibrate([], grid)` da
`ZeroDivisionError` sin mensaje; arrays de viento 3D más cortos que
`n_days` dan `IndexError` sin nombrar la causa; `SpreadParameters` no es
hasheable pese a `frozen=True` (por el campo `dict`); `run-ca --size 0`
da un traceback de Typer sin validar; `docs/cellular-automata.md` no
lleva el aviso de "herramienta de investigación" (consistente con
`fire-events.md`/`dataset-card.md`, no una regresión nueva).

## `shared/model_protocol.py`: la interfaz común de modelo vive en `shared/`, tipada contra `xr.DataArray`

El enunciado pidió el `Protocol` en `shared/` explícitamente, para que
`models/cellular_automata` (P7) y el futuro U-Net (P9-P11) lo
implementen igual sin que `models/evaluation/backtest.py` ni el reporte
final (P16) dupliquen lógica por modelo. `shared/` gana `xarray` como
dependencia SOLO para tipar `event: xr.DataArray` concretamente (el
mismo tensor que `features/dataset/assemble.py` ya produce) -- `shared/`
NO importa `features/` ni `models/`, preservando la dirección de
dependencia establecida (todo depende de `shared/`, nunca al revés).

## `models/evaluation/db.py`: cada backtest es un experimento nuevo, no un upsert

A diferencia de `features/dataset/db.py::persist_fire_event_metadata`
(upsert por `firms_event_id`, corregido en la revisión final de ese
módulo), `persist_backtest_run` hace un `insert` liso en `model_run` +
`evaluation_result` cada vez. Es la semántica correcta para un registro
de experimento de ML: re-correr un backtest con parámetros distintos (o
incluso los mismos, para verificar reproducibilidad) es una corrida
NUEVA que vale la pena conservar, no una corrección de la anterior.

## Cierre de Etapa 3: modelo funcional y medible de punta a punta

Con `models/cellular_automata/` (P7) implementando
`shared.model_protocol.FireSpreadModel` y `models/evaluation/backtest.py`
corriendo métricas per-evento + agregadas con bootstrap contra
cualquier modelo que cumpla esa interfaz, el proyecto tiene su primer
modelo funcional de punta a punta: ingesta → features → grilla/eventos →
dataset → simulación → evaluación. Es deliberadamente simple (un
autómata celular sin calibrar contra incendios reales, ver
`docs/cellular-automata.md`/`docs/limitations.md`) — el valor de este
cierre de etapa es que el ARNÉS completo (Protocol, backtest, métricas,
persistencia, CLI) ya existe y es reutilizable sin cambios cuando P9-P11
agregue el U-Net como segundo implementador de `FireSpreadModel`.

## `run_backtest`: se compara "¿ha ardido esta celda alguna vez?" acumulado, no la extensión activa diaria

Hallazgo de la revisión final del 2026-09-28: el canal `fire_mask` de
`features/dataset/` es la extensión ACTIVA de fuego por día (una celda
que ardió ayer y no hoy vuelve a `False`, ver
`features/fire_state/rasterize.py`), pero `CellularAutomatonModel`
predice el estado ACUMULADO -- `simulate.py` documenta explícitamente
que `burning` es monótono no decreciente porque el autómata no modela
extinción. Comparar la predicción acumulada contra la verdad cruda día a
día penaliza al modelo por una diferencia de CONVENCIÓN (una celda que
FIRMS dejó de ver activa) y no por un error real de propagación --
antes de este fix, un backtest sobre un evento realista de varios días
daba IoU/ECE dramáticamente peores de lo que la calidad real de la
propagación explicaría.

Decisión: `run_backtest` acumula la verdad con
`np.logical_or.accumulate` antes de calcular las 4 métricas, así ambas
series (predicha y real) miden la misma pregunta: "¿esta celda ha
ardido alguna vez hasta el día d?". Esto NO corrige la simplificación
de que el autómata celular no modela extinción -- esa sigue siendo una
limitación real y documentada (`docs/cellular-automata.md`,
`docs/limitations.md`) -- solo evita medir esa simplificación ya
conocida como si fuera un error adicional del backtest. Alternativa
descartada: comparar solo el último día del evento (pierde toda la
trayectoria intermedia, que es justamente lo que el backtest existe
para evaluar).

## `models/evaluation/db.py::persist_backtest_run`: una transacción para todo el backtest, no una por evento

Hallazgo de la revisión final del 2026-09-28: la versión original abría
una transacción `engine.begin()` POR EVENTO dentro del loop del CLI. Si
el evento `k` de N tenía un `firms_event_id` que ya no existe en
`fire_event` (p. ej. los archivos Zarr sobrevivieron a una base de
datos recreada -- un escenario real en este proyecto, donde
`data/processed/` está fuera de git y Postgres se puede recrear desde
cero), `scalar_one()` levantaba `NoResultFound` sin nombrar el evento
ni la causa, y los primeros `k-1` eventos ya habían sido persistidos --
dejando `model_run` parcialmente poblado sin ningún aviso.

`persist_backtest_run` ahora recibe la lista completa de
`EventMetrics` y persiste todo en UNA transacción: resuelve todos los
`firms_event_id` primero (fallando con `UnknownFireEventError`,
nombrando el id exacto, antes de escribir nada) y solo entonces
inserta. Un backtest de N eventos ahora es atómico: todo o nada.

## `models/deep/tfrecord_reader.py`: TFRecord leído a mano, sin `tensorflow` ni el paquete `tfrecord`

`models/` depende de `torch`, no de `tensorflow` — agregar
`tensorflow` (varios cientos de MB, un segundo framework de deep
learning completo) solo para leer un formato de archivo no se
justifica. El paquete `tfrecord` de PyPI tiene un conflicto de versión
de `protobuf` conocido (encontrado durante la investigación previa a
este plan, en un proyecto de terceros que enfrentó el mismo problema
con este mismo dataset). Se implementó un lector mínimo (framing
TFRecord + un decodificador protobuf acotado a los wire types que
`tf.train.Example` realmente usa) verificado por round-trip contra un
encoder propio en los tests -- sin acceso de red en este entorno para
validar contra un archivo real de Kaggle, documentado explícitamente
como una limitación en `docs/public-dataset.md`.

## `models` gana `features` como dependencia: reutiliza `assemble_event_tensor`, no lo duplica

`models/deep/public_dataset.py` necesita producir EXACTAMENTE el mismo
tensor `(day, channel, y, x)` que `features/dataset/assemble.py`
produce para un evento real de Chile -- reimplementar esa construcción
en `models/` arriesgaría que ambos esquemas se desincronicen en
silencio. `models → features` es la dirección "mainline" documentada
en el diagrama de arquitectura de CLAUDE.md
(`ingestion → features → models → evaluation → serving`), ya aceptada
como no problemática en revisiones anteriores de este proyecto (a
diferencia de `features → ingestion`, que sí invertiría la dirección
establecida). También reutiliza `features/terrain/slope_aspect.py`
para derivar pendiente/orientación desde la elevación de NDWS, en vez
de duplicar la fórmula de Horn (1981).

## Humedad relativa de NDWS: fórmula de presión de vapor, no reutiliza `relative_humidity_approx` de PyroCast directamente

`features/weather/derive.py::relative_humidity_approx` espera
temperatura + punto de rocío (dewpoint); NDWS no trae dewpoint, trae
humedad específica (`sph`, kg/kg) -- una cantidad física distinta. Se
implementó la conversión humedad-específica-a-relativa estándar de la
OMM (presión de vapor real desde `sph` + presión, presión de
saturación vía Magnus-Tetens con los MISMOS coeficientes de Alduchov &
Eskridge 1996 que `relative_humidity_approx` ya usa, para mantener
consistencia física entre ambas fórmulas) en vez de forzar una
conversión intermedia sph→dewpoint que agregaría un paso de error
adicional. Asume presión estándar a nivel del mar (101325 Pa) porque
NDWS no trae presión de superficie -- documentado como aproximación en
`docs/limitations.md`.
