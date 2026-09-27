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
y sea correcta; migrar los pipelines existentes queda para
`features/dataset/` (ver más abajo — implementado a continuación de este
módulo), que es el consumidor que en la práctica necesita alineación
píxel-a-píxel entre todas las capas.

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
de detecciones, no la respuesta de una sola descarga — quedó conectado a
`pyrocast-features build-dataset` en cuanto ese módulo existió (ver más
abajo), no antes.

## `pyproj` declarado explícitamente en `features/pyproject.toml`

`features/fire_state/rasterize.py` importa `pyproj` directamente (para
reproyectar detecciones WGS84 al CRS de la grilla). Ya estaba instalado
de forma transitiva vía `rasterio`, pero depender de eso sin declararlo
es frágil — se agrega como dependencia directa del paquete `features`.
