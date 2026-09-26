# Fuentes de datos

## NASA FIRMS

**Qué entrega:** detecciones activas de fuego casi en tiempo real
(VIIRS 375 m por defecto en este proyecto). El Area API también acepta
MODIS 1 km vía `--sensor` (`MODIS_NRT`/`MODIS_SP`); LANDSAT
(`LANDSAT_NRT`) **es exclusivo de EE.UU./Canadá** y no sirve para Chile
pese a estar en la lista de `SOURCE` válidos. Solo los sensores
`VIIRS_*` están probados end-to-end en este proyecto — `MODIS_*` usa
una escala de `confidence` numérica distinta (ver más abajo) que no se
ha ejercitado con datos reales.

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

**Cacheo:** el resultado final (mosaico + reproyección) se cachea con
un nombre que incluye un hash de `(bbox, resolución, CRS)` — si se pide
el mismo bbox/resolución de nuevo, no se vuelve a descargar ni
reprocesar nada. Los tiles crudos individuales también se cachean por
su nombre (son globales/estáticos, reutilizables entre bboxes distintos
que compartan un tile).

**Pendiente y orientación (fórmula y unidades):** ver
`features/terrain/slope_aspect.py` — método de Horn (1981), el mismo
que usan GDAL `gdaldem` y ESRI. Con la ventana 3x3 `a b c / d e f / g h
i` centrada en el píxel (fila = eje Y hacia el sur, columna = eje X
hacia el este):

```
dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize_x)
dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize_y)

slope_deg  = grados(atan(hipot(dz/dx, dz/dy)))          # [0, 90]
aspect_deg = (grados(atan2(-dz/dx, dz/dy))) mod 360      # [0, 360), -1 si es plana
```

Pendiente en **grados** (0-90). Orientación en **grados de rumbo**
(0-360, sentido horario desde el norte: 0=N, 90=E, 180=S, 270=O), con
**-1** para celdas planas. Se calculan sobre el DEM ya reproyectado a
EPSG:32719 (metros) — `compute_and_save_terrain` rechaza explícitamente
un DEM en CRS geográfico (grados), porque el tamaño de celda en metros
variaría con la latitud y la pendiente quedaría mal calculada.

**Manejo de nodata:** los tiles reales de Copernicus DEM declaran
`nodata=None` — sin un valor propio, cualquier hueco del mosaico (tile
faltante, borde del área pedida) se rellenaría con `0.0` sin marcar, y
`features/terrain` lo leería como terreno real a nivel del mar. El
pipeline fuerza un nodata propio (`-32767.0` para el DEM, `-9999.0`
para pendiente/orientación) y lo propaga; las celdas cuya elevación de
origen es nodata se marcan como nodata en la salida, no se fabrica un
valor.

**Limitaciones conocidas:**
- **GLO-30 Public tiene huecos**: una fracción de tiles globales no
  está liberada públicamente por el programa Copernicus (variante
  `COP-DEM-GLO-30-R` vs. `Public`), y los tiles oceánicos genuinamente
  no existen (verificado: varios tiles costeros/oceánicos cercanos al
  área de estudio devuelven 404). El pipeline tolera tiles individuales
  faltantes (quedan como hueco marcado con nodata) y solo falla si
  **ningún** tile del bbox pudo descargarse.
- **Bordes de huecos de datos no son confiables**: una celda cuya
  elevación de origen es nodata se marca como nodata en la salida, pero
  las celdas VECINAS a ese hueco siguen usando el hueco dentro de su
  kernel 3x3 — su pendiente/orientación calculada no es confiable.
- **Salida no recortada al bbox exacto ni anclada a una grilla
  canónica**: el resultado cubre el mosaico completo de tiles enteros
  (puede exceder el bbox pedido), y el origen de la grilla de 250 m sale
  de los bounds reproyectados, no de un ancla fija — dos bboxes distintos
  dentro del mismo conjunto de tiles hoy producen rasters con orígenes
  de píxel distintos. Diferido a `features/grid/` (aún no implementado),
  que debe definir la grilla canónica de 250 m que todas las fuentes
  compartan.
- **Resolución nativa ~30 m, remuestreada a 250 m**: se pierde detalle
  de microrelieve; consistente con la simplificación deliberada de
  resolución ya documentada para todo el proyecto (ver limitations.md).

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
