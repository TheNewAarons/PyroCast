# Fuentes de datos

## Licencias y atribución (resumen)

Todas las fuentes son abiertas y gratuitas. **Este repositorio no
redistribuye ningún dato**: `data/` está en `.gitignore` y cada persona
descarga los datos por su cuenta con sus propias credenciales. Aun así,
cualquier resultado, mapa o publicación derivada debe llevar las
atribuciones de abajo. Última verificación contra las páginas oficiales:
**2026-10-01** (las URL están en la columna "Fuente de la licencia"); la
columna "Verificado" dice qué se confirmó ese día y qué no.

| Fuente | Licencia / términos | Atribución requerida | Fuente de la licencia | Verificado |
|---|---|---|---|---|
| **NASA FIRMS** (VIIRS 375 m) | Política de datos abiertos de NASA (uso libre, sin restricción de redistribución); la API exige un `MAP_KEY` gratuito | "We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms), part of NASA's Earth Science Data and Information System (ESDIS)." Para VIIRS 375 m, citar además a Schroeder et al. (2014), *Remote Sensing of Environment* 143, 85-96, doi:10.1016/j.rse.2013.12.008 | [Earthdata: Data Use and Citation Guidance](https://www.earthdata.nasa.gov/learn/find-data/near-real-time/citation) | Texto de agradecimiento confirmado; la cita de Schroeder et al. es la que indica la documentación de FIRMS pero no se re-verificó la página ese día |
| **Copernicus DEM GLO-30** | Licencia Copernicus DEM: GLO-30 "Public" libre para el público general bajo los términos de su licencia (es decir, no es dominio público) | "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved". Si los datos se modifican (aquí se remuestrean y derivan pendiente/orientación): "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved". Citar además "Copernicus Digital Elevation Model (DEM) was accessed on `FECHA` from https://registry.opendata.aws/copernicus-dem" | [Registro AWS Open Data](https://registry.opendata.aws/copernicus-dem/) y el texto de licencia que enlaza | Aviso de atribución y texto de acceso confirmados (resultados de búsqueda y registro AWS); **el documento de licencia completo no se leyó** — leerlo antes de redistribuir derivados |
| **ERA5-Land** (Copernicus CDS) | CC BY 4.0 ("CC-BY licence" en la página del dataset); además se deben aceptar los términos del dataset en el sitio de CDS antes de descargar | Citar: Muñoz Sabater, J. (2019): *ERA5-Land hourly data from 1950 to present*. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), doi:10.24381/cds.e2161bac; y el artículo Muñoz-Sabater et al. (2021), *Earth System Science Data* 13, 4349-4383, doi:10.5194/essd-13-4349-2021. Aviso recomendado de C3S: "Generated using Copernicus Climate Change Service information [año]" | [CDS: ERA5-Land hourly](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | Licencia y DOI confirmados en la página del dataset; el texto exacto del aviso de C3S no estaba en la página consultada (es el aviso estándar de C3S) |
| **Sentinel-2 L2A** (Copernicus Data Space Ecosystem, vía openEO) | Datos Sentinel de Copernicus: acceso "libre, completo y abierto" (Legal Notice on the use of Copernicus Sentinel Data); el uso de la plataforma CDSE se rige además por sus términos de uso y por su política de créditos/cuotas | Datos modificados (aquí: composición mensual + NDVI): "Contains modified Copernicus Sentinel data [año]". Sin modificar: "Copernicus Sentinel data [año]" | [Copernicus Sentinel data licence](https://cds.climate.copernicus.eu/licences/ec-sentinel); [Legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice) | Fórmulas de atribución confirmadas; los términos de uso de CDSE (cuotas) no se leyeron completos |
| **ESA WorldCover 10 m 2021 v200** | CC BY 4.0 | "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium". Citar: Zanaga, D. et al. (2022). *ESA WorldCover 10 m 2021 v200*. doi:10.5281/zenodo.7254221; y "ESA WorldCover was accessed on `FECHA` from https://registry.opendata.aws/esa-worldcover-vito" | [Registro AWS Open Data](https://registry.opendata.aws/esa-worldcover-vito/) y [esa-worldcover.org/data-access](https://esa-worldcover.org/en/data-access) | Licencia, DOI y textos de atribución confirmados |
| **Next Day Wildfire Spread (NDWS)** — dataset público de preentrenamiento, vía Kaggle | CC BY 4.0 | Citar: Huot, F., Hu, R. L., Goyal, N., Sankar, T., Ihme, M., Chen, Y.-F. (2022). "Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data". *IEEE Transactions on Geoscience and Remote Sensing* 60, 1-13 | [Kaggle: Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread); detalle en `docs/public-dataset.md` | Licencia y cita confirmadas en fuentes secundarias (la página de Kaggle exige navegador autenticado y no se pudo leer directamente). **El U-Net evaluado en `docs/results.md` NO usó este dataset** (se entrenó desde cero con eventos de Chile); el código de preentrenamiento existe pero nunca se corrió con datos reales |

**Otras atribuciones de la interfaz web** (`serving/web/`): las teselas
del mapa son de OpenStreetMap — "© OpenStreetMap contributors" (ODbL;
[copyright](https://www.openstreetmap.org/copyright)), ya incluido en la
atribución del mapa; el servidor público de teselas de OSM tiene una
[política de uso](https://operations.osmfoundation.org/policies/tiles/)
(adecuada para uso de investigación local, no para tráfico masivo).
Leaflet se carga desde unpkg con SRI (licencia BSD-2-Clause).

**Derivados que produce este proyecto** (`docs/results.md`, mapas de la
interfaz, tensores Zarr) mezclan varias de estas fuentes: mantener todas
las atribuciones aplicables al compartirlos. Los modelos entrenados
heredan la condición de atribución de CC BY 4.0 de sus datos de
entrenamiento si se redistribuyen.


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
precipitación, agregado a diario por este proyecto: media para
temperatura/punto de rocío/viento. Para precipitación, **el total
diario NO es la suma de las 24 muestras horarias** — ERA5-Land acumula
`tp` de forma corrida desde las 00 UTC de cada día (verificado contra
la documentación de ECMWF), así que el total real del día d es la
muestra de (d+1) a las 00 UTC. Sumar las 24 muestras horarias del día
sobrecuenta por ~11-12x y mezcla el acumulado del día anterior — un
error real encontrado y corregido en la revisión final de este módulo.
Se pide el dataset horario crudo (`reanalysis-era5-land`), no el
derivado de estadísticas diarias de CDS, porque este último excluye
variables acumuladas (incluida precipitación total).

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
fuente). Puede tardar minutos u horas según la carga del servicio. Un
rango de fechas que cruza un límite de mes/año se parte en una
solicitud por mes calendario (`ingestion/era5/client.py:month_chunks`)
— las listas `year`/`month`/`day` de CDS producen el producto
cartesiano si no se hace esto, lo que puede pedir fechas futuras
inexistentes. Cacheo por hash de `(rango de fechas, variables
solicitadas)` — el bbox no forma parte de la clave (se asume el bbox de
estudio fijo del proyecto).

**Reproyección y remuestreo:** reproyectado a `EPSG:32719` en la
resolución de trabajo con remuestreo **bilineal** (magnitudes
continuas). `features/weather/derive.py` también deriva velocidad/
dirección del viento desde u/v y humedad relativa aproximada desde
temperatura/punto de rocío (Magnus-Tetens, coeficientes de Alduchov &
Eskridge 1996 — válida -40°C a 50°C, error máximo documentado ±0.4%
RH, recortada a [0,100] porque ERA5-Land puede entregar Td > T).

**Limitaciones conocidas:**
- **Downscaling por interpolación, no física**: ~9 km a 250 m es
  puramente geométrico — no introduce detalle real de sub-grilla.
- Humedad relativa es una aproximación, no una medición real.
- Salida no recortada al bbox exacto ni anclada a una grilla canónica
  compartida entre fuentes — misma limitación que Copernicus DEM,
  diferida a `features/grid/`.
- La semántica exacta de "DATE = primer día del rango" en el Area API
  de FIRMS y el comportamiento de `year`/`month`/`day` de CDS están
  verificados contra documentación, no contra una llamada real
  autenticada (este entorno no tiene credenciales reales) — ver
  `docs/decisions.md`.

## Sentinel-2 L2A (Copernicus Data Space Ecosystem, vía openEO)

**Qué entrega:** composición mensual de menor nubosidad (mediana
temporal tras enmascarar nubes por SCL) de las bandas B04 (rojo) y B08
(NIR), usada para calcular NDVI — `features/vegetation/ndvi.py`. La
banda SCL se usa solo para construir la máscara de nubes server-side y
se descarta antes de la reducción temporal (es un código categórico;
una mediana temporal sobre SCL fabricaría clases inexistentes — ver
`docs/decisions.md`), así que el GeoTIFF descargado tiene 2 bandas, no 3:

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
de nube, nube prob. media/alta, cirros delgados), aplicado ÚNICAMENTE
server-side en el grafo openEO — no hay una segunda pasada local en
`features/vegetation/ndvi.py` (la habíamos diseñado como "defensa en
profundidad" en la primera versión, pero requería conservar SCL
post-reducción temporal, y eso es exactamente lo que fabrica clases
inexistentes; ver `docs/decisions.md`). Cacheado por hash de `(bbox,
año, mes)`. Descarga atómica (`.part` + `os.replace`).

**Corrección radiométrica:** los DN de reflectancia se corrigen con el
offset aditivo `BOA_ADD_OFFSET = -1000` del processing baseline 04.00+
(vigente desde 2022-01-25, cubre toda la temporada 2025-26 de este
proyecto) antes de calcular el cociente NDVI — la colección
`SENTINEL2_L2A` de CDSE no publica esta metadata, así que es un
supuesto explícito, no un valor leído del dato (ver
`features/vegetation/ndvi.py`).

**Reproyección y remuestreo:** NDVI reproyectado a `EPSG:32719` en la
resolución de trabajo con remuestreo **bilineal** (magnitud continua,
igual que DEM y ERA5-Land). Nodata del composite (declarado por la
fuente, o NaN) se propaga explícitamente a nodata de NDVI antes de
reproyectar — nunca se deja que un sentinel de nodata entero se lea
como reflectancia real.

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
- **Escala no probada contra el bbox de estudio real**: a 10 m nativos,
  el composite sobre el bbox por defecto (~2.7°x2.8°) es del orden de
  ~31000x24000 píxeles. `compute_and_save_vegetation` lee ambas bandas
  completas en memoria (no hay lectura por bloques/ventanas) — esto es
  del orden de decenas de GB para ese tamaño y no ha sido probado a esa
  escala. Además, `composite.download()` es una descarga openEO
  síncrona; CDSE limita el tamaño/tiempo de procesamiento síncrono, y un
  producto de este tamaño probablemente requiere un batch job
  (`create_job`/`start_and_wait`) en vez de descarga directa. Ninguno de
  los dos problemas está resuelto todavía — quedan diferidos a un
  trabajo futuro de procesamiento por bloques + batch jobs.

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
suroeste, mismo esquema semiabierto que Copernicus DEM. Un tile
faltante no aborta el mosaico completo (mismo criterio de tolerancia
que DEM). El mosaico se recorta al bbox pedido (`merge(..., bounds=bbox)`)
en vez de materializar la unión completa de tiles enteros de 3°x3° —
para el bbox de estudio por defecto, evita leer un array de varios GB
donde la mayor parte del área no es necesaria. Resultado final cacheado
por hash de `(bbox, resolución, CRS, versión, año)`.

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
- **El modelo asume que las áreas urbanas (clase Built-up) no
  propagan fuego** (`FUEL_URBANO_NO_COMBUSTIBLE`) — una simplificación
  de v1 que impide representar la interfaz urbano-forestal (WUI), el
  escenario detrás de las viviendas destruidas que motivan este
  proyecto. Ver el docstring de `fuel_type.py`.
- La tabla de mapeo a tipo de combustible es una simplificación
  heurística de una persona, no un sistema de combustibles validado en
  terreno (Fireline/Behave/Scott-Burgan) — ver el docstring de
  `fuel_type.py`.
- Producto de un único año (2021 por defecto; el año/versión son
  parámetros de `build_worldcover`); no captura cambios de uso de suelo
  posteriores (p. ej. cosecha de plantaciones, incendios previos que ya
  cambiaron la cobertura).
- Salida no anclada a una grilla canónica compartida entre fuentes —
  misma limitación que Copernicus DEM, diferida a `features/grid/`
  (el mosaico sí se recorta al bbox pedido, a diferencia de DEM).
