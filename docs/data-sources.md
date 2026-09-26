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
