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
