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
- **Permisos del volumen `./data` en Linux**: el contenedor `api` corre
  como usuario no-root `pyrocast` (uid 1000). Con un bind mount
  (`./data:/data`), el propietario efectivo dentro del contenedor es el
  del directorio del host, no el `chown` hecho en el Dockerfile — en
  macOS/Docker Desktop esto normalmente funciona sin fricción, pero en
  Linux con un host cuyo uid de usuario no sea 1000 puede producir
  errores de permiso al escribir en `/data`. Si eso ocurre, ajustar los
  permisos de `./data` en el host (o cambiar a un volumen nombrado).
- **DEM: salida no recortada al bbox exacto ni anclada a una grilla
  canónica**: `ingestion/dem/pipeline.py` produce un raster que cubre el
  mosaico completo de tiles enteros de 1°x1° (puede exceder el bbox
  pedido), con el origen de la grilla de 250 m determinado por los
  bounds reproyectados, no por un ancla fija — dos bboxes distintos que
  comparten los mismos tiles hoy producen orígenes de píxel distintos.
  Esto se descubrió en la revisión final de `ingestion/dem/` (un
  revisor independiente cuantificó, para un bbox de prueba grande, que
  ~6% de la salida caía fuera de la cobertura real de datos). Se corrigió
  el síntoma más grave — esas celdas ahora se marcan con nodata en vez
  de fabricarse como `0.0` (terreno a nivel del mar) — pero el recorte
  exacto al bbox y el anclaje a una grilla canónica de 250 m compartida
  entre todas las fuentes (DEM, ERA5-Land, vegetación) queda diferido a
  `features/grid/`, todavía sin implementar.
- **DEM: bordes de huecos de datos no confiables**: una celda cuya
  elevación de origen es nodata se marca como nodata en la pendiente y
  orientación de salida, pero las celdas vecinas a ese hueco siguen
  usando el hueco dentro de su kernel 3x3 de cálculo — su valor no es
  confiable cerca del borde de cualquier hueco de datos.
- **`docker compose up` no verificado end-to-end**: el bootstrap inicial
  se hizo en un entorno sin Docker instalado (ver `docs/decisions.md`).
  El Dockerfile y compose se revisaron estáticamente y se simuló la
  resolución de dependencias fuera de Docker, pero nadie ha confirmado
  todavía que el contenedor `api` realmente arranca y pasa su healthcheck
  dentro de Docker real.

## Herramienta de investigación

Herramienta de investigación. No usar para decisiones operativas de
combate de incendios sin validación de CONAF/SENAPRED.
