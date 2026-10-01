# Limitaciones conocidas

Este documento se actualiza con cada hallazgo real de la evaluación
contra incendios de Chile. Nunca se suaviza ni se elimina una métrica
negativa para que el proyecto "se vea mejor" (ver CLAUDE.md).

## Limitaciones por área

Agrupadas por área del sistema; dentro de cada área, en el orden en que
se fueron encontrando. Las limitaciones nuevas de la etapa de
endurecimiento (servido, ingesta, seguridad) están al final. Las ya
resueltas están en "Resueltas (historial)", no se borran.

### Fuentes de datos, resolución y datos de entrada

- **Resolución de ERA5-Land vs. grilla de trabajo**: ERA5-Land tiene
  resolución nativa de ~9 km. `features/weather/derive.py` la
  reproyecta y remuestrea a la grilla de 250 m del proyecto mediante
  interpolación bilineal — un downscaling puramente geométrico, **no**
  una modelación física de procesos atmosféricos de sub-grilla. Los
  campos de viento/temperatura/humedad/precipitación tendrán
  variabilidad espacial artificialmente suave dentro de cada celda de
  9 km original; la variabilidad real por debajo de esa escala
  simplemente no está en los datos de origen. Debe mencionarse
  explícitamente en cualquier resultado que use clima como insumo.
- **Humedad relativa de ERA5-Land es aproximada, no medida**: se deriva
  de temperatura y punto de rocío con la fórmula de Magnus-Tetens
  (coeficientes de Alduchov & Eskridge, 1996), válida entre -40°C y
  50°C con un error máximo documentado de ±0.4% RH en ese rango — es una
  aproximación estándar en meteorología, pero no reemplaza una medición
  real de humedad relativa.
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
  entre todas las fuentes (DEM, ERA5-Land, vegetación) queda diferido:
  `features/grid/` ya existe (define la grilla canónica de forma
  determinista), pero este pipeline de DEM en particular todavía no
  reproyecta contra ella — sigue calculando su propio destino desde los
  bounds de su mosaico.
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
  Actualización (endurecimiento): la imagen ahora incluye `features`, `models` e `ingestion` (arrastra `torch`, imagen pesada) y `serving/web`; tampoco se construyó aquí. El job `docker` de CI es lo que la valida.
- **Vegetación (Sentinel-2/NDVI): escala a 10 m no probada contra el
  bbox de estudio real**: `features/vegetation/ndvi.py` lee ambas
  bandas del composite completas en memoria (sin lectura por
  ventanas/bloques). A 10 m nativos sobre el bbox por defecto
  (~2.7°x2.8°), el composite es del orden de ~31000x24000 píxeles —
  decenas de GB en memoria, no probado a esa escala. Además,
  `Sentinel2Client.fetch_monthly_composite` descarga el composite de
  forma síncrona (`composite.download()`); CDSE limita el tamaño/tiempo
  de procesamiento síncrono, y un producto de este tamaño
  probablemente requiere un batch job (`create_job`/`start_and_wait`)
  en vez de descarga directa. Encontrado en la revisión final de
  `ingestion/sentinel2`+`features/vegetation` (2026-09-26); la mitigación
  contenida (mosaico de WorldCover recortado al bbox, NDVI en float32 en
  vez de float64) se aplicó, pero el procesamiento por bloques y la
  conversión a batch job quedan pendientes — ver `docs/decisions.md`.
  Mitigación operativa vigente: pasar `--bbox` por evento (como en `docs/backtest-2026.md`) en vez del bbox de estudio completo.
- **Fuel-type: áreas urbanas asumidas no combustibles**: la clase
  Built-up de WorldCover se mapea a `FUEL_URBANO_NO_COMBUSTIBLE`
  (`ingestion/worldcover/fuel_type.py`) — un supuesto de v1, no una
  verificación empírica. Esto significa que el modelo, tal como está,
  no puede representar la propagación de fuego hacia la interfaz
  urbano-forestal (WUI), que es precisamente el escenario detrás de las
  muertes y viviendas destruidas que motivan este proyecto (ver
  CLAUDE.md). Encontrado en la revisión final del 2026-09-26.
- **ERA5-Land solo cubre tierra: eventos cercanos a la costa producen
  NaN parcial en los canales de clima incluso en días con datos
  disponibles** -- verificado contra datos reales de la temporada
  2025-2026 (revisión final del 2026-09-29): la interpolación bilineal
  de `features/dataset/pipeline.py::build_dataset_for_event` entre una
  celda de tierra válida de ERA5-Land y una celda de océano (sin datos)
  produce NaN en una fracción de píxeles del recorte del evento -- hasta
  ~13% de los píxeles de clima en el evento más grande de la temporada
  (`event_1277049523`, cercano a la costa de Biobío). Sin tratar, este
  NaN envenena la convolución del U-Net en TODA la imagen, no solo en la
  celda afectada. Se rellena con el promedio de los píxeles VÁLIDOS del
  archivo fuente COMPLETO (la región de estudio entera, no el recorte
  del evento, que podría no tener ningún píxel de tierra propio) --
  `_fill_weather_nan_with_source_mean` en `pipeline.py`. Esto es una
  imputación real, documentada explícitamente: para un evento muy
  costero, el clima "local" en la práctica es el promedio regional, no
  una medición específica de ese punto.

### Estado del fuego y dataset (`features/`)

- **Reconstrucción de eventos de incendio: buffer + interpolación lineal,
  no kriging**: `features/fire_state/` reconstruye la extensión ACTIVA de
  fuego diaria de un evento (no la superficie quemada acumulada — cada
  máscara diaria es independiente, no incluye lo ya quemado en días
  anteriores) con un buffer espacial fijo alrededor de cada detección
  FIRMS más interpolación temporal lineal (equivalente a unión de
  máscaras) entre días con detección — WildfireCube (paper de referencia)
  usa kriging espaciotemporal, que estima incertidumbre espacial y
  produce una reconstrucción más plausible físicamente. Un incendio que
  se apaga y se reactiva en otro punto dentro de la misma ventana
  `temporal_eps` (2 días por defecto) se rellena como si hubiera seguido
  ardiendo en ambos lugares durante el hueco, sobreestimando la extensión
  activa en ese caso. Ver `docs/fire-events.md`.
- **El buffer de rasterización es un radio de 375 m, no el "tamaño de
  píxel"**: cubre ~3.1x el área nominal de un píxel VIIRS (140 625 m² vs.
  ~441 786 m² del buffer) — una sobre-cobertura deliberada (compensa
  incertidumbre de geolocalización y crecimiento de píxel fuera de
  nadir), no calibrada contra incendios reales. Encontrado en la revisión
  final del 2026-09-27 — la documentación original describía el valor
  como "tamaño de píxel" sin aclarar que se usa como radio, lo que
  ocultaba la sobreestimación de área. Ver `docs/fire-events.md`.
- **El "encadenamiento" (chaining) del clustering no tiene límite
  temporal ni espacial acotado por evento** — solo detección-a-detección:
  una cadena de detecciones a ≤2 días/≤750 m cada una puede encadenar un
  evento arbitrariamente largo en el tiempo (verificado: 30 días) o en el
  espacio (verificado: 83 km). Esto puede tanto sobre-fusionar episodios
  distintos como, a la inversa, fragmentar de más un incendio disperso
  por nubosidad/humo bajo condiciones de viento fuerte (p. ej. Puelche).
  Ver `docs/fire-events.md`.
- **Parámetros de clustering de eventos sin calibrar contra incendios
  reales**: `spatial_eps_m=750m` y `temporal_eps=2 días`
  (`features/fire_state/clustering.py`) son heurísticas basadas en la
  resolución nominal de VIIRS (375 m, revisita diaria), no un ajuste
  contra el historial real de incendios de Chile — el backtest ya
  existe (`docs/backtest-2026.md`), pero no se usó para ajustar estos
  parámetros: siguen sin calibrar.
- **`features/dataset/`: un único DEM/estudio de área asumido**:
  `resolve_event_sources` espera exactamente un archivo bajo
  `data_processed_dir/dem/` (convención de nombre con hash de
  bbox+resolución+CRS, un único estudio de área configurado a la vez) —
  levanta un error claro si encuentra 0 o más de 1, pero no soporta
  múltiples estudios de área simultáneos.
- **`features/dataset/`: canales sin cobertura se rellenan con NaN, sin
  error explícito**: si el rango de fechas ingerido con `pyrocast-ingest`
  no cubre los días de padding previos a un evento (o el mes de NDVI más
  cercano no existe en absoluto), el canal correspondiente queda en NaN
  para esos días — un dataset con muchos NaN no falla ruidosamente, hay
  que inspeccionarlo. Ver `docs/dataset-card.md`.
  `serving/` sí falla explícitamente (`weather_unavailable`, `terrain_coverage_unavailable`) si falta un canal que el modelo usa; `build-dataset` no.
- **`features/dataset/`: cada evento tiene su propia grilla, no la grilla
  de estudio completa**: dos tensores de eventos distintos no son
  comparables píxel a píxel sin un paso de reproyección adicional — ver
  `docs/dataset-card.md`.
- **`features/dataset/`: `pyrocast-features build-dataset` no es
  transaccional entre eventos**: un fallo a mitad de una corrida con
  varios eventos deja los anteriores con Zarr escrito y fila de PostGIS
  insertada/actualizada (el upsert por `firms_event_id` hace esto
  seguro para re-intentos), pero sin `splits.json` y sin un resumen de
  qué evento falló. Encontrado en la revisión final del 2026-09-27, ver
  `docs/dataset-card.md`.
- **`features/dataset/`: deduplicación de detecciones FIRMS no
  verificada contra datos reales**: `firms_loader.py` deduplica por
  `(fecha/hora, lat/lon redondeados a 6 decimales, satélite)` — asume
  que dos detecciones físicas distintas casi nunca coinciden en los tres
  campos, una suposición razonable pero no verificada contra el
  historial real de FIRMS de Chile (dos satélites VIIRS pueden detectar
  el mismo incendio con timestamps/coordenadas ligeramente distintos, lo
  cual es correcto que NO se deduplique). Ver `docs/dataset-card.md`.

### Autómata celular (`models/cellular_automata/`)

- **Autómata celular: sin modelo de extinción/consumo de combustible**:
  `models/cellular_automata/simulate.py` trata el estado "en llamas" como
  monótono — una celda encendida nunca se "apaga" dentro del horizonte
  simulado. Un incendio real se extingue al consumir el combustible
  disponible; esto no está modelado. Ver `docs/cellular-automata.md`.
- **Autómata celular: parámetros libres sin calibrar contra incendios
  reales**: `base_spread_prob=0.3`, `slope_coefficient=4.0`,
  `wind_coefficient=0.2` y la tabla `fuel_flammability` son heurísticas
  elegidas para que el comportamiento cualitativo pedido (más vecinos en
  llamas = mayor probabilidad, cuesta arriba más rápido, a favor del
  viento se alarga) sea verificable, no un ajuste contra el historial
  real de incendios de Chile. `calibrate.py` existe pero necesita
  eventos de entrenamiento reales (de `features/dataset/`, que a su vez
  necesita datos ingeridos reales) para producir valores con algún
  sentido — ver `docs/dataset-card.md`.
  El backtest de `docs/backtest-2026.md` y `serving/` usan estos valores por defecto.
- **`calibrate.py` solo calibra los tres parámetros escalares**:
  `fuel_flammability` (un dict, no un escalar) no participa del grid
  search — calibrarlo requeriría un espacio de búsqueda combinatorio
  mucho más costoso. Queda como trabajo futuro.
- **`calibrate.py`'s scoring de un solo paso**: `TrainingSample` compara
  contra `observed_final_mask` simulando un único día desde
  `initial_burning` — válido para eventos de un paso, pero requeriría
  ajustarse (ejecutar la trayectoria completa de varios días) para
  calibrar contra un evento real de `features/dataset/` que abarca
  varios días. Consecuencia directa verificada en la revisión final del
  2026-09-27: `seed` no tiene ningún efecto en `_score_sample` (con
  `n_days=1`, el único array de probabilidad se calcula antes del primer
  sorteo aleatorio).
- **`calibrate.py` con métrica IoU y un `TrainingSample` de un solo
  paso NO recupera el valor exacto de `base_spread_prob`**: cuando cada
  celda del anillo de ignición tiene exactamente un vecino en llamas,
  `P(ignición)=base_spread_prob` para todas ellas, y el IoU umbralizado
  en `>=0.5` da el MISMO score para cualquier candidato `>=0.5` sin
  importar cuán cerca esté del valor verdadero — el grid search solo
  distingue de qué lado del umbral 0.5 cae cada candidato, no recupera
  el valor real. Verificado: con candidatos `[0.5, 0.9]` y un valor
  verdadero de 0.9, el grid search elige **0.5** (el primero con el
  mismo score), no 0.9. Encontrado en la revisión final del 2026-09-27 —
  ver `models/tests/test_cellular_automata_calibrate.py` para el test
  que documenta este comportamiento explícitamente.
- **8 direcciones (vecindad de Moore), no propagación continua**: la
  distancia diagonal se calcula correctamente (`resolución_m·√2`), pero
  la resolución angular de la propagación está limitada a 8 direcciones
  discretas por celda — un frente de fuego real no está limitado así.
- **Autómata celular: `np.clip(p_dir, 0.0, 1.0)` satura a spread CIERTO
  en condiciones reales de incendios chilenos, no solo en casos extremos
  de laboratorio**: con los parámetros por defecto a 250 m de
  resolución, `p_dir` supera 1.0 (spread determinista, no
  probabilístico) por encima de ~22° de pendiente o ~8 m/s (30 km/h) de
  viento alineado con la propagación — velocidades de viento comunes en
  episodios de viento Puelche que impulsan los megaincendios de
  Biobío/Ñuble/Araucanía. Ninguno de los tests actuales ejercita este
  régimen (usan pendientes/vientos moderados que se quedan dentro del
  rango no saturado). Encontrado en la revisión final del 2026-09-27 —
  ver `docs/cellular-automata.md` para la tabla de valores exactos.
- **`simulate_fire_spread` rechaza `elevation`/`wind_u`/`wind_v` no
  finitos (NaN/inf) con un error explícito** — necesario porque
  `features/dataset/resample.py` rellena huecos de cobertura con NaN
  (nunca fabrica un valor), y esos son precisamente los arrays que
  alimentarían este simulador en una integración futura con eventos
  reales. Antes de este fix (revisión final del 2026-09-27), un solo
  NaN de elevación se propagaba en silencio y reducía una simulación de
  55 celdas encendidas a 1, sin ningún aviso. El autómata ya está
  conectado a eventos reales (backtest y `serving/`): este chequeo es lo
  que convierte un NaN en un error explícito (en la API, un 422
  `terrain_coverage_unavailable` / `weather_unavailable`).

### Evaluación y backtest

- **Backtest: la evaluación del día 0 es tautológica por construcción**:
  `CellularAutomatonModel.predict` siembra `initial_burning` desde el
  propio `fire_mask` del día 0 del evento (no hay "día -1" del que
  sembrar) -- así que la predicción del día 0 para las celdas ya en
  llamas coincide con la verdad por definición, no porque el modelo haya
  "acertado" nada. Incluir el día 0 en las métricas agregadas del
  backtest sesga (levemente, hacia arriba) el desempeño reportado. Ver
  `docs/decisions.md`.
  Las métricas de `docs/results.md` conservan esa convención (incluyen el día 0); su efecto no se cuantificó por separado.
- **Bootstrap del backtest remuestrea valores por evento, no eventos
  reales ni píxeles**: con pocos eventos de test (realista en las
  primeras corridas de este proyecto), el intervalo de confianza
  bootstrap es necesariamente ancho y poco informativo -- es la
  limitación estadística esperada de tener pocos eventos, no un error
  de implementación.
- **`ece_score` con probabilidades sin calibrar en absoluto** (p. ej. el
  autómata celular, que nunca se calibró explícitamente para producir
  probabilidades bien calibradas, solo para que la propagación
  cualitativa sea correcta) puede dar valores de ECE altos que no
  reflejan un error de implementación sino la falta de calibración
  probabilística real del modelo. Ver `docs/cellular-automata.md`.
- **El autómata celular sigue sin modelar extinción, y el backtest ya no
  penaliza esa simplificación como si fuera un error adicional (pero
  la simplificación en sí sigue ahí)**: `run_backtest` compara la
  predicción (acumulada, monótona) contra la verdad TAMBIÉN acumulada
  -- ver `docs/decisions.md`. Esto hace que las métricas midan
  correctamente "¿la propagación predicha coincide con dónde ardió
  realmente, alguna vez?", pero un incendio real que se apaga y el
  autómata sigue "quemando" esa zona en cada día posterior todavía es
  una limitación real del modelo (ver más arriba, "no hay modelo de
  extinción/consumo de combustible") -- el backtest deja de castigar
  DOS VECES la misma simplificación ya conocida, no la elimina.
- **Acumular detecciones FIRMS día a día como proxy de "superficie
  quemada acumulada" no es lo mismo que la superficie quemada
  acumulada real**: VIIRS tiene huecos de revisita y cobertura de
  nubes -- una celda que ardió mientras el satélite no pasó (o pasó con
  nubes) nunca aparece en `fire_mask` ningún día, y por lo tanto tampoco
  en la verdad acumulada que usa `run_backtest`. El backtest mide contra
  la mejor proxy disponible con datos abiertos, no contra la superficie
  quemada real.
- **El autómata celular usado en `docs/backtest-2026.md` NO está
  calibrado contra incendios reales de Chile** -- `calibrate.py` (P7)
  sigue sin soportar eventos multi-día reales de `features/dataset/`
  (su `_score_sample` solo compara un único paso simulado, ver más
  arriba en este documento); extenderlo a trayectorias multi-día reales
  quedó fuera de alcance de esta tarea. El backtest reportado usa los
  parámetros HEURÍSTICOS por defecto (`base_spread_prob=0.3`, etc.), no
  valores ajustados contra el historial real -- ver `docs/backtest-2026.md`.
- **El backtest tiene 2 eventos de test, y el "test interno" del dataset de
  Chile es ese mismo conjunto**: el split 70/15/15 por evento sobre 15
  eventos deja 2 de test. No hay ninguna evaluación sobre NDWS ni sobre
  otra temporada. Los intervalos bootstrap con n=2 no sostienen
  afirmaciones de superioridad de un modelo (ver `docs/results.md`
  secciones 1 y 5; el análisis de fallas son hipótesis no verificadas).
- **Selección de eventos por umbral de detecciones (≥40)**: sesga la
  muestra hacia incendios grandes y bien detectados por VIIRS; no
  representa la cola larga de incendios pequeños (`docs/backtest-2026.md`).

### Dataset público NDWS y preentrenamiento

- **La humedad relativa derivada de NDWS asume presión estándar a
  nivel del mar (101325 Pa), no la presión real de cada ubicación**:
  NDWS no trae un campo de presión de superficie; ignorar la variación
  real de presión con la elevación introduce un error que crece con la
  altitud del recorte. Es una aproximación aceptada para datos de
  PREENTRENAMIENTO (no la señal de evaluación final contra incendios
  de Chile) -- ver `docs/decisions.md` y `docs/public-dataset.md`.
- **`fuel_type` de las muestras de NDWS es SIEMPRE "desconocido" (código
  99)**: NDWS no tiene ningún canal de cobertura de suelo o tipo de
  vegetación categórico. Un U-Net preentrenado con estas muestras nunca
  ve una señal real de `fuel_type` durante el preentrenamiento -- solo
  durante el fine-tuning sobre eventos de Chile (que sí trae
  `fuel_type` real de ESA WorldCover) aprende a usar ese canal. Esto es
  intencional (ver `docs/decisions.md`: inventar una clase de
  combustible desde NDVI sería un dato fabricado, peor que admitir que
  no existe), pero significa que el preentrenamiento por sí solo no
  enseña nada sobre `fuel_type`.
- **`models/deep/tfrecord_reader.py` nunca se probó contra un archivo
  real descargado de Kaggle** -- este entorno no tiene acceso de red
  para descargarlo. Verificado en tres niveles distintos (ver
  `docs/public-dataset.md` para el detalle): el CRC32C y su fórmula de
  máscara están verificados contra el vector de control ESTÁNDAR de la
  especificación (no solo contra el encoder de test propio, que
  duplica la misma implementación y por eso no podía por sí solo
  detectar un polinomio o máscara incorrectos -- corregido en la
  revisión final del 2026-09-28, antes solo había verificación por
  round-trip); el anidamiento protobuf y el framing TFRecord (incluido
  gzip, soportado y probado) son autoconsistentes por round-trip. Lo
  que sigue sin probar es la compatibilidad byte a byte contra un
  archivo real de Kaggle.
- **El nombre exacto de los archivos dentro del zip de Kaggle no se
  pudo confirmar** al escribir `docs/public-dataset.md` (la página
  requiere una sesión de navegador autenticada). `models/deep/public_dataset.py`
  no asume ningún nombre -- recibe una lista explícita de rutas del
  caller -- pero si Kaggle distribuye un formato distinto de TFRecord
  (algunas re-subidas de terceros ofrecen `.npy`), este loader no lo
  soporta.
- **El decodificador protobuf de `tfrecord_reader.py` no soporta
  `float_list` no empaquetado (wire type 5) ni un campo `packed`
  partido en varios chunks length-delimited del mismo número de campo**
  -- ambas son codificaciones legales del protobuf de
  `tf.train.Example`, aunque el escritor real de TensorFlow nunca las
  produce (siempre un único chunk packed). Riesgo bajo en la práctica,
  documentado para no sobre-afirmar conformidad total con la
  especificación. Encontrado en la revisión final del 2026-09-28.
- **Features que no son `float_list` (p. ej. `int64_list`) se
  descartan en silencio** por `tfrecord_reader.py`, en vez de fallar --
  si Kaggle alguna vez cambia la codificación de una feature de NDWS,
  `transform_ndws_record` la reportaría como "feature faltante", no
  como "feature con tipo inesperado". Encontrado en la revisión final
  del 2026-09-28.
- **`split_public_dataset` con menos de 2 shards da `val` vacío sin
  aviso** (0 shards → todo vacío; 1 shard → todo a `train`) -- mismo
  tipo de degradación silenciosa que `features/dataset/split.py` tenía
  antes de su propia revisión final, sin el mismo aviso explícito acá
  todavía. Riesgo bajo (un usuario real de NDWS tendrá muchos más de 2
  shards), documentado para no ocultarlo. Encontrado en la revisión
  final del 2026-09-28.
- **El U-Net evaluado se entrenó desde cero, sin preentrenamiento con el
  dataset público NDWS**: sin credenciales de Kaggle no había dataset
  público, así que `pyrocast-train finetune` (que permite omitir el
  checkpoint preentrenado) entrenó un `SmallUNet` nuevo solo sobre ~11
  eventos reales de Chile, órdenes de magnitud menos que la literatura
  (NDWS: 18.545 chips). Cualquier resultado débil del U-Net debe leerse
  a la luz de este tamaño de entrenamiento, no como evidencia de que la
  arquitectura sea inadecuada. El camino de preentrenamiento NDWS existe
  en el código pero nunca se ejecutó contra datos reales.

### Entrenamiento del U-Net

- **El tiempo de entrenamiento sobre NDWS nunca se midió**: el único
  entrenamiento real fue sobre los 11 eventos pequeños de Chile (se
  registra por época en `history.csv`, columna `seconds`; del orden de
  1-5 s por época en una máquina sin GPU), más el smoke test sintético
  (`docs/model-card.md`). Cualquier estimación de horas/época sobre el
  NDWS completo en `docs/model-card.md` es un orden de magnitud, no una
  medición.
- **`batch_size=1` por defecto, sin soporte de relleno/recorte para
  batir eventos de distinto tamaño** -- entrenar con más de un evento
  de Chile por batch requeriría lógica de padding/cropping no
  construida en este plan (ver `docs/model-card.md`, "Tamaño de
  batch").
- **La decisión de fine-tuning (sin capas congeladas, learning rate
  reducido) no está validada empíricamente contra la alternativa
  (congelar el encoder)** -- es la opción más conservadora dado el
  desajuste de dominio NDWS/Chile, pero ninguna corrida real comparó
  ambas estrategias en este proyecto todavía.
- **`finetune` no resume el estado del optimizador de `pretrain`** --
  cada fase crea un `Adam` nuevo dentro de `train_model`. Es una
  simplificación deliberada (retomar los momentos de Adam ajustados a
  la tasa de aprendizaje de preentrenamiento no tiene un beneficio
  claro al cambiar de fase/LR), no un descuido, pero significa que
  `finetune` no es un "continuar entrenando" literal, solo una
  inicialización de pesos.
- **`ChileFinetuneDataset` con `batch_size > 1` sobre eventos de
  distinto tamaño espacial fallaría en el collate de `DataLoader`** --
  no probado ni protegido explícitamente, porque `batch_size=1` (el
  default) lo evita por construcción; ver el punto de batch_size más
  arriba.
- **`pretrain`/`finetune` ahora EXIGEN un split de val no vacío
  (corregido en la revisión final del 2026-09-29 -- antes, un val
  vacío hacía que `val_loss` se reportara como `0.0` fabricado,
  arruinando early stopping y dejando `best.pt` sin entrenar de
  verdad, en silencio)**. Esto significa que, con muy pocos eventos de
  Chile (`features/dataset/split.py` da `val: []` con menos de 3
  eventos), `finetune` simplemente se niega a correr en vez de
  entrenar con una validación degradada -- correcto para no fabricar
  una métrica, pero significa que el proyecto necesita un mínimo de
  eventos reales de Chile antes de poder hacer fine-tuning en
  absoluto. Ver `docs/decisions.md`.
- **`pretrain --max-samples` es un tope manual, no una solución de
  streaming real**: sin él, cargar el dataset NDWS oficial completo
  (18.545 chips, ~4.1 GB medidos, ver `docs/model-card.md`) en una
  máquina de 8 GB sin GPU puede agotar la memoria antes de la primera
  época. El usuario debe elegir el tope a mano; no hay un `Dataset`
  perezoso indexado por posición de archivo (requeriría escanear los
  shards TFRecord una vez para construir un índice de offsets, no
  construido en este plan).
- **`SmallUNet` falla con un `RuntimeError` de PyTorch poco claro (no
  un `ValueError` con mensaje propio) si `H` o `W` es menor que
  `2**depth`** -- verificado con `depth=3` sobre una grilla de 9x7.
  Los eventos reales de Chile (`features/dataset/pipeline.py`,
  `DEFAULT_CONTEXT_BUFFER_M`) son suficientemente grandes en la
  práctica, pero `depth` es un flag de CLI y nada impide pasarlo con
  una grilla más chica.
- **`last.pt` puede registrar un `best_val_loss` desactualizado**: se
  guarda antes de la comparación de esa misma época contra el mejor
  histórico, así que en una corrida de una sola época `last.pt` queda
  con `best_val_loss=inf` en vez del valor real de esa época. Cosmético
  (no afecta qué checkpoint es "best", ni su contenido), pero confuso
  si alguien inspecciona `last.pt` directamente.
- **Reentrenar sobre un `run_dir` ya usado mezcla dos corridas**:
  `history.csv` se trunca al empezar, pero `best.pt` de la corrida
  anterior sobrevive si la nueva corrida nunca lo supera -- el
  directorio puede terminar describiendo el historial de la corrida B
  con el checkpoint de la corrida A. No se valida que `run_dir` esté
  vacío.
- **`finetune` no valida que `in_channels` del checkpoint coincida con
  el tensor de Chile antes de fallar** -- si algún día cambia
  `CHANNEL_ORDER` o se usa un checkpoint de otro esquema de canales, el
  error aparece como una forma incompatible dentro de la primera
  `Conv2d`, no como un mensaje que nombre el desajuste.
- **La pérdida promedio por época promedia sobre BATCHES, no sobre
  muestras** -- con `batch_size=1` (el default de producción) esto es
  irrelevante (cada batch es una muestra), pero con un `batch_size`
  mayor y un dataset cuyo tamaño no es múltiplo exacto, el último batch
  (más chico) pesa lo mismo que los demás en el promedio.

### Calibración y ensamble

- **La tabla antes/después de `docs/calibration.md` es de un checkpoint
  de FIXTURE sintético** (`pyrocast-calibrate run --fixture`): prueba
  que el pipeline funciona de punta a punta, no que el U-Net real esté
  bien calibrado. La calibración del checkpoint real
  (`runs/finetune_2026_v3`, `--chile-val`, solo 2 eventos de val) está
  en `docs/results.md` sección 3.
- **Sin split de calibración separado del de validación** -- ver
  `docs/decisions.md`. El ECE post-calibración de `make calibrate` da
  0.0000 exacto (1024 celdas de fixture), y esto es estructural, NO un
  artefacto de pocas muestras -- verificado con hasta 100.000 muestras
  sintéticas, sigue siendo de orden `1e-17`. Agregar MÁS datos al mismo
  set de ajuste-y-evaluación no reduce este efecto; solo evaluar sobre
  un split de calibración SEPARADO del usado para ajustar lo haría. No
  construido en este plan porque el enunciado pide explícitamente
  ajustar y comparar "sobre el set de validación" (ver
  `docs/decisions.md`).
  `docs/results.md` reporta además la calibración sobre test (fuera de muestra) para no presentar el ECE≈0 de val como evidencia.
- **`CalibratedUNet` no valida `in_channels` contra el tensor de
  entrada antes de fallar** -- mismo patrón (y misma limitación
  todavía sin resolver) que `models/deep/train.py::finetune`, ya
  ledgeado en una revisión anterior.
- **El día 0 de `CalibratedUNet.predict` y de `CellularAutomatonModel.predict`
  no son bit-idénticos, pese a compartir el mismo convenio de "el día 0
  es el ancla conocida"**: `CalibratedUNet` devuelve el `fire_mask`
  real del día 0 sin modificar; `CellularAutomatonModel` (por cómo está
  estructurado su bucle de simulación, `models/cellular_automata/simulate.py`,
  ya revisado y aceptado en una revisión anterior) devuelve para el día
  0 el ancla MÁS una capa de probabilidad de ignición hacia celdas
  vecinas no quemadas -- un artefacto preexistente de esa simulación,
  no introducido por esta revisión. La diferencia es pequeña (una sola
  celda de margen) pero real; no se modificó el autómata celular ya
  aceptado para "emparejar" este detalle.
- **`_checkpoint_fingerprint` carga el checkpoint completo en memoria
  para hashear su `state_dict`** (en vez de leer bytes del archivo en
  streaming) -- razonable para los checkpoints diminutos de este
  proyecto, pero costoso si algún día se usan checkpoints reales de
  cientos de MB en la máquina sin GPU de `docs/model-card.md`.
- **El CLI de calibración no soporta `--max-samples` para la ruta NDWS
  real** (`--checkpoint` + `--shard-dir`), a diferencia de
  `pyrocast-train pretrain`, que sí lo tiene por la misma razón (el
  dataset NDWS completo pesa ~4.1 GB, ver `docs/model-card.md`).
- **Ensamble CA + U-Net: pesos ajustados sobre un val contaminado**: el
  peso del blend y los coeficientes del stacking (`models/deep/ensemble.py`)
  se ajustan sobre los 2 eventos de val, los mismos que calibraron el
  calibrador isotónico del U-Net: las salidas del U-Net ahí son
  optimistas y sesgan el ensamble a favorecerlo. Con n=2 en val y n=2 en
  test no hay validación cruzada posible; la ventaja del stacking
  (`docs/backtest-2026.md` sección 8) no es concluyente y por eso el
  ensamble no es el modelo por defecto de `serving/`.

### Servido: API y mapa web (`serving/`)

- **`/predict` no pronostica: solo usa clima ya procesado**: ERA5-Land es
  un reanálisis histórico, no un pronóstico, y la API no lo descarga bajo
  demanda (una solicitud a CDS tarda minutos y requiere credenciales).
  Fechas recientes o futuras sin clima procesado devuelven
  `weather_unavailable`. Un uso real de pronóstico necesitaría una fuente
  de pronóstico meteorológico (p. ej. GFS/ECMWF IFS) que no está integrada.
- **Ignición de una sola celda, horizonte máximo de 7 días y
  probabilidades no calibradas**: `/predict` marca solo la celda de 250 m
  que contiene el punto (no un perímetro), y el modelo servido (autómata
  celular) no está calibrado: sus probabilidades son un puntaje relativo
  (`docs/api.md`, `docs/backtest-2026.md`).
- **Caché de predicciones en memoria, sin expiración ni invalidación**: se
  pierde al reiniciar y no se entera de que se re-ingirieron datos
  (`docs/api.md`).
- **La API no tiene autenticación ni límite de tasa**: pensada para uso
  local o de investigación. `/active-fires` consume la `FIRMS_MAP_KEY` del
  servidor (con caché de 10 min por la cuota de 5000 transacciones/10 min).
  CORS no se habilita por defecto; no hay CSP.
- **La interfaz web no se probó con un navegador automatizado**: hay tests
  de servidor (página, estáticos, endpoints) y la sintaxis del JS se
  verificó, pero el flujo clic → predicción → deslizador solo se validó
  contra las respuestas de la API, no con un navegador real. Depende de
  CDN externos (Leaflet en unpkg con SRI, teselas de OpenStreetMap).

### Ingesta: robustez y seguridad

- **La clasificación de errores de ERA5-Land y Sentinel-2 no se verificó
  contra errores reales**: cuota agotada, credenciales inválidas y fallas
  transitorias se distinguen por el texto del error (`cdsapi`) o por el
  código HTTP (`openeo`), cubiertos con fakes; nunca se forzó una cuota
  real. Un mensaje de formato inesperado cae en el error genérico (sigue
  siendo un mensaje claro, sin traceback ni credenciales).
- **Los reintentos están acotados y una falla persistente aborta el
  comando**: es deliberado (seguir con un tile o mes faltante dejaría un
  hueco silencioso). Solo un 404 de tile (DEM, WorldCover) se tolera. Lo
  ya descargado queda en caché y se reutiliza (tiles, tramos mensuales
  de ERA5, consultas de FIRMS; para FIRMS el CLI indica desde qué fecha
  reanudar).
- **`pip-audit` solo detecta vulnerabilidades conocidas en las
  dependencias bloqueadas (`uv.lock`)**: no escanea la imagen Docker ni
  el código propio, y no reemplaza una revisión de seguridad.

## Resueltas (historial)

- ~~`shared.model_protocol.FireSpreadModel` no está implementado para SmallUNet~~ -- resuelto: `CalibratedUNet` (`models/deep/calibration.py`) lo implementa (sigmoid, calibración, predicción autorregresiva).
- ~~El CLI de calibración no soporta calibrar contra un set de validación real de eventos de Chile~~ -- resuelto: `pyrocast-calibrate run --checkpoint ... --chile-val`.
- ~~`pyrocast-models backtest` solo corría el autómata celular y no registraba comando ni commit~~ -- resuelto: `--model unet|blend|stacking`, y todo resultado de `bench/results/` incluye `"command"` y `"git_commit"`.

## Herramienta de investigación

Herramienta de investigación. No usar para decisiones operativas de
combate de incendios sin validación de CONAF/SENAPRED.
