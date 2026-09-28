# Limitaciones conocidas

Este documento se actualiza con cada hallazgo real de la evaluación
contra incendios de Chile. Nunca se suaviza ni se elimina una métrica
negativa para que el proyecto "se vea mejor" (ver CLAUDE.md).

## Limitaciones de diseño (conocidas desde el bootstrap, no hallazgos de evaluación)

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
- **Fuel-type: áreas urbanas asumidas no combustibles**: la clase
  Built-up de WorldCover se mapea a `FUEL_URBANO_NO_COMBUSTIBLE`
  (`ingestion/worldcover/fuel_type.py`) — un supuesto de v1, no una
  verificación empírica. Esto significa que el modelo, tal como está,
  no puede representar la propagación de fuego hacia la interfaz
  urbano-forestal (WUI), que es precisamente el escenario detrás de las
  muertes y viviendas destruidas que motivan este proyecto (ver
  CLAUDE.md). Encontrado en la revisión final del 2026-09-26.
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
  contra el historial real de incendios de Chile — ese ajuste
  corresponde a `models/evaluation/` (backtesting), sin implementar.
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
  55 celdas encendidas a 1, sin ningún aviso. `models/cellular_automata/`
  todavía no está conectado a `features/dataset/` -- este riesgo no se
  ha materializado en producción, pero queda cerrado antes de esa
  integración en vez de esperar a que alguien lo redescubra.
- **Backtest: la evaluación del día 0 es tautológica por construcción**:
  `CellularAutomatonModel.predict` siembra `initial_burning` desde el
  propio `fire_mask` del día 0 del evento (no hay "día -1" del que
  sembrar) -- así que la predicción del día 0 para las celdas ya en
  llamas coincide con la verdad por definición, no porque el modelo haya
  "acertado" nada. Incluir el día 0 en las métricas agregadas del
  backtest sesga (levemente, hacia arriba) el desempeño reportado. Ver
  `docs/decisions.md`.
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
- **Ningún tiempo de entrenamiento real sobre NDWS o sobre eventos
  reales de Chile fue medido en esta sesión** -- solo el smoke test
  sintético (medido, ~8.4s, ver `docs/model-card.md`). El tiempo real
  depende del tamaño real de cada evento de Chile (variable, no
  medido) y del volumen real de NDWS descargado (no disponible en este
  entorno sin red). Cualquier estimación de horas/época en
  `docs/model-card.md` es un orden de magnitud, no una medición.
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
- **`shared.model_protocol.FireSpreadModel` no está implementado para
  SmallUNet** -- el modelo devuelve logits `(batch, 1, H, W)`, no
  `(day, y, x)` en `[0, 1]` como el Protocol exige; un wrapper que lo
  implemente (aplicar sigmoid, adaptar la forma) queda para cuando se
  necesite correr `models/evaluation/backtest.py` contra el U-Net.
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

## Herramienta de investigación

Herramienta de investigación. No usar para decisiones operativas de
combate de incendios sin validación de CONAF/SENAPRED.
