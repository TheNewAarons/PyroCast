# models/deep/public_dataset.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document and load "Next Day Wildfire Spread" (NDWS, Huot et
al., ICDM 2022) as a pretraining source for the future U-Net (P9-P11):
a dependency-free TFRecord reader, a channel-mapping adapter that
transforms NDWS samples into PyroCast's own `(day, channel, y, x)`
tensor schema (reusing `features.dataset.assemble` verbatim, not a
parallel implementation), and a train/val split of the public dataset
kept strictly separate from `features/dataset/split.py`'s Chile-event
split.

**Architecture:** `models/deep/tfrecord_reader.py` is a small,
domain-agnostic, dependency-free (stdlib only) reader for the TFRecord
container format + the `tf.train.Example` protobuf schema — verified
against the public TFRecord/protobuf wire-format specification and
round-trip-tested against a hand-written encoder (this environment has
no network access to a real Kaggle download to test against; that
caveat is documented, not hidden). `models/deep/public_dataset.py` is
the domain adapter: it knows NDWS's specific channel names (verified
against the dataset's own official export code, not guessed), converts
each to PyroCast's units/convention (documented per-channel, including
the two channels with NO equivalent), builds a `WorkGrid` +
`EventChannels` and calls `features.dataset.assemble.assemble_event_tensor`
directly (so the output tensor is byte-for-byte the same shape/dims/
channel-order/dtype convention as a real Chile event, not a lookalike),
and separately returns NDWS's own next-day label (no PyroCast tensor
channel is a "supervised label", so it travels alongside the tensor,
not inside it). `split_public_dataset` splits at the TFRecord-shard
level (the natural atomic unit here, same "never split below the
natural grouping" rule `features/dataset/split.py` already applies to
events), producing `{"train": [...], "val": [...]}` — no `test` key,
and no shared code path with the Chile split, so a caller cannot
accidentally mix the two.

**Tech Stack:** stdlib only for the TFRecord/protobuf reader (`struct`,
no `tensorflow`, no `tfrecord` pip package — both were considered and
rejected, see Global Constraints). `numpy` for the channel math.
`features` added as a new dependency of `models` (workspace source, no
version pin needed) to reuse `features.dataset.assemble.assemble_event_tensor`
and `features.grid.grid.WorkGrid` verbatim instead of re-implementing
the tensor-construction contract.

**Spec:** the user's request (quoted below), governed by
`/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md`.

```
Implementa models/deep/public_dataset.py.

1. Documenta en docs/public-dataset.md el dataset público "Next Day
   Wildfire Spread" (Huot et al.) como fuente de preentrenamiento:
   dónde se obtiene, su licencia, su esquema de canales y resolución, y
   cómo difiere del esquema de canales propio de PyroCast (P6).
   Verifica la disponibilidad y el formato actuales de ese dataset
   antes de escribir el loader; no asumas una URL o formato de archivo
   sin confirmarlo.
2. Un adaptador que cargue ese dataset externo (una vez descargado
   manualmente por el usuario, dado que puede requerir aceptar
   términos) y lo transforme al mismo esquema de tensor (orden de
   canales, normalización) que usa el dataset de PyroCast (P6),
   documentando explícitamente qué canales no tienen equivalente
   directo (por ejemplo, si el dataset público no tiene tipo de
   combustible detallado, documenta el mapeo o el relleno usado).
3. Split propio del dataset público (train/val), separado del split de
   eventos de Chile, para que el fine-tuning posterior no mezcle ambos
   de forma confusa.

Tests: el adaptador transforma un archivo de fixture pequeño (formato
del dataset público simulado) al esquema esperado, con las dimensiones
y el orden de canales correctos.
Criterios de aceptación: mypy y ruff limpios; docs/public-dataset.md
completo, incluyendo instrucciones para que cualquiera pueda obtener el
dataset por su cuenta (sin redistribuirlo en el repo).
```

## Research performed before writing this plan (verified, not assumed)

Verified via live web search/fetch before any code or doc was written
(sources cited in `docs/public-dataset.md`, Task 1):

- **Distribution**: primary channel is Kaggle,
  `kaggle.com/datasets/fantineh/next-day-wildfire-spread`, gated behind
  a free Kaggle account + accepting the dataset's terms (no anonymous
  download) — confirmed by multiple independent sources, including the
  dataset's own README pointer from the authors' Google Research
  export-code repository.
- **License**: CC BY 4.0 — confirmed by two independent sources
  (a third-party project's citation of the license, and a web-search
  synthesis of the Kaggle listing).
- **Format**: TFRecord, gzip-compressed shards (`*.tfrecord.gz`) —
  confirmed directly from the authors' own export-code repository
  (`google-research/google-research/simulation_research/next_day_wildfire_spread`),
  which also confirms the TFRecord parsing contract used by the
  official code: `tf.io.FixedLenFeature(shape=[64, 64], dtype=tf.float32)`
  per feature name (each feature stored as a flat length-4096 packed
  float32 array, reshaped to 64×64 on read).
- **Channels** (from the authors' own `constants.py`, fetched
  verbatim): 12 input channels — `elevation, pdsi, NDVI, pr, sph, th,
  tmmn, tmmx, vs, erc, population, PrevFireMask` — plus one label
  channel, `FireMask` (tomorrow's fire mask; NOT one of the 12 model
  inputs). Per-channel normalization stats `(min_clip, max_clip, mean,
  std)` were also fetched verbatim — NOT applied by this adapter (see
  Global Constraints: PyroCast's own P6 tensors store raw physical
  units, never z-scored, so applying NDWS's z-score stats here would
  make this adapter's output inconsistent with P6's own convention,
  not consistent with it).
- **Split**: the paper's own train/eval/test split is 14,979 / 1,877 /
  1,689 chips (8:1:1 by week, 2012–2020) — confirmed by an independent
  citation. This plan does NOT reuse that exact split (task 3 asks for
  PyroCast's own train/val split, kept separate from the Chile split,
  operating over whatever shard files the user actually has on disk —
  see Task 4).
- **Exact current Kaggle file-naming convention could not be confirmed
  with full confidence** (Kaggle's dataset page requires an
  authenticated browser session that this environment's fetch tools
  cannot reach; multiple secondary sources hedge on this point too,
  one explicitly saying to "look for `X.npy` + `y.npy`, or sharded
  `shard_*.npz`" as an alternative to TFRecord). This plan does NOT
  hardcode a Kaggle filename pattern — `models/deep/public_dataset.py`
  takes an explicit list/glob of local file paths from the caller
  (never constructs a URL or a guessed filename itself), so whatever
  the user actually has on disk after downloading works, as long as it
  is the official TFRecord format. `docs/public-dataset.md` documents
  this uncertainty honestly instead of asserting a specific filename
  pattern as fact.

## Global Constraints

- **No `tensorflow` or `tfrecord` pip dependency.** `models/`
  currently depends on `torch`, not `tensorflow` — adding a second,
  much larger deep-learning framework as a dependency purely to read
  one file format is not justified (CLAUDE.md: "No agregues
  dependencias sin justificarlas"). `models/deep/tfrecord_reader.py`
  implements the TFRecord container framing (length + masked CRC32C +
  payload) and a minimal `tf.train.Example` protobuf decoder (only the
  wire types this schema actually uses: varint and length-delimited)
  by hand, stdlib only. This mirrors a real prior-art solution found
  during research (a third-party project's README states it built "a
  dependency-free TFRecord parser (no TensorFlow, no `tfrecord` pip
  package — the latter has a known protobuf conflict)" for this exact
  dataset).
- **CRC32C is implemented and used, not skipped.** A large Kaggle
  download that gets truncated or corrupted should fail loudly at the
  first bad record, not silently return partial/garbage data. The
  masking formula (`((crc >> 15) | (crc << 17) + 0xa282ead8) mod
  2^32`) is the standard TFRecord masked-CRC32C scheme. **Honesty
  caveat, stated in both the code and `docs/public-dataset.md`**: this
  implementation is verified by round-tripping through a hand-written
  encoder in this same codebase (`models/tests/test_tfrecord_reader.py`)
  — this environment has no network access to validate it against a
  real Kaggle-downloaded file, so a byte-for-byte compatibility claim
  against real NDWS shards is NOT made; only "correct per the publicly
  documented format, and internally self-consistent" is claimed.
- **Output tensor is PyroCast's real tensor, not a lookalike.**
  `models/deep/public_dataset.py` imports and calls
  `features.dataset.assemble.assemble_event_tensor` directly (same
  `CHANNEL_ORDER`, same `EventChannels` dataclass, same `WorkGrid`
  contract) — it does not re-implement tensor construction. This
  requires adding `features` as a new dependency of `models`, which is
  the documented "mainline" direction in CLAUDE.md's architecture
  diagram (`ingestion → features → models → evaluation → serving`),
  already established as acceptable in this session's prior reviews —
  not a new precedent, but still gets one `docs/decisions.md` line
  since CLAUDE.md requires justifying every new dependency.
- **No normalization/z-scoring applied.** Verified before writing this
  plan: neither `features/dataset/assemble.py` nor
  `features/dataset/resample.py` normalizes any channel — PyroCast's
  P6 tensors store raw physical values (meters, Kelvin, m/s, etc.).
  This adapter converts NDWS's raw values to PyroCast's UNITS (see
  per-channel mapping below) but does not z-score or min-max scale
  anything, matching P6's own convention exactly.
- **Per-channel mapping, verified against the actual source code each
  PyroCast channel comes from (not assumed):**
  - `elevation` → NDWS `elevation` directly. Both meters (NDWS: SRTM;
    PyroCast: Copernicus DEM GLO-30, `ingestion/dem/`) — same physical
    quantity, same unit, direct copy.
  - `slope_deg`, `aspect_deg` → **derived from NDWS's own `elevation`**
    channel via the exact Horn (1981) 3×3-kernel formula PyroCast
    already uses in `features/terrain/slope_aspect.py::compute_slope_aspect`,
    called with `cellsize_x = cellsize_y = 1000.0` (NDWS's native 1 km
    resolution, confirmed in Research above) — reused via direct
    import (`features` is already a dependency for the tensor
    assembly, so this is not an extra cost), not duplicated.
  - `wind_u`, `wind_v` → **derived** from NDWS `th` (wind direction,
    degrees, meteorological convention — the direction the wind is
    blowing FROM, confirmed against GRIDMET's own variable
    documentation) and `vs` (wind speed, m/s) via the standard
    from-direction-to-vector conversion `u = -speed·sin(θ)`, `v =
    -speed·cos(θ)`. PyroCast's own `wind_u`/`wind_v`
    (`features/weather/derive.py`) are the raw ERA5 `u10`/`v10`
    components, which point where the wind blows TO (the opposite
    convention from `th`) — verified by reading
    `features/weather/derive.py`'s own docstring before writing this
    mapping, not assumed.
  - `temperature` → **derived**: `mean(tmmn, tmmx)` (both already
    Kelvin — GRIDMET's native unit, matching PyroCast's `t2m`-sourced
    `temperature` channel, which is also Kelvin, verified against
    `features/weather/derive.py`). PyroCast stores a single daily
    temperature (ERA5 `t2m`, not a min/max pair); NDWS has no single
    "the" temperature, so the mean of its min/max is the least-lossy
    single value.
  - `relative_humidity` → **derived, and NOT a direct formula reuse**:
    NDWS's `sph` is SPECIFIC humidity (kg/kg, a different physical
    quantity from relative humidity — verified this is NOT the same
    input PyroCast's own `relative_humidity_approx(temp_k, dewpoint_k)`
    (`features/weather/derive.py`) expects, which needs a dewpoint
    PyroCast doesn't have from NDWS). Converts specific humidity to
    approximate relative humidity via the standard WMO vapor-pressure
    formula: `e = q·p / (0.622 + 0.378·q)` (actual vapor pressure from
    specific humidity `q` and pressure `p`), `e_sat = 611.2 ·
    exp(17.625·T_C / (243.04 + T_C))` (Magnus-Tetens saturation vapor
    pressure — same Alduchov & Eskridge 1996 coefficients PyroCast's
    own `relative_humidity_approx` already uses, for consistency),
    `RH = 100 · e / e_sat`, clipped to `[0, 100]`. **Approximation
    documented as a limitation**: `p` is fixed at standard sea-level
    pressure (101325 Pa) since NDWS provides no surface pressure field
    — real surface pressure varies with elevation (NDWS's own
    `elevation` channel could estimate it via the barometric formula,
    not done here — YAGNI, this is pretraining data, not the final
    evaluation signal).
  - `precipitation` → NDWS `pr` is in **millimeters** (GRIDMET);
    PyroCast's `precipitation` channel is in **meters** — verified by
    reading `features/weather/derive.py`, which stores ERA5-Land's
    native `tp` field UNCONVERTED (no `*1000`/`/1000` anywhere in that
    file). Without this unit check this would have been a silent
    1000× error. Converts via `pr / 1000.0`.
  - `ndvi` → NDWS `NDVI` is scaled by 10,000 (integer-friendly MODIS/
    VIIRS convention, confirmed by `constants.py`'s own clip range of
    roughly ±10,000); PyroCast's `ndvi` (`features/vegetation/ndvi.py`)
    is the raw `(NIR-RED)/(NIR+RED)` ratio, range `[-1, 1]`. Converts
    via `NDVI / 10000.0`, clipped to `[-1, 1]`.
  - `fuel_type` → **NO equivalent in NDWS at all** (no land-cover/
    vegetation-type channel of any kind in the 12 input channels).
    Filled with the constant `shared`-independent value `99`
    (`ingestion/worldcover/fuel_type.py::FUEL_TYPE_UNKNOWN`, duplicated
    here as a literal rather than importing `ingestion` — `models`
    depending on `ingestion` would invert the established dependency
    direction, so this one constant is duplicated with a comment
    pointing at its source, same pattern as `models/cellular_automata/rules.py`'s
    `DEFAULT_FUEL_FLAMMABILITY`). **Explicitly not derived from NDVI**
    (fabricating a land-cover class from a vegetation-greenness index
    would be inventing a categorical signal NDWS never measured — a
    worse honesty violation than admitting "unknown").
  - `fire_mask` → NDWS `PrevFireMask` (the known state at the sample's
    reference day — the closest analog to "day 0's own fire_mask" that
    `models/cellular_automata/model.py` already reads from a real
    PyroCast tensor). NDWS's mask uses `{-1: uncertain/no-observation,
    0: no fire, 1: fire}`; PyroCast's is `{0, 1}`. Maps via
    `np.clip(mask, 0.0, 1.0)` — uncertain is conservatively treated as
    "no observed fire", documented as a limitation (NOT the same as
    genuinely observing no fire).
  - **`FireMask` (NDWS's next-day label) has no PyroCast tensor
    channel at all** — P6's schema has no "label" concept (a model
    learns to predict a *later day's* `fire_mask` from *earlier days'*
    channels within one multi-day tensor; NDWS is fundamentally a
    single-timestep before/after pair, not a multi-day sequence). It is
    returned alongside the tensor as a separate array on
    `PublicDatasetSample.next_day_fire_mask`, not stuffed into the
    11-channel schema, same `{-1,0,1}→{0,1}` clip as `PrevFireMask`.
  - NDWS's `pdsi` (drought index) and `erc` (fire-danger index) and
    `population` have **no PyroCast equivalent and are not used to
    derive anything** — genuinely dropped. Documented in
    `docs/public-dataset.md` as available-but-unused, not silently
    discarded without a trace.
- **The `WorkGrid`/tensor `day`/coordinates NDWS samples get are
  nominal, not real geography** — NDWS chips are US CONUS patches, not
  Chile-projected. `resolution_m=1000.0` (NDWS's real resolution) is
  meaningful; `crs`, `transform` origin, and the placeholder `day` date
  are NOT — documented explicitly wherever they appear (docstring,
  `docs/public-dataset.md`) so nobody mistakes a public-dataset
  tensor's `x`/`y` coordinates for real Chile coordinates.
- **Split granularity is the TFRecord shard file, not the individual
  record** — reading every record up front just to assign a split
  would mean reading the entire (potentially multi-GB) dataset twice
  for no benefit; one shard file is the natural atomic unit here, the
  same "never split below the natural grouping" principle
  `features/dataset/split.py` already applies to fire events.

## Review Focus

- A TFRecord shard file that is truncated mid-record (a common failure
  mode for a large, manually-downloaded Kaggle file) — the reader must
  raise a clear error, not silently return a partial record or hang.
- A record missing one of the 12 expected NDWS feature names (a
  version-skew risk if Kaggle's schema ever changes) — must raise a
  clear error naming the missing feature, not a `KeyError` with no
  context.
- `relative_humidity` conversion at the physical extremes (`sph=0`,
  bone dry) — must not divide by zero or produce a negative percentage.
- `split_public_dataset` called with fewer than 2 shard files — must
  not silently produce an empty `val` (same class of bug
  `features/dataset/split.py` was fixed for earlier this project, for
  a different dataset).
- The NDVI/precipitation/humidity unit conversions are exactly the
  kind of silent-order-of-magnitude bug a reviewer might not think to
  check without reading both source modules' actual units — each has
  its own hand-verified test with an exact expected numeric value, not
  just a "runs without crashing" test.

---

## Task 1: `docs/public-dataset.md`

**Files:**
- Create: `docs/public-dataset.md`

**Interfaces:**
- Produces: the document Task 2's code comments and Task 3's docstrings
  point back to for the full narrative (download instructions,
  license, channel table, mapping table). No code interface — this is
  the authoritative source Tasks 2–3 must stay consistent with.

- [ ] **Step 1: Write the document**

Create `docs/public-dataset.md`:

```markdown
# Dataset público de preentrenamiento: Next Day Wildfire Spread

`models/deep/public_dataset.py` carga este dataset externo como fuente
de preentrenamiento para el futuro U-Net (P9-P11) — el dataset propio
de PyroCast (P6, Chile) es demasiado pequeño por sí solo para entrenar
una red profunda desde cero.

**Aviso de honestidad (CLAUDE.md):** este documento describe un dataset
de OTRO PAÍS (Estados Unidos continental) usado únicamente para
preentrenamiento. Ningún resultado reportado sobre incendios chilenos
debe mezclar métricas de este dataset con las de `docs/results.md` sin
decirlo explícitamente.

## Qué es

"Next Day Wildfire Spread" (NDWS), Huot, Hu, Goyal, Sankar, Ihme y
Chen, "Next Day Wildfire Spread: A Machine Learning Data Set to Predict
Wildfire Spreading from Remote-Sensing Data", IEEE Transactions on
Geoscience and Remote Sensing / ICDM 2022 (preprint:
[arXiv:2112.02447](https://arxiv.org/abs/2112.02447)). ~18.545 recortes
("chips") de 64×64 píxeles a 1 km de resolución sobre incendios reales
de EE.UU. continental (2012-2020), cada uno con 12 variables de entrada
del día `t` y la máscara de fuego del día `t+1` como etiqueta.

## Cómo obtenerlo (no redistribuido en este repo)

1. Crear una cuenta gratuita en [Kaggle](https://www.kaggle.com/).
2. Ir a [kaggle.com/datasets/fantineh/next-day-wildfire-spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)
   y aceptar los términos del dataset (Kaggle exige esto antes de
   permitir la descarga — no hay forma de automatizarlo, ni debería
   automatizarse).
3. Descargar con la Kaggle CLI (`pip install kaggle`, configurar
   `~/.kaggle/kaggle.json` con tu API token de Kaggle):
   ```
   kaggle datasets download -d fantineh/next-day-wildfire-spread -p <dónde-quieras>
   unzip <dónde-quieras>/next-day-wildfire-spread.zip -d <dónde-quieras>
   ```
4. Apuntar `models/deep/public_dataset.py` a los archivos `.tfrecord`
   (o `.tfrecord.gz`) descomprimidos — el loader recibe una lista de
   rutas o un patrón glob explícito, nunca asume un nombre de archivo
   ni construye una URL por su cuenta.

**Nota de incertidumbre, honesta a propósito**: el nombre EXACTO de los
archivos dentro del zip de Kaggle no se pudo confirmar con
certeza total al escribir este documento (la página de Kaggle requiere
una sesión de navegador autenticada que no se pudo inspeccionar
directamente) — puede variar según cuándo se descargue. El formato
TFRecord en sí (confirmado contra el código de exportación oficial de
los autores) y el esquema de features (confirmado contra su
`constants.py`) son estables independientemente del nombre de archivo
exacto. Si el archivo descargado NO es TFRecord (algunas re-subidas de
terceros en Kaggle ofrecen versiones pre-convertidas a `.npy`/`.npz`),
`models/deep/public_dataset.py` no lo soporta — habría que agregar un
loader separado para ese formato, no asumirlo aquí.

**Código de exportación original** (Google Earth Engine, para
reconstruir o actualizar el dataset desde cero, no necesario para usar
el CSV ya publicado): [google-research/google-research/simulation_research/next_day_wildfire_spread](https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread)

## Licencia

**CC BY 4.0** (Creative Commons Attribution 4.0). Permite uso,
modificación y redistribución con atribución — pero este proyecto NO
redistribuye el dataset (ver sección anterior): cada usuario lo
descarga por su cuenta directamente desde Kaggle.

## Formato

TFRecord (posiblemente comprimido con gzip, `.tfrecord.gz`), el formato
de contenedor binario de TensorFlow — cada archivo es una secuencia de
registros `longitud + CRC32C enmascarado + datos + CRC32C enmascarado`,
donde `datos` es un mensaje protobuf `tf.train.Example` serializado.
`models/deep/tfrecord_reader.py` implementa un lector de este formato
SIN depender de `tensorflow` ni del paquete `tfrecord` de PyPI (ver
`docs/decisions.md`) — ver ese módulo para el detalle exacto del
parseo, verificado contra la especificación pública del formato.

Cada `Example` trae 13 features, cada una un `float_list` empaquetado
de 4096 valores (64×64 aplanado, reconstruido en ese orden por
`tfrecord_reader.py`):

| Feature NDWS | Significado | Fuente | Unidad |
|---|---|---|---|
| `elevation` | Elevación | SRTM | metros |
| `pdsi` | Índice de sequía Palmer | GRIDMET | índice adimensional |
| `NDVI` | Vegetación (escalado ×10000) | VIIRS VNP13A1 | adimensional (÷10000 → [-1,1]) |
| `pr` | Precipitación | GRIDMET | mm |
| `sph` | Humedad específica | GRIDMET | kg/kg |
| `th` | Dirección del viento (desde dónde sopla) | GRIDMET | grados |
| `tmmn` | Temperatura mínima diaria | GRIDMET | Kelvin |
| `tmmx` | Temperatura máxima diaria | GRIDMET | Kelvin |
| `vs` | Velocidad del viento | GRIDMET | m/s |
| `erc` | Energy Release Component (peligro de incendio) | NFDRS | índice (BTU/pie²) |
| `population` | Densidad poblacional | GPWv4 | personas/km² |
| `PrevFireMask` | Máscara de fuego día `t` | MOD14A1 V6 | {-1, 0, 1} |
| `FireMask` (**etiqueta**, no una de las 12 de entrada) | Máscara de fuego día `t+1` | MOD14A1 V6 | {-1, 0, 1} |

Estadísticas de normalización oficiales (min/max de recorte, media,
desviación estándar por canal) están publicadas en el `constants.py`
de los autores, pero **no se aplican en este adaptador** — ver
`docs/decisions.md`: los tensores propios de PyroCast (P6) guardan
valores físicos crudos, sin normalizar, y este adaptador iguala esa
convención en vez de la de NDWS.

## Split oficial del paper (no el que usa este proyecto)

14.979 train / 1.877 val / 1.689 test, separado por semanas
(2012-2020) en proporción 8:1:1. `models/deep/public_dataset.py::split_public_dataset`
NO reutiliza este split — genera su propio split determinista
train/val sobre los archivos que el usuario efectivamente tenga en
disco (ver Task 4 del plan de implementación), separado por completo
del split de eventos de Chile (`features/dataset/split.py`) para que
el fine-tuning posterior no mezcle ambos splits de forma confusa.

## Cómo difiere del esquema de PyroCast (P6)

`features/dataset/assemble.py::CHANNEL_ORDER` tiene 11 canales:
`elevation, slope_deg, aspect_deg, wind_u, wind_v, temperature,
relative_humidity, precipitation, ndvi, fuel_type, fire_mask`. NDWS no
tiene el mismo esquema — la tabla completa de mapeo (qué se copia
directo, qué se deriva, qué canal de PyroCast queda relleno sin
equivalente, y las fórmulas exactas) está documentada en el docstring
de `models/deep/public_dataset.py`, no duplicada aquí para evitar que
ambas fuentes se desincronicen. Resumen:

- **Copiado directo** (misma magnitud, misma unidad): `elevation`.
- **Derivado con una fórmula de PyroCast ya existente**: `slope_deg`,
  `aspect_deg` (Horn 1981, desde la elevación de NDWS).
- **Derivado con conversión de unidad/convención**: `wind_u`/`wind_v`
  (dirección+velocidad → componentes), `temperature` (promedio
  min/max), `relative_humidity` (humedad específica → relativa,
  aproximada), `precipitation` (mm → m), `ndvi` (÷10000).
- **Sin equivalente, relleno explícito**: `fuel_type` (NDWS no tiene
  ningún canal de cobertura de suelo/vegetación categórica — se rellena
  con el código "desconocido" de PyroCast, código 99, nunca inventado
  desde NDVI).
- **Sin canal correspondiente en el tensor de PyroCast en absoluto**:
  la etiqueta `FireMask` (día `t+1`) de NDWS — P6 no tiene concepto de
  "etiqueta" dentro del tensor; se devuelve por separado.
- **De NDWS, sin uso en absoluto**: `pdsi`, `erc`, `population` — no
  tienen equivalente y no se usan para derivar nada (documentado, no
  descartado en silencio).
```

- [ ] **Step 2: Read it back once, end to end, checking every claim against the real code once Task 4 is done**

This step is revisited at the end of Task 4 (the doc is written first
so Tasks 2–4's docstrings can point at it, but its channel-mapping
summary must match `transform_ndws_record`'s real code exactly once
written — see Task 4 Step 6).

- [ ] **Step 3: Commit**

```bash
git add docs/public-dataset.md
git commit -m "docs: add public-dataset.md for Next Day Wildfire Spread (NDWS)"
```

---

## Task 2: `models/deep/tfrecord_reader.py` — dependency-free TFRecord + tf.Example reader

**Files:**
- Create: `models/src/models/deep/tfrecord_reader.py`
- Test: `models/tests/test_tfrecord_reader.py`

**Interfaces:**
- Produces: `read_tf_examples(path: Path) -> Iterator[dict[str, np.ndarray]]`
  — yields one dict per `Example` record in the file, each value a 2D
  `float32` array (the flat `float_list` reshaped to a square — `side =
  sqrt(len(values))`, validated to be a whole number). Task 3 consumes
  this directly.
- Consumes: nothing from other tasks (self-contained, domain-agnostic).

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_tfrecord_reader.py`:

```python
"""Tests del lector TFRecord + tf.Example sin dependencias (sin
tensorflow, sin el paquete `tfrecord` de PyPI -- ver docs/decisions.md).
Verificado por round-trip contra un ENCODER escrito a mano en este mismo
archivo de test (no hay acceso de red en este entorno para probar
contra un archivo real descargado de Kaggle -- ver
docs/public-dataset.md, sección de incertidumbre)."""
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from models.deep.tfrecord_reader import CorruptTFRecordError, read_tf_examples

_CRC32C_POLY = 0x82F63B78  # polinomio Castagnoli reflejado


def _crc32c_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLY if crc & 1 else 0)
        table.append(crc)
    return table


_CRC32C_TABLE = _crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _mask_crc(crc: int) -> int:
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_length_delimited(field_number: int, payload: bytes) -> bytes:
    return _encode_tag(field_number, 2) + _encode_varint(len(payload)) + payload


def _encode_float_list_feature(values: np.ndarray) -> bytes:
    packed = struct.pack(f"<{values.size}f", *values.ravel().tolist())
    float_list = _encode_length_delimited(1, packed)  # FloatList.value (packed)
    return _encode_length_delimited(2, float_list)  # Feature.float_list


def _encode_example(features: dict[str, np.ndarray]) -> bytes:
    feature_entries = b""
    for name, values in features.items():
        key_bytes = _encode_length_delimited(1, name.encode("utf-8"))
        value_bytes = _encode_length_delimited(2, _encode_float_list_feature(values))
        entry = key_bytes + value_bytes
        feature_entries += _encode_length_delimited(1, entry)  # Features.feature (map entry)
    features_message = feature_entries
    example = _encode_length_delimited(1, features_message)  # Example.features
    return example


def _write_tfrecord(path: Path, records: list[dict[str, np.ndarray]]) -> None:
    with open(path, "wb") as f:
        for record in records:
            data = _encode_example(record)
            length_bytes = struct.pack("<Q", len(data))
            length_crc = struct.pack("<I", _mask_crc(_crc32c(length_bytes)))
            data_crc = struct.pack("<I", _mask_crc(_crc32c(data)))
            f.write(length_bytes + length_crc + data + data_crc)


def test_read_tf_examples_round_trips_a_single_record(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    original = {
        "elevation": np.arange(16, dtype="float32").reshape(4, 4),
        "NDVI": np.full((4, 4), 5000.0, dtype="float32"),
    }
    _write_tfrecord(path, [original])

    records = list(read_tf_examples(path))
    assert len(records) == 1
    np.testing.assert_array_equal(records[0]["elevation"], original["elevation"])
    np.testing.assert_array_equal(records[0]["NDVI"], original["NDVI"])


def test_read_tf_examples_round_trips_multiple_records(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    records_in = [
        {"elevation": np.zeros((2, 2), dtype="float32")},
        {"elevation": np.ones((2, 2), dtype="float32")},
        {"elevation": np.full((2, 2), 7.0, dtype="float32")},
    ]
    _write_tfrecord(path, records_in)

    records_out = list(read_tf_examples(path))
    assert len(records_out) == 3
    for expected, actual in zip(records_in, records_out, strict=True):
        np.testing.assert_array_equal(actual["elevation"], expected["elevation"])


def test_read_tf_examples_rejects_a_non_square_feature(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    # 6 valores -- no es un cuadrado perfecto (sqrt(6) no es entero).
    _write_tfrecord(path, [{"elevation": np.zeros(6, dtype="float32")}])
    with pytest.raises(CorruptTFRecordError, match="elevation"):
        list(read_tf_examples(path))


def test_read_tf_examples_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    _write_tfrecord(path, [{"elevation": np.zeros((2, 2), dtype="float32")}])
    truncated = path.read_bytes()[:-5]  # corta a mitad del último CRC
    path.write_bytes(truncated)
    with pytest.raises(CorruptTFRecordError):
        list(read_tf_examples(path))


def test_read_tf_examples_rejects_a_corrupted_length_crc(tmp_path):
    path = tmp_path / "fixture.tfrecord"
    _write_tfrecord(path, [{"elevation": np.zeros((2, 2), dtype="float32")}])
    raw = bytearray(path.read_bytes())
    raw[8] ^= 0xFF  # corrompe un byte del CRC de longitud (offset 8-11)
    path.write_bytes(bytes(raw))
    with pytest.raises(CorruptTFRecordError, match="CRC"):
        list(read_tf_examples(path))


def test_read_tf_examples_on_an_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "empty.tfrecord"
    path.write_bytes(b"")
    assert list(read_tf_examples(path)) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_tfrecord_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.deep.tfrecord_reader'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/deep/tfrecord_reader.py`:

```python
"""Lector del formato contenedor TFRecord + el esquema protobuf
`tf.train.Example`, sin depender de `tensorflow` ni del paquete
`tfrecord` de PyPI (este último tiene un conflicto de versión de
protobuf conocido) -- ver docs/decisions.md. Domain-agnostic: no sabe
nada de NDWS ni de PyroCast, solo sabe leer el formato binario. Ver
docs/public-dataset.md para el contexto de qué dataset lo usa.

Formato TFRecord (contenedor, por registro):
    uint64 longitud                              (little-endian)
    uint32 CRC32C enmascarado de los 8 bytes de longitud
    byte   datos[longitud]                       (un Example serializado)
    uint32 CRC32C enmascarado de datos

CRC32C (Castagnoli) se implementa y se VALIDA (no se omite) -- un
archivo grande descargado manualmente que llega truncado o corrupto
debe fallar ruidosamente en el primer registro malo, no devolver datos
parciales en silencio. La fórmula de "máscara" es la estándar de
TFRecord: `mask(crc) = ((crc >> 15) | (crc << 17)) + 0xa282ead8 (mod
2^32)`.

`tf.train.Example` (protobuf, solo los wire types que este esquema usa
-- varint y length-delimited, nunca fixed32/fixed64 sueltos, porque los
floats van "packed" dentro de un length-delimited):
    Example { Features features = 1; }
    Features { map<string, Feature> feature = 1; }   -- un map en wire
               format es un repeated de mensajes {key=1, value=2}
    Feature { oneof { FloatList float_list = 2; ... } }  -- solo
               float_list está implementado: es lo único que NDWS usa.
    FloatList { repeated float value = 1 [packed=true]; }  -- packed:
               los floats van concatenados como bytes crudos dentro de
               UN campo length-delimited, no como floats sueltos.

*** LIMITACIÓN DE VERIFICACIÓN, DOCUMENTADA A PROPÓSITO: este lector se
verificó por round-trip contra un encoder escrito a mano en
models/tests/test_tfrecord_reader.py, siguiendo la especificación
pública del formato. Este entorno no tiene acceso de red para
descargar un archivo real de Kaggle y probar compatibilidad byte a
byte contra él -- ver docs/public-dataset.md. ***
"""
import math
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_CRC32C_POLY = 0x82F63B78  # polinomio Castagnoli reflejado
_CRC_MASK_DELTA = 0xA282EAD8


class CorruptTFRecordError(ValueError):
    """El archivo TFRecord está truncado, tiene un CRC inválido, o un
    Example con una feature que no se puede interpretar como una
    grilla cuadrada."""


def _crc32c_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLY if crc & 1 else 0)
        table.append(crc)
    return table


_CRC32C_TABLE = _crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _mask_crc(crc: int) -> int:
    return (((crc >> 15) | (crc << 17)) + _CRC_MASK_DELTA) & 0xFFFFFFFF


def _read_exact(f: "object", n: int) -> bytes:
    data: bytes = f.read(n)  # type: ignore[attr-defined]
    if len(data) != n:
        raise CorruptTFRecordError(
            f"archivo TFRecord truncado -- se esperaban {n} bytes, se leyeron {len(data)}."
        )
    return data


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise CorruptTFRecordError("varint truncado dentro de un Example.")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_top_level_fields(data: bytes) -> Iterator[tuple[int, bytes | int]]:
    """Itera los campos de nivel superior de un mensaje protobuf,
    soportando solo wire type 0 (varint) y 2 (length-delimited) -- los
    únicos que el esquema de tf.train.Example usa."""
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:
            value, pos = _read_varint(data, pos)
            yield field_number, value
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            if pos + length > len(data):
                raise CorruptTFRecordError("submensaje length-delimited truncado.")
            yield field_number, data[pos : pos + length]
            pos += length
        else:
            raise CorruptTFRecordError(f"wire type {wire_type} no soportado.")


def _parse_float_list(feature_bytes: bytes) -> np.ndarray | None:
    for field_number, value in _iter_top_level_fields(feature_bytes):
        if field_number == 2 and isinstance(value, bytes):  # Feature.float_list
            for inner_field, inner_value in _iter_top_level_fields(value):
                if inner_field == 1 and isinstance(inner_value, bytes):  # FloatList.value
                    count = len(inner_value) // 4
                    return np.frombuffer(inner_value, dtype="<f4", count=count).astype("float32")
    return None


def _parse_example(data: bytes) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for field_number, features_bytes in _iter_top_level_fields(data):
        if field_number != 1 or not isinstance(features_bytes, bytes):
            continue  # Example.features
        for entry_field, entry_bytes in _iter_top_level_fields(features_bytes):
            if entry_field != 1 or not isinstance(entry_bytes, bytes):
                continue  # Features.feature map entry
            name = None
            flat: np.ndarray | None = None
            for inner_field, inner_value in _iter_top_level_fields(entry_bytes):
                if inner_field == 1 and isinstance(inner_value, bytes):
                    name = inner_value.decode("utf-8")
                elif inner_field == 2 and isinstance(inner_value, bytes):
                    flat = _parse_float_list(inner_value)
            if name is not None and flat is not None:
                side = math.isqrt(flat.size)
                if side * side != flat.size:
                    raise CorruptTFRecordError(
                        f"feature '{name}' tiene {flat.size} valores, que no es un "
                        f"cuadrado perfecto -- no se puede reformar a una grilla."
                    )
                result[name] = flat.reshape(side, side)
    return result


def read_tf_examples(path: Path) -> Iterator[dict[str, np.ndarray]]:
    with open(path, "rb") as f:
        while True:
            length_bytes = f.read(8)
            if length_bytes == b"":
                return
            if len(length_bytes) != 8:
                raise CorruptTFRecordError("longitud de registro truncada.")
            (length,) = struct.unpack("<Q", length_bytes)

            length_crc_bytes = _read_exact(f, 4)
            (length_crc,) = struct.unpack("<I", length_crc_bytes)
            if _mask_crc(_crc32c(length_bytes)) != length_crc:
                raise CorruptTFRecordError("CRC de longitud inválido -- archivo corrupto.")

            data = _read_exact(f, length)

            data_crc_bytes = _read_exact(f, 4)
            (data_crc,) = struct.unpack("<I", data_crc_bytes)
            if _mask_crc(_crc32c(data)) != data_crc:
                raise CorruptTFRecordError("CRC de datos inválido -- archivo corrupto.")

            yield _parse_example(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_tfrecord_reader.py -v`
Expected: `6 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/deep/tfrecord_reader.py models/tests/test_tfrecord_reader.py
git commit -m "feat: add dependency-free TFRecord + tf.Example reader (models/deep)"
```

---

## Task 3: `models/pyproject.toml` — add `features` dependency

**Files:**
- Modify: `models/pyproject.toml`

**Interfaces:**
- Produces: `models` can import `features.dataset.assemble` and
  `features.grid.grid`. Task 4 consumes this.

- [ ] **Step 1: Add the dependency**

In `models/pyproject.toml`, add `"features",` to `dependencies` and
`features = { workspace = true }` to `[tool.uv.sources]`:

```toml
dependencies = [
    "shared",
    "features",
    "numpy>=2.0",
    "torch>=2.4",
    "scikit-learn>=1.5",
    "typer>=0.12",
    "xarray>=2024.7",
    "zarr>=2.18",
]

[tool.uv.sources]
shared = { workspace = true }
features = { workspace = true }
```

- [ ] **Step 2: Sync**

Run: `uv sync --all-packages` (never a scoped `uv sync` — see Global
Constraints of every prior plan this session).
Expected: resolves cleanly, `models` now importable against `features`.

- [ ] **Step 3: Verify the import works**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models python3 -c "from features.dataset.assemble import assemble_event_tensor, CHANNEL_ORDER, EventChannels; from features.grid.grid import WorkGrid; print(CHANNEL_ORDER)"`
Expected: prints the 11-tuple `CHANNEL_ORDER`, no import error.

- [ ] **Step 4: Commit**

```bash
git add models/pyproject.toml uv.lock
git commit -m "chore: add features as a models dependency (reuse assemble_event_tensor)"
```

---

## Task 4: `models/deep/public_dataset.py` — channel-mapping adapter + split

**Files:**
- Create: `models/src/models/deep/public_dataset.py`
- Modify: `models/src/models/deep/__init__.py`
- Test: `models/tests/test_public_dataset.py`

**Interfaces:**
- Consumes: `models.deep.tfrecord_reader.read_tf_examples`,
  `features.dataset.assemble.{assemble_event_tensor, CHANNEL_ORDER,
  EventChannels}`, `features.grid.grid.WorkGrid`,
  `features.terrain.slope_aspect.compute_slope_aspect`.
- Produces: `PublicDatasetSample` (frozen dataclass: `tensor:
  xr.DataArray`, `next_day_fire_mask: np.ndarray`),
  `transform_ndws_record(record: dict[str, np.ndarray], sample_id: int)
  -> PublicDatasetSample`, `load_public_dataset_samples(paths:
  list[Path]) -> Iterator[PublicDatasetSample]`,
  `split_public_dataset(shard_paths: list[Path], seed: int = 42,
  train_frac: float = 0.85) -> dict[str, list[Path]]`.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_public_dataset.py`:

```python
"""Tests del adaptador NDWS -> esquema de tensor de PyroCast. El
fixture simula el formato del dataset público (un registro TFRecord de
verdad, escrito con el mismo encoder de test_tfrecord_reader.py) con
los 13 nombres de feature reales de NDWS, en una grilla pequeña (4x4,
no 64x64 -- las dimensiones no importan para probar el MAPEO de
canales, solo que se preserven)."""
import struct
from pathlib import Path

import numpy as np
import pytest
from features.dataset.assemble import CHANNEL_ORDER
from models.deep.public_dataset import (
    PublicDatasetSample,
    load_public_dataset_samples,
    split_public_dataset,
    transform_ndws_record,
)

_SIZE = 4


def _make_ndws_record(
    th_deg: float = 90.0,
    vs: float = 10.0,
    tmmn: float = 280.0,
    tmmx: float = 300.0,
    sph: float = 0.0,
    pr_mm: float = 5.0,
    ndvi_raw: float = 5000.0,
    prev_fire: float = 1.0,
    next_fire: float = 0.0,
) -> dict[str, np.ndarray]:
    def const(value: float) -> np.ndarray:
        return np.full((_SIZE, _SIZE), value, dtype="float32")

    return {
        "elevation": np.arange(_SIZE * _SIZE, dtype="float32").reshape(_SIZE, _SIZE),
        "pdsi": const(1.0),
        "NDVI": const(ndvi_raw),
        "pr": const(pr_mm),
        "sph": const(sph),
        "th": const(th_deg),
        "tmmn": const(tmmn),
        "tmmx": const(tmmx),
        "vs": const(vs),
        "erc": const(30.0),
        "population": const(5.0),
        "PrevFireMask": const(prev_fire),
        "FireMask": const(next_fire),
    }


def test_transform_ndws_record_produces_the_pyrocast_channel_order():
    record = _make_ndws_record()
    sample = transform_ndws_record(record, sample_id=1)
    assert isinstance(sample, PublicDatasetSample)
    assert list(sample.tensor.coords["channel"].values) == list(CHANNEL_ORDER)


def test_transform_ndws_record_produces_the_expected_shape():
    record = _make_ndws_record()
    sample = transform_ndws_record(record, sample_id=1)
    # (day=1, channel=11, y=4, x=4)
    assert sample.tensor.shape == (1, len(CHANNEL_ORDER), _SIZE, _SIZE)
    assert sample.next_day_fire_mask.shape == (_SIZE, _SIZE)


def test_transform_ndws_record_copies_elevation_directly():
    record = _make_ndws_record()
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    elevation = sample.tensor.values[0, channels.index("elevation")]
    np.testing.assert_array_equal(elevation, record["elevation"])


def test_transform_ndws_record_converts_precipitation_mm_to_meters():
    record = _make_ndws_record(pr_mm=5.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    precip = sample.tensor.values[0, channels.index("precipitation")]
    assert precip[0, 0] == pytest.approx(0.005)


def test_transform_ndws_record_converts_temperature_to_mean_of_min_max():
    record = _make_ndws_record(tmmn=280.0, tmmx=300.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    temperature = sample.tensor.values[0, channels.index("temperature")]
    assert temperature[0, 0] == pytest.approx(290.0)


def test_transform_ndws_record_converts_ndvi_by_dividing_by_ten_thousand():
    record = _make_ndws_record(ndvi_raw=5000.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    ndvi = sample.tensor.values[0, channels.index("ndvi")]
    assert ndvi[0, 0] == pytest.approx(0.5)


def test_transform_ndws_record_converts_wind_direction_and_speed_to_components():
    # th=90 (viento SOPLA DESDE el este) -> avanza hacia el oeste ->
    # componente u (este) NEGATIVA, v (norte) CERO.
    record = _make_ndws_record(th_deg=90.0, vs=10.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    wind_u = sample.tensor.values[0, channels.index("wind_u")]
    wind_v = sample.tensor.values[0, channels.index("wind_v")]
    assert wind_u[0, 0] == pytest.approx(-10.0, abs=1e-6)
    assert wind_v[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_transform_ndws_record_fills_fuel_type_with_the_unknown_code():
    record = _make_ndws_record()
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    fuel_type = sample.tensor.values[0, channels.index("fuel_type")]
    assert np.all(fuel_type == 99.0)


def test_transform_ndws_record_clips_uncertain_fire_mask_to_not_fire():
    record = _make_ndws_record(prev_fire=-1.0, next_fire=-1.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    fire_mask = sample.tensor.values[0, channels.index("fire_mask")]
    assert np.all(fire_mask == 0.0)
    assert np.all(sample.next_day_fire_mask == 0.0)


def test_transform_ndws_record_relative_humidity_is_zero_when_bone_dry():
    record = _make_ndws_record(sph=0.0)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    rh = sample.tensor.values[0, channels.index("relative_humidity")]
    assert rh[0, 0] == pytest.approx(0.0)


def test_transform_ndws_record_relative_humidity_is_self_consistent_at_saturation():
    # construye sph EXACTAMENTE en el punto de saturación a la misma T
    # que usa la conversión (mean(tmmn,tmmx)) -- debe dar ~100% RH,
    # verificando la fórmula contra sí misma en vez de un número
    # calculado a mano (evita un error de aritmética en el test).
    from models.deep.public_dataset import _saturation_specific_humidity

    temp_k = 290.0
    q_sat = _saturation_specific_humidity(temp_k)
    record = _make_ndws_record(tmmn=temp_k, tmmx=temp_k, sph=q_sat)
    sample = transform_ndws_record(record, sample_id=1)
    channels = list(sample.tensor.coords["channel"].values)
    rh = sample.tensor.values[0, channels.index("relative_humidity")]
    assert rh[0, 0] == pytest.approx(100.0, abs=0.01)


def test_transform_ndws_record_rejects_a_record_missing_a_required_feature():
    record = _make_ndws_record()
    del record["th"]
    with pytest.raises(ValueError, match="th"):
        transform_ndws_record(record, sample_id=1)


def test_load_public_dataset_samples_reads_a_real_fixture_file(tmp_path):
    from models.tests.test_tfrecord_reader import _write_tfrecord

    path = tmp_path / "fixture.tfrecord"
    _write_tfrecord(path, [_make_ndws_record(), _make_ndws_record()])

    samples = list(load_public_dataset_samples([path]))
    assert len(samples) == 2
    assert all(isinstance(s, PublicDatasetSample) for s in samples)


def test_split_public_dataset_is_deterministic_and_covers_every_shard():
    shards = [Path(f"shard_{i}.tfrecord") for i in range(10)]
    first = split_public_dataset(shards, seed=1)
    second = split_public_dataset(shards, seed=1)
    assert first == second
    assert set(first["train"]) | set(first["val"]) == set(shards)
    assert set(first["train"]).isdisjoint(first["val"])
    assert "test" not in first


def test_split_public_dataset_never_leaves_val_empty_with_enough_shards():
    shards = [Path(f"shard_{i}.tfrecord") for i in range(5)]
    result = split_public_dataset(shards, seed=1)
    assert len(result["val"]) >= 1


def test_split_public_dataset_is_disjoint_from_a_different_seed():
    shards = [Path(f"shard_{i}.tfrecord") for i in range(20)]
    a = split_public_dataset(shards, seed=1)
    b = split_public_dataset(shards, seed=2)
    assert a != b  # distinto seed -> partición distinta (con alta probabilidad, N=20)
```

Note: this test imports the test-only encoder helper
`_write_tfrecord` from `test_tfrecord_reader.py` — both are in
`models/tests/`, a plain sibling import (no package boundary issue
since `models/tests/` isn't a namespace package requiring `__init__.py`
for this, matching how pytest already discovers both files).

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_public_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.deep.public_dataset'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/deep/public_dataset.py`:

```python
"""Adaptador de 'Next Day Wildfire Spread' (NDWS, Huot et al.) al
esquema de tensor de PyroCast (`features.dataset.assemble.CHANNEL_ORDER`)
-- ver docs/public-dataset.md para el contexto completo (de dónde se
obtiene, licencia, formato) y la tabla resumen de mapeo. Este docstring
es la fuente de verdad de las fórmulas exactas; docs/public-dataset.md
la resume pero no la duplica en detalle, para que no se desincronicen.

Mapeo canal por canal (PyroCast <- NDWS):
    elevation          <- elevation                          (copia directa, metros)
    slope_deg,         <- derivado de elevation de NDWS vía
    aspect_deg            features.terrain.slope_aspect (Horn 1981),
                          cellsize=1000.0 (resolución nativa de NDWS)
    wind_u, wind_v     <- derivado de th (dirección, grados, convención
                          meteorológica "desde dónde sopla") + vs
                          (velocidad, m/s): u=-vs*sin(th), v=-vs*cos(th)
    temperature        <- mean(tmmn, tmmx)                    (Kelvin)
    relative_humidity  <- derivado de sph (humedad específica, kg/kg) +
                          temperature, vía la fórmula de presión de
                          vapor de la OMM + Magnus-Tetens (Alduchov &
                          Eskridge 1996, mismos coeficientes que
                          features.weather.derive.relative_humidity_approx),
                          asumiendo presión estándar a nivel del mar
                          (101325 Pa) -- NDWS no trae presión de
                          superficie real. Aproximación documentada.
    precipitation      <- pr / 1000.0                         (mm -> m)
    ndvi               <- NDVI / 10000.0, clip [-1, 1]
    fuel_type          <- SIN equivalente en NDWS -- relleno constante
                          99 (ingestion.worldcover.fuel_type.FUEL_TYPE_UNKNOWN,
                          duplicado aquí como literal -- models no
                          depende de ingestion, mismo patrón que
                          models/cellular_automata/rules.py)
    fire_mask          <- PrevFireMask, clip [0, 1] (-1 "incierto" se
                          trata conservadoramente como "sin fuego
                          observado")

Sin canal correspondiente en el tensor de 11 canales:
    FireMask (etiqueta día t+1) -> PublicDatasetSample.next_day_fire_mask,
    mismo clip [0, 1] que fire_mask.

Sin uso en absoluto (NDWS los trae, PyroCast no tiene dónde ponerlos ni
se derivan): pdsi, erc, population.

*** El WorkGrid/día que se construye para llamar a
assemble_event_tensor es NOMINAL, no geografía real -- NDWS son
recortes de EE.UU. continental, no de Chile. Solo resolution_m
(1000.0, la resolución real de NDWS) tiene significado; crs, el origen
de la transform, y la fecha de "day" son placeholders. Ver
docs/public-dataset.md. ***
"""
import datetime as dt
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine
from features.dataset.assemble import CHANNEL_ORDER, EventChannels, assemble_event_tensor
from features.grid.grid import WorkGrid
from features.terrain.slope_aspect import compute_slope_aspect

from models.deep.tfrecord_reader import read_tf_examples

_NDWS_RESOLUTION_M = 1000.0
_NDWS_STANDARD_PRESSURE_PA = 101325.0
_FUEL_TYPE_UNKNOWN = 99.0  # ingestion.worldcover.fuel_type.FUEL_TYPE_UNKNOWN, duplicado
_NOMINAL_DAY = dt.date(1970, 1, 1)  # NDWS no tiene fecha real por recorte -- ver docstring

_REQUIRED_NDWS_FEATURES = (
    "elevation", "pdsi", "NDVI", "pr", "sph", "th", "tmmn", "tmmx",
    "vs", "erc", "population", "PrevFireMask", "FireMask",
)


@dataclass(frozen=True)
class PublicDatasetSample:
    tensor: xr.DataArray
    next_day_fire_mask: np.ndarray


def _saturation_specific_humidity(temp_k: float, pressure_pa: float = _NDWS_STANDARD_PRESSURE_PA) -> float:
    temp_c = temp_k - 273.15
    e_sat = 611.2 * math.exp((17.625 * temp_c) / (243.04 + temp_c))
    return 0.622 * e_sat / (pressure_pa - 0.378 * e_sat)


def _specific_humidity_to_relative_humidity(
    specific_humidity: np.ndarray, temp_k: np.ndarray, pressure_pa: float = _NDWS_STANDARD_PRESSURE_PA
) -> np.ndarray:
    temp_c = temp_k - 273.15
    e_sat = 611.2 * np.exp((17.625 * temp_c) / (243.04 + temp_c))
    e = specific_humidity * pressure_pa / (0.622 + 0.378 * specific_humidity)
    rh = 100.0 * e / e_sat
    return np.clip(rh, 0.0, 100.0)


def transform_ndws_record(record: dict[str, np.ndarray], sample_id: int) -> PublicDatasetSample:
    missing = [name for name in _REQUIRED_NDWS_FEATURES if name not in record]
    if missing:
        raise ValueError(
            f"registro NDWS le faltan feature(s) requerida(s): {missing} -- "
            f"esperadas: {_REQUIRED_NDWS_FEATURES}."
        )

    elevation = record["elevation"].astype("float64")
    height, width = elevation.shape

    slope_deg, aspect_deg = compute_slope_aspect(elevation, _NDWS_RESOLUTION_M, _NDWS_RESOLUTION_M)

    th_rad = np.radians(record["th"].astype("float64"))
    vs = record["vs"].astype("float64")
    wind_u = -vs * np.sin(th_rad)
    wind_v = -vs * np.cos(th_rad)

    temperature = (record["tmmn"].astype("float64") + record["tmmx"].astype("float64")) / 2.0
    relative_humidity = _specific_humidity_to_relative_humidity(
        record["sph"].astype("float64"), temperature
    )
    precipitation = record["pr"].astype("float64") / 1000.0
    ndvi = np.clip(record["NDVI"].astype("float64") / 10000.0, -1.0, 1.0)
    fuel_type = np.full((height, width), _FUEL_TYPE_UNKNOWN, dtype="float64")
    fire_mask = np.clip(record["PrevFireMask"].astype("float64"), 0.0, 1.0)
    next_day_fire_mask = np.clip(record["FireMask"].astype("float64"), 0.0, 1.0)

    channels = EventChannels(
        days=(_NOMINAL_DAY,),
        static={
            "elevation": elevation,
            "slope_deg": slope_deg,
            "aspect_deg": aspect_deg,
            "fuel_type": fuel_type,
        },
        dynamic={
            "wind_u": {_NOMINAL_DAY: wind_u},
            "wind_v": {_NOMINAL_DAY: wind_v},
            "temperature": {_NOMINAL_DAY: temperature},
            "relative_humidity": {_NOMINAL_DAY: relative_humidity},
            "precipitation": {_NOMINAL_DAY: precipitation},
            "ndvi": {_NOMINAL_DAY: ndvi},
            "fire_mask": {_NOMINAL_DAY: fire_mask},
        },
    )
    grid = WorkGrid(
        crs="EPSG:32719",  # nominal -- ver docstring del módulo
        transform=Affine(_NDWS_RESOLUTION_M, 0.0, 0.0, 0.0, -_NDWS_RESOLUTION_M, 0.0),
        width=width,
        height=height,
        resolution_m=_NDWS_RESOLUTION_M,
    )
    tensor = assemble_event_tensor(channels, grid, event_id=sample_id)
    assert list(tensor.coords["channel"].values) == list(CHANNEL_ORDER)  # noqa: S101

    return PublicDatasetSample(tensor=tensor, next_day_fire_mask=next_day_fire_mask)


def load_public_dataset_samples(paths: list[Path]) -> Iterator[PublicDatasetSample]:
    sample_id = 0
    for path in paths:
        for record in read_tf_examples(path):
            yield transform_ndws_record(record, sample_id=sample_id)
            sample_id += 1


def split_public_dataset(
    shard_paths: list[Path], seed: int = 42, train_frac: float = 0.85
) -> dict[str, list[Path]]:
    # split al nivel de ARCHIVO (shard), no de registro individual --
    # leer todo el dataset por adelantado solo para asignar un split
    # significaría leerlo dos veces; un shard es la unidad atómica
    # natural acá, igual que un evento lo es en
    # features/dataset/split.py.
    sorted_shards = sorted(shard_paths, key=str)
    rng = random.Random(seed)
    shuffled = sorted_shards[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n < 2:
        return {"train": shuffled, "val": []}

    n_val = max(1, round(n * (1 - train_frac)))
    n_val = min(n_val, n - 1)  # nunca deja train vacío
    return {"train": shuffled[: n - n_val], "val": shuffled[n - n_val :]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_public_dataset.py -v`
Expected: `16 passed`

- [ ] **Step 5: Update `models/src/models/deep/__init__.py`**

```python
"""Módulo de modelos: deep. `public_dataset.py` (preentrenamiento
sobre Next Day Wildfire Spread, ver docs/public-dataset.md) está
implementado; el U-Net propio (P9-P11) todavía no."""
```

- [ ] **Step 6: Re-read `docs/public-dataset.md` against the real code**

Confirm every claim in the "Cómo difiere del esquema de PyroCast" table
and the feature table matches `transform_ndws_record`'s actual
docstring exactly (channel names, formulas, the fuel_type=99 fill, the
FireMask-has-no-channel note). Fix any drift found.

- [ ] **Step 7: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep && uv run ruff check models/`
Expected: both clean

- [ ] **Step 8: Commit**

```bash
git add models/src/models/deep/public_dataset.py models/src/models/deep/__init__.py models/tests/test_public_dataset.py docs/public-dataset.md
git commit -m "feat: add NDWS-to-PyroCast channel adapter and public dataset split"
```

---

## Task 5: Documentation — `docs/decisions.md`, `docs/limitations.md`

**Files:**
- Modify: `docs/decisions.md`
- Modify: `docs/limitations.md`

**Interfaces:**
- Consumes: the final shipped behavior from Tasks 1–4 — no code
  interface.

- [ ] **Step 1: Append to `docs/decisions.md`**

```markdown
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
```

- [ ] **Step 2: Append to `docs/limitations.md`**

```markdown
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
  real descargado de Kaggle** (solo contra un encoder propio, por
  round-trip) -- este entorno no tiene acceso de red para descargarlo.
  Es correcto según la especificación pública del formato TFRecord y
  autoconsistente, pero una incompatibilidad real con el archivo exacto
  que Kaggle distribuye hoy no puede descartarse con 100% de certeza
  hasta probarlo contra un archivo real. Ver `docs/public-dataset.md`.
- **El nombre exacto de los archivos dentro del zip de Kaggle no se
  pudo confirmar** al escribir `docs/public-dataset.md` (la página
  requiere una sesión de navegador autenticada). `models/deep/public_dataset.py`
  no asume ningún nombre -- recibe una lista/glob explícito de rutas
  del caller -- pero si Kaggle distribuye un formato distinto de
  TFRecord (algunas re-subidas de terceros ofrecen `.npy`), este
  loader no lo soporta.
```

- [ ] **Step 3: Run the full verification sweep**

Run:
```bash
env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests -v
uv run ruff check .
uv run mypy --strict models/src/models/deep
uv run mypy --strict shared/src features/src
```
Expected: all green/clean (models test count: 60 existing + 6
(tfrecord_reader) + 16 (public_dataset) = 82 passed + 1 skipped).

- [ ] **Step 4: Commit**

```bash
git add docs/decisions.md docs/limitations.md
git commit -m "docs: document NDWS pretraining adapter design decisions and limitations"
```
