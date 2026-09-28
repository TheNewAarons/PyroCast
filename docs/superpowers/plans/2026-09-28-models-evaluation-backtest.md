# models/evaluation/metrics.py + backtest.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Round out `models/evaluation/metrics.py` with ECE and Dice
(IoU/Brier already exist from the `models/cellular_automata` plan), add
`backtest.py` (per-event + bootstrap-aggregated metrics against any model
implementing a common `Protocol`), define that Protocol in `shared/` so
P7 (cellular automaton) and the future P9-P11 (U-Net) share it, wire the
cellular automaton as its first implementer, and ship a
`pyrocast-models backtest` CLI that writes a deterministic,
git-versionable `bench/results/baseline.json` and persists to PostGIS.

**Architecture:** `shared/model_protocol.py` defines `FireSpreadModel`
(a `Protocol` with one method, `predict(event: xr.DataArray) ->
np.ndarray`, `event` being exactly the `(day, channel, y, x)` tensor
`features.dataset.assemble.assemble_event_tensor` already produces —
`shared` gains an `xarray` dependency to type this concretely, but never
imports `features` itself, preserving the existing dependency direction).
`models/cellular_automata/model.py` adapts the already-implemented
`simulate_fire_spread` to that Protocol (slicing named channels out of
the tensor by position, seeding `initial_burning` from the tensor's own
day-0 `fire_mask`). `models/evaluation/backtest.py` runs any
`FireSpreadModel` over a list of test events, computes all four metrics
per event, and bootstrap-aggregates each metric (percentile method,
deterministic under a fixed seed). `models/cli.py` gains a `backtest`
command that loads the P6 test split + its Zarr events by the same
filename convention `features/dataset/` already established, runs the
calibrated cellular automaton, writes `bench/results/baseline.json`
(deterministic JSON: fixed seeds throughout, sorted keys), and persists
per-event metrics to `PostGIS` (`model_run` + `evaluation_result`,
already-existing tables from the bootstrap schema).

Every metric formula and the Zarr tensor's exact runtime shape
(`dims`, `channel` coordinate contents, `attrs["resolution_m"]`/
`attrs["event_id"]` types) were verified by actually running the code
before this plan was written — see Global Constraints.

**Tech Stack:** `numpy` only for `metrics.py` (no new dependency).
`xarray` added to `shared/pyproject.toml` (new — for the Protocol's type
hint only, justified in Global Constraints). `xarray>=2024.7` and
`zarr>=2.18` added to `models/pyproject.toml` (new — `models/` needs to
open the Zarr events `features/dataset/` produces, same versions already
pinned in `features/pyproject.toml`).

**Spec:** the user's request (quoted below), governed by `/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md`.

```
Implementa models/evaluation/metrics.py y models/evaluation/backtest.py,
y evalúa el autómata celular de P7.

1. metrics.py: implementa Brier score, ECE (expected calibration error,
   con binning configurable), IoU y coeficiente de Dice entre máscara
   predicha y máscara real, todas vectorizadas con numpy y con tests
   contra valores calculados a mano en casos pequeños conocidos.
2. backtest.py: dado un modelo (cualquier objeto con una interfaz común
   `predict(event) -> probability_masks`) y un conjunto de eventos de
   test (del split de P6), corre las predicciones y calcula todas las
   métricas por evento y agregadas, con bootstrap para intervalos de
   confianza.
3. Define la interfaz común de modelo en shared/ (un Protocol de
   Python) para que el autómata celular (P7) y el U-Net (P9-P11) la
   implementen igual, y así el backtest y el reporte final (P16) no
   dupliquen lógica por modelo.
4. Corre el backtest del autómata celular calibrado contra el split de
   test y guarda el resultado en la tabla `evaluation_result` (PostGIS)
   y en bench/results/ como JSON (mismo criterio de reproducibilidad
   que en P6: nada de resultados no versionables).

Tests: cada métrica con casos calculados a mano, el backtest con un
modelo dummy de fixture (predicción constante) para verificar que el
pipeline entero corre y guarda resultados correctamente.
Criterios de aceptación: `make backtest` corre el autómata celular
contra los eventos de fixture y deja métricas en
bench/results/baseline.json; mypy --strict y ruff limpios en
models/evaluation/. Cierra la etapa 3: a partir de aquí el proyecto ya
tiene un modelo funcional y medible de punta a punta, aunque sea
simple.
```

## Global Constraints

- `models/evaluation/metrics.py` already has `iou_score`/`brier_score`
  from the `models/cellular_automata` plan (already shipped, already
  reviewed clean) — this plan ADDS `dice_score`/`ece_score` to the same
  file, doesn't recreate it.
- **`FireSpreadModel.predict`'s `event` parameter is `xr.DataArray`**,
  not a bespoke class — verified by actually building, saving, and
  re-opening a real event tensor before writing this plan:
  `dims=('day','channel','y','x')`, `channel` coord values are (numpy)
  strings matching `features.dataset.assemble.CHANNEL_ORDER` exactly,
  `attrs["resolution_m"]` round-trips as a plain Python `float`,
  `attrs["event_id"]` as a plain Python `int`. `predict` returns a plain
  `np.ndarray` of shape `(day, y, x)`, values in `[0, 1]` — same spatial
  dims and day count as the input event, never a different shape.
- **`shared/` gains `xarray` as a dependency, but never imports
  `features`** — the Protocol needs a concrete type for `event`, and
  `xr.DataArray` (a third-party type) is exactly what `features/dataset/`
  already produces, so typing against it costs nothing structurally
  while keeping `shared` at the bottom of the dependency graph (every
  other package depends on `shared`, never the reverse — `models`
  depending on `features` is fine, since `ingestion → features → models`
  is the documented mainline pipeline direction in CLAUDE.md's
  architecture diagram, unlike the `ingestion`/`features` cross-deps
  ruled on earlier this session).
- **Verified exact values** (computed and cross-checked against the
  actual formulas before writing this plan, not just derived on paper):
  - `dice_score([T,T,F,F], [T,F,T,F])` (intersection=1, total=2+2=4) =
    `0.5`.
  - `dice_score` on two all-empty masks = `1.0` (same "both-empty is a
    trivial perfect match" convention as `iou_score`, for consistency).
  - `ece_score([0.1,0.4,0.6,0.9], [0,0,1,1], n_bins=2)`: bin 0 (`p<=0.5`)
    has predictions `{0.1,0.4}` (confidence 0.25) vs. truth `{0,0}`
    (accuracy 0.0); bin 1 (`p>0.5`) has `{0.6,0.9}` (confidence 0.75) vs.
    `{1,1}` (accuracy 1.0). `ECE = 0.5*|0.25-0| + 0.5*|0.75-1| = 0.25`.
  - `np.digitize(predictions, bin_edges[1:-1], right=True)` is the
    correct binning call for `n_bins` equal-width bins over `[0,1]` —
    verified it reproduces the hand-computed bin assignment above
    exactly (`right=True` puts a value equal to an edge in the LOWER
    bin, which is why `p=0.4` and `p=0.6` land in different bins for the
    2-bin case above, not both in the same one).
- **`fuel_type` in the tensor is `float32`, not `int`** — the adapter
  must `.astype(int)` the sliced band before calling
  `flammability_grid` (which expects integer codes) — this was already
  the pattern `resample_to_grid`/`assemble_event_tensor` established,
  and verified precision-safe for fuel codes 1-99 in the `features/dataset`
  review (float32's 24-bit mantissa is exact well past 99).
- **Day 0 of a backtest is evaluated somewhat tautologically**: the
  adapter seeds `initial_burning` from the event's own day-0 `fire_mask`
  (the only sensible seed — there's no "day -1" to seed from), so the
  model's day-0 output for already-burning cells is `1.0` by
  construction, matching the ground truth trivially. This is documented
  as a known modeling property in `docs/limitations.md` (Task 6), not
  hidden — excluding day 0 from evaluation is a reasonable future
  refinement, not implemented here (YAGNI: no test or acceptance
  criterion asks for it).
- **Bootstrap CI**: percentile method, `numpy.random.default_rng(seed)`,
  resampling event-level metric VALUES with replacement (never
  resampling pixels — consistent with this project's established
  "splits/resampling happen at the EVENT level, never below it" rule
  from `features/dataset/split.py`). `n=0` samples → all-NaN CI (no
  crash); `n=1` → point estimate with lower=upper=that same value (no
  crash, no fabricated spread).
- **`bench/results/baseline.json` must be deterministic and
  git-versionable**: `bench/` is NOT in `.gitignore` (verified — only
  `data/raw|interim|processed/` and raster/Zarr file extensions are
  ignored), so this file is meant to be committed. Every seed
  (simulation, bootstrap) defaults to a fixed value; `json.dumps(...,
  sort_keys=True, indent=2)` for stable, diffable formatting.
- **PostGIS persistence reuses the bootstrap schema as-is**: `model_run`
  (`event_id` FK to `fire_event.id`, `model_name`, `config` JSON) and
  `evaluation_result` (`run_id` FK, `metric_name`, `value`, `split`) —
  no schema migration needed. `model_run.event_id` is `fire_event.id`
  (the serial PK), NOT the content-hash `firms_event_id` the tensor's
  `attrs["event_id"]` carries — persisting a backtest result for a real
  event therefore requires looking up `fire_event.id` by
  `firms_event_id` first. Every backtest run legitimately gets its own
  new `model_run` row (unlike `fire_event`'s upsert-by-content-hash from
  the `features/dataset` review fix — a backtest is an experiment
  record, re-running it is a new experiment, not a correction to the
  same one).

## Review Focus

- **`ece_score` with `n_bins` larger than the number of distinct
  predicted values** (e.g. 100 bins over 4 predictions): most bins are
  empty and must be skipped without dividing by zero or contributing a
  spurious term — covered by a dedicated Task 1 test.
- **`dice_score`/`iou_score` called with float arrays instead of
  boolean** (a caller passing raw probabilities without thresholding
  first): both already `.astype(bool)` internally (any nonzero float
  becomes `True`) — a real footgun if a caller forgets to threshold at
  0.5 first. Covered by a dedicated Task 1 test asserting the exact
  (surprising) behavior, so it's documented as a known contract rather
  than silently discovered later.
- **`run_backtest` given a `FireSpreadModel` whose `predict` returns the
  WRONG shape** (different day count or spatial dims than the input
  event): should fail loudly (a numpy broadcast/shape error surfacing
  immediately) rather than silently comparing mismatched arrays — verify
  this is what actually happens and document it, not silently guessed at.
- **`run_backtest` with a single test event**: bootstrap CI on n=1
  aggregated metric values must not crash or fabricate a confidence
  interval — covered by a dedicated Task 4 test (this is also exactly
  the fixture-CLI acceptance path, which may realistically only have 1-2
  fixture events).
- **The cellular-automaton adapter given an event whose `fire_mask` at
  day 0 is entirely `False`** (a real possibility if `features/dataset/`'s
  padding days precede the first detection): `initial_burning` would be
  all-`False`, and `simulate_fire_spread` already handles an all-`False`
  `initial_burning` without crashing (never contradicted by existing
  tests) — covered by a dedicated Task 3 test confirming the adapter
  doesn't add a NEW failure mode on top of that.

---

## Task 1: `models/evaluation/metrics.py` — add `dice_score`, `ece_score`

**Files:**
- Modify: `models/src/models/evaluation/metrics.py`
- Test: `models/tests/test_evaluation_metrics.py` (modify — add new tests, keep existing ones)

**Interfaces:**
- Produces: `dice_score(predicted_mask: np.ndarray, true_mask: np.ndarray)
  -> float`, `ece_score(predicted_prob: np.ndarray, true_binary:
  np.ndarray, n_bins: int = 10) -> float`. Task 4 (`backtest.py`)
  consumes both directly, alongside the existing `iou_score`/`brier_score`.

- [ ] **Step 1: Write the failing tests**

Append to `models/tests/test_evaluation_metrics.py`:

```python
from models.evaluation.metrics import dice_score, ece_score


def test_dice_score_known_partial_overlap():
    # intersección=1, total=2+2=4 -> dice=2*1/4=0.5
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert dice_score(a, b) == pytest.approx(0.5)


def test_dice_score_is_one_for_identical_masks():
    mask = np.array([[True, False], [False, True]])
    assert dice_score(mask, mask) == 1.0


def test_dice_score_both_empty_is_one_not_a_division_by_zero():
    empty = np.zeros((3, 3), dtype=bool)
    assert dice_score(empty, empty) == 1.0


def test_dice_score_treats_any_nonzero_float_as_true():
    # dice_score hace .astype(bool) internamente -- CUALQUIER float
    # distinto de cero (incluida una probabilidad de 0.01 sin
    # umbralizar) cuenta como "True". Documentado como contrato
    # explícito, no un bug sorpresa a descubrir después: quien llame a
    # dice_score/iou_score con probabilidades crudas sin umbralizar
    # primero obtiene un resultado silenciosamente distinto del
    # esperado.
    a = np.array([0.01, 0.0, 0.99])
    b = np.array([1.0, 0.0, 1.0])
    assert dice_score(a, b) == dice_score(
        np.array([True, False, True]), np.array([True, False, True])
    )


def test_ece_score_known_value_two_bins():
    pred = np.array([0.1, 0.4, 0.6, 0.9])
    true = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece_score(pred, true, n_bins=2) == pytest.approx(0.25)


def test_ece_score_is_zero_for_perfect_calibration():
    # cada bin tiene confianza == exactitud exacta
    pred = np.array([0.0, 0.0, 1.0, 1.0])
    true = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece_score(pred, true, n_bins=2) == pytest.approx(0.0)


def test_ece_score_handles_more_bins_than_distinct_values_without_crashing():
    # la mayoría de los 100 bins quedan vacíos -- deben saltarse, no
    # dividir por cero ni sumar un término espurio.
    pred = np.array([0.1, 0.4, 0.6, 0.9])
    true = np.array([0.0, 0.0, 1.0, 1.0])
    result = ece_score(pred, true, n_bins=100)
    assert 0.0 <= result <= 1.0


def test_ece_score_empty_input_is_zero_not_a_crash():
    assert ece_score(np.array([]), np.array([])) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package models pytest models/tests/test_evaluation_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'dice_score' from 'models.evaluation.metrics'`

- [ ] **Step 3: Write minimal implementation**

Append to `models/src/models/evaluation/metrics.py`:

```python
def dice_score(predicted_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Coeficiente de Dice (2·|A∩B| / (|A|+|B|)) entre dos máscaras
    binarias -- ambas vacías -> 1.0 (misma convención que iou_score, no
    una división por cero). Hace `.astype(bool)` internamente: CUALQUIER
    valor no-cero cuenta como True, incluida una probabilidad cruda sin
    umbralizar -- ver test_dice_score_treats_any_nonzero_float_as_true."""
    predicted = predicted_mask.astype(bool)
    true = true_mask.astype(bool)
    total = predicted.sum() + true.sum()
    if total == 0:
        return 1.0
    intersection = np.logical_and(predicted, true).sum()
    return float(2 * intersection) / float(total)


def ece_score(predicted_prob: np.ndarray, true_binary: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: agrupa las predicciones en `n_bins`
    bins de ancho igual sobre [0,1], y para cada bin no vacío suma
    |confianza_promedio - exactitud_promedio| ponderado por la fracción
    de muestras en ese bin. 0.0 = perfectamente calibrado. Array vacío
    -> 0.0 (sin evidencia de descalibración, no un error)."""
    predicted = predicted_prob.astype("float64").ravel()
    true = true_binary.astype("float64").ravel()
    n = predicted.size
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=True: un valor IGUAL a un borde interno cae en el bin de
    # ABAJO (verificado con el caso conocido de 2 bins -- ver Global
    # Constraints del plan). clip por seguridad ante errores de punto
    # flotante en el borde superior.
    bin_indices = np.clip(np.digitize(predicted, bin_edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(predicted[in_bin].mean())
        accuracy = float(true[in_bin].mean())
        ece += (count / n) * abs(confidence - accuracy)
    return float(ece)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package models pytest models/tests/test_evaluation_metrics.py -v`
Expected: `15 passed` (7 existing + 8 new)

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/evaluation && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/evaluation/metrics.py models/tests/test_evaluation_metrics.py
git commit -m "feat: add Dice and ECE metrics (models/evaluation/metrics.py)"
```

---

## Task 2: `shared/model_protocol.py` — the common model interface

**Files:**
- Create: `shared/src/shared/model_protocol.py`
- Modify: `shared/pyproject.toml` (add `xarray>=2024.7`)
- Test: `shared/tests/test_model_protocol.py`

**Interfaces:**
- Produces: `FireSpreadModel` (a `@runtime_checkable` `Protocol` with one
  method: `predict(self, event: xr.DataArray) -> np.ndarray`). Task 3
  (the cellular-automaton adapter) implements it; Task 4 (`backtest.py`)
  consumes it as the type of the `model` parameter.

- [ ] **Step 1: Write the failing test**

Create `shared/tests/test_model_protocol.py`:

```python
"""Tests de la interfaz común de modelo: verifica que un modelo dummy
que implementa `predict(event) -> np.ndarray` satisface el Protocol
estructuralmente (sin heredar de nada)."""
import numpy as np
import xarray as xr
from shared.model_protocol import FireSpreadModel


class _DummyConstantModel:
    """No hereda de FireSpreadModel -- el Protocol es estructural."""

    def predict(self, event: xr.DataArray) -> np.ndarray:
        n_days, _, height, width = event.shape
        return np.full((n_days, height, width), 0.5, dtype="float64")


def test_dummy_model_satisfies_the_protocol_structurally():
    model = _DummyConstantModel()
    assert isinstance(model, FireSpreadModel)


def test_protocol_typed_function_accepts_the_dummy_model():
    def run(model: FireSpreadModel) -> np.ndarray:
        fake_event = xr.DataArray(
            np.zeros((2, 3, 4, 5), dtype="float32"),
            dims=("day", "channel", "y", "x"),
        )
        return model.predict(fake_event)

    result = run(_DummyConstantModel())
    assert result.shape == (2, 4, 5)
    assert np.all(result == 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --package shared pytest shared/tests/test_model_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shared.model_protocol'`

- [ ] **Step 3: Add `xarray` to `shared/pyproject.toml`**

In `shared/pyproject.toml`, add `"xarray>=2024.7",` to `dependencies`.
Run `uv sync --all-packages` — **never** a scoped `uv sync`/`uv sync
--package X` in this workspace (it silently prunes the shared `.venv` —
see the `features/grid`+`fire_state` plan's ledger).

- [ ] **Step 4: Write minimal implementation**

Create `shared/src/shared/model_protocol.py`:

```python
"""Interfaz común de modelo de propagación de incendios -- implementada
por igual por `models/cellular_automata` (P7) y el futuro U-Net
(P9-P11), para que `models/evaluation/backtest.py` y el reporte final
(P16) no dupliquen lógica por modelo.

Vive en `shared/` (no en `models/`) para que cualquier paquete pueda
tipar contra ella sin depender de `models/` -- p. ej. un futuro
`serving/` que sirva predicciones no necesitaría importar todo
`models/cellular_automata` solo por el tipo.
"""
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr


@runtime_checkable
class FireSpreadModel(Protocol):
    def predict(self, event: xr.DataArray) -> np.ndarray:
        """`event`: tensor `(day, channel, y, x)` tal como lo produce
        `features.dataset.assemble.assemble_event_tensor` (mismo orden
        de canales, `CHANNEL_ORDER`, mismos atributos `crs`/`transform`/
        `resolution_m`/`event_id`). Devuelve un array `(day, y, x)` de
        probabilidad de fuego por celda y por día, en `[0, 1]` -- mismas
        dimensiones espaciales y mismo número de días que `event`."""
        ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run --package shared pytest shared/tests/test_model_protocol.py -v`
Expected: `2 passed`

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict shared/src && uv run ruff check shared/`
Expected: both clean

- [ ] **Step 7: Commit**

```bash
git add shared/src/shared/model_protocol.py shared/pyproject.toml shared/tests/test_model_protocol.py uv.lock
git commit -m "feat: add common FireSpreadModel Protocol (shared/model_protocol.py)"
```

---

## Task 3: `models/cellular_automata/model.py` — adapt `simulate_fire_spread` to the Protocol

**Files:**
- Create: `models/src/models/cellular_automata/model.py`
- Modify: `models/pyproject.toml` (add `xarray>=2024.7`, `zarr>=2.18`)
- Test: `models/tests/test_cellular_automata_model.py`

**Interfaces:**
- Consumes: `models.cellular_automata.rules.SpreadParameters`,
  `models.cellular_automata.simulate.simulate_fire_spread`,
  `shared.model_protocol.FireSpreadModel`.
- Produces: `CellularAutomatonModel` (a class implementing
  `FireSpreadModel`: `__init__(self, params: SpreadParameters =
  SpreadParameters(), seed: int = 42)`, `predict(self, event:
  xr.DataArray) -> np.ndarray`). Task 4 (`backtest.py`) and Task 5 (CLI)
  consume this directly.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_cellular_automata_model.py`:

```python
"""Tests del adaptador que expone simulate_fire_spread como
FireSpreadModel -- construye un evento sintético (mismo formato que
features.dataset.assemble.assemble_event_tensor produce, verificado
contra el objeto real antes de escribir este archivo) y verifica que la
predicción tiene la forma correcta y usa las capas correctas."""
import numpy as np
import xarray as xr
from models.cellular_automata.model import CellularAutomatonModel
from models.cellular_automata.rules import SpreadParameters
from shared.model_protocol import FireSpreadModel

CHANNEL_ORDER = (
    "elevation", "slope_deg", "aspect_deg",
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
    "ndvi", "fuel_type", "fire_mask",
)


def _make_event(
    size: int = 21, n_days: int = 5, ignite_center: bool = True
) -> xr.DataArray:
    center = size // 2
    data = np.zeros((n_days, len(CHANNEL_ORDER), size, size), dtype="float32")
    fuel_idx = CHANNEL_ORDER.index("fuel_type")
    data[:, fuel_idx, :, :] = 1.0  # pastizal, combustible, homogéneo
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    if ignite_center:
        data[0, fire_idx, center, center] = 1.0
    return xr.DataArray(
        data,
        dims=("day", "channel", "y", "x"),
        coords={
            "day": [f"2026-01-{d + 1:02d}" for d in range(n_days)],
            "channel": list(CHANNEL_ORDER),
        },
        name="fire_event_tensor",
        attrs={"crs": "EPSG:32719", "transform": (100.0, 0.0, 0.0, 0.0, -100.0, 0.0),
               "resolution_m": 100.0, "event_id": 42},
    )


def test_cellular_automaton_model_satisfies_the_protocol():
    assert isinstance(CellularAutomatonModel(), FireSpreadModel)


def test_predict_returns_expected_shape():
    event = _make_event(size=21, n_days=5)
    model = CellularAutomatonModel(seed=1)
    result = model.predict(event)
    assert result.shape == (5, 21, 21)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_predict_seeds_initial_burning_from_day_zero_fire_mask():
    event = _make_event(size=21, n_days=3)
    model = CellularAutomatonModel(seed=1)
    result = model.predict(event)
    center = 21 // 2
    assert result[0, center, center] == 1.0  # ya en llamas en el día 0


def test_predict_handles_an_event_with_no_fire_on_day_zero():
    # capas de padding antes de la primera detección real -- day-0
    # fire_mask enteramente False, no debe fallar.
    event = _make_event(size=11, n_days=3, ignite_center=False)
    model = CellularAutomatonModel(seed=1)
    result = model.predict(event)
    assert result.shape == (3, 11, 11)
    assert np.all(result == 0.0)  # sin ignición y sin vecinos en llamas -> nunca se enciende


def test_predict_is_deterministic_with_a_fixed_seed():
    event = _make_event(size=15, n_days=4)
    first = CellularAutomatonModel(seed=7).predict(event)
    second = CellularAutomatonModel(seed=7).predict(event)
    assert np.array_equal(first, second)


def test_predict_uses_the_configured_spread_parameters():
    event = _make_event(size=21, n_days=6)
    aggressive = CellularAutomatonModel(
        params=SpreadParameters(base_spread_prob=0.99), seed=1
    ).predict(event)
    timid = CellularAutomatonModel(
        params=SpreadParameters(base_spread_prob=0.01), seed=1
    ).predict(event)
    assert np.sum(aggressive[-1] >= 0.5) > np.sum(timid[-1] >= 0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package models pytest models/tests/test_cellular_automata_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.cellular_automata.model'`

- [ ] **Step 3: Add `xarray`/`zarr` to `models/pyproject.toml`**

In `models/pyproject.toml`, add `"xarray>=2024.7",` and `"zarr>=2.18",`
to `dependencies`. Run `uv sync --all-packages`.

- [ ] **Step 4: Write minimal implementation**

Create `models/src/models/cellular_automata/model.py`:

```python
"""Adapta `simulate_fire_spread` a `shared.model_protocol.FireSpreadModel`
-- el autómata celular (P7) es el primer implementador de esa interfaz
común (el futuro U-Net, P9-P11, será el segundo)."""
import numpy as np
import xarray as xr

from models.cellular_automata.rules import SpreadParameters
from models.cellular_automata.simulate import simulate_fire_spread

_STATIC_CHANNELS = ("elevation", "fuel_type")
_WIND_CHANNELS = ("wind_u", "wind_v")


class CellularAutomatonModel:
    def __init__(
        self, params: SpreadParameters = SpreadParameters(), seed: int = 42
    ) -> None:
        self.params = params
        self.seed = seed

    def predict(self, event: xr.DataArray) -> np.ndarray:
        channels = list(event.coords["channel"].values)

        def band(name: str) -> np.ndarray:
            result: np.ndarray = event.values[:, channels.index(name), :, :]
            return result

        # capas estáticas: mismo valor todos los días -- se toma el día 0.
        elevation = band("elevation")[0]
        # fuel_type se guarda como float32 en el tensor (mismo criterio
        # que ingestion/worldcover, ver docs/decisions.md) -- las tablas
        # de flammability usan códigos int.
        fuel_type = band("fuel_type")[0].astype(int)
        wind_u = band("wind_u")
        wind_v = band("wind_v")

        fire_mask = band("fire_mask")
        # día 0 es el único "ancla" real disponible -- no existe un
        # "día -1" del que sembrar. Un evento sin fuego en el día 0 (p.
        # ej. un día de padding antes de la primera detección) produce
        # initial_burning todo-False, que simulate_fire_spread ya maneja
        # sin fallar.
        initial_burning = fire_mask[0] > 0.5

        resolution_m = float(event.attrs["resolution_m"])
        n_days = int(event.sizes["day"])

        result: np.ndarray = simulate_fire_spread(
            initial_burning, elevation, wind_u, wind_v, fuel_type,
            resolution_m, n_days, params=self.params, seed=self.seed,
        )
        return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --package models pytest models/tests/test_cellular_automata_model.py -v`
Expected: `6 passed`

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/cellular_automata && uv run ruff check models/`
Expected: both clean

- [ ] **Step 7: Commit**

```bash
git add models/src/models/cellular_automata/model.py models/pyproject.toml models/tests/test_cellular_automata_model.py uv.lock
git commit -m "feat: adapt cellular automaton to FireSpreadModel Protocol"
```

---

## Task 4: `models/evaluation/backtest.py` — per-event + bootstrapped metrics

**Files:**
- Create: `models/src/models/evaluation/backtest.py`
- Test: `models/tests/test_evaluation_backtest.py`

**Interfaces:**
- Consumes: `shared.model_protocol.FireSpreadModel`,
  `models.evaluation.metrics.{iou_score, dice_score, brier_score, ece_score}`.
- Produces: `EventMetrics` (frozen dataclass: `event_id: int`, `iou:
  float`, `dice: float`, `brier: float`, `ece: float`), `BootstrapCI`
  (frozen dataclass: `point_estimate: float`, `lower: float`, `upper:
  float`), `BacktestResult` (frozen dataclass: `per_event:
  list[EventMetrics]`, `aggregate: dict[str, BootstrapCI]`),
  `run_backtest(model: FireSpreadModel, events: list[xr.DataArray],
  n_bootstrap: int = 1000, seed: int = 42, ece_bins: int = 10,
  confidence: float = 0.95) -> BacktestResult`. Task 5 (CLI) consumes
  `run_backtest`, `BacktestResult`, `EventMetrics`, `BootstrapCI` directly
  (for JSON serialization and PostGIS persistence).

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_evaluation_backtest.py`:

```python
"""Tests del backtest: modelo dummy de fixture (predicción constante),
verifica que el pipeline entero corre y calcula métricas por evento y
agregadas con bootstrap."""
import numpy as np
import xarray as xr
from models.evaluation.backtest import BacktestResult, run_backtest

CHANNEL_ORDER = (
    "elevation", "slope_deg", "aspect_deg",
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
    "ndvi", "fuel_type", "fire_mask",
)


class _ConstantModel:
    """Predicción constante de 0.5 en todo el evento -- fixture simple,
    no necesita conocer nada del contenido real del tensor."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def predict(self, event: xr.DataArray) -> np.ndarray:
        n_days, _, height, width = event.shape
        return np.full((n_days, height, width), self.value, dtype="float64")


def _make_event(event_id: int, n_days: int = 3, size: int = 4, fire_frac: float = 0.5) -> xr.DataArray:
    data = np.zeros((n_days, len(CHANNEL_ORDER), size, size), dtype="float32")
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    n_fire_cells = int(size * size * fire_frac)
    flat = data[:, fire_idx, :, :].reshape(n_days, -1)
    flat[:, :n_fire_cells] = 1.0
    data[:, fire_idx, :, :] = flat.reshape(n_days, size, size)
    return xr.DataArray(
        data,
        dims=("day", "channel", "y", "x"),
        coords={"day": [f"2026-01-{d + 1:02d}" for d in range(n_days)], "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor",
        attrs={"event_id": event_id, "resolution_m": 100.0},
    )


def test_run_backtest_computes_all_four_metrics_per_event():
    events = [_make_event(event_id=1), _make_event(event_id=2)]
    result = run_backtest(_ConstantModel(0.5), events, n_bootstrap=100, seed=1)
    assert isinstance(result, BacktestResult)
    assert len(result.per_event) == 2
    for metrics in result.per_event:
        assert metrics.event_id in (1, 2)
        for value in (metrics.iou, metrics.dice, metrics.brier, metrics.ece):
            assert 0.0 <= value <= 1.0


def test_run_backtest_aggregate_has_a_bootstrap_ci_per_metric():
    events = [_make_event(event_id=i) for i in range(1, 6)]
    result = run_backtest(_ConstantModel(0.5), events, n_bootstrap=200, seed=1)
    assert set(result.aggregate) == {"iou", "dice", "brier", "ece"}
    for ci in result.aggregate.values():
        assert ci.lower <= ci.point_estimate <= ci.upper


def test_run_backtest_is_deterministic_with_a_fixed_seed():
    events = [_make_event(event_id=i) for i in range(1, 4)]
    first = run_backtest(_ConstantModel(0.5), events, n_bootstrap=100, seed=3)
    second = run_backtest(_ConstantModel(0.5), events, n_bootstrap=100, seed=3)
    assert first == second


def test_run_backtest_single_event_bootstrap_ci_does_not_crash():
    events = [_make_event(event_id=1)]
    result = run_backtest(_ConstantModel(0.5), events, n_bootstrap=50, seed=1)
    for ci in result.aggregate.values():
        assert ci.lower == ci.point_estimate == ci.upper


def test_run_backtest_perfect_model_scores_perfectly():
    class _PerfectModel:
        def predict(self, event: xr.DataArray) -> np.ndarray:
            channels = list(event.coords["channel"].values)
            fire_idx = channels.index("fire_mask")
            result: np.ndarray = event.values[:, fire_idx, :, :].astype("float64")
            return result

    events = [_make_event(event_id=1, fire_frac=0.3)]
    result = run_backtest(_PerfectModel(), events, n_bootstrap=10, seed=1)
    metrics = result.per_event[0]
    assert metrics.iou == 1.0
    assert metrics.dice == 1.0
    assert metrics.brier == 0.0
    assert metrics.ece == pytest.approx(0.0)
```

Note the last test needs `import pytest` added to the top of the file
alongside the other imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package models pytest models/tests/test_evaluation_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.evaluation.backtest'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/evaluation/backtest.py`:

```python
"""Corre un `FireSpreadModel` (shared.model_protocol) contra un conjunto
de eventos de test, calcula las 4 métricas de `metrics.py` por evento, y
agrega cada una con un intervalo de confianza bootstrap (método
percentil, remuestreo de los VALORES por evento, nunca de píxeles --
mismo criterio de "nunca por debajo del nivel de evento" que
`features/dataset/split.py`)."""
from dataclasses import dataclass

import numpy as np
import xarray as xr

from models.evaluation.metrics import brier_score, dice_score, ece_score, iou_score
from shared.model_protocol import FireSpreadModel

_FIRE_MASK_THRESHOLD = 0.5


@dataclass(frozen=True)
class EventMetrics:
    event_id: int
    iou: float
    dice: float
    brier: float
    ece: float


@dataclass(frozen=True)
class BootstrapCI:
    point_estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class BacktestResult:
    per_event: list[EventMetrics]
    aggregate: dict[str, BootstrapCI]


def _bootstrap_ci(
    values: list[float], n_bootstrap: int, seed: int, confidence: float
) -> BootstrapCI:
    array = np.array(values, dtype="float64")
    if array.size == 0:
        return BootstrapCI(point_estimate=float("nan"), lower=float("nan"), upper=float("nan"))
    point = float(array.mean())
    if array.size == 1:
        return BootstrapCI(point_estimate=point, lower=point, upper=point)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_bootstrap, array.size))
    resampled_means = array[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lower = float(np.quantile(resampled_means, alpha))
    upper = float(np.quantile(resampled_means, 1 - alpha))
    return BootstrapCI(point_estimate=point, lower=lower, upper=upper)


def run_backtest(
    model: FireSpreadModel,
    events: list[xr.DataArray],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ece_bins: int = 10,
    confidence: float = 0.95,
) -> BacktestResult:
    per_event: list[EventMetrics] = []
    for event in events:
        predicted_prob = model.predict(event)
        channels = list(event.coords["channel"].values)
        fire_idx = channels.index("fire_mask")
        true_prob = event.values[:, fire_idx, :, :].astype("float64")
        true_mask = true_prob >= _FIRE_MASK_THRESHOLD
        predicted_mask = predicted_prob >= _FIRE_MASK_THRESHOLD

        per_event.append(
            EventMetrics(
                event_id=int(event.attrs["event_id"]),
                iou=iou_score(predicted_mask, true_mask),
                dice=dice_score(predicted_mask, true_mask),
                brier=brier_score(predicted_prob, true_prob),
                ece=ece_score(predicted_prob, true_prob, n_bins=ece_bins),
            )
        )

    aggregate = {
        name: _bootstrap_ci(
            [getattr(m, name) for m in per_event], n_bootstrap, seed, confidence
        )
        for name in ("iou", "dice", "brier", "ece")
    }
    return BacktestResult(per_event=per_event, aggregate=aggregate)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package models pytest models/tests/test_evaluation_backtest.py -v`
Expected: `5 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/evaluation && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/evaluation/backtest.py models/tests/test_evaluation_backtest.py
git commit -m "feat: add backtest runner with per-event and bootstrapped metrics"
```

---

## Task 5: `models/evaluation/db.py` + `pyrocast-models backtest` CLI + Makefile

**Files:**
- Create: `models/src/models/evaluation/db.py`
- Modify: `models/src/models/cli.py`
- Modify: `Makefile`
- Test: `models/tests/test_evaluation_db.py`
- Test: `models/tests/test_cli_backtest.py`

**Interfaces:**
- Consumes: `shared.db.schema.{model_run, evaluation_result, fire_event}`,
  `models.evaluation.backtest.{run_backtest, BacktestResult, EventMetrics}`,
  `models.cellular_automata.model.CellularAutomatonModel`.
- Produces: `persist_backtest_run(engine, firms_event_id: int,
  model_name: str, config: dict, split: str, metrics: EventMetrics) ->
  int` (returns the new `model_run.id`); the `pyrocast-models backtest`
  command; `bench/results/baseline.json`.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_evaluation_db.py`:

```python
"""Tests de persistencia de resultados de backtest en `model_run` +
`evaluation_result` (PostGIS). El insert real solo se ejercita contra un
Postgres real (guardado con skipif, mismo patrón que el resto de este
proyecto) -- lo demás se verifica sin red/DB."""
import os

import pytest
from models.evaluation.backtest import EventMetrics
from models.evaluation.db import persist_backtest_run
from shared.config import Settings
from shared.db.schema import evaluation_result, fire_event, metadata, model_run
from sqlalchemy import create_engine, insert, select, text


def _live_settings() -> Settings | None:
    required = [
        "FIRMS_MAP_KEY", "CDS_API_URL", "CDS_API_KEY",
        "COPERNICUS_DATASPACE_CLIENT_ID", "COPERNICUS_DATASPACE_CLIENT_SECRET",
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
        "POSTGRES_USER", "POSTGRES_PASSWORD",
    ]
    if not all(os.getenv(k) for k in required):
        return None
    return Settings()


@pytest.mark.skipif(_live_settings() is None, reason="requiere POSTGRES_* de un contenedor real")
def test_persist_backtest_run_roundtrip_against_real_postgis():
    settings = _live_settings()
    assert settings is not None
    engine = create_engine(settings.postgres_dsn)
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        metadata.create_all(conn)

        fire_event_id = conn.execute(
            insert(fire_event).values(
                bbox="-73.0,-38.0,-72.0,-37.0", start_date="2026-01-10",
                end_date="2026-01-15", source="test", firms_event_id=999,
                geom="SRID=4326;POLYGON((-73 -38, -73 -37, -72 -37, -72 -38, -73 -38))",
            ).returning(fire_event.c.id)
        ).scalar_one()

        metrics = EventMetrics(event_id=999, iou=0.5, dice=0.6, brier=0.1, ece=0.05)
        run_id = persist_backtest_run(
            engine=conn, firms_event_id=999, model_name="cellular_automata",
            config={"base_spread_prob": 0.3}, split="test", metrics=metrics,
        )

        stored_model_name = conn.execute(
            select(model_run.c.model_name).where(model_run.c.id == run_id)
        ).scalar_one()
        assert stored_model_name == "cellular_automata"
        stored_event_id = conn.execute(
            select(model_run.c.event_id).where(model_run.c.id == run_id)
        ).scalar_one()
        assert stored_event_id == fire_event_id

        stored_metrics = conn.execute(
            select(evaluation_result.c.metric_name, evaluation_result.c.value)
            .where(evaluation_result.c.run_id == run_id)
        ).all()
        stored = {row.metric_name: row.value for row in stored_metrics}
        assert stored == {"iou": 0.5, "dice": 0.6, "brier": 0.1, "ece": 0.05}
    finally:
        trans.rollback()
        conn.close()
```

Create `models/tests/test_cli_backtest.py`:

```python
"""Test de humo del CLI `pyrocast-models backtest`: sin red, sin
Postgres real, sin Zarr real -- eventos de fixture inyectados, pero el
backtest y la escritura de bench/results corren de verdad."""
import datetime as dt
import json

import numpy as np
import xarray as xr
from models.cli import app
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}

_CHANNEL_ORDER = (
    "elevation", "slope_deg", "aspect_deg",
    "wind_u", "wind_v", "temperature", "relative_humidity", "precipitation",
    "ndvi", "fuel_type", "fire_mask",
)


def _fixture_event(event_id: int) -> xr.DataArray:
    size, n_days = 11, 3
    data = np.zeros((n_days, len(_CHANNEL_ORDER), size, size), dtype="float32")
    data[:, _CHANNEL_ORDER.index("fuel_type"), :, :] = 1.0
    data[0, _CHANNEL_ORDER.index("fire_mask"), size // 2, size // 2] = 1.0
    return xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={
            "day": [f"2026-01-{d + 1:02d}" for d in range(n_days)],
            "channel": list(_CHANNEL_ORDER),
        },
        name="fire_event_tensor",
        attrs={"event_id": event_id, "resolution_m": 100.0},
    )


def test_backtest_cli_writes_bench_results_json(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)

    from shared.config import get_settings

    get_settings.cache_clear()

    fixture_events = [_fixture_event(1), _fixture_event(2)]
    monkeypatch.setattr(
        "models.cli.load_test_events", lambda dataset_dir: fixture_events
    )
    monkeypatch.setattr("models.cli.persist_backtest_run", lambda **kwargs: 1)

    result = runner.invoke(app, ["backtest", "--n-bootstrap", "50"])
    assert result.exit_code == 0, result.output

    results_path = tmp_path / "bench" / "results" / "baseline.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text())
    assert payload["model_name"] == "cellular_automata"
    assert len(payload["per_event"]) == 2
    assert set(payload["aggregate"]) == {"iou", "dice", "brier", "ece"}
    get_settings.cache_clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package models pytest models/tests/test_evaluation_db.py models/tests/test_cli_backtest.py -v`
Expected: FAIL — `test_evaluation_db.py` collection error
(`ModuleNotFoundError: No module named 'models.evaluation.db'`);
`test_cli_backtest.py` fails because `models.cli.load_test_events` etc.
don't exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/evaluation/db.py`:

```python
"""Persiste un resultado de backtest en `model_run` + `evaluation_result`
(PostGIS). `model_run.event_id` es el `fire_event.id` serial (la PK real
de la tabla), NO el `firms_event_id` (hash de contenido) que trae el
tensor -- hay que resolverlo primero. Cada corrida de backtest es un
registro de experimento nuevo (a diferencia del upsert de
`features/dataset/db.py::persist_fire_event_metadata`): re-correr un
backtest no "corrige" el anterior, agrega otro."""
from typing import Any

from sqlalchemy import Connection, Engine, insert, select

from models.evaluation.backtest import EventMetrics
from shared.db.schema import evaluation_result, fire_event, model_run


def _resolve_fire_event_id(conn: Any, firms_event_id: int) -> int:
    row_id = conn.execute(
        select(fire_event.c.id).where(fire_event.c.firms_event_id == firms_event_id)
    ).scalar_one()
    result: int = row_id
    return result


def persist_backtest_run(
    engine: Engine | Connection,
    firms_event_id: int,
    model_name: str,
    config: dict[str, float],
    split: str,
    metrics: EventMetrics,
) -> int:
    if isinstance(engine, Engine):
        with engine.begin() as conn:
            return _persist(conn, firms_event_id, model_name, config, split, metrics)
    return _persist(engine, firms_event_id, model_name, config, split, metrics)


def _persist(
    conn: Any,
    firms_event_id: int,
    model_name: str,
    config: dict[str, float],
    split: str,
    metrics: EventMetrics,
) -> int:
    resolved_event_id = _resolve_fire_event_id(conn, firms_event_id)
    run_id = conn.execute(
        insert(model_run)
        .values(event_id=resolved_event_id, model_name=model_name, config=config)
        .returning(model_run.c.id)
    ).scalar_one()
    for metric_name in ("iou", "dice", "brier", "ece"):
        conn.execute(
            insert(evaluation_result).values(
                run_id=run_id,
                metric_name=metric_name,
                value=getattr(metrics, metric_name),
                split=split,
            )
        )
    result: int = run_id
    return result
```

Add `backtest` to `models/src/models/cli.py`:

```python
"""Punto de entrada del CLI de modelos: `pyrocast-models`."""
import json
from pathlib import Path

import numpy as np
import typer
import xarray as xr
from sqlalchemy import create_engine

from models.cellular_automata.model import CellularAutomatonModel
from models.cellular_automata.simulate import simulate_fire_spread
from models.evaluation.backtest import BacktestResult, run_backtest
from models.evaluation.db import persist_backtest_run
from shared.config import get_settings

app = typer.Typer()


@app.callback()
def _callback() -> None:
    """CLI de modelos de PyroCast."""


def run_ca(
    n_days: int = typer.Option(10, help="Días a simular"),
    size: int = typer.Option(41, help="Tamaño (alto=ancho) de la grilla de fixture"),
    seed: int = typer.Option(42, help="Semilla del componente probabilístico"),
) -> None:
    """Simula un evento de FIXTURE (terreno plano, sin viento, combustible
    homogéneo, ignición en el centro) con el autómata celular, de
    principio a fin -- no requiere un evento real de features/dataset/."""
    center = size // 2
    initial_burning = np.zeros((size, size), dtype=bool)
    initial_burning[center, center] = True
    elevation = np.zeros((size, size))
    wind_u = np.zeros((size, size))
    wind_v = np.zeros((size, size))
    fuel_type = np.ones((size, size), dtype=int)  # 1 = pastizal, homogéneo

    probabilities = simulate_fire_spread(
        initial_burning, elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=n_days, seed=seed,
    )
    for day in range(n_days):
        burning_cells = int(np.sum(probabilities[day] >= 0.5))
        typer.echo(f"Día {day + 1}: {burning_cells} celda(s) con probabilidad >= 0.5")


def load_test_events(dataset_dir: Path) -> list[xr.DataArray]:
    """Lee `splits.json` + los eventos Zarr del split "test", por la
    misma convención de rutas que `features/dataset/` ya establece."""
    splits = json.loads((dataset_dir / "splits.json").read_text())
    events = []
    for event_id in splits["test"]:
        zarr_path = dataset_dir / f"event_{event_id:04d}.zarr"
        opened = xr.open_zarr(zarr_path)["fire_event_tensor"]
        events.append(opened)
    return events


def _result_to_json(result: BacktestResult, model_name: str, config: dict[str, float]) -> dict:
    return {
        "model_name": model_name,
        "config": config,
        "split": "test",
        "n_events": len(result.per_event),
        "per_event": [
            {
                "event_id": m.event_id, "iou": m.iou, "dice": m.dice,
                "brier": m.brier, "ece": m.ece,
            }
            for m in result.per_event
        ],
        "aggregate": {
            name: {"point_estimate": ci.point_estimate, "lower": ci.lower, "upper": ci.upper}
            for name, ci in result.aggregate.items()
        },
    }


def backtest(
    n_bootstrap: int = typer.Option(1000, help="Número de remuestreos bootstrap"),
    seed: int = typer.Option(42, help="Semilla del bootstrap y de la simulación"),
) -> None:
    """Corre el backtest del autómata celular calibrado contra el split
    de test de `features/dataset/` y guarda el resultado en
    bench/results/baseline.json y en PostGIS (model_run +
    evaluation_result). Requiere que `pyrocast-features build-dataset`
    ya haya corrido -- ver docs/dataset-card.md."""
    settings = get_settings()
    dataset_dir = settings.data_processed_dir / "dataset"
    events = load_test_events(dataset_dir)
    if not events:
        typer.echo("El split de test no tiene eventos.")
        raise typer.Exit(code=0)

    params = {"base_spread_prob": 0.3, "slope_coefficient": 4.0, "wind_coefficient": 0.2}
    model = CellularAutomatonModel(seed=seed)
    result = run_backtest(model, events, n_bootstrap=n_bootstrap, seed=seed)

    output_dir = Path("bench") / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _result_to_json(result, "cellular_automata", params)
    (output_dir / "baseline.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2)
    )
    typer.echo(f"Backtest: {len(result.per_event)} evento(s) -> {output_dir / 'baseline.json'}")

    engine = create_engine(settings.postgres_dsn)
    for metrics in result.per_event:
        persist_backtest_run(
            engine=engine, firms_event_id=metrics.event_id,
            model_name="cellular_automata", config=params, split="test", metrics=metrics,
        )
    typer.echo(f"Métricas persistidas en PostGIS ({len(result.per_event)} evaluation_result).")


app.command("run-ca")(run_ca)
app.command("backtest")(backtest)
```

Update `Makefile`'s `backtest` target:

```makefile
backtest:
	uv run --package models pyrocast-models backtest
```

(replace `@echo "pendiente: models/evaluation (backtesting) aún no implementado"`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package models pytest models/tests/test_evaluation_db.py models/tests/test_cli_backtest.py -v`
Expected: `test_evaluation_db.py` → `1 skipped` (no live Postgres in this
environment — expected); `test_cli_backtest.py` → `1 passed`.

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/evaluation models/src/models/cli.py && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Manually verify the installed console script and the fixture acceptance path**

The `backtest` command needs real `data/processed/dataset/splits.json` +
Zarr events to run for real — none exist in this environment (no real
ingestion has been run). Verify the wiring is real and correct anyway:

```bash
uv sync --all-packages
uv run --package models pyrocast-models backtest --help
```

Expected: shows `--n-bootstrap`/`--seed` options. Running `make backtest`
for real against real data is a future step once `pyrocast-ingest`
and `pyrocast-features build-dataset` have populated real events — this
is the same documented operational precondition every prior CLI in this
project has (`pyrocast-features build-dataset` itself needs
`pyrocast-ingest` first). The acceptance criterion's fixture path is
validated by Step 4's `test_cli_backtest.py`, matching the established
pattern for every CLI smoke test this session.

- [ ] **Step 7: Commit**

```bash
git add models/src/models/evaluation/db.py models/src/models/cli.py Makefile models/tests/test_evaluation_db.py models/tests/test_cli_backtest.py
git commit -m "feat: add pyrocast-models backtest command with PostGIS + bench/results persistence"
```

---

## Task 6: Documentation — `docs/decisions.md`, `docs/limitations.md`

**Files:**
- Modify: `docs/decisions.md`
- Modify: `docs/limitations.md`

**Interfaces:**
- Consumes: the final shipped behavior from Tasks 1–5 — no code
  interface, this task only writes prose that must match the shipped
  code exactly.

- [ ] **Step 1: Append to `docs/decisions.md`**

```markdown
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
```

- [ ] **Step 2: Append to `docs/limitations.md`**

```markdown
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
```

- [ ] **Step 3: Run the full verification sweep**

Run:
```bash
env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests -v
env -i PATH="$PATH" HOME="$HOME" uv run --package shared pytest shared/tests -v
uv run ruff check .
uv run mypy --strict models/src/models/evaluation
uv run mypy --strict shared/src features/src
```
Expected: all green/clean (models test count: 33 existing + 8 (metrics)
+ 6 (model.py) + 5 (backtest.py) + 1 (cli backtest, db test skipped) =
53 passed + 1 skipped; shared: 15 existing + 2 (protocol) = 17 passed +
1 skipped).

- [ ] **Step 4: Commit**

```bash
git add docs/decisions.md docs/limitations.md
git commit -m "docs: document models/evaluation backtest design decisions, close Stage 3"
```
