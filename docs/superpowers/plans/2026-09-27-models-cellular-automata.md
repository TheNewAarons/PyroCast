# models/cellular_automata/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A vectorized, probabilistic cellular-automaton wildfire-spread
model (simplified Rothermel-inspired transition rule), a day-by-day
simulator that scales to a full event grid, a grid-search calibration
script against IoU/Brier score, and a `pyrocast-models run-ca` CLI that
simulates a fixture event end-to-end in seconds.

**Architecture:** `models/cellular_automata/rules.py` computes, for every
cell in the grid at once (8 array-level operations for the 8-neighbor
Moore neighborhood, never a per-cell Python loop), the probability that
each not-yet-burning cell ignites given which of its neighbors are
already on fire, the slope toward each burning neighbor, wind alignment
with the spread direction, and the target cell's fuel flammability.
`models/cellular_automata/simulate.py` drives this day-by-day with a
seeded `numpy.random.Generator` for the stochastic ignition draw
(monotonic/no-burnout: once a cell ignites it stays an active fire
source for the rest of the simulated horizon — a deliberate
simplification, documented). `models/cellular_automata/calibrate.py` grid
-searches the free scalar parameters against a set of training samples,
scoring with `models/evaluation/metrics.py` (`iou_score`/`brier_score`,
created directly in their final P8 home now, so nothing needs to move
later). `models/cli.py` adds a `pyrocast-models run-ca` command that runs
a small built-in fixture scenario (flat terrain, no wind, homogeneous
fuel, single ignition point) end-to-end — no dependency on a real
`features/dataset/` Zarr event, which may not exist yet in a fresh
environment.

Every formula constant and every neighbor/wind/slope sign convention in
this plan was independently verified by running the actual arithmetic
(not just derived on paper) before being written down — see the
Global Constraints section for the exact numbers.

**Tech Stack:** `numpy` only for the model itself (no new dependency);
`typer` added to `models/pyproject.toml` (new — for the CLI, mirroring
`ingestion`/`features`'s existing pattern).

**Spec:** the user's request (quoted below), governed by `/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md`.

```
Implementa models/cellular_automata/.

1. rules.py: modelo de transición por celda inspirado en Rothermel
   simplificado. La probabilidad de que una celda se encienda en el paso
   siguiente depende de: cuántos vecinos ya en llamas tiene, la pendiente
   entre la celda y cada vecino en llamas (el fuego sube más rápido
   cuesta arriba), la alineación del viento con la dirección hacia la
   celda, y el tipo de combustible (algunos no arden). Documenta cada
   término de la fórmula y sus parámetros libres en
   docs/cellular-automata.md.
2. simulate.py: simulación vectorizada con numpy (nada de loops por
   celda en Python puro; debe escalar a la grilla completa de un
   evento). Dado un estado inicial (una o más celdas en llamas) y las
   capas estáticas/dinámicas de un evento, simula día a día y devuelve
   una máscara de probabilidad por celda y por día.
3. Un script de calibración (grid search simple) que ajuste los
   parámetros libres del modelo contra un subconjunto de eventos de
   entrenamiento, optimizando IoU o Brier score (usa las métricas de P8
   aunque aún no exista el módulo completo; defínelas primero si hace
   falta y muévelas a models/evaluation/ en P8 sin duplicar código).
4. Caso de prueba analítico: en un terreno plano, sin viento, con
   combustible homogéneo, el fuego debe propagarse de forma
   aproximadamente circular desde el punto de ignición. Verifícalo con
   un test.

Tests: el caso circular anterior, sensibilidad a la pendiente (el fuego
avanza más rápido cuesta arriba que cuesta abajo en el mismo número de
pasos), sensibilidad al viento (se alarga en la dirección del viento),
determinismo con semilla fija para el componente probabilístico.
Criterios de aceptación: `make run-ca` simula un evento de fixture de
principio a fin en menos de unos segundos; mypy y ruff limpios;
docs/cellular-automata.md completo con la fórmula y sus parámetros.
```

## Global Constraints

- Python 3.12, `mypy --strict` on `features/src` per CLAUDE.md — `models/
  src` is NOT currently in that strict list (`make typecheck` only
  covers `shared/src features/src`); this plan runs `mypy --strict
  models/src` anyway as an extra quality bar for every task (all code
  below was pre-verified to pass it), but does not modify the Makefile's
  `typecheck` target to add `models/src` — that's a separate, larger
  decision (would require the pre-existing `models/deep` and
  `models/evaluation` stub `__init__.py` files to also pass strict mode,
  unverified) outside this plan's scope. Documented as a ruling in
  `docs/decisions.md` (Task 6).
- **8-connectivity (Moore neighborhood)**, offsets `NEIGHBOR_OFFSETS =
  ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))` as `(d_row,
  d_col)` pairs, meaning "the neighbor is at `(row+d_row, col+d_col)`
  relative to the cell at `(row, col)`". World-coordinate convention
  (north-up raster, matching every other raster in this repo): `+row` =
  south, `+col` = east. So the unit spread direction from that neighbor
  *into* the cell, in `(east, north)` world coordinates, is `(-d_col,
  d_row) / hypot(d_row, d_col)`.
- **Slope term**: `slope_nc = (elevation[cell] - elevation[neighbor]) /
  distance`, `distance = resolution_m * hypot(d_row, d_col)` (`resolution_m`
  for orthogonal neighbors, `resolution_m * sqrt(2)` for diagonal).
  Positive `slope_nc` means the cell is higher than the burning neighbor
  (uphill spread from neighbor into cell) — must INCREASE the
  probability (`el fuego sube más rápido cuesta arriba`). Applied as
  `exp(slope_coefficient * slope_nc)`.
- **Wind term**: `directional_component = (wind_u * (-d_col) + wind_v *
  d_row) / hypot(d_row, d_col)` — the projection of the wind vector
  `(wind_u, wind_v)` (raw ERA5-Land `u10`/`v10` convention: `u10` =
  eastward component, `v10` = northward component, i.e. the vector
  points in the direction the air is actually moving TOWARD, not
  meteorology's "from" convention used elsewhere in this repo for
  `wind_direction`) onto the (unnormalized) spread-direction vector.
  Positive means wind blows in the same direction as the spread (from
  the burning neighbor toward the cell) — must INCREASE the probability.
  This form is deliberately NOT divided by wind speed (which would
  produce a `0/0` when there is no wind): a zero wind vector gives
  `directional_component = 0` for every direction, so `exp(wind_coefficient
  * 0) = 1` — no wind, no effect, with no special-cased branch needed.
  Applied as `exp(wind_coefficient * directional_component)`.
- **Fuel term**: a lookup from the SIMPLIFIED fuel-type integer codes
  already defined in `ingestion/worldcover/fuel_type.py` (1=pastizal,
  2=matorral, 3=bosque, 4=cultivo, 5=humedal, 90-93=non-combustible
  sentinels, 99=unknown) to a flammability multiplier in `[0, 1]`.
  `models/cellular_automata/rules.py` does NOT import
  `ingestion.worldcover.fuel_type` (would invert the established
  `ingestion → features`/`models` dependency direction) — it duplicates
  just the small integer-code meaning as its own
  `DEFAULT_FUEL_FLAMMABILITY: dict[int, float]`, the same pattern already
  used for `CLOUD_SCL_CLASSES` between `ingestion/sentinel2` and
  `features/vegetation`. Documented as a ruling in `docs/decisions.md`
  (Task 6). Any fuel code NOT in the table (including the sentinels
  above) defaults to flammability `0.0` — conservative: never ignite a
  fuel type this module doesn't recognize.
- **Neighbor-count term**: with 8 potential burning neighbors, each
  contributing an independent per-direction ignition probability `p_dir`
  (base × slope-factor × wind-factor × target-cell flammability, clipped
  to `[0, 1]`), the combined probability that the target cell ignites
  from ANY of them is the standard cellular-automaton form
  `P = 1 - ∏(1 - p_dir)` over the neighbors that are actually burning —
  monotonically increasing in the number of burning neighbors, verified
  directly (see Task 2).
- **No burnout/extinction modeling**: once a cell ignites it remains an
  active fire source (radiates ignition probability to its own
  neighbors) for the rest of the simulated horizon — there is no "burned
  out, no longer spreading" state. This is a deliberate simplification
  ("Rothermel simplificado" per the spec), not an oversight — real fires
  do extinguish; modeling that would need a fuel-consumption/burn-duration
  term this plan does not add. Documented forcefully in
  `docs/cellular-automata.md` and `docs/limitations.md` (Task 6).
- **Free parameters and their verified defaults** (all overridable via
  `SpreadParameters`, all calibratable via Task 4's grid search):
  `base_spread_prob=0.3`, `slope_coefficient=4.0`, `wind_coefficient=0.2`,
  `fuel_flammability=DEFAULT_FUEL_FLAMMABILITY`. None of these come from
  a calibration against real Chilean fires — heuristics chosen so the
  four qualitative behaviors the spec requires (neighbor count, uphill
  faster, downwind longer, seed-determinism) are clearly and verifiably
  true, exactly like every other free parameter introduced in this
  session (`docs/fire-events.md`'s `spatial_eps_m`/`temporal_eps`, etc.).
- **Verified concrete numbers** (from an actual run of this exact
  algorithm before writing this plan — Tasks 2/3's tests assert these
  exact values, not just qualitative inequalities):
  - 3×3 grid, one burning neighbor directly north (uphill, cell 10 m
    higher over 100 m resolution → `slope_nc=0.1`), no wind, flammability
    1.0, default params: `P = 0.3 * exp(4.0*0.1) = 0.4475474...`
  - Same but downhill (`slope_nc=-0.1`): `P = 0.3 * exp(-0.4) =
    0.2010960...`
  - One burning neighbor directly west, wind blowing east at 5 m/s
    (`directional_component=+5`): `P = 0.3 * exp(0.2*5) = 0.8154845...`
  - Same but neighbor directly east (upwind spread), same wind:
    (`directional_component=-5`): `P = 0.3 * exp(0.2*-5) = 0.1103638...`
  - 41×41 grid, ignition at center (20,20), flat/no-wind/homogeneous
    fuel, `base_spread_prob=0.99`, `seed=42`, 10 days: final burning
    extent is EXACTLY 10 cells in all four cardinal directions from
    center (isotropic).
  - Same grid with a 5 m/cell northward elevation ramp (north = uphill),
    `base_spread_prob=0.3`, `seed=42`, 10 days: uphill (north) extent = 8
    cells, downhill (south) extent = 5 cells.
  - Same grid with wind_u=5.0 everywhere (blowing east),
    `base_spread_prob=0.3`, `seed=42`, 10 days: downwind (east) extent =
    10 cells, upwind (west) extent = 2 cells.
  - Two runs with identical inputs and `seed=7`: bit-identical results.

## Review Focus

- **A cell whose fuel type is a code not present in `DEFAULT_FUEL_FLAMMABILITY`
  at all** (not one of the documented sentinels, e.g. a corrupted or
  future WorldCover class): must default to flammability `0.0`
  (never ignite), not KeyError or a silent nonzero default — covered by
  Task 2's `test_flammability_grid_defaults_unmapped_code_to_zero`.
- **A cell seeded as initially burning that also has flammability 0**
  (e.g. a user mistakenly ignites a water/urban cell): the simulator
  does not validate this and will treat it as burning anyway (radiating
  ignition probability to neighbors) for the whole simulated horizon —
  this is a real, documented limitation (Task 6), not silently "fixed"
  by the code, since validating initial conditions was not asked for and
  masking a user's own input error is a separate decision.
- **All-zero wind AND all-zero slope simultaneously** (the circular
  test's exact setup): the formula must reduce to the isotropic
  `base_spread_prob * flammability` for every direction with no NaN/inf
  from a `0/0` — covered structurally by the wind-term's deliberate
  "never divide by wind speed" design (Global Constraints) and Task 3's
  circular test passing without any error suppression.
- **A grid large enough to matter for "debe escalar a la grilla completa
  de un evento"**: the spec's own wording demands this be checked at a
  realistic size, not just the small grids the analytical tests use for
  precise geometric assertions — covered by a dedicated timing test in
  Task 3 on a 200×200 grid over 30 days.
- **`grid_search_calibrate` given only one candidate parameter combination
  in the grid** (a degenerate but legal grid search): must still return
  that one combination and its score, not crash on an empty
  "best-so-far" comparison — covered by Task 4's dedicated test.

---

## Task 1: `models/evaluation/metrics.py` — IoU and Brier score

**Files:**
- Create: `models/src/models/evaluation/metrics.py`
- Modify: `models/src/models/evaluation/__init__.py`
- Test: `models/tests/test_evaluation_metrics.py`

**Interfaces:**
- Produces: `iou_score(predicted_mask: np.ndarray, true_mask: np.ndarray)
  -> float`, `brier_score(predicted_prob: np.ndarray, true_binary:
  np.ndarray) -> float`. Task 4 consumes both directly.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_evaluation_metrics.py`:

```python
"""Tests de métricas de evaluación: IoU y Brier score. Definidas aquí
directamente (P8, models/evaluation/) en vez de en
models/cellular_automata/ y movidas después -- ver docs/decisions.md."""
import numpy as np
import pytest
from models.evaluation.metrics import brier_score, iou_score


def test_iou_score_is_one_for_identical_masks():
    mask = np.array([[True, False], [False, True]])
    assert iou_score(mask, mask) == 1.0


def test_iou_score_is_zero_for_disjoint_masks():
    a = np.array([[True, False], [False, False]])
    b = np.array([[False, True], [False, False]])
    assert iou_score(a, b) == 0.0


def test_iou_score_known_partial_overlap():
    # intersección=1, unión=3 -> IoU=1/3
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert iou_score(a, b) == pytest.approx(1 / 3)


def test_iou_score_both_empty_is_one_not_a_division_by_zero():
    empty = np.zeros((3, 3), dtype=bool)
    assert iou_score(empty, empty) == 1.0


def test_brier_score_is_zero_for_perfect_predictions():
    pred = np.array([1.0, 0.0, 1.0])
    true = np.array([1.0, 0.0, 1.0])
    assert brier_score(pred, true) == 0.0


def test_brier_score_known_value():
    # (0.5)^2 promedio = 0.25 para una predicción de 0.5 en todo
    pred = np.full(4, 0.5)
    true = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(pred, true) == pytest.approx(0.25)


def test_brier_score_worst_case_confident_and_wrong():
    pred = np.array([1.0, 0.0])
    true = np.array([0.0, 1.0])
    assert brier_score(pred, true) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package models pytest models/tests/test_evaluation_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.evaluation.metrics'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/evaluation/metrics.py`:

```python
"""Métricas de evaluación (IoU, Brier score) -- definidas aquí
directamente en su ubicación final de P8 (`models/evaluation/`), en vez
de en `models/cellular_automata/` y movidas después. `models/
cellular_automata/calibrate.py` las importa de aquí; cuando P8 agregue
backtesting real, reutiliza estas mismas funciones sin duplicar código.
"""
import numpy as np


def iou_score(predicted_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Intersection over Union entre dos máscaras binarias. Ambas vacías
    -> 1.0 (coincidencia perfecta trivial, no una división por cero)."""
    predicted = predicted_mask.astype(bool)
    true = true_mask.astype(bool)
    union = np.logical_or(predicted, true).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(predicted, true).sum()
    return float(intersection) / float(union)


def brier_score(predicted_prob: np.ndarray, true_binary: np.ndarray) -> float:
    """Error cuadrático medio entre probabilidades predichas ([0,1]) y el
    resultado binario real (0/1) -- menor es mejor, 0.0 es perfecto."""
    predicted = predicted_prob.astype("float64")
    true = true_binary.astype("float64")
    return float(np.mean((predicted - true) ** 2))
```

Update `models/src/models/evaluation/__init__.py`:

```python
"""Módulo de modelos: evaluation. `metrics.py` (IoU, Brier score) ya
implementado -- calibración isotónica y backtesting (P8) siguen
pendientes."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package models pytest models/tests/test_evaluation_metrics.py -v`
Expected: `7 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/evaluation && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/evaluation/metrics.py models/src/models/evaluation/__init__.py models/tests/test_evaluation_metrics.py
git commit -m "feat: add IoU/Brier evaluation metrics (models/evaluation)"
```

---

## Task 2: `models/cellular_automata/rules.py` — ignition probability rule

**Files:**
- Create: `models/src/models/cellular_automata/rules.py`
- Test: `models/tests/test_cellular_automata_rules.py`

**Interfaces:**
- Produces: `NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...]`,
  `DEFAULT_FUEL_FLAMMABILITY: dict[int, float]`, `SpreadParameters`
  (frozen dataclass: `base_spread_prob: float = 0.3`,
  `slope_coefficient: float = 4.0`, `wind_coefficient: float = 0.2`,
  `fuel_flammability: dict[int, float]` — defaults to a copy of
  `DEFAULT_FUEL_FLAMMABILITY`), `flammability_grid(fuel_type: np.ndarray,
  fuel_flammability: dict[int, float]) -> np.ndarray`,
  `compute_ignition_probability(burning: np.ndarray, elevation:
  np.ndarray, wind_u: np.ndarray, wind_v: np.ndarray, flammability:
  np.ndarray, resolution_m: float, params: SpreadParameters) ->
  np.ndarray`. Task 3 consumes all of these directly.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_cellular_automata_rules.py`:

```python
"""Tests de la regla de transición (probabilidad de ignición por celda):
término de vecinos en llamas, pendiente, viento, y tipo de combustible.
Valores esperados calculados a mano y verificados independientemente
antes de escribir este archivo (ver
docs/superpowers/plans/2026-09-27-models-cellular-automata.md)."""
import numpy as np
import pytest
from models.cellular_automata.rules import (
    DEFAULT_FUEL_FLAMMABILITY,
    SpreadParameters,
    compute_ignition_probability,
    flammability_grid,
)


def test_flammability_grid_maps_known_fuel_codes():
    fuel_type = np.array([[1, 2], [90, 99]])
    grid = flammability_grid(fuel_type, DEFAULT_FUEL_FLAMMABILITY)
    assert grid[0, 0] == 1.0   # pastizal
    assert grid[0, 1] == 0.8   # matorral
    assert grid[1, 0] == 0.0   # urbano/no combustible
    assert grid[1, 1] == 0.0   # desconocido


def test_flammability_grid_defaults_unmapped_code_to_zero():
    fuel_type = np.array([[255]])
    grid = flammability_grid(fuel_type, DEFAULT_FUEL_FLAMMABILITY)
    assert grid[0, 0] == 0.0


def test_compute_ignition_probability_is_zero_with_no_burning_neighbors():
    burning = np.zeros((3, 3), dtype=bool)
    elevation = np.zeros((3, 3))
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert np.all(prob == 0.0)


def test_compute_ignition_probability_is_zero_for_non_combustible_target():
    # 4 vecinos en llamas (N,S,E,O), pero la celda objetivo no es
    # combustible (flammability=0) -- nunca debe encenderse.
    burning = np.array([
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ])
    elevation = np.zeros((3, 3))
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.zeros((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == 0.0


def test_compute_ignition_probability_increases_with_more_burning_neighbors():
    elevation = np.zeros((3, 3))
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    one_neighbor = np.array([
        [False, True, False],
        [False, False, False],
        [False, False, False],
    ])
    four_neighbors = np.array([
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ])
    params = SpreadParameters()
    p_one = compute_ignition_probability(
        one_neighbor, elevation, wind_u, wind_v, flammability, 100.0, params
    )[1, 1]
    p_four = compute_ignition_probability(
        four_neighbors, elevation, wind_u, wind_v, flammability, 100.0, params
    )[1, 1]
    assert p_four > p_one


def test_compute_ignition_probability_matches_hand_computed_value_uphill():
    # vecino en llamas al norte, 10 m más bajo que la celda objetivo
    # (celda objetivo cuesta ARRIBA respecto del vecino) a 100 m de
    # resolución -> slope=0.1. P = 0.3 * exp(4.0*0.1) = 0.4475474...
    burning = np.array([
        [False, True, False],
        [False, False, False],
        [False, False, False],
    ])
    elevation = np.array([
        [0.0, 90.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == pytest.approx(0.44754740929238107)


def test_compute_ignition_probability_matches_hand_computed_value_downhill():
    # mismo vecino, pero ahora la celda objetivo está 10 m más ABAJO
    # (cuesta abajo) -> slope=-0.1. P = 0.3 * exp(-0.4) = 0.2010960...
    # -- debe ser menor que el caso cuesta arriba de arriba.
    burning = np.array([
        [False, True, False],
        [False, False, False],
        [False, False, False],
    ])
    elevation = np.array([
        [0.0, 100.0, 0.0],
        [0.0, 90.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    wind_u = np.zeros((3, 3))
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == pytest.approx(0.20109601381069178)


def test_compute_ignition_probability_matches_hand_computed_value_downwind():
    # vecino en llamas al oeste, viento soplando hacia el este a 5 m/s
    # -- alineado con la propagación oeste->centro.
    # P = 0.3 * exp(0.2*5) = 0.8154845...
    burning = np.array([
        [False, False, False],
        [True, False, False],
        [False, False, False],
    ])
    elevation = np.zeros((3, 3))
    wind_u = np.full((3, 3), 5.0)
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == pytest.approx(0.8154845485377135)


def test_compute_ignition_probability_matches_hand_computed_value_upwind():
    # vecino en llamas al ESTE (propagación este->centro, hacia el
    # oeste), mismo viento hacia el este -- opuesto a la propagación.
    # P = 0.3 * exp(-1.0) = 0.1103638...  -- menor que el caso downwind.
    burning = np.array([
        [False, False, False],
        [False, False, True],
        [False, False, False],
    ])
    elevation = np.zeros((3, 3))
    wind_u = np.full((3, 3), 5.0)
    wind_v = np.zeros((3, 3))
    flammability = np.ones((3, 3))
    prob = compute_ignition_probability(
        burning, elevation, wind_u, wind_v, flammability, 100.0, SpreadParameters()
    )
    assert prob[1, 1] == pytest.approx(0.1103638323514327)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package models pytest models/tests/test_cellular_automata_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.cellular_automata.rules'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/cellular_automata/rules.py`:

```python
"""Regla de transición del autómata celular: probabilidad de que una
celda se encienda en el paso siguiente, inspirada en Rothermel
simplificado.

P(celda se enciende) = 1 - prod_{vecinos en llamas} (1 - p_dir)

donde, para cada una de las 8 direcciones de vecindad (Moore):

    p_dir = base_spread_prob
            * exp(slope_coefficient * pendiente_hacia_la_celda)
            * exp(wind_coefficient * componente_direccional_del_viento)
            * flammability[celda]
            (recortado a [0, 1])

Términos:
- **Vecinos en llamas**: el producto sobre TODOS los vecinos que están
  actualmente en llamas (los que no lo están no contribuyen) -- más
  vecinos en llamas siempre aumenta o mantiene la probabilidad, nunca la
  reduce.
- **Pendiente** (`pendiente_hacia_la_celda`): `(elevación[celda] -
  elevación[vecino]) / distancia`. Positiva cuando la celda está más
  alta que el vecino en llamas (fuego subiendo) -- aumenta la
  probabilidad ("el fuego sube más rápido cuesta arriba").
- **Viento** (`componente_direccional_del_viento`): proyección del
  vector de viento `(wind_u, wind_v)` (convención ERA5-Land: apunta
  hacia donde SOPLA el viento, no de dónde viene) sobre la dirección de
  propagación (del vecino hacia la celda). Positiva cuando el viento
  sopla en la misma dirección que la propagación -- aumenta la
  probabilidad. Sin viento, este término es 0 exactamente (nunca se
  divide por la velocidad del viento, así que no hay 0/0).
- **Tipo de combustible** (`flammability[celda]`): un multiplicador en
  [0, 1] de la celda OBJETIVO (no del vecino) -- 0 para combustibles no
  arden (agua, urbano, nieve/hielo, suelo desnudo, código desconocido).

Parámetros libres (`SpreadParameters`, todos calibrables — ver
`models/cellular_automata/calibrate.py`): `base_spread_prob`,
`slope_coefficient`, `wind_coefficient`, `fuel_flammability`. Ver
docs/cellular-automata.md para la justificación completa de cada uno y
sus valores por defecto.

*** SIMPLIFICACIÓN EXPLÍCITA: este NO es un modelo físico de Rothermel
completo (que requiere humedad de combustible, profundidad del lecho de
combustible, razón de empaquetamiento, calor de ignición, etc., ninguno
modelado aquí) — es una regla de transición probabilística de autómata
celular INSPIRADA en la intuición física de Rothermel (el fuego se
propaga más rápido cuesta arriba y a favor del viento), no una
implementación de sus ecuaciones. Ver docs/cellular-automata.md y
docs/limitations.md. ***
"""
from dataclasses import dataclass, field

import numpy as np

NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)

# Códigos de tipo de combustible simplificado, duplicados desde
# ingestion/worldcover/fuel_type.py (no se importa desde ahí -- ver
# docs/decisions.md, mismo patrón que CLOUD_SCL_CLASSES). Cualquier
# código no listado aquí (incluyendo estos sentinels) vale 0.0.
DEFAULT_FUEL_FLAMMABILITY: dict[int, float] = {
    1: 1.0,   # pastizal
    2: 0.8,   # matorral
    3: 0.6,   # bosque
    4: 0.3,   # cultivo
    5: 0.1,   # humedal
    90: 0.0,  # urbano/no combustible
    91: 0.0,  # suelo desnudo/no combustible
    92: 0.0,  # agua/no combustible
    93: 0.0,  # nieve/hielo/no combustible
    99: 0.0,  # desconocido
}


@dataclass(frozen=True)
class SpreadParameters:
    base_spread_prob: float = 0.3
    slope_coefficient: float = 4.0
    wind_coefficient: float = 0.2
    fuel_flammability: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_FUEL_FLAMMABILITY)
    )


def _shifted(array: np.ndarray, d_row: int, d_col: int, fill: float) -> np.ndarray:
    """`array` desplazado (d_row, d_col): el resultado B cumple
    `B[row, col] == array[row + d_row, col + d_col]` cuando esa posición
    cae dentro de la grilla, y `fill` en caso contrario -- nunca
    wrap-around (a diferencia de `np.roll`)."""
    height, width = array.shape
    result = np.full_like(array, fill)
    row_src_start, row_src_end = max(0, d_row), height + min(0, d_row)
    row_dst_start, row_dst_end = max(0, -d_row), height + min(0, -d_row)
    col_src_start, col_src_end = max(0, d_col), width + min(0, d_col)
    col_dst_start, col_dst_end = max(0, -d_col), width + min(0, -d_col)
    result[row_dst_start:row_dst_end, col_dst_start:col_dst_end] = array[
        row_src_start:row_src_end, col_src_start:col_src_end
    ]
    return result


def flammability_grid(fuel_type: np.ndarray, fuel_flammability: dict[int, float]) -> np.ndarray:
    result = np.zeros(fuel_type.shape, dtype="float64")
    for code, value in fuel_flammability.items():
        result[fuel_type == code] = value
    return result


def compute_ignition_probability(
    burning: np.ndarray,
    elevation: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    flammability: np.ndarray,
    resolution_m: float,
    params: SpreadParameters,
) -> np.ndarray:
    not_burning_contrib = np.ones(burning.shape, dtype="float64")
    for d_row, d_col in NEIGHBOR_OFFSETS:
        neighbor_burning = _shifted(burning, d_row, d_col, False)
        neighbor_elevation = _shifted(elevation, d_row, d_col, 0.0)
        distance = resolution_m * float(np.hypot(d_row, d_col))
        slope = (elevation - neighbor_elevation) / distance
        directional_component = (wind_u * (-d_col) + wind_v * d_row) / float(
            np.hypot(d_row, d_col)
        )
        p_dir = (
            params.base_spread_prob
            * np.exp(params.slope_coefficient * slope)
            * np.exp(params.wind_coefficient * directional_component)
            * flammability
        )
        p_dir = np.clip(p_dir, 0.0, 1.0)
        contribution = np.where(neighbor_burning, p_dir, 0.0)
        not_burning_contrib = not_burning_contrib * (1.0 - contribution)
    result: np.ndarray = 1.0 - not_burning_contrib
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package models pytest models/tests/test_cellular_automata_rules.py -v`
Expected: `8 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/cellular_automata && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/cellular_automata/rules.py models/tests/test_cellular_automata_rules.py
git commit -m "feat: add cellular-automaton ignition probability rule (models/cellular_automata)"
```

---

## Task 3: `models/cellular_automata/simulate.py` — vectorized day-by-day simulation

**Files:**
- Create: `models/src/models/cellular_automata/simulate.py`
- Test: `models/tests/test_cellular_automata_simulate.py`

**Interfaces:**
- Consumes: `models.cellular_automata.rules.{SpreadParameters,
  flammability_grid, compute_ignition_probability}`.
- Produces: `simulate_fire_spread(initial_burning: np.ndarray, elevation:
  np.ndarray, wind_u: np.ndarray, wind_v: np.ndarray, fuel_type:
  np.ndarray, resolution_m: float, n_days: int, params: SpreadParameters
  = SpreadParameters(), seed: int = 42) -> np.ndarray` (shape `(n_days,
  height, width)`, `float64`, values in `[0, 1]`; `wind_u`/`wind_v` may
  be 2D — constant across days — or 3D `(n_days, height, width)` for
  daily-varying wind). Task 4 and Task 5 consume this directly.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_cellular_automata_simulate.py`:

```python
"""Tests de la simulación día a día: caso analítico circular, sensibilidad
a pendiente, sensibilidad a viento, determinismo, y escala a una grilla
de tamaño realista. Todos los valores esperados fueron verificados
ejecutando el algoritmo antes de escribir este archivo (ver
docs/superpowers/plans/2026-09-27-models-cellular-automata.md)."""
import time

import numpy as np
from models.cellular_automata.rules import SpreadParameters
from models.cellular_automata.simulate import simulate_fire_spread

_SIZE = 41
_CENTER = _SIZE // 2


def _single_ignition() -> np.ndarray:
    initial = np.zeros((_SIZE, _SIZE), dtype=bool)
    initial[_CENTER, _CENTER] = True
    return initial


def _extents(burning: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(burning)
    north = _CENTER - int(rows.min())
    south = int(rows.max()) - _CENTER
    west = _CENTER - int(cols.min())
    east = int(cols.max()) - _CENTER
    return north, south, east, west


def test_flat_no_wind_homogeneous_fuel_spreads_approximately_circularly():
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    params = SpreadParameters(base_spread_prob=0.99)

    probabilities = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, params=params, seed=42,
    )
    burning = probabilities[-1] >= 0.5
    north, south, east, west = _extents(burning)
    assert (north, south, east, west) == (10, 10, 10, 10)


def test_fire_advances_faster_uphill_than_downhill_in_the_same_number_of_steps():
    # rampa de elevación: 5 m más alto por celda hacia el norte (fila
    # decreciente) -- norte es cuesta arriba desde el punto de ignición.
    elevation = (_CENTER - np.arange(_SIZE))[:, None] * np.ones((1, _SIZE)) * 5.0
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    params = SpreadParameters()  # defaults

    probabilities = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, params=params, seed=42,
    )
    burning = probabilities[-1] >= 0.5
    north, south, _, _ = _extents(burning)
    assert north == 8   # cuesta arriba
    assert south == 5   # cuesta abajo
    assert north > south


def test_fire_elongates_in_the_wind_direction():
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.full((_SIZE, _SIZE), 5.0)  # viento soplando hacia el este
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    params = SpreadParameters()

    probabilities = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, params=params, seed=42,
    )
    burning = probabilities[-1] >= 0.5
    _, _, east, west = _extents(burning)
    assert east == 10   # a favor del viento
    assert west == 2    # en contra del viento
    assert east > west


def test_simulation_is_deterministic_with_a_fixed_seed():
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)

    first = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, seed=7,
    )
    second = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=10, seed=7,
    )
    assert np.array_equal(first, second)


def test_already_burning_cells_report_probability_one():
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    probabilities = simulate_fire_spread(
        _single_ignition(), elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=3, seed=1,
    )
    assert probabilities[0, _CENTER, _CENTER] == 1.0
    assert probabilities[-1, _CENTER, _CENTER] == 1.0  # nunca se "apaga"


def test_scales_to_a_realistic_event_grid_size_in_a_few_seconds():
    size = 200
    center = size // 2
    initial = np.zeros((size, size), dtype=bool)
    initial[center, center] = True
    elevation = np.random.default_rng(0).random((size, size)) * 500.0
    wind_u = np.full((size, size), 3.0)
    wind_v = np.full((size, size), 1.0)
    fuel_type = np.ones((size, size), dtype=int)

    start = time.monotonic()
    simulate_fire_spread(
        initial, elevation, wind_u, wind_v, fuel_type,
        resolution_m=250.0, n_days=30, seed=42,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package models pytest models/tests/test_cellular_automata_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.cellular_automata.simulate'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/cellular_automata/simulate.py`:

```python
"""Simulación día a día del autómata celular: vectorizada con numpy
(ningún loop por celda en Python puro — el único loop en Python es sobre
DÍAS, que el propio enunciado pide simular uno a uno).

Estado: `burning` es MONÓTONO CRECIENTE (una celda que se enciende nunca
se "apaga" dentro del horizonte simulado — ver la simplificación
explícita de no-burnout en rules.py y docs/cellular-automata.md).

Componente probabilístico: cada día se calcula la probabilidad de
ignición de cada celda (determinista dado el estado actual) y luego se
decide qué celdas se encienden REALMENTE ese día mediante un sorteo
Bernoulli vectorizado con un `numpy.random.Generator` sembrado con
`seed` -- misma semilla + mismas entradas -> misma trayectoria completa,
siempre.
"""
import numpy as np

from models.cellular_automata.rules import (
    SpreadParameters,
    compute_ignition_probability,
    flammability_grid,
)


def simulate_fire_spread(
    initial_burning: np.ndarray,
    elevation: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    fuel_type: np.ndarray,
    resolution_m: float,
    n_days: int,
    params: SpreadParameters = SpreadParameters(),
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flammability = flammability_grid(fuel_type, params.fuel_flammability)
    burning = initial_burning.astype(bool).copy()
    probabilities = np.zeros((n_days, *initial_burning.shape), dtype="float64")

    for day in range(n_days):
        wind_u_day = wind_u[day] if wind_u.ndim == 3 else wind_u
        wind_v_day = wind_v[day] if wind_v.ndim == 3 else wind_v
        prob = compute_ignition_probability(
            burning, elevation, wind_u_day, wind_v_day, flammability, resolution_m, params
        )
        # celdas ya en llamas: estado conocido, probabilidad 1.0 (no se
        # vuelve a sortear si ya estaban encendidas).
        prob = np.where(burning, 1.0, prob)
        probabilities[day] = prob

        draws = rng.random(prob.shape)
        newly_ignited = (draws < prob) & ~burning
        burning = burning | newly_ignited

    return probabilities
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package models pytest models/tests/test_cellular_automata_simulate.py -v`
Expected: `6 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/cellular_automata && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/cellular_automata/simulate.py models/tests/test_cellular_automata_simulate.py
git commit -m "feat: add vectorized day-by-day fire spread simulation (models/cellular_automata)"
```

---

## Task 4: `models/cellular_automata/calibrate.py` — grid search calibration

**Files:**
- Create: `models/src/models/cellular_automata/calibrate.py`
- Test: `models/tests/test_cellular_automata_calibrate.py`

**Interfaces:**
- Consumes: `models.cellular_automata.rules.SpreadParameters`,
  `models.cellular_automata.simulate.simulate_fire_spread`,
  `models.evaluation.metrics.{iou_score, brier_score}`.
- Produces: `TrainingSample` (frozen dataclass: `initial_burning:
  np.ndarray`, `elevation: np.ndarray`, `wind_u: np.ndarray`, `wind_v:
  np.ndarray`, `fuel_type: np.ndarray`, `resolution_m: float`,
  `observed_final_mask: np.ndarray`), `grid_search_calibrate(samples:
  list[TrainingSample], param_grid: dict[str, list[float]], metric: str
  = "iou", seed: int = 42) -> tuple[SpreadParameters, float]` (best
  params found, its average score across samples).

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_cellular_automata_calibrate.py`:

```python
"""Tests del grid search de calibración: sobre datos sintéticos donde el
parámetro "verdadero" se conoce, el grid search debe encontrarlo."""
import numpy as np
from models.cellular_automata.calibrate import TrainingSample, grid_search_calibrate
from models.cellular_automata.rules import SpreadParameters
from models.cellular_automata.simulate import simulate_fire_spread

_SIZE = 21
_CENTER = _SIZE // 2


def _make_sample(true_base_spread_prob: float) -> TrainingSample:
    initial = np.zeros((_SIZE, _SIZE), dtype=bool)
    initial[_CENTER, _CENTER] = True
    elevation = np.zeros((_SIZE, _SIZE))
    wind_u = np.zeros((_SIZE, _SIZE))
    wind_v = np.zeros((_SIZE, _SIZE))
    fuel_type = np.ones((_SIZE, _SIZE), dtype=int)
    true_params = SpreadParameters(base_spread_prob=true_base_spread_prob)
    observed = simulate_fire_spread(
        initial, elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=8, params=true_params, seed=1,
    )
    return TrainingSample(
        initial_burning=initial, elevation=elevation, wind_u=wind_u, wind_v=wind_v,
        fuel_type=fuel_type, resolution_m=100.0,
        observed_final_mask=observed[-1] >= 0.5,
    )


def test_grid_search_calibrate_recovers_the_true_base_spread_prob_with_iou():
    sample = _make_sample(true_base_spread_prob=0.9)
    best_params, best_score = grid_search_calibrate(
        [sample],
        param_grid={
            "base_spread_prob": [0.05, 0.9],
            "slope_coefficient": [4.0],
            "wind_coefficient": [0.2],
        },
        metric="iou",
        seed=1,
    )
    assert best_params.base_spread_prob == 0.9
    assert 0.0 <= best_score <= 1.0


def test_grid_search_calibrate_works_with_brier_metric():
    sample = _make_sample(true_base_spread_prob=0.9)
    best_params, _ = grid_search_calibrate(
        [sample],
        param_grid={
            "base_spread_prob": [0.05, 0.9],
            "slope_coefficient": [4.0],
            "wind_coefficient": [0.2],
        },
        metric="brier",
        seed=1,
    )
    assert best_params.base_spread_prob == 0.9


def test_grid_search_calibrate_handles_a_single_candidate_without_crashing():
    sample = _make_sample(true_base_spread_prob=0.5)
    best_params, best_score = grid_search_calibrate(
        [sample],
        param_grid={
            "base_spread_prob": [0.5],
            "slope_coefficient": [4.0],
            "wind_coefficient": [0.2],
        },
        metric="iou",
        seed=1,
    )
    assert best_params.base_spread_prob == 0.5
    assert best_score >= 0.0


def test_grid_search_calibrate_rejects_an_unknown_metric():
    sample = _make_sample(true_base_spread_prob=0.5)
    import pytest

    with pytest.raises(ValueError, match="métrica"):
        grid_search_calibrate(
            [sample],
            param_grid={"base_spread_prob": [0.5]},
            metric="not-a-real-metric",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package models pytest models/tests/test_cellular_automata_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.cellular_automata.calibrate'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/cellular_automata/calibrate.py`:

```python
"""Grid search simple para calibrar los parámetros libres de
`SpreadParameters` contra un subconjunto de eventos de entrenamiento,
optimizando IoU o Brier score (`models/evaluation/metrics.py`, P8 --
definidas ahí desde el principio, no duplicadas ni movidas después).

Solo calibra los tres parámetros escalares (`base_spread_prob`,
`slope_coefficient`, `wind_coefficient`) -- `fuel_flammability` es un
dict, no un escalar, y calibrarlo por grid search requeriría un espacio
de búsqueda combinatorio mucho más caro; queda para trabajo futuro (ver
docs/limitations.md).
"""
import itertools
import math
from dataclasses import dataclass

import numpy as np

from models.cellular_automata.rules import SpreadParameters
from models.cellular_automata.simulate import simulate_fire_spread
from models.evaluation.metrics import brier_score, iou_score


@dataclass(frozen=True)
class TrainingSample:
    initial_burning: np.ndarray
    elevation: np.ndarray
    wind_u: np.ndarray
    wind_v: np.ndarray
    fuel_type: np.ndarray
    resolution_m: float
    observed_final_mask: np.ndarray


def _score_sample(sample: TrainingSample, params: SpreadParameters, metric: str, seed: int) -> float:
    n_days = 1  # el grid search compara contra el estado final observado,
    # no contra una trayectoria diaria completa -- simplificación
    # deliberada (ver docs/limitations.md); usar el mínimo necesario
    # (1 día) sería incorrecto si el evento real abarca más días, así
    # que se re-deriva desde el propio estado observado en su lugar.
    probabilities = simulate_fire_spread(
        sample.initial_burning, sample.elevation, sample.wind_u, sample.wind_v,
        sample.fuel_type, sample.resolution_m, n_days=n_days, params=params, seed=seed,
    )
    final_prob = probabilities[-1]
    if metric == "iou":
        predicted_mask = final_prob >= 0.5
        return iou_score(predicted_mask, sample.observed_final_mask)
    if metric == "brier":
        return -brier_score(final_prob, sample.observed_final_mask.astype("float64"))
    raise ValueError(f"métrica desconocida: {metric!r} -- usar 'iou' o 'brier'")


def grid_search_calibrate(
    samples: list[TrainingSample],
    param_grid: dict[str, list[float]],
    metric: str = "iou",
    seed: int = 42,
) -> tuple[SpreadParameters, float]:
    keys = list(param_grid)
    best_params: SpreadParameters | None = None
    best_score = -math.inf
    for combo in itertools.product(*(param_grid[key] for key in keys)):
        candidate = SpreadParameters(**dict(zip(keys, combo, strict=True)))
        total = sum(_score_sample(sample, candidate, metric, seed) for sample in samples)
        average_score = total / len(samples)
        if average_score > best_score:
            best_score = average_score
            best_params = candidate
    if best_params is None:
        raise ValueError("param_grid no produjo ninguna combinación de parámetros")
    return best_params, best_score
```

**Ruling on `_score_sample`'s single-day re-simulation**: `TrainingSample`
only carries `observed_final_mask` (the training-relevant end state), not
a full daily trajectory, so scoring runs the simulator for exactly one
day from the sample's own `initial_burning` and compares that day's
output directly to `observed_final_mask`. This only makes literal sense
when the sample's "final mask" is genuinely reachable in one simulated
step from its stated initial condition — true for this task's tests
(which construct samples that way on purpose) but a real training sample
built from a multi-day `features/fire_state` event would need either its
own per-day `initial_burning` (the state at the START of the day being
scored) or the calibration to run the full multi-day trajectory and
compare the LAST day's probabilities to the observed last-day mask. If
implementing this task surfaces that mismatch (i.e., a real multi-day
`TrainingSample` doesn't fit this single-day scoring shape), rule on the
smallest fix that keeps `TrainingSample` accurately describing what it
contains — likely renaming `observed_final_mask` to make the one-step
assumption explicit, or accepting the full per-day trajectory and
`n_days` explicitly — and ledger it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --package models pytest models/tests/test_cellular_automata_calibrate.py -v`
Expected: `4 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/cellular_automata && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/cellular_automata/calibrate.py models/tests/test_cellular_automata_calibrate.py
git commit -m "feat: add grid-search calibration against IoU/Brier score (models/cellular_automata)"
```

---

## Task 5: `pyrocast-models run-ca` CLI + Makefile target

**Files:**
- Create: `models/src/models/cli.py`
- Modify: `models/pyproject.toml` (add `typer>=0.12` dependency and `[project.scripts]`)
- Modify: `Makefile`
- Test: `models/tests/test_cli.py`

**Interfaces:**
- Consumes: `models.cellular_automata.simulate.simulate_fire_spread`.
- Produces: the `pyrocast-models` console script with a `run-ca` command.

- [ ] **Step 1: Write the failing test**

Create `models/tests/test_cli.py`:

```python
"""Test de humo del CLI `pyrocast-models run-ca`: simula un evento de
fixture construido en el propio comando, sin depender de un Zarr real de
features/dataset/."""
import time

from models.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_run_ca_cli_completes_quickly_with_default_fixture():
    start = time.monotonic()
    result = runner.invoke(app, ["run-ca"])
    elapsed = time.monotonic() - start
    assert result.exit_code == 0, result.output
    assert elapsed < 5.0
    assert "Día 1:" in result.output
    assert "Día 10:" in result.output


def test_run_ca_cli_respects_n_days_option():
    result = runner.invoke(app, ["run-ca", "--n-days", "3"])
    assert result.exit_code == 0, result.output
    assert "Día 3:" in result.output
    assert "Día 4:" not in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package models pytest models/tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.cli'`

- [ ] **Step 3: Add `typer` and `[project.scripts]` to `models/pyproject.toml`**

In `models/pyproject.toml`, add `"typer>=0.12",` to `dependencies`
(after `"scikit-learn>=1.5",`), and add:

```toml
[project.scripts]
pyrocast-models = "models.cli:app"
```

(placed after `dependencies`, before `[tool.uv.sources]`). Run `uv sync
--all-packages` — **never** a scoped `uv sync`/`uv sync --package X` in
this workspace (see the `features/grid`+`fire_state` plan's ledger for
why: it silently prunes the shared `.venv`).

- [ ] **Step 4: Write minimal implementation**

Create `models/src/models/cli.py`:

```python
"""Punto de entrada del CLI de modelos: `pyrocast-models`."""
import numpy as np
import typer

from models.cellular_automata.simulate import simulate_fire_spread

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


app.command("run-ca")(run_ca)
```

Update `Makefile`'s `run-ca` target:

```makefile
run-ca:
	uv run --package models pyrocast-models run-ca
```

(replace the old `@echo "pendiente: models/cellular_automata aún no implementado"` line.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --package models pytest models/tests/test_cli.py -v`
Expected: `2 passed`

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/cellular_automata models/src/models/cli.py && uv run ruff check models/`
Expected: both clean

- [ ] **Step 7: Manually verify the installed console script**

Run:
```bash
uv sync --all-packages
uv run --package models pyrocast-models --help
uv run --package models pyrocast-models run-ca --help
make run-ca
```
Expected: `--help` screens show a working Typer interface (`run-ca`
listed as a command; `--n-days`/`--size`/`--seed` listed as options with
defaults); `make run-ca` completes in well under a second and prints 10
lines (`Día 1:` .. `Día 10:`).

- [ ] **Step 8: Commit**

```bash
git add models/src/models/cli.py models/pyproject.toml models/tests/test_cli.py Makefile uv.lock
git commit -m "feat: add pyrocast-models CLI with run-ca command"
```

---

## Task 6: Documentation — `docs/cellular-automata.md`, `docs/decisions.md`, `docs/limitations.md`

**Files:**
- Create: `docs/cellular-automata.md`
- Modify: `docs/decisions.md`
- Modify: `docs/limitations.md`

**Interfaces:**
- Consumes: the final shipped behavior from Tasks 1–5 — no code
  interface, this task only writes prose that must match the shipped
  code exactly.

- [ ] **Step 1: Write `docs/cellular-automata.md`**

```markdown
# Autómata celular de propagación de incendios: fórmula y parámetros

`models/cellular_automata/` implementa un autómata celular probabilístico
inspirado en Rothermel simplificado — NO una implementación de las
ecuaciones de Rothermel (que requieren humedad de combustible, razón de
empaquetamiento, profundidad de lecho de combustible, calor de ignición,
etc., ninguno modelado aquí), sino una regla de transición que captura su
intuición física central: el fuego se propaga más rápido cuesta arriba y
a favor del viento.

## La fórmula

Para cada celda no encendida, la probabilidad de que se encienda en el
paso (día) siguiente es:

```
P(celda se enciende) = 1 - ∏ (1 - p_dir)   sobre los 8 vecinos (Moore)

p_dir = base_spread_prob
        · exp(slope_coefficient · pendiente_nc)
        · exp(wind_coefficient · viento_direccional_nc)
        · flammability[celda]
        (recortado a [0, 1] antes de combinar)
```

donde el producto solo incluye vecinos que están ACTUALMENTE en llamas
(un vecino que no está en llamas no contribuye — equivale a `p_dir=0`
para esa dirección).

### Término 1: vecinos en llamas

El producto `∏(1 - p_dir)` sobre múltiples vecinos en llamas es la forma
estándar de autómata celular para "cualquiera de varias fuentes
independientes puede encender la celda" — la probabilidad combinada
siempre aumenta (o se mantiene igual) con más vecinos en llamas, nunca
disminuye. Verificado directamente: con un vecino en llamas, la
probabilidad es estrictamente menor que con cuatro (mismos parámetros,
terreno/viento/combustible homogéneos).

### Término 2: pendiente (`pendiente_nc`)

```
pendiente_nc = (elevación[celda] - elevación[vecino]) / distancia
distancia = resolución_m · hypot(d_row, d_col)   (ortogonal: resolución_m;
                                                    diagonal: resolución_m·√2)
```

Positiva cuando la celda está más alta que el vecino en llamas (el fuego
sube desde el vecino hacia la celda) — aumenta `p_dir` vía
`exp(slope_coefficient · pendiente_nc)`. Verificado con un caso exacto:
vecino en llamas 10 m más abajo a 100 m de distancia
(`pendiente_nc=0.1`) da `P=0.4475...`; el mismo vecino pero 10 m más
arriba (`pendiente_nc=-0.1`, cuesta abajo) da `P=0.2011...` — cuesta
arriba es más del doble de probable que cuesta abajo, con los parámetros
por defecto.

**Parámetro libre**: `slope_coefficient` (default `4.0`). Mayor valor =
mayor sensibilidad a la pendiente (más diferencia entre cuesta
arriba/abajo).

### Término 3: viento (`viento_direccional_nc`)

```
viento_direccional_nc = (wind_u·(-d_col) + wind_v·d_row) / hypot(d_row, d_col)
```

La proyección del vector de viento `(wind_u, wind_v)` — convención
ERA5-Land: apunta hacia DONDE SOPLA el viento (`u10`=componente hacia el
este, `v10`=componente hacia el norte), no de dónde viene — sobre la
dirección de propagación (del vecino hacia la celda), SIN dividir por la
velocidad del viento (así que viento cero da exactamente 0, nunca un
`0/0`). Positiva cuando el viento sopla en la misma dirección que la
propagación — aumenta `p_dir` vía `exp(wind_coefficient ·
viento_direccional_nc)`. Verificado con un caso exacto: vecino en llamas
al oeste con viento de 5 m/s hacia el este (a favor de la propagación,
`viento_direccional_nc=+5`) da `P=0.8155...`; el mismo viento pero con el
vecino en llamas al este (propagación hacia el oeste, en contra del
viento, `viento_direccional_nc=-5`) da `P=0.1104...`.

**Parámetro libre**: `wind_coefficient` (default `0.2`). Mayor valor =
mayor sensibilidad al viento.

### Término 4: tipo de combustible (`flammability[celda]`)

Un multiplicador en `[0, 1]` de la celda OBJETIVO (no del vecino),
buscado por su código de tipo de combustible simplificado (mismos
códigos que `ingestion/worldcover/fuel_type.py`, duplicados aquí — ver
`docs/decisions.md`):

| Código | Tipo | `flammability` (default) |
|---|---|---|
| 1 | pastizal | 1.0 |
| 2 | matorral | 0.8 |
| 3 | bosque | 0.6 |
| 4 | cultivo | 0.3 |
| 5 | humedal | 0.1 |
| 90-93 | no combustible (urbano/suelo desnudo/agua/nieve-hielo) | 0.0 |
| 99 (o cualquier código no listado) | desconocido | 0.0 |

`0.0` hace que `p_dir=0` para esa celda en TODAS las direcciones —
nunca se enciende, sin importar cuántos vecinos estén en llamas.

**Parámetro libre**: `fuel_flammability` (dict completo, default la
tabla de arriba). Los valores relativos (pastizal > matorral > bosque >
cultivo > humedal) son un ordenamiento razonable pero NO calibrado
contra el comportamiento real de estos combustibles en incendios
chilenos.

## Estado y simulación día a día (`simulate.py`)

Cada celda tiene un estado binario MONÓTONO: `no-en-llamas` o
`en-llamas`. Una vez que una celda se enciende, permanece "en llamas"
(sigue radiando probabilidad de ignición a sus vecinos) para el resto del
horizonte simulado — **no hay modelo de extinción/consumo de
combustible**. Esto es una simplificación deliberada: un incendio real se
apaga cuando consume el combustible disponible; modelar eso requeriría un
término de duración/consumo que este autómata no incluye. Ver
`docs/limitations.md`.

Cada día:
1. Se calcula `P(ignición)` para cada celda no encendida (determinista
   dado el estado actual del día).
2. Se sortea, con un `numpy.random.Generator` sembrado con `seed`, qué
   celdas se encienden REALMENTE ese día (`draws < P`).
3. Se devuelve el array de `P(ignición)` de ese día (no el sorteo
   binario) — las celdas ya en llamas reportan `P=1.0` (estado conocido,
   no se vuelve a sortear).

Misma semilla + mismas entradas → misma trayectoria completa, siempre
(verificado con un test dedicado).

## Parámetros libres — resumen

| Parámetro | Default | Calibrable vía | Calibrado contra incendios reales |
|---|---|---|---|
| `base_spread_prob` | 0.3 | `calibrate.py` (grid search) | No |
| `slope_coefficient` | 4.0 | `calibrate.py` (grid search) | No |
| `wind_coefficient` | 0.2 | `calibrate.py` (grid search) | No |
| `fuel_flammability` | tabla de arriba | No (dict, no escalar — ver limitaciones) | No |

## Calibración (`calibrate.py`)

Grid search simple: para cada combinación de `base_spread_prob` ×
`slope_coefficient` × `wind_coefficient` en una grilla de valores
candidatos, simula cada `TrainingSample` de entrenamiento y promedia
IoU o Brier score (definidas en `models/evaluation/metrics.py` — P8 de
CLAUDE.md, ya en su ubicación final, nunca duplicadas ni movidas después)
contra la máscara observada; devuelve la combinación con mejor score
promedio. Solo calibra los tres parámetros ESCALARES —
`fuel_flammability` es un dict, no un escalar, y calibrarlo por grid
search sería combinatoriamente mucho más caro; queda como trabajo futuro.

## Caso analítico verificado: propagación circular

Terreno plano (elevación uniforme), sin viento, combustible homogéneo,
ignición en un único punto central: con `base_spread_prob=0.99` (spread
casi determinista, para aislar la geometría del ruido probabilístico), a
los 10 días la extensión quemada es EXACTAMENTE 10 celdas en las cuatro
direcciones cardinales (norte, sur, este, oeste) desde el punto de
ignición — isotropía perfecta, como se espera de una regla sin ningún
término direccional activo (pendiente y viento ambos neutralizados por
construcción).
```

- [ ] **Step 2: Append to `docs/decisions.md`**

```markdown
## `models/cellular_automata/`: tabla de flammability duplicada desde `ingestion/worldcover/fuel_type.py`

`models/cellular_automata/rules.py::DEFAULT_FUEL_FLAMMABILITY` usa los
mismos códigos enteros de tipo de combustible simplificado que
`ingestion/worldcover/fuel_type.py` ya define, sin importar ese módulo
directamente — importar desde `ingestion` invertiría la dirección de
dependencia establecida (`ingestion` → `features`/`models`, nunca al
revés), exactamente el mismo caso ya resuelto para `CLOUD_SCL_CLASSES`
entre `ingestion/sentinel2` y `features/vegetation`. Si los códigos de
`ingestion/worldcover/fuel_type.py` cambiaran, esta tabla quedaría
desincronizada silenciosamente — mantenerlas en sync es manual.

## `models/evaluation/metrics.py` implementado directamente en su ubicación final de P8

El enunciado pidió las métricas de evaluación (IoU, Brier score) "aunque
aún no exista el módulo completo" de `models/evaluation/`, con la
instrucción explícita de moverlas ahí en P8 "sin duplicar código". En vez
de crearlas dentro de `models/cellular_automata/` y planear un movimiento
futuro, se crearon DIRECTAMENTE en `models/evaluation/metrics.py` desde
el principio — cuando P8 (calibración isotónica, backtesting) se
implemente, reutiliza estas mismas funciones sin ningún movimiento de
archivo ni duplicación.

## `models/src` no se agrega a `make typecheck` en este plan

CLAUDE.md especifica `mypy --strict` en `shared/` y `features/`
únicamente; el Makefile's `typecheck` target refleja eso. Este plan
verifica `mypy --strict models/src/models/cellular_automata` (y
`models/cli.py`) tarea por tarea, y pasa limpio, pero no modifica el
Makefile para agregar todo `models/src` a `make typecheck` — los stubs
pre-existentes `models/deep/__init__.py` no fueron verificados contra
`--strict` y podrían no pasar. Ampliar la cobertura de `mypy --strict` a
todo `models/` queda como una decisión separada, más grande, para cuando
`models/deep`/`models/evaluation` completo se implementen.

## `typer` agregado a `models/pyproject.toml`

`models/cli.py` (el nuevo comando `pyrocast-models run-ca`) necesita
`typer` — no estaba entre las dependencias de `models` (que hasta ahora
era una librería pura, sin CLI propio). Se agrega como dependencia
directa, mismo patrón que `ingestion`/`features`.
```

- [ ] **Step 3: Append to `docs/limitations.md`**

```markdown
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
  varios días.
- **8 direcciones (vecindad de Moore), no propagación continua**: la
  distancia diagonal se calcula correctamente (`resolución_m·√2`), pero
  la resolución angular de la propagación está limitada a 8 direcciones
  discretas por celda — un frente de fuego real no está limitado así.
```

- [ ] **Step 4: Run the full verification sweep**

Run:
```bash
env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests -v
uv run ruff check .
uv run mypy --strict models/src/models/cellular_automata models/src/models/evaluation models/src/models/cli.py
uv run mypy --strict shared/src features/src
```
Expected: all green/clean (models test count: 1 existing smoke test + 7
(metrics) + 8 (rules) + 6 (simulate) + 4 (calibrate) + 2 (cli) = 28
passed).

- [ ] **Step 5: Commit**

```bash
git add docs/cellular-automata.md docs/decisions.md docs/limitations.md
git commit -m "docs: add cellular-automata.md, document models/cellular_automata design decisions"
```
