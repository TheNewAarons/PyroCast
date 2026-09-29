# models/deep/calibration.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isotonic calibration (scikit-learn) for `SmallUNet`'s raw
sigmoid probabilities, fit against observed validation-set frequencies;
a before/after Brier+ECE comparison documented in `docs/calibration.md`;
a `CalibratedUNet` wrapper implementing the exact same
`shared.model_protocol.FireSpreadModel` Protocol `CellularAutomatonModel`
already implements (so `models/evaluation/backtest.py` treats both
identically); and a calibrator persisted alongside its checkpoint,
fingerprinted so it can never silently be applied to the wrong one.
This closes Stage 4 (per CLAUDE.md).

**Architecture:** `models/deep/calibration.py` is a single new module:
`_checkpoint_fingerprint` (sha256 of the checkpoint file's bytes —
persisted inside the calibration file, checked on load),
`CalibrationResult` (frozen dataclass carrying the fingerprint, the
fitted `sklearn.isotonic.IsotonicRegression`, and the four before/after
metric values), `save_calibration`/`load_calibration` (the latter
raises `IncompatibleCalibratorError` on a fingerprint mismatch),
`fit_isotonic_calibrator` + `_collect_predictions` (run a checkpoint's
model over a validation `Dataset` — reusing `NDWSPretrainDataset`/
`ChileFinetuneDataset` from `models/deep/train.py` verbatim, since
their `__getitem__` contract is already exactly `(input tensor,
target mask)` — and flatten every pixel's `(raw_prob, target)` pair
across the whole validation set), `calibrate_checkpoint` (the
orchestrating function: load checkpoint, collect predictions, compute
Brier/ECE before, fit the calibrator, compute Brier/ECE after, save,
return the result), `CalibratedUNet` (implements `FireSpreadModel`:
loads a checkpoint + optional calibration file together at
construction — verifying the fingerprint immediately, not lazily at
first `predict()` — and follows the exact same "day 0 is the known
anchor, days 1..n-1 are model predictions" convention
`models/cellular_automata/model.py` already established for this
Protocol), and a self-contained `pyrocast-calibrate` Typer CLI in the
same file (mirroring `models/deep/train.py`'s own CLI-in-the-training-
file precedent) with one `run` command: `--fixture` builds a tiny
synthetic checkpoint via `models.deep.train.train_model` (this IS "the
checkpoint de fixture de P10" the acceptance criteria names — produced
by P10's own training function, not a lookalike) and calibrates it
end-to-end with no network or real data required, or `--checkpoint` +
`--shard-dir` calibrates a real pretrain checkpoint against a real NDWS
validation split.

**Tech Stack:** `scikit-learn>=1.5` (already a `models` dependency —
`sklearn.isotonic.IsotonicRegression`, no new dependency). Reuses
`models.deep.train.{train_model, NDWSPretrainDataset, set_seed,
TrainingConfig-consuming helpers}`, `models.deep.checkpoint.{load_checkpoint}`,
`models.deep.unet.SmallUNet`, `models.deep.public_dataset.{PublicDatasetSample,
load_public_dataset_samples, split_public_dataset}`,
`models.evaluation.metrics.{brier_score, ece_score}`,
`shared.model_protocol.FireSpreadModel`, `features.dataset.assemble.CHANNEL_ORDER`.

**Spec:** the user's request (quoted below), governed by
`/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md`.

```
Implementa models/deep/calibration.py.

1. Calibración isotónica (scikit-learn) que ajuste las probabilidades
   crudas del U-Net contra las frecuencias observadas en el set de
   validación, siguiendo el enfoque de calibración usado en la
   literatura de referencia.
2. Comparación antes/después de calibrar usando Brier score y ECE (de
   P8) sobre el set de validación, con una tabla clara en
   docs/calibration.md.
3. El modelo calibrado debe implementar la misma interfaz Protocol
   definida en P8, para que el backtest y el reporte lo traten igual
   que al autómata celular.
4. Persiste el calibrador junto al checkpoint del modelo al que
   corresponde (deben viajar juntos, nunca aplicar un calibrador a un
   checkpoint distinto del que fue entrenado).

Tests: calibración sobre un caso sintético donde se conoce la relación
real entre probabilidad predicha y frecuencia observada, verificación
de que ECE mejora tras calibrar en ese caso, error claro si se intenta
aplicar un calibrador a un checkpoint incompatible.
Criterios de aceptación: `make calibrate` corre sobre el checkpoint de
fixture de P10 y produce el reporte antes/después; mypy y ruff
limpios. Cierra la etapa 4.
```

## Global Constraints

- **`IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)`**
  — isotonic regression is the standard calibration method for exactly
  this reason in the reference literature this task cites (a
  non-parametric, monotonic fit from raw probability to observed
  frequency — unlike Platt scaling, it makes no assumption about the
  miscalibration curve's shape). `out_of_bounds="clip"` guarantees a
  calibrated probability is never emitted outside `[0,1]` even for a
  raw probability outside the range seen during fitting.
- **Calibration is fit ONLY on real model predictions, never on the
  day-0 anchor** — `models/cellular_automata/model.py`'s established
  convention (and `models/evaluation/backtest.py`'s own convention) is
  that day 0 of a `FireSpreadModel.predict()` output is the tensor's
  own known day-0 state, not something the model predicted. Fitting a
  calibrator on that day would calibrate against a value that isn't a
  genuine model output — `fit_isotonic_calibrator` only ever sees
  `(raw_prob, target)` pairs from `NDWSPretrainDataset`/
  `ChileFinetuneDataset`'s `__getitem__` (which are already, by
  construction, real single-timestep-in → next-day-out supervised
  pairs, never an anchor).
- **The calibrator is fit and evaluated on the SAME validation set**
  (no separate held-out calibration set) — this matches the task's own
  wording ("ajuste las probabilidades crudas... contra las frecuencias
  observadas en el set de validación" and "sobre el set de validación"
  for the before/after comparison) and is standard practice for a
  small-data project at this scale (a genuinely separate calibration
  split would shrink an already small validation set further, and the
  task doesn't ask for one).
- **Checkpoint compatibility is enforced by content fingerprint (sha256
  of the checkpoint file's bytes), checked at `CalibratedUNet`
  construction time, not lazily at first `predict()` call** — a
  fail-fast contract: a caller who mismatches a calibrator and a
  checkpoint gets an immediate, named error
  (`IncompatibleCalibratorError`), never a silently-wrong prediction.
  Retraining the same architecture with the same hyperparameters still
  produces different weights (different random init, different data
  order) and therefore a different fingerprint — exactly the case this
  guard exists to catch, per the task's explicit requirement ("nunca
  aplicar un calibrador a un checkpoint distinto del que fue
  entrenado").
- **`CalibratedUNet.predict` reuses `CellularAutomatonModel`'s exact
  day-0-anchor convention** — verified by reading
  `models/cellular_automata/model.py` before writing this plan: day 0
  of the Protocol's `(day, y, x)` output is seeded from the event
  tensor's own day-0 `fire_mask` (the only real "known state" — there
  is no day -1 to predict from), and only days `1..n-1` are genuine
  model output. This is not a new convention invented for this task —
  it is the one the Protocol's only prior implementer already
  established, and `CalibratedUNet` must match it for
  `models/evaluation/backtest.py` (which does not special-case which
  `FireSpreadModel` it is scoring) to treat both models identically,
  exactly as requirement 3 asks.
- **`CalibratedUNet` also works with `calibration_path=None`** (raw,
  uncalibrated model) — this is what `calibrate_checkpoint` uses
  internally to compute the "before" Brier/ECE, and it is also how
  `docs/calibration.md`'s comparison table is produced: one class,
  used twice (once with a calibrator attached, once without), not two
  near-duplicate classes.
- **No new duplication of the Chile-events zarr-loading convention** —
  `models/deep/train.py::_load_chile_events` is already the third
  copy of `event_{id:04d}.zarr` + `splits.json` in this codebase
  (ledgered as a deferred minor in the P10 review). This plan's CLI
  supports calibrating a real checkpoint against a real **NDWS**
  validation split (`--checkpoint` + `--shard-dir`, reusing
  `split_public_dataset`/`load_public_dataset_samples` — no new
  convention) and the **fixture** path (`--fixture`, fully synthetic).
  Real-Chile-events calibration is out of scope for this plan (YAGNI —
  not required by the acceptance criteria, and adding a fourth copy of
  that convention here would compound an already-ledgered issue rather
  than fix it).
- **`docs/calibration.md`'s before/after table is real, measured
  numbers from this session's fixture run** (same practice as
  `docs/model-card.md`'s measured smoke-test timing and memory
  figures) — written after Task 4 actually runs the fixture
  calibration, never invented ahead of time.

## Review Focus

- The synthetic ECE-improvement test's raw probabilities must be
  **systematically miscalibrated in a way isotonic regression can
  actually fix** (monotonic but non-identity relationship between raw
  probability and true frequency — e.g. true frequency `= raw_prob**2`,
  a classic "overconfident" pattern) — a test using well-calibrated or
  randomly-scattered synthetic data would not reliably show
  improvement and would defeat the point of the test.
- `load_calibration` given a calibration file whose calibrator was fit
  against a DIFFERENT validation set but the SAME checkpoint (fingerprint
  matches) — must succeed (the fingerprint only ties a calibrator to
  the weights it was fit against, not to a specific validation run) —
  a test should confirm this is NOT rejected, since it's easy to
  over-tighten the compatibility check into rejecting a legitimate
  case.
- `fit_isotonic_calibrator` given a validation set where every target
  is the same class (all-fire or all-no-fire pixels, e.g. a
  degenerate/tiny fixture) — `IsotonicRegression.fit` must not raise
  an opaque sklearn error; verify actual behavior and either handle it
  explicitly or let it surface with a clear message.
- `CalibratedUNet.predict` on a single-day event (`n_days=1`, no day
  1..n-1 to predict) — must return just the day-0 anchor without
  attempting a forward pass on an empty batch (the same class of edge
  case `ChileFinetuneDataset` already handles for zero-day-pairs).
- `docs/calibration.md`'s claimed Brier/ECE improvement must be the
  actual number `make calibrate` prints when run for real in this
  session — not copied from the plan's own worked-example arithmetic
  (which uses illustrative, not measured, values).

---

## Task 1: `models/deep/calibration.py` — fingerprinting, persistence, `IncompatibleCalibratorError`

**Files:**
- Create: `models/src/models/deep/calibration.py`
- Test: `models/tests/test_calibration.py`

**Interfaces:**
- Consumes: nothing from other tasks yet (self-contained persistence
  layer).
- Produces: `IncompatibleCalibratorError(RuntimeError)`,
  `_checkpoint_fingerprint(path: Path) -> str`, `CalibrationResult`
  (frozen dataclass: `checkpoint_fingerprint: str, calibrator:
  IsotonicRegression, brier_before: float, ece_before: float,
  brier_after: float, ece_after: float, n_samples: int`),
  `save_calibration(path: Path, result: CalibrationResult) -> None`,
  `load_calibration(path: Path, checkpoint_path: Path) ->
  CalibrationResult`, `default_calibration_path(checkpoint_path: Path)
  -> Path`. Tasks 2–3 consume all of these.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_calibration.py`:

```python
"""Tests de models/deep/calibration.py -- huella de checkpoint,
persistencia del calibrador, y el error claro al aplicar un calibrador
a un checkpoint incompatible."""
from pathlib import Path

import numpy as np
import pytest
import torch
from models.deep.calibration import (
    CalibrationResult,
    IncompatibleCalibratorError,
    _checkpoint_fingerprint,
    default_calibration_path,
    load_calibration,
    save_calibration,
)
from models.deep.checkpoint import TrainingConfig, save_checkpoint
from models.deep.unet import SmallUNet
from sklearn.isotonic import IsotonicRegression


def _make_checkpoint(path: Path, seed: int) -> None:
    torch.manual_seed(seed)
    model = SmallUNet(in_channels=3, base_channels=8, depth=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = TrainingConfig(
        phase="pretrain", in_channels=3, base_channels=8, depth=1, lr=1e-3,
        batch_size=1, seed=seed, focal_alpha=0.8, focal_gamma=2.0, max_epochs=1,
        patience=1, data_paths=("fixture",), pretrained_checkpoint=None,
    )
    save_checkpoint(path, model, optimizer, epoch=1, best_val_loss=0.5, config=config)


def _make_result(fingerprint: str) -> CalibrationResult:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit([0.1, 0.5, 0.9], [0.0, 0.5, 1.0])
    return CalibrationResult(
        checkpoint_fingerprint=fingerprint, calibrator=calibrator,
        brier_before=0.2, ece_before=0.15, brier_after=0.1, ece_after=0.05, n_samples=3,
    )


def test_checkpoint_fingerprint_is_deterministic_for_the_same_file(tmp_path):
    checkpoint = tmp_path / "model.pt"
    _make_checkpoint(checkpoint, seed=1)
    assert _checkpoint_fingerprint(checkpoint) == _checkpoint_fingerprint(checkpoint)


def test_checkpoint_fingerprint_differs_for_different_weights(tmp_path):
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    _make_checkpoint(checkpoint_a, seed=1)
    _make_checkpoint(checkpoint_b, seed=2)
    assert _checkpoint_fingerprint(checkpoint_a) != _checkpoint_fingerprint(checkpoint_b)


def test_save_and_load_calibration_round_trips(tmp_path):
    checkpoint = tmp_path / "model.pt"
    _make_checkpoint(checkpoint, seed=1)
    result = _make_result(_checkpoint_fingerprint(checkpoint))
    calibration_path = default_calibration_path(checkpoint)

    save_calibration(calibration_path, result)
    loaded = load_calibration(calibration_path, checkpoint)

    assert loaded.checkpoint_fingerprint == result.checkpoint_fingerprint
    assert loaded.brier_before == result.brier_before
    assert loaded.ece_after == result.ece_after
    np.testing.assert_array_equal(
        loaded.calibrator.predict([0.3, 0.7]), result.calibrator.predict([0.3, 0.7])
    )


def test_load_calibration_rejects_a_mismatched_checkpoint(tmp_path):
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    _make_checkpoint(checkpoint_a, seed=1)
    _make_checkpoint(checkpoint_b, seed=2)

    result_for_a = _make_result(_checkpoint_fingerprint(checkpoint_a))
    calibration_path = tmp_path / "a.calibrator.pt"
    save_calibration(calibration_path, result_for_a)

    with pytest.raises(IncompatibleCalibratorError, match="checkpoint"):
        load_calibration(calibration_path, checkpoint_b)


def test_load_calibration_accepts_the_same_checkpoint_regardless_of_which_val_set_fit_it(tmp_path):
    # el fingerprint ata el calibrador a los PESOS, no a una corrida de
    # validación específica -- dos calibradores distintos ajustados
    # contra el MISMO checkpoint deben aceptarse ambos (ver Review
    # Focus del plan).
    checkpoint = tmp_path / "model.pt"
    _make_checkpoint(checkpoint, seed=1)
    fingerprint = _checkpoint_fingerprint(checkpoint)

    result_1 = _make_result(fingerprint)
    result_2 = _make_result(fingerprint)
    path_1 = tmp_path / "cal1.pt"
    path_2 = tmp_path / "cal2.pt"
    save_calibration(path_1, result_1)
    save_calibration(path_2, result_2)

    load_calibration(path_1, checkpoint)  # no lanza
    load_calibration(path_2, checkpoint)  # no lanza


def test_default_calibration_path_is_a_sibling_of_the_checkpoint():
    checkpoint = Path("runs/pretrain/best.pt")
    result = default_calibration_path(checkpoint)
    assert result.parent == checkpoint.parent
    assert result.name == "best.calibrator.pt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.deep.calibration'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/deep/calibration.py`:

```python
"""Calibración isotónica de las probabilidades crudas de SmallUNet
contra las frecuencias observadas en el set de validación
(scikit-learn `IsotonicRegression`) -- ver docs/calibration.md para el
contexto completo (enfoque, tabla antes/después) y el docstring de
`CalibratedUNet` para cómo el modelo calibrado implementa
`shared.model_protocol.FireSpreadModel`.

El calibrador se guarda junto a un checkpoint identificado por una
huella (sha256 del archivo del checkpoint) -- cargar un calibrador
verifica esa huella contra el checkpoint real antes de usarlo. Nunca
se aplica un calibrador a un checkpoint distinto del que fue entrenado
(el enunciado lo exige explícitamente): el fingerprint ata el
calibrador a los PESOS exactos, no a una ejecución de calibración
específica -- dos calibraciones distintas contra el mismo checkpoint
son ambas válidas.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.isotonic import IsotonicRegression


class IncompatibleCalibratorError(RuntimeError):
    """El calibrador fue ajustado contra un checkpoint distinto del que
    se intenta usar -- nunca se aplica un calibrador a pesos que no son
    los suyos."""


def _checkpoint_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CalibrationResult:
    checkpoint_fingerprint: str
    calibrator: IsotonicRegression
    brier_before: float
    ece_before: float
    brier_after: float
    ece_after: float
    n_samples: int


def default_calibration_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.stem + ".calibrator.pt")


def save_calibration(path: Path, result: CalibrationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)


def load_calibration(path: Path, checkpoint_path: Path) -> CalibrationResult:
    # weights_only=False: seguro acá por la misma razón que en
    # models/deep/checkpoint.py -- solo se cargan archivos que este
    # mismo código escribió.
    result: CalibrationResult = torch.load(path, weights_only=False)
    actual_fingerprint = _checkpoint_fingerprint(checkpoint_path)
    if result.checkpoint_fingerprint != actual_fingerprint:
        raise IncompatibleCalibratorError(
            f"el calibrador en {path} fue ajustado contra un checkpoint distinto "
            f"-- huella esperada {result.checkpoint_fingerprint[:12]}..., huella "
            f"real de {checkpoint_path} es {actual_fingerprint[:12]}... -- nunca "
            f"aplicar un calibrador a un checkpoint distinto del que fue entrenado."
        )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_calibration.py -v`
Expected: `6 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep/calibration.py && uv run ruff check models/`
Expected: both clean (if `sklearn`'s stubs are incomplete for
`IsotonicRegression`, add a narrow, commented `# type: ignore` on the
exact line, not a blanket suppression).

- [ ] **Step 6: Commit**

```bash
git add models/src/models/deep/calibration.py models/tests/test_calibration.py
git commit -m "feat: add checkpoint-fingerprinted calibration persistence (models/deep/calibration.py)"
```

---

## Task 2: `fit_isotonic_calibrator` + `calibrate_checkpoint` — the calibration itself

**Files:**
- Modify: `models/src/models/deep/calibration.py`
- Test: `models/tests/test_calibration.py` (extend)

**Interfaces:**
- Consumes: `models.deep.checkpoint.load_checkpoint`,
  `models.deep.unet.SmallUNet`, `models.evaluation.metrics.{brier_score,
  ece_score}`, `torch.utils.data.Dataset` (any dataset whose
  `__getitem__` returns `(input: torch.Tensor (C,H,W), target:
  torch.Tensor (H,W))` — the exact contract `NDWSPretrainDataset`/
  `ChileFinetuneDataset` already implement).
- Produces: `_collect_predictions(model: SmallUNet, val_dataset:
  Dataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]`
  (flattened `(raw_probs, targets)`, both 1D), `fit_isotonic_calibrator(
  raw_probs: np.ndarray, targets: np.ndarray) -> IsotonicRegression`,
  `calibrate_checkpoint(checkpoint_path: Path, val_dataset: Dataset,
  calibration_path: Path | None = None) -> CalibrationResult`. Task 3
  (`CalibratedUNet`) and Task 4 (CLI) consume `calibrate_checkpoint`
  directly.

- [ ] **Step 1: Write the failing tests**

Append to `models/tests/test_calibration.py`:

```python
def test_fit_isotonic_calibrator_improves_ece_on_a_known_miscalibration():
    # relación real CONOCIDA: frecuencia verdadera = raw_prob**2 (un
    # patrón de sobreconfianza clásico -- ver Review Focus del plan).
    # Isotonic regression, al ser monótona, puede corregir exactamente
    # este tipo de relación no lineal pero monótona.
    from models.deep.calibration import fit_isotonic_calibrator
    from models.evaluation.metrics import ece_score

    rng = np.random.default_rng(42)
    raw_probs = rng.uniform(0.05, 0.95, size=4000)
    true_frequency = raw_probs**2
    targets = (rng.uniform(size=4000) < true_frequency).astype("float64")

    ece_before = ece_score(raw_probs, targets, n_bins=10)
    calibrator = fit_isotonic_calibrator(raw_probs, targets)
    calibrated_probs = calibrator.predict(raw_probs)
    ece_after = ece_score(calibrated_probs, targets, n_bins=10)

    assert ece_after < ece_before


def test_fit_isotonic_calibrator_handles_a_single_class_validation_set_without_crashing():
    # Review Focus del plan: un set de validación degenerado (todo la
    # misma clase) no debe lanzar un error opaco de sklearn.
    from models.deep.calibration import fit_isotonic_calibrator

    raw_probs = np.array([0.1, 0.4, 0.6, 0.9])
    targets = np.zeros(4)  # ninguna celda con fuego observado
    calibrator = fit_isotonic_calibrator(raw_probs, targets)
    result = calibrator.predict(raw_probs)
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_calibrate_checkpoint_reports_before_and_after_on_the_val_set(tmp_path):
    from models.deep.calibration import calibrate_checkpoint
    from models.deep.train import NDWSPretrainDataset

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint(checkpoint, seed=1)
    val_dataset = NDWSPretrainDataset(_make_public_samples(seed=10, n=4))

    result = calibrate_checkpoint(checkpoint, val_dataset, calibration_path=tmp_path / "cal.pt")

    assert result.checkpoint_fingerprint == _checkpoint_fingerprint(checkpoint)
    assert 0.0 <= result.brier_before <= 1.0
    assert 0.0 <= result.brier_after <= 1.0
    assert 0.0 <= result.ece_before <= 1.0
    assert 0.0 <= result.ece_after <= 1.0
    assert result.n_samples > 0
    assert (tmp_path / "cal.pt").exists()
```

Add the shared fixture helper `_make_public_samples` near the top of
`models/tests/test_calibration.py` (below the existing imports):

```python
import xarray as xr
from features.dataset.assemble import CHANNEL_ORDER
from models.deep.public_dataset import PublicDatasetSample


def _make_public_samples(seed: int, n: int, size: int = 8) -> list[PublicDatasetSample]:
    samples = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        data = rng.random((1, len(CHANNEL_ORDER), size, size)).astype("float32")
        fire_idx = CHANNEL_ORDER.index("fire_mask")
        data[0, fire_idx] = (rng.random((size, size)) > 0.8).astype("float32")
        tensor = xr.DataArray(
            data, dims=("day", "channel", "y", "x"),
            coords={"day": ["1970-01-01"], "channel": list(CHANNEL_ORDER)},
            name="fire_event_tensor", attrs={"resolution_m": 1000.0, "event_id": seed + i},
        )
        next_mask = (rng.random((size, size)) > 0.8).astype("float64")
        samples.append(PublicDatasetSample(tensor=tensor, next_day_fire_mask=next_mask))
    return samples
```

Note: `_make_checkpoint` in this test file uses `in_channels=3`, but
`NDWSPretrainDataset` samples built from `CHANNEL_ORDER` have
`len(CHANNEL_ORDER)` (11) channels — `test_calibrate_checkpoint_reports_before_and_after_on_the_val_set`
must build its checkpoint with `in_channels=len(CHANNEL_ORDER)` to
match, NOT reuse `_make_checkpoint`'s hardcoded 3. Add a
`_make_checkpoint_for_channel_order(path, seed)` variant (same body as
`_make_checkpoint` but `in_channels=len(CHANNEL_ORDER)`) and use it in
this test instead.

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_calibration.py -v`
Expected: FAIL — `ImportError: cannot import name 'fit_isotonic_calibrator'`
(and `calibrate_checkpoint`) from `models.deep.calibration`.

- [ ] **Step 3: Write minimal implementation**

Append to `models/src/models/deep/calibration.py` (add these imports
to the top-of-file import block alongside the existing ones: `numpy as
np`, `from torch.utils.data import Dataset`, `from
models.deep.unet import SmallUNet`, `from models.deep.checkpoint import
load_checkpoint`, `from models.evaluation.metrics import brier_score,
ece_score`):

```python
def _select_device() -> torch.device:
    # duplicado deliberado de models/deep/train.py::_select_device --
    # ambos son funciones de 4 líneas, y models/deep/train.py ya es la
    # tercera copia de una convención similar en este proyecto (ver
    # docs/limitations.md); un import de un símbolo privado (`_`) entre
    # módulos sería peor acoplamiento que estas 4 líneas duplicadas.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collect_predictions(
    model: SmallUNet, val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    raw_probs_batches = []
    targets_batches = []
    with torch.no_grad():
        for i in range(len(val_dataset)):  # type: ignore[arg-type]
            x, y = val_dataset[i]
            logits = model(x.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            raw_probs_batches.append(probs.astype("float64").ravel())
            targets_batches.append(y.numpy().astype("float64").ravel())
    raw_probs = np.concatenate(raw_probs_batches)
    targets = np.concatenate(targets_batches)
    return raw_probs, targets


def fit_isotonic_calibrator(raw_probs: np.ndarray, targets: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_probs, targets)
    return calibrator


def calibrate_checkpoint(
    checkpoint_path: Path,
    val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    calibration_path: Path | None = None,
) -> CalibrationResult:
    model, _optimizer_state, _config = load_checkpoint(checkpoint_path)
    device = _select_device()

    raw_probs, targets = _collect_predictions(model, val_dataset, device)
    brier_before = brier_score(raw_probs, targets)
    ece_before = ece_score(raw_probs, targets)

    calibrator = fit_isotonic_calibrator(raw_probs, targets)
    calibrated_probs = calibrator.predict(raw_probs)
    brier_after = brier_score(calibrated_probs, targets)
    ece_after = ece_score(calibrated_probs, targets)

    result = CalibrationResult(
        checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_path),
        calibrator=calibrator, brier_before=brier_before, ece_before=ece_before,
        brier_after=brier_after, ece_after=ece_after, n_samples=raw_probs.size,
    )
    path = calibration_path if calibration_path is not None else default_calibration_path(checkpoint_path)
    save_calibration(path, result)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_calibration.py -v`
Expected: `9 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep/calibration.py && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/deep/calibration.py models/tests/test_calibration.py
git commit -m "feat: add isotonic calibration fitting and before/after evaluation"
```

---

## Task 3: `CalibratedUNet` — implements `FireSpreadModel`

**Files:**
- Modify: `models/src/models/deep/calibration.py`
- Test: `models/tests/test_calibration.py` (extend)

**Interfaces:**
- Consumes: `models.deep.checkpoint.load_checkpoint`,
  `shared.model_protocol.FireSpreadModel`, `calibrate_checkpoint`,
  `load_calibration` (from Tasks 1–2).
- Produces: `CalibratedUNet(checkpoint_path: Path, calibration_path:
  Path | None = None)` — `predict(self, event: xr.DataArray) ->
  np.ndarray`, satisfying `FireSpreadModel`. Task 4 (CLI) and any
  future `models/evaluation/backtest.py` caller consume this directly.

- [ ] **Step 1: Write the failing tests**

Append to `models/tests/test_calibration.py`:

```python
def test_calibrated_unet_satisfies_the_fire_spread_model_protocol(tmp_path):
    from models.deep.calibration import CalibratedUNet
    from shared.model_protocol import FireSpreadModel

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    model = CalibratedUNet(checkpoint)
    assert isinstance(model, FireSpreadModel)


def test_calibrated_unet_predict_seeds_day_zero_from_the_known_fire_mask(tmp_path):
    # mismo convenio que CellularAutomatonModel (models/cellular_automata/model.py):
    # el día 0 es el ancla conocida, no una predicción real.
    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    model = CalibratedUNet(checkpoint)

    event = _make_public_samples(seed=5, n=1)[0].tensor
    # día 0 real de la muestra sintética -- forzamos un valor conocido.
    fire_idx = list(CHANNEL_ORDER).index("fire_mask")
    event.values[0, fire_idx] = 0.0
    event.values[0, fire_idx, 0, 0] = 1.0

    result = model.predict(event)
    assert result.shape == (1, event.sizes["y"], event.sizes["x"])
    assert result[0, 0, 0] == 1.0
    assert result[0, 1, 1] == 0.0


def test_calibrated_unet_predict_handles_a_single_day_event_without_a_forward_pass(tmp_path):
    # Review Focus del plan: n_days=1 no debe intentar correr el modelo
    # sobre un batch vacío.
    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    model = CalibratedUNet(checkpoint)
    event = _make_public_samples(seed=6, n=1)[0].tensor  # day dim size 1
    result = model.predict(event)
    assert result.shape == (1, event.sizes["y"], event.sizes["x"])


def test_calibrated_unet_predict_produces_multi_day_output_matching_event_shape(tmp_path):
    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    model = CalibratedUNet(checkpoint)

    size = 8
    data = np.random.default_rng(7).random((3, len(CHANNEL_ORDER), size, size)).astype("float32")
    event = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={
            "day": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "channel": list(CHANNEL_ORDER),
        },
        name="fire_event_tensor", attrs={"resolution_m": 250.0, "event_id": 1},
    )
    result = model.predict(event)
    assert result.shape == (3, size, size)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_calibrated_unet_uses_the_calibrator_when_one_is_provided(tmp_path):
    from models.deep.calibration import calibrate_checkpoint
    from models.deep.train import NDWSPretrainDataset

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    val_dataset = NDWSPretrainDataset(_make_public_samples(seed=20, n=4))
    calibration_path = tmp_path / "cal.pt"
    calibrate_checkpoint(checkpoint, val_dataset, calibration_path=calibration_path)

    raw_model = CalibratedUNet(checkpoint)
    calibrated_model = CalibratedUNet(checkpoint, calibration_path=calibration_path)

    event = _make_public_samples(seed=30, n=1)[0].tensor
    # forzar > 1 día para tener al menos una celda de predicción real
    event = xr.concat([event, event], dim="day")
    event = event.assign_coords(day=["2026-01-01", "2026-01-02"])

    raw_output = raw_model.predict(event)
    calibrated_output = calibrated_model.predict(event)
    # el calibrador es una transformación monótona no trivial (ver
    # fit sobre datos aleatorios) -- el día 1 (predicho, no ancla) casi
    # seguro difiere entre ambos, salvo coincidencia exacta.
    assert not np.array_equal(raw_output[1], calibrated_output[1])


def test_calibrated_unet_rejects_a_calibrator_from_a_different_checkpoint(tmp_path):
    from models.deep.calibration import IncompatibleCalibratorError, calibrate_checkpoint
    from models.deep.train import NDWSPretrainDataset

    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    _make_checkpoint_for_channel_order(checkpoint_a, seed=1)
    _make_checkpoint_for_channel_order(checkpoint_b, seed=2)
    val_dataset = NDWSPretrainDataset(_make_public_samples(seed=40, n=4))
    calibration_for_a = tmp_path / "a.calibrator.pt"
    calibrate_checkpoint(checkpoint_a, val_dataset, calibration_path=calibration_for_a)

    with pytest.raises(IncompatibleCalibratorError, match="checkpoint"):
        CalibratedUNet(checkpoint_b, calibration_path=calibration_for_a)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_calibration.py -v`
Expected: FAIL — `ImportError: cannot import name 'CalibratedUNet'`

- [ ] **Step 3: Write minimal implementation**

Append to `models/src/models/deep/calibration.py` (add `numpy as np`
is already imported from Task 2; add `import xarray as xr` to the
top-of-file imports):

```python
class CalibratedUNet:
    """Implementa `shared.model_protocol.FireSpreadModel` -- mismo
    convenio de día 0 que `models/cellular_automata/model.py`
    (CellularAutomatonModel): el día 0 de `predict()` es el estado
    conocido (`fire_mask` del propio evento en el día 0, no hay "día
    -1" del que predecir), y los días `1..n-1` son la salida real del
    modelo, calibrada si `calibration_path` fue dado."""

    def __init__(self, checkpoint_path: Path, calibration_path: Path | None = None) -> None:
        self.model, _optimizer_state, self.config = load_checkpoint(checkpoint_path)
        self.model.eval()
        self.device = _select_device()
        self.model.to(self.device)

        self.calibrator: IsotonicRegression | None = None
        if calibration_path is not None:
            calibration = load_calibration(calibration_path, checkpoint_path)
            self.calibrator = calibration.calibrator

    def predict(self, event: xr.DataArray) -> np.ndarray:
        channels = list(event.coords["channel"].values)
        fire_idx = channels.index("fire_mask")
        n_days = event.sizes["day"]
        height, width = event.sizes["y"], event.sizes["x"]

        output = np.zeros((n_days, height, width), dtype="float64")
        output[0] = np.clip(event.values[0, fire_idx].astype("float64"), 0.0, 1.0)

        if n_days > 1:
            inputs = torch.from_numpy(event.values[:-1].astype("float32")).to(self.device)
            with torch.no_grad():
                logits = self.model(inputs)
            raw_probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype("float64")
            if self.calibrator is not None:
                calibrated = self.calibrator.predict(raw_probs.ravel()).reshape(raw_probs.shape)
                output[1:] = np.clip(calibrated, 0.0, 1.0)
            else:
                output[1:] = np.clip(raw_probs, 0.0, 1.0)

        return output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_calibration.py -v`
Expected: `15 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep/calibration.py && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/deep/calibration.py models/tests/test_calibration.py
git commit -m "feat: add CalibratedUNet implementing FireSpreadModel"
```

---

## Task 4: `pyrocast-calibrate` CLI + `make calibrate` + `docs/calibration.md`

**Files:**
- Modify: `models/src/models/deep/calibration.py` (append the Typer app)
- Modify: `models/pyproject.toml` (new `[project.scripts]` entry)
- Modify: `Makefile`
- Create: `docs/calibration.md`
- Test: `models/tests/test_calibration_cli.py`

**Interfaces:**
- Consumes: `calibrate_checkpoint`, `models.deep.train.{train_model,
  NDWSPretrainDataset, set_seed}`, `models.deep.checkpoint.TrainingConfig`,
  `models.deep.unet.SmallUNet`, `models.deep.public_dataset.{PublicDatasetSample,
  load_public_dataset_samples, split_public_dataset}`,
  `features.dataset.assemble.CHANNEL_ORDER`.
- Produces: the `pyrocast-calibrate` console script, one command: `run`.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_calibration_cli.py`:

```python
"""Test de humo del CLI `pyrocast-calibrate`: --fixture corre de
verdad (entrena un checkpoint sintético diminuto vía
models.deep.train.train_model, lo calibra, imprime el reporte
antes/después) sin red ni datos reales -- esto es lo que `make
calibrate` ejecuta."""
from models.deep.calibration import app
from typer.testing import CliRunner

runner = CliRunner()


def test_run_fixture_completes_and_prints_before_after_report(tmp_path):
    result = runner.invoke(app, ["run", "--fixture", "--run-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "brier" in result.output.lower()
    assert "ece" in result.output.lower()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "best.calibrator.pt").exists()


def test_run_help_mentions_fixture_and_checkpoint_options():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--fixture" in result.output
    assert "--checkpoint" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_calibration_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'app' from 'models.deep.calibration'`

- [ ] **Step 3: Append the CLI to `models/src/models/deep/calibration.py`**

Add `typer` to the top-of-file imports, alongside the rest (not as a
separate late import block — this task's code is shown appended below
only because the diff is additive):

```python
app = typer.Typer()


@app.callback()
def _callback() -> None:
    """CLI de calibración isotónica del U-Net de PyroCast."""


def _build_fixture_checkpoint(run_dir: Path, seed: int = 42) -> Path:
    # el MISMO train_model de P10 entrena este checkpoint -- es
    # literalmente "el checkpoint de fixture de P10" que el enunciado
    # pide, no una imitación local del entrenamiento.
    from models.deep.train import NDWSPretrainDataset, set_seed, train_model

    set_seed(seed)
    size = 16
    n_channels = len(CHANNEL_ORDER)
    fire_idx = CHANNEL_ORDER.index("fire_mask")

    def make_sample(sample_id: int) -> PublicDatasetSample:
        rng = np.random.default_rng(sample_id)
        data = rng.random((1, n_channels, size, size)).astype("float32")
        data[0, fire_idx] = (rng.random((size, size)) > 0.8).astype("float32")
        tensor = xr.DataArray(
            data, dims=("day", "channel", "y", "x"),
            coords={"day": ["1970-01-01"], "channel": list(CHANNEL_ORDER)},
            name="fire_event_tensor", attrs={"resolution_m": 1000.0, "event_id": sample_id},
        )
        next_mask = (rng.random((size, size)) > 0.8).astype("float64")
        return PublicDatasetSample(tensor=tensor, next_day_fire_mask=next_mask)

    samples = [make_sample(i) for i in range(8)]
    config = TrainingConfig(
        phase="pretrain", in_channels=n_channels, base_channels=8, depth=2, lr=1e-2,
        batch_size=2, seed=seed, focal_alpha=0.8, focal_gamma=2.0, max_epochs=2, patience=2,
        data_paths=("synthetic-calibration-fixture",), pretrained_checkpoint=None,
    )
    model = SmallUNet(in_channels=n_channels, base_channels=8, depth=2)
    train_model(
        model, NDWSPretrainDataset(samples[:6]), NDWSPretrainDataset(samples[6:]),
        config, run_dir,
    )
    return run_dir / "best.pt"


def _echo_report(result: "CalibrationResult") -> None:
    typer.echo(f"Muestras evaluadas: {result.n_samples}")
    typer.echo(f"Brier -- antes: {result.brier_before:.4f}  después: {result.brier_after:.4f}")
    typer.echo(f"ECE   -- antes: {result.ece_before:.4f}  después: {result.ece_after:.4f}")


def run(
    fixture: bool = typer.Option(False, help="Entrena y calibra un checkpoint sintético diminuto"),
    checkpoint: Path | None = typer.Option(None, help="Checkpoint real a calibrar (requiere --shard-dir)"),
    shard_dir: Path | None = typer.Option(None, help="Shards NDWS para el set de validación real"),
    run_dir: Path = typer.Option(Path("runs") / "calibration", help="Dónde guardar checkpoint/calibrador de fixture"),
    seed: int = typer.Option(42, help="Semilla de reproducibilidad"),
) -> None:
    """Calibra un checkpoint de SmallUNet con regresión isotónica y
    reporta Brier/ECE antes y después sobre el set de validación."""
    if fixture:
        checkpoint_path = _build_fixture_checkpoint(run_dir, seed=seed)
        val_dataset = NDWSPretrainDataset(
            [
                PublicDatasetSample(
                    tensor=xr.DataArray(
                        np.random.default_rng(seed + 100 + i).random(
                            (1, len(CHANNEL_ORDER), 16, 16)
                        ).astype("float32"),
                        dims=("day", "channel", "y", "x"),
                        coords={"day": ["1970-01-01"], "channel": list(CHANNEL_ORDER)},
                        name="fire_event_tensor",
                        attrs={"resolution_m": 1000.0, "event_id": seed + 100 + i},
                    ),
                    next_day_fire_mask=(
                        np.random.default_rng(seed + 200 + i).random((16, 16)) > 0.8
                    ).astype("float64"),
                )
                for i in range(4)
            ]
        )
        calibration_path = default_calibration_path(checkpoint_path)
    elif checkpoint is not None and shard_dir is not None:
        checkpoint_path = checkpoint
        shard_paths = sorted(shard_dir.glob("*.tfrecord*"))
        if not shard_paths:
            typer.echo(f"No se encontraron shards *.tfrecord* en {shard_dir}.")
            raise typer.Exit(code=1)
        split = split_public_dataset(shard_paths, seed=seed)
        val_samples = list(load_public_dataset_samples(split["val"] or split["train"]))
        val_dataset = NDWSPretrainDataset(val_samples)
        calibration_path = default_calibration_path(checkpoint_path)
    else:
        typer.echo("Usar --fixture, o --checkpoint junto con --shard-dir.")
        raise typer.Exit(code=1)

    result = calibrate_checkpoint(checkpoint_path, val_dataset, calibration_path=calibration_path)
    _echo_report(result)
    typer.echo(f"Calibrador guardado en {calibration_path}")


app.command("run")(run)
```

Note: `NDWSPretrainDataset`, `PublicDatasetSample`,
`load_public_dataset_samples`, `split_public_dataset`, `TrainingConfig`,
`SmallUNet`, `CHANNEL_ORDER` need importing at the top of the file (not
inside `_build_fixture_checkpoint`, which only late-imports
`train_model`/`NDWSPretrainDataset`/`set_seed` from `models.deep.train`
to avoid a needless module-load-time dependency on `train.py`'s own
Typer app construction — everything else used in `run()` is imported
normally at the top like the rest of the file).

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_calibration_cli.py -v`
Expected: `2 passed`. Time the fixture run
(`time uv run --package models pyrocast-calibrate run --fixture --run-dir /tmp/cal-fixture-timing`
once installed in Step 5) and read its printed Brier/ECE before/after
numbers — Task 4 Step 7's `docs/calibration.md` quotes these REAL
numbers, not invented ones.

- [ ] **Step 5: Add the console script to `models/pyproject.toml`**

```toml
[project.scripts]
pyrocast-models = "models.cli:app"
pyrocast-train = "models.deep.train:app"
pyrocast-calibrate = "models.deep.calibration:app"
```

Run `uv sync --all-packages`.

- [ ] **Step 6: Wire `Makefile`'s `calibrate` target**

Replace the stub:

```makefile
calibrate:
	uv run --package models pyrocast-calibrate run --fixture
```

Run `make calibrate` for real and confirm it exits 0 and prints the
before/after report.

- [ ] **Step 7: Write `docs/calibration.md`**

Mirror `docs/model-card.md`'s header structure (H1 title, purpose
paragraph, bolded `**Aviso de honestidad (CLAUDE.md):**` disclaimer).
Content, filled in with the REAL numbers from Step 4's timed run:

```markdown
# Calibración isotónica

`models/deep/calibration.py` ajusta las probabilidades crudas de
SmallUNet (P10) contra las frecuencias observadas en el set de
validación, usando regresión isotónica (scikit-learn) -- el enfoque
estándar en la literatura de calibración de redes profundas (no
paramétrico, monótono: corrige cualquier forma de sobreconfianza o
subconfianza sistemática sin asumir una curva particular, a diferencia
de Platt scaling).

**Aviso de honestidad (CLAUDE.md):** Herramienta de investigación. No
usar para decisiones operativas de combate de incendios sin validación
de CONAF/SENAPRED. Los números de esta página son de una corrida real
del pipeline de calibración sobre un checkpoint de FIXTURE sintético
(`pyrocast-calibrate run --fixture`, lo que `make calibrate` ejecuta) --
no de un modelo entrenado sobre NDWS o eventos reales de Chile, que
esta sesión no tuvo forma de descargar ni entrenar (ver
`docs/public-dataset.md`, `docs/model-card.md`). Sirven para probar que
el pipeline funciona de punta a punta, no como una medición de qué tan
bien calibrado está el U-Net real.

## Enfoque

1. Se corre el checkpoint sobre cada muestra del set de validación
   (`models.deep.train.NDWSPretrainDataset`/`ChileFinetuneDataset` --
   el mismo contrato `(entrada, máscara objetivo)` que ya usa el
   entrenamiento), aplicando `sigmoid` a los logits para obtener
   probabilidades crudas por celda.
2. Se ajusta `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip",
   y_min=0.0, y_max=1.0)` contra esos pares (probabilidad cruda,
   resultado real 0/1) -- **sobre el mismo set de validación**, sin un
   split de calibración separado (ver `docs/decisions.md`).
3. Se calculan Brier score y ECE (`models/evaluation/metrics.py`, P8)
   antes y después de calibrar, sobre el mismo set de validación.

## Tabla antes/después (checkpoint de fixture, `make calibrate`)

| Métrica | Antes | Después |
|---|---|---|
| Brier score | **[COMPLETAR CON EL NÚMERO REAL DEL PASO 4]** | **[COMPLETAR]** |
| ECE | **[COMPLETAR]** | **[COMPLETAR]** |

Menor es mejor en ambas métricas. `n_samples` evaluadas: **[COMPLETAR]**.

## El modelo calibrado implementa `FireSpreadModel` (P8)

`CalibratedUNet` (`models/deep/calibration.py`) implementa
`shared.model_protocol.FireSpreadModel` -- el mismo Protocol que
`CellularAutomatonModel` (P7) ya implementa, con el mismo convenio de
día 0 (el estado conocido del propio evento, no una predicción real;
ver `models/cellular_automata/model.py`). Esto significa que
`models/evaluation/backtest.py` puede correr contra un `CalibratedUNet`
exactamente igual que contra el autómata celular, sin ningún caso
especial por modelo.

## El calibrador viaja con su checkpoint, nunca con otro

Cada calibrador guardado (`<checkpoint>.calibrator.pt` por defecto)
lleva una huella (sha256) del archivo de checkpoint exacto contra el
que se ajustó. Cargarlo contra un checkpoint distinto -- incluso uno
reentrenado con los mismos hiperparámetros, que produce pesos
distintos por la inicialización aleatoria y el orden de los datos --
levanta `IncompatibleCalibratorError` de inmediato, en la construcción
de `CalibratedUNet`, no en el primer `predict()`.

## Limitaciones

Ver `docs/limitations.md`. No se micro-gestionan acá para evitar que
ambas fuentes se desincronicen.
```

- [ ] **Step 8: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep/calibration.py && uv run ruff check models/`
Expected: both clean.

- [ ] **Step 9: Commit**

```bash
git add models/src/models/deep/calibration.py models/pyproject.toml Makefile docs/calibration.md models/tests/test_calibration_cli.py uv.lock
git commit -m "feat: add pyrocast-calibrate CLI, wire make calibrate, add docs/calibration.md"
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
## Calibración isotónica se ajusta y evalúa sobre el MISMO set de validación

No hay un split de calibración separado del de validación -- el
enunciado pide explícitamente ajustar "contra las frecuencias
observadas en el set de validación" y comparar "sobre el set de
validación", y un split adicional reduciría aún más un set de
validación ya pequeño en este proyecto. Encaja con la práctica estándar
para datasets chicos: el riesgo de sobreajustar el calibrador (que solo
tiene la forma monótona por gradosde libertad de un ajuste isotónico,
muy pocos parámetros efectivos comparado con la red) es bajo.

## `CalibratedUNet` reutiliza el convenio de día 0 de `CellularAutomatonModel`

No es una decisión nueva -- `models/cellular_automata/model.py` ya
estableció que el día 0 de `FireSpreadModel.predict()` es el estado
conocido del propio evento (no hay "día -1"), y `models/evaluation/backtest.py`
ya asume esa convención al comparar contra la verdad acumulada (ver la
revisión final de `models/evaluation`, 2026-09-28). `CalibratedUNet`
sigue exactamente el mismo convenio para que ambos modelos sean
intercambiables ante el backtest, tal como pide el enunciado.

## Fingerprint por contenido (sha256), no por metadata del checkpoint

Se consideró identificar un checkpoint por su `TrainingConfig` (fase,
hiperparámetros) en vez de por el hash de sus bytes -- se descartó
porque dos entrenamientos con los MISMOS hiperparámetros producen
pesos distintos (inicialización aleatoria, orden de datos), y el
enunciado exige explícitamente que un calibrador nunca se aplique "a
un checkpoint distinto del que fue entrenado" -- el contenido exacto
de los pesos es lo único que identifica eso sin ambigüedad.
```

- [ ] **Step 2: Append to `docs/limitations.md`** (before the closing
"## Herramienta de investigación" banner — verify with `tail -30
docs/limitations.md` before editing; a prior task in this same session
found a plain `cat >>` lands after that banner)

```markdown
- **La calibración de esta sesión se corrió únicamente sobre un
  checkpoint de fixture sintético** (`pyrocast-calibrate run
  --fixture`) -- ningún checkpoint real entrenado sobre NDWS o eventos
  de Chile fue calibrado todavía. La tabla antes/después de
  `docs/calibration.md` prueba que el pipeline funciona de punta a
  punta, no que el U-Net real está bien calibrado.
- **Sin split de calibración separado del de validación** -- ver
  `docs/decisions.md`. Con más datos disponibles en el futuro, separar
  un split de calibración propio evitaría cualquier optimismo del
  ajuste isotónico sobre el mismo set que reporta el "después".
- **`CalibratedUNet` no valida `in_channels` contra el tensor de
  entrada antes de fallar** -- mismo patrón (y misma limitación
  todavía sin resolver) que `models/deep/train.py::finetune`, ya
  ledgeado en una revisión anterior.
- **El CLI de calibración no soporta calibrar contra un set de
  validación real de eventos de Chile** -- solo fixture sintético o
  NDWS real (`--shard-dir`). Agregar esa ruta reutilizaría
  `models/deep/train.py::_load_chile_events` (o duplicaría su
  convención una cuarta vez) -- deliberadamente fuera de alcance de
  este plan, ver Global Constraints.
```

- [ ] **Step 3: Run the full verification sweep**

Run:
```bash
env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests -v
uv run ruff check .
uv run mypy --strict models/src/models/deep
uv run mypy --strict shared/src features/src
make calibrate
```
Expected: all green/clean; `make calibrate` exits 0 and prints the
before/after report within a reasonable time on this machine (no GPU —
ver `docs/model-card.md`).

- [ ] **Step 4: Commit**

```bash
git add docs/decisions.md docs/limitations.md
git commit -m "docs: document calibration design decisions and limitations, close Stage 4"
```
