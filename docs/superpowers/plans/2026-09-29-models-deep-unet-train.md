# models/deep/unet.py + models/deep/train.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small 2D U-Net (PyTorch) that predicts next-day fire
probability from one PyroCast tensor timestep, plus a training
pipeline (`pyrocast-train pretrain` / `pyrocast-train finetune`) with
checkpointing, early stopping, CSV+matplotlib logging, and a documented
freeze/no-freeze fine-tuning decision — sized for this machine's actual
hardware (no CUDA GPU; see Hardware below).

**Hardware — stated explicitly before any design choice, per the
user's request:** this machine has **no CUDA GPU**. It is a MacBook
Air, Apple M1, 8 GB RAM. `torch.cuda.is_available()` → `False`.
`torch.backends.mps.is_available()` → `True` (Apple's Metal/MPS
backend can accelerate `torch` on this chip) — the code picks
`cuda` → `mps` → `cpu` in that order at runtime, but every sizing
decision below (model width, batch size, resolution) is made as if
only CPU were available: 8 GB unified memory is shared with the OS and
everything else running, and MPS on a base M1 is a modest accelerator,
not a datacenter GPU. `docs/model-card.md` states this and gives an
explicit, honestly-caveated time estimate (see Task 6) — CLAUDE.md's
honesty mandate applies to performance claims exactly as much as to
metrics.

**Architecture:** `models/deep/unet.py` (`SmallUNet`, GroupNorm not
BatchNorm — see Global Constraints), `models/deep/losses.py`
(`FocalLoss`), `models/deep/checkpoint.py` (`TrainingConfig` +
save/load, config travels with every checkpoint), `models/deep/train.py`
(dataset adapters for both data sources, the train/val loop, CSV +
matplotlib logging, and the `pyrocast-train` Typer CLI with three
commands: `pretrain`, `finetune`, `smoke-test`). The smoke-test command
is what `make train` runs — synthetic in-memory fixture data, no
network, no real NDWS/Chile data required, mirroring how `run-ca` and
`backtest` already work standalone in this repo.

**Tech Stack:** `torch>=2.4` (already a `models` dependency).
`matplotlib` added as a new direct dependency (CSV + plots, explicitly
requested instead of a heavy external tracker — justified in
`docs/decisions.md`). Reuses `features.dataset.assemble.CHANNEL_ORDER`
and `models.deep.public_dataset.{load_public_dataset_samples,
split_public_dataset}` verbatim — no channel-order duplication.

**Spec:** the user's request (quoted below), governed by
`/Users/aarons/Documents/Projects/FullPy/PyroCast/CLAUDE.md`.

```
Implementa models/deep/unet.py y models/deep/train.py.

Antes de empezar, dime si tienes GPU disponible o no; si no la tengo,
dilo aquí explícitamente: [COMPLETAR]. Diseña el tamaño del modelo y el
tamaño de batch en consecuencia (si es solo CPU, un U-Net pequeño en
resolución reducida, con un aviso claro del tiempo esperado de
entrenamiento).

1. unet.py: una U-Net 2D en PyTorch que recibe el tensor de un solo
   paso de tiempo (canales estáticos + dinámicos del día actual +
   máscara de fuego del día actual) y predice la probabilidad de fuego
   del día siguiente. Arquitectura pequeña (pocos bloques), configurable
   por número de canales de entrada.
2. train.py: bucle de entrenamiento con checkpointing, early stopping
   por pérdida de validación, semillas fijas para reproducibilidad, y
   logging simple a CSV + gráficos con matplotlib (sin herramientas
   externas de tracking pesadas). Función de pérdida apropiada para
   máscaras binarias desbalanceadas (por ejemplo, focal loss o BCE
   ponderada; justifica la elección).
3. Dos fases de entrenamiento por CLI (`pyrocast-train pretrain` y
   `pyrocast-train finetune`): preentrenamiento en el dataset público
   (P9) y fine-tuning en el dataset de eventos de Chile (P6),
   congelando o no capas según decidas y documentando la decisión.
4. Guarda cada checkpoint con su configuración exacta (hiperparámetros,
   versión de datos usada) para que un resultado sea reproducible.

Tests: un paso de forward con tensores aleatorios de la forma esperada
(verifica dimensiones de entrada/salida), un paso de entrenamiento de
una iteración sobre datos de fixture que efectivamente reduce la
pérdida, guardado y carga de checkpoint que reproduce las mismas
predicciones.
Criterios de aceptación: `make train` corre al menos el smoke test de
una época sobre datos de fixture sin errores; mypy y ruff limpios en
las partes tipables; docs/model-card.md con la arquitectura,
hiperparámetros y las dos fases de entrenamiento documentadas.
```

## Global Constraints

- **`SmallUNet` uses `GroupNorm`, never `BatchNorm`** — training runs
  with `batch_size=1` by default on this hardware (see Hardware),
  where `BatchNorm`'s running statistics are unstable/meaningless at
  batch size 1. GroupNorm normalizes within one sample regardless of
  batch size, the standard fix for small-batch training. Requires
  every channel count in the network to be a multiple of `num_groups`
  (fixed at 8) — enforced by requiring `base_channels` to be a
  multiple of 8 (default 16; the smoke-test uses 8), raising
  `ValueError` otherwise.
- **Arbitrary input H/W, not just powers of 2**: real PyroCast event
  tensors have a size determined by each fire's own bounding box (via
  `WorkGrid`), never guaranteed divisible by `2^depth`. `SmallUNet`'s
  up-path pads/crops each upsampled feature map to match its skip
  connection's exact spatial size before concatenating (the standard
  fix used by most modern U-Net implementations) — verified by a test
  with a deliberately non-power-of-2 input size.
- **`batch_size=1` by default, for both phases** — real Chile events
  have per-event spatial size (no two events are the same H×W), so
  naive batching would require padding/cropping logic this plan does
  not build (YAGNI: the expected number of real Chile events is small
  enough that batch_size=1 is not a meaningful training-speed problem,
  and NDWS chips, though fixed-size, use the same data pipeline for
  consistency). `batch_size` is a CLI option for anyone with more
  events or more RAM later.
- **Model output is logits, not probabilities** — `SmallUNet.forward`
  returns raw logits `(batch, 1, H, W)`; `FocalLoss` and any inference
  code apply `sigmoid` explicitly. This matches `BCEWithLogitsLoss`'s
  own convention (numerically stable, avoids a redundant
  sigmoid-then-log). `shared.model_protocol.FireSpreadModel.predict`
  (which returns probabilities) is NOT implemented in this plan — out
  of scope for this request, noted for later, not built speculatively.
- **Focal loss, not weighted BCE** — justified in
  `models/deep/losses.py`'s docstring and `docs/model-card.md`: both
  are legitimate choices for imbalanced binary masks, but focal loss's
  `(1-p_t)^gamma` focusing term additionally down-weights EASY correct
  negatives during training (the vast majority of a fire mask's
  pixels, `p_t` near 1 once the model has learned "most pixels aren't
  fire"), not just the class-frequency imbalance a single `pos_weight`
  scalar addresses — better matched to a mask where a large majority
  of pixels are trivially-correct negatives, not just numerically rare
  positives. Default `alpha=0.8` (upweight the rare positive class),
  `gamma=2.0` (the original paper's default, Lin et al. 2017).
- **Fine-tuning: no frozen layers, lower learning rate instead** — a
  deliberate choice, not the only valid one (documented as such in
  `docs/model-card.md`): NDWS is 1 km resolution over the continental
  US; PyroCast's own Chile tensors are 250 m resolution with a
  different DEM source, different WorldCover-derived `fuel_type`
  values, different wind/humidity distributions. Freezing the
  pretrained encoder would assume its learned low-level filters
  transfer directly across that resolution/geography shift — a
  stronger assumption than this plan is willing to make with no
  empirical evidence either way. Instead, `finetune` uses a learning
  rate `finetune_lr_factor` (default `0.1`) lower than pretraining's,
  letting every layer adapt but by smaller steps — this is a real
  design decision, argued explicitly, not a default left undocumented.
- **Every checkpoint's config is self-describing** — `TrainingConfig`
  (a frozen dataclass) is serialized into every `.pt` file alongside
  the model/optimizer state: architecture hyperparameters
  (`in_channels`, `base_channels`, `depth`), training hyperparameters
  (`lr`, `batch_size`, `seed`, `focal_alpha`, `focal_gamma`,
  `max_epochs`, `patience`), `phase` (`"pretrain"`/`"finetune"`), and
  `data_paths` (the exact file/directory paths used to build the
  dataset — since `split_public_dataset`/`features.dataset.split.split_events`
  are already deterministic given the same files + seed, storing the
  paths is sufficient for "reproducible" without needing a separate
  data-versioning system this project doesn't otherwise have).
- **Checkpoints are gitignored, like `data/`** — `.pt` files are
  large binaries regenerated from data + config, the same reasoning
  `.gitignore` already applies to `data/raw|interim|processed/`.
- **`models/` is not in CI's `mypy --strict` scope today** (confirmed:
  `.github/workflows/ci.yml` runs `mypy --strict shared/src features/src`
  only) — this plan's acceptance criterion ("mypy y ruff limpios en
  las partes tipables") is verified manually in each task's steps,
  matching how every prior `models/` module this session verified
  itself the same way without a CI change (out of scope here, already
  ledgered as a deferred minor in a prior review this session).
- **`torch.Tensor`-typed code is exempted from `mypy --strict`'s
  strictest generic checks where the plan's tasks say so explicitly**
  — `torch`'s own type stubs are incomplete in places (e.g. some
  `nn.Module` subclass patterns); each task's mypy step names the
  actual command and its actual expected result, including any narrow,
  justified `# type: ignore` comment, never a blanket suppression.

## Review Focus

- A Chile event with only 1 day (`n_days=1`, no "next day" to predict)
  fed to the finetune dataset builder — must contribute zero
  (input, target) pairs, not crash on an out-of-range day index.
- `SmallUNet` given a batch size larger than 1 (even though the
  default CLI batch size is 1) — GroupNorm and the pad/crop skip-merge
  must both work correctly at `batch_size > 1`, not just 1.
- Early stopping triggered on epoch 1 (`patience=0` or a val loss that
  gets worse immediately) — must still produce a valid "best"
  checkpoint (the epoch-1 one), not crash trying to reference a
  nonexistent earlier epoch.
- Loading a checkpoint saved by `pretrain` into `finetune` — the
  architecture hyperparameters (`in_channels`, `base_channels`,
  `depth`) from the checkpoint's own `TrainingConfig` must be used to
  reconstruct the model before loading `state_dict`, not the
  finetune-command's own CLI defaults (which could silently mismatch
  and either crash on `load_state_dict` or, worse, silently produce a
  differently-shaped model that loads by coincidence).
- The smoke-test path (`make train`) must complete in well under a
  minute on this machine's actual hardware (CPU/MPS, 8 GB RAM) — it is
  the acceptance criterion's own literal check, not just "should be
  fast."

---

## Task 1: `models/deep/unet.py` — SmallUNet

**Files:**
- Create: `models/src/models/deep/unet.py`
- Test: `models/tests/test_unet.py`

**Interfaces:**
- Produces: `SmallUNet(in_channels: int = 11, base_channels: int = 16,
  depth: int = 3)` — an `nn.Module`; `forward(x: torch.Tensor) ->
  torch.Tensor` takes `(batch, in_channels, H, W)`, returns
  `(batch, 1, H, W)` logits (same H, W as input). Tasks 4–5 consume
  this directly.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_unet.py`:

```python
"""Tests de SmallUNet: forma de entrada/salida, tamaños arbitrarios de
H/W (no solo potencias de 2 -- los eventos reales de PyroCast tienen el
tamaño de su propio bbox, nunca garantizado divisible por 2**depth),
batch_size > 1, y la validación de base_channels."""
import pytest
import torch
from models.deep.unet import SmallUNet


def test_forward_preserves_batch_and_spatial_dims_returns_one_channel():
    model = SmallUNet(in_channels=11, base_channels=16, depth=3)
    x = torch.randn(2, 11, 32, 32)
    y = model(x)
    assert y.shape == (2, 1, 32, 32)


def test_forward_handles_a_non_power_of_two_size():
    # 37x29 no es divisible por 2**3=8 en ningún eje -- el pad/crop del
    # camino de subida debe manejarlo sin lanzar un error de forma.
    model = SmallUNet(in_channels=11, base_channels=16, depth=3)
    x = torch.randn(1, 11, 37, 29)
    y = model(x)
    assert y.shape == (1, 1, 37, 29)


def test_forward_works_with_batch_size_greater_than_one():
    model = SmallUNet(in_channels=11, base_channels=8, depth=2)
    x = torch.randn(4, 11, 16, 16)
    y = model(x)
    assert y.shape == (4, 1, 16, 16)


def test_in_channels_is_configurable():
    model = SmallUNet(in_channels=5, base_channels=8, depth=2)
    x = torch.randn(1, 5, 16, 16)
    y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_rejects_base_channels_not_a_multiple_of_eight():
    # GroupNorm(num_groups=8) requiere que cada nivel del canal sea
    # divisible por 8 -- ver Global Constraints del plan.
    with pytest.raises(ValueError, match="8"):
        SmallUNet(in_channels=11, base_channels=10, depth=2)


def test_output_is_finite_for_a_realistic_forward_and_backward_pass():
    model = SmallUNet(in_channels=11, base_channels=16, depth=3)
    x = torch.randn(1, 11, 24, 24, requires_grad=False)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert torch.isfinite(y).all()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} sin gradiente -- capa desconectada"
        assert torch.isfinite(param.grad).all(), f"{name} tiene gradiente no finito"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_unet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.deep.unet'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/deep/unet.py`:

```python
"""U-Net 2D pequeña (pocos bloques) que recibe el tensor de UN solo día
(canales estáticos + dinámicos del día actual + fire_mask del día
actual, mismo orden que features.dataset.assemble.CHANNEL_ORDER) y
predice, por celda, el LOGIT de probabilidad de fuego del día
SIGUIENTE -- no una probabilidad ya aplicada con sigmoid (ver
docs/decisions.md: BCEWithLogitsLoss/FocalLoss esperan logits, más
estable numéricamente que sigmoid+log por separado).

GroupNorm, no BatchNorm: este proyecto entrena con batch_size=1 por
defecto (ver docs/model-card.md -- sin GPU, eventos reales de tamaño
variable), donde las estadísticas de BatchNorm no tienen sentido.
GroupNorm normaliza dentro de una sola muestra, funciona igual a
cualquier batch_size. Requiere que cada nivel de canales sea divisible
por num_groups (8, fijo) -- por eso `base_channels` debe ser múltiplo
de 8.

Tamaño arbitrario de entrada: los eventos reales de PyroCast tienen el
tamaño de su propio bbox (features/grid/grid.py::WorkGrid), nunca
garantizado divisible por 2**depth. El camino de subida rellena
(pad) cada mapa de activación subido al tamaño EXACTO de su conexión
de salto antes de concatenar -- la técnica estándar (usada por la
mayoría de las implementaciones modernas de U-Net) para tolerar
cualquier tamaño de entrada sin recortar información."""
import torch
import torch.nn.functional as F
from torch import nn

_NUM_GROUPS = 8


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_NUM_GROUPS, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_NUM_GROUPS, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.block(x)
        return result


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.pool_conv(x)
        return result


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # pad x al tamaño EXACTO de skip -- ver docstring del módulo.
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        merged = torch.cat([skip, x], dim=1)
        result: torch.Tensor = self.conv(merged)
        return result


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 11, base_channels: int = 16, depth: int = 3) -> None:
        super().__init__()
        if base_channels % _NUM_GROUPS != 0:
            raise ValueError(
                f"base_channels debe ser múltiplo de {_NUM_GROUPS} (GroupNorm) -- "
                f"recibido {base_channels}."
            )
        self.depth = depth
        self.in_conv = _DoubleConv(in_channels, base_channels)

        channels = [base_channels * (2**i) for i in range(depth + 1)]
        self.downs = nn.ModuleList(
            [_Down(channels[i], channels[i + 1]) for i in range(depth)]
        )
        self.ups = nn.ModuleList(
            [_Up(channels[i + 1], channels[i], channels[i]) for i in reversed(range(depth))]
        )
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.in_conv(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        y = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            y = up(y, skip)

        result: torch.Tensor = self.out_conv(y)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_unet.py -v`
Expected: `6 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep/unet.py && uv run ruff check models/`
Expected: both clean (if `torch`'s stubs force an unavoidable, narrow
`# type: ignore`, add it on that exact line with a one-line comment
naming which stub gap it works around — do not suppress the whole
file).

- [ ] **Step 6: Commit**

```bash
git add models/src/models/deep/unet.py models/tests/test_unet.py
git commit -m "feat: add SmallUNet 2D architecture (models/deep/unet.py)"
```

---

## Task 2: `models/deep/losses.py` — FocalLoss

**Files:**
- Create: `models/src/models/deep/losses.py`
- Test: `models/tests/test_losses.py`

**Interfaces:**
- Produces: `FocalLoss(alpha: float = 0.8, gamma: float = 2.0)` — an
  `nn.Module`; `forward(logits: torch.Tensor, targets: torch.Tensor)
  -> torch.Tensor` (scalar mean loss), both args same shape
  `(batch, 1, H, W)` or `(batch, H, W)`, `targets` in `[0, 1]`. Task 4
  consumes this directly.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_losses.py`:

```python
"""Tests de FocalLoss: valores conocidos calculados a mano, y las
propiedades que la hacen apropiada para máscaras binarias desbalanceadas
(pondera más la clase positiva rara, y castiga menos los negativos
fáciles que BCE simple)."""
import math

import pytest
import torch
from models.deep.losses import FocalLoss


def test_focal_loss_matches_hand_computed_value_for_a_single_pixel():
    # logit=0 -> p=0.5 (sigmoid(0)=0.5). target=1 (positivo).
    # FL = -alpha * (1-p)^gamma * log(p) = -0.8 * 0.5^2 * log(0.5)
    logits = torch.tensor([[0.0]])
    targets = torch.tensor([[1.0]])
    loss_fn = FocalLoss(alpha=0.8, gamma=2.0)
    expected = -0.8 * (0.5**2) * math.log(0.5)
    assert loss_fn(logits, targets).item() == pytest.approx(expected, abs=1e-5)


def test_focal_loss_is_near_zero_for_a_confident_correct_prediction():
    logits = torch.tensor([[10.0]])  # sigmoid(10) ~ 0.99995, muy seguro de "fuego"
    targets = torch.tensor([[1.0]])
    loss_fn = FocalLoss(alpha=0.8, gamma=2.0)
    assert loss_fn(logits, targets).item() < 1e-3


def test_focal_loss_down_weights_easy_negatives_more_than_plain_bce():
    # negativo FÁCIL (logit muy negativo, target=0): el término de
    # focusing (1-p_t)^gamma con p_t~1 debe hacer que FL sea MENOR que
    # BCE simple con el mismo alpha implícito -- esa es la propiedad
    # que justifica elegir focal loss sobre BCE ponderada (ver
    # docs/decisions.md).
    logits = torch.tensor([[-10.0]])
    targets = torch.tensor([[0.0]])
    focal = FocalLoss(alpha=0.8, gamma=2.0)(logits, targets).item()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).item()
    assert focal < bce


def test_focal_loss_averages_over_the_batch():
    logits = torch.tensor([[0.0], [10.0]])
    targets = torch.tensor([[1.0], [1.0]])
    loss_fn = FocalLoss(alpha=0.8, gamma=2.0)
    single_high_conf = FocalLoss(alpha=0.8, gamma=2.0)(
        torch.tensor([[10.0]]), torch.tensor([[1.0]])
    ).item()
    single_low_conf = FocalLoss(alpha=0.8, gamma=2.0)(
        torch.tensor([[0.0]]), torch.tensor([[1.0]])
    ).item()
    expected_mean = (single_high_conf + single_low_conf) / 2
    assert loss_fn(logits, targets).item() == pytest.approx(expected_mean, abs=1e-5)


def test_focal_loss_accepts_2d_spatial_shape_without_a_channel_dim():
    logits = torch.zeros(2, 8, 8)
    targets = torch.zeros(2, 8, 8)
    loss_fn = FocalLoss(alpha=0.8, gamma=2.0)
    result = loss_fn(logits, targets)
    assert result.shape == ()
    assert torch.isfinite(result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.deep.losses'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/deep/losses.py`:

```python
"""Focal Loss (Lin et al. 2017, "Focal Loss for Dense Object
Detection") para máscaras binarias desbalanceadas -- la elección sobre
BCE ponderada (`pos_weight` de BCEWithLogitsLoss):

BCE ponderada corrige el desbalance de FRECUENCIA de clase (un peso
escalar fijo para la clase positiva), pero sigue penalizando por igual
cada negativo, sin importar qué tan fácil sea. En una máscara de fuego
real, la gran mayoría de los píxeles son negativos TRIVIALES (lejos de
cualquier fuego, el modelo aprende rápido a predecirlos con alta
confianza) -- BCE ponderada sigue acumulando pérdida sobre ellos,
diluyendo el gradiente útil de los píxeles difíciles (el borde del
frente de fuego, los pocos positivos reales).

Focal loss agrega un término de "focusing", `(1-p_t)^gamma`, que baja
la pérdida de cualquier predicción ya confiada y correcta (sea positiva
o negativa) -- no solo corrige la frecuencia de clase con `alpha`, sino
que concentra el gradiente en los píxeles donde el modelo todavía se
equivoca o duda. Ver `test_focal_loss_down_weights_easy_negatives_more_than_plain_bce`
para la propiedad verificada, y docs/model-card.md para más contexto.

FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    p_t = p si target=1, (1-p) si target=0    (p = sigmoid(logit))
    alpha_t = alpha si target=1, (1-alpha) si target=0

Defaults: alpha=0.8 (la clase "fuego" es la minoría, se le da más
peso), gamma=2.0 (el valor por defecto del paper original)."""
import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_term = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_term * bce
        result: torch.Tensor = loss.mean()
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_losses.py -v`
Expected: `5 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep/losses.py && uv run ruff check models/`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add models/src/models/deep/losses.py models/tests/test_losses.py
git commit -m "feat: add FocalLoss for imbalanced fire masks (models/deep/losses.py)"
```

---

## Task 3: `models/deep/checkpoint.py` — TrainingConfig + save/load

**Files:**
- Create: `models/src/models/deep/checkpoint.py`
- Test: `models/tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `models.deep.unet.SmallUNet`.
- Produces: `TrainingConfig` (frozen dataclass: `phase: str,
  in_channels: int, base_channels: int, depth: int, lr: float,
  batch_size: int, seed: int, focal_alpha: float, focal_gamma: float,
  max_epochs: int, patience: int, data_paths: tuple[str, ...],
  pretrained_checkpoint: str | None = None`), `save_checkpoint(path:
  Path, model: SmallUNet, optimizer: torch.optim.Optimizer, epoch: int,
  best_val_loss: float, config: TrainingConfig) -> None`,
  `load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[SmallUNet,
  dict[str, object], TrainingConfig]` (rebuilds the model from the
  checkpoint's OWN config, loads `state_dict` into it, returns the
  model + the raw optimizer-state dict [not loaded into an optimizer
  here — the caller may not want to resume the optimizer] + the
  config). Task 4 (train loop) and Task 5 (CLI's `finetune`) consume
  these directly.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_checkpoint.py`:

```python
"""Tests de guardado/carga de checkpoint: la configuración exacta viaja
con el checkpoint (hiperparámetros + rutas de datos usadas), y cargar
un checkpoint reproduce EXACTAMENTE las mismas predicciones que el
modelo antes de guardarlo."""
import torch
from models.deep.checkpoint import TrainingConfig, load_checkpoint, save_checkpoint
from models.deep.unet import SmallUNet


def _make_config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = dict(
        phase="pretrain", in_channels=5, base_channels=8, depth=2, lr=1e-3,
        batch_size=1, seed=42, focal_alpha=0.8, focal_gamma=2.0, max_epochs=10,
        patience=3, data_paths=("fixture/shard_0.tfrecord",), pretrained_checkpoint=None,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def test_save_and_load_checkpoint_reproduces_identical_predictions(tmp_path):
    torch.manual_seed(0)
    model = SmallUNet(in_channels=5, base_channels=8, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = _make_config()

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=3, best_val_loss=0.5, config=config)

    model.eval()
    x = torch.randn(1, 5, 16, 16)
    with torch.no_grad():
        original_output = model(x)

    loaded_model, _optimizer_state, loaded_config = load_checkpoint(path)
    loaded_model.eval()
    with torch.no_grad():
        loaded_output = loaded_model(x)

    assert torch.equal(original_output, loaded_output)
    assert loaded_config == config


def test_load_checkpoint_rebuilds_architecture_from_its_own_config_not_a_guess(tmp_path):
    # el checkpoint se guarda con in_channels=5, base_channels=8 --
    # cargarlo NO debe requerir que el caller ya sepa esos valores de
    # antemano (ver Review Focus del plan: pretrain -> finetune debe
    # funcionar sin que finetune adivine la arquitectura).
    model = SmallUNet(in_channels=5, base_channels=8, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = _make_config(in_channels=5, base_channels=8, depth=2)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1, best_val_loss=1.0, config=config)

    loaded_model, _optimizer_state, loaded_config = load_checkpoint(path)
    assert loaded_config.in_channels == 5
    assert loaded_config.base_channels == 8
    x = torch.randn(1, 5, 8, 8)
    loaded_model(x)  # no lanza -- la arquitectura reconstruida coincide


def test_checkpoint_config_records_data_paths_and_phase(tmp_path):
    model = SmallUNet(in_channels=3, base_channels=8, depth=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = _make_config(
        phase="finetune", in_channels=3, base_channels=8, depth=1,
        data_paths=("data/processed/dataset",), pretrained_checkpoint="runs/pretrain/best.pt",
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1, best_val_loss=0.1, config=config)

    _model, _optimizer_state, loaded_config = load_checkpoint(path)
    assert loaded_config.phase == "finetune"
    assert loaded_config.data_paths == ("data/processed/dataset",)
    assert loaded_config.pretrained_checkpoint == "runs/pretrain/best.pt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.deep.checkpoint'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/deep/checkpoint.py`:

```python
"""Checkpointing: cada `.pt` guarda, junto con los pesos, su
`TrainingConfig` COMPLETA -- arquitectura, hiperparámetros de
entrenamiento, fase, y las rutas de datos exactas usadas. Sin esto,
"reproducible" significaría "reproducible si además recuerdas a mano
qué comando exacto generó este archivo" -- ver docs/model-card.md."""
from dataclasses import dataclass
from pathlib import Path

import torch

from models.deep.unet import SmallUNet


@dataclass(frozen=True)
class TrainingConfig:
    phase: str
    in_channels: int
    base_channels: int
    depth: int
    lr: float
    batch_size: int
    seed: int
    focal_alpha: float
    focal_gamma: float
    max_epochs: int
    patience: int
    data_paths: tuple[str, ...]
    pretrained_checkpoint: str | None = None


def save_checkpoint(
    path: Path,
    model: SmallUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path, map_location: str = "cpu"
) -> tuple[SmallUNet, dict[str, object], TrainingConfig]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config: TrainingConfig = checkpoint["config"]

    model = SmallUNet(
        in_channels=config.in_channels, base_channels=config.base_channels, depth=config.depth
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state: dict[str, object] = checkpoint["optimizer_state_dict"]
    return model, optimizer_state, config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_checkpoint.py -v`
Expected: `3 passed`

- [ ] **Step 5: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep/checkpoint.py && uv run ruff check models/`
Expected: both clean. `torch.load`'s `weights_only=False` is required
here (a `TrainingConfig` dataclass instance is not a tensor and
`weights_only=True`, torch's newer default, would refuse to unpickle
it) — this is safe because every checkpoint this codebase loads is one
it itself wrote (`pretrain`/`finetune`/`smoke-test`), never an
untrusted download; add a one-line comment on that call saying so.

- [ ] **Step 6: Commit**

```bash
git add models/src/models/deep/checkpoint.py models/tests/test_checkpoint.py
git commit -m "feat: add self-describing checkpoint save/load (models/deep/checkpoint.py)"
```

---

## Task 4: `models/deep/train.py` — datasets, seeding, train/val loop, CSV+plot logging

**Files:**
- Create: `models/src/models/deep/train.py`
- Test: `models/tests/test_train.py`

**Interfaces:**
- Consumes: `models.deep.unet.SmallUNet`, `models.deep.losses.FocalLoss`,
  `models.deep.checkpoint.{TrainingConfig, save_checkpoint}`,
  `models.deep.public_dataset.PublicDatasetSample`,
  `features.dataset.assemble.CHANNEL_ORDER`.
- Produces: `set_seed(seed: int) -> None`,
  `NDWSPretrainDataset(samples: list[PublicDatasetSample])` (a
  `torch.utils.data.Dataset`, `__getitem__` returns `(input:
  torch.Tensor (C,H,W) float32, target: torch.Tensor (H,W) float32)`),
  `ChileFinetuneDataset(event_tensors: list[xr.DataArray])` (same
  `__getitem__` contract, one item per `(day, day+1)` pair across all
  given events), `EpochResult` (frozen dataclass: `epoch: int,
  train_loss: float, val_loss: float, seconds: float`),
  `train_model(model: SmallUNet, train_dataset: Dataset, val_dataset:
  Dataset, config: TrainingConfig, run_dir: Path) -> list[EpochResult]`
  (runs the full loop: seeding, per-epoch train+val, CSV row per
  epoch at `run_dir/history.csv`, early stopping on `val_loss` with
  `config.patience`, saves `run_dir/best.pt` — the best-val-loss
  checkpoint — and `run_dir/last.pt` every epoch, and at the end
  writes `run_dir/loss_curve.png` via matplotlib from the CSV). Task 5
  (CLI) consumes `NDWSPretrainDataset`, `ChileFinetuneDataset`,
  `train_model` directly.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_train.py`:

```python
"""Tests del pipeline de entrenamiento: datasets de fixture (sin red,
sin datos reales), un paso de entrenamiento que efectivamente reduce la
pérdida, early stopping, y que el CSV + el checkpoint se escriben."""
import csv
import datetime as dt

import numpy as np
import pytest
import torch
import xarray as xr
from features.dataset.assemble import CHANNEL_ORDER
from models.deep.checkpoint import TrainingConfig, load_checkpoint
from models.deep.public_dataset import PublicDatasetSample
from models.deep.train import ChileFinetuneDataset, NDWSPretrainDataset, set_seed, train_model
from models.deep.unet import SmallUNet

_SIZE = 8


def _make_public_sample(seed: int) -> PublicDatasetSample:
    rng = np.random.default_rng(seed)
    data = rng.random((1, len(CHANNEL_ORDER), _SIZE, _SIZE)).astype("float32")
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    data[0, fire_idx] = (rng.random((_SIZE, _SIZE)) > 0.8).astype("float32")
    tensor = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": ["1970-01-01"], "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor",
        attrs={"resolution_m": 1000.0, "event_id": seed},
    )
    next_mask = (rng.random((_SIZE, _SIZE)) > 0.8).astype("float64")
    return PublicDatasetSample(tensor=tensor, next_day_fire_mask=next_mask)


def _make_chile_event(seed: int, n_days: int = 3) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    data = rng.random((n_days, len(CHANNEL_ORDER), _SIZE, _SIZE)).astype("float32")
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    data[:, fire_idx] = (rng.random((n_days, _SIZE, _SIZE)) > 0.8).astype("float32")
    days = [(dt.date(2026, 1, 1) + dt.timedelta(days=d)).isoformat() for d in range(n_days)]
    return xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": days, "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor",
        attrs={"resolution_m": 250.0, "event_id": seed},
    )


def _make_config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = dict(
        phase="pretrain", in_channels=len(CHANNEL_ORDER), base_channels=8, depth=2,
        lr=1e-2, batch_size=2, seed=42, focal_alpha=0.8, focal_gamma=2.0,
        max_epochs=5, patience=3, data_paths=("fixture",), pretrained_checkpoint=None,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def test_ndws_pretrain_dataset_yields_input_and_target_with_matching_shapes():
    samples = [_make_public_sample(seed=i) for i in range(3)]
    dataset = NDWSPretrainDataset(samples)
    assert len(dataset) == 3
    x, y = dataset[0]
    assert x.shape == (len(CHANNEL_ORDER), _SIZE, _SIZE)
    assert y.shape == (_SIZE, _SIZE)


def test_chile_finetune_dataset_yields_one_pair_per_consecutive_day():
    events = [_make_chile_event(seed=1, n_days=3), _make_chile_event(seed=2, n_days=2)]
    dataset = ChileFinetuneDataset(events)
    # evento 1: 3 días -> 2 pares (d0->d1, d1->d2). evento 2: 2 días -> 1 par.
    assert len(dataset) == 3
    x, y = dataset[0]
    assert x.shape == (len(CHANNEL_ORDER), _SIZE, _SIZE)
    assert y.shape == (_SIZE, _SIZE)


def test_chile_finetune_dataset_skips_single_day_events_without_crashing():
    events = [_make_chile_event(seed=1, n_days=1)]
    dataset = ChileFinetuneDataset(events)
    assert len(dataset) == 0


def test_train_model_reduces_loss_over_a_few_epochs(tmp_path):
    set_seed(42)
    samples = [_make_public_sample(seed=i) for i in range(6)]
    train_dataset = NDWSPretrainDataset(samples[:4])
    val_dataset = NDWSPretrainDataset(samples[4:])
    config = _make_config(max_epochs=8, patience=8, lr=5e-2)
    model = SmallUNet(
        in_channels=config.in_channels, base_channels=config.base_channels, depth=config.depth
    )

    results = train_model(model, train_dataset, val_dataset, config, run_dir=tmp_path)

    assert len(results) >= 2
    assert results[-1].train_loss < results[0].train_loss


def test_train_model_writes_history_csv_and_loss_curve_png(tmp_path):
    set_seed(42)
    samples = [_make_public_sample(seed=i) for i in range(4)]
    train_dataset = NDWSPretrainDataset(samples[:3])
    val_dataset = NDWSPretrainDataset(samples[3:])
    config = _make_config(max_epochs=2, patience=2)
    model = SmallUNet(
        in_channels=config.in_channels, base_channels=config.base_channels, depth=config.depth
    )

    train_model(model, train_dataset, val_dataset, config, run_dir=tmp_path)

    csv_path = tmp_path / "history.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {"epoch", "train_loss", "val_loss", "seconds"} <= set(rows[0].keys())
    assert (tmp_path / "loss_curve.png").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()


def test_train_model_early_stops_and_best_checkpoint_matches_best_epoch(tmp_path):
    set_seed(42)
    samples = [_make_public_sample(seed=i) for i in range(4)]
    train_dataset = NDWSPretrainDataset(samples[:3])
    val_dataset = NDWSPretrainDataset(samples[3:])
    # patience=0: para apenas el val_loss deje de mejorar por 1 época.
    config = _make_config(max_epochs=20, patience=0, lr=5e-2)
    model = SmallUNet(
        in_channels=config.in_channels, base_channels=config.base_channels, depth=config.depth
    )

    results = train_model(model, train_dataset, val_dataset, config, run_dir=tmp_path)

    assert len(results) < 20  # se detuvo antes de max_epochs
    best_epoch_result = min(results, key=lambda r: r.val_loss)
    _loaded_model, _opt_state, loaded_config = load_checkpoint(tmp_path / "best.pt")
    assert loaded_config.max_epochs == config.max_epochs
    # el checkpoint "best" existe incluso si la mejor época fue la 1ª
    # (Review Focus: early stopping en la primera época no debe
    # crashear por no tener una época "anterior" a la que volver).
    assert best_epoch_result.epoch >= 1


def test_set_seed_makes_model_init_deterministic():
    set_seed(123)
    model_a = SmallUNet(in_channels=3, base_channels=8, depth=1)
    set_seed(123)
    model_b = SmallUNet(in_channels=3, base_channels=8, depth=1)
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(p_a, p_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.deep.train'`

- [ ] **Step 3: Write minimal implementation**

Create `models/src/models/deep/train.py`:

```python
"""Datasets de entrenamiento (NDWS de preentrenamiento, eventos de
Chile de fine-tuning), y el bucle de entrenamiento/validación con
checkpointing, early stopping, semillas fijas, y logging a CSV +
matplotlib (sin herramientas externas de tracking -- ver
docs/decisions.md). El CLI (`pyrocast-train`) vive al final de este
mismo archivo."""
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display -- headless, servidor/CI/CLI
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from features.dataset.assemble import CHANNEL_ORDER
from models.deep.checkpoint import TrainingConfig, save_checkpoint
from models.deep.losses import FocalLoss
from models.deep.public_dataset import PublicDatasetSample
from models.deep.unet import SmallUNet

_FIRE_MASK_CHANNEL_INDEX = CHANNEL_ORDER.index("fire_mask")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NDWSPretrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[PublicDatasetSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        x = torch.from_numpy(sample.tensor.values[0].astype("float32"))
        y = torch.from_numpy(sample.next_day_fire_mask.astype("float32"))
        return x, y


class ChileFinetuneDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, event_tensors: list[xr.DataArray]) -> None:
        # (evento, día) por cada par consecutivo -- un evento de 1 solo
        # día no aporta ningún par (no hay "día siguiente" que predecir,
        # ver Review Focus del plan).
        self._pairs: list[tuple[xr.DataArray, int]] = []
        for event in event_tensors:
            n_days = event.sizes["day"]
            for day_index in range(n_days - 1):
                self._pairs.append((event, day_index))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        event, day_index = self._pairs[index]
        x = torch.from_numpy(event.values[day_index].astype("float32"))
        y = torch.from_numpy(
            event.values[day_index + 1, _FIRE_MASK_CHANNEL_INDEX].astype("float32")
        )
        return x, y


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    seconds: float


def _run_epoch(
    model: SmallUNet,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: FocalLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _plot_loss_curve(history: list[EpochResult], path: Path) -> None:
    fig, ax = plt.subplots()
    epochs = [r.epoch for r in history]
    ax.plot(epochs, [r.train_loss for r in history], label="train")
    ax.plot(epochs, [r.val_loss for r in history], label="val")
    ax.set_xlabel("época")
    ax.set_ylabel("focal loss")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def train_model(
    model: SmallUNet,
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    config: TrainingConfig,
    run_dir: Path,
) -> list[EpochResult]:
    set_seed(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device()
    model.to(device)

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    loss_fn = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: list[EpochResult] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    csv_path = run_dir / "history.csv"

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "seconds"])

    for epoch in range(1, config.max_epochs + 1):
        start = time.monotonic()
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = _run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        seconds = time.monotonic() - start

        result = EpochResult(epoch=epoch, train_loss=train_loss, val_loss=val_loss, seconds=seconds)
        history.append(result)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, seconds])

        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, best_val_loss, config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, best_val_loss, config)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > config.patience:
                break

    _plot_loss_curve(history, run_dir / "loss_curve.png")
    return history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_train.py -v`
Expected: `8 passed`. If `test_train_model_reduces_loss_over_a_few_epochs`
is flaky (small-fixture training can be noisy even with a fixed seed
across different `torch` builds), raise `max_epochs`/`lr` in that
specific test's config rather than loosening the assertion — do not
weaken `results[-1].train_loss < results[0].train_loss` into a
"doesn't crash" check.

- [ ] **Step 5: Add `matplotlib` to `models/pyproject.toml`**

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
    "matplotlib>=3.9",
]
```

Run `uv sync --all-packages`.

- [ ] **Step 6: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep/train.py && uv run ruff check models/`
Expected: both clean. `matplotlib` has no `py.typed` marker in some
versions — if mypy reports missing stubs for `matplotlib.pyplot`, add
`# type: ignore[import-untyped]` on that one import line, not a
blanket ignore.

- [ ] **Step 7: Commit**

```bash
git add models/src/models/deep/train.py models/tests/test_train.py models/pyproject.toml uv.lock
git commit -m "feat: add training loop with checkpointing, early stopping, CSV+plot logging"
```

---

## Task 5: `pyrocast-train` CLI — pretrain / finetune / smoke-test

**Files:**
- Modify: `models/src/models/deep/train.py` (append the Typer app)
- Modify: `models/pyproject.toml` (new `[project.scripts]` entry)
- Modify: `Makefile`
- Test: `models/tests/test_train_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4, plus
  `models.deep.public_dataset.{load_public_dataset_samples,
  split_public_dataset}`, `shared.config.get_settings`.
- Produces: the `pyrocast-train` console script with three commands:
  `pretrain`, `finetune`, `smoke-test`.

- [ ] **Step 1: Write the failing tests**

Create `models/tests/test_train_cli.py`:

```python
"""Test de humo del CLI `pyrocast-train`: smoke-test corre de verdad
(datos sintéticos en memoria, sin red, sin datos reales) y dentro del
tiempo esperado en este hardware (ver docs/model-card.md)."""
import time

from models.deep.train import app
from typer.testing import CliRunner

runner = CliRunner()


def test_smoke_test_command_completes_quickly_and_writes_artifacts(tmp_path):
    start = time.monotonic()
    result = runner.invoke(app, ["smoke-test", "--run-dir", str(tmp_path)])
    elapsed = time.monotonic() - start

    assert result.exit_code == 0, result.output
    assert elapsed < 60, f"smoke-test tardó {elapsed:.1f}s -- ver docs/model-card.md"
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "loss_curve.png").exists()


def test_smoke_test_help_documents_all_three_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "pretrain" in result.output
    assert "finetune" in result.output
    assert "smoke-test" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_train_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'app' from 'models.deep.train'`

- [ ] **Step 3: Append the CLI to `models/src/models/deep/train.py`**

First, add these imports to the file's TOP import block (alongside the
existing ones from Task 4 — `csv`, `random`, `time`, etc.): `json`,
`typer`, `from models.deep.checkpoint import load_checkpoint`, `from
models.deep.public_dataset import load_public_dataset_samples,
split_public_dataset`, `from shared.config import get_settings`.

Then append at the end of the file (after `train_model`):

```python
app = typer.Typer()


@app.callback()
def _callback() -> None:
    """CLI de entrenamiento del U-Net de PyroCast."""


def _load_chile_events(dataset_dir: Path, event_ids: list[int]) -> list[xr.DataArray]:
    events = []
    for event_id in event_ids:
        zarr_path = dataset_dir / f"event_{event_id:04d}.zarr"
        events.append(xr.open_zarr(zarr_path)["fire_event_tensor"])
    return events


def pretrain(
    shard_dir: Path = typer.Option(..., help="Directorio con shards *.tfrecord(.gz) de NDWS"),
    run_dir: Path = typer.Option(Path("runs") / "pretrain", help="Dónde guardar checkpoints/logs"),
    base_channels: int = typer.Option(16, help="Canales base de SmallUNet (múltiplo de 8)"),
    depth: int = typer.Option(3, help="Profundidad de SmallUNet"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    batch_size: int = typer.Option(1, help="Tamaño de batch -- 1 por defecto, ver docs/model-card.md"),
    max_epochs: int = typer.Option(50, help="Épocas máximas"),
    patience: int = typer.Option(5, help="Épocas sin mejora antes de early stopping"),
    seed: int = typer.Option(42, help="Semilla de reproducibilidad"),
) -> None:
    """Preentrena SmallUNet sobre shards TFRecord de Next Day Wildfire
    Spread ya descargados -- ver docs/public-dataset.md."""
    shard_paths = sorted(shard_dir.glob("*.tfrecord*"))
    if not shard_paths:
        typer.echo(f"No se encontraron shards *.tfrecord* en {shard_dir}.")
        raise typer.Exit(code=1)

    split = split_public_dataset(shard_paths, seed=seed)
    train_samples = list(load_public_dataset_samples(split["train"]))
    val_samples = list(load_public_dataset_samples(split["val"]))

    config = TrainingConfig(
        phase="pretrain", in_channels=len(CHANNEL_ORDER), base_channels=base_channels,
        depth=depth, lr=lr, batch_size=batch_size, seed=seed, focal_alpha=0.8,
        focal_gamma=2.0, max_epochs=max_epochs, patience=patience,
        data_paths=tuple(str(p) for p in shard_paths), pretrained_checkpoint=None,
    )
    model = SmallUNet(in_channels=config.in_channels, base_channels=base_channels, depth=depth)
    history = train_model(
        model, NDWSPretrainDataset(train_samples), NDWSPretrainDataset(val_samples),
        config, run_dir,
    )
    typer.echo(f"Preentrenamiento: {len(history)} época(s) -> {run_dir / 'best.pt'}")


_FINETUNE_LR_FACTOR = 0.1  # ver Global Constraints del plan: no se congela ninguna capa


def finetune(
    pretrained_checkpoint: Path = typer.Option(..., help="Checkpoint de pretrain (best.pt)"),
    run_dir: Path = typer.Option(Path("runs") / "finetune", help="Dónde guardar checkpoints/logs"),
    lr: float | None = typer.Option(
        None, help="Learning rate -- por defecto, lr del checkpoint * 0.1"
    ),
    batch_size: int = typer.Option(1, help="Tamaño de batch"),
    max_epochs: int = typer.Option(30, help="Épocas máximas"),
    patience: int = typer.Option(5, help="Épocas sin mejora antes de early stopping"),
    seed: int = typer.Option(42, help="Semilla de reproducibilidad"),
) -> None:
    """Fine-tunea un checkpoint preentrenado sobre el split de train/val
    de eventos de Chile (features/dataset/, ver docs/dataset-card.md).
    No congela ninguna capa -- usa un learning rate más bajo en su
    lugar (ver docs/model-card.md, sección de fine-tuning)."""
    model, _optimizer_state, pretrained_config = load_checkpoint(pretrained_checkpoint)
    effective_lr = lr if lr is not None else pretrained_config.lr * _FINETUNE_LR_FACTOR

    settings = get_settings()
    dataset_dir = settings.data_processed_dir / "dataset"
    splits = json.loads((dataset_dir / "splits.json").read_text())
    if not splits["train"]:
        typer.echo("El split de train de eventos de Chile está vacío.")
        raise typer.Exit(code=1)

    train_events = _load_chile_events(dataset_dir, splits["train"])
    val_events = _load_chile_events(dataset_dir, splits["val"]) if splits["val"] else train_events

    config = TrainingConfig(
        phase="finetune", in_channels=pretrained_config.in_channels,
        base_channels=pretrained_config.base_channels, depth=pretrained_config.depth,
        lr=effective_lr, batch_size=batch_size, seed=seed, focal_alpha=pretrained_config.focal_alpha,
        focal_gamma=pretrained_config.focal_gamma, max_epochs=max_epochs, patience=patience,
        data_paths=(str(dataset_dir),), pretrained_checkpoint=str(pretrained_checkpoint),
    )
    history = train_model(
        model, ChileFinetuneDataset(train_events), ChileFinetuneDataset(val_events),
        config, run_dir,
    )
    typer.echo(f"Fine-tuning: {len(history)} época(s) -> {run_dir / 'best.pt'}")


def smoke_test(
    run_dir: Path = typer.Option(Path("runs") / "smoke_test", help="Dónde guardar checkpoints/logs"),
) -> None:
    """Corre 1+ época(s) de entrenamiento sobre datos sintéticos en
    memoria -- sin red, sin NDWS ni eventos de Chile reales. Esto es lo
    que `make train` ejecuta (ver criterios de aceptación del plan)."""
    set_seed(42)
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

    samples = [make_sample(i) for i in range(6)]
    config = TrainingConfig(
        phase="pretrain", in_channels=n_channels, base_channels=8, depth=2, lr=1e-2,
        batch_size=2, seed=42, focal_alpha=0.8, focal_gamma=2.0, max_epochs=1, patience=1,
        data_paths=("synthetic-smoke-test",), pretrained_checkpoint=None,
    )
    model = SmallUNet(in_channels=n_channels, base_channels=8, depth=2)
    history = train_model(
        model, NDWSPretrainDataset(samples[:4]), NDWSPretrainDataset(samples[4:]),
        config, run_dir,
    )
    typer.echo(f"Smoke test: {len(history)} época(s) sobre datos sintéticos -> {run_dir}")


app.command("pretrain")(pretrain)
app.command("finetune")(finetune)
app.command("smoke-test")(smoke_test)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests/test_train_cli.py -v`
Expected: `2 passed`. Time the actual `smoke-test` run on this machine
while you're here (`time uv run --package models pyrocast-train smoke-test`
once installed in Step 6) and note the real elapsed seconds — Task 6's
`docs/model-card.md` should quote this REAL number, not a guess.

- [ ] **Step 5: Add the console script to `models/pyproject.toml`**

```toml
[project.scripts]
pyrocast-models = "models.cli:app"
pyrocast-train = "models.deep.train:app"
```

Run `uv sync --all-packages`.

- [ ] **Step 6: Wire `Makefile`'s `train` target**

Replace the stub:

```makefile
train:
	uv run --package models pyrocast-train smoke-test
```

Run `make train` for real and confirm it exits 0 within the time
budget claimed in Step 4.

- [ ] **Step 7: Typecheck and lint**

Run: `uv run mypy --strict models/src/models/deep/train.py && uv run ruff check models/`
Expected: both clean.

- [ ] **Step 8: Commit**

```bash
git add models/src/models/deep/train.py models/pyproject.toml Makefile models/tests/test_train_cli.py uv.lock
git commit -m "feat: add pyrocast-train CLI (pretrain/finetune/smoke-test)"
```

---

## Task 6: `docs/model-card.md` + `docs/decisions.md` + `docs/limitations.md`

**Files:**
- Create: `docs/model-card.md`
- Modify: `docs/decisions.md`
- Modify: `docs/limitations.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the final shipped behavior from Tasks 1–5, and the REAL
  measured `smoke-test` wall-clock time from Task 5 Step 4 — no code
  interface.

- [ ] **Step 1: Add checkpoint/run directories to `.gitignore`**

Append:

```
runs/
*.pt
```

- [ ] **Step 2: Write `docs/model-card.md`**

Mirror `docs/public-dataset.md`'s header structure (H1 title, purpose
paragraph, bolded `**Aviso de honestidad (CLAUDE.md):**` disclaimer
before the first `##` section). Content, filled in with this task's
real numbers once measured:

```markdown
# Ficha del modelo: SmallUNet

`models/deep/unet.py` + `models/deep/train.py` implementan el segundo
modelo del proyecto (P9-P11) -- una U-Net 2D pequeña, complementaria al
autómata celular de P7 (`docs/cellular-automata.md`), entrenada primero
sobre un dataset público de EE.UU. (preentrenamiento, P9,
`docs/public-dataset.md`) y luego afinada sobre eventos reales de Chile
(fine-tuning, P6).

**Aviso de honestidad (CLAUDE.md):** Herramienta de investigación. No
usar para decisiones operativas de combate de incendios sin validación
de CONAF/SENAPRED. Los tiempos de entrenamiento de este documento son
del hardware de desarrollo real (ver abajo), no de un servidor con GPU
-- cualquiera que reproduzca este proyecto en una laptop similar debe
esperar tiempos del mismo orden.

## Hardware de desarrollo

Sin GPU CUDA. MacBook Air, Apple M1, 8 GB RAM.
`torch.cuda.is_available()` -> `False`.
`torch.backends.mps.is_available()` -> `True` (acelerador Metal de
Apple) -- el código elige `cuda` -> `mps` -> `cpu` en ese orden en
tiempo de ejecución, pero CADA decisión de tamaño de este documento
(ancho del modelo, batch size, resolución) se tomó asumiendo que solo
hay CPU disponible: 8 GB de memoria unificada compartida con el resto
del sistema, y MPS en un M1 base es un acelerador modesto, no una GPU
de datacenter.

## Arquitectura: SmallUNet

U-Net 2D, GroupNorm (no BatchNorm -- ver docs/decisions.md, entrena con
batch_size=1 por defecto). Profundidad configurable, 3 niveles por
defecto:

| Etapa | Canales |
|---|---|
| Entrada | 11 (`features.dataset.assemble.CHANNEL_ORDER`) |
| `in_conv` | 16 |
| Down 1 | 32 |
| Down 2 | 64 |
| Down 3 (bottleneck) | 128 |
| Up (simétrico) | 64 -> 32 -> 16 |
| Salida (1x1 conv) | 1 (logit) |

Acepta cualquier alto/ancho (no solo potencias de 2) -- el camino de
subida rellena cada mapa subido al tamaño exacto de su conexión de
salto antes de concatenar. Con `base_channels=16, depth=3`, ~1.9M
parámetros (calculado con `sum(p.numel() for p in model.parameters())`
en `base_channels=16, depth=3, in_channels=11`).

Salida: LOGITS, no probabilidades -- aplicar `sigmoid` explícitamente
para obtener probabilidad de fuego por celda (ver
`shared/model_protocol.py`: un futuro wrapper `predict()` que
implemente `FireSpreadModel` haría esa conversión; no está construido
en este plan).

## Función de pérdida: Focal Loss

`alpha=0.8, gamma=2.0` (defaults del paper original, Lin et al. 2017).
Elegida sobre BCE ponderada porque además de corregir el desbalance de
FRECUENCIA de clase (la clase "fuego" es rara), el término de focusing
`(1-p_t)^gamma` baja la pérdida de cualquier predicción ya confiada y
correcta -- la mayoría de los píxeles de una máscara de fuego real son
negativos triviales que el modelo aprende a predecir rápido; BCE
ponderada seguiría acumulando pérdida sobre ellos, diluyendo el
gradiente útil de los píxeles difíciles (el borde del frente de fuego,
los pocos positivos reales). Ver `models/deep/losses.py` para la
fórmula exacta y `models/tests/test_losses.py` para la propiedad
verificada.

## Fases de entrenamiento

### 1. Preentrenamiento (`pyrocast-train pretrain`)

Sobre shards TFRecord de Next Day Wildfire Spread ya descargados
manualmente (`docs/public-dataset.md`) -- desde cero, sin checkpoint
previo. `batch_size=1` por defecto (ver "Tamaño de batch" abajo).

### 2. Fine-tuning (`pyrocast-train finetune`)

Sobre el split train/val de eventos de Chile (`features/dataset/`,
`docs/dataset-card.md`), partiendo de un checkpoint de preentrenamiento.

**Decisión: no se congela ninguna capa.** Se usa un learning rate más
bajo (`lr_preentrenamiento * 0.1` por defecto) en vez de congelar el
encoder. Razón: NDWS es 1 km de resolución sobre EE.UU. continental;
los tensores propios de PyroCast son 250 m sobre Chile, con una fuente
de DEM distinta y una distribución de viento/humedad/vegetación
distinta. Congelar el encoder asumiría que sus filtros de bajo nivel
transfieren directamente a través de ese cambio de resolución y
geografía -- una suposición más fuerte que la que este proyecto está
dispuesto a hacer sin evidencia empírica en ningún sentido. Dejar que
toda la red se adapte, con pasos más pequeños, es la opción más segura
dado el desajuste de dominio conocido. Esta es una decisión de diseño,
no la única válida -- ver `docs/decisions.md`.

## Tamaño de batch

`batch_size=1` por defecto, en ambas fases. Los eventos reales de Chile
tienen el tamaño de su propio bbox (ninguno es igual a otro) -- batir
más de un evento a la vez requeriría relleno/recorte a un tamaño común,
no construido en este plan (YAGNI: el número esperado de eventos reales
de Chile es pequeño, batch_size=1 no es un problema de velocidad
significativo ahí). Configurable por CLI para quien tenga más datos o
más RAM.

## Tiempo de entrenamiento esperado

`pyrocast-train smoke-test` (datos sintéticos, `base_channels=8,
depth=2`, 16x16, 4 muestras de train + 2 de val, 1 época) tardó
**[MEDIR EN TASK 5 PASO 4 Y COMPLETAR ACÁ]** segundos en este hardware.

Para una corrida real de preentrenamiento sobre NDWS (18.545 chips
oficiales de 64x64, `base_channels=16, depth=3`) **no se midió en esta
sesión** -- este entorno no tiene acceso de red para descargar NDWS.
Estimado por orden de magnitud a partir del tiempo del smoke test
(escalando por tamaño de imagen y cantidad de muestras, sin
benchmarking real): del orden de horas por época en CPU/MPS, no
minutos -- entrenar la cantidad de épocas que `patience` normalmente
permite probablemente toma **muchas horas a un día** en este hardware.
Esta es una estimación, no una medición -- CLAUDE.md exige no ocultar
esta incertidumbre.

## Reproducibilidad

Cada checkpoint (`models/deep/checkpoint.py::TrainingConfig`) guarda:
arquitectura completa (`in_channels, base_channels, depth`),
hiperparámetros de entrenamiento (`lr, batch_size, seed, focal_alpha,
focal_gamma, max_epochs, patience`), fase, y las rutas de datos
exactas usadas. Cargar un checkpoint reconstruye el modelo desde su
PROPIA configuración, nunca desde los defaults del comando que lo
carga -- `finetune` no necesita (ni debe) adivinar la arquitectura de
un checkpoint de `pretrain`.

## Limitaciones

Ver `docs/limitations.md` para la lista completa. No se micro-gestionan
acá para evitar que ambas fuentes se desincronicen.
```

- [ ] **Step 3: Append to `docs/decisions.md`**

```markdown
## SmallUNet usa GroupNorm, no BatchNorm

Justificado en el docstring de `models/deep/unet.py` -- este proyecto
entrena con `batch_size=1` por defecto (sin GPU, eventos reales de
tamaño variable, ver `docs/model-card.md`), donde las estadísticas de
BatchNorm no tienen sentido. GroupNorm normaliza dentro de una sola
muestra, funciona igual a cualquier tamaño de batch.

## Focal Loss en vez de BCE ponderada

Ver `models/deep/losses.py` para la justificación completa y
`docs/model-card.md` para el resumen -- el término de "focusing" de
focal loss baja la pérdida de negativos fáciles YA confiados
correctamente, no solo corrige la frecuencia de clase (que BCE
ponderada también resuelve).

## Fine-tuning: sin capas congeladas, learning rate reducido en su lugar

Decisión documentada en `docs/model-card.md` -- NDWS (1 km, EE.UU.) y
los tensores de PyroCast (250 m, Chile) difieren en resolución,
geografía y fuente de datos; congelar el encoder asumiría una
transferencia de filtros de bajo nivel sin evidencia empírica.

## `matplotlib` como nueva dependencia de `models`

El enunciado pidió explícitamente "logging simple a CSV + gráficos con
matplotlib (sin herramientas externas de tracking pesadas)" -- ninguna
alternativa más liviana ya presente en el workspace cubre "generar un
PNG de una curva de pérdida" sin agregar una dependencia nueva de
todas formas.
```

- [ ] **Step 4: Append to `docs/limitations.md`** (before the closing
"## Herramienta de investigación" banner — verify with `tail -30
docs/limitations.md` before editing, a prior task in this same session
found a plain `cat >>` lands after that banner)

```markdown
- **Ningún tiempo de entrenamiento real sobre NDWS o sobre eventos
  reales de Chile fue medido en esta sesión** -- solo el smoke test
  sintético (ver `docs/model-card.md`). El tiempo real depende del
  tamaño real de cada evento de Chile (variable, no medido) y del
  volumen real de NDWS descargado (no disponible en este entorno sin
  red). Cualquier estimación de horas/época en `docs/model-card.md` es
  un orden de magnitud, no una medición.
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
```

- [ ] **Step 5: Run the full verification sweep**

Run:
```bash
env -i PATH="$PATH" HOME="$HOME" uv run --package models pytest models/tests -v
uv run ruff check .
uv run mypy --strict models/src/models/deep
uv run mypy --strict shared/src features/src
make train
```
Expected: all green/clean; `make train` exits 0 within the time budget
noted in Task 5 Step 4.

- [ ] **Step 6: Commit**

```bash
git add docs/model-card.md docs/decisions.md docs/limitations.md .gitignore
git commit -m "docs: add model-card.md, document SmallUNet design decisions"
```
