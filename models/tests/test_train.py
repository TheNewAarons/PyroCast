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


def _make_sentinel_event(event_id: int, n_days: int) -> xr.DataArray:
    # cada (día, canal) tiene un valor único y reconocible -- permite
    # verificar que ChileFinetuneDataset mapea (evento, día) al par
    # EXACTO (día -> fire_mask del día+1), no solo a algo de la forma
    # correcta (hallazgo de la revisión final del 2026-09-29: los tests
    # anteriores solo afirmaban formas, nunca contenido).
    data = np.zeros((n_days, len(CHANNEL_ORDER), _SIZE, _SIZE), dtype="float32")
    for day in range(n_days):
        for channel in range(len(CHANNEL_ORDER)):
            data[day, channel] = event_id * 1000 + day * 10 + channel
    days = [(dt.date(2026, 1, 1) + dt.timedelta(days=d)).isoformat() for d in range(n_days)]
    return xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": days, "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor", attrs={"resolution_m": 250.0, "event_id": event_id},
    )


def test_chile_finetune_dataset_maps_each_pair_to_the_exact_day_and_next_day_fire_mask():
    events = [
        _make_sentinel_event(event_id=1, n_days=4),
        _make_sentinel_event(event_id=2, n_days=2),
    ]
    dataset = ChileFinetuneDataset(events, cumulative=False)  # mapeo crudo de valores
    assert len(dataset) == 4  # evento 1: 3 pares (d0-1,d1-2,d2-3); evento 2: 1 par (d0-1)

    fire_idx = CHANNEL_ORDER.index("fire_mask")

    # par 0: evento 1, día 0 -> día 1
    x, y = dataset[0]
    assert x[0, 0, 0].item() == 1000  # evento 1, día 0, canal 0
    assert x[fire_idx, 0, 0].item() == 1000 + fire_idx  # evento 1, día 0, canal fire_mask
    assert y[0, 0].item() == 1000 + 10 + fire_idx  # evento 1, día 1, fire_mask

    # par 2: evento 1, día 2 -> día 3 (el último par del evento 1;
    # el evento 1 con n_days=4 aporta 3 pares, índices 0,1,2)
    x, y = dataset[2]
    assert x[0, 0, 0].item() == 1000 + 20
    assert y[0, 0].item() == 1000 + 30 + fire_idx

    # índice 3: primer (y único) par del evento 2, INMEDIATAMENTE
    # después del último par del evento 1 -- confirma que el límite
    # entre eventos no se pierde ni se desplaza.
    x, y = dataset[3]
    assert x[0, 0, 0].item() == 2000  # evento 2, día 0, canal 0
    assert y[0, 0].item() == 2000 + 10 + fire_idx  # evento 2, día 1, fire_mask


def test_chile_finetune_dataset_replaces_residual_nan_with_zero():
    # eventos reales cerca de bordes de cobertura (WorldCover, Sentinel-2)
    # pueden dejar una fracción minúscula de NaN residual en canales
    # NO climáticos (fuel_type, ndvi) incluso después del relleno con el
    # promedio regional de clima (docs/limitations.md) -- verificado
    # contra los 15 eventos reales de docs/backtest-2026.md (event_12676775:
    # fuel_type NaN en 990/19600 celdas de borde). NaN sin tratar
    # envenena la convolución del U-Net en toda la imagen del batch, no
    # solo en esa celda.
    n_days, size = 2, 4
    data = np.zeros((n_days, len(CHANNEL_ORDER), size, size), dtype="float32")
    fuel_idx = CHANNEL_ORDER.index("fuel_type")
    data[:, fuel_idx, 0, 0] = np.nan
    days = [(dt.date(2026, 1, 1) + dt.timedelta(days=d)).isoformat() for d in range(n_days)]
    event = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": days, "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor", attrs={"resolution_m": 250.0, "event_id": 1},
    )
    dataset = ChileFinetuneDataset([event])
    x, y = dataset[0]
    assert not torch.isnan(x).any()
    assert x[fuel_idx, 0, 0].item() == 0.0
    assert not torch.isnan(y).any()


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
    # el checkpoint "best" DEBE corresponder exactamente a la época con
    # menor val_loss -- no cualquier época válida (hallazgo de la
    # revisión final del 2026-09-29: la versión anterior de este test
    # no afirmaba nada real sobre esto).
    best_checkpoint = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert best_checkpoint["epoch"] == best_epoch_result.epoch
    _loaded_model, _opt_state, loaded_config = load_checkpoint(tmp_path / "best.pt")
    assert loaded_config.max_epochs == config.max_epochs
    # el checkpoint "best" existe incluso si la mejor época fue la 1ª
    # (Review Focus: early stopping en la primera época no debe
    # crashear por no tener una época "anterior" a la que volver).
    assert best_epoch_result.epoch >= 1


def test_train_model_rejects_an_empty_val_dataset(tmp_path):
    # sin esto, un val vacío hacía que val_loss reportara 0.0 (0/1)
    # de forma silenciosa: early stopping paraba al azar y best.pt
    # quedaba en la época 1, sin entrenar -- ver docs/limitations.md
    # y la revisión final del 2026-09-29.
    samples = [_make_public_sample(seed=i) for i in range(3)]
    config = _make_config(max_epochs=3, patience=3)
    model = SmallUNet(
        in_channels=config.in_channels, base_channels=config.base_channels, depth=config.depth
    )
    with pytest.raises(ValueError, match="val"):
        train_model(
            model, NDWSPretrainDataset(samples), NDWSPretrainDataset([]), config, run_dir=tmp_path
        )


def test_train_model_rejects_an_empty_train_dataset(tmp_path):
    samples = [_make_public_sample(seed=i) for i in range(3)]
    config = _make_config(max_epochs=3, patience=3)
    model = SmallUNet(
        in_channels=config.in_channels, base_channels=config.base_channels, depth=config.depth
    )
    with pytest.raises(ValueError, match="train"):
        train_model(
            model, NDWSPretrainDataset([]), NDWSPretrainDataset(samples), config, run_dir=tmp_path
        )


def test_train_model_works_with_the_production_default_batch_size_of_one(tmp_path):
    # batch_size=1 es el default real de ambas fases del CLI (ver
    # docs/model-card.md) -- las fixtures de los demás tests usan
    # batch_size=2, este es el único que cubre el valor por defecto
    # real.
    set_seed(42)
    samples = [_make_public_sample(seed=i) for i in range(3)]
    config = _make_config(max_epochs=2, patience=2, batch_size=1)
    model = SmallUNet(
        in_channels=config.in_channels, base_channels=config.base_channels, depth=config.depth
    )
    results = train_model(
        model, NDWSPretrainDataset(samples[:2]), NDWSPretrainDataset(samples[2:]),
        config, run_dir=tmp_path,
    )
    assert len(results) == 2


def test_set_seed_makes_model_init_deterministic():
    set_seed(123)
    model_a = SmallUNet(in_channels=3, base_channels=8, depth=1)
    set_seed(123)
    model_b = SmallUNet(in_channels=3, base_channels=8, depth=1)
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(p_a, p_b)


def test_load_chile_events_trims_leading_days_without_fire_like_the_evaluation(tmp_path):
    import numpy as np
    import xarray as xr
    from features.dataset.assemble import CHANNEL_ORDER
    from models.deep.train import _load_chile_events

    data = np.zeros((6, len(CHANNEL_ORDER), 4, 4), dtype="float32")
    data[3:, CHANNEL_ORDER.index("fire_mask"), 1, 1] = 1.0  # fuego desde el día 3
    event = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": [f"2026-01-{d + 1:02d}" for d in range(6)],
                "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor", attrs={"resolution_m": 250.0, "event_id": 7},
    )
    event.to_dataset().to_zarr(tmp_path / "event_0007.zarr", mode="w")
    (loaded,) = _load_chile_events(tmp_path, [7])
    assert loaded.sizes["day"] == 3
    assert str(loaded.coords["day"].values[0]) == "2026-01-04"


def test_chile_finetune_dataset_default_uses_cumulative_burned_area_for_input_and_target():
    # fuego ACTIVO que aparece, "desaparece" (hueco de pasada del satélite) y reaparece
    data = np.zeros((4, len(CHANNEL_ORDER), 3, 3), dtype="float32")
    fire = CHANNEL_ORDER.index("fire_mask")
    data[0, fire, 0, 0] = 1.0
    data[2, fire, 1, 1] = 1.0
    event = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": ["2026-01-0%d" % (d + 1) for d in range(4)],
                "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor", attrs={"resolution_m": 250.0, "event_id": 1},
    )
    dataset = ChileFinetuneDataset([event])
    # par día 1 -> 2: el fuego activo del día 1 es todo ceros, pero el área
    # quemada acumulada ya incluye la celda (0, 0) del día 0
    x, y = dataset[1]
    assert float(x[fire, 0, 0]) == 1.0 and float(y[0, 0]) == 1.0 and float(y[1, 1]) == 1.0
    for i in range(len(dataset)):
        xi, yi = dataset[i]
        assert (yi >= xi[fire]).all()  # el objetivo nunca "pierde" fuego
    raw = ChileFinetuneDataset([event], cumulative=False)
    assert float(raw[1][0][fire, 0, 0]) == 0.0  # crudo: el fuego activo sí desaparece
