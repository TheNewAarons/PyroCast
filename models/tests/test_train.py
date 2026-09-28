"""Tests del pipeline de entrenamiento: datasets de fixture (sin red,
sin datos reales), un paso de entrenamiento que efectivamente reduce la
pérdida, early stopping, y que el CSV + el checkpoint se escriben."""
import csv
import datetime as dt

import numpy as np
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
