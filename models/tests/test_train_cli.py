"""Test de humo del CLI `pyrocast-train`: smoke-test corre de verdad
(datos sintéticos en memoria, sin red, sin datos reales) y dentro del
tiempo esperado en este hardware (ver docs/model-card.md)."""
import json
import time

import numpy as np
import torch
import xarray as xr
from features.dataset.assemble import CHANNEL_ORDER
from models.deep.checkpoint import TrainingConfig, load_checkpoint, save_checkpoint
from models.deep.train import app
from models.deep.unet import SmallUNet
from typer.testing import CliRunner

runner = CliRunner()

REQUIRED_ENV = {
    "FIRMS_MAP_KEY": "x", "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "x", "COPERNICUS_DATASPACE_CLIENT_ID": "id",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "secret", "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432", "POSTGRES_DB": "pyrocast", "POSTGRES_USER": "pyrocast",
    "POSTGRES_PASSWORD": "x",
}


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


def test_pretrain_cli_rejects_an_empty_val_split(tmp_path):
    # un solo shard -> split_public_dataset da val=[] (necesita al
    # menos 2 para separar train/val) -- debe rechazarse ANTES de
    # entrenar con un mensaje claro, no silenciosamente validar contra
    # el propio train (encontrado en la revisión final del 2026-09-29).
    # El contenido del shard es irrelevante: se sale antes de leerlo.
    (tmp_path / "shard_0.tfrecord").write_bytes(b"")
    result = runner.invoke(app, ["pretrain", "--shard-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "val" in result.output.lower()


def _make_chile_event_for_cli(event_id: int, n_days: int = 2, size: int = 8) -> xr.DataArray:
    data = np.zeros((n_days, len(CHANNEL_ORDER), size, size), dtype="float32")
    fire_idx = CHANNEL_ORDER.index("fire_mask")
    data[:, fire_idx] = (np.random.default_rng(event_id).random((n_days, size, size)) > 0.8)
    return xr.DataArray(
        data.astype("float32"), dims=("day", "channel", "y", "x"),
        coords={
            "day": [f"2026-01-{d + 1:02d}" for d in range(n_days)],
            "channel": list(CHANNEL_ORDER),
        },
        name="fire_event_tensor",
        attrs={"resolution_m": 250.0, "event_id": event_id},
    )


def test_finetune_cli_rejects_an_empty_val_split(tmp_path, monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    from shared.config import get_settings

    get_settings.cache_clear()

    checkpoint_path = tmp_path / "pretrained.pt"
    model = SmallUNet(in_channels=len(CHANNEL_ORDER), base_channels=8, depth=1)
    save_checkpoint(
        checkpoint_path, model, torch.optim.Adam(model.parameters(), lr=1e-3), epoch=1,
        best_val_loss=0.1,
        config=TrainingConfig(
            phase="pretrain", in_channels=len(CHANNEL_ORDER), base_channels=8, depth=1,
            lr=1e-3, batch_size=1, seed=42, focal_alpha=0.8, focal_gamma=2.0,
            max_epochs=1, patience=1, data_paths=("fixture",), pretrained_checkpoint=None,
        ),
    )

    dataset_dir = tmp_path / "data" / "processed" / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits.json").write_text(json.dumps({"train": [1], "val": []}))
    _make_chile_event_for_cli(1).to_dataset().to_zarr(dataset_dir / "event_0001.zarr", mode="w")

    result = runner.invoke(app, ["finetune", "--pretrained-checkpoint", str(checkpoint_path)])
    assert result.exit_code == 1
    assert "val" in result.output.lower()
    get_settings.cache_clear()


def test_finetune_cli_inherits_architecture_from_the_pretrained_checkpoint(tmp_path, monkeypatch):
    # Review Focus del plan: finetune debe reconstruir la arquitectura
    # desde la config del PROPIO checkpoint, no desde sus propios
    # defaults de CLI -- se usa base_channels=24 (no-default, el
    # default de finetune/pretrain es 16) para probar que de verdad se
    # hereda, no que coincide por casualidad.
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    from shared.config import get_settings

    get_settings.cache_clear()

    checkpoint_path = tmp_path / "pretrained.pt"
    non_default_base_channels = 24
    model = SmallUNet(
        in_channels=len(CHANNEL_ORDER), base_channels=non_default_base_channels, depth=2
    )
    save_checkpoint(
        checkpoint_path, model, torch.optim.Adam(model.parameters(), lr=1e-2), epoch=1,
        best_val_loss=0.1,
        config=TrainingConfig(
            phase="pretrain", in_channels=len(CHANNEL_ORDER),
            base_channels=non_default_base_channels, depth=2, lr=1e-2, batch_size=1,
            seed=42, focal_alpha=0.8, focal_gamma=2.0, max_epochs=1, patience=1,
            data_paths=("fixture",), pretrained_checkpoint=None,
        ),
    )

    dataset_dir = tmp_path / "data" / "processed" / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits.json").write_text(json.dumps({"train": [1, 2], "val": [3]}))
    for event_id in (1, 2, 3):
        _make_chile_event_for_cli(event_id).to_dataset().to_zarr(
            dataset_dir / f"event_{event_id:04d}.zarr", mode="w"
        )

    run_dir = tmp_path / "finetune_run"
    result = runner.invoke(
        app,
        [
            "finetune", "--pretrained-checkpoint", str(checkpoint_path),
            "--run-dir", str(run_dir), "--max-epochs", "1", "--patience", "1",
        ],
    )
    assert result.exit_code == 0, result.output

    _loaded_model, _opt_state, finetuned_config = load_checkpoint(run_dir / "best.pt")
    assert finetuned_config.base_channels == non_default_base_channels
    assert finetuned_config.depth == 2
    assert finetuned_config.phase == "finetune"
    assert finetuned_config.lr == 1e-2 * 0.1  # ver _FINETUNE_LR_FACTOR
    get_settings.cache_clear()


def test_finetune_cli_trains_from_scratch_without_a_pretrained_checkpoint(tmp_path, monkeypatch):
    # El usuario eligió explícitamente "sin preentrenamiento, solo
    # fine-tuning en Chile" (sin credenciales de Kaggle/NDWS
    # disponibles, ver docs/backtest-2026.md) -- finetune() debía
    # exigir SIEMPRE un checkpoint preentrenado, dejando este camino sin
    # forma de entrenar un U-Net solo con eventos reales de Chile.
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    from shared.config import get_settings

    get_settings.cache_clear()

    dataset_dir = tmp_path / "data" / "processed" / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits.json").write_text(json.dumps({"train": [1, 2], "val": [3]}))
    for event_id in (1, 2, 3):
        _make_chile_event_for_cli(event_id).to_dataset().to_zarr(
            dataset_dir / f"event_{event_id:04d}.zarr", mode="w"
        )

    run_dir = tmp_path / "scratch_run"
    result = runner.invoke(
        app,
        [
            "finetune", "--run-dir", str(run_dir), "--max-epochs", "1", "--patience", "1",
            "--base-channels", "8", "--depth", "1", "--lr", "1e-2",
        ],
    )
    assert result.exit_code == 0, result.output

    _loaded_model, _opt_state, config = load_checkpoint(run_dir / "best.pt")
    assert config.base_channels == 8
    assert config.depth == 1
    assert config.phase == "finetune"
    assert config.pretrained_checkpoint is None
    assert config.lr == 1e-2
    get_settings.cache_clear()
