"""Datasets de entrenamiento (NDWS de preentrenamiento, eventos de
Chile de fine-tuning), y el bucle de entrenamiento/validación con
checkpointing, early stopping, semillas fijas, y logging a CSV +
matplotlib (sin herramientas externas de tracking -- ver
docs/decisions.md). El CLI (`pyrocast-train`) vive al final de este
mismo archivo."""
import csv
import itertools
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display -- headless, servidor/CI/CLI
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import xarray as xr
from features.dataset.assemble import CHANNEL_ORDER
from shared.config import get_settings
from torch.utils.data import DataLoader, Dataset

from models.deep.checkpoint import TrainingConfig, load_checkpoint, save_checkpoint
from models.deep.losses import FocalLoss
from models.deep.public_dataset import (
    PublicDatasetSample,
    load_public_dataset_samples,
    split_public_dataset,
)
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
    # sin esto, un dataset vacío hace que _run_epoch divida por
    # max(0,1)=1 y reporte una pérdida de 0.0 -- un número FABRICADO
    # (no medido) que además hace que early stopping y la selección
    # de "best" se comporten como si el modelo hubiera validado
    # perfecto desde la época 1. Encontrado en la revisión final del
    # 2026-09-29. Ver docs/limitations.md.
    if len(train_dataset) == 0:  # type: ignore[arg-type]
        raise ValueError(
            "train_dataset está vacío -- no hay nada con qué entrenar."
        )
    if len(val_dataset) == 0:  # type: ignore[arg-type]
        raise ValueError(
            "val_dataset está vacío -- early stopping y la selección del checkpoint "
            "'best' necesitan un val real, no se puede continuar en silencio."
        )

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
    batch_size: int = typer.Option(
        1, help="Tamaño de batch -- 1 por defecto, ver docs/model-card.md"
    ),
    max_epochs: int = typer.Option(50, help="Épocas máximas"),
    patience: int = typer.Option(5, help="Épocas sin mejora antes de early stopping"),
    seed: int = typer.Option(42, help="Semilla de reproducibilidad"),
    max_samples: int | None = typer.Option(
        None,
        help=(
            "Tope de muestras a cargar en memoria por split (train y val, cada uno "
            "hasta este número) -- cada muestra NDWS de 64x64 pesa ~215 KB una vez "
            "cargada; los 18.545 chips oficiales completos pesan ~4 GB, más el "
            "modelo/optimizador/activaciones. En una máquina de 8 GB sin GPU (ver "
            "docs/model-card.md), cargar el dataset completo sin este tope puede "
            "agotar la memoria. Sin tope por defecto -- el usuario decide."
        ),
    ),
) -> None:
    """Preentrena SmallUNet sobre shards TFRecord de Next Day Wildfire
    Spread ya descargados -- ver docs/public-dataset.md."""
    shard_paths = sorted(shard_dir.glob("*.tfrecord*"))
    if not shard_paths:
        typer.echo(f"No se encontraron shards *.tfrecord* en {shard_dir}.")
        raise typer.Exit(code=1)

    split = split_public_dataset(shard_paths, seed=seed)
    if not split["val"]:
        typer.echo(
            f"El split de val de NDWS quedó vacío (solo {len(shard_paths)} shard(s) -- "
            f"se necesitan al menos 2 para separar train/val). Sin val real no se "
            f"puede hacer early stopping honesto (ver docs/limitations.md)."
        )
        raise typer.Exit(code=1)

    train_samples = list(
        itertools.islice(load_public_dataset_samples(split["train"]), max_samples)
    )
    val_samples = list(itertools.islice(load_public_dataset_samples(split["val"]), max_samples))

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

    if not splits["val"]:
        typer.echo(
            "El split de val de eventos de Chile está vacío -- no se puede hacer "
            "early stopping honesto validando contra el propio train. Agrega más "
            "eventos o corre `pyrocast-features build-dataset` de nuevo (ver "
            "docs/limitations.md)."
        )
        raise typer.Exit(code=1)

    train_events = _load_chile_events(dataset_dir, splits["train"])
    val_events = _load_chile_events(dataset_dir, splits["val"])

    config = TrainingConfig(
        phase="finetune", in_channels=pretrained_config.in_channels,
        base_channels=pretrained_config.base_channels, depth=pretrained_config.depth,
        lr=effective_lr, batch_size=batch_size, seed=seed,
        focal_alpha=pretrained_config.focal_alpha, focal_gamma=pretrained_config.focal_gamma,
        max_epochs=max_epochs, patience=patience,
        data_paths=(str(dataset_dir),), pretrained_checkpoint=str(pretrained_checkpoint),
    )
    history = train_model(
        model, ChileFinetuneDataset(train_events), ChileFinetuneDataset(val_events),
        config, run_dir,
    )
    typer.echo(f"Fine-tuning: {len(history)} época(s) -> {run_dir / 'best.pt'}")


def smoke_test(
    run_dir: Path = typer.Option(
        Path("runs") / "smoke_test", help="Dónde guardar checkpoints/logs"
    ),
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
