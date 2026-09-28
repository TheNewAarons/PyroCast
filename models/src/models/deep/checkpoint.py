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
    # weights_only=False: seguro acá porque todo checkpoint que este
    # código carga es uno que él mismo escribió (pretrain/finetune/
    # smoke-test), nunca una descarga de terceros -- TrainingConfig no
    # es un tensor y weights_only=True (default más nuevo de torch) se
    # negaría a deserializarlo.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config: TrainingConfig = checkpoint["config"]

    model = SmallUNet(
        in_channels=config.in_channels, base_channels=config.base_channels, depth=config.depth
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state: dict[str, object] = checkpoint["optimizer_state_dict"]
    return model, optimizer_state, config
