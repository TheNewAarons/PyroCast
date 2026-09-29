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

import numpy as np
import torch
import xarray as xr
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import Dataset

from models.deep.checkpoint import load_checkpoint
from models.deep.unet import SmallUNet
from models.evaluation.metrics import brier_score, ece_score


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
    path = (
        calibration_path
        if calibration_path is not None
        else default_calibration_path(checkpoint_path)
    )
    save_calibration(path, result)
    return result


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
