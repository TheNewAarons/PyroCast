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
