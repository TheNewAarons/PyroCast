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
