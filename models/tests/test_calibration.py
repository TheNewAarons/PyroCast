"""Tests de models/deep/calibration.py -- huella de checkpoint,
persistencia del calibrador, y el error claro al aplicar un calibrador
a un checkpoint incompatible."""
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from features.dataset.assemble import CHANNEL_ORDER
from models.deep.calibration import (
    CalibrationResult,
    IncompatibleCalibratorError,
    _checkpoint_fingerprint,
    default_calibration_path,
    load_calibration,
    save_calibration,
)
from models.deep.checkpoint import TrainingConfig, save_checkpoint
from models.deep.public_dataset import PublicDatasetSample
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


def _make_checkpoint_for_channel_order(path: Path, seed: int) -> None:
    # mismo cuerpo que _make_checkpoint, pero in_channels=len(CHANNEL_ORDER)
    # para calzar con muestras construidas por _make_public_samples.
    torch.manual_seed(seed)
    model = SmallUNet(in_channels=len(CHANNEL_ORDER), base_channels=8, depth=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = TrainingConfig(
        phase="pretrain", in_channels=len(CHANNEL_ORDER), base_channels=8, depth=1,
        lr=1e-3, batch_size=1, seed=seed, focal_alpha=0.8, focal_gamma=2.0,
        max_epochs=1, patience=1, data_paths=("fixture",), pretrained_checkpoint=None,
    )
    save_checkpoint(path, model, optimizer, epoch=1, best_val_loss=0.5, config=config)


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
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    val_dataset = NDWSPretrainDataset(_make_public_samples(seed=10, n=4))

    result = calibrate_checkpoint(checkpoint, val_dataset, calibration_path=tmp_path / "cal.pt")

    assert result.checkpoint_fingerprint == _checkpoint_fingerprint(checkpoint)
    assert 0.0 <= result.brier_before <= 1.0
    assert 0.0 <= result.brier_after <= 1.0
    assert 0.0 <= result.ece_before <= 1.0
    assert 0.0 <= result.ece_after <= 1.0
    assert result.n_samples > 0
    assert (tmp_path / "cal.pt").exists()
