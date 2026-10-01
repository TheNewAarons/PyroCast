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


def test_checkpoint_fingerprint_is_stable_across_resaves_of_identical_weights(tmp_path):
    # hallazgo de la revisión final del 2026-09-29: hashear los bytes
    # CRUDOS del archivo (en vez del contenido de los pesos) ataba la
    # huella al nombre del archivo -- torch.save nombra las entradas del
    # zip según el nombre del archivo, así que guardar los MISMOS pesos
    # bajo dos rutas distintas (p. ej. last.pt y best.pt en la misma
    # época) daba huellas DISTINTAS, contradiciendo el docstring del
    # módulo ("ata el calibrador a los PESOS exactos"). El fingerprint
    # ahora hashea el state_dict, no el archivo.
    torch.manual_seed(1)
    model = SmallUNet(in_channels=3, base_channels=8, depth=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = TrainingConfig(
        phase="pretrain", in_channels=3, base_channels=8, depth=1, lr=1e-3,
        batch_size=1, seed=1, focal_alpha=0.8, focal_gamma=2.0, max_epochs=1,
        patience=1, data_paths=("fixture",), pretrained_checkpoint=None,
    )
    path_a = tmp_path / "last.pt"
    path_b = tmp_path / "best.pt"
    save_checkpoint(path_a, model, optimizer, epoch=1, best_val_loss=0.5, config=config)
    save_checkpoint(path_b, model, optimizer, epoch=1, best_val_loss=0.5, config=config)

    assert _checkpoint_fingerprint(path_a) == _checkpoint_fingerprint(path_b)


def test_fit_isotonic_calibrator_improves_ece_on_a_known_miscalibration():
    # relación real CONOCIDA: frecuencia verdadera = raw_prob**2 (un
    # patrón de sobreconfianza clásico -- ver Review Focus del plan).
    # Isotonic regression, al ser monótona, puede corregir exactamente
    # este tipo de relación no lineal pero monótona.
    #
    # AJUSTAR y EVALUAR sobre el MISMO set (como hace `calibrate_checkpoint`
    # a propósito, ver docs/decisions.md) hace que el ECE "después" sea
    # ~0 estructuralmente para CUALQUIER dato -- incluso ruido aleatorio
    # sin relación real con el target, o datos ya bien calibrados
    # (verificado en la revisión final del 2026-09-29) -- así que ese
    # diseño NO sirve para probar que la calibración generaliza. Este
    # test ajusta sobre una mitad y evalúa ECE sobre la OTRA mitad
    # (held-out) para verificar una mejora real y no trivial, y además
    # comprueba que el mapeo aprendido se acerca a la relación real
    # (x**2), no solo que "algún número bajó".
    from models.deep.calibration import fit_isotonic_calibrator
    from models.evaluation.metrics import ece_score

    rng = np.random.default_rng(42)
    raw_probs = rng.uniform(0.05, 0.95, size=8000)
    true_frequency = raw_probs**2
    targets = (rng.uniform(size=8000) < true_frequency).astype("float64")

    fit_probs, eval_probs = raw_probs[:4000], raw_probs[4000:]
    fit_targets, eval_targets = targets[:4000], targets[4000:]

    calibrator = fit_isotonic_calibrator(fit_probs, fit_targets)

    ece_before = ece_score(eval_probs, eval_targets, n_bins=10)
    calibrated_eval_probs = calibrator.predict(eval_probs)
    ece_after = ece_score(calibrated_eval_probs, eval_targets, n_bins=10)
    assert ece_after < ece_before

    # el mapeo aprendido debe acercarse a x**2, la relación real -- no
    # solo bajar el ECE por casualidad.
    check_points = np.array([0.2, 0.5, 0.8])
    learned = calibrator.predict(check_points)
    expected = check_points**2
    assert np.max(np.abs(learned - expected)) < 0.1


def test_fit_isotonic_calibrator_in_sample_ece_is_near_zero_regardless_of_real_relationship():
    # hallazgo IMPORTANTE de la revisión final del 2026-09-29: ajustar
    # y evaluar sobre el MISMO set hace que el ECE "después" sea
    # estructuralmente ~0 -- incluso cuando los targets son RUIDO
    # ALEATORIO sin relación real con raw_probs. Esto no es un defecto
    # de `fit_isotonic_calibrator` (hace exactamente lo que se le pide:
    # ajustar en el mismo set), pero significa que `calibrate_checkpoint`
    # (que por diseño ajusta y evalúa sobre el mismo set de validación,
    # ver docs/decisions.md) SIEMPRE va a reportar un ECE "después"
    # cercano a 0, sin importar qué tan bueno sea el modelo -- por eso
    # docs/calibration.md documenta ese número como una prueba de que
    # el pipeline corre, no como una medición de calibración real.
    from models.deep.calibration import fit_isotonic_calibrator
    from models.evaluation.metrics import ece_score

    rng = np.random.default_rng(7)
    raw_probs = rng.uniform(0.05, 0.95, size=2000)
    targets = rng.integers(0, 2, size=2000).astype("float64")  # sin relación real

    calibrator = fit_isotonic_calibrator(raw_probs, targets)
    calibrated = calibrator.predict(raw_probs)
    ece_after = ece_score(calibrated, targets, n_bins=10)

    assert ece_after < 0.01


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


def test_calibrate_checkpoint_rejects_a_val_sample_with_non_finite_values(tmp_path):
    from models.deep.calibration import calibrate_checkpoint
    from models.deep.train import NDWSPretrainDataset

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    samples = _make_public_samples(seed=10, n=4)
    samples[0].tensor.values[0, 0, 0, 0] = np.nan
    val_dataset = NDWSPretrainDataset(samples)

    with pytest.raises(ValueError, match="finita"):
        calibrate_checkpoint(checkpoint, val_dataset, calibration_path=tmp_path / "cal.pt")


def test_calibrated_unet_satisfies_the_fire_spread_model_protocol(tmp_path):
    from models.deep.calibration import CalibratedUNet
    from shared.model_protocol import FireSpreadModel

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    model = CalibratedUNet(checkpoint)
    assert isinstance(model, FireSpreadModel)


def test_calibrated_unet_predict_seeds_day_zero_from_the_known_fire_mask(tmp_path):
    # mismo convenio que CellularAutomatonModel (models/cellular_automata/model.py):
    # el día 0 es el ancla conocida, no una predicción real.
    from models.deep.calibration import CalibratedUNet

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    model = CalibratedUNet(checkpoint)

    event = _make_public_samples(seed=5, n=1)[0].tensor
    # día 0 real de la muestra sintética -- forzamos un valor conocido.
    fire_idx = list(CHANNEL_ORDER).index("fire_mask")
    event.values[0, fire_idx] = 0.0
    event.values[0, fire_idx, 0, 0] = 1.0

    result = model.predict(event)
    assert result.shape == (1, event.sizes["y"], event.sizes["x"])
    assert result[0, 0, 0] == 1.0
    assert result[0, 1, 1] == 0.0


def test_calibrated_unet_predict_handles_a_single_day_event_without_a_forward_pass(tmp_path):
    # Review Focus del plan: n_days=1 no debe intentar correr el modelo
    # sobre un batch vacío.
    from models.deep.calibration import CalibratedUNet

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    model = CalibratedUNet(checkpoint)
    event = _make_public_samples(seed=6, n=1)[0].tensor  # day dim size 1
    result = model.predict(event)
    assert result.shape == (1, event.sizes["y"], event.sizes["x"])


def test_calibrated_unet_predict_produces_multi_day_output_matching_event_shape(tmp_path):
    from models.deep.calibration import CalibratedUNet

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    model = CalibratedUNet(checkpoint)

    size = 8
    data = np.random.default_rng(7).random((3, len(CHANNEL_ORDER), size, size)).astype("float32")
    event = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={
            "day": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "channel": list(CHANNEL_ORDER),
        },
        name="fire_event_tensor", attrs={"resolution_m": 250.0, "event_id": 1},
    )
    result = model.predict(event)
    assert result.shape == (3, size, size)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_calibrated_unet_uses_the_calibrator_when_one_is_provided(tmp_path):
    from models.deep.calibration import CalibratedUNet, calibrate_checkpoint
    from models.deep.train import NDWSPretrainDataset

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    val_dataset = NDWSPretrainDataset(_make_public_samples(seed=20, n=4))
    calibration_path = tmp_path / "cal.pt"
    calibrate_checkpoint(checkpoint, val_dataset, calibration_path=calibration_path)

    raw_model = CalibratedUNet(checkpoint)
    calibrated_model = CalibratedUNet(checkpoint, calibration_path=calibration_path)

    event = _make_public_samples(seed=30, n=1)[0].tensor
    # forzar > 1 día para tener al menos una celda de predicción real
    event = xr.concat([event, event], dim="day")
    event = event.assign_coords(day=["2026-01-01", "2026-01-02"])

    raw_output = raw_model.predict(event)
    calibrated_output = calibrated_model.predict(event)
    # el calibrador es una transformación monótona no trivial (ver
    # fit sobre datos aleatorios) -- el día 1 (predicho, no ancla) casi
    # seguro difiere entre ambos, salvo coincidencia exacta.
    assert not np.array_equal(raw_output[1], calibrated_output[1])


def test_calibrated_unet_rejects_a_calibrator_from_a_different_checkpoint(tmp_path):
    from models.deep.calibration import CalibratedUNet, calibrate_checkpoint
    from models.deep.train import NDWSPretrainDataset

    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    _make_checkpoint_for_channel_order(checkpoint_a, seed=1)
    _make_checkpoint_for_channel_order(checkpoint_b, seed=2)
    val_dataset = NDWSPretrainDataset(_make_public_samples(seed=40, n=4))
    calibration_for_a = tmp_path / "a.calibrator.pt"
    calibrate_checkpoint(checkpoint_a, val_dataset, calibration_path=calibration_for_a)

    with pytest.raises(IncompatibleCalibratorError, match="checkpoint"):
        CalibratedUNet(checkpoint_b, calibration_path=calibration_for_a)


def test_calibrated_unet_predict_is_autoregressive_not_teacher_forced(tmp_path):
    # hallazgo CRÍTICO de la revisión final del 2026-09-29:
    # CalibratedUNet alimentaba el fire_mask REAL (verdad de terreno)
    # del día d-1 como entrada para predecir el día d -- "teacher
    # forcing" de un solo paso, no una propagación multi-día genuina
    # como CellularAutomatonModel (que evoluciona SU PROPIO estado
    # desde el día 0, sin volver a leer el fire_mask real de días
    # posteriores). Esto hacía que ambos modelos NO fueran
    # intercambiables ante el backtest como el enunciado exige.
    #
    # Este test fuerza un fire_mask REAL del día 1 completamente en
    # cero (irreal si el modelo predijo propagación en el día 1) y
    # verifica que la entrada usada para predecir el día 2 usa la
    # PROPIA predicción del modelo para el día 1 (probablemente no
    # todo cero), NO el fire_mask real del día 1.
    from models.deep.calibration import CalibratedUNet

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=1)
    model = CalibratedUNet(checkpoint)

    size = 8
    n_days = 3
    data = np.random.default_rng(9).random((n_days, len(CHANNEL_ORDER), size, size)).astype(
        "float32"
    )
    fire_idx = list(CHANNEL_ORDER).index("fire_mask")
    data[:, fire_idx] = 0.0
    data[0, fire_idx, 0, 0] = 1.0  # ancla día 0: una celda en llamas
    data[1, fire_idx] = 0.0  # fire_mask REAL del día 1: nada en llamas (irreal)
    event = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={
            "day": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "channel": list(CHANNEL_ORDER),
        },
        name="fire_event_tensor", attrs={"resolution_m": 250.0, "event_id": 1},
    )

    captured_inputs = []
    original_forward = model.model.forward

    def spy_forward(x: torch.Tensor) -> torch.Tensor:
        captured_inputs.append(x.clone())
        result: torch.Tensor = original_forward(x)
        return result

    model.model.forward = spy_forward  # type: ignore[method-assign]
    model.predict(event)

    assert len(captured_inputs) == 2  # una llamada por día predicho (día 1, día 2)
    day_2_input_fire_channel = captured_inputs[1][0, fire_idx].cpu().numpy()
    real_day_1_fire_mask = data[1, fire_idx]
    # la entrada del día 2 NO debe ser el fire_mask REAL del día 1
    # (todo cero) -- debe ser la predicción propia del modelo.
    assert not np.allclose(day_2_input_fire_channel, real_day_1_fire_mask)


def test_calibrated_unet_predict_rejects_non_finite_input():
    # mismo criterio que models/cellular_automata/simulate.py::_require_finite
    # -- features/dataset/resample.py rellena huecos de cobertura con
    # NaN; sin este chequeo, un NaN de entrada se propaga en silencio a
    # la predicción y pasa desapercibido por
    # models/evaluation/backtest.py (las comparaciones con NaN son
    # siempre False).
    from models.deep.calibration import CalibratedUNet

    class _FakeModel:
        def eval(self) -> None:
            pass

        def to(self, device: object) -> None:
            pass

    instance = CalibratedUNet.__new__(CalibratedUNet)
    instance.model = _FakeModel()  # type: ignore[assignment]
    instance.device = torch.device("cpu")
    instance.calibrator = None

    size = 4
    data = np.random.default_rng(1).random((2, len(CHANNEL_ORDER), size, size)).astype("float32")
    # el día 0 es el único que se lee como ENTRADA para predecir el día
    # 1 en un evento de 2 días (`event.values[:-1]`) -- el NaN debe ir
    # ahí para que el chequeo lo detecte.
    data[0, 0, 0, 0] = np.nan
    event = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": ["2026-01-01", "2026-01-02"], "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor", attrs={"resolution_m": 250.0, "event_id": 1},
    )
    with pytest.raises(ValueError, match="finita"):
        instance.predict(event)


def test_calibrated_unet_prediction_is_monotone_non_decreasing_in_time(tmp_path):
    # la probabilidad es ACUMULADA ("¿ha ardido alguna vez?"): nunca baja de un día a otro
    from models.deep.calibration import CalibratedUNet

    checkpoint = tmp_path / "model.pt"
    _make_checkpoint_for_channel_order(checkpoint, seed=3)
    size, n_days = 8, 5
    data = np.random.default_rng(11).random(
        (n_days, len(CHANNEL_ORDER), size, size)).astype("float32")
    event = xr.DataArray(
        data, dims=("day", "channel", "y", "x"),
        coords={"day": [f"2026-01-0{d + 1}" for d in range(n_days)],
                "channel": list(CHANNEL_ORDER)},
        name="fire_event_tensor", attrs={"resolution_m": 250.0, "event_id": 1},
    )
    result = CalibratedUNet(checkpoint).predict(event)
    assert np.all(np.diff(result, axis=0) >= -1e-12)
