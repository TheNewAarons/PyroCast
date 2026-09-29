"""Calibración isotónica de las probabilidades crudas de SmallUNet
contra las frecuencias observadas en el set de validación
(scikit-learn `IsotonicRegression`) -- ver docs/calibration.md para el
contexto completo (enfoque, tabla antes/después) y el docstring de
`CalibratedUNet` para cómo el modelo calibrado implementa
`shared.model_protocol.FireSpreadModel`.

El calibrador se guarda junto a un checkpoint identificado por una
huella (sha256 del `state_dict` del checkpoint, NO de los bytes crudos
del archivo -- torch.save nombra las entradas de su zip según el
nombre del archivo, así que hashear el archivo ataría la huella al
NOMBRE, no a los pesos: guardar los mismos pesos como `last.pt` y
`best.pt` en la misma época daría huellas distintas. Encontrado en la
revisión final del 2026-09-29) -- cargar un calibrador verifica esa
huella contra el checkpoint real antes de usarlo. Nunca se aplica un
calibrador a un checkpoint distinto del que fue entrenado (el
enunciado lo exige explícitamente): el fingerprint ata el calibrador a
los PESOS exactos, no a una ejecución de calibración específica -- dos
calibraciones distintas contra el mismo checkpoint son ambas válidas.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import typer
import xarray as xr
from features.dataset.assemble import CHANNEL_ORDER
from shared.config import get_settings
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import Dataset

from models.deep.checkpoint import TrainingConfig, load_checkpoint
from models.deep.public_dataset import (
    PublicDatasetSample,
    load_public_dataset_samples,
    split_public_dataset,
)
from models.deep.unet import SmallUNet
from models.evaluation.metrics import brier_score, ece_score


class IncompatibleCalibratorError(RuntimeError):
    """El calibrador fue ajustado contra un checkpoint distinto del que
    se intenta usar -- nunca se aplica un calibrador a pesos que no son
    los suyos."""


def _checkpoint_fingerprint(path: Path) -> str:
    # hashea el CONTENIDO del state_dict (nombre de parámetro + bytes
    # del tensor, en orden determinista), no los bytes crudos del
    # archivo -- ver docstring del módulo.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        hasher.update(state_dict[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


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


def _require_finite(name: str, array: np.ndarray) -> None:
    # mismo criterio que models/cellular_automata/simulate.py::_require_finite
    # -- features/dataset/resample.py rellena huecos de cobertura con
    # NaN (nunca fabrica un valor); sin este chequeo, un NaN de entrada
    # se propaga en silencio a la predicción/calibración, y
    # models/evaluation/backtest.py no lo detecta (las comparaciones
    # con NaN son siempre False). Encontrado en la revisión final del
    # 2026-09-29.
    if not np.all(np.isfinite(array)):
        bad = int(np.sum(~np.isfinite(array)))
        raise ValueError(
            f"{name} tiene {bad} celda(s) no finita(s) (NaN/inf) -- no se puede "
            f"continuar con datos de entrada incompletos. Ver docs/limitations.md."
        )


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
            _require_finite(f"muestra {i} de val_dataset", x.numpy())
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
    -1" del que predecir).

    Días `1..n-1`: predicción AUTORREGRESIVA, no "teacher forcing".
    Para predecir el día `d`, se usan los canales REALES del evento en
    el día `d-1` (terreno, clima -- igual que CellularAutomatonModel,
    que también consume viento/clima real por día) EXCEPTO el canal
    `fire_mask`, que se reemplaza por la PROPIA predicción del modelo
    para el día `d-1` (calibrada, si corresponde), no por el fire_mask
    real observado. Sin esto, cada predicción sería un pronóstico "a un
    día" con acceso al fire_mask real del día anterior -- mucho más
    fácil que la propagación multi-día genuina que hace
    CellularAutomatonModel evolucionando su propio estado desde el día
    0, y ambos modelos NO serían intercambiables ante el backtest como
    el enunciado exige. Encontrado en la revisión final del 2026-09-29
    -- ver `test_calibrated_unet_predict_is_autoregressive_not_teacher_forced`."""

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
            _require_finite("event", event.values[:-1])
            current_fire_state = output[0].astype("float32")
            for day in range(1, n_days):
                day_input = event.values[day - 1].copy().astype("float32")
                day_input[fire_idx] = current_fire_state
                x = torch.from_numpy(day_input[np.newaxis]).to(self.device)
                with torch.no_grad():
                    logits = self.model(x)
                raw_prob = (
                    torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy().astype("float64")
                )
                if self.calibrator is not None:
                    prob = self.calibrator.predict(raw_prob.ravel()).reshape(raw_prob.shape)
                else:
                    prob = raw_prob
                prob = np.clip(prob, 0.0, 1.0)
                output[day] = prob
                current_fire_state = prob.astype("float32")

        return output


app = typer.Typer()


@app.callback()
def _callback() -> None:
    """CLI de calibración isotónica del U-Net de PyroCast."""


def _build_fixture_checkpoint(run_dir: Path, seed: int = 42) -> Path:
    # el MISMO train_model de P10 entrena este checkpoint -- es
    # literalmente "el checkpoint de fixture de P10" que el enunciado
    # pide, no una imitación local del entrenamiento.
    from models.deep.train import NDWSPretrainDataset, set_seed, train_model

    set_seed(seed)
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

    samples = [make_sample(i) for i in range(8)]
    config = TrainingConfig(
        phase="pretrain", in_channels=n_channels, base_channels=8, depth=2, lr=1e-2,
        batch_size=2, seed=seed, focal_alpha=0.8, focal_gamma=2.0, max_epochs=2, patience=2,
        data_paths=("synthetic-calibration-fixture",), pretrained_checkpoint=None,
    )
    model = SmallUNet(in_channels=n_channels, base_channels=8, depth=2)
    train_model(
        model, NDWSPretrainDataset(samples[:6]), NDWSPretrainDataset(samples[6:]),
        config, run_dir,
    )
    return run_dir / "best.pt"


def _build_fixture_val_dataset(seed: int) -> "Dataset[tuple[torch.Tensor, torch.Tensor]]":
    from models.deep.train import NDWSPretrainDataset

    size = 16
    n_channels = len(CHANNEL_ORDER)

    def make_sample(sample_id: int) -> PublicDatasetSample:
        rng = np.random.default_rng(sample_id)
        data = rng.random((1, n_channels, size, size)).astype("float32")
        tensor = xr.DataArray(
            data, dims=("day", "channel", "y", "x"),
            coords={"day": ["1970-01-01"], "channel": list(CHANNEL_ORDER)},
            name="fire_event_tensor", attrs={"resolution_m": 1000.0, "event_id": sample_id},
        )
        next_mask = (np.random.default_rng(sample_id + 1000).random((size, size)) > 0.8).astype(
            "float64"
        )
        return PublicDatasetSample(tensor=tensor, next_day_fire_mask=next_mask)

    samples = [make_sample(seed + 100 + i) for i in range(4)]
    return NDWSPretrainDataset(samples)


def _echo_report(result: CalibrationResult) -> None:
    typer.echo(f"Muestras evaluadas: {result.n_samples}")
    typer.echo(f"Brier -- antes: {result.brier_before:.4f}  después: {result.brier_after:.4f}")
    typer.echo(f"ECE   -- antes: {result.ece_before:.4f}  después: {result.ece_after:.4f}")


def run(
    fixture: bool = typer.Option(
        False, help="Entrena y calibra un checkpoint sintético diminuto"
    ),
    checkpoint: Path | None = typer.Option(
        None, help="Checkpoint real a calibrar (requiere --shard-dir)"
    ),
    shard_dir: Path | None = typer.Option(
        None, help="Shards NDWS para el set de validación real"
    ),
    chile_val: bool = typer.Option(
        False,
        help=(
            "Calibra contra el split 'val' de eventos reales de Chile "
            "(features/dataset/, requiere --checkpoint) en vez de shards NDWS -- "
            "para cuando no hay dataset público disponible, ver docs/backtest-2026.md."
        ),
    ),
    run_dir: Path = typer.Option(
        Path("runs") / "calibration", help="Dónde guardar checkpoint/calibrador de fixture"
    ),
    seed: int = typer.Option(42, help="Semilla de reproducibilidad"),
) -> None:
    """Calibra un checkpoint de SmallUNet con regresión isotónica y
    reporta Brier/ECE antes y después sobre el set de validación."""
    if fixture:
        checkpoint_path = _build_fixture_checkpoint(run_dir, seed=seed)
        val_dataset = _build_fixture_val_dataset(seed=seed)
        calibration_path = default_calibration_path(checkpoint_path)
    elif checkpoint is not None and chile_val:
        checkpoint_path = checkpoint
        if not checkpoint_path.exists():
            typer.echo(f"No existe el checkpoint {checkpoint_path}.")
            raise typer.Exit(code=1)
        settings = get_settings()
        dataset_dir = settings.data_processed_dir / "dataset"
        splits = json.loads((dataset_dir / "splits.json").read_text())
        if not splits["val"]:
            typer.echo(
                "El split de val de eventos de Chile está vacío -- no se puede "
                "reportar un antes/después honesto (ver docs/limitations.md)."
            )
            raise typer.Exit(code=1)
        from models.deep.train import ChileFinetuneDataset, _load_chile_events

        val_events = _load_chile_events(dataset_dir, splits["val"])
        val_dataset = ChileFinetuneDataset(val_events)
        calibration_path = default_calibration_path(checkpoint_path)
    elif checkpoint is not None and shard_dir is not None:
        checkpoint_path = checkpoint
        if not checkpoint_path.exists():
            typer.echo(f"No existe el checkpoint {checkpoint_path}.")
            raise typer.Exit(code=1)
        shard_paths = sorted(shard_dir.glob("*.tfrecord*"))
        if not shard_paths:
            typer.echo(f"No se encontraron shards *.tfrecord* en {shard_dir}.")
            raise typer.Exit(code=1)
        split = split_public_dataset(shard_paths, seed=seed)
        if not split["val"]:
            typer.echo(
                f"El split de val de NDWS quedó vacío (solo {len(shard_paths)} shard(s) -- "
                f"se necesitan al menos 2 para separar train/val). Sin val real no se "
                f"puede reportar un antes/después honesto (ver docs/limitations.md)."
            )
            raise typer.Exit(code=1)
        val_samples = list(load_public_dataset_samples(split["val"]))
        from models.deep.train import NDWSPretrainDataset

        val_dataset = NDWSPretrainDataset(val_samples)
        calibration_path = default_calibration_path(checkpoint_path)
    else:
        typer.echo(
            "Usar --fixture, --checkpoint junto con --shard-dir, o --checkpoint "
            "junto con --chile-val."
        )
        raise typer.Exit(code=1)

    result = calibrate_checkpoint(checkpoint_path, val_dataset, calibration_path=calibration_path)
    _echo_report(result)
    typer.echo(f"Calibrador guardado en {calibration_path}")


app.command("run")(run)
