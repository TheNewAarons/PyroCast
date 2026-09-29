"""Punto de entrada del CLI de modelos: `pyrocast-models`."""
import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np
import typer
import xarray as xr
from shared.config import get_settings
from shared.model_protocol import FireSpreadModel
from sqlalchemy import create_engine

from models.cellular_automata.model import CellularAutomatonModel
from models.cellular_automata.simulate import simulate_fire_spread
from models.deep.calibration import CalibratedUNet, default_calibration_path
from models.evaluation.backtest import BacktestResult, run_backtest
from models.evaluation.db import persist_backtest_run

_ECE_BINS = 10
_CONFIDENCE = 0.95


def _git_commit_hash() -> str:
    # bench/results/ debe ser reproducible por otra persona con las
    # mismas credenciales (docs/backtest-2026.md, instrucción 3) -- sin
    # el commit exacto, no hay forma de saber con qué versión del
    # código se generó un resultado.
    # cwd fijo al repo (no al cwd del proceso): el comando se invoca
    # desde cualquier directorio de trabajo -- p. ej. los tests
    # aíslan la salida de bench/results/ en un tmp_path sin .git --,
    # pero el commit reportado siempre debe ser el de ESTE código, no
    # el de un repo git que resulte estar en el cwd de turno.
    repo_root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True,
            check=True,
        )
        return completed.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _exact_command(args: list[str]) -> str:
    # se reconstruye desde los valores REALES de los parámetros del
    # comando en vez de leer sys.argv -- sys.argv no refleja la
    # invocación bajo CliRunner (tests) ni es estable entre formas de
    # invocar el mismo comando (uv run, el entry point instalado, etc).
    return " ".join(["pyrocast-models", "backtest", *args])

app = typer.Typer()


@app.callback()
def _callback() -> None:
    """CLI de modelos de PyroCast."""


def run_ca(
    n_days: int = typer.Option(10, help="Días a simular"),
    size: int = typer.Option(41, help="Tamaño (alto=ancho) de la grilla de fixture"),
    seed: int = typer.Option(42, help="Semilla del componente probabilístico"),
) -> None:
    """Simula un evento de FIXTURE (terreno plano, sin viento, combustible
    homogéneo, ignición en el centro) con el autómata celular, de
    principio a fin -- no requiere un evento real de features/dataset/."""
    center = size // 2
    initial_burning = np.zeros((size, size), dtype=bool)
    initial_burning[center, center] = True
    elevation = np.zeros((size, size))
    wind_u = np.zeros((size, size))
    wind_v = np.zeros((size, size))
    fuel_type = np.ones((size, size), dtype=int)  # 1 = pastizal, homogéneo

    probabilities = simulate_fire_spread(
        initial_burning, elevation, wind_u, wind_v, fuel_type,
        resolution_m=100.0, n_days=n_days, seed=seed,
    )
    for day in range(n_days):
        burning_cells = int(np.sum(probabilities[day] >= 0.5))
        typer.echo(f"Día {day + 1}: {burning_cells} celda(s) con probabilidad >= 0.5")


def load_test_events(dataset_dir: Path) -> list[xr.DataArray]:
    """Lee `splits.json` + los eventos Zarr del split "test", por la
    misma convención de rutas que `features/dataset/` ya establece."""
    splits = json.loads((dataset_dir / "splits.json").read_text())
    events = []
    for event_id in splits["test"]:
        zarr_path = dataset_dir / f"event_{event_id:04d}.zarr"
        opened = xr.open_zarr(zarr_path)["fire_event_tensor"]
        events.append(opened)
    return events


def _result_to_json(
    result: BacktestResult,
    model_name: str,
    config: dict[str, object],
    seed: int,
    n_bootstrap: int,
    ece_bins: int,
    confidence: float,
    command_args: list[str],
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "config": config,
        "split": "test",
        "command": _exact_command(command_args),
        "git_commit": _git_commit_hash(),
        "n_events": len(result.per_event),
        # sin estos cuatro valores, baseline.json no se puede
        # regenerar a partir de su propio contenido -- violaría el
        # mismo criterio de reproducibilidad que P6 (splits.json:
        # "nada de resultados no versionables"). Encontrado en la
        # revisión final del 2026-09-28.
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "ece_bins": ece_bins,
        "confidence": confidence,
        "per_event": [
            {
                "event_id": m.event_id, "iou": m.iou, "dice": m.dice,
                "brier": m.brier, "ece": m.ece,
            }
            for m in result.per_event
        ],
        "aggregate": {
            name: {"point_estimate": ci.point_estimate, "lower": ci.lower, "upper": ci.upper}
            for name, ci in result.aggregate.items()
        },
    }


def backtest(
    model: str = typer.Option(
        "cellular_automata",
        help="Modelo a evaluar: 'cellular_automata' o 'unet' (requiere --checkpoint)",
    ),
    checkpoint: Path | None = typer.Option(
        None, help="Checkpoint de U-Net a evaluar (solo --model unet)"
    ),
    calibration: Path | None = typer.Option(
        None,
        help=(
            "Calibrador isotónico del checkpoint (solo --model unet) -- por "
            "defecto, <checkpoint>.calibrator.pt (ver models.deep.calibration."
            "default_calibration_path). Si no existe, corre sin calibrar."
        ),
    ),
    n_bootstrap: int = typer.Option(1000, help="Número de remuestreos bootstrap"),
    seed: int = typer.Option(42, help="Semilla del bootstrap y de la simulación"),
) -> None:
    """Corre un modelo (autómata celular, parámetros por defecto de
    SpreadParameters, SIN CALIBRAR contra incendios reales -- ver
    docs/limitations.md y docs/cellular-automata.md; o U-Net calibrado
    vía --model unet) contra el split de test de `features/dataset/` y
    guarda el resultado en bench/results/<model>.json y en PostGIS
    (model_run + evaluation_result). Requiere que `pyrocast-features
    build-dataset` ya haya corrido -- ver docs/dataset-card.md."""
    settings = get_settings()
    dataset_dir = settings.data_processed_dir / "dataset"
    events = load_test_events(dataset_dir)
    if not events:
        typer.echo("El split de test no tiene eventos.")
        raise typer.Exit(code=0)

    command_args = [f"--n-bootstrap {n_bootstrap}", f"--seed {seed}"]
    if model == "cellular_automata":
        spread_model: FireSpreadModel = CellularAutomatonModel(seed=seed)
        # derivado del objeto real que corrió, no copiado a mano -- si
        # alguien cambia un default en rules.py, el registro sigue
        # describiendo la corrida real (encontrado en la revisión final
        # del 2026-09-28: la versión anterior tenía un dict hardcodeado
        # que podía desincronizarse en silencio, y omitía
        # fuel_flammability).
        config: dict[str, object] = dataclasses.asdict(spread_model.params)
        output_name = "baseline.json"
    elif model == "unet":
        if checkpoint is None:
            typer.echo("--model unet requiere --checkpoint.")
            raise typer.Exit(code=1)
        calibration_path = calibration if calibration is not None else default_calibration_path(
            checkpoint
        )
        if not calibration_path.exists():
            calibration_path_or_none = None
        else:
            calibration_path_or_none = calibration_path
        spread_model = CalibratedUNet(checkpoint, calibration_path_or_none)
        config = {
            "checkpoint": str(checkpoint),
            "calibration": str(calibration_path_or_none) if calibration_path_or_none else None,
        }
        output_name = "unet.json"
        command_args = [f"--model unet --checkpoint {checkpoint}", *command_args]
    else:
        typer.echo(f"Modelo desconocido: {model!r} -- usar 'cellular_automata' o 'unet'.")
        raise typer.Exit(code=1)

    result = run_backtest(
        spread_model, events, n_bootstrap=n_bootstrap, seed=seed, ece_bins=_ECE_BINS,
        confidence=_CONFIDENCE,
    )

    output_dir = Path("bench") / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _result_to_json(
        result, model, config, seed, n_bootstrap, _ECE_BINS, _CONFIDENCE, command_args
    )
    (output_dir / output_name).write_text(json.dumps(payload, sort_keys=True, indent=2))
    typer.echo(f"Backtest: {len(result.per_event)} evento(s) -> {output_dir / output_name}")

    engine = create_engine(settings.postgres_dsn)
    persist_backtest_run(
        engine=engine, per_event=result.per_event,
        model_name=model, config=config, split="test",
    )
    typer.echo(f"Métricas persistidas en PostGIS ({len(result.per_event)} model_run).")


app.command("run-ca")(run_ca)
app.command("backtest")(backtest)
