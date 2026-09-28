"""Punto de entrada del CLI de modelos: `pyrocast-models`."""
import json
from pathlib import Path

import numpy as np
import typer
import xarray as xr
from shared.config import get_settings
from sqlalchemy import create_engine

from models.cellular_automata.model import CellularAutomatonModel
from models.cellular_automata.simulate import simulate_fire_spread
from models.evaluation.backtest import BacktestResult, run_backtest
from models.evaluation.db import persist_backtest_run

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
    result: BacktestResult, model_name: str, config: dict[str, float]
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "config": config,
        "split": "test",
        "n_events": len(result.per_event),
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
    n_bootstrap: int = typer.Option(1000, help="Número de remuestreos bootstrap"),
    seed: int = typer.Option(42, help="Semilla del bootstrap y de la simulación"),
) -> None:
    """Corre el backtest del autómata celular calibrado contra el split
    de test de `features/dataset/` y guarda el resultado en
    bench/results/baseline.json y en PostGIS (model_run +
    evaluation_result). Requiere que `pyrocast-features build-dataset`
    ya haya corrido -- ver docs/dataset-card.md."""
    settings = get_settings()
    dataset_dir = settings.data_processed_dir / "dataset"
    events = load_test_events(dataset_dir)
    if not events:
        typer.echo("El split de test no tiene eventos.")
        raise typer.Exit(code=0)

    params = {"base_spread_prob": 0.3, "slope_coefficient": 4.0, "wind_coefficient": 0.2}
    model = CellularAutomatonModel(seed=seed)
    result = run_backtest(model, events, n_bootstrap=n_bootstrap, seed=seed)

    output_dir = Path("bench") / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _result_to_json(result, "cellular_automata", params)
    (output_dir / "baseline.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2)
    )
    typer.echo(f"Backtest: {len(result.per_event)} evento(s) -> {output_dir / 'baseline.json'}")

    engine = create_engine(settings.postgres_dsn)
    for metrics in result.per_event:
        persist_backtest_run(
            engine=engine, firms_event_id=metrics.event_id,
            model_name="cellular_automata", config=params, split="test", metrics=metrics,
        )
    typer.echo(f"Métricas persistidas en PostGIS ({len(result.per_event)} evaluation_result).")


app.command("run-ca")(run_ca)
app.command("backtest")(backtest)
