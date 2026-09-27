"""Punto de entrada del CLI de modelos: `pyrocast-models`."""
import numpy as np
import typer

from models.cellular_automata.simulate import simulate_fire_spread

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


app.command("run-ca")(run_ca)
