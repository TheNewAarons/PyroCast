"""Punto de entrada del CLI de ingesta: `pyrocast-ingest`."""
import typer

from ingestion.dem.cli import dem as dem_command
from ingestion.era5.cli import era5 as era5_command
from ingestion.firms.cli import firms as firms_command
from ingestion.sentinel2.cli import sentinel2 as sentinel2_command
from ingestion.worldcover.cli import worldcover as worldcover_command

app = typer.Typer()


@app.callback()
def _callback() -> None:
    """CLI de ingesta de datos abiertos para PyroCast."""


# Un callback vacío fuerza a Typer a tratar esto como un grupo de
# subcomandos ("pyrocast-ingest firms ..."), en vez de colapsar al modo
# de comando único que usaría si `firms` fuera el único comando
# registrado (comportamiento documentado de Typer).
app.command("firms")(firms_command)
app.command("dem")(dem_command)
app.command("era5")(era5_command)
app.command("sentinel2")(sentinel2_command)
app.command("worldcover")(worldcover_command)
