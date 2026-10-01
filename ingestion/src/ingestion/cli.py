"""Punto de entrada del CLI de ingesta: `pyrocast-ingest`."""
import typer

from ingestion.dem.cli import dem as dem_command
from ingestion.era5.cli import era5 as era5_command
from ingestion.firms.cli import firms as firms_command
from ingestion.resilience import handle_errors
from ingestion.sentinel2.cli import sentinel2 as sentinel2_command
from ingestion.worldcover.cli import worldcover as worldcover_command

# show_locals=False: los locals de un traceback incluyen credenciales
# (map_key, client_secret...) -- nunca deben imprimirse.
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.callback()
def _callback() -> None:
    """CLI de ingesta de datos abiertos para PyroCast."""


# Un callback vacío fuerza a Typer a tratar esto como un grupo de
# subcomandos ("pyrocast-ingest firms ..."), en vez de colapsar al modo
# de comando único que usaría si `firms` fuera el único comando
# registrado (comportamiento documentado de Typer).
app.command("firms")(handle_errors(firms_command))
app.command("dem")(handle_errors(dem_command))
app.command("era5")(handle_errors(era5_command))
app.command("sentinel2")(handle_errors(sentinel2_command))
app.command("worldcover")(handle_errors(worldcover_command))
