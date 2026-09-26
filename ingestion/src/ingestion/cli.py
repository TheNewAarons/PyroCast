"""Punto de entrada del CLI de ingesta: `pyrocast-ingest`."""
import typer

from ingestion.firms.cli import firms as firms_command

app = typer.Typer()


@app.callback()
def _callback() -> None:
    """CLI de ingesta de datos abiertos para PyroCast."""


# Un callback vacío fuerza a Typer a tratar esto como un grupo de
# subcomandos ("pyrocast-ingest firms ..."), en vez de colapsar al modo
# de comando único que usaría si `firms` fuera el único comando
# registrado (comportamiento documentado de Typer).
app.command("firms")(firms_command)
