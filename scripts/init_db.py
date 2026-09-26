"""CLI para crear las tablas iniciales en PostGIS.

Uso: uv run --package serving python scripts/init_db.py create-all
(usa --package serving porque es el único miembro del workspace que ya
depende de typer y de shared; scripts/ no es un miembro propio).
"""
import typer
from sqlalchemy import create_engine, text

from shared.config import get_settings
from shared.db.schema import metadata

app = typer.Typer()


@app.command("create-all")
def create_all() -> None:
    settings = get_settings()
    engine = create_engine(settings.postgres_dsn)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        metadata.create_all(conn)
    typer.echo("Tablas creadas: fire_event, model_run, evaluation_result")


if __name__ == "__main__":
    app()
