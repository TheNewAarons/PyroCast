"""Comando `era5` del CLI de ingesta: descarga diaria agregada + derivados
de clima (viento, humedad, reproyección)."""
import datetime as dt

import typer
from features.weather.derive import compute_and_save_weather
from shared.config import get_settings

from ingestion.era5.client import Era5Client
from ingestion.era5.pipeline import fetch_daily_era5


def era5(
    start: dt.datetime = typer.Option(
        ..., formats=["%Y-%m-%d"], help="Fecha de inicio (YYYY-MM-DD)"
    ),
    end: dt.datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Fecha de fin (YYYY-MM-DD)"),
    timeout_seconds: float = typer.Option(
        3600.0, help="Timeout total de espera por solicitud CDS (segundos)"
    ),
) -> None:
    """Descarga ERA5-Land agregado a diario para el bbox de estudio y
    calcula viento/humedad reproyectados a la grilla de trabajo."""
    settings = get_settings()
    client = Era5Client(url=settings.cds_api_url, key=settings.cds_api_key)

    daily_path = fetch_daily_era5(
        bbox=settings.study_area_bbox,
        start=start.date(),
        end=end.date(),
        era5_client=client,
        raw_dir=settings.data_raw_dir / "era5",
        cache_dir=settings.data_processed_dir / "era5",
        timeout_seconds=timeout_seconds,
    )
    typer.echo(f"ERA5-Land diario: {daily_path}")

    paths = compute_and_save_weather(
        daily_path,
        settings.data_processed_dir / "weather",
        target_crs=settings.crs,
        target_resolution_m=settings.spatial_resolution_m,
    )
    for field_name, per_date in paths.items():
        typer.echo(f"{field_name}: {len(per_date)} día(s)")
