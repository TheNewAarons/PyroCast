"""Comando `sentinel2` del CLI de ingesta: composición mensual + NDVI."""
import typer
from features.vegetation.ndvi import compute_and_save_vegetation
from shared.config import get_settings

from ingestion.sentinel2.client import Sentinel2Client
from ingestion.sentinel2.pipeline import fetch_sentinel2


def sentinel2(
    year: int = typer.Option(..., help="Año (YYYY)"),
    month: int = typer.Option(..., min=1, max=12, help="Mes (1-12)"),
) -> None:
    """Descarga una composición mensual de menor nubosidad de Sentinel-2
    L2A y calcula NDVI reproyectado a la grilla de trabajo."""
    settings = get_settings()
    client = Sentinel2Client(
        client_id=settings.copernicus_dataspace_client_id,
        client_secret=settings.copernicus_dataspace_client_secret,
    )
    composite_path = fetch_sentinel2(
        bbox=settings.study_area_bbox,
        year=year,
        month=month,
        client=client,
        cache_dir=settings.data_raw_dir / "sentinel2",
    )
    typer.echo(f"Composición Sentinel-2: {composite_path}")

    ndvi_path = compute_and_save_vegetation(
        composite_path,
        settings.data_processed_dir / "vegetation",
        target_crs=settings.crs,
        target_resolution_m=settings.spatial_resolution_m,
    )
    typer.echo(f"NDVI: {ndvi_path}")
