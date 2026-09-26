"""Comando `worldcover` del CLI de ingesta: descarga+mosaico+tipo de
combustible."""
import rasterio
import typer
from shared.config import get_settings

from ingestion.worldcover.fuel_type import map_worldcover_to_fuel_type
from ingestion.worldcover.pipeline import build_worldcover


def worldcover() -> None:
    """Descarga ESA WorldCover para el bbox de estudio, lo mosaica y
    reproyecta (nearest), y calcula el tipo de combustible simplificado."""
    settings = get_settings()
    worldcover_path = build_worldcover(
        bbox=settings.study_area_bbox,
        resolution_m=settings.spatial_resolution_m,
        crs=settings.crs,
        raw_tiles_dir=settings.data_raw_dir / "worldcover",
        cache_dir=settings.data_processed_dir / "worldcover",
    )
    typer.echo(f"WorldCover: {worldcover_path}")

    with rasterio.open(worldcover_path) as src:
        classes = src.read(1)
        profile = src.profile

    fuel_type = map_worldcover_to_fuel_type(classes)
    output_dir = settings.data_processed_dir / "vegetation"
    output_dir.mkdir(parents=True, exist_ok=True)
    fuel_type_path = output_dir / "fuel_type.tif"
    fuel_profile = {**profile, "dtype": "int32", "nodata": None}
    with rasterio.open(fuel_type_path, "w", **fuel_profile) as dst:
        dst.write(fuel_type.astype("int32"), 1)
    typer.echo(f"Tipo de combustible: {fuel_type_path}")
