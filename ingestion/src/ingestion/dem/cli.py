"""Comando `dem` del CLI de ingesta: descarga+mosaico+reproyección del DEM
de Copernicus, y cálculo de pendiente/orientación (derivados) sobre el
resultado — un solo comando porque `make ingest-terrain` (Makefile) está
documentado como "DEM + derivados" en un solo paso.

Nota de arquitectura: esto hace que `ingestion` dependa de `features`
(ver ingestion/pyproject.toml), invirtiendo la dirección habitual
ingestion -> features del diagrama de CLAUDE.md. Es una dependencia
acotada a este módulo de CLI únicamente — ingestion/dem/pipeline.py en
sí no importa features. Ver docs/decisions.md.
"""
import typer
from features.terrain.slope_aspect import compute_and_save_terrain
from shared.config import get_settings

from ingestion.dem.pipeline import build_dem


def dem() -> None:
    """Descarga Copernicus DEM GLO-30 para el bbox de estudio, lo mosaica
    y reproyecta a la grilla del proyecto, y calcula pendiente/orientación."""
    settings = get_settings()
    dem_path = build_dem(
        bbox=settings.study_area_bbox,
        resolution_m=settings.spatial_resolution_m,
        crs=settings.crs,
        raw_tiles_dir=settings.data_raw_dir / "dem",
        cache_dir=settings.data_processed_dir / "dem",
    )
    typer.echo(f"DEM: {dem_path}")

    slope_path, aspect_path = compute_and_save_terrain(
        dem_path, settings.data_processed_dir / "terrain"
    )
    typer.echo(f"Pendiente: {slope_path}")
    typer.echo(f"Orientación: {aspect_path}")
