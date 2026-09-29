"""Comando `sentinel2` del CLI de ingesta: composición mensual + NDVI."""
import typer
from features.vegetation.ndvi import compute_and_save_vegetation
from shared.config import get_settings

from ingestion.sentinel2.client import Sentinel2Client
from ingestion.sentinel2.pipeline import fetch_sentinel2


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    # mismo parseo que ingestion/firms/cli.py::_parse_bbox -- duplicado
    # a propósito (4 líneas, cada CLI de ingesta es independiente, ver
    # docs/decisions.md sobre duplicación deliberada de fragmentos
    # chicos en este proyecto).
    parts_str = value.split(",")
    if len(parts_str) != 4:
        raise typer.BadParameter("bbox debe tener 4 valores: west,south,east,north")
    try:
        parts = [float(p) for p in parts_str]
    except ValueError as exc:
        raise typer.BadParameter(f"bbox debe ser 4 números separados por coma: {exc}") from exc
    return (parts[0], parts[1], parts[2], parts[3])


def sentinel2(
    year: int = typer.Option(..., help="Año (YYYY)"),
    month: int = typer.Option(..., min=1, max=12, help="Mes (1-12)"),
    bbox: str | None = typer.Option(
        None,
        help=(
            "west,south,east,north — por defecto, el bbox de shared.config. "
            "El bbox completo de la zona de estudio excede el límite de "
            "píxeles de un job síncrono de openEO a 10 m nativos (~20000x20000) "
            "-- para esa zona hace falta un bbox más chico, p. ej. el de un "
            "evento real."
        ),
    ),
) -> None:
    """Descarga una composición mensual de menor nubosidad de Sentinel-2
    L2A y calcula NDVI reproyectado a la grilla de trabajo."""
    settings = get_settings()
    area = _parse_bbox(bbox) if bbox else settings.study_area_bbox
    client = Sentinel2Client(
        client_id=settings.copernicus_dataspace_client_id,
        client_secret=settings.copernicus_dataspace_client_secret,
    )
    composite_path = fetch_sentinel2(
        bbox=area,
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
    # compute_and_save_vegetation siempre escribe "ndvi.tif" -- sin este
    # renombrado, una segunda invocación para otro mes pisaría el NDVI del
    # mes anterior, dejando siempre un único archivo en disco. El nombre
    # con mes es lo que permite a features/dataset/ elegir "el composite
    # mensual más cercano" entre varios meses ya ingeridos.
    month_stamped_path = ndvi_path.parent / f"ndvi_{year:04d}-{month:02d}.tif"
    ndvi_path.replace(month_stamped_path)
    typer.echo(f"NDVI: {month_stamped_path}")
