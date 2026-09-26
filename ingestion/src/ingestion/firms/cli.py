"""Comando `firms` del CLI de ingesta: descarga, persiste y normaliza."""
import datetime as dt

import typer
from shared.config import get_settings

from ingestion.firms.client import FirmsClient
from ingestion.firms.parser import parse_csv_to_detections
from ingestion.firms.storage import save_raw_response

# SOURCE values documentados por el Area API de FIRMS (ver
# docs/superpowers/plans/2026-09-26-ingestion-firms.md para la
# verificación contra la documentación vigente). LANDSAT_NRT es
# exclusivo de US/Canadá — inútil para Chile, pero se deja en la lista
# para no romper a alguien que sí lo necesite fuera de este proyecto.
KNOWN_SENSORS = (
    "LANDSAT_NRT",
    "MODIS_NRT",
    "MODIS_SP",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT",
    "VIIRS_SNPP_SP",
)


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts_str = value.split(",")
    if len(parts_str) != 4:
        raise typer.BadParameter("bbox debe tener 4 valores: west,south,east,north")
    try:
        parts = [float(p) for p in parts_str]
    except ValueError as exc:
        raise typer.BadParameter(f"bbox debe ser 4 números separados por coma: {exc}") from exc
    return (parts[0], parts[1], parts[2], parts[3])


def firms(
    start: dt.datetime = typer.Option(
        ..., formats=["%Y-%m-%d"], help="Fecha de inicio (YYYY-MM-DD)"
    ),
    end: dt.datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Fecha de fin (YYYY-MM-DD)"),
    bbox: str | None = typer.Option(
        None, help="west,south,east,north — por defecto, el bbox de shared.config"
    ),
    sensor: str = typer.Option("VIIRS_SNPP_NRT", help="SOURCE del Area API de FIRMS"),
) -> None:
    """Descarga detecciones activas de FIRMS para un rango de fechas,
    las persiste en Parquet crudo y muestra un resumen normalizado."""
    if sensor not in KNOWN_SENSORS:
        raise typer.BadParameter(
            f"sensor desconocido {sensor!r}. Válidos: {', '.join(KNOWN_SENSORS)}"
        )
    if end.date() < start.date():
        raise typer.BadParameter(f"--end ({end.date()}) es anterior a --start ({start.date()})")

    settings = get_settings()
    area = _parse_bbox(bbox) if bbox else settings.study_area_bbox

    client = FirmsClient(map_key=settings.firms_map_key)
    chunks = client.fetch_range(area, sensor=sensor, start=start.date(), end=end.date())

    total_detections = 0
    total_chunks = 0
    for chunk_start, chunk_end, raw_csv in chunks:
        save_raw_response(
            raw_csv_text=raw_csv,
            query_start=chunk_start,
            query_end=chunk_end,
            bbox=area,
            sensor=sensor,
            downloaded_at=dt.datetime.now(dt.UTC),
            base_dir=settings.data_raw_dir,
        )
        detections = parse_csv_to_detections(raw_csv)
        total_detections += len(detections)
        total_chunks += 1
        typer.echo(f"{chunk_start}..{chunk_end}: {len(detections)} detecciones")

    typer.echo(f"Total: {total_detections} detecciones en {total_chunks} consulta(s)")
