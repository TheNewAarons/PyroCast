"""Comando `firms` del CLI de ingesta: descarga, persiste y normaliza."""
import datetime as dt

import typer
from shared.config import get_settings

from ingestion.firms.client import FirmsClient
from ingestion.firms.parser import parse_csv_to_detections
from ingestion.firms.storage import save_raw_response


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("bbox debe tener 4 valores: west,south,east,north")
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
    settings = get_settings()
    area = _parse_bbox(bbox) if bbox else settings.study_area_bbox

    client = FirmsClient(map_key=settings.firms_map_key)
    chunks = client.fetch_range(area, sensor=sensor, start=start.date(), end=end.date())

    total_detections = 0
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
        typer.echo(f"{chunk_start}..{chunk_end}: {len(detections)} detecciones")

    typer.echo(f"Total: {total_detections} detecciones en {len(chunks)} consulta(s)")
