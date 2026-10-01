"""Punto de entrada del CLI de features: `pyrocast-features`."""
import datetime as dt
import json

import typer
from shared.config import get_settings
from sqlalchemy import create_engine

from features.dataset.assemble import save_event_to_zarr
from features.dataset.db import persist_fire_event_metadata
from features.dataset.firms_loader import load_firms_detections
from features.dataset.pipeline import (
    build_dataset_for_event,
    padded_days_for_event,
    resolve_event_sources,
)
from features.dataset.split import split_events
from features.fire_state.clustering import build_fire_events

# show_locals=False: los locals de un traceback pueden incluir credenciales
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.callback()
def _callback() -> None:
    """CLI de features de PyroCast."""


def _parse_event_ids(value: str) -> frozenset[int]:
    try:
        return frozenset(int(part) for part in value.split(","))
    except ValueError as exc:
        raise typer.BadParameter(
            f"event-ids debe ser una lista de enteros separados por coma: {exc}"
        ) from exc


def build_dataset(
    start: dt.datetime = typer.Option(
        ..., formats=["%Y-%m-%d"], help="Fecha de inicio (YYYY-MM-DD)"
    ),
    end: dt.datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Fecha de fin (YYYY-MM-DD)"),
    event_ids_filter: str | None = typer.Option(
        None,
        "--event-ids",
        help=(
            "IDs de evento separados por coma -- si se pasa, procesa SOLO esos "
            "eventos entre los clusterizados en el rango de fechas, en vez de "
            "todos. Un rango de fechas real de temporada completa clusteriza en "
            "cientos o miles de eventos, la mayoría ruido de 1-2 detecciones; "
            "este filtro permite acotar a un subconjunto elegido explícitamente "
            "(p. ej. por cantidad de detecciones) de forma reproducible."
        ),
    ),
) -> None:
    """Ensambla el dataset espaciotemporal por evento (P2-P6): clusteriza
    detecciones FIRMS ya ingeridas en eventos, arma el tensor de cada uno
    a partir de las capas ya procesadas (DEM/ERA5-Land/Sentinel-2/
    WorldCover -- deben haberse ingerido antes con `pyrocast-ingest`, ver
    docs/dataset-card.md), lo persiste en Zarr, guarda sus metadatos en
    PostGIS, y calcula el split train/val/test reproducible."""
    settings = get_settings()

    detections = load_firms_detections(settings.data_raw_dir, start.date(), end.date())
    events = build_fire_events(detections)
    if event_ids_filter is not None:
        wanted = _parse_event_ids(event_ids_filter)
        events = [event for event in events if event.event_id in wanted]
    if not events:
        typer.echo("No se encontraron eventos de incendio en el rango pedido.")
        raise typer.Exit(code=0)

    output_dir = settings.data_processed_dir / "dataset"
    engine = create_engine(settings.postgres_dsn)

    event_ids: list[int] = []
    for event in events:
        # padded_days_for_event es el ÚNICO lugar que calcula esta
        # ventana -- antes recalculaba un `5` hardcodeado por su cuenta,
        # pudiendo desincronizarse de DEFAULT_PRE_EVENT_PADDING_DAYS
        # (encontrado en la revisión final del 2026-09-27).
        days = padded_days_for_event(event)
        sources = resolve_event_sources(days, settings)
        tensor, bbox_cut = build_dataset_for_event(
            event, sources, settings.spatial_resolution_m, settings.crs, event_id=event.event_id
        )
        zarr_path = save_event_to_zarr(tensor, output_dir, event.event_id)
        persist_fire_event_metadata(
            engine=engine,
            event_id=event.event_id,
            bbox_cut=bbox_cut,
            start_date=event.start_date,
            end_date=event.end_date,
        )
        event_ids.append(event.event_id)
        typer.echo(f"Evento {event.event_id}: {zarr_path}")

    splits = split_events(event_ids)
    (output_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    typer.echo(
        f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )


app.command("build-dataset")(build_dataset)
