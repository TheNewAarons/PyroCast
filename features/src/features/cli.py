"""Punto de entrada del CLI de features: `pyrocast-features`."""
import datetime as dt
import json

import typer
from shared.config import get_settings
from sqlalchemy import create_engine

from features.dataset.assemble import save_event_to_zarr
from features.dataset.db import persist_fire_event_metadata
from features.dataset.firms_loader import load_firms_detections
from features.dataset.pipeline import build_dataset_for_event, resolve_event_sources
from features.dataset.split import split_events
from features.fire_state.clustering import build_fire_events

app = typer.Typer()


@app.callback()
def _callback() -> None:
    """CLI de features de PyroCast."""


def build_dataset(
    start: dt.datetime = typer.Option(
        ..., formats=["%Y-%m-%d"], help="Fecha de inicio (YYYY-MM-DD)"
    ),
    end: dt.datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Fecha de fin (YYYY-MM-DD)"),
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
    if not events:
        typer.echo("No se encontraron eventos de incendio en el rango pedido.")
        raise typer.Exit(code=0)

    output_dir = settings.data_processed_dir / "dataset"
    engine = create_engine(settings.postgres_dsn)

    event_ids: list[int] = []
    for event in events:
        padded_start = event.start_date - dt.timedelta(days=5)
        days = [
            padded_start + dt.timedelta(days=i)
            for i in range((event.end_date - padded_start).days + 1)
        ]
        sources = resolve_event_sources(days, settings)
        tensor, bbox_cut = build_dataset_for_event(
            event, sources, settings.spatial_resolution_m, settings.crs
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
