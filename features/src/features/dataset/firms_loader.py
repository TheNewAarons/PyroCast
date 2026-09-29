"""Lee las detecciones FIRMS crudas directamente desde el parquet que
`ingestion.firms.storage.save_raw_response` ya escribe -- sin importar
`ingestion` (ver docs/decisions.md: `features` nunca depende de
`ingestion`, es al revés). Duplica ~20 líneas de lectura de un formato de
almacenamiento simple (columnas string, mismo esquema que el CSV del Area
API de FIRMS) en vez de invertir esa dependencia."""
import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq
from shared.schemas import FireDetection


def _parse_detected_at(acq_date: str, acq_time: str) -> dt.datetime:
    # acq_time representa "HHMM" sin separador -- PERO el Area API de
    # FIRMS lo serializa como campo NUMÉRICO: los ceros a la izquierda se
    # pierden ("517" en el parquet real significa 05:17 UTC, no "51:7").
    # Verificado contra datos reales (revisión final del 2026-09-29,
    # mismo hallazgo que ingestion/firms/parser.py -- esta función lo
    # duplica a propósito, así que también duplicaba el bug). zfill(4)
    # restaura el ancho fijo antes de cortar HH/MM.
    padded = acq_time.zfill(4)
    hour = int(padded[:2])
    minute = int(padded[2:])
    date = dt.date.fromisoformat(acq_date)
    return dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=dt.UTC)


def load_firms_detections(
    base_dir: Path, start: dt.date, end: dt.date
) -> list[FireDetection]:
    firms_dir = base_dir / "firms"
    if not firms_dir.exists():
        return []

    detections: list[FireDetection] = []
    # Re-correr `pyrocast-ingest firms` para un rango ya cubierto (p. ej.
    # en un día de descarga distinto) escribe una partición NUEVA con las
    # mismas filas -- sin dedup, la misma detección física física se
    # cuenta dos veces, cambiando silenciosamente el event_id (hash de
    # contenido) y el conteo de detecciones del evento. Se deduplica por
    # (fecha/hora, coordenadas redondeadas, satélite): dos overpasses
    # reales distintos casi nunca coinciden en los tres a la vez.
    seen: set[tuple[str, float, float, str]] = set()
    for parquet_path in sorted(firms_dir.glob("**/*.parquet")):
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            acq_date = dt.date.fromisoformat(row["acq_date"])
            if not (start <= acq_date <= end):
                continue
            dedup_key = (
                row["acq_date"] + row["acq_time"],
                round(float(row["latitude"]), 6),
                round(float(row["longitude"]), 6),
                row["satellite"],
            )
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            frp_raw = (row.get("frp") or "").strip()
            detections.append(
                FireDetection(
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    detected_at=_parse_detected_at(row["acq_date"], row["acq_time"]),
                    frp=float(frp_raw) if frp_raw else None,
                    confidence=row["confidence"],
                    satellite=row["satellite"],
                    instrument=row["instrument"],
                )
            )
    return detections
