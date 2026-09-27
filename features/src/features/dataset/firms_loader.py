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
    # acq_time viene como "HHMM" sin separador, con cero a la izquierda
    # (p. ej. "0005" = 00:05 UTC) -- NO tratar como entero, se pierde el
    # cero inicial. (Mismo cuidado que ingestion/firms/parser.py.)
    hour = int(acq_time[:2])
    minute = int(acq_time[2:])
    date = dt.date.fromisoformat(acq_date)
    return dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=dt.UTC)


def load_firms_detections(
    base_dir: Path, start: dt.date, end: dt.date
) -> list[FireDetection]:
    firms_dir = base_dir / "firms"
    if not firms_dir.exists():
        return []

    detections: list[FireDetection] = []
    for parquet_path in sorted(firms_dir.glob("**/*.parquet")):
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            acq_date = dt.date.fromisoformat(row["acq_date"])
            if not (start <= acq_date <= end):
                continue
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
