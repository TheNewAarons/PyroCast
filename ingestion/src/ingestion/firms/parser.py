"""Normaliza el CSV crudo del Area API de FIRMS a shared.schemas.FireDetection."""
import csv
import datetime as dt
import io

from shared.schemas import FireDetection


def _parse_detected_at(acq_date: str, acq_time: str) -> dt.datetime:
    # acq_time viene como "HHMM" sin separador, con cero a la izquierda
    # (p. ej. "0005" = 00:05 UTC) — NO tratar como entero, se pierde el
    # cero inicial.
    hour = int(acq_time[:2])
    minute = int(acq_time[2:])
    date = dt.date.fromisoformat(acq_date)
    return dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=dt.UTC)


def parse_csv_to_detections(raw_csv_text: str) -> list[FireDetection]:
    reader = csv.DictReader(io.StringIO(raw_csv_text))
    detections: list[FireDetection] = []
    for row in reader:
        frp_raw = row.get("frp", "").strip()
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
