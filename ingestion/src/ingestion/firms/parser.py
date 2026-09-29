"""Normaliza el CSV crudo del Area API de FIRMS a shared.schemas.FireDetection."""
import csv
import datetime as dt
import io

from shared.schemas import FireDetection


def _parse_detected_at(acq_date: str, acq_time: str) -> dt.datetime:
    # acq_time representa "HHMM" sin separador -- PERO el Area API de
    # FIRMS lo serializa como campo NUMÉRICO, no como texto de ancho
    # fijo: los ceros a la izquierda se pierden ("517" en el CSV real
    # significa 05:17 UTC, "5" significa 00:05 UTC). Verificado contra
    # una respuesta real de VIIRS_SNPP_SP (revisión final del
    # 2026-09-29) -- el comentario anterior de este archivo asumía "sí
    # viene con cero a la izquierda" sin haberlo confirmado contra datos
    # reales, y fallaba con "hour must be in 0..23" apenas se usó con
    # una descarga real en vez de fixtures. zfill(4) restaura el ancho
    # fijo antes de cortar HH/MM.
    padded = acq_time.zfill(4)
    hour = int(padded[:2])
    minute = int(padded[2:])
    date = dt.date.fromisoformat(acq_date)
    return dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=dt.UTC)


def parse_csv_to_detections(raw_csv_text: str) -> list[FireDetection]:
    reader = csv.DictReader(io.StringIO(raw_csv_text))
    detections: list[FireDetection] = []
    for row in reader:
        # row.get(...) puede devolver None (no solo levantar KeyError) si
        # una fila truncada a mitad de columna deja menos valores que
        # encabezados — csv.DictReader rellena esos con None, no "".
        frp_raw = (row.get("frp") or "").strip()
        try:
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
        except KeyError as exc:
            raise ValueError(
                f"Fila de CSV de FIRMS sin la columna requerida {exc}: {row!r}"
            ) from exc
    return detections
