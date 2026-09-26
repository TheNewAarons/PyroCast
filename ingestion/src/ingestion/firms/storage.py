"""Persistencia cruda de respuestas del Area API de FIRMS.

Cada respuesta (ya sea de una consulta completa o de un chunk de ≤5
días) se guarda tal cual — mismas columnas que entrega la API, sin
normalizar — en Parquet, particionado por fecha de DESCARGA (no de
detección), junto a un .meta.json con los datos de la consulta.
"""
import csv
import datetime as dt
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def save_raw_response(
    raw_csv_text: str,
    query_start: dt.date,
    query_end: dt.date,
    bbox: tuple[float, float, float, float],
    sensor: str,
    downloaded_at: dt.datetime,
    base_dir: Path,
) -> Path:
    reader = csv.DictReader(io.StringIO(raw_csv_text))
    fieldnames = reader.fieldnames or []
    rows = list(reader)

    columns: dict[str, list[str]] = {name: [] for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            columns[name].append(row[name])
    table = (
        pa.table({name: pa.array(values) for name, values in columns.items()})
        if fieldnames
        else pa.table({})
    )

    partition_dir = base_dir / "firms" / f"download_date={downloaded_at.date().isoformat()}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{query_start.isoformat()}_{query_end.isoformat()}_{sensor}"
    parquet_path = partition_dir / f"{stem}.parquet"
    pq.write_table(table, parquet_path)

    meta = {
        "query_start": query_start.isoformat(),
        "query_end": query_end.isoformat(),
        "bbox": list(bbox),
        "sensor": sensor,
        "downloaded_at": downloaded_at.isoformat(),
        "row_count": len(rows),
    }
    parquet_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    return parquet_path
