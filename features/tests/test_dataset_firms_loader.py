"""Tests del lector de detecciones FIRMS crudas (parquet), sin depender
de `ingestion` -- ver docs/decisions.md."""
import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from features.dataset.firms_loader import load_firms_detections


def _write_fixture_parquet(base_dir: Path, rows: list[dict], filename: str) -> None:
    partition_dir = base_dir / "firms" / "download_date=2026-01-20"
    partition_dir.mkdir(parents=True, exist_ok=True)
    columns = {name: [row[name] for row in rows] for name in rows[0]}
    table = pa.table({name: pa.array(values) for name, values in columns.items()})
    pq.write_table(table, partition_dir / filename)


def test_load_firms_detections_filters_by_date_range(tmp_path):
    _write_fixture_parquet(
        tmp_path,
        [
            {
                "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2026-01-10",
                "acq_time": "0130", "confidence": "n", "satellite": "N",
                "instrument": "VIIRS", "frp": "12.3",
            },
            {
                "latitude": "-37.6", "longitude": "-72.4", "acq_date": "2026-01-15",
                "acq_time": "0140", "confidence": "h", "satellite": "N",
                "instrument": "VIIRS", "frp": "8.0",
            },
            {
                "latitude": "-37.7", "longitude": "-72.5", "acq_date": "2026-02-01",
                "acq_time": "0150", "confidence": "n", "satellite": "N",
                "instrument": "VIIRS", "frp": "5.0",
            },
        ],
        "chunk1.parquet",
    )
    detections = load_firms_detections(tmp_path, dt.date(2026, 1, 12), dt.date(2026, 1, 20))
    assert len(detections) == 1
    assert detections[0].latitude == -37.6
    assert detections[0].detected_at == dt.datetime(2026, 1, 15, 1, 40, tzinfo=dt.UTC)


def test_load_firms_detections_preserves_leading_zero_in_acq_time(tmp_path):
    _write_fixture_parquet(
        tmp_path,
        [{
            "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2026-01-15",
            "acq_time": "0005", "confidence": "n", "satellite": "N",
            "instrument": "VIIRS", "frp": "1.0",
        }],
        "chunk1.parquet",
    )
    detections = load_firms_detections(tmp_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert detections[0].detected_at.hour == 0
    assert detections[0].detected_at.minute == 5


def test_load_firms_detections_handles_a_real_firms_time_with_dropped_leading_zeros(tmp_path):
    # Mismo hallazgo que ingestion/firms/parser.py (revisión final del
    # 2026-09-29, contra datos REALES de FIRMS VIIRS_SNPP_SP): el Area
    # API de FIRMS serializa acq_time como campo NUMÉRICO -- "517" en el
    # parquet real significa 05:17 UTC, no "51:7" (ValueError: hour must
    # be in 0..23). Esta función DUPLICA _parse_detected_at de
    # ingestion/firms/parser.py a propósito (features nunca depende de
    # ingestion) -- pero eso significa que también duplicó el bug.
    _write_fixture_parquet(
        tmp_path,
        [{
            "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2025-11-01",
            "acq_time": "517", "confidence": "n", "satellite": "N",
            "instrument": "VIIRS", "frp": "1.4",
        }],
        "chunk1.parquet",
    )
    detections = load_firms_detections(tmp_path, dt.date(2025, 11, 1), dt.date(2025, 11, 1))
    assert detections[0].detected_at == dt.datetime(2025, 11, 1, 5, 17, tzinfo=dt.UTC)


def test_load_firms_detections_handles_missing_frp(tmp_path):
    _write_fixture_parquet(
        tmp_path,
        [{
            "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2026-01-15",
            "acq_time": "0130", "confidence": "n", "satellite": "N",
            "instrument": "VIIRS", "frp": "",
        }],
        "chunk1.parquet",
    )
    detections = load_firms_detections(tmp_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert detections[0].frp is None


def test_load_firms_detections_returns_empty_list_when_no_files(tmp_path):
    assert load_firms_detections(tmp_path, dt.date(2026, 1, 1), dt.date(2026, 1, 31)) == []


def test_load_firms_detections_deduplicates_across_overlapping_parquet_files(tmp_path):
    # Re-correr `make ingest-firms` para el mismo rango en un día
    # distinto escribe una SEGUNDA partición `download_date=...` con las
    # mismas filas -- sin dedup, la misma detección física cuenta dos
    # veces, cambiando `event_id` (hash de contenido) y el conteo de
    # detecciones del evento. Encontrado en la revisión final del
    # 2026-09-27.
    row = {
        "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2026-01-15",
        "acq_time": "0130", "confidence": "n", "satellite": "N",
        "instrument": "VIIRS", "frp": "1.0",
    }
    _write_fixture_parquet(tmp_path, [row], "chunk_run1.parquet")
    partition_dir_2 = tmp_path / "firms" / "download_date=2026-01-21"
    partition_dir_2.mkdir(parents=True, exist_ok=True)
    table = pa.table({name: pa.array([value]) for name, value in row.items()})
    pq.write_table(table, partition_dir_2 / "chunk_run2.parquet")

    detections = load_firms_detections(tmp_path, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert len(detections) == 1


def test_load_firms_detections_reads_across_multiple_parquet_files(tmp_path):
    _write_fixture_parquet(
        tmp_path,
        [{
            "latitude": "-37.5", "longitude": "-72.3", "acq_date": "2026-01-15",
            "acq_time": "0130", "confidence": "n", "satellite": "N",
            "instrument": "VIIRS", "frp": "1.0",
        }],
        "chunk1.parquet",
    )
    _write_fixture_parquet(
        tmp_path,
        [{
            "latitude": "-38.5", "longitude": "-73.3", "acq_date": "2026-01-16",
            "acq_time": "0230", "confidence": "h", "satellite": "N",
            "instrument": "VIIRS", "frp": "2.0",
        }],
        "chunk2.parquet",
    )
    detections = load_firms_detections(tmp_path, dt.date(2026, 1, 1), dt.date(2026, 1, 31))
    assert len(detections) == 2
