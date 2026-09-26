"""Tests de persistencia cruda de FIRMS: Parquet + metadatos, sin red."""
import datetime as dt
import json

import pyarrow.parquet as pq
from ingestion.firms.storage import save_raw_response

RAW_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "-37.4689,-72.3524,335.2,0.42,0.39,2026-01-15,0005,N,VIIRS,n,2.0NRT,289.1,12.3,N\n"
)
BBOX = (-73.7, -39.3, -71.0, -36.5)


def test_save_raw_response_writes_parquet_partitioned_by_download_date(tmp_path):
    downloaded_at = dt.datetime(2026, 1, 20, 10, 30, tzinfo=dt.UTC)
    parquet_path = save_raw_response(
        raw_csv_text=RAW_CSV,
        query_start=dt.date(2026, 1, 15),
        query_end=dt.date(2026, 1, 15),
        bbox=BBOX,
        sensor="VIIRS_SNPP_NRT",
        downloaded_at=downloaded_at,
        base_dir=tmp_path,
    )
    assert parquet_path.exists()
    assert parquet_path.parent == tmp_path / "firms" / "download_date=2026-01-20"

    table = pq.read_table(parquet_path)
    assert table.num_rows == 1
    assert "latitude" in table.column_names
    # Se guarda tal cual (string), sin coerción de tipos — la
    # normalización a tipos numéricos ocurre en el parser, no aquí.
    assert table.column("latitude")[0].as_py() == "-37.4689"


def test_save_raw_response_writes_metadata_sidecar(tmp_path):
    downloaded_at = dt.datetime(2026, 1, 20, 10, 30, tzinfo=dt.UTC)
    parquet_path = save_raw_response(
        raw_csv_text=RAW_CSV,
        query_start=dt.date(2026, 1, 15),
        query_end=dt.date(2026, 1, 16),
        bbox=BBOX,
        sensor="VIIRS_SNPP_NRT",
        downloaded_at=downloaded_at,
        base_dir=tmp_path,
    )
    meta_path = parquet_path.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["query_start"] == "2026-01-15"
    assert meta["query_end"] == "2026-01-16"
    assert meta["bbox"] == list(BBOX)
    assert meta["sensor"] == "VIIRS_SNPP_NRT"
    assert meta["row_count"] == 1


def test_save_raw_response_handles_empty_result_without_raising(tmp_path):
    empty_csv = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
    )
    downloaded_at = dt.datetime(2026, 1, 20, 10, 30, tzinfo=dt.UTC)
    parquet_path = save_raw_response(
        raw_csv_text=empty_csv,
        query_start=dt.date(2026, 1, 15),
        query_end=dt.date(2026, 1, 15),
        bbox=BBOX,
        sensor="VIIRS_SNPP_NRT",
        downloaded_at=downloaded_at,
        base_dir=tmp_path,
    )
    table = pq.read_table(parquet_path)
    assert table.num_rows == 0
    meta = json.loads(parquet_path.with_suffix(".meta.json").read_text())
    assert meta["row_count"] == 0
