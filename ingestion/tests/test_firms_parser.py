"""Tests del parser FIRMS → shared.schemas.FireDetection."""
import datetime as dt
from pathlib import Path

from ingestion.firms.parser import parse_csv_to_detections

FIXTURES = Path(__file__).parent / "fixtures"
OK_CSV = (FIXTURES / "firms_area_ok.csv").read_text()


def test_parse_csv_to_detections_normalizes_all_rows():
    detections = parse_csv_to_detections(OK_CSV)
    assert len(detections) == 3
    first = detections[0]
    assert first.latitude == -37.4689
    assert first.longitude == -72.3524
    assert first.frp == 12.3
    assert first.confidence == "n"
    assert first.satellite == "N"
    assert first.instrument == "VIIRS"


def test_parse_csv_to_detections_combines_date_and_zero_padded_time_utc():
    detections = parse_csv_to_detections(OK_CSV)
    # acq_date=2026-01-15, acq_time="0005" -> 00:05 UTC, not 05:00 and not
    # dropped-leading-zero "5" minutes.
    assert detections[0].detected_at == dt.datetime(2026, 1, 15, 0, 5, tzinfo=dt.UTC)


def test_parse_csv_to_detections_handles_empty_result_without_raising():
    empty_csv = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
    )
    assert parse_csv_to_detections(empty_csv) == []
