"""Tests del parser FIRMS → shared.schemas.FireDetection."""
import datetime as dt
from pathlib import Path

import pytest
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


def test_parse_csv_to_detections_raises_clear_value_error_on_missing_column():
    # Falta la columna "instrument" por completo (p. ej. una fuente con un
    # esquema distinto) — debe fallar con un mensaje diagnosticable, no un
    # KeyError crudo.
    csv_without_instrument = (
        "latitude,longitude,acq_date,acq_time,satellite,confidence,frp\n"
        "-37.4689,-72.3524,2026-01-15,0005,N,n,12.3\n"
    )
    with pytest.raises(ValueError, match="instrument"):
        parse_csv_to_detections(csv_without_instrument)


def test_parse_csv_to_detections_handles_a_real_firms_time_with_dropped_leading_zeros():
    # Encontrado contra datos REALES de FIRMS (VIIRS_SNPP_SP, 2025-11-01,
    # revisión final del 2026-09-29): el Area API de FIRMS devuelve
    # acq_time como un campo NUMÉRICO -- el cero (o los ceros) a la
    # izquierda se pierden en la serialización a CSV. "517" en el CSV
    # real significa 05:17 UTC, no "51:7" (que ni siquiera es una hora
    # válida -- ValueError: hour must be in 0..23, que es exactamente lo
    # que pasaba antes de este fix). El comentario original de este
    # archivo asumía "siempre viene con cero a la izquierda" sin
    # verificarlo contra una respuesta real -- estaba mal.
    csv_with_dropped_leading_zeros = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
        "-37.4689,-72.3524,335.2,0.42,0.39,2025-11-01,517,N,VIIRS,n,2,307.6,1.4,D\n"
        "-37.4690,-72.3525,335.2,0.42,0.39,2025-11-01,5,N,VIIRS,n,2,307.6,1.4,D\n"
    )
    detections = parse_csv_to_detections(csv_with_dropped_leading_zeros)
    assert len(detections) == 2
    assert detections[0].detected_at == dt.datetime(2025, 11, 1, 5, 17, tzinfo=dt.UTC)
    assert detections[1].detected_at == dt.datetime(2025, 11, 1, 0, 5, tzinfo=dt.UTC)


def test_parse_csv_to_detections_handles_ragged_row_with_missing_frp_value():
    # Una fila truncada a mitad de columna (p. ej. corte de conexión) deja
    # csv.DictReader rellenando los valores faltantes con None, no "" —
    # row.get("frp", "").strip() fallaría con AttributeError sobre None.
    ragged_csv = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
        "-37.4689,-72.3524,335.2,0.42,0.39,2026-01-15,0005,N,VIIRS,n,2.0NRT\n"
    )
    detections = parse_csv_to_detections(ragged_csv)
    assert len(detections) == 1
    assert detections[0].frp is None
