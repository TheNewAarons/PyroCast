"""Tests del cliente FIRMS: cero llamadas de red reales (usa `responses`)."""
import datetime as dt
from pathlib import Path

import pytest
import responses
from ingestion.firms.client import FirmsApiError, FirmsClient

FIXTURES = Path(__file__).parent / "fixtures"
OK_CSV = (FIXTURES / "firms_area_ok.csv").read_text()
ERROR_BODY = (FIXTURES / "firms_area_error.txt").read_text()

BBOX = (-73.7, -39.3, -71.0, -36.5)  # west, south, east, north


def _url(map_key: str = "test-key", sensor: str = "VIIRS_SNPP_NRT", day_range: int = 5,
          date: str | None = "2026-01-15") -> str:
    coords = f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"
    base = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{sensor}/{coords}/{day_range}"
    return f"{base}/{date}" if date else base


@responses.activate
def test_fetch_area_csv_returns_raw_text_on_success():
    responses.add(responses.GET, _url(), body=OK_CSV, status=200)
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    result = client.fetch_area_csv(
        BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15)
    )
    assert result == OK_CSV


@responses.activate
def test_fetch_area_csv_rejects_day_range_above_five():
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    with pytest.raises(ValueError, match="day_range"):
        client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=6, date=dt.date(2026, 1, 15))


@responses.activate
def test_fetch_area_csv_retries_on_429_then_succeeds():
    responses.add(responses.GET, _url(), status=429)
    responses.add(responses.GET, _url(), body=OK_CSV, status=200)
    sleeps: list[float] = []
    # min_request_interval_seconds=0 isolates retry/backoff from rate
    # limiting (a separate concern, covered by its own test below) — with
    # a non-zero interval, the no-op sleep_fn doesn't actually advance
    # real time, so _rate_limit would also fire a sleep before the retry.
    client = FirmsClient(
        map_key="test-key", max_retries=3, sleep_fn=sleeps.append, min_request_interval_seconds=0
    )
    result = client.fetch_area_csv(
        BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15)
    )
    assert result == OK_CSV
    assert len(sleeps) == 1  # one retry happened, one backoff sleep


@responses.activate
def test_fetch_area_csv_retries_on_500_then_raises_after_exhausting_retries():
    for _ in range(4):
        responses.add(responses.GET, _url(), status=500)
    client = FirmsClient(map_key="test-key", max_retries=3, sleep_fn=lambda _seconds: None)
    with pytest.raises(FirmsApiError, match="500"):
        client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15))


@responses.activate
def test_fetch_area_csv_raises_on_error_body_without_retrying():
    responses.add(responses.GET, _url(), body=ERROR_BODY, status=200)
    calls = {"n": 0}

    def _counting_sleep(_seconds: float) -> None:
        calls["n"] += 1

    client = FirmsClient(map_key="test-key", max_retries=3, sleep_fn=_counting_sleep)
    with pytest.raises(FirmsApiError, match="Invalid MAP_KEY"):
        client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15))
    assert calls["n"] == 0  # no retry for a bad-key body — retrying can't fix it


@responses.activate
def test_fetch_area_csv_rate_limits_between_requests():
    responses.add(responses.GET, _url(date="2026-01-15"), body=OK_CSV, status=200)
    responses.add(responses.GET, _url(date="2026-01-16"), body=OK_CSV, status=200)
    ticks = iter([0.0, 0.05, 0.05])  # third call is the post-sleep re-check
    monotonic = lambda: next(ticks, 10.0)  # noqa: E731
    sleeps: list[float] = []
    client = FirmsClient(
        map_key="test-key",
        min_request_interval_seconds=0.2,
        sleep_fn=sleeps.append,
        monotonic_fn=monotonic,
    )
    client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 15))
    client.fetch_area_csv(BBOX, sensor="VIIRS_SNPP_NRT", day_range=5, date=dt.date(2026, 1, 16))
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.15, abs=0.01)  # 0.2 - (0.05 - 0.0)


def test_fetch_range_chunks_into_windows_of_at_most_five_days():
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    chunks = client._chunk_date_range(dt.date(2026, 1, 1), dt.date(2026, 1, 12))
    assert chunks == [
        (dt.date(2026, 1, 1), dt.date(2026, 1, 5)),
        (dt.date(2026, 1, 6), dt.date(2026, 1, 10)),
        (dt.date(2026, 1, 11), dt.date(2026, 1, 12)),
    ]


def test_fetch_range_chunks_single_day():
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    chunks = client._chunk_date_range(dt.date(2026, 1, 1), dt.date(2026, 1, 1))
    assert chunks == [(dt.date(2026, 1, 1), dt.date(2026, 1, 1))]


@responses.activate
def test_fetch_range_makes_one_request_per_chunk_and_returns_all():
    responses.add(responses.GET, _url(date="2026-01-01", day_range=5), body=OK_CSV, status=200)
    responses.add(responses.GET, _url(date="2026-01-06", day_range=2), body=OK_CSV, status=200)
    client = FirmsClient(map_key="test-key", sleep_fn=lambda _seconds: None)
    results = client.fetch_range(
        BBOX, sensor="VIIRS_SNPP_NRT", start=dt.date(2026, 1, 1), end=dt.date(2026, 1, 7)
    )
    assert [(s, e) for s, e, _ in results] == [
        (dt.date(2026, 1, 1), dt.date(2026, 1, 5)),
        (dt.date(2026, 1, 6), dt.date(2026, 1, 7)),
    ]
    assert all(raw == OK_CSV for _, _, raw in results)
