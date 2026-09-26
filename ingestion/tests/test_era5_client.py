"""Tests del cliente ERA5: cdsapi.Client mockeado por completo, sin red."""
import datetime as dt
from pathlib import Path

import pytest
from ingestion.era5.client import (
    ERA5_VARIABLES,
    Era5Client,
    Era5RequestFailedError,
    Era5RequestTimeoutError,
    build_request,
)

BBOX = (-73.7, -39.3, -71.0, -36.5)  # west, south, east, north


def test_build_request_converts_bbox_to_cds_north_west_south_east_order():
    request = build_request(BBOX, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert request["area"] == [-36.5, -73.7, -39.3, -71.0]  # N, W, S, E


def test_build_request_includes_all_five_variables_by_default():
    request = build_request(BBOX, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert set(request["variable"]) == set(ERA5_VARIABLES)
    assert len(ERA5_VARIABLES) == 5


def test_build_request_uses_unarchived_netcdf_format():
    request = build_request(BBOX, dt.date(2026, 1, 15), dt.date(2026, 1, 15))
    assert request["data_format"] == "netcdf"
    assert request["download_format"] == "unarchived"


def test_build_request_spans_year_month_day_across_a_range():
    request = build_request(BBOX, dt.date(2026, 1, 30), dt.date(2026, 2, 2))
    assert request["year"] == ["2026"]
    assert request["month"] == ["01", "02"]
    assert set(request["day"]) == {"30", "31", "01", "02"}
    assert request["time"] == [f"{h:02d}:00" for h in range(24)]


class _FakeResultShapedRemote:
    """Simula cdsapi.api.Result: reply['state'] + update() + download()."""

    def __init__(self, states: list[str]):
        self._states = list(states)
        self.reply = {"state": self._states[0]}
        self.downloaded_to: str | None = None

    def update(self) -> None:
        if len(self._states) > 1:
            self._states.pop(0)
        self.reply = {"state": self._states[0]}

    def download(self, target: str) -> str:
        self.downloaded_to = target
        Path(target).write_bytes(b"fake-netcdf-bytes")
        return target


class _FakeRemoteShapedRemote:
    """Simula datastores.Remote: propiedad .status + update() + download()."""

    def __init__(self, states: list[str]):
        self._states = list(states)
        self.downloaded_to: str | None = None

    @property
    def status(self) -> str:
        return self._states[0]

    def update(self) -> None:
        if len(self._states) > 1:
            self._states.pop(0)

    def download(self, target: str) -> str:
        self.downloaded_to = target
        Path(target).write_bytes(b"fake-netcdf-bytes")
        return target


class _FakeCdsapiClient:
    def __init__(self, remote):
        self._remote = remote
        self.retrieve_calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, key: str, wait_until_complete: bool) -> "_FakeCdsapiClient":
        assert wait_until_complete is False
        return self

    def retrieve(self, dataset: str, request: dict):
        self.retrieve_calls.append((dataset, request))
        return self._remote


def test_download_hourly_succeeds_with_result_shaped_remote(tmp_path):
    remote = _FakeResultShapedRemote(["queued", "running", "completed"])
    factory = _FakeCdsapiClient(remote)
    client = Era5Client(url="https://x", key="k", client_factory=factory)
    target = tmp_path / "out.nc"
    result = client.download_hourly(
        "reanalysis-era5-land", {}, target, poll_interval_seconds=0, sleep_fn=lambda _s: None
    )
    assert result == target
    assert target.read_bytes() == b"fake-netcdf-bytes"


def test_download_hourly_succeeds_with_remote_shaped_remote(tmp_path):
    remote = _FakeRemoteShapedRemote(["accepted", "running", "successful"])
    factory = _FakeCdsapiClient(remote)
    client = Era5Client(url="https://x", key="k", client_factory=factory)
    target = tmp_path / "out.nc"
    result = client.download_hourly(
        "reanalysis-era5-land", {}, target, poll_interval_seconds=0, sleep_fn=lambda _s: None
    )
    assert result == target


def test_download_hourly_raises_on_failed_state_not_infinite_loop(tmp_path):
    remote = _FakeResultShapedRemote(["queued", "running", "failed"])
    factory = _FakeCdsapiClient(remote)
    client = Era5Client(url="https://x", key="k", client_factory=factory)
    with pytest.raises(Era5RequestFailedError, match="failed"):
        client.download_hourly(
            "reanalysis-era5-land", {}, tmp_path / "out.nc",
            poll_interval_seconds=0, sleep_fn=lambda _s: None,
        )


def test_download_hourly_raises_on_timeout_not_treated_as_success(tmp_path):
    remote = _FakeResultShapedRemote(["queued", "running", "running", "running"])
    factory = _FakeCdsapiClient(remote)
    client = Era5Client(url="https://x", key="k", client_factory=factory)
    ticks = iter([0.0, 1.0, 2.0, 100.0])  # jumps past timeout on the 4th check
    with pytest.raises(Era5RequestTimeoutError, match="running"):
        client.download_hourly(
            "reanalysis-era5-land", {}, tmp_path / "out.nc",
            poll_interval_seconds=0, timeout_seconds=10.0,
            sleep_fn=lambda _s: None, monotonic_fn=lambda: next(ticks),
        )
