"""Tests del pipeline Sentinel-2: cache + orquestación, openEO mockeado."""
from ingestion.sentinel2.pipeline import fetch_sentinel2

BBOX = (-73.7, -39.3, -71.0, -36.5)


class _FakeSentinel2Client:
    def __init__(self):
        self.calls: list = []

    def fetch_monthly_composite(self, bbox, year, month, target, **kwargs):
        self.calls.append(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-composite")
        return target


def test_fetch_sentinel2_produces_composite(tmp_path):
    client = _FakeSentinel2Client()
    result = fetch_sentinel2(
        bbox=BBOX, year=2026, month=1, client=client, cache_dir=tmp_path / "cache"
    )
    assert result.exists()
    assert len(client.calls) == 1


def test_fetch_sentinel2_cache_hit_skips_fetch_entirely(tmp_path):
    client = _FakeSentinel2Client()
    first = fetch_sentinel2(
        bbox=BBOX, year=2026, month=1, client=client, cache_dir=tmp_path / "cache"
    )
    second = fetch_sentinel2(
        bbox=BBOX, year=2026, month=1, client=client, cache_dir=tmp_path / "cache"
    )
    assert second == first
    assert len(client.calls) == 1  # no segunda llamada


def test_fetch_sentinel2_different_month_is_a_cache_miss(tmp_path):
    client = _FakeSentinel2Client()
    fetch_sentinel2(bbox=BBOX, year=2026, month=1, client=client, cache_dir=tmp_path / "cache")
    fetch_sentinel2(bbox=BBOX, year=2026, month=2, client=client, cache_dir=tmp_path / "cache")
    assert len(client.calls) == 2
