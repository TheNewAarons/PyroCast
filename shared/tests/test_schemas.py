"""Tests de shared.schemas: el modelo normalizado de detecciones de fuego."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from shared.schemas import FireDetection


def test_fire_detection_accepts_valid_fields():
    detection = FireDetection(
        latitude=-37.4689,
        longitude=-72.3524,
        detected_at=datetime(2026, 1, 15, 5, 12, tzinfo=timezone.utc),
        frp=12.3,
        confidence="n",
        satellite="N",
        instrument="VIIRS",
    )
    assert detection.latitude == -37.4689
    assert detection.frp == 12.3
    assert detection.confidence == "n"


def test_fire_detection_allows_frp_none():
    detection = FireDetection(
        latitude=-37.4689,
        longitude=-72.3524,
        detected_at=datetime(2026, 1, 15, 5, 12, tzinfo=timezone.utc),
        frp=None,
        confidence="low",
        satellite="Terra",
        instrument="MODIS",
    )
    assert detection.frp is None


def test_fire_detection_rejects_out_of_range_latitude():
    with pytest.raises(ValidationError):
        FireDetection(
            latitude=95.0,
            longitude=-72.3524,
            detected_at=datetime(2026, 1, 15, 5, 12, tzinfo=timezone.utc),
            frp=None,
            confidence="n",
            satellite="N",
            instrument="VIIRS",
        )
