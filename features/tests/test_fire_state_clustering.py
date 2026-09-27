"""Tests de clustering espaciotemporal de detecciones FIRMS en eventos."""
import datetime as dt

from features.fire_state.clustering import (
    DEFAULT_SPATIAL_EPS_M,
    DEFAULT_TEMPORAL_EPS,
    FireEvent,
    build_fire_events,
    cluster_detections,
)
from pyproj import Transformer
from shared.schemas import FireDetection

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)


def _det(lat: float, lon: float, at: dt.datetime) -> FireDetection:
    return FireDetection(
        latitude=lat, longitude=lon, detected_at=at, frp=1.0,
        confidence="n", satellite="N", instrument="VIIRS",
    )


def _det_offset_m(base: FireDetection, dx: float, dy: float, at: dt.datetime) -> FireDetection:
    x, y = _TO_UTM.transform(base.longitude, base.latitude)
    lon, lat = _TO_WGS84.transform(x + dx, y + dy)
    return _det(lat, lon, at)


def test_two_fires_separated_in_space_and_time_are_distinct_events():
    base = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    fire_a = [_det(-37.0, -72.0, base), _det(-37.001, -72.001, base + dt.timedelta(hours=6))]
    fire_b = [_det(-39.0, -71.0, base), _det(-39.001, -71.001, base + dt.timedelta(hours=6))]
    labels = cluster_detections(fire_a + fire_b)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_contiguous_detections_chain_into_one_event_even_if_endpoints_alone_would_not_merge():
    # day1 y day3 quedan a ~1400 m entre sí (por encima de spatial_eps
    # =750 m), pero day2 está a ~700 m de ambos -- deben unirse los 3 vía
    # transitividad de union-find, no solo comparación directa por pares.
    base = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    day1 = _det(-37.0, -72.0, base)
    day2 = _det_offset_m(day1, dx=700.0, dy=0.0, at=base + dt.timedelta(days=1))
    day3 = _det_offset_m(day1, dx=1400.0, dy=0.0, at=base + dt.timedelta(days=2))
    labels = cluster_detections([day1, day2, day3], spatial_eps_m=750.0)
    assert labels[0] == labels[1] == labels[2]


def test_detections_beyond_temporal_eps_are_distinct_events_even_if_colocated():
    base = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.UTC)
    same_spot_early = _det(-37.0, -72.0, base)
    same_spot_late = _det(-37.0, -72.0, base + dt.timedelta(days=30))
    labels = cluster_detections(
        [same_spot_early, same_spot_late], temporal_eps=dt.timedelta(days=2)
    )
    assert labels[0] != labels[1]


def test_duplicate_detection_same_instant_same_coordinates_does_not_raise():
    at = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.UTC)
    det = _det(-37.0, -72.0, at)
    labels = cluster_detections([det, det])
    assert labels[0] == labels[1]


def test_clustering_handles_new_year_boundary_correctly():
    # dt.timedelta sobre datetimes timezone-aware maneja el rollover de
    # año de forma nativa -- este test lo pin-ea explícitamente en vez de
    # confiar en que "debería funcionar".
    dec_31 = dt.datetime(2025, 12, 31, 23, 0, tzinfo=dt.UTC)
    jan_1 = dt.datetime(2026, 1, 1, 1, 0, tzinfo=dt.UTC)  # 2 horas después
    labels = cluster_detections(
        [_det(-37.0, -72.0, dec_31), _det(-37.001, -72.001, jan_1)],
        temporal_eps=dt.timedelta(days=2),
    )
    assert labels[0] == labels[1]


def test_defaults_are_the_documented_values():
    assert DEFAULT_SPATIAL_EPS_M == 750.0
    assert DEFAULT_TEMPORAL_EPS == dt.timedelta(days=2)


def test_build_fire_events_groups_detections_and_exposes_date_range():
    base = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    fire_a = [_det(-37.0, -72.0, base), _det(-37.001, -72.001, base + dt.timedelta(days=1))]
    fire_b = [_det(-39.0, -71.0, base)]
    events = build_fire_events(fire_a + fire_b)
    assert len(events) == 2
    assert all(isinstance(e, FireEvent) for e in events)
    sizes = sorted(len(e.detections) for e in events)
    assert sizes == [1, 2]
    two_detection_event = next(e for e in events if len(e.detections) == 2)
    assert two_detection_event.start_date == base.date()
    assert two_detection_event.end_date == (base + dt.timedelta(days=1)).date()


def test_single_detection_event_start_and_end_date_are_the_same_day():
    at = dt.datetime(2026, 3, 1, 8, 0, tzinfo=dt.UTC)
    events = build_fire_events([_det(-37.0, -72.0, at)])
    assert len(events) == 1
    assert events[0].start_date == events[0].end_date == at.date()
