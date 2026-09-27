"""Tests de rasterización de eventos de incendio: buffer espacial +
interpolación temporal lineal (unión de máscaras ancla) sobre días sin
detección dentro del rango del evento."""
import datetime as dt

import numpy as np
from features.fire_state.clustering import FireEvent
from features.fire_state.rasterize import (
    build_fire_state,
    fill_temporal_gaps,
    rasterize_daily_masks,
)
from features.grid.grid import WorkGrid
from rasterio.transform import from_origin
from shared.schemas import FireDetection

# Grilla de 20x20 @ 250 m en EPSG:32719, con origen elegido a mano para
# que el punto (lat=-38.0, lon=-72.5) caiga en el píxel (10, 10) --
# verificado independientemente con pyproj antes de escribir el test
# (ver docs/superpowers/plans/2026-09-27-features-grid-fire-state.md).
_GRID = WorkGrid(
    crs="EPSG:32719", transform=from_origin(190000, 5791000, 250, 250),
    width=20, height=20, resolution_m=250.0,
)


def _det(lat: float, lon: float, at: dt.datetime) -> FireDetection:
    return FireDetection(
        latitude=lat, longitude=lon, detected_at=at, frp=1.0,
        confidence="n", satellite="N", instrument="VIIRS",
    )


def test_rasterize_daily_masks_known_coordinate_lands_on_expected_pixel():
    at = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.UTC)
    event = FireEvent(event_id=0, detections=(_det(-38.0, -72.5, at),))
    masks = rasterize_daily_masks(event, _GRID, buffer_m=300.0)
    assert list(masks.keys()) == [at.date()]
    mask = masks[at.date()]
    assert mask.shape == (20, 20)
    assert mask.dtype == np.bool_
    assert mask[10, 10]  # centro esperado -- verificado independientemente
    assert not mask[0, 0]  # esquina, a >3000 m del punto -- fuera del buffer


def test_fill_temporal_gaps_interpolates_as_union_of_neighboring_anchors():
    shape = (20, 20)
    day1 = dt.date(2026, 1, 1)
    day2 = dt.date(2026, 1, 2)  # día sin detección propia -- se rellena
    day3 = dt.date(2026, 1, 3)

    mask_day1 = np.zeros(shape, dtype=bool)
    mask_day1[5, 5] = True
    mask_day1[10, 10] = True  # presente en AMBAS anclas
    mask_day3 = np.zeros(shape, dtype=bool)
    mask_day3[8, 8] = True
    mask_day3[10, 10] = True  # presente en AMBAS anclas

    filled = fill_temporal_gaps({day1: mask_day1, day3: mask_day3}, day1, day3)

    assert set(filled.keys()) == {day1, day2, day3}
    assert filled[day1][5, 5] and not filled[day1][8, 8]
    assert filled[day3][8, 8] and not filled[day3][5, 5]
    # día intermedio: unión de ambas anclas (interpolación lineal del
    # indicador 0/1 nunca cruza 0 salvo que ambas anclas sean 0) -- un
    # píxel presente en AMBAS anclas debe resolver a True, no quedar en
    # un valor fraccionario sin umbralizar.
    assert filled[day2][10, 10]
    assert filled[day2][5, 5]
    assert filled[day2][8, 8]
    assert not filled[day2][0, 0]


def test_fill_temporal_gaps_never_extrapolates_past_a_single_sided_anchor():
    shape = (5, 5)
    day1 = dt.date(2026, 1, 5)  # única ancla, a mitad del rango pedido
    only_mask = np.zeros(shape, dtype=bool)
    only_mask[2, 2] = True

    filled = fill_temporal_gaps(
        {day1: only_mask}, dt.date(2026, 1, 3), dt.date(2026, 1, 7)
    )

    assert filled[day1][2, 2]
    for day in (dt.date(2026, 1, 3), dt.date(2026, 1, 4)):
        assert not np.any(filled[day])  # antes de la única ancla -- vacío
    for day in (dt.date(2026, 1, 6), dt.date(2026, 1, 7)):
        assert not np.any(filled[day])  # después de la única ancla -- vacío


def test_build_fire_state_combines_rasterize_and_fill_across_a_gap_day():
    day1 = dt.datetime(2026, 2, 1, 12, 0, tzinfo=dt.UTC)
    day3 = dt.datetime(2026, 2, 3, 12, 0, tzinfo=dt.UTC)  # deja el 2 sin detección
    event = FireEvent(
        event_id=0,
        detections=(_det(-38.0, -72.5, day1), _det(-38.0, -72.5, day3)),
    )
    masks = build_fire_state(event, _GRID, buffer_m=300.0)
    assert set(masks.keys()) == {day1.date(), dt.date(2026, 2, 2), day3.date()}
    assert masks[dt.date(2026, 2, 2)][10, 10]  # relleno del día intermedio
