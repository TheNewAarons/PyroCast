"""Rasterización de un `FireEvent` a una máscara binaria diaria de
EXTENSIÓN ACTIVA DE FUEGO (no superficie quemada acumulada — cada día es
independiente, una celda que ardió ayer y no hoy vuelve a `False`; ver
docs/fire-events.md) sobre la grilla de trabajo.

*** SIMPLIFICACIÓN EXPLÍCITA frente a la literatura: WildfireCube
reconstruye la superficie quemada mediante kriging espaciotemporal sobre
las detecciones. Este proyecto usa en cambio (a) un buffer espacial FIJO
alrededor de cada detección puntual (nunca una interpolación
geoestadística) y (b) interpolación temporal LINEAL del indicador binario
entre días con detección para rellenar días sin observación dentro del
rango del evento — nunca extrapola fuera de ese rango. Ver
docs/limitations.md. ***
"""
import datetime as dt

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import Point

from features.fire_state.clustering import FireEvent
from features.grid.grid import WorkGrid

DEFAULT_BUFFER_M = 375.0
# RADIO del buffer, no el tamaño de píxel -- un círculo de 375 m de radio
# cubre ~441 786 m², ~3.1x el área de un píxel VIIRS de 375x375 m
# (140 625 m²). Es una sobre-cobertura DELIBERADA (no un intento fallido
# de igualar el área del píxel): compensa la incertidumbre real de
# geolocalización del sensor y el crecimiento del píxel fuera de nadir,
# ninguno de los dos modelado explícitamente aquí. Sin calibrar contra
# incendios reales de Chile -- ver docs/fire-events.md y
# docs/limitations.md.


def rasterize_daily_masks(
    event: FireEvent, grid: WorkGrid, buffer_m: float = DEFAULT_BUFFER_M
) -> dict[dt.date, np.ndarray]:
    """Una máscara booleana (grid.height, grid.width) por cada día CON
    detección en el evento — un círculo de radio `buffer_m` (metros, en
    `grid.crs`) alrededor de cada detección de ese día."""
    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    geoms_by_day: dict[dt.date, list[Point]] = {}
    for detection in event.detections:
        day = detection.detected_at.date()
        x, y = transformer.transform(detection.longitude, detection.latitude)
        geoms_by_day.setdefault(day, []).append(Point(x, y).buffer(buffer_m))

    masks: dict[dt.date, np.ndarray] = {}
    for day, geoms in geoms_by_day.items():
        raw = rasterize(
            [(geom, 1) for geom in geoms],
            out_shape=(grid.height, grid.width),
            transform=grid.transform,
            fill=0,
            dtype="uint8",
        )
        mask = raw.astype(bool)
        if not mask.any():
            # Un día CON detecciones reales que termina sin ningún píxel
            # marcado (buffer_m demasiado chico frente al tamaño de
            # celda, detección fuera de los bounds de la grilla, o una
            # reproyección degenerada) es indistinguible de "no hubo
            # fuego real" si se deja pasar en silencio -- el mismo patrón
            # de nodata fabricado/silencioso ya encontrado varias veces en
            # este repo. Fallar ruidosamente en vez de devolver una
            # máscara toda-False.
            raise ValueError(
                f"El día {day.isoformat()} tiene {len(geoms)} detección(es) pero "
                f"ningún píxel quedó marcado tras rasterizar -- revisar buffer_m "
                f"(demasiado chico frente a la resolución de la grilla) o si las "
                f"detecciones caen fuera de los bounds de la grilla."
            )
        masks[day] = mask
    return masks


def fill_temporal_gaps(
    masks_by_day: dict[dt.date, np.ndarray], start_date: dt.date, end_date: dt.date
) -> dict[dt.date, np.ndarray]:
    """Para cada día en `[start_date, end_date]` sin máscara propia,
    interpola LINEALMENTE el indicador binario entre el día ancla anterior
    y el siguiente con detección, marcando como "fuego" todo lo que el
    valor interpolado sea > 0 — equivalente a la unión de las máscaras
    ancla anterior/siguiente para ese píxel (un 0/1 interpolado
    linealmente entre dos anclas solo da 0 si AMBAS anclas son 0). Un día
    sin ancla en algún lado (antes de la primera detección del evento, o
    después de la última) se deja vacío — nunca se extrapola."""
    anchor_days = sorted(masks_by_day)
    if not anchor_days:
        return {}
    shape = next(iter(masks_by_day.values())).shape

    filled: dict[dt.date, np.ndarray] = {}
    total_days = (end_date - start_date).days + 1
    for offset in range(total_days):
        day = start_date + dt.timedelta(days=offset)
        if day in masks_by_day:
            filled[day] = masks_by_day[day]
            continue

        earlier = [a for a in anchor_days if a < day]
        later = [a for a in anchor_days if a > day]
        if not earlier or not later:
            filled[day] = np.zeros(shape, dtype=bool)
            continue

        prev_day, next_day = earlier[-1], later[0]
        prev_mask, next_mask = masks_by_day[prev_day], masks_by_day[next_day]
        span_days = (next_day - prev_day).days
        weight = (day - prev_day).days / span_days
        interpolated = (1 - weight) * prev_mask.astype(np.float64) + weight * next_mask.astype(
            np.float64
        )
        filled[day] = interpolated > 0.0
    return filled


def build_fire_state(
    event: FireEvent, grid: WorkGrid, buffer_m: float = DEFAULT_BUFFER_M
) -> dict[dt.date, np.ndarray]:
    daily_masks = rasterize_daily_masks(event, grid, buffer_m)
    return fill_temporal_gaps(daily_masks, event.start_date, event.end_date)
