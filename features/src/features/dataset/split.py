"""Split train/val/test reproducible, POR EVENTO -- nunca por píxel ni
por día dentro de un mismo evento (evitaría fuga de datos: días
consecutivos del mismo incendio son casi idénticos, y modelo entrenado
con un día y evaluado con el día siguiente del MISMO evento mediría
memorización, no generalización).

*** Separar por evento NO basta *** (revisión independiente, docs/review.md,
hallazgo C1): el clustering FIRMS puede partir UN mismo complejo de
incendios en varios "eventos" contiguos que caen en splits distintos
(medido: dos eventos de val tenían la extensión de fuego pegada a un evento
de test, con los mismos días). `split_events_grouped` agrupa los eventos
acoplados espaciotemporalmente y reparte GRUPOS, no eventos;
`find_split_leakage` audita un split ya hecho. `split_events` (por evento
suelto) queda solo para uso sin información de ubicación y fechas."""
import datetime as dt
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from pyproj import Transformer

DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_VAL_FRAC = 0.15
# el resto (0.15 por defecto) va a test.


def split_events(
    event_ids: list[int],
    seed: int = 42,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> dict[str, list[int]]:
    # ordenar antes de mezclar: el resultado depende solo del CONJUNTO de
    # ids y de la semilla, nunca del orden en que la lista de entrada los
    # trae.
    ids_sorted = sorted(event_ids)
    rng = random.Random(seed)
    shuffled = ids_sorted[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n < 3:
        # sin eventos suficientes para un split con sentido -- todo a
        # train, explícito, no un val/test vacío "por accidente" del
        # redondeo (encontrado en la revisión final del 2026-09-27: antes
        # de este fix, n=3..5 podían dejar val o test vacíos sin avisar,
        # contradiciendo el 70/15/15 documentado en docs/dataset-card.md).
        return {"train": shuffled, "val": [], "test": []}

    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * (1 - train_frac - val_frac)))
    while n_val + n_test >= n:
        if n_val >= n_test:
            n_val -= 1
        else:
            n_test -= 1
    n_train = n - n_val - n_test

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


DEFAULT_MAX_GAP_KM = 10.0
# ~ el tamaño de celda nativo de ERA5-Land (~9 km): dos incendios a menos de
# eso comparten (casi) el mismo clima de entrada, además de ser el mismo
# complejo de fuego o vecinos inmediatos.
DEFAULT_MAX_GAP_DAYS = 3
# un día más que `temporal_eps` del clustering (2 días): eventos con fuego en
# fechas casi consecutivas comparten el régimen sinóptico (p. ej. viento
# Puelche).

_KM_PER_DEG_LAT = 110.57
_KM_PER_DEG_LON_EQUATOR = 111.32


@dataclass(frozen=True)
class EventFootprint:
    """Dónde (bbox WGS84 de la extensión de fuego) y cuándo (primer y último
    día con fuego) estuvo activo un evento."""

    west: float
    south: float
    east: float
    north: float
    first_day: dt.date
    last_day: dt.date


def footprint_from_fire_event(event: Any) -> EventFootprint:
    """Huella desde las detecciones de un `FireEvent` (usado por
    `build-dataset`, que ya tiene los eventos clusterizados en memoria)."""
    lons = [d.longitude for d in event.detections]
    lats = [d.latitude for d in event.detections]
    return EventFootprint(min(lons), min(lats), max(lons), max(lats),
                          event.start_date, event.end_date)


def footprint_from_tensor(tensor: xr.DataArray) -> EventFootprint:
    """Huella desde un tensor de evento ya construido: extensión de las celdas
    con fuego en cualquier día, y primer/último día con fuego."""
    channels = list(tensor.coords["channel"].values)
    fire = tensor.values[:, channels.index("fire_mask")] >= 0.5
    rows, cols = np.where(fire.any(axis=0))
    if rows.size == 0:
        raise ValueError(
            f"El evento {tensor.attrs.get('event_id')} no tiene ninguna celda en llamas."
        )
    xs, ys = tensor.coords["x"].values, tensor.coords["y"].values
    to_wgs84 = Transformer.from_crs(str(tensor.attrs.get("crs", "EPSG:32719")), "EPSG:4326",
                                    always_xy=True)
    west, south = to_wgs84.transform(xs[cols.min()], ys[rows.max()])
    east, north = to_wgs84.transform(xs[cols.max()], ys[rows.min()])
    days = [dt.date.fromisoformat(str(d)[:10]) for d in tensor.coords["day"].values]
    fire_days = [days[i] for i in range(len(days)) if fire[i].any()]
    return EventFootprint(float(west), float(south), float(east), float(north),
                          fire_days[0], fire_days[-1])


def footprint_gap_km(a: EventFootprint, b: EventFootprint) -> float:
    """Distancia mínima (km) entre los bbox de dos huellas (0 si se tocan o se
    solapan). Aproximación plana local: sobra para umbrales de ~10 km."""
    mean_lat = math.radians((a.south + a.north + b.south + b.north) / 4)
    dx_deg = max(0.0, max(a.west, b.west) - min(a.east, b.east))
    dy_deg = max(0.0, max(a.south, b.south) - min(a.north, b.north))
    return math.hypot(dx_deg * _KM_PER_DEG_LON_EQUATOR * math.cos(mean_lat),
                      dy_deg * _KM_PER_DEG_LAT)


def footprint_gap_days(a: EventFootprint, b: EventFootprint) -> int:
    """Días entre los intervalos de fuego de dos eventos (0 si se solapan)."""
    return max(0, (max(a.first_day, b.first_day) - min(a.last_day, b.last_day)).days)


def _coupled(a: EventFootprint, b: EventFootprint, max_gap_km: float, max_gap_days: int) -> bool:
    return footprint_gap_km(a, b) <= max_gap_km and footprint_gap_days(a, b) <= max_gap_days


def group_events(
    footprints: Mapping[int, EventFootprint],
    max_gap_km: float = DEFAULT_MAX_GAP_KM,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> list[list[int]]:
    """Componentes conexas de la relación "acoplados" (transitiva: A-B y B-C
    acoplados dejan a A, B y C en el mismo grupo). Cada grupo y la lista de
    grupos salen ordenados, así el resultado no depende del orden de entrada."""
    ids = sorted(footprints)
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pos, i in enumerate(ids):
        for j in ids[pos + 1:]:
            if _coupled(footprints[i], footprints[j], max_gap_km, max_gap_days):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def split_events_grouped(
    footprints: Mapping[int, EventFootprint],
    seed: int = 42,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    max_gap_km: float = DEFAULT_MAX_GAP_KM,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> dict[str, list[int]]:
    """Reparte GRUPOS de eventos acoplados (nunca un grupo entre dos splits).
    Los objetivos de tamaño (en eventos) son los de `split_events`; como un
    grupo es indivisible, el tamaño real puede desviarse del objetivo. Exige
    al menos 3 grupos independientes: con menos, no hay forma de armar
    train/val/test sin fuga y se levanta `ValueError` (no se degrada en
    silencio)."""
    groups = group_events(footprints, max_gap_km, max_gap_days)
    if len(groups) < 3:
        raise ValueError(
            f"Solo {len(groups)} grupos independientes de eventos (acoplamiento "
            f"<= {max_gap_km} km y <= {max_gap_days} días): no se puede formar "
            f"train/val/test sin fuga entre splits. Hacen falta más eventos o un "
            f"umbral de acoplamiento menor."
        )
    n = sum(len(g) for g in groups)
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * (1 - train_frac - val_frac)))
    rng = random.Random(seed)
    remaining = groups[:]
    rng.shuffle(remaining)

    def take(need: int) -> list[int]:
        # el primer grupo (en orden aleatorio) que cabe en lo que falta; si
        # ninguno cabe, el más chico -- así el tamaño real se acerca al objetivo.
        fits = [g for g in remaining if len(g) <= need]
        chosen = fits[0] if fits else min(remaining, key=len)
        remaining.remove(chosen)
        return chosen

    test: list[int] = []
    while len(test) < n_test and len(remaining) > 2:  # dejar >=1 para val y >=1 para train
        test += take(n_test - len(test))
    val: list[int] = []
    while len(val) < n_val and len(remaining) > 1:
        val += take(n_val - len(val))
    train = [e for g in remaining for e in g]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def find_split_leakage(
    splits: Mapping[str, list[int]],
    footprints: Mapping[int, EventFootprint],
    max_gap_km: float = DEFAULT_MAX_GAP_KM,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> list[dict[str, Any]]:
    """Pares de eventos acoplados que quedaron en splits DISTINTOS (vacío =
    sin fuga según estos umbrales). Cada elemento trae ids, splits, distancia
    (km) y separación temporal (días)."""
    where = {e: s for s, ids in splits.items() for e in ids}
    ids = sorted(e for e in where if e in footprints)
    leaks: list[dict[str, Any]] = []
    for pos, i in enumerate(ids):
        for j in ids[pos + 1:]:
            if where[i] == where[j]:
                continue
            if _coupled(footprints[i], footprints[j], max_gap_km, max_gap_days):
                leaks.append({
                    "event_a": i, "split_a": where[i], "event_b": j, "split_b": where[j],
                    "gap_km": footprint_gap_km(footprints[i], footprints[j]),
                    "gap_days": footprint_gap_days(footprints[i], footprints[j]),
                })
    return leaks
