"""Definición de "evento de incendio" como clúster espaciotemporal de
detecciones FIRMS — ver docs/fire-events.md para la justificación completa
de parámetros y la comparación con kriging (WildfireCube).

Algoritmo: unión de conjuntos (union-find) sobre una relación de vecindad
— "las detecciones A y B pertenecen al mismo evento si están a lo sumo
`spatial_eps_m` de distancia (haversine) Y a lo sumo `temporal_eps` de
diferencia temporal" — equivalente a DBSCAN con `min_samples=1` y una
métrica precomputada que combina distancia espacial y temporal, pero
implementado directamente en vez de agregar `scikit-learn` como
dependencia nueva de `features` (ver docs/decisions.md): para el volumen
de detecciones de este proyecto (unos pocos miles por temporada en el
área de estudio) un O(n²) de unión de conjuntos es más que suficiente.

Es intencionalmente transitivo/por cadena ("chaining"): si A-B están
cerca y B-C están cerca, A y C quedan en el mismo evento aunque A-C por
sí solas no lo estén — esto modela un incendio que se mueve/crece de
forma continua en el tiempo, no una bola fija alrededor de un punto.
"""
import datetime as dt
import math
from dataclasses import dataclass

from shared.schemas import FireDetection

_EARTH_RADIUS_M = 6_371_000.0

DEFAULT_SPATIAL_EPS_M = 750.0
# ~2x el tamaño de píxel nominal de VIIRS (375 m, CLAUDE.md) -- une
# detecciones contiguas del mismo incendio sin fusionar focos separados
# por más de un par de píxeles VIIRS.
DEFAULT_TEMPORAL_EPS = dt.timedelta(days=2)
# VIIRS revisita el área ~1 vez/día; 2 días tolera un día de nubosidad
# perdido sin fusionar incendios de episodios distintos.


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    # min(1.0, ...) evita un ValueError de asin por overshoot de punto
    # flotante cuando a es minúsculamente > 1.0 (dos detecciones idénticas).
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b


def cluster_detections(
    detections: list[FireDetection],
    spatial_eps_m: float = DEFAULT_SPATIAL_EPS_M,
    temporal_eps: dt.timedelta = DEFAULT_TEMPORAL_EPS,
) -> list[int]:
    """labels[i] = id de evento para detections[i] — nunca -1: toda
    detección pertenece a algún evento, aunque sea de un solo elemento."""
    n = len(detections)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            time_diff = abs(detections[i].detected_at - detections[j].detected_at)
            if time_diff > temporal_eps:
                continue
            distance_m = _haversine_m(
                detections[i].latitude, detections[i].longitude,
                detections[j].latitude, detections[j].longitude,
            )
            if distance_m <= spatial_eps_m:
                uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    # Re-etiquetar raíces (arbitrarias) a ids consecutivos 0..k-1 en orden
    # de primera aparición -- resultado determinista, no depende del valor
    # interno de cada raíz de union-find.
    label_by_root: dict[int, int] = {}
    labels: list[int] = []
    for root in roots:
        if root not in label_by_root:
            label_by_root[root] = len(label_by_root)
        labels.append(label_by_root[root])
    return labels


@dataclass(frozen=True)
class FireEvent:
    event_id: int
    detections: tuple[FireDetection, ...]

    @property
    def start_date(self) -> dt.date:
        # anotación explícita de la lista intermedia: sin ella, mypy
        # --strict infiere `Any` para el resultado de `min()` sobre el
        # generador y se queja de "Returning Any from function declared
        # to return date".
        dates: list[dt.date] = [d.detected_at.date() for d in self.detections]
        return min(dates)

    @property
    def end_date(self) -> dt.date:
        dates: list[dt.date] = [d.detected_at.date() for d in self.detections]
        return max(dates)


def build_fire_events(
    detections: list[FireDetection],
    spatial_eps_m: float = DEFAULT_SPATIAL_EPS_M,
    temporal_eps: dt.timedelta = DEFAULT_TEMPORAL_EPS,
) -> list[FireEvent]:
    labels = cluster_detections(detections, spatial_eps_m, temporal_eps)
    grouped: dict[int, list[FireDetection]] = {}
    for label, detection in zip(labels, detections, strict=True):
        grouped.setdefault(label, []).append(detection)
    return [
        FireEvent(event_id=label, detections=tuple(dets))
        for label, dets in sorted(grouped.items())
    ]
