"""Adaptador de 'Next Day Wildfire Spread' (NDWS, Huot et al.) al
esquema de tensor de PyroCast (`features.dataset.assemble.CHANNEL_ORDER`)
-- ver docs/public-dataset.md para el contexto completo (de dónde se
obtiene, licencia, formato) y la tabla resumen de mapeo. Este docstring
es la fuente de verdad de las fórmulas exactas; docs/public-dataset.md
la resume pero no la duplica en detalle, para que no se desincronicen.

Mapeo canal por canal (PyroCast <- NDWS):
    elevation          <- elevation                          (copia directa, metros)
    slope_deg,         <- derivado de elevation de NDWS vía
    aspect_deg            features.terrain.slope_aspect (Horn 1981),
                          cellsize=1000.0 (resolución nativa de NDWS)
    wind_u, wind_v     <- derivado de th (dirección, grados, convención
                          meteorológica "desde dónde sopla") + vs
                          (velocidad, m/s): u=-vs*sin(th), v=-vs*cos(th)
    temperature        <- mean(tmmn, tmmx)                    (Kelvin)
    relative_humidity  <- derivado de sph (humedad específica, kg/kg) +
                          temperature, vía la fórmula de presión de
                          vapor de la OMM + Magnus-Tetens (Alduchov &
                          Eskridge 1996, mismos coeficientes que
                          features.weather.derive.relative_humidity_approx),
                          asumiendo presión estándar a nivel del mar
                          (101325 Pa) -- NDWS no trae presión de
                          superficie real. Aproximación documentada.
    precipitation      <- pr / 1000.0                         (mm -> m)
    ndvi               <- NDVI / 10000.0, clip [-1, 1]
    fuel_type          <- SIN equivalente en NDWS -- relleno constante
                          99 (ingestion.worldcover.fuel_type.FUEL_TYPE_UNKNOWN,
                          duplicado aquí como literal -- models no
                          depende de ingestion, mismo patrón que
                          models/cellular_automata/rules.py)
    fire_mask          <- PrevFireMask, clip [0, 1] (-1 "incierto" se
                          trata conservadoramente como "sin fuego
                          observado")

Sin canal correspondiente en el tensor de 11 canales:
    FireMask (etiqueta día t+1) -> PublicDatasetSample.next_day_fire_mask,
    mismo clip [0, 1] que fire_mask.

Sin uso en absoluto (NDWS los trae, PyroCast no tiene dónde ponerlos ni
se derivan): pdsi, erc, population.

*** El WorkGrid/día que se construye para llamar a
assemble_event_tensor es NOMINAL, no geografía real -- NDWS son
recortes de EE.UU. continental, no de Chile. Solo resolution_m
(1000.0, la resolución real de NDWS) tiene significado; crs, el origen
de la transform, y la fecha de "day" son placeholders. Ver
docs/public-dataset.md. ***
"""
import datetime as dt
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine
from features.dataset.assemble import CHANNEL_ORDER, EventChannels, assemble_event_tensor
from features.grid.grid import WorkGrid
from features.terrain.slope_aspect import compute_slope_aspect

from models.deep.tfrecord_reader import read_tf_examples

_NDWS_RESOLUTION_M = 1000.0
_NDWS_STANDARD_PRESSURE_PA = 101325.0
_FUEL_TYPE_UNKNOWN = 99.0  # ingestion.worldcover.fuel_type.FUEL_TYPE_UNKNOWN, duplicado
_NOMINAL_DAY = dt.date(1970, 1, 1)  # NDWS no tiene fecha real por recorte -- ver docstring

_REQUIRED_NDWS_FEATURES = (
    "elevation", "pdsi", "NDVI", "pr", "sph", "th", "tmmn", "tmmx",
    "vs", "erc", "population", "PrevFireMask", "FireMask",
)


@dataclass(frozen=True)
class PublicDatasetSample:
    tensor: xr.DataArray
    next_day_fire_mask: np.ndarray


def _saturation_specific_humidity(
    temp_k: float, pressure_pa: float = _NDWS_STANDARD_PRESSURE_PA
) -> float:
    temp_c = temp_k - 273.15
    e_sat = 611.2 * math.exp((17.625 * temp_c) / (243.04 + temp_c))
    return 0.622 * e_sat / (pressure_pa - 0.378 * e_sat)


def _specific_humidity_to_relative_humidity(
    specific_humidity: np.ndarray,
    temp_k: np.ndarray,
    pressure_pa: float = _NDWS_STANDARD_PRESSURE_PA,
) -> np.ndarray:
    temp_c = temp_k - 273.15
    e_sat = 611.2 * np.exp((17.625 * temp_c) / (243.04 + temp_c))
    e = specific_humidity * pressure_pa / (0.622 + 0.378 * specific_humidity)
    rh = 100.0 * e / e_sat
    return np.clip(rh, 0.0, 100.0)


def transform_ndws_record(record: dict[str, np.ndarray], sample_id: int) -> PublicDatasetSample:
    missing = [name for name in _REQUIRED_NDWS_FEATURES if name not in record]
    if missing:
        raise ValueError(
            f"registro NDWS le faltan feature(s) requerida(s): {missing} -- "
            f"esperadas: {_REQUIRED_NDWS_FEATURES}."
        )

    elevation = record["elevation"].astype("float64")
    height, width = elevation.shape

    # sin esto, una feature con una forma distinta a `elevation` (p.
    # ej. un `th` de 1x1 en vez de la grilla completa) hace que numpy
    # la haga BROADCAST en silencio sobre toda la grilla más abajo --
    # un único valor "fabricado" repetido como si fuera un campo real.
    # Encontrado en la revisión final del 2026-09-28.
    mismatched = {
        name: record[name].shape
        for name in _REQUIRED_NDWS_FEATURES
        if record[name].shape != elevation.shape
    }
    if mismatched:
        raise ValueError(
            f"feature(s) NDWS con forma distinta a elevation {elevation.shape}: "
            f"{mismatched} -- todas las features de un registro deben cubrir la "
            f"misma grilla."
        )

    slope_deg, aspect_deg = compute_slope_aspect(elevation, _NDWS_RESOLUTION_M, _NDWS_RESOLUTION_M)

    th_rad = np.radians(record["th"].astype("float64"))
    vs = record["vs"].astype("float64")
    wind_u = -vs * np.sin(th_rad)
    wind_v = -vs * np.cos(th_rad)

    temperature = (record["tmmn"].astype("float64") + record["tmmx"].astype("float64")) / 2.0
    relative_humidity = _specific_humidity_to_relative_humidity(
        record["sph"].astype("float64"), temperature
    )
    precipitation = record["pr"].astype("float64") / 1000.0
    ndvi = np.clip(record["NDVI"].astype("float64") / 10000.0, -1.0, 1.0)
    fuel_type = np.full((height, width), _FUEL_TYPE_UNKNOWN, dtype="float64")
    fire_mask = np.clip(record["PrevFireMask"].astype("float64"), 0.0, 1.0)
    next_day_fire_mask = np.clip(record["FireMask"].astype("float64"), 0.0, 1.0)

    channels = EventChannels(
        days=(_NOMINAL_DAY,),
        static={
            "elevation": elevation,
            "slope_deg": slope_deg,
            "aspect_deg": aspect_deg,
            "fuel_type": fuel_type,
        },
        dynamic={
            "wind_u": {_NOMINAL_DAY: wind_u},
            "wind_v": {_NOMINAL_DAY: wind_v},
            "temperature": {_NOMINAL_DAY: temperature},
            "relative_humidity": {_NOMINAL_DAY: relative_humidity},
            "precipitation": {_NOMINAL_DAY: precipitation},
            "ndvi": {_NOMINAL_DAY: ndvi},
            "fire_mask": {_NOMINAL_DAY: fire_mask},
        },
    )
    grid = WorkGrid(
        crs="EPSG:32719",  # nominal -- ver docstring del módulo
        transform=Affine(_NDWS_RESOLUTION_M, 0.0, 0.0, 0.0, -_NDWS_RESOLUTION_M, 0.0),
        width=width,
        height=height,
        resolution_m=_NDWS_RESOLUTION_M,
    )
    tensor = assemble_event_tensor(channels, grid, event_id=sample_id)
    assert list(tensor.coords["channel"].values) == list(CHANNEL_ORDER)
    # `assemble_event_tensor` copia crs/transform de `grid` a los attrs
    # del tensor -- sin esto, un DataArray/Zarr de NDWS afirmaría en
    # silencio ser un recorte de Chile en EPSG:32719, cuando en
    # realidad es un chip de EE.UU. con un CRS/transform nominales (ver
    # docstring del módulo). Encontrado en la revisión final del
    # 2026-09-28.
    tensor.attrs["georeference"] = "nominal"
    tensor.attrs["source_dataset"] = "NDWS"

    return PublicDatasetSample(tensor=tensor, next_day_fire_mask=next_day_fire_mask)


def load_public_dataset_samples(paths: list[Path]) -> Iterator[PublicDatasetSample]:
    sample_id = 0
    for path in paths:
        for record in read_tf_examples(path):
            yield transform_ndws_record(record, sample_id=sample_id)
            sample_id += 1


def split_public_dataset(
    shard_paths: list[Path], seed: int = 42, train_frac: float = 0.85
) -> dict[str, list[Path]]:
    # split al nivel de ARCHIVO (shard), no de registro individual --
    # leer todo el dataset por adelantado solo para asignar un split
    # significaría leerlo dos veces; un shard es la unidad atómica
    # natural acá, igual que un evento lo es en
    # features/dataset/split.py.
    sorted_shards = sorted(shard_paths, key=str)
    rng = random.Random(seed)
    shuffled = sorted_shards[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n < 2:
        return {"train": shuffled, "val": []}

    n_val = max(1, round(n * (1 - train_frac)))
    n_val = min(n_val, n - 1)  # nunca deja train vacío
    return {"train": shuffled[: n - n_val], "val": shuffled[n - n_val :]}
