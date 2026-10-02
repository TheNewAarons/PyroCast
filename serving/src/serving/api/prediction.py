"""Pipeline de /predict: punto de ignición + fecha + horizonte ->
tensor de features (reusando capas estáticas ya procesadas en disco,
clima/NDVI buscados por fecha) -> modelo -> GeoJSON de celdas.

Convención de probabilidad (heredada del backtest, docs/backtest-2026.md):
ACUMULADA -- "¿ha ardido esta celda alguna vez hasta el día d?". El día 0
es la ignición (estado conocido) y no se devuelve.

Nada se rellena ni se inventa: si falta clima para algún día requerido
por el modelo, o el punto cae fuera de la cobertura de los rasters, se
levanta un error explícito (`serving.api.errors`).
"""
import datetime as dt
import math
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine
from features.dataset.pipeline import (
    DEFAULT_CONTEXT_BUFFER_M,
    build_dataset_for_event,
    resolve_event_sources,
)
from features.fire_state.clustering import FireEvent
from pyproj import Transformer
from shared.config import Settings
from shared.schemas import FireDetection

from serving.api.cache import LRUCache
from serving.api.errors import (
    OutsideStudyAreaError,
    StaticLayersUnavailableError,
    TerrainCoverageError,
    WeatherUnavailableError,
)
from serving.api.model_registry import LoadedModel
from serving.api.schemas import (
    DayInfo,
    GridInfo,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    RequestEcho,
)

_WEATHER_FIELDS = ("wind_u", "wind_v", "temperature", "relative_humidity", "precipitation")
_CONTEXT_CELLS_BEYOND_HORIZON = 4
_PROB_DECIMALS = 4
_COORD_DECIMALS = 6

CacheKey = tuple[int, int, dt.date, int, str]


def _iso(days: list[dt.date]) -> list[str]:
    return [d.isoformat() for d in days]


class PredictionService:
    def __init__(
        self, settings: Settings, loaded: LoadedModel, cache: LRUCache[PredictResponse]
    ) -> None:
        self._settings = settings
        self._loaded = loaded
        self._cache = cache
        self._to_crs = Transformer.from_crs("EPSG:4326", settings.crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(settings.crs, "EPSG:4326", always_xy=True)

    def predict(self, request: PredictRequest) -> PredictResponse:
        west, south, east, north = self._settings.study_area_bbox
        if not (west <= request.lon <= east and south <= request.lat <= north):
            raise OutsideStudyAreaError(
                f"El punto ({request.lat}, {request.lon}) está fuera del área de estudio "
                f"(lon {west}..{east}, lat {south}..{north}).",
                {"study_area_bbox": [west, south, east, north]},
            )

        # el punto se ajusta a la celda de la grilla que lo contiene: el
        # cálculo usa el CENTRO de esa celda, así que dos puntos de la
        # misma celda dan exactamente el mismo resultado (y comparten caché).
        res = float(self._settings.spatial_resolution_m)
        x, y = self._to_crs.transform(request.lon, request.lat)
        ix, iy = math.floor(x / res), math.floor(y / res)
        key: CacheKey = (ix, iy, request.date, request.horizon_days, self._loaded.name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.model_copy(update={"cached": True})

        snapped_lon, snapped_lat = self._to_wgs84.transform((ix + 0.5) * res, (iy + 0.5) * res)
        response = self._compute(request, snapped_lat, snapped_lon)
        self._cache.put(key, response)
        return response

    def _compute(
        self, request: PredictRequest, snapped_lat: float, snapped_lon: float
    ) -> PredictResponse:
        res = float(self._settings.spatial_resolution_m)
        horizon = request.horizon_days
        days = [request.date + dt.timedelta(days=i) for i in range(horizon + 1)]

        sources = self._resolve_sources(days)
        self._require_weather(days)

        # detección sintética: SOLO ancla el estado de fuego del día 0 en
        # la celda pedida; no es una detección real de FIRMS.
        ignition = FireDetection(
            latitude=snapped_lat, longitude=snapped_lon,
            detected_at=dt.datetime(
                request.date.year, request.date.month, request.date.day, 12, tzinfo=dt.UTC
            ),
            frp=None, confidence="user", satellite="user", instrument="user_ignition",
        )
        tensor, _bbox = build_dataset_for_event(
            FireEvent(event_id=0, detections=(ignition,)),
            sources,
            resolution_m=res,
            crs=self._settings.crs,
            pre_event_padding_days=0,
            # el fuego puede avanzar ~1 celda/día: contexto suficiente
            # para el horizonte pedido.
            context_buffer_m=max(
                DEFAULT_CONTEXT_BUFFER_M, (horizon + _CONTEXT_CELLS_BEYOND_HORIZON) * res
            ),
            # radio de media celda: marca solo la celda de ignición.
            fire_buffer_m=res / 2,
            post_event_days=horizon,
        )
        warnings = self._check_tensor(tensor)

        probabilities = self._loaded.model.predict(tensor)[1:]
        return PredictResponse(
            features=self._cells_to_features(tensor, probabilities),
            model=ModelInfo(name=self._loaded.name, calibrated=self._loaded.calibrated),
            request=RequestEcho(
                lat=request.lat, lon=request.lon, snapped_lat=snapped_lat,
                snapped_lon=snapped_lon, date=request.date, horizon_days=horizon,
            ),
            grid=self._grid_info(tensor),
            days=[DayInfo(day=i, date=days[i]) for i in range(1, horizon + 1)],
            warnings=warnings,
        )

    def _resolve_sources(self, days: list[dt.date]):  # type: ignore[no-untyped-def]
        try:
            sources = resolve_event_sources(days, self._settings)
        except ValueError as exc:
            raise StaticLayersUnavailableError(
                f"Capas estáticas (DEM) no disponibles: {exc} Ejecutar `make ingest-terrain`."
            ) from exc
        missing = [
            p.name for p in (sources.slope_path, sources.aspect_path, sources.fuel_type_path)
            if not p.exists()
        ]
        if missing:
            raise StaticLayersUnavailableError(
                f"Faltan capas estáticas: {', '.join(missing)}. Ejecutar `make ingest-terrain` "
                f"y `make ingest-vegetation`.",
                {"missing": missing},
            )
        return sources

    @property
    def model_name(self) -> str:
        return self._loaded.name

    def static_layers_problem(self) -> str | None:
        """None si las capas estáticas (DEM, pendiente, orientación,
        combustible) están; si no, el mensaje de por qué no (para /api/health/)."""
        try:
            self._resolve_sources([])
        except StaticLayersUnavailableError as exc:
            return exc.message
        return None

    def weather_range(self) -> tuple[dt.date, dt.date] | None:
        """Primer y último día con TODO el clima que requiere el modelo
        (None si no hay ninguno procesado). Solo informativo para la UI:
        puede haber huecos dentro del rango, /predict los valida igual."""
        available = self._available_weather_dates(
            self._settings.data_processed_dir / "weather", self._required_weather_fields()
        )
        return (min(available), max(available)) if available else None

    def _required_weather_fields(self) -> list[str]:
        return [c for c in self._loaded.required_channels if c in _WEATHER_FIELDS]

    def _require_weather(self, days: list[dt.date]) -> None:
        weather_dir = self._settings.data_processed_dir / "weather"
        fields = self._required_weather_fields()
        missing_days = [
            d for d in days
            if not all((weather_dir / f"{f}_{d.isoformat()}.tif").exists() for f in fields)
        ]
        if not missing_days:
            return
        available = self._available_weather_dates(weather_dir, fields)
        available_range = [min(available).isoformat(), max(available).isoformat()] if available \
            else None
        raise WeatherUnavailableError(
            f"No hay datos de clima ({', '.join(fields)}) para: {', '.join(_iso(missing_days))}. "
            f"Rango disponible: "
            f"{' a '.join(available_range) if available_range else 'ninguno'}. "
            f"ERA5-Land es reanálisis histórico (no un pronóstico): ejecutar "
            f"`make ingest-weather` para esas fechas, o pedir una fecha dentro del rango.",
            {"missing_days": _iso(missing_days), "available_range": available_range},
        )

    @staticmethod
    def _available_weather_dates(weather_dir: Path, fields: list[str]) -> list[dt.date]:
        per_field: list[set[dt.date]] = []
        for field in fields:
            found: set[dt.date] = set()
            for path in weather_dir.glob(f"{field}_*.tif"):
                try:
                    found.add(dt.date.fromisoformat(path.stem.removeprefix(f"{field}_")))
                except ValueError:
                    continue
            per_field.append(found)
        return sorted(set.intersection(*per_field)) if per_field else []

    def _check_tensor(self, tensor: xr.DataArray) -> list[str]:
        channels = list(tensor.coords["channel"].values)
        warnings: list[str] = []
        for idx, name in enumerate(channels):
            if name == "fire_mask":
                continue
            bad = int(np.count_nonzero(~np.isfinite(tensor.values[:, idx])))
            if bad == 0:
                continue
            total = int(tensor.values[:, idx].size)
            if name in self._loaded.required_channels:
                error = WeatherUnavailableError if name in _WEATHER_FIELDS else TerrainCoverageError
                raise error(
                    f"El canal '{name}', requerido por el modelo {self._loaded.name}, no tiene "
                    f"datos en {bad} de {total} celdas del área de la predicción (el punto está "
                    f"en el borde o fuera de la cobertura de los datos procesados). "
                    f"No se rellena con valores inventados.",
                    {"channel": name, "missing_cells": bad, "total_cells": total},
                )
            warnings.append(
                f"{name}: sin datos en {bad} de {total} celdas; el modelo "
                f"{self._loaded.name} no usa este canal, no afecta esta predicción."
            )
        return warnings

    def _grid_info(self, tensor: xr.DataArray) -> GridInfo:
        rows, cols = int(tensor.sizes["y"]), int(tensor.sizes["x"])
        lons, lats = self._cell_vertices(tensor)
        return GridInfo(
            crs=str(tensor.attrs["crs"]),
            resolution_m=float(tensor.attrs["resolution_m"]),
            rows=rows, cols=cols,
            bbox_wgs84=(
                round(float(lons.min()), _COORD_DECIMALS),
                round(float(lats.min()), _COORD_DECIMALS),
                round(float(lons.max()), _COORD_DECIMALS),
                round(float(lats.max()), _COORD_DECIMALS),
            ),
        )

    def _cell_vertices(self, tensor: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
        """Vértices (rows+1, cols+1) de la grilla en WGS84 (lon, lat)."""
        affine = Affine(*tensor.attrs["transform"])
        rows, cols = int(tensor.sizes["y"]), int(tensor.sizes["x"])
        xs = affine.c + affine.a * np.arange(cols + 1)
        ys = affine.f + affine.e * np.arange(rows + 1)
        grid_x, grid_y = np.meshgrid(xs, ys)
        lons, lats = self._to_wgs84_for(str(tensor.attrs["crs"])).transform(grid_x, grid_y)
        return np.asarray(lons), np.asarray(lats)

    def _to_wgs84_for(self, crs: str) -> Transformer:
        return self._to_wgs84 if crs == self._settings.crs else Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        )

    def _cells_to_features(
        self, tensor: xr.DataArray, probabilities: np.ndarray
    ) -> list[dict[str, object]]:
        lons, lats = self._cell_vertices(tensor)
        rows, cols = int(tensor.sizes["y"]), int(tensor.sizes["x"])
        features: list[dict[str, object]] = []
        for r in range(rows):
            for c in range(cols):
                ring = [
                    [round(float(lons[rr, cc]), _COORD_DECIMALS),
                     round(float(lats[rr, cc]), _COORD_DECIMALS)]
                    for rr, cc in ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c), (r, c))
                ]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "row": r, "col": c,
                        "probability_by_day": [
                            round(float(p), _PROB_DECIMALS) for p in probabilities[:, r, c]
                        ],
                    },
                })
        return features
