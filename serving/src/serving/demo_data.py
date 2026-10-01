"""Datos SINTÉTICOS para probar la API y el mapa sin credenciales ni
descargas: rasters chicos (terreno plano, pastizal, viento constante del
oeste) con la misma convención de nombres que produce la ingesta real.

NO son datos reales y no dicen nada sobre ningún incendio: sirven solo
para ver el sistema funcionando de punta a punta (`make demo`) y para los
tests de `serving/`. Uso: `python -m serving.demo_data <data/processed/>`.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

DEMO_LAT = -37.5
DEMO_LON = -72.5
DEMO_FIRST_DAY = dt.date(2026, 1, 10)
DEMO_DAYS = 5
_SIZE = 80  # celdas de 250 m -> 20 km de lado, centrado en el punto
_WEATHER_FIELDS = ("wind_u", "wind_v", "temperature", "relative_humidity", "precipitation")
_WEATHER_VALUES = {"wind_u": 3.0, "wind_v": 0.0, "temperature": 25.0,
                   "relative_humidity": 30.0, "precipitation": 0.0}


def _write_tif(path: Path, value: float, transform: rasterio.Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((_SIZE, _SIZE), value, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=_SIZE, width=_SIZE, count=1, dtype="float32",
        crs="EPSG:32719", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def write_demo_dataset(processed: Path) -> Path:
    """Escribe DEM, pendiente, orientación, combustible, NDVI y 5 días de
    clima bajo `processed/` (el `DATA_PROCESSED_DIR` de la API)."""
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True).transform(
        DEMO_LON, DEMO_LAT
    )
    half = _SIZE * 250 / 2
    transform = from_origin(x - half, y + half, 250, 250)
    _write_tif(processed / "dem" / "dem_demo.tif", 100.0, transform)
    _write_tif(processed / "terrain" / "slope_deg.tif", 0.0, transform)
    _write_tif(processed / "terrain" / "aspect_deg.tif", -1.0, transform)
    _write_tif(processed / "vegetation" / "fuel_type.tif", 1.0, transform)  # 1 = pastizal
    _write_tif(processed / "vegetation" / "ndvi_2026-01.tif", 0.5, transform)
    for i in range(DEMO_DAYS):
        day = DEMO_FIRST_DAY + dt.timedelta(days=i)
        for field in _WEATHER_FIELDS:
            _write_tif(
                processed / "weather" / f"{field}_{day.isoformat()}.tif",
                _WEATHER_VALUES[field], transform,
            )
    return processed


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_data/processed")
    write_demo_dataset(target)
    last = DEMO_FIRST_DAY + dt.timedelta(days=DEMO_DAYS - 1)
    print(f"Datos sintéticos de demo en {target} (clima {DEMO_FIRST_DAY} a {last}, "
          f"centro lat {DEMO_LAT}, lon {DEMO_LON}). NO son datos reales.")
