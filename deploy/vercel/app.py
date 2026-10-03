"""Entrypoint de Vercel para PyroCast (`tool.vercel.entrypoint = "app:app"`).

`build.py` deja el código de los paquetes del monorepo en `_vendor/` y los datos
compactos en `data/processed/`. Aquí solo se ajusta `sys.path` y el entorno, y
se construye la app real de `serving/` sin cambios.
"""
import ctypes
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Bibliotecas del sistema que el runtime de Vercel no trae y que rasterio necesita
# (build.py las copia a _vendor/lib). RTLD_GLOBAL: así el cargador ya las tiene
# cuando rasterio las pide por nombre.
for _lib in sorted((HERE / "_vendor" / "lib").glob("*.so*")):
    ctypes.CDLL(str(_lib), mode=ctypes.RTLD_GLOBAL)
for package in ("shared", "features", "models", "ingestion", "serving"):
    sys.path.insert(0, str(HERE / "_vendor" / package / "src"))

# Datos empaquetados en el despliegue (solo lectura; las funciones no escriben disco).
os.environ.setdefault("DATA_PROCESSED_DIR", str(HERE / "data" / "processed"))

# `shared.config.Settings` exige estas credenciales porque el PIPELINE las usa
# (ingesta ERA5/Sentinel-2, PostGIS). La app servida no las toca: se rellenan
# con un marcador para no obligar a cargar secretos inútiles en Vercel.
for name, placeholder in {
    "CDS_API_URL": "https://cds.climate.copernicus.eu/api",
    "CDS_API_KEY": "unused-in-deploy",
    "COPERNICUS_DATASPACE_CLIENT_ID": "unused-in-deploy",
    "COPERNICUS_DATASPACE_CLIENT_SECRET": "unused-in-deploy",
    "POSTGRES_HOST": "unused-in-deploy",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "unused-in-deploy",
    "POSTGRES_USER": "unused-in-deploy",
    "POSTGRES_PASSWORD": "unused-in-deploy",
    # sin clave real, /active-fires responde un error claro (no tumba la app)
    "FIRMS_MAP_KEY": "not-configured",
}.items():
    os.environ.setdefault(name, placeholder)

from serving.api.main import create_app  # noqa: E402

app = create_app()
