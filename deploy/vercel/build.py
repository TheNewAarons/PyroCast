"""Build Command de Vercel (`tool.vercel.scripts.build`), también usable local.

1. Copia el código de los paquetes del monorepo a `_vendor/` (Vercel solo empaqueta
   archivos dentro del Root Directory; los paquetes viven en `../../<pkg>/src`).
2. Descarga el paquete de datos compacto (URL y sha256 en `data-bundle.json`, o
   `PYROCAST_DATA_URL` + `PYROCAST_DATA_SHA256`), verifica el sha256 y lo extrae en
   `data/processed/`. Un archivo local: `python build.py --data-file ruta.tar.gz`.

Solo biblioteca estándar: corre antes de que importe nada del proyecto.
"""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PACKAGES = ("shared", "features", "models", "ingestion", "serving")
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "tests")


def vendor_sources(repo_root: Path = REPO_ROOT, target: Path = HERE / "_vendor") -> None:
    if target.exists():
        shutil.rmtree(target)
    for package in PACKAGES:
        src = repo_root / package / "src" / package
        if not src.is_dir():
            raise SystemExit(f"No se encontró {src}: ¿Root Directory = deploy/vercel?")
        shutil.copytree(src, target / package / "src" / package, ignore=_IGNORE)
    # serving/api/main.py busca la web en ../../../web relativo a sí mismo
    shutil.copytree(repo_root / "serving" / "web", target / "serving" / "web", ignore=_IGNORE)


# Bibliotecas del sistema que los wheels de rasterio (GDAL) esperan encontrar y que
# el runtime de funciones de Vercel no trae (verificado: "libexpat.so.1: cannot open
# shared object file"). Se copian desde la imagen de build a _vendor/lib y app.py
# las precarga antes de importar rasterio.
SYSTEM_LIBS = ("libexpat.so.1",)
_LIB_DIRS = ("/usr/lib64", "/lib64", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu",
             "/usr/lib")


def vendor_system_libs(target: Path = HERE / "_vendor" / "lib") -> list[str]:
    """Copia las bibliotecas de SYSTEM_LIBS que existan en esta máquina (solo Linux;
    en macOS no aplica). Devuelve las que no encontró."""
    import sys

    if not sys.platform.startswith("linux"):
        return []
    target.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in SYSTEM_LIBS:
        found = next((Path(d) / name for d in _LIB_DIRS if (Path(d) / name).exists()), None)
        if found is None:
            missing.append(name)
            continue
        shutil.copy2(found.resolve(), target / name)
    return missing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_data(url: str, sha256: str, destination: Path = HERE / "data") -> Path:
    """Descarga (o copia, si `url` es una ruta local) y verifica antes de extraer:
    un paquete con otro sha256 NO se usa."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bundle.tar.gz"
        if Path(url).exists():
            shutil.copyfile(url, archive)
        else:
            with urllib.request.urlopen(url, timeout=120) as response, open(archive, "wb") as out:
                shutil.copyfileobj(response, out)
        actual = _sha256(archive)
        if actual != sha256:
            raise SystemExit(
                f"sha256 del paquete de datos no coincide (esperado {sha256}, obtenido "
                f"{actual}): no se despliega con datos no verificados."
            )
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(destination, filter="data")
    processed = destination / "processed"
    if not any(processed.glob("dem/*.tif")):
        raise SystemExit(f"El paquete no trae DEM en {processed}/dem.")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", help="paquete local en vez de descargarlo")
    parser.add_argument("--skip-data", action="store_true", help="solo copiar el código")
    args = parser.parse_args()

    vendor_sources()
    print(f"Código copiado a {HERE / '_vendor'}")
    missing = vendor_system_libs()
    if missing:
        print(f"AVISO: no se encontraron {missing} en la imagen de build: rasterio puede "
              f"fallar al importar en el runtime.")
    if args.skip_data:
        return
    manifest = json.loads((HERE / "data-bundle.json").read_text())
    url = os.environ.get("PYROCAST_DATA_URL") or manifest["url"]
    sha256 = os.environ.get("PYROCAST_DATA_SHA256") or manifest["sha256"]
    if os.environ.get("PYROCAST_DATA_URL") and not os.environ.get("PYROCAST_DATA_SHA256"):
        raise SystemExit("PYROCAST_DATA_URL requiere PYROCAST_DATA_SHA256.")
    processed = fetch_data(args.data_file or url, sha256)
    print(f"Datos verificados (sha256 {sha256[:12]}…) en {processed}")


if __name__ == "__main__":
    main()
