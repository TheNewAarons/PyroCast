"""Repara los tensores Zarr locales construidos con el NDVI incorrecto (offset BOA
duplicado, docs/review.md C2) sin volver a descargar nada.

Uso: `uv run --package features python scripts/migrate_ndvi_offset.py`
(lee DATA_PROCESSED_DIR / DATA_RAW_DIR de .env). Hace una copia de respaldo del
dataset en `<dataset>_pre_ndvi_fix/` y escribe `bench/results/ndvi_fix_audit.json`.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from features.vegetation.migrate import migrate_event_ndvi
from shared.config import get_settings


def main() -> None:
    settings = get_settings()
    dataset_dir = settings.data_processed_dir / "dataset"
    backup = dataset_dir.with_name(dataset_dir.name + "_pre_ndvi_fix")
    if backup.exists():
        raise SystemExit(f"Ya existe {backup}: la migración ya se hizo (o bórralo a conciencia).")
    shutil.copytree(dataset_dir, backup)
    composites = sorted((settings.data_raw_dir / "sentinel2").glob("sentinel2_*.tif"))
    audit: list[dict[str, object]] = []
    for zarr_path in sorted(dataset_dir.glob("event_*.zarr")):
        tensor = xr.open_zarr(zarr_path)["fire_event_tensor"].load()
        channels = list(tensor.coords["channel"].values)
        result = migrate_event_ndvi(tensor, composites)
        old = tensor.values[0, channels.index("ndvi")]
        fire = tensor.values[:, channels.index("fire_mask")] >= 0.5
        burned = np.logical_or.accumulate(fire, axis=0)[-1]
        new = result.new_ndvi
        ok = np.isfinite(new)
        values = tensor.values.copy()
        values[:, channels.index("ndvi")] = new  # NDVI estático por evento: mismo en todos los días
        patched = tensor.copy(data=values)
        patched.attrs = dict(tensor.attrs)
        patched.to_dataset().to_zarr(zarr_path, mode="w")
        audit.append({
            "event_id": int(tensor.attrs["event_id"]),
            "composite": result.composite.name,
            "old_match_max_abs_diff": result.old_match_max_abs_diff,
            "old_ndvi_median": float(np.nanmedian(old)), "old_ndvi_max": float(np.nanmax(old)),
            "new_ndvi_median": float(np.nanmedian(new)),
            "new_ndvi_min": float(np.nanmin(new)), "new_ndvi_max": float(np.nanmax(new)),
            "new_ndvi_mean_burned": (
                float(np.nanmean(new[burned & ok])) if (burned & ok).any() else None
            ),
            "new_ndvi_mean_unburned": float(np.nanmean(new[~burned & ok])),
        })
        print(audit[-1]["event_id"], result.composite.name, f"{result.old_match_max_abs_diff:.1e}")
    out = Path("bench/results/ndvi_fix_audit.json")
    out.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(f"Auditoría -> {out}; respaldo -> {backup}")


if __name__ == "__main__":
    main()
