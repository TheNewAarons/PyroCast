"""Calcula los insumos del reporte que NO caben en los JSON de backtest
(curvas de confiabilidad, predicciones por evento para los mapas,
descriptores de evento para el análisis de fallas) y los guarda en
`bench/results/` (`report_artifacts.json` + `report_examples.npz`), con su
comando, commit y fecha de generación.

Es el único paso del reporte que necesita el checkpoint y `data/processed/`
(ninguno versionado); `models.evaluation.report` solo lee lo que este paso
deja en `bench/results/`, así `make report` es reproducible sin esos datos.

Convenciones idénticas al backtest (`docs/backtest-2026.md`): eventos
recortados a su primer día con fuego, verdad acumulada, umbral 0.5, y el
ensamble usa el peso del blend ya registrado en `bench/results/blend.json`
y un stacking ajustado sobre val (NaN residual de val -> 0, como en el CLI).
"""
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from features.dataset.split import (
    DEFAULT_MAX_GAP_DAYS,
    DEFAULT_MAX_GAP_KM,
    find_split_leakage,
    footprint_from_tensor,
    footprint_gap_days,
    footprint_gap_km,
    group_events,
)
from shared.model_protocol import FireSpreadModel

from models.cellular_automata.model import CellularAutomatonModel
from models.cli import _exact_command, _git_commit_hash
from models.deep.calibration import (
    CalibratedUNet,
    _checkpoint_fingerprint,
    _collect_predictions,
    default_calibration_path,
)
from models.deep.ensemble import BlendEnsemble, StackingEnsemble, select_blend_weight
from models.deep.train import ChileFinetuneDataset
from models.evaluation.backtest import run_backtest
from models.evaluation.metrics import brier_score, ece_score, reliability_bins
from models.events import load_split_events

N_BINS = 10
THRESHOLD = 0.5


class _Precomputed:
    """Devuelve una predicción ya calculada -- para reutilizar `run_backtest`
    (mismas métricas que el reporte principal) sin predecir dos veces."""

    def __init__(self, prediction: np.ndarray) -> None:
        self._prediction = prediction

    def predict(self, event: xr.DataArray) -> np.ndarray:
        return self._prediction


def _cumulative_truth(event: xr.DataArray) -> np.ndarray:
    fire_idx = list(event.coords["channel"].values).index("fire_mask")
    result: np.ndarray = np.logical_or.accumulate(event.values[:, fire_idx] >= THRESHOLD, axis=0)
    return result


def _curve(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    return {
        "brier": brier_score(pred, true), "ece": ece_score(pred, true, n_bins=N_BINS),
        "bins": reliability_bins(pred, true, n_bins=N_BINS),
    }


def _channel(event: xr.DataArray, name: str) -> np.ndarray:
    idx = list(event.coords["channel"].values).index(name)
    result: np.ndarray = event.values[:, idx]
    return result


def describe_event(
    event: xr.DataArray, split: str, flammability: dict[int, float]
) -> dict[str, Any]:
    """Descriptores del evento que usa el análisis de fallas (todo calculado
    del tensor, nada a mano)."""
    n_days = int(event.sizes["day"])
    u = np.nanmean(_channel(event, "wind_u"), axis=(1, 2))
    v = np.nanmean(_channel(event, "wind_v"), axis=(1, 2))
    speed = np.hypot(u, v)
    if n_days > 1 and np.all(np.isfinite(u)) and np.all(np.isfinite(v)):
        angles = np.arctan2(v, u)
        resultant = math.hypot(float(np.cos(angles).mean()), float(np.sin(angles).mean()))
        circ_std = math.degrees(math.sqrt(max(-2.0 * math.log(max(resultant, 1e-12)), 0.0)))
    else:
        circ_std = 0.0
    fuel = _channel(event, "fuel_type")[0]
    finite = np.isfinite(fuel)
    flam = np.array([flammability.get(int(c), 0.0) for c in fuel[finite]])
    days = [str(d) for d in event.coords["day"].values]
    return {
        "event_id": int(event.attrs["event_id"]), "split": split, "n_days": n_days,
        "date_first": days[0], "date_last": days[-1],
        "shape": [int(event.sizes["y"]), int(event.sizes["x"])],
        "true_cells_final": int(_cumulative_truth(event)[-1].sum()),
        "mean_wind_speed": float(np.nanmean(speed)),
        "wind_dir_circ_std_deg": float(circ_std),
        "elevation_std_m": float(np.nanstd(_channel(event, "elevation")[0])),
        "mean_slope_deg": float(np.nanmean(_channel(event, "slope_deg")[0])),
        "nonflammable_fraction": float(np.mean(flam == 0.0)) if flam.size else None,
        "mean_temperature": float(np.nanmean(_channel(event, "temperature"))),
        "mean_relative_humidity": float(np.nanmean(_channel(event, "relative_humidity"))),
    }


def build_artifacts(
    dataset_dir: Path, checkpoint: Path, results_dir: Path, seed: int = 42,
    calibration: Path | None = None,
) -> Path:
    splits = json.loads((dataset_dir / "splits.json").read_text())
    events = {s: load_split_events(dataset_dir, s) for s in ("train", "val", "test")}
    # val: NaN residual -> 0, como en el CLI de backtest (solo afecta ajuste/evaluación de val)
    events["val"] = [e.copy(data=np.nan_to_num(e.values, nan=0.0)) for e in events["val"]]

    ca = CellularAutomatonModel(seed=seed)
    calibration_path = calibration or default_calibration_path(checkpoint)
    unet = CalibratedUNet(checkpoint, calibration_path if calibration_path.exists() else None)
    if unet.calibrator is None:
        raise RuntimeError(f"No hay calibrador en {calibration_path}: el reporte lo necesita.")

    blend_file = results_dir / "blend.json"
    if blend_file.exists():
        weight = float(json.loads(blend_file.read_text())["config"]["weight_unet"])
    else:
        weight = select_blend_weight(ca, unet, events["val"])
    models: dict[str, FireSpreadModel] = {
        "cellular_automata": ca, "unet": unet,
        "blend": BlendEnsemble(ca, unet, weight_unet=weight),
        "stacking": StackingEnsemble(ca, unet).fit(events["val"]),
    }

    model_events: list[dict[str, Any]] = []
    examples: dict[str, np.ndarray] = {}
    backtest_pred: dict[str, list[np.ndarray]] = {m: [] for m in models}
    backtest_true: list[np.ndarray] = []
    for split in ("test", "val"):
        for event in events[split]:
            event_id = int(event.attrs["event_id"])
            truth = _cumulative_truth(event)
            examples[f"{event_id}__truth_final"] = truth[-1]
            if split == "test":
                backtest_true.append(truth[1:].astype("float64").ravel())  # sin el ancla
            for name, model in models.items():
                pred = model.predict(event)
                metrics = run_backtest(_Precomputed(pred), [event], n_bootstrap=1, seed=seed,
                                       ece_bins=N_BINS).per_event[0]
                model_events.append({
                    "split": split, "event_id": event_id, "model": name, "iou": metrics.iou,
                    "dice": metrics.dice, "brier": metrics.brier, "ece": metrics.ece,
                    "pred_cells_final": int((pred[-1] >= THRESHOLD).sum()),
                    "true_cells_final": int(truth[-1].sum()),
                })
                examples[f"{event_id}__{name}_final"] = pred[-1].astype("float32")
                if split == "test":
                    backtest_pred[name].append(pred[1:].ravel())

    calibration_one_step: dict[str, Any] = {}
    for split in ("val", "test"):
        if not events[split]:
            continue
        raw, targets = _collect_predictions(
            unet.model, ChileFinetuneDataset(events[split]), unet.device
        )
        calibration_one_step[split] = {
            "n": int(raw.size), "raw": _curve(raw, targets),
            "calibrated": _curve(unet.calibrator.predict(raw), targets),
        }

    reliability = {
        "test": {
            name: _curve(np.concatenate(preds), np.concatenate(backtest_true))
            for name, preds in backtest_pred.items() if preds
        }
    }
    flammability = {int(k): float(v) for k, v in ca.params.fuel_flammability.items()}
    descriptors = [
        describe_event(event, split, flammability)
        for split in ("train", "val", "test") for event in events[split]
    ]
    split_audit = _split_audit(events, splits)
    resolution = float(events["test"][0].attrs["resolution_m"]) if events["test"] else 250.0

    results_dir.mkdir(parents=True, exist_ok=True)
    npz_name = "report_examples.npz"
    np.savez_compressed(results_dir / npz_name, **examples)
    payload = {
        "kind": "report_artifacts",
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit_hash(),
        "command": _exact_command_artifacts(checkpoint, seed),
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint),
        "seed": seed, "blend_weight_unet": weight, "resolution_m": resolution,
        "splits": splits, "split_audit": split_audit, "events": descriptors,
        "model_events": model_events,
        "calibration": {"unet_one_step": calibration_one_step},
        "reliability_backtest": reliability,
        "examples": {"file": npz_name},
    }
    out = results_dir / "report_artifacts.json"
    out.write_text(json.dumps(payload, sort_keys=True, indent=2))
    return out


def _split_audit(
    events: dict[str, list[xr.DataArray]], splits: dict[str, list[int]]
) -> dict[str, Any]:
    """Auditoría de fuga espacio-temporal entre splits (docs/review.md, C1):
    pares acoplados en splits distintos y, por evento retenido, el vecino más
    cercano en OTRO split."""
    footprints = {
        int(e.attrs["event_id"]): footprint_from_tensor(e)
        for split_events in events.values() for e in split_events
    }
    where = {e: s for s, ids in splits.items() for e in ids}
    nearest: list[dict[str, Any]] = []
    for eid in sorted(footprints):
        if where.get(eid) not in ("val", "test"):
            continue
        others = [o for o in footprints if where.get(o) != where[eid]]
        best = min(others, key=lambda o: (
            footprint_gap_km(footprints[eid], footprints[o]), footprint_gap_days(
                footprints[eid], footprints[o])))
        nearest.append({
            "event_id": eid, "split": where[eid], "nearest_event_id": best,
            "nearest_split": where[best],
            "gap_km": footprint_gap_km(footprints[eid], footprints[best]),
            "gap_days": footprint_gap_days(footprints[eid], footprints[best]),
        })
    return {
        "max_gap_km": DEFAULT_MAX_GAP_KM, "max_gap_days": DEFAULT_MAX_GAP_DAYS,
        "n_groups": len(group_events(footprints)),
        "violations": find_split_leakage(splits, footprints),
        "nearest_cross_split": nearest,
    }


def _exact_command_artifacts(checkpoint: Path, seed: int) -> str:
    return _exact_command([]).replace(
        "backtest", f"report-artifacts --checkpoint {checkpoint} --seed {seed}"
    ).strip()
