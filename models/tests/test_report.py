"""Generador de reporte: todo número sale de bench/results/, determinista,
secciones explícitas cuando falta un insumo, y un extremo a extremo
artefactos -> reporte con un checkpoint y eventos de fixture."""
import json
from pathlib import Path

import numpy as np
import pytest
from models.evaluation.report import (
    DISCLAIMER,
    FAIL_IOU,
    build_report,
    parse_limitation_bullets,
)

MODELS = ("cellular_automata", "unet", "blend", "stacking")
BASE = {"cellular_automata": 0.311, "unet": 0.222, "blend": 0.333, "stacking": 0.444}


def _bins(ece_shift: float = 0.1) -> list[dict]:
    return [
        {"lo": i / 10, "hi": (i + 1) / 10, "count": 10 if i % 2 == 0 else 0,
         "mean_pred": (i + 0.5) / 10 if i % 2 == 0 else None,
         "mean_true": min(1.0, (i + 0.5) / 10 + ece_shift) if i % 2 == 0 else None}
        for i in range(10)
    ]


def _result(model: str, base: float) -> dict:
    agg = {
        m: {"point_estimate": base + k * 0.01, "lower": base - 0.05 + k * 0.01,
            "upper": base + 0.05 + k * 0.01}
        for k, m in enumerate(("iou", "dice", "brier", "ece"))
    }
    return {
        "model_name": model, "config": {}, "split": "test", "n_events": 2, "seed": 42,
        "n_bootstrap": 1000, "ece_bins": 10, "confidence": 0.95,
        "command": f"pyrocast-models backtest --model {model}",
        "git_commit": "abcdef1234567890" * 2 + "abcdef12",
        "per_event": [
            {"event_id": 11, "iou": base, "dice": base + 0.1, "brier": 0.1, "ece": 0.05},
            {"event_id": 22, "iou": base / 10, "dice": 0.05, "brier": 0.2, "ece": 0.07},
        ],
        "aggregate": agg,
    }


def _event(event_id: int, split: str, **over) -> dict:
    d = {"event_id": event_id, "split": split, "n_days": 4, "date_first": "2026-01-15",
         "date_last": "2026-01-18", "shape": [10, 12], "true_cells_final": 100,
         "mean_wind_speed": 3.0, "wind_dir_circ_std_deg": 20.0, "elevation_std_m": 50.0,
         "mean_slope_deg": 4.0, "nonflammable_fraction": 0.05}
    d.update(over)
    return d


def _artifacts(file: str = "report_examples.npz") -> dict:
    curve = {"brier": 0.07, "ece": 0.04, "bins": _bins()}
    me = [
        {"split": "test", "event_id": 22, "model": m, "iou": 0.05 if m == "unet" else 0.6,
         "dice": 0.1, "brier": 0.2, "ece": 0.07,
         "pred_cells_final": 0 if m == "unet" else 90, "true_cells_final": 460}
        for m in MODELS
    ] + [
        {"split": "test", "event_id": 11, "model": m, "iou": 0.7, "dice": 0.8, "brier": 0.1,
         "ece": 0.05, "pred_cells_final": 100, "true_cells_final": 113} for m in MODELS
    ]
    events = [_event(i, "train", true_cells_final=100 + i, elevation_std_m=20.0 + i,
                     wind_dir_circ_std_deg=10.0 + i) for i in range(1, 6)]
    events += [_event(11, "test"), _event(22, "test", true_cells_final=460,
                                          elevation_std_m=999.0)]
    return {
        "kind": "report_artifacts", "generated_at": "2026-10-01T12:00:00Z",
        "git_commit": "0123456789abcdef", "command": "pyrocast-models report-artifacts --seed 42",
        "checkpoint": "runs/x/best.pt", "checkpoint_fingerprint": "f" * 64, "seed": 42,
        "blend_weight_unet": 0.4, "resolution_m": 250.0,
        "splits": {"train": [1, 2, 3, 4, 5], "val": [], "test": [11, 22]},
        "events": events, "model_events": me,
        "calibration": {"unet_one_step": {
            "val": {"n": 500, "raw": curve, "calibrated": curve},
            "test": {"n": 300, "raw": {**curve, "ece": 0.1389}, "calibrated": curve}}},
        "reliability_backtest": {"test": {m: curve for m in MODELS}},
        "examples": {"file": file},
    }


def _write_examples(path: Path) -> None:
    rng = np.random.default_rng(0)
    arrays = {}
    for eid in (11, 22):
        truth = rng.random((10, 12)) > 0.6
        arrays[f"{eid}__truth_final"] = truth
        for m in MODELS:
            arrays[f"{eid}__{m}_final"] = rng.random((10, 12)).astype("float32")
    np.savez_compressed(path, **arrays)


LIMITATIONS = """# Limitaciones conocidas

## Diseño

- **Resolución espacio-temporal reducida frente a la literatura**: 250 m diario en vez de
  30 m / 3 h. Texto UNICO-RESOLUCION.
- **Resolución de ERA5-Land vs. grilla de trabajo**: downscaling bilineal UNICO-ERA5.
- **Reconstrucción de eventos de incendio: buffer**: UNICO-RECONSTRUCCION.
- **El lector nunca se probó contra un archivo real de Kaggle**: UNICO-NDWS.

## Herramienta de investigación

Herramienta de investigación. No usar sin validación de CONAF/SENAPRED.
"""


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    results, docs = tmp_path / "bench", tmp_path / "docs"
    results.mkdir()
    docs.mkdir()
    for m in MODELS:
        (results / f"{m}.json").write_text(json.dumps(_result(m, BASE[m])))
    (results / "report_artifacts.json").write_text(json.dumps(_artifacts()))
    _write_examples(results / "report_examples.npz")
    (docs / "limitations.md").write_text(LIMITATIONS)
    (docs / "api.md").write_text("# API de PyroCast\n")
    return results, docs


def test_report_writes_markdown_html_and_figures(workspace, tmp_path):
    results, docs = workspace
    md, html = build_report(results, docs, tmp_path / "out")
    assert md.name == "results.md" and html.name == "results.html"
    figures = {p.name for p in (tmp_path / "out" / "figures").glob("*.png")}
    assert {"calibration_unet.png", "reliability_backtest.png", "per_event_iou.png",
            "map_test_11.png", "map_test_22.png"} <= figures
    text = md.read_text()
    for heading in ("Resumen", "Tabla comparativa", "Calibración", "Predicciones contra",
                    "Dónde falla", "Limitaciones", "Metodología"):
        assert heading in text


def test_every_comparison_number_comes_from_the_result_files(workspace, tmp_path):
    results, docs = workspace
    text = build_report(results, docs, tmp_path / "out")[0].read_text()
    for model in MODELS:
        for metric, data in json.loads((results / f"{model}.json").read_text())[
            "aggregate"
        ].items():
            expected = (f"{data['point_estimate']:.3f} [{data['lower']:.3f}, "
                        f"{data['upper']:.3f}]")
            assert expected in text, (model, metric)


def test_changing_a_result_file_changes_the_report(workspace, tmp_path):
    results, docs = workspace
    before = build_report(results, docs, tmp_path / "a")[0].read_text()
    data = json.loads((results / "unet.json").read_text())
    data["aggregate"]["iou"] = {"point_estimate": 0.987, "lower": 0.986, "upper": 0.988}
    (results / "unet.json").write_text(json.dumps(data))
    after = build_report(results, docs, tmp_path / "b")[0].read_text()
    assert "0.987 [0.986, 0.988]" in after and "0.987" not in before


def test_report_is_deterministic_byte_for_byte(workspace, tmp_path):
    results, docs = workspace
    a_md, a_html = build_report(results, docs, tmp_path / "a")
    b_md, b_html = build_report(results, docs, tmp_path / "b")
    assert a_md.read_bytes() == b_md.read_bytes()
    assert a_html.read_bytes() == b_html.read_bytes()
    for png in (tmp_path / "a" / "figures").glob("*.png"):
        assert png.read_bytes() == (tmp_path / "b" / "figures" / png.name).read_bytes()


def test_disclaimer_is_visible_in_both_outputs_and_html_is_self_contained(workspace, tmp_path):
    results, docs = workspace
    md, html = build_report(results, docs, tmp_path / "out")
    assert DISCLAIMER in md.read_text()
    page = html.read_text()
    assert DISCLAIMER in page and 'class="notice"' in page
    assert page.count("data:image/png;base64,") >= 5
    assert 'src="http' not in page and 'src="figures' not in page
    assert "figures/calibration_unet.png" in md.read_text()


def test_missing_ensembles_and_artifacts_are_stated_not_hidden(workspace, tmp_path):
    results, docs = workspace
    for name in ("blend.json", "stacking.json", "report_artifacts.json", "report_examples.npz"):
        (results / name).unlink()
    text = build_report(results, docs, tmp_path / "out")[0].read_text()
    assert "No hay resultados de ensamble" in text
    assert text.count("**No disponible**") >= 3  # calibración, mapas, fallas


def test_internal_test_split_absence_is_declared_honestly(workspace, tmp_path):
    results, docs = workspace
    text = build_report(results, docs, tmp_path / "out")[0].read_text()
    assert "No existe en `bench/results/` ningún resultado de un test interno" in text


def test_failure_analysis_flags_low_iou_cases_with_evidence_based_hypotheses(workspace, tmp_path):
    results, docs = workspace
    text = build_report(results, docs, tmp_path / "out")[0].read_text()
    section = text.split("## 5. Dónde falla el modelo")[1].split("## 6.")[0]
    assert f"IoU < {FAIL_IOU:.2f}" in section
    assert "U-Net (calibrado): IoU 0.050" in section
    assert "### Evento 22 (test)" in section
    assert "**subpredice**" in section  # 0 celdas predichas frente a 460
    assert "fuera del rango de entrenamiento" in section  # elevation_std_m 999
    assert "terreno más complejo" in section
    assert "evento 11" not in section  # IoU 0.7: no es una falla
    assert "no verificadas" in section


def test_failure_analysis_without_supporting_descriptors_says_cause_unidentified(
    workspace, tmp_path
):
    results, docs = workspace
    art = json.loads((results / "report_artifacts.json").read_text())
    for e in art["events"]:
        if e["event_id"] == 22:
            e.update(elevation_std_m=22.0, true_cells_final=102, wind_dir_circ_std_deg=12.0)
    (results / "report_artifacts.json").write_text(json.dumps(art))
    text = build_report(results, docs, tmp_path / "out")[0].read_text()
    assert "estos datos no respaldan una hipótesis específica" in text


def test_limitations_are_consolidated_from_the_docs_file(workspace, tmp_path):
    results, docs = workspace
    text = build_report(results, docs, tmp_path / "out")[0].read_text()
    section = text.split("## 6. Limitaciones")[1].split("## 7.")[0]
    for marker in ("UNICO-RESOLUCION", "UNICO-ERA5", "UNICO-RECONSTRUCCION", "UNICO-NDWS"):
        assert marker in section
    assert "CONAF/SENAPRED" in section
    (docs / "limitations.md").write_text("# L\n\n- **Otra cosa**: nada.\n")
    text = build_report(results, docs, tmp_path / "out2")[0].read_text()
    assert "sin entrada específica en `docs/limitations.md`" in text


def test_methodology_lists_command_commit_and_data_dates(workspace, tmp_path):
    results, docs = workspace
    text = build_report(results, docs, tmp_path / "out")[0].read_text()
    section = text.split("## 7.")[1]
    assert "pyrocast-models backtest --model unet" in section
    assert "abcdef123456" in section and "0123456789ab" in section
    assert "2026-01-15" in section and "2026-01-18" in section
    assert "make report" in section


def test_parse_limitation_bullets_handles_wrapped_bullets():
    bullets = parse_limitation_bullets(LIMITATIONS)
    assert bullets[0][0].startswith("Resolución espacio-temporal")
    assert "30 m / 3 h" in bullets[0][1] and len(bullets) == 4


def test_artifacts_to_report_end_to_end_with_fixture_checkpoint(tmp_path, monkeypatch):
    """build_artifacts (checkpoint + eventos de fixture) alimenta a build_report."""
    from models.deep.calibration import calibrate_checkpoint
    from models.deep.train import ChileFinetuneDataset
    from models.evaluation.report_artifacts import build_artifacts
    from test_calibration_cli import _make_chile_event_for_calibration_cli, _make_real_checkpoint

    monkeypatch.chdir(tmp_path)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    splits = {"train": [1, 2, 3], "val": [4, 5], "test": [6, 7]}
    (dataset_dir / "splits.json").write_text(json.dumps(splits))
    events = {}
    for eid in range(1, 8):
        event = _make_chile_event_for_calibration_cli(eid, n_days=4, size=8)
        event.to_dataset().to_zarr(dataset_dir / f"event_{eid:04d}.zarr", mode="w")
        events[eid] = event
    checkpoint = tmp_path / "model.pt"
    _make_real_checkpoint(checkpoint)
    calibrate_checkpoint(checkpoint, ChileFinetuneDataset([events[4], events[5]]))

    results = tmp_path / "bench"
    out = build_artifacts(dataset_dir, checkpoint, results, seed=42)
    data = json.loads(out.read_text())
    assert data["kind"] == "report_artifacts"
    assert {r["split"] for r in data["model_events"]} == {"test", "val"}
    assert {r["model"] for r in data["model_events"]} == set(MODELS)
    assert len(data["events"]) == 7
    assert set(data["calibration"]["unet_one_step"]) == {"val", "test"}
    assert (results / "report_examples.npz").exists()

    md, _html = build_report(results, tmp_path / "docs", tmp_path / "docs")
    text = md.read_text()
    assert "map_test_6.png" in text and "calibration_unet.png" in text
    assert "No hay resultados de backtest" in text  # sin JSON de modelos en este bench
