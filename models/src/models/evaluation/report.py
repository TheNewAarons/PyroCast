"""Generador de `docs/results.md` y `docs/results.html`.

Entradas: `bench/results/*.json` (resultados de backtest, uno por modelo,
con `model_name`; más `report_artifacts.json` + `report_examples.npz`, que
produce `pyrocast-models report-artifacts` a partir del checkpoint y los
datos reales) y `docs/*.md` (limitaciones, documentos relacionados).

Regla central (CLAUDE.md): NINGÚN número del reporte se escribe a mano.
Toda cifra sale de esos archivos; el texto solo da forma. Si un insumo
falta (p. ej. no hay artefactos, o no hay ensamble), la sección lo dice
explícitamente en vez de omitirse o rellenarse. La salida es
determinista: mismos insumos -> mismos bytes (sin fecha de "ahora").
"""
import base64
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DISCLAIMER = (
    "Herramienta de investigación. No usar para decisiones operativas de combate de "
    "incendios sin validación de CONAF/SENAPRED."
)
METRICS = ("iou", "dice", "brier", "ece")
METRIC_LABEL = {"iou": "IoU ↑", "dice": "Dice ↑", "brier": "Brier ↓", "ece": "ECE ↓"}
HIGHER_IS_BETTER = {"iou": True, "dice": True, "brier": False, "ece": False}
MODEL_ORDER = ("cellular_automata", "unet", "blend", "stacking")
MODEL_LABEL = {
    "cellular_automata": "Autómata celular (sin calibrar)",
    "unet": "U-Net (calibrado)",
    "blend": "Ensamble: blend",
    "stacking": "Ensamble: stacking",
}
# paleta apta para daltonismo (Okabe-Ito), un color fijo por modelo
MODEL_COLOR = {
    "cellular_automata": "#0072B2", "unet": "#D55E00", "blend": "#009E73", "stacking": "#CC79A7",
}
FAIL_IOU = 0.30  # un caso con IoU por debajo de esto se analiza en "dónde falla"
RANK_HIGH, RANK_LOW = 0.75, 0.25
RESULTS_JSON_ARTIFACTS = "report_artifacts.json"
# bullets de docs/limitations.md que se consolidan, por tema: (tema, regex sobre
# título del bullet, regex de respaldo sobre el cuerpo)
LIMITATION_TOPICS: tuple[tuple[str, str, str], ...] = (
    ("Resolución 250 m / diaria en vez de 30 m / 3 h", r"Resolución espacio-temporal", ""),
    ("Downscaling de ERA5-Land (~9 km → 250 m)", r"Resolución de ERA5-Land", ""),
    ("Reconstrucción simplificada del estado del fuego", r"Reconstrucción de eventos", ""),
    ("Dependencia de un dataset público externo para preentrenar",
     r"sin preentrenamiento", r"Kaggle"),
    ("Ausencia de validación operativa con CONAF/SENAPRED",
     r"CONAF|SENAPRED|operativ", r"SENAPRED"),
)


# ---------------------------------------------------------------- documento
@dataclass
class Doc:
    blocks: list[tuple[str, Any]] = field(default_factory=list)

    def h(self, level: int, text: str) -> None:
        self.blocks.append(("h", (level, text)))

    def p(self, text: str) -> None:
        self.blocks.append(("p", text))

    def ul(self, items: list[str]) -> None:
        self.blocks.append(("ul", items))

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        self.blocks.append(("table", (headers, rows)))

    def fig(self, filename: str, caption: str) -> None:
        self.blocks.append(("fig", (filename, caption)))

    def code(self, text: str) -> None:
        self.blocks.append(("code", text))

    def quote(self, text: str) -> None:
        self.blocks.append(("quote", text))


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(doc: Doc) -> str:
    out: list[str] = []
    for kind, data in doc.blocks:
        if kind == "h":
            out.append(f"{'#' * data[0]} {data[1]}\n")
        elif kind == "p":
            out.append(f"{data}\n")
        elif kind == "quote":
            out.append(f"> {data}\n")
        elif kind == "ul":
            out.append("\n".join(f"- {item}" for item in data) + "\n")
        elif kind == "code":
            out.append(f"```\n{data}\n```\n")
        elif kind == "table":
            headers, rows = data
            lines = ["| " + " | ".join(_cell(h) for h in headers) + " |",
                     "|" + "|".join("---" for _ in headers) + "|"]
            lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
            out.append("\n".join(lines) + "\n")
        elif kind == "fig":
            out.append(f"![{data[1]}](figures/{data[0]})\n\n*{data[1]}*\n")
    return "\n".join(out)


def _inline_html(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


_CSS = """
:root{--bg:#fff;--fg:#1b1b1b;--muted:#555;--line:#d4d4d0;--panel:#f6f6f4;--warn-bg:#fff3cd;
--warn-fg:#4d3b00;--warn-line:#e0b84c}
@media (prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#ececec;--muted:#b0b0b0;
--line:#3a3d41;--panel:#202225;--warn-bg:#3d3210;--warn-fg:#ffe9a6;--warn-line:#8a7220}}
body{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,sans-serif;
margin:0;padding:0 16px 48px}
main{max-width:980px;margin:0 auto}
.notice{background:var(--warn-bg);color:var(--warn-fg);border:2px solid var(--warn-line);
padding:10px 14px;margin:16px 0;border-radius:4px;font-weight:600}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px;display:block;
overflow-x:auto}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:var(--panel)}
figure{margin:20px 0}img{max-width:100%;height:auto;background:#fff}
figcaption{color:var(--muted);font-size:14px}
code,pre{background:var(--panel);padding:1px 4px;border-radius:3px;font-size:13px}
pre{padding:10px;overflow-x:auto}
blockquote{border-left:4px solid var(--line);margin:12px 0;padding:2px 12px;color:var(--muted)}
"""


def render_html(doc: Doc, figures_dir: Path, title: str) -> str:
    body: list[str] = []
    for kind, data in doc.blocks:
        if kind == "h":
            body.append(f"<h{data[0]}>{_inline_html(data[1])}</h{data[0]}>")
        elif kind == "p":
            css = ' class="notice"' if data.startswith("**" + DISCLAIMER) else ""
            body.append(f"<p{css}>{_inline_html(data)}</p>")
        elif kind == "quote":
            body.append(f"<blockquote>{_inline_html(data)}</blockquote>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{_inline_html(i)}</li>" for i in data) + "</ul>")
        elif kind == "code":
            body.append(f"<pre>{html.escape(data)}</pre>")
        elif kind == "table":
            headers, rows = data
            head = "".join(f"<th>{_inline_html(h)}</th>" for h in headers)
            trs = "".join(
                "<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in row) + "</tr>"
                for row in rows
            )
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{trs}</tbody></table>")
        elif kind == "fig":
            png = base64.b64encode((figures_dir / data[0]).read_bytes()).decode("ascii")
            alt = html.escape(data[1], quote=True)
            body.append(
                f'<figure><img src="data:image/png;base64,{png}" alt="{alt}">'
                f"<figcaption>{_inline_html(data[1])}</figcaption></figure>"
            )
    return (
        '<!doctype html>\n<html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>\n"
    )


# ------------------------------------------------------------------- carga
@dataclass(frozen=True)
class Inputs:
    results: dict[str, dict[str, Any]]  # model_name -> JSON de backtest
    result_files: dict[str, str]  # model_name -> nombre de archivo
    artifacts: dict[str, Any] | None
    examples: dict[str, np.ndarray] | None
    docs_dir: Path


def load_inputs(results_dir: Path, docs_dir: Path) -> Inputs:
    results: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}
    artifacts: dict[str, Any] | None = None
    examples: dict[str, np.ndarray] | None = None
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("kind") == "report_artifacts":
            artifacts = data
        elif "model_name" in data and "per_event" in data:
            results[data["model_name"]] = data
            files[data["model_name"]] = path.name
    if artifacts is not None:
        npz_path = results_dir / artifacts["examples"]["file"]
        if npz_path.exists():
            with np.load(npz_path) as npz:
                examples = {key: npz[key] for key in npz.files}
    return Inputs(results, files, artifacts, examples, docs_dir)


def _ordered(models: dict[str, Any]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(m for m in models if m not in MODEL_ORDER)


def _label(model: str) -> str:
    return MODEL_LABEL.get(model, model)


def _ci(ci: dict[str, float]) -> str:
    return f"{ci['point_estimate']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]"


# ---------------------------------------------------------------- secciones
def _summary(doc: Doc, inp: Inputs) -> None:
    doc.h(2, "1. Resumen")
    models = _ordered(inp.results)
    if not models:
        doc.p("**No hay resultados de backtest en `bench/results/`.** Nada que resumir.")
        return
    n_events = {inp.results[m]["n_events"] for m in models}
    doc.p(
        f"Backtest sobre el split de test de incendios reales de Chile 2025-2026 "
        f"(n = {', '.join(str(n) for n in sorted(n_events))} eventos de test). "
        f"Modelos con resultados: {', '.join(_label(m) for m in models)}."
    )
    bullets: list[str] = []
    for metric in METRICS:
        best = (max if HIGHER_IS_BETTER[metric] else min)(
            models, key=lambda m: inp.results[m]["aggregate"][metric]["point_estimate"]
        )
        best_ci = inp.results[best]["aggregate"][metric]
        overlapping = [
            m for m in models if m != best and _intervals_overlap(
                best_ci, inp.results[m]["aggregate"][metric]
            )
        ]
        verdict = (
            f"intervalo solapado con {', '.join(_label(m) for m in overlapping)}: "
            f"**diferencia no distinguible** con este n"
            if overlapping else "intervalo sin solape con los demás modelos"
        )
        bullets.append(
            f"{METRIC_LABEL[metric]}: mejor {_label(best)} ({_ci(best_ci)}); {verdict}."
        )
    doc.ul(bullets)
    doc.p(
        "Con tan pocos eventos de test los intervalos bootstrap (remuestreo de valores por "
        "evento) son anchos o, con n pequeño, engañosamente angostos: **esto no es una "
        "comparación estadísticamente robusta**. Ver las secciones 2 y 5."
    )


def _intervals_overlap(a: dict[str, float], b: dict[str, float]) -> bool:
    return a["lower"] <= b["upper"] and b["lower"] <= a["upper"]


def _comparison(doc: Doc, inp: Inputs) -> None:
    doc.h(2, "2. Tabla comparativa")
    models = _ordered(inp.results)
    doc.h(3, "2.1 Backtest real 2026 (P12) y ensambles (P13)")
    if not models:
        doc.p("No hay resultados disponibles.")
    else:
        rows = [
            [METRIC_LABEL[metric]] + [_ci(inp.results[m]["aggregate"][metric]) for m in models]
            for metric in METRICS
        ]
        doc.table(["Métrica (media [IC 95 % bootstrap])"] + [_label(m) for m in models], rows)
        if "blend" not in inp.results and "stacking" not in inp.results:
            doc.p("**No hay resultados de ensamble en `bench/results/`**: la tabla solo compara "
                  "los modelos individuales.")
    doc.h(3, "2.2 Test interno (P8)")
    doc.p(
        "El 'split de test interno' del dataset de eventos de Chile **es** el conjunto "
        "evaluado en 2.1: `features.dataset.split` reserva esos eventos por evento (nunca por "
        "píxel) y el backtest los usa. **No existe en `bench/results/` ningún resultado de un "
        "test interno distinto** (p. ej. sobre el dataset público NDWS, que el U-Net nunca vio: "
        "se entrenó solo con eventos de Chile). No se rellena con números de fixtures sintéticos."
    )
    doc.h(3, "2.3 Por evento")
    rows = []
    event_ids = sorted({e["event_id"] for m in models for e in inp.results[m]["per_event"]})
    for event_id in event_ids:
        for m in models:
            per = next((e for e in inp.results[m]["per_event"] if e["event_id"] == event_id), None)
            if per is None:
                continue
            rows.append([str(event_id), _label(m)] + [f"{per[k]:.3f}" for k in METRICS])
    doc.table(["event_id", "modelo"] + [METRIC_LABEL[k] for k in METRICS], rows)
    _figure_per_event(inp, event_ids)
    doc.fig("per_event_iou.png", "IoU por evento de test y modelo (datos de bench/results/).")


def _figure_per_event(inp: Inputs, event_ids: list[int]) -> None:
    models = _ordered(inp.results)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = []
        for event_id in event_ids:
            per = next((e for e in inp.results[m]["per_event"] if e["event_id"] == event_id), None)
            vals.append(per["iou"] if per else np.nan)
        ax.bar(np.arange(len(event_ids)) + i * width, vals, width, label=_label(m),
               color=MODEL_COLOR.get(m))
    ax.set_xticks(np.arange(len(event_ids)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels([str(e) for e in event_ids])
    ax.set_ylabel("IoU")
    ax.set_xlabel("event_id (test)")
    ax.set_ylim(0, 1)
    if models:
        ax.legend(fontsize=8)
    ax.set_title("IoU por evento de test")
    _save(fig, inp, "per_event_iou.png")


# ------------------------------------------------------------- calibración
def _calibration(doc: Doc, inp: Inputs) -> None:
    doc.h(2, "3. Calibración")
    art = inp.artifacts
    if art is None:
        doc.p("**No disponible**: falta `bench/results/report_artifacts.json` "
              "(`make report-artifacts`, requiere checkpoint y datos locales).")
        return
    cal = art["calibration"]["unet_one_step"]
    doc.h(3, "3.1 U-Net: antes / después de la calibración isotónica (P11)")
    rows = []
    for split, label in (("val", "val (conjunto de AJUSTE del calibrador)"),
                         ("test", "test (fuera de muestra)")):
        c = cal.get(split)
        if not c:
            continue
        rows.append([
            label, str(c["n"]), f"{c['raw']['brier']:.4f}", f"{c['calibrated']['brier']:.4f}",
            f"{c['raw']['ece']:.4f}", f"{c['calibrated']['ece']:.4f}",
        ])
    doc.table(["conjunto", "n celdas", "Brier antes", "Brier después", "ECE antes",
               "ECE después"], rows)
    doc.p(
        "Pares (entrada del día d, máscara del día d+1) de un solo paso, el mismo contrato con "
        "que se entrenó y calibró. **El ECE 'después' sobre val es estructuralmente ~0** (el "
        "calibrador se ajusta sobre esos mismos datos, `docs/calibration.md`): solo la fila de "
        "test mide calibración fuera de muestra."
    )
    _figure_calibration(inp, cal)
    doc.fig("calibration_unet.png", "Diagramas de confiabilidad del U-Net, crudo vs. calibrado.")
    doc.h(3, "3.2 Los cuatro modelos en el backtest (predicción acumulada multi-día)")
    rel = art["reliability_backtest"].get("test", {})
    if rel:
        _figure_reliability(inp, rel)
        doc.fig("reliability_backtest.png",
                "Confiabilidad de cada modelo sobre los eventos de test (misma convención "
                "acumulada que las métricas de 2.1).")
        doc.table(
            ["modelo", "Brier", "ECE"],
            [[_label(m), f"{rel[m]['brier']:.4f}", f"{rel[m]['ece']:.4f}"] for m in _ordered(rel)],
        )
    else:
        doc.p("No hay datos de confiabilidad del backtest en los artefactos.")


def _plot_bins(ax: Any, bins: list[dict[str, Any]], label: str, color: str, marker: str) -> None:
    pts = [(b["mean_pred"], b["mean_true"]) for b in bins if b["count"]]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, color=color, label=label,
                lw=1.5, ms=5)


def _figure_calibration(inp: Inputs, cal: dict[str, Any]) -> None:
    splits = [s for s in ("val", "test") if cal.get(s)]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.3), squeeze=False)
    for ax, split in zip(axes[0], splits, strict=True):
        c = cal[split]
        ax.plot([0, 1], [0, 1], "--", color="#888", label="calibración perfecta")
        _plot_bins(ax, c["raw"]["bins"], f"crudo (ECE {c['raw']['ece']:.3f})", "#D55E00", "o")
        _plot_bins(ax, c["calibrated"]["bins"],
                   f"calibrado (ECE {c['calibrated']['ece']:.3f})", "#0072B2", "s")
        ax.set_title("val (ajuste)" if split == "val" else "test (fuera de muestra)")
        ax.set_xlabel("probabilidad predicha")
        ax.set_ylabel("frecuencia observada")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
    _save(fig, inp, "calibration_unet.png")


def _figure_reliability(inp: Inputs, rel: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="calibración perfecta")
    for m in _ordered(rel):
        _plot_bins(ax, rel[m]["bins"], f"{_label(m)} (ECE {rel[m]['ece']:.3f})",
                   MODEL_COLOR.get(m, "#444"), "o")
    ax.set_xlabel("probabilidad predicha")
    ax.set_ylabel("frecuencia observada")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Confiabilidad en el backtest (test)")
    _save(fig, inp, "reliability_backtest.png")


# -------------------------------------------------------------------- mapas
def _maps(doc: Doc, inp: Inputs) -> None:
    doc.h(2, "4. Predicciones contra el incendio real")
    art, ex = inp.artifacts, inp.examples
    if art is None or ex is None:
        doc.p("**No disponible**: faltan `report_artifacts.json` / `report_examples.npz` "
              "(`make report-artifacts`).")
        return
    doc.p(
        "**Criterio de selección: ninguno que favorezca al modelo.** Se muestran *todos* los "
        "eventos retenidos (no vistos al entrenar los pesos): los de **test** (evidencia fuera "
        "de muestra) y los de **val** (con rótulo: val se usó para calibrar el U-Net y elegir "
        "el peso/coeficientes del ensamble, así que NO es evidencia fuera de muestra). Cada "
        "caso se rotula 'bueno' o 'malo' según el IoU del U-Net, calculado de los datos. "
        "Cada panel muestra el último día del horizonte; el contorno negro es el área "
        "realmente quemada acumulada."
    )
    model_events = art["model_events"]
    cases = sorted(
        {(r["split"], r["event_id"]) for r in model_events},
        key=lambda c: (0 if c[0] == "test" else 1, c[1]),
    )
    for split, event_id in cases:
        rows = {r["model"]: r for r in model_events
                if r["split"] == split and r["event_id"] == event_id}
        anchor = rows.get("unet") or next(iter(rows.values()))
        quality = "bueno" if anchor["iou"] >= FAIL_IOU else "malo"
        fname = f"map_{split}_{event_id}.png"
        if _figure_map(inp, split, event_id, rows):
            doc.h(3, f"Evento {event_id} ({split}) — caso {quality} para "
                     f"{_label('unet') if 'unet' in rows else _label(anchor['model'])} "
                     f"(IoU {anchor['iou']:.3f})")
            doc.fig(fname, f"Evento {event_id} ({split}): verdad acumulada y probabilidad "
                           f"predicha por modelo, último día.")


def _figure_map(inp: Inputs, split: str, event_id: int, rows: dict[str, Any]) -> bool:
    assert inp.examples is not None and inp.artifacts is not None
    truth = inp.examples.get(f"{event_id}__truth_final")
    if truth is None:
        return False
    models = [m for m in _ordered(rows) if f"{event_id}__{m}_final" in inp.examples]
    resolution = float(inp.artifacts["resolution_m"])
    height, width = truth.shape
    extent = (0, width * resolution / 1000, 0, height * resolution / 1000)
    n = len(models) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.3), squeeze=False)
    axes[0][0].imshow(truth.astype(float), cmap="Greys", vmin=0, vmax=1, extent=extent)
    axes[0][0].set_title("verdad (quemado acum.)", fontsize=8)
    image = None
    for ax, m in zip(axes[0][1:], models, strict=True):
        pred = inp.examples[f"{event_id}__{m}_final"]
        image = ax.imshow(pred, cmap="YlOrRd", vmin=0, vmax=1, extent=extent)
        ax.contour(truth.astype(float), levels=[0.5], colors="k", linewidths=0.8,
                   extent=(0, extent[1], extent[3], 0))
        ax.set_title(f"{_label(m)}\nIoU {rows[m]['iou']:.2f}", fontsize=8)
    for ax in axes[0]:
        ax.set_xlabel("km", fontsize=8)
        ax.tick_params(labelsize=7)
    if image is not None:
        fig.colorbar(image, ax=list(axes[0]), shrink=0.8, label="probabilidad")
    _save(fig, inp, f"map_{split}_{event_id}.png")
    return True


# ------------------------------------------------------------------- fallas
def _percentile_rank(value: float, train_values: list[float]) -> float:
    # rango medio: los empates cuentan la mitad, así un descriptor sin
    # variación en el entrenamiento queda en 0.5 (ni alto ni bajo).
    below = sum(v < value for v in train_values)
    ties = sum(v == value for v in train_values)
    return float((below + 0.5 * ties) / len(train_values))


DESCRIPTORS: tuple[tuple[str, str, str], ...] = (
    ("true_cells_final", "celdas realmente quemadas (último día)", "size"),
    ("wind_dir_circ_std_deg", "variabilidad de la dirección diaria del viento (desv. circular, °)",
     "wind"),
    ("mean_wind_speed", "velocidad media del viento (m/s)", "windspeed"),
    ("elevation_std_m", "desv. estándar de la elevación (m)", "terrain"),
    ("mean_slope_deg", "pendiente media (°)", "terrain"),
    ("nonflammable_fraction", "fracción de celdas no combustibles", "fuel"),
)
_HYPOTHESIS = {
    "size": "evento más chico que la mayoría de los de entrenamiento (fuera de la "
            "distribución por tamaño): el modelo vio pocos casos así",
    "wind": "viento cambiante no capturado por agregados diarios: la dirección varía y el "
            "modelo recibe un único vector medio por día",
    "windspeed": "viento más intenso que en la mayoría de los eventos de entrenamiento: "
                 "régimen poco representado",
    "terrain": "terreno más complejo que en la mayoría de los eventos de entrenamiento: la "
               "propagación depende de la pendiente/elevación y a 250 m se resuelve de "
               "forma gruesa",
    "fuel": "combustible: gran parte de la zona es no combustible según WorldCover y la "
            "clasificación de combustible es un proxy grueso",
}


def _fmt(key: str, value: float) -> str:
    return f"{value:.0f}" if key == "true_cells_final" else f"{value:.2f}"


def _descriptor_analysis(
    ev: dict[str, Any], train: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    hypotheses: list[str] = []
    for key, label, theme in DESCRIPTORS:
        values = [t[key] for t in train if t.get(key) is not None]
        value = ev.get(key)
        if value is None or not values:
            continue
        rank = _percentile_rank(value, values)
        outside = value < min(values) or value > max(values)
        flag = ""
        if outside:
            flag = " **fuera del rango de entrenamiento**"
        elif rank >= RANK_HIGH:
            flag = " (alto frente al entrenamiento)"
        elif rank <= RANK_LOW:
            flag = " (bajo frente al entrenamiento)"
        lines.append(
            f"{label}: {_fmt(key, value)}; entrenamiento "
            f"{_fmt(key, min(values))}–{_fmt(key, max(values))}; "
            f"percentil {rank * 100:.0f}{flag}"
        )
        high_theme = theme in ("wind", "terrain", "fuel", "windspeed")
        small_event = theme == "size" and rank <= RANK_LOW
        if outside or (rank >= RANK_HIGH and high_theme) or small_event:
            hypotheses.append(f"{_HYPOTHESIS[theme]} ({label}: {_fmt(key, value)}).")
    return lines, hypotheses


def _failures(doc: Doc, inp: Inputs) -> None:
    doc.h(2, "5. Dónde falla el modelo")
    art = inp.artifacts
    if art is None:
        doc.p("**No disponible**: sin `report_artifacts.json` no hay diagnóstico por evento.")
        return
    n_train = sum(1 for e in art["events"] if e["split"] == "train")
    doc.p(
        f"Se analizan todos los pares modelo-evento retenidos con **IoU < {FAIL_IOU:.2f}** "
        f"(umbral fijo de este reporte, no ajustado a los resultados), agrupados por evento. "
        f"Se comparan descriptores del evento contra los {n_train} eventos de entrenamiento. "
        f"**Las hipótesis NO están verificadas**: con tan pocos eventos no se puede aislar "
        f"una causa."
    )
    events = {e["event_id"]: e for e in art["events"]}
    train = [e for e in art["events"] if e["split"] == "train"]
    failing = [r for r in art["model_events"] if r["iou"] < FAIL_IOU]
    if not failing:
        doc.p(f"**Ningún caso retenido tiene IoU < {FAIL_IOU:.2f}.**")
        return
    cases = sorted({(r["split"], r["event_id"]) for r in failing},
                   key=lambda c: (0 if c[0] == "test" else 1, c[1]))
    for split, event_id in cases:
        doc.h(3, f"Evento {event_id} ({split})"
              + ("" if split == "test" else " — val: no es evidencia fuera de muestra"))
        rows = sorted((r for r in failing if (r["split"], r["event_id"]) == (split, event_id)),
                      key=lambda r: _ordered_index(r["model"]))
        bullets = []
        for r in rows:
            ratio = r["pred_cells_final"] / r["true_cells_final"] if r["true_cells_final"] else None
            direction = ""
            if ratio is not None:
                direction = (" — **subpredice**" if ratio < 0.5
                             else " — **sobrepredice**" if ratio > 2 else "")
            bullets.append(
                f"{_label(r['model'])}: IoU {r['iou']:.3f}, Dice {r['dice']:.3f}, "
                f"Brier {r['brier']:.3f}; celdas con prob ≥ 0,5 en el último día: "
                f"{r['pred_cells_final']} frente a {r['true_cells_final']} realmente quemadas"
                + (f" (razón {ratio:.2f})" if ratio is not None else "") + direction + "."
            )
        doc.ul(bullets)
        ev = events.get(event_id)
        if ev is None or not train:
            doc.p("Sin descriptores de evento / de entrenamiento para comparar.")
            continue
        lines, hypotheses = _descriptor_analysis(ev, train)
        doc.p("Descriptores del evento frente a los eventos de entrenamiento:")
        doc.ul(lines)
        if hypotheses:
            doc.p("**Hipótesis respaldadas por los descriptores** (no verificadas):")
            doc.ul(sorted(set(hypotheses)))
        else:
            doc.p("**Ningún descriptor se aparta del entrenamiento**: estos datos no respaldan "
                  "una hipótesis específica; la causa queda sin identificar.")
        if any(r["model"] == "unet" and r["pred_cells_final"] < 0.5 * r["true_cells_final"]
               for r in rows):
            doc.p("Para el U-Net, la subpredicción sistemática es coherente con un modelo "
                  "entrenado desde cero con muy pocos eventos (sin preentrenamiento), que no "
                  "aprendió a propagar el fuego: hipótesis, sin experimento que la aísle.")


def _ordered_index(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


# ------------------------------------------------------------- limitaciones
def parse_limitation_bullets(text: str) -> list[tuple[str, str]]:
    """(título, cuerpo) de cada bullet `- **Título**: cuerpo` de limitations.md."""
    bullets: list[tuple[str, str]] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        raw = " ".join(s.strip() for s in current)
        match = re.match(r"-\s+\*\*(.+?)\*\*:?\s*(.*)", raw)
        if match:
            bullets.append((match.group(1).strip(), match.group(2).strip()))

    for line in text.splitlines():
        if line.startswith("- "):
            flush()
            current = [line]
        elif current and (line.startswith("  ") or not line.strip()):
            if line.strip():
                current.append(line)
            else:
                flush()
                current = []
        elif line.startswith("#"):
            flush()
            current = []
    flush()
    return bullets


def _first_sentences(text: str, limit: int = 520) -> str:
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: end + 1] if end > 120 else cut.rstrip()) + " […]"


def _limitations(doc: Doc, inp: Inputs) -> None:
    doc.h(2, "6. Limitaciones")
    path = inp.docs_dir / "limitations.md"
    if not path.exists():
        doc.p("**`docs/limitations.md` no existe**: no se puede consolidar.")
        return
    text = path.read_text()
    bullets = parse_limitation_bullets(text)
    doc.p("Consolidado desde `docs/limitations.md` (ahí está el detalle completo y cada "
          "hallazgo con su fecha).")
    items: list[str] = []
    for topic, title_re, body_re in LIMITATION_TOPICS:
        match = next((b for b in bullets if re.search(title_re, b[0], re.I)), None)
        if match is None and body_re:
            match = next((b for b in bullets if re.search(body_re, b[1])), None)
        if match is not None:
            items.append(f"**{topic}** — *{match[0]}*: {_first_sentences(match[1])}")
        elif re.search(r"CONAF|SENAPRED", topic) and re.search(r"CONAF/SENAPRED", text):
            items.append(f"**{topic}** — `docs/limitations.md` declara que esta es una "
                         f"herramienta de investigación y que cualquier uso operativo requiere "
                         f"validación de CONAF/SENAPRED; no se hizo ninguna validación operativa.")
        else:
            items.append(f"**{topic}** — *sin entrada específica en `docs/limitations.md`.*")
    doc.ul(items)
    doc.p("**Además** (de la evaluación, no solo del diseño): n = 2 eventos de test; autómata "
          "celular sin calibrar contra incendios reales; U-Net entrenado desde cero con eventos "
          "de Chile; pesos del ensamble ajustados sobre un val que también calibró el U-Net. "
          "Ver `docs/backtest-2026.md` y `docs/limitations.md`.")
    doc.p(f"`docs/limitations.md` registra {len(bullets)} limitaciones en total. Títulos:")
    doc.ul([title for title, _ in bullets])


# -------------------------------------------------------------- metodología
def _methodology(doc: Doc, inp: Inputs) -> None:
    doc.h(2, "7. Metodología y reproducibilidad")
    doc.p("Regenerar este reporte (determinista: mismos insumos, mismos bytes):")
    doc.code("make report-artifacts   # requiere checkpoint + data/processed (no versionados)\n"
             "make report             # solo lee bench/results/ y docs/")
    rows = []
    for m in _ordered(inp.results):
        r = inp.results[m]
        rows.append([f"`{inp.result_files[m]}`", _label(m), f"`{r['command']}`",
                     f"`{r['git_commit'][:12]}`", str(r["seed"]), str(r["n_bootstrap"]),
                     str(r["n_events"])])
    doc.table(["archivo", "modelo", "comando exacto", "commit", "semilla", "remuestreos",
               "eventos"], rows)
    art = inp.artifacts
    if art is None:
        doc.p("**Sin `report_artifacts.json`**: no hay procedencia de calibración, mapas ni "
              "fechas de datos.")
        return
    fingerprint = art["checkpoint_fingerprint"][:16]
    doc.p(f"Artefactos (`{RESULTS_JSON_ARTIFACTS}`, `{art['examples']['file']}`): comando "
          f"`{art['command']}`, commit `{art['git_commit'][:12]}`, "
          f"generados {art['generated_at']}, checkpoint `{art['checkpoint']}` "
          f"(huella sha256 `{fingerprint}…`), semilla {art['seed']}, "
          f"peso U-Net del blend {art['blend_weight_unet']}.")
    doc.h(3, "Fechas de los datos usados")
    doc.table(
        ["event_id", "split", "primer día", "último día", "días", "grilla (y×x)"],
        [[str(e["event_id"]), e["split"], e["date_first"], e["date_last"], str(e["n_days"]),
          f"{e['shape'][0]}×{e['shape'][1]}"]
         for e in sorted(art["events"], key=lambda e: (e["split"], e["event_id"]))],
    )
    firsts = [e["date_first"] for e in art["events"]]
    lasts = [e["date_last"] for e in art["events"]]
    doc.p(f"Rango total de los eventos: {min(firsts)} a {max(lasts)}. Cada evento se recorta a "
          f"su primer día con fuego antes de evaluar (`docs/backtest-2026.md`). Resolución "
          f"{art['resolution_m']:.0f} m, paso diario.")
    _split_audit(doc, inp)


def _split_audit(doc: Doc, inp: Inputs) -> None:
    audit = inp.artifacts.get("split_audit") if inp.artifacts else None
    doc.h(3, "Auditoría de fuga entre splits")
    if not audit:
        doc.p("**No disponible**: estos artefactos no incluyen la auditoría de fuga "
              "(`make report-artifacts`).")
        return
    doc.p(
        f"El split reparte *grupos* de eventos acoplados (<= {audit['max_gap_km']:.0f} km y "
        f"<= {audit['max_gap_days']} días entre extensiones de fuego), no eventos sueltos: "
        f"{audit['n_groups']} grupos independientes. Para cada evento retenido, el vecino más "
        f"cercano en OTRO split:"
    )
    doc.table(
        ["evento", "split", "vecino más cercano", "su split", "distancia (km)",
         "separación (días)"],
        [[str(r["event_id"]), r["split"], str(r["nearest_event_id"]), r["nearest_split"],
          f"{r['gap_km']:.1f}", str(r["gap_days"])] for r in audit["nearest_cross_split"]],
    )
    if audit["violations"]:
        doc.p("**FUGA DETECTADA**: pares acoplados en splits distintos:")
        doc.ul([f"{v['event_a']} ({v['split_a']}) y {v['event_b']} ({v['split_b']}): "
                f"{v['gap_km']:.1f} km, {v['gap_days']} días" for v in audit["violations"]])
    else:
        doc.p("Ningún par de eventos acoplados quedó en splits distintos. Sigue habiendo "
              "dependencia climática gruesa (el mismo episodio sinóptico de enero de 2026 cubre "
              "casi todos los eventos): el split controla la fuga espacial y temporal directa, "
              "no la correlación meteorológica regional.")


def _related(doc: Doc, inp: Inputs) -> None:
    doc.h(2, "8. Documentos relacionados")
    items = []
    for path in sorted(inp.docs_dir.glob("*.md")):
        if path.name == "results.md":
            continue
        first = next((ln[2:].strip() for ln in path.read_text().splitlines()
                      if ln.startswith("# ")), path.stem)
        items.append(f"`docs/{path.name}` — {first}")
    doc.ul(items)


# -------------------------------------------------------------- README
README_START = "<!-- results-summary:start (generado por `make report`, no editar a mano) -->"
README_END = "<!-- results-summary:end -->"


def render_readme_summary(inp: Inputs) -> str:
    """Resumen corto para el README: misma fuente (bench/results/) que el
    reporte completo, así el README no puede desincronizarse a mano."""
    models = _ordered(inp.results)
    if not models:
        return "No hay resultados de backtest en `bench/results/`."
    n = sorted({inp.results[m]["n_events"] for m in models})
    rows = [
        [_label(m)] + [f"{inp.results[m]['aggregate'][k]['point_estimate']:.3f}" for k in METRICS]
        for m in models
    ]
    header = ["Modelo"] + [METRIC_LABEL[k] for k in METRICS]
    table = "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )
    indistinct = [
        METRIC_LABEL[k] for k in METRICS
        if _all_overlap([inp.results[m]["aggregate"][k] for m in models])
    ]
    note = (
        f"Backtest sobre incendios reales de Chile 2025-2026, media sobre "
        f"**{', '.join(str(x) for x in n)} eventos de test**. "
        + (f"En {', '.join(indistinct)} todos los intervalos se solapan. " if indistinct else "")
        + "Con tan pocos eventos **no es una comparación estadísticamente robusta**: ningún "
        "modelo queda demostrado como mejor. Intervalos, mapas, calibración y análisis de "
        "fallas en [`docs/results.md`](docs/results.md)."
    )
    return f"{table}\n\n{note}"


def _all_overlap(cis: list[dict[str, float]]) -> bool:
    return max(c["lower"] for c in cis) <= min(c["upper"] for c in cis)


def update_readme(readme_path: Path, summary: str) -> bool:
    """Reemplaza el bloque entre los marcadores; False si no hay README o
    marcadores (no se toca nada)."""
    if not readme_path.exists():
        return False
    text = readme_path.read_text()
    if README_START not in text or README_END not in text:
        return False
    before, rest = text.split(README_START, 1)
    _, after = rest.split(README_END, 1)
    readme_path.write_text(f"{before}{README_START}\n{summary}\n{README_END}{after}")
    return True


# ---------------------------------------------------------------- principal
def _save(fig: Any, inp: Inputs, filename: str) -> None:
    out = _FIG_DIR[0]
    out.mkdir(parents=True, exist_ok=True)
    fig.set_layout_engine("constrained")
    # sin metadatos de fecha/versión: la salida debe ser reproducible byte a byte
    fig.savefig(out / filename, dpi=110, metadata={"Software": None})
    plt.close(fig)


_FIG_DIR: list[Path] = [Path("docs/figures")]


def build_report(
    results_dir: Path, docs_dir: Path, out_dir: Path | None = None,
    readme_path: Path | None = None,
) -> tuple[Path, Path]:
    """Escribe `results.md`, `results.html` y `figures/*.png` en `out_dir`
    (por defecto `docs_dir`). Devuelve las rutas de md y html."""
    out_dir = out_dir or docs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _FIG_DIR[0] = out_dir / "figures"
    inp = load_inputs(results_dir, docs_dir)

    doc = Doc()
    doc.h(1, "Resultados de PyroCast")
    doc.p(f"**{DISCLAIMER}**")
    doc.quote("Este documento lo genera `make report` a partir de `bench/results/` y `docs/`. "
              "Ningún número se escribió a mano. Léanse junto con `docs/limitations.md`.")
    _summary(doc, inp)
    _comparison(doc, inp)
    _calibration(doc, inp)
    _maps(doc, inp)
    _failures(doc, inp)
    _limitations(doc, inp)
    _methodology(doc, inp)
    _related(doc, inp)

    md_path, html_path = out_dir / "results.md", out_dir / "results.html"
    md_path.write_text(render_markdown(doc))
    html_path.write_text(render_html(doc, out_dir / "figures", "PyroCast — resultados"))
    update_readme(
        readme_path if readme_path is not None else docs_dir.parent / "README.md",
        render_readme_summary(inp),
    )
    return md_path, html_path
