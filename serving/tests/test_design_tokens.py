"""Contraste WCAG AA de los tokens de diseño, calculado (no a ojo) desde `:root`
de serving/web/static/app.css. Los paneles son translúcidos: se evalúan contra el
PEOR fondo posible (el panel compuesto sobre blanco puro)."""
import re
from pathlib import Path

import pytest

CSS = (Path(__file__).resolve().parents[1] / "web" / "static" / "app.css").read_text()
AA_TEXT = 4.5
AA_UI = 3.0


def tokens() -> dict[str, str]:
    root = re.search(r":root\s*\{(.*?)\n\}", CSS, re.S)
    assert root, ":root no encontrado"
    return dict(re.findall(r"--([\w-]+):\s*([^;]+);", root.group(1)))


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def luminance(hex_color: str) -> float:
    def channel(c: int) -> float:
        v = c / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def over(fg: str, alpha: float, bg: str) -> str:
    r, g, b = (
        round(alpha * f + (1 - alpha) * k) for f, k in zip(rgb(fg), rgb(bg), strict=True)
    )
    return f"#{r:02x}{g:02x}{b:02x}"


T = tokens()
SURFACE = T["surface-solid"]
ALPHA = float(T["surface-alpha"])
PANEL_WORST = over(SURFACE, ALPHA, "#ffffff")  # panel sobre un mapa blanco (imposible, cota)
_dim = [round(float(T["map-dim"]) * c) for c in rgb("#3b3b3d")]
DIM_TILE = f"#{_dim[0]:02x}{_dim[1]:02x}{_dim[2]:02x}"


@pytest.mark.parametrize("fg", ["text", "text-strong", "text-dim"])
@pytest.mark.parametrize("bg_name", ["bg", "surface-solid"])
def test_text_meets_aa_on_the_base_surfaces(fg, bg_name):
    assert ratio(T[fg], T[bg_name]) >= AA_TEXT, (fg, bg_name)


@pytest.mark.parametrize("fg", ["text", "text-strong", "text-dim"])
def test_text_meets_aa_even_on_the_worst_case_translucent_panel(fg):
    assert ratio(T[fg], PANEL_WORST) >= AA_TEXT, fg


def test_text_on_the_accent_button_meets_aa():
    assert ratio(T["accent-ink"], T["accent"]) >= AA_TEXT


def test_accent_and_control_borders_meet_the_non_text_minimum():
    assert ratio(T["accent"], SURFACE) >= AA_UI
    assert ratio(T["line-ui"], SURFACE) >= AA_UI
    assert ratio(T["line-ui"], PANEL_WORST) >= AA_UI
    assert ratio(T["focus"], SURFACE) >= AA_UI


def test_every_probability_class_stands_out_from_the_dimmed_base_map():
    for i in range(1, 6):
        assert ratio(T[f"ramp-{i}"], DIM_TILE) >= AA_UI, i


def test_probability_ramp_is_monotonic_in_luminance():
    lums = [luminance(T[f"ramp-{i}"]) for i in range(1, 6)]
    assert lums == sorted(lums) and len(set(lums)) == 5


def test_design_uses_tokens_not_loose_colors_outside_root():
    outside = CSS.split("\n}", 1)[1]
    loose = re.findall(r"#[0-9a-fA-F]{3,8}\b", outside)
    # permitidos: la capa blanca translúcida del banner de error y el hover de zoom
    assert set(loose) <= {"#151d28"}, loose


def test_html_declares_both_font_families_and_a_dark_color_scheme(tmp_path):
    html = (Path(__file__).resolve().parents[1] / "web" / "templates" / "index.html").read_text()
    assert "family=Jost" in html and "family=Martian+Mono" in html
    assert 'name="color-scheme" content="dark"' in html
    assert "Jost" in T["font-sans"] and "Martian Mono" in T["font-mono"]
