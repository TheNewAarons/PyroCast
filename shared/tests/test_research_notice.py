"""CLAUDE.md exige el aviso "Herramienta de investigación..." visible en todo README e
informe: se verifica en README.md y en CADA docs/*.md (en sus primeras líneas, no
enterrado al final)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTICE = (
    "Herramienta de investigación. No usar para decisiones operativas de combate de "
    "incendios sin validación de CONAF/SENAPRED."
)


def _head(path: Path, n: int = 12) -> str:
    text = " ".join(path.read_text().splitlines()[:n])
    for noise in (">", "*"):  # citas y negritas de Markdown
        text = text.replace(noise, " ")
    return " ".join(text.split())


def test_readme_has_the_notice_at_the_very_top():
    assert NOTICE in _head(ROOT / "README.md", 8)


def test_every_doc_has_the_notice_near_the_top():
    docs = sorted((ROOT / "docs").glob("*.md"))
    assert len(docs) >= 10
    missing = [d.name for d in docs if NOTICE not in _head(d)]
    assert not missing, f"sin aviso de herramienta de investigación arriba: {missing}"


def test_generated_html_report_has_the_notice():
    html = (ROOT / "docs" / "results.html").read_text().replace("<strong>", "").replace(
        "</strong>", "")
    assert NOTICE in " ".join(html.split())
