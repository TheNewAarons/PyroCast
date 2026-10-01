"""Ninguna credencial en el repo: escanea los archivos versionados en busca
de patrones de claves reales (no de los placeholders de test/ejemplo)."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SKIP_SUFFIXES = {".png", ".npz", ".lock", ".pt", ".tif", ".nc"}
PATTERNS = {
    "clave hex de 32 caracteres (p. ej. MAP_KEY de FIRMS)":
        re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{32}(?![0-9A-Za-z])"),
    "UUID (p. ej. API key de CDS)": re.compile(
        r"(?<![0-9a-f-])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f-])"
    ),
    "clave AWS": re.compile(r"AKIA[0-9A-Z]{16}"),
    "clave privada": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "asignación de credencial con valor real": re.compile(
        r"(FIRMS_MAP_KEY|CDS_API_KEY|CLIENT_SECRET|POSTGRES_PASSWORD)\s*[=:]\s*['\"]?"
        r"(?![-\s]*(?:changeme|dev-placeholder|ci-placeholder|x['\"]?$|test|<|\$|\{))[A-Za-z0-9_\-]{12,}"
    ),
}


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("no es un repositorio git")
    return [ROOT / line for line in out.splitlines() if (ROOT / line).suffix not in _SKIP_SUFFIXES]


def test_no_credential_like_strings_are_tracked_in_git():
    findings = []
    for path in _tracked_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(f"{path.relative_to(ROOT)}: {label}: {match.group(0)[:12]}…")
    assert not findings, "posibles credenciales versionadas:\n" + "\n".join(findings)


def test_env_files_are_ignored_and_not_tracked():
    tracked = {p.name for p in _tracked_files()}
    assert ".env" not in tracked
    gitignore = (ROOT / ".gitignore").read_text()
    assert ".env" in gitignore.splitlines()
