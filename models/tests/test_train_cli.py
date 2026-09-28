"""Test de humo del CLI `pyrocast-train`: smoke-test corre de verdad
(datos sintéticos en memoria, sin red, sin datos reales) y dentro del
tiempo esperado en este hardware (ver docs/model-card.md)."""
import time

from models.deep.train import app
from typer.testing import CliRunner

runner = CliRunner()


def test_smoke_test_command_completes_quickly_and_writes_artifacts(tmp_path):
    start = time.monotonic()
    result = runner.invoke(app, ["smoke-test", "--run-dir", str(tmp_path)])
    elapsed = time.monotonic() - start

    assert result.exit_code == 0, result.output
    assert elapsed < 60, f"smoke-test tardó {elapsed:.1f}s -- ver docs/model-card.md"
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "loss_curve.png").exists()


def test_smoke_test_help_documents_all_three_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "pretrain" in result.output
    assert "finetune" in result.output
    assert "smoke-test" in result.output
