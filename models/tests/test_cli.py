"""Test de humo del CLI `pyrocast-models run-ca`: simula un evento de
fixture construido en el propio comando, sin depender de un Zarr real de
features/dataset/."""
import time

from models.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_run_ca_cli_completes_quickly_with_default_fixture():
    start = time.monotonic()
    result = runner.invoke(app, ["run-ca"])
    elapsed = time.monotonic() - start
    assert result.exit_code == 0, result.output
    assert elapsed < 5.0
    assert "Día 1:" in result.output
    assert "Día 10:" in result.output


def test_run_ca_cli_respects_n_days_option():
    result = runner.invoke(app, ["run-ca", "--n-days", "3"])
    assert result.exit_code == 0, result.output
    assert "Día 3:" in result.output
    assert "Día 4:" not in result.output
