"""Test de humo del CLI `pyrocast-calibrate`: --fixture corre de
verdad (entrena un checkpoint sintético diminuto vía
models.deep.train.train_model, lo calibra, imprime el reporte
antes/después) sin red ni datos reales -- esto es lo que `make
calibrate` ejecuta."""
from models.deep.calibration import app
from typer.testing import CliRunner

runner = CliRunner()


def test_run_fixture_completes_and_prints_before_after_report(tmp_path):
    result = runner.invoke(app, ["run", "--fixture", "--run-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "brier" in result.output.lower()
    assert "ece" in result.output.lower()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "best.calibrator.pt").exists()


def test_run_help_mentions_fixture_and_checkpoint_options():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--fixture" in result.output
    assert "--checkpoint" in result.output
