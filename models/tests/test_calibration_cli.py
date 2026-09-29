"""Test de humo del CLI `pyrocast-calibrate`: --fixture corre de
verdad (entrena un checkpoint sintético diminuto vía
models.deep.train.train_model, lo calibra, imprime el reporte
antes/después) sin red ni datos reales -- esto es lo que `make
calibrate` ejecuta. También cubre la ruta real --checkpoint +
--shard-dir (nunca antes ejercitada, ver revisión final del
2026-09-29), incluida su negativa a correr con un split de val vacío."""
from pathlib import Path

import numpy as np
import torch
from features.dataset.assemble import CHANNEL_ORDER
from models.deep.calibration import app
from models.deep.checkpoint import TrainingConfig, save_checkpoint
from models.deep.unet import SmallUNet
from tfrecord_fixtures import write_tfrecord
from typer.testing import CliRunner

runner = CliRunner()

_SIZE = 4


def _make_ndws_record(sample_id: int) -> dict[str, np.ndarray]:
    def const(value: float) -> np.ndarray:
        return np.full((_SIZE, _SIZE), value, dtype="float32")

    rng = np.random.default_rng(sample_id)
    return {
        "elevation": rng.random((_SIZE, _SIZE)).astype("float32") * 1000,
        "pdsi": const(1.0), "NDVI": const(5000.0), "pr": const(5.0), "sph": const(0.005),
        "th": const(90.0), "tmmn": const(280.0), "tmmx": const(300.0), "vs": const(5.0),
        "erc": const(30.0), "population": const(5.0),
        "PrevFireMask": (rng.random((_SIZE, _SIZE)) > 0.8).astype("float32"),
        "FireMask": (rng.random((_SIZE, _SIZE)) > 0.8).astype("float32"),
    }


def _make_real_checkpoint(path: Path) -> None:
    torch.manual_seed(1)
    model = SmallUNet(in_channels=len(CHANNEL_ORDER), base_channels=8, depth=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = TrainingConfig(
        phase="pretrain", in_channels=len(CHANNEL_ORDER), base_channels=8, depth=1,
        lr=1e-3, batch_size=1, seed=1, focal_alpha=0.8, focal_gamma=2.0, max_epochs=1,
        patience=1, data_paths=("fixture",), pretrained_checkpoint=None,
    )
    save_checkpoint(path, model, optimizer, epoch=1, best_val_loss=0.5, config=config)


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


def test_run_with_real_checkpoint_and_shard_dir_completes_end_to_end(tmp_path):
    # nunca antes ejercitada por la suite (revisión final del
    # 2026-09-29) -- 2 shards reales de NDWS para que el split
    # train/val no quede vacío.
    checkpoint_path = tmp_path / "model.pt"
    _make_real_checkpoint(checkpoint_path)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    write_tfrecord(shard_dir / "shard_0.tfrecord", [_make_ndws_record(1), _make_ndws_record(2)])
    write_tfrecord(shard_dir / "shard_1.tfrecord", [_make_ndws_record(3), _make_ndws_record(4)])

    result = runner.invoke(
        app, ["run", "--checkpoint", str(checkpoint_path), "--shard-dir", str(shard_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "brier" in result.output.lower()
    assert (checkpoint_path.with_name("model.calibrator.pt")).exists()


def test_run_with_a_single_shard_refuses_instead_of_validating_on_train(tmp_path):
    # hallazgo IMPORTANTE de la revisión final del 2026-09-29: con un
    # solo shard, split_public_dataset da val=[] -- antes, el CLI caía
    # en silencio a calibrar contra el propio train y lo reportaba como
    # "el set de validación". Ahora se niega, igual que
    # pyrocast-train pretrain.
    checkpoint_path = tmp_path / "model.pt"
    _make_real_checkpoint(checkpoint_path)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    write_tfrecord(shard_dir / "shard_0.tfrecord", [_make_ndws_record(1), _make_ndws_record(2)])

    result = runner.invoke(
        app, ["run", "--checkpoint", str(checkpoint_path), "--shard-dir", str(shard_dir)]
    )
    assert result.exit_code == 1
    assert "val" in result.output.lower()


def test_run_with_a_nonexistent_checkpoint_gives_a_clear_message(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    write_tfrecord(shard_dir / "shard_0.tfrecord", [_make_ndws_record(1)])
    write_tfrecord(shard_dir / "shard_1.tfrecord", [_make_ndws_record(2)])

    result = runner.invoke(
        app,
        ["run", "--checkpoint", str(tmp_path / "missing.pt"), "--shard-dir", str(shard_dir)],
    )
    assert result.exit_code == 1
    assert "checkpoint" in result.output.lower()
