"""Tests de guardado/carga de checkpoint: la configuración exacta viaja
con el checkpoint (hiperparámetros + rutas de datos usadas), y cargar
un checkpoint reproduce EXACTAMENTE las mismas predicciones que el
modelo antes de guardarlo."""
import torch
from models.deep.checkpoint import TrainingConfig, load_checkpoint, save_checkpoint
from models.deep.unet import SmallUNet


def _make_config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = dict(
        phase="pretrain", in_channels=5, base_channels=8, depth=2, lr=1e-3,
        batch_size=1, seed=42, focal_alpha=0.8, focal_gamma=2.0, max_epochs=10,
        patience=3, data_paths=("fixture/shard_0.tfrecord",), pretrained_checkpoint=None,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def test_save_and_load_checkpoint_reproduces_identical_predictions(tmp_path):
    torch.manual_seed(0)
    model = SmallUNet(in_channels=5, base_channels=8, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = _make_config()

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=3, best_val_loss=0.5, config=config)

    model.eval()
    x = torch.randn(1, 5, 16, 16)
    with torch.no_grad():
        original_output = model(x)

    loaded_model, _optimizer_state, loaded_config = load_checkpoint(path)
    loaded_model.eval()
    with torch.no_grad():
        loaded_output = loaded_model(x)

    assert torch.equal(original_output, loaded_output)
    assert loaded_config == config


def test_load_checkpoint_rebuilds_architecture_from_its_own_config_not_a_guess(tmp_path):
    # el checkpoint se guarda con in_channels=5, base_channels=8 --
    # cargarlo NO debe requerir que el caller ya sepa esos valores de
    # antemano (ver Review Focus del plan: pretrain -> finetune debe
    # funcionar sin que finetune adivine la arquitectura).
    model = SmallUNet(in_channels=5, base_channels=8, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = _make_config(in_channels=5, base_channels=8, depth=2)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1, best_val_loss=1.0, config=config)

    loaded_model, _optimizer_state, loaded_config = load_checkpoint(path)
    assert loaded_config.in_channels == 5
    assert loaded_config.base_channels == 8
    x = torch.randn(1, 5, 8, 8)
    loaded_model(x)  # no lanza -- la arquitectura reconstruida coincide


def test_checkpoint_config_records_data_paths_and_phase(tmp_path):
    model = SmallUNet(in_channels=3, base_channels=8, depth=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = _make_config(
        phase="finetune", in_channels=3, base_channels=8, depth=1,
        data_paths=("data/processed/dataset",), pretrained_checkpoint="runs/pretrain/best.pt",
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1, best_val_loss=0.1, config=config)

    _model, _optimizer_state, loaded_config = load_checkpoint(path)
    assert loaded_config.phase == "finetune"
    assert loaded_config.data_paths == ("data/processed/dataset",)
    assert loaded_config.pretrained_checkpoint == "runs/pretrain/best.pt"
