"""Normalización fija de entradas del U-Net (revisión independiente, hallazgo H1:
las entradas crudas mezclaban escalas de 1e-3 a 1e3 y códigos de combustible
categóricos tratados como números)."""
import pytest
import torch
from features.dataset.assemble import CHANNEL_ORDER
from models.deep.checkpoint import TrainingConfig, load_checkpoint, save_checkpoint
from models.deep.normalization import normalize_inputs
from models.deep.unet import SmallUNet


def _raw(**values: float) -> torch.Tensor:
    x = torch.zeros(1, len(CHANNEL_ORDER), 2, 2)
    for name, value in values.items():
        x[0, CHANNEL_ORDER.index(name)] = value
    return x


def _chan(x: torch.Tensor, name: str) -> float:
    return float(x[0, CHANNEL_ORDER.index(name), 0, 0])


def test_none_is_the_identity():
    x = _raw(elevation=2500.0, temperature=305.0)
    assert torch.equal(normalize_inputs(x, "none"), x)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="input_norm"):
        normalize_inputs(_raw(), "zscore-of-who-knows")


def test_v1_brings_realistic_raw_values_to_order_one():
    raw = _raw(elevation=2500.0, slope_deg=30.0, aspect_deg=180.0, wind_u=10.0, wind_v=-10.0,
               temperature=305.0, relative_humidity=100.0, precipitation=0.03, ndvi=0.8,
               fire_mask=1.0)
    out = normalize_inputs(raw, "v1")
    for name in ("elevation", "slope_deg", "aspect_deg", "wind_u", "wind_v", "temperature",
                 "relative_humidity", "precipitation", "ndvi", "fire_mask"):
        assert abs(_chan(out, name)) <= 3.0, name
    assert _chan(out, "fire_mask") == 1.0  # la máscara no se toca
    assert _chan(out, "wind_u") == pytest.approx(2.0)


@pytest.mark.parametrize("code, expected", [(1, 1.0), (3, 0.6), (4, 0.3), (92, 0.0), (99, 0.0),
                                            (0, 0.0), (57, 0.0)])
def test_v1_encodes_fuel_type_codes_by_flammability_not_as_ordinal_numbers(code, expected):
    out = normalize_inputs(_raw(fuel_type=float(code)), "v1")
    assert _chan(out, "fuel_type") == pytest.approx(expected)


def test_unet_with_v1_runs_on_raw_inputs_and_adds_no_parameters_or_state():
    plain = SmallUNet(in_channels=len(CHANNEL_ORDER), base_channels=8, depth=2)
    normed = SmallUNet(in_channels=len(CHANNEL_ORDER), base_channels=8, depth=2, input_norm="v1")
    assert plain.state_dict().keys() == normed.state_dict().keys()
    out = normed(_raw(elevation=2500.0, temperature=305.0, fuel_type=1.0).repeat(1, 1, 8, 8))
    assert torch.isfinite(out).all() and out.shape == (1, 1, 16, 16)


def test_checkpoint_roundtrip_preserves_input_norm_and_old_configs_default_to_none(tmp_path):
    config = TrainingConfig(
        phase="finetune", in_channels=len(CHANNEL_ORDER), base_channels=8, depth=2, lr=1e-3,
        batch_size=1, seed=1, focal_alpha=0.8, focal_gamma=2.0, max_epochs=1, patience=1,
        data_paths=("x",), pretrained_checkpoint=None, input_norm="v1",
    )
    model = SmallUNet(in_channels=len(CHANNEL_ORDER), base_channels=8, depth=2, input_norm="v1")
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(tmp_path / "c.pt", model, optimizer, 1, 0.5, config)
    loaded, _opt, cfg = load_checkpoint(tmp_path / "c.pt")
    assert loaded.input_norm == "v1" and cfg.input_norm == "v1"
    # un checkpoint ANTERIOR (sin el campo) se carga sin normalizar, como se entrenó
    old = TrainingConfig(**{**config.__dict__, "input_norm": "none"})
    object.__delattr__(old, "input_norm")  # simula un pickle viejo
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": {},
                "epoch": 1, "best_val_loss": 0.5, "config": old}, tmp_path / "old.pt")
    assert load_checkpoint(tmp_path / "old.pt")[0].input_norm == "none"
