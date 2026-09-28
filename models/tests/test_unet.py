"""Tests de SmallUNet: forma de entrada/salida, tamaños arbitrarios de
H/W (no solo potencias de 2 -- los eventos reales de PyroCast tienen el
tamaño de su propio bbox, nunca garantizado divisible por 2**depth),
batch_size > 1, y la validación de base_channels."""
import pytest
import torch
from models.deep.unet import SmallUNet


def test_forward_preserves_batch_and_spatial_dims_returns_one_channel():
    model = SmallUNet(in_channels=11, base_channels=16, depth=3)
    x = torch.randn(2, 11, 32, 32)
    y = model(x)
    assert y.shape == (2, 1, 32, 32)


def test_forward_handles_a_non_power_of_two_size():
    # 37x29 no es divisible por 2**3=8 en ningún eje -- el pad/crop del
    # camino de subida debe manejarlo sin lanzar un error de forma.
    model = SmallUNet(in_channels=11, base_channels=16, depth=3)
    x = torch.randn(1, 11, 37, 29)
    y = model(x)
    assert y.shape == (1, 1, 37, 29)


def test_forward_works_with_batch_size_greater_than_one():
    model = SmallUNet(in_channels=11, base_channels=8, depth=2)
    x = torch.randn(4, 11, 16, 16)
    y = model(x)
    assert y.shape == (4, 1, 16, 16)


def test_in_channels_is_configurable():
    model = SmallUNet(in_channels=5, base_channels=8, depth=2)
    x = torch.randn(1, 5, 16, 16)
    y = model(x)
    assert y.shape == (1, 1, 16, 16)


def test_rejects_base_channels_not_a_multiple_of_eight():
    # GroupNorm(num_groups=8) requiere que cada nivel del canal sea
    # divisible por 8 -- ver Global Constraints del plan.
    with pytest.raises(ValueError, match="8"):
        SmallUNet(in_channels=11, base_channels=10, depth=2)


def test_output_is_finite_for_a_realistic_forward_and_backward_pass():
    model = SmallUNet(in_channels=11, base_channels=16, depth=3)
    x = torch.randn(1, 11, 24, 24, requires_grad=False)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert torch.isfinite(y).all()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} sin gradiente -- capa desconectada"
        assert torch.isfinite(param.grad).all(), f"{name} tiene gradiente no finito"
