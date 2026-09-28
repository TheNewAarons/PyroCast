"""Tests de FocalLoss: valores conocidos calculados a mano, y las
propiedades que la hacen apropiada para máscaras binarias desbalanceadas
(pondera más la clase positiva rara, y castiga menos los negativos
fáciles que BCE simple)."""
import math

import pytest
import torch
from models.deep.losses import FocalLoss


def test_focal_loss_matches_hand_computed_value_for_a_single_pixel():
    # logit=0 -> p=0.5 (sigmoid(0)=0.5). target=1 (positivo).
    # FL = -alpha * (1-p)^gamma * log(p) = -0.8 * 0.5^2 * log(0.5)
    logits = torch.tensor([[0.0]])
    targets = torch.tensor([[1.0]])
    loss_fn = FocalLoss(alpha=0.8, gamma=2.0)
    expected = -0.8 * (0.5**2) * math.log(0.5)
    assert loss_fn(logits, targets).item() == pytest.approx(expected, abs=1e-5)


def test_focal_loss_is_near_zero_for_a_confident_correct_prediction():
    logits = torch.tensor([[10.0]])  # sigmoid(10) ~ 0.99995, muy seguro de "fuego"
    targets = torch.tensor([[1.0]])
    loss_fn = FocalLoss(alpha=0.8, gamma=2.0)
    assert loss_fn(logits, targets).item() < 1e-3


def test_focal_loss_down_weights_easy_negatives_more_than_plain_bce():
    # negativo FÁCIL (logit muy negativo, target=0): el término de
    # focusing (1-p_t)^gamma con p_t~1 debe hacer que FL sea MENOR que
    # BCE simple con el mismo alpha implícito -- esa es la propiedad
    # que justifica elegir focal loss sobre BCE ponderada (ver
    # docs/decisions.md).
    logits = torch.tensor([[-10.0]])
    targets = torch.tensor([[0.0]])
    focal = FocalLoss(alpha=0.8, gamma=2.0)(logits, targets).item()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).item()
    assert focal < bce


def test_focal_loss_averages_over_the_batch():
    logits = torch.tensor([[0.0], [10.0]])
    targets = torch.tensor([[1.0], [1.0]])
    loss_fn = FocalLoss(alpha=0.8, gamma=2.0)
    single_high_conf = FocalLoss(alpha=0.8, gamma=2.0)(
        torch.tensor([[10.0]]), torch.tensor([[1.0]])
    ).item()
    single_low_conf = FocalLoss(alpha=0.8, gamma=2.0)(
        torch.tensor([[0.0]]), torch.tensor([[1.0]])
    ).item()
    expected_mean = (single_high_conf + single_low_conf) / 2
    assert loss_fn(logits, targets).item() == pytest.approx(expected_mean, abs=1e-5)


def test_focal_loss_accepts_2d_spatial_shape_without_a_channel_dim():
    logits = torch.zeros(2, 8, 8)
    targets = torch.zeros(2, 8, 8)
    loss_fn = FocalLoss(alpha=0.8, gamma=2.0)
    result = loss_fn(logits, targets)
    assert result.shape == ()
    assert torch.isfinite(result)
