"""Tests de métricas de evaluación: IoU y Brier score. Definidas aquí
directamente (P8, models/evaluation/) en vez de en
models/cellular_automata/ y movidas después -- ver docs/decisions.md."""
import numpy as np
import pytest
from models.evaluation.metrics import brier_score, dice_score, ece_score, iou_score


def test_iou_score_is_one_for_identical_masks():
    mask = np.array([[True, False], [False, True]])
    assert iou_score(mask, mask) == 1.0


def test_iou_score_is_zero_for_disjoint_masks():
    a = np.array([[True, False], [False, False]])
    b = np.array([[False, True], [False, False]])
    assert iou_score(a, b) == 0.0


def test_iou_score_known_partial_overlap():
    # intersección=1, unión=3 -> IoU=1/3
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert iou_score(a, b) == pytest.approx(1 / 3)


def test_iou_score_both_empty_is_one_not_a_division_by_zero():
    empty = np.zeros((3, 3), dtype=bool)
    assert iou_score(empty, empty) == 1.0


def test_brier_score_is_zero_for_perfect_predictions():
    pred = np.array([1.0, 0.0, 1.0])
    true = np.array([1.0, 0.0, 1.0])
    assert brier_score(pred, true) == 0.0


def test_brier_score_known_value():
    # (0.5)^2 promedio = 0.25 para una predicción de 0.5 en todo
    pred = np.full(4, 0.5)
    true = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(pred, true) == pytest.approx(0.25)


def test_brier_score_worst_case_confident_and_wrong():
    pred = np.array([1.0, 0.0])
    true = np.array([0.0, 1.0])
    assert brier_score(pred, true) == 1.0


def test_dice_score_known_partial_overlap():
    # intersección=1, total=2+2=4 -> dice=2*1/4=0.5
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert dice_score(a, b) == pytest.approx(0.5)


def test_dice_score_is_one_for_identical_masks():
    mask = np.array([[True, False], [False, True]])
    assert dice_score(mask, mask) == 1.0


def test_dice_score_both_empty_is_one_not_a_division_by_zero():
    empty = np.zeros((3, 3), dtype=bool)
    assert dice_score(empty, empty) == 1.0


def test_dice_score_treats_any_nonzero_float_as_true():
    # dice_score hace .astype(bool) internamente -- CUALQUIER float
    # distinto de cero (incluida una probabilidad de 0.01 sin
    # umbralizar) cuenta como "True". Documentado como contrato
    # explícito, no un bug sorpresa a descubrir después: quien llame a
    # dice_score/iou_score con probabilidades crudas sin umbralizar
    # primero obtiene un resultado silenciosamente distinto del
    # esperado.
    a = np.array([0.01, 0.0, 0.99])
    b = np.array([1.0, 0.0, 1.0])
    assert dice_score(a, b) == dice_score(
        np.array([True, False, True]), np.array([True, False, True])
    )


def test_ece_score_known_value_two_bins():
    pred = np.array([0.1, 0.4, 0.6, 0.9])
    true = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece_score(pred, true, n_bins=2) == pytest.approx(0.25)


def test_ece_score_is_zero_for_perfect_calibration():
    # cada bin tiene confianza == exactitud exacta
    pred = np.array([0.0, 0.0, 1.0, 1.0])
    true = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece_score(pred, true, n_bins=2) == pytest.approx(0.0)


def test_ece_score_handles_more_bins_than_distinct_values_without_crashing():
    # la mayoría de los 100 bins quedan vacíos -- deben saltarse, no
    # dividir por cero ni sumar un término espurio.
    pred = np.array([0.1, 0.4, 0.6, 0.9])
    true = np.array([0.0, 0.0, 1.0, 1.0])
    result = ece_score(pred, true, n_bins=100)
    assert 0.0 <= result <= 1.0


def test_ece_score_empty_input_is_zero_not_a_crash():
    assert ece_score(np.array([]), np.array([])) == 0.0
