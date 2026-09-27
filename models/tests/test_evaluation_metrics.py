"""Tests de métricas de evaluación: IoU y Brier score. Definidas aquí
directamente (P8, models/evaluation/) en vez de en
models/cellular_automata/ y movidas después -- ver docs/decisions.md."""
import numpy as np
import pytest
from models.evaluation.metrics import brier_score, iou_score


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
