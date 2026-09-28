"""Tests de la interfaz común de modelo: verifica que un modelo dummy
que implementa `predict(event) -> np.ndarray` satisface el Protocol
estructuralmente (sin heredar de nada)."""
import numpy as np
import xarray as xr
from shared.model_protocol import FireSpreadModel


class _DummyConstantModel:
    """No hereda de FireSpreadModel -- el Protocol es estructural."""

    def predict(self, event: xr.DataArray) -> np.ndarray:
        n_days, _, height, width = event.shape
        return np.full((n_days, height, width), 0.5, dtype="float64")


def test_dummy_model_satisfies_the_protocol_structurally():
    model = _DummyConstantModel()
    assert isinstance(model, FireSpreadModel)


def test_protocol_typed_function_accepts_the_dummy_model():
    def run(model: FireSpreadModel) -> np.ndarray:
        fake_event = xr.DataArray(
            np.zeros((2, 3, 4, 5), dtype="float32"),
            dims=("day", "channel", "y", "x"),
        )
        return model.predict(fake_event)

    result = run(_DummyConstantModel())
    assert result.shape == (2, 4, 5)
    assert np.all(result == 0.5)
