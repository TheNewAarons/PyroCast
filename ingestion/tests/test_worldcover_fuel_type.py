"""Tests del mapeo WorldCover -> tipo de combustible."""
import numpy as np
from ingestion.worldcover.fuel_type import (
    CLASS_TO_FUEL_TYPE,
    FUEL_BOSQUE,
    FUEL_PASTIZAL,
    FUEL_TYPE_UNKNOWN,
    map_worldcover_to_fuel_type,
)


def test_all_11_documented_classes_are_mapped():
    assert set(CLASS_TO_FUEL_TYPE.keys()) == {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}


def test_map_worldcover_to_fuel_type_known_classes():
    classes = np.array([[10, 30]])
    result = map_worldcover_to_fuel_type(classes)
    assert result[0, 0] == FUEL_BOSQUE
    assert result[0, 1] == FUEL_PASTIZAL


def test_map_worldcover_to_fuel_type_unmapped_class_is_explicit_unknown():
    # p. ej. un código corrupto o de una versión futura de WorldCover
    classes = np.array([[255]])
    result = map_worldcover_to_fuel_type(classes)
    assert result[0, 0] == FUEL_TYPE_UNKNOWN
