"""Verifica la forma de las tablas Core (sin conexión real a Postgres)."""
from shared.db.schema import evaluation_result, fire_event, metadata, model_run


def test_metadata_has_expected_tables():
    assert set(metadata.tables) == {
        "fire_event",
        "model_run",
        "evaluation_result",
    }


def test_fire_event_columns():
    cols = set(fire_event.columns.keys())
    assert {"id", "bbox", "start_date", "end_date", "source", "geom"} <= cols


def test_model_run_has_fk_to_fire_event():
    fk_targets = {fk.column.table.name for fk in model_run.foreign_keys}
    assert fk_targets == {"fire_event"}


def test_evaluation_result_has_fk_to_model_run():
    fk_targets = {fk.column.table.name for fk in evaluation_result.foreign_keys}
    assert fk_targets == {"model_run"}
