"""Tests del split train/val/test reproducible POR EVENTO."""
from features.dataset.split import split_events


def test_split_events_no_event_appears_in_two_splits():
    event_ids = list(range(20))
    splits = split_events(event_ids, seed=1)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(event_ids)


def test_split_events_is_reproducible_with_the_same_seed():
    event_ids = list(range(50))
    first = split_events(event_ids, seed=7)
    second = split_events(event_ids, seed=7)
    assert first == second


def test_split_events_handles_small_event_count_without_crashing():
    event_ids = [101, 102, 103]
    splits = split_events(event_ids, seed=1)
    all_assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_assigned) == event_ids


def test_split_events_guarantees_at_least_one_per_split_when_n_is_at_least_three():
    # Antes del fix, n=3..5 podian dejar val o test completamente vacios
    # (contradiciendo el 70/15/15 documentado) -- verificado en la
    # revision final del 2026-09-27. A partir de 3 eventos, cada split
    # debe tener al menos 1.
    for n in range(3, 12):
        splits = split_events(list(range(n)), seed=1)
        assert len(splits["train"]) >= 1
        assert len(splits["val"]) >= 1
        assert len(splits["test"]) >= 1
        assert sum(len(v) for v in splits.values()) == n


def test_split_events_with_fewer_than_three_events_puts_everything_in_train():
    # Sin suficientes eventos para un split con sentido, todo va a train
    # -- documentado explicitamente, no un accidente silencioso.
    assert split_events([1], seed=1) == {"train": [1], "val": [], "test": []}
    assert sorted(split_events([1, 2], seed=1)["train"]) == [1, 2]


def test_split_events_input_order_does_not_change_the_result():
    # el split depende del contenido del conjunto de ids, no del orden en
    # que la lista de entrada los trae (los ids se ordenan antes de
    # mezclar con la semilla) -- dos llamadas con los mismos ids en
    # distinto orden de entrada deben dar el mismo split.
    forward = split_events([1, 2, 3, 4, 5], seed=3)
    shuffled_input = split_events([5, 3, 1, 4, 2], seed=3)
    assert forward == shuffled_input
