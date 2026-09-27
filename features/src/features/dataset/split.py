"""Split train/val/test reproducible, POR EVENTO -- nunca por píxel ni
por día dentro de un mismo evento (evitaría fuga de datos: días
consecutivos del mismo incendio son casi idénticos, y modelo entrenado
con un día y evaluado con el día siguiente del MISMO evento mediría
memorización, no generalización)."""
import random

DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_VAL_FRAC = 0.15
# el resto (0.15 por defecto) va a test.


def split_events(
    event_ids: list[int],
    seed: int = 42,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> dict[str, list[int]]:
    # ordenar antes de mezclar: el resultado depende solo del CONJUNTO de
    # ids y de la semilla, nunca del orden en que la lista de entrada los
    # trae.
    ids_sorted = sorted(event_ids)
    rng = random.Random(seed)
    shuffled = ids_sorted[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
