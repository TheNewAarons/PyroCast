"""Normalización FIJA de las entradas del U-Net.

Hallazgo H1 de la revisión independiente (`docs/review.md`): el U-Net recibía
el tensor crudo, con canales de escalas enormemente distintas (elevación en
cientos de metros, temperatura en ~290 K, precipitación en ~1e-3 m, viento en
unidades de m/s) y `fuel_type` como un código de clase (1..99) tratado como un
número ordinal. Con `batch_size=1` y sin estadísticas de entrada, la señal que
importa (la máscara de fuego 0/1) quedaba aplastada por canales de magnitud
1e3. Se aplica una transformación FIJA y documentada (constantes físicas, NO
estadísticos ajustados a los datos -- así no hay nada que pueda filtrarse entre
splits):

    elevation          / 1000        (km)
    slope_deg          / 30
    aspect_deg         / 360         (circular; el centinela -1 de terreno plano
                                      queda ~0 -- simplificación documentada)
    wind_u, wind_v     / 5           (m/s)
    temperature        (K - 290) / 10
    relative_humidity  / 100         (% -> fracción)
    precipitation      * 100         (m -> cm; ~[0, 3])
    ndvi, fire_mask    sin cambio
    fuel_type          código -> flamabilidad en [0, 1] (la misma tabla del
                       autómata celular, `DEFAULT_FUEL_FLAMMABILITY`; un código
                       desconocido o 0 -> 0)

Es un modo explícito del modelo (`input_norm`), guardado en el checkpoint:
los checkpoints anteriores sin el campo se siguen cargando como "none".
"""
import torch
from features.dataset.assemble import CHANNEL_ORDER

from models.cellular_automata.rules import DEFAULT_FUEL_FLAMMABILITY

INPUT_NORM_MODES = ("none", "v1")
_SCALE_SHIFT: dict[str, tuple[float, float]] = {
    # canal: (shift, scale) con x_norm = (x - shift) / scale
    "elevation": (0.0, 1000.0),
    "slope_deg": (0.0, 30.0),
    "aspect_deg": (0.0, 360.0),
    "wind_u": (0.0, 5.0),
    "wind_v": (0.0, 5.0),
    "temperature": (290.0, 10.0),
    "relative_humidity": (0.0, 100.0),
    "precipitation": (0.0, 0.01),
}
_FUEL_CODES = 100


def _fuel_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    table = torch.zeros(_FUEL_CODES, dtype=dtype, device=device)
    for code, flammability in DEFAULT_FUEL_FLAMMABILITY.items():
        table[code] = flammability
    return table


def normalize_inputs(x: torch.Tensor, mode: str) -> torch.Tensor:
    """`x`: (batch, canal, alto, ancho) en el orden de `CHANNEL_ORDER`."""
    if mode not in INPUT_NORM_MODES:
        raise ValueError(f"input_norm debe ser uno de {INPUT_NORM_MODES}, recibido {mode!r}")
    if mode == "none":
        return x
    if x.shape[1] != len(CHANNEL_ORDER):
        raise ValueError(
            f"input_norm='v1' espera {len(CHANNEL_ORDER)} canales (CHANNEL_ORDER), "
            f"recibido {x.shape[1]}"
        )
    out = x.clone()
    for name, (shift, scale) in _SCALE_SHIFT.items():
        idx = CHANNEL_ORDER.index(name)
        out[:, idx] = (x[:, idx] - shift) / scale
    fuel_idx = CHANNEL_ORDER.index("fuel_type")
    codes = x[:, fuel_idx].round().long().clamp(0, _FUEL_CODES - 1)
    out[:, fuel_idx] = _fuel_table(x.device, x.dtype)[codes]
    return out
