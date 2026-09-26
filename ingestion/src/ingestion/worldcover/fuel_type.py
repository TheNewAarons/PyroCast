"""Mapeo heurístico de clase de cobertura ESA WorldCover a "tipo de
combustible" simplificado — usado por el autómata celular y el U-Net
como proxy grosero de carga/tipo de combustible, NO un mapa de
combustibles forestales validado en terreno (p. ej. no distingue los
sistemas Fireline/Behave/Scott-Burgan).

*** LIMITACIÓN EXPLÍCITA: WorldCover no distingue bosque nativo de
plantación forestal (ambos caen en la clase 10 "Tree cover") — una
distinción crítica para el comportamiento del fuego en Biobío/Ñuble/
Araucanía (plantaciones de Pinus radiata y Eucalyptus se comportan de
forma muy distinta a bosque nativo de Nothofagus). Separar ambos
requeriría una fuente adicional (p. ej. catastro de CONAF), no
integrada en este proyecto. ***
"""
import numpy as np

FUEL_PASTIZAL = 1
FUEL_MATORRAL = 2
FUEL_BOSQUE = 3
FUEL_CULTIVO = 4
FUEL_HUMEDAL = 5
FUEL_URBANO_NO_COMBUSTIBLE = 90
FUEL_SUELO_DESNUDO_NO_COMBUSTIBLE = 91
FUEL_AGUA_NO_COMBUSTIBLE = 92
FUEL_NIEVE_HIELO_NO_COMBUSTIBLE = 93
FUEL_TYPE_UNKNOWN = 99

FUEL_TYPE_LABELS: dict[int, str] = {
    FUEL_PASTIZAL: "pastizal",
    FUEL_MATORRAL: "matorral",
    FUEL_BOSQUE: "bosque (nativo o plantación — no distinguible)",
    FUEL_CULTIVO: "cultivo",
    FUEL_HUMEDAL: "humedal",
    FUEL_URBANO_NO_COMBUSTIBLE: "urbano/no combustible",
    FUEL_SUELO_DESNUDO_NO_COMBUSTIBLE: "suelo desnudo/no combustible",
    FUEL_AGUA_NO_COMBUSTIBLE: "agua/no combustible",
    FUEL_NIEVE_HIELO_NO_COMBUSTIBLE: "nieve o hielo/no combustible",
    FUEL_TYPE_UNKNOWN: "desconocido (clase WorldCover no mapeada)",
}

# Las 11 clases documentadas de ESA WorldCover 2021 v200, verificadas
# contra https://collections.sentinel-hub.com/worldcover/readme.html
CLASS_TO_FUEL_TYPE: dict[int, int] = {
    10: FUEL_BOSQUE,                        # Tree cover
    20: FUEL_MATORRAL,                       # Shrubland
    30: FUEL_PASTIZAL,                       # Grassland
    40: FUEL_CULTIVO,                        # Cropland
    50: FUEL_URBANO_NO_COMBUSTIBLE,           # Built-up
    60: FUEL_SUELO_DESNUDO_NO_COMBUSTIBLE,    # Bare / sparse vegetation
    70: FUEL_NIEVE_HIELO_NO_COMBUSTIBLE,      # Snow and ice
    80: FUEL_AGUA_NO_COMBUSTIBLE,             # Permanent water bodies
    90: FUEL_HUMEDAL,                         # Herbaceous wetland
    95: FUEL_HUMEDAL,                         # Mangroves
    # Moss and lichen (proxy: vegetación rala no combustible)
    100: FUEL_NIEVE_HIELO_NO_COMBUSTIBLE,
}


def map_worldcover_to_fuel_type(classes: np.ndarray) -> np.ndarray:
    result = np.full(classes.shape, FUEL_TYPE_UNKNOWN, dtype="int32")
    for wc_class, fuel_type in CLASS_TO_FUEL_TYPE.items():
        result[classes == wc_class] = fuel_type
    return result
