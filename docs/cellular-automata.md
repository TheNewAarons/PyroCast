# Autómata celular de propagación de incendios: fórmula y parámetros

`models/cellular_automata/` implementa un autómata celular probabilístico
inspirado en Rothermel simplificado — NO una implementación de las
ecuaciones de Rothermel (que requieren humedad de combustible, razón de
empaquetamiento, profundidad de lecho de combustible, calor de ignición,
etc., ninguno modelado aquí), sino una regla de transición que captura su
intuición física central: el fuego se propaga más rápido cuesta arriba y
a favor del viento.

## La fórmula

`compute_ignition_probability` devuelve un valor para TODA celda de la
grilla (encendida o no) — es `simulate_fire_spread` quien sobrescribe
las celdas ya en llamas con `P=1.0` (ver más abajo). Conceptualmente,
para cada celda no encendida, la probabilidad de que se encienda en el
paso (día) siguiente es:

```
P(celda se enciende) = 1 - ∏ (1 - p_dir)   sobre los 8 vecinos (Moore)

p_dir = base_spread_prob
        · exp(slope_coefficient · pendiente_nc)
        · exp(wind_coefficient · viento_direccional_nc)
        · flammability[celda]
        (recortado a [0, 1] antes de combinar)
```

donde el producto solo incluye vecinos que están ACTUALMENTE en llamas
(un vecino que no está en llamas no contribuye — equivale a `p_dir=0`
para esa dirección).

### Término 1: vecinos en llamas

El producto `∏(1 - p_dir)` sobre múltiples vecinos en llamas es la forma
estándar de autómata celular para "cualquiera de varias fuentes
independientes puede encender la celda" — la probabilidad combinada
siempre aumenta (o se mantiene igual) con más vecinos en llamas, nunca
disminuye. Verificado directamente: con un vecino en llamas, la
probabilidad es estrictamente menor que con cuatro (mismos parámetros,
terreno/viento/combustible homogéneos).

### Término 2: pendiente (`pendiente_nc`)

```
pendiente_nc = (elevación[celda] - elevación[vecino]) / distancia
distancia = resolución_m · hypot(d_row, d_col)   (ortogonal: resolución_m;
                                                    diagonal: resolución_m·√2)
```

Positiva cuando la celda está más alta que el vecino en llamas (el fuego
sube desde el vecino hacia la celda) — aumenta `p_dir` vía
`exp(slope_coefficient · pendiente_nc)`. Verificado con un caso exacto:
vecino en llamas 10 m más abajo a 100 m de distancia
(`pendiente_nc=0.1`) da `P=0.4475...`; el mismo vecino pero 10 m más
arriba (`pendiente_nc=-0.1`, cuesta abajo) da `P=0.2011...` — cuesta
arriba es más del doble de probable que cuesta abajo, con los parámetros
por defecto.

**Parámetro libre**: `slope_coefficient` (default `4.0`). Mayor valor =
mayor sensibilidad a la pendiente (más diferencia entre cuesta
arriba/abajo).

### Término 3: viento (`viento_direccional_nc`)

```
viento_direccional_nc = (wind_u·(-d_col) + wind_v·d_row) / hypot(d_row, d_col)
```

La proyección del vector de viento `(wind_u, wind_v)` — convención
ERA5-Land: apunta hacia DONDE SOPLA el viento, no de dónde viene
(`u10`=componente hacia el este, `v10`=componente hacia el norte) —
sobre la dirección de propagación (del vecino hacia la celda), SIN
dividir por la velocidad del viento (así que viento cero da exactamente
0, nunca un `0/0`). Positiva cuando el viento sopla en la misma
dirección que la propagación — aumenta `p_dir` vía
`exp(wind_coefficient · viento_direccional_nc)`. Verificado con un caso
exacto: vecino en llamas al oeste con viento de 5 m/s hacia el este (a
favor de la propagación, `viento_direccional_nc=+5`) da `P=0.8155...`;
el mismo viento pero con el vecino en llamas al este (propagación hacia
el oeste, en contra del viento, `viento_direccional_nc=-5`) da
`P=0.1104...`.

**Parámetro libre**: `wind_coefficient` (default `0.2`). Mayor valor =
mayor sensibilidad al viento.

### Término 4: tipo de combustible (`flammability[celda]`)

Un multiplicador en `[0, 1]` de la celda OBJETIVO (no del vecino),
buscado por su código de tipo de combustible simplificado (mismos
códigos que `ingestion/worldcover/fuel_type.py`, duplicados aquí — ver
`docs/decisions.md`):

| Código | Tipo | `flammability` (default) |
|---|---|---|
| 1 | pastizal | 1.0 |
| 2 | matorral | 0.8 |
| 3 | bosque | 0.6 |
| 4 | cultivo | 0.3 |
| 5 | humedal | 0.1 |
| 90-93 | no combustible (urbano/suelo desnudo/agua/nieve-hielo) | 0.0 |
| 99 (o cualquier código no listado) | desconocido | 0.0 |

`0.0` hace que `p_dir=0` para esa celda en TODAS las direcciones —
nunca se enciende, sin importar cuántos vecinos estén en llamas.

**Parámetro libre**: `fuel_flammability` (dict completo, default la
tabla de arriba). Los valores relativos (pastizal > matorral > bosque >
cultivo > humedal) son un ordenamiento razonable pero NO calibrado
contra el comportamiento real de estos combustibles en incendios
chilenos.

## Estado y simulación día a día (`simulate.py`)

Cada celda tiene un estado binario MONÓTONO: `no-en-llamas` o
`en-llamas`. Una vez que una celda se enciende, permanece "en llamas"
(sigue radiando probabilidad de ignición a sus vecinos) para el resto del
horizonte simulado — **no hay modelo de extinción/consumo de
combustible**. Esto es una simplificación deliberada: un incendio real se
apaga cuando consume el combustible disponible; modelar eso requeriría un
término de duración/consumo que este autómata no incluye. Ver
`docs/limitations.md`.

Cada día:
1. Se calcula `P(ignición)` para cada celda no encendida (determinista
   dado el estado actual del día).
2. Se sortea, con un `numpy.random.Generator` sembrado con `seed`, qué
   celdas se encienden REALMENTE ese día (`draws < P`).
3. Se devuelve el array de `P(ignición)` de ese día (no el sorteo
   binario) — las celdas ya en llamas reportan `P=1.0` (estado conocido,
   no se vuelve a sortear).

Misma semilla + mismas entradas → misma trayectoria completa, siempre
(verificado con un test dedicado).

## Parámetros libres — resumen

| Parámetro | Default | Calibrable vía | Calibrado contra incendios reales |
|---|---|---|---|
| `base_spread_prob` | 0.3 | `calibrate.py` (grid search) | No |
| `slope_coefficient` | 4.0 | `calibrate.py` (grid search) | No |
| `wind_coefficient` | 0.2 | `calibrate.py` (grid search) | No |
| `fuel_flammability` | tabla de arriba | No (dict, no escalar — ver limitaciones) | No |

## Calibración (`calibrate.py`)

Grid search simple: para cada combinación de `base_spread_prob` ×
`slope_coefficient` × `wind_coefficient` en una grilla de valores
candidatos, simula cada `TrainingSample` de entrenamiento y promedia
IoU o Brier score (definidas en `models/evaluation/metrics.py` — P8 de
CLAUDE.md, ya en su ubicación final, nunca duplicadas ni movidas después)
contra la máscara observada; devuelve la combinación con mejor score
promedio. Solo calibra los tres parámetros ESCALARES —
`fuel_flammability` es un dict, no un escalar, y calibrarlo por grid
search sería combinatoriamente mucho más caro; queda como trabajo futuro.

## Saturación por recorte (`np.clip(p_dir, 0.0, 1.0)`)

Con los parámetros por defecto, a la resolución de trabajo del proyecto
(250 m, `shared/config.py`), `p_dir` supera 1.0 (spread CIERTO, no
probabilístico) en condiciones que ocurren en incendios chilenos reales
de verano, no solo en casos extremos de laboratorio:

| Condición | `p_dir` sin recortar | Resultado |
|---|---|---|
| pendiente 75 m/celda (16.7°) | 0.996 | al borde |
| pendiente 100 m/celda (21.8°) | 1.486 | **recortado a 1.0** |
| pendiente 150 m/celda (31°) | 3.307 | **recortado a 1.0** |
| viento alineado 5 m/s | 0.815 | al borde |
| viento alineado 8 m/s | 1.486 | **recortado a 1.0** |
| viento alineado 10 m/s | 2.217 | **recortado a 1.0** |
| viento alineado 20 m/s | 16.379 | **recortado a 1.0** |

Es decir: por encima de ~22° de pendiente o ~8 m/s de viento alineado
(30 km/h — común en episodios de viento Puelche que impulsan los
megaincendios de Biobío/Ñuble/Araucanía), el modelo deja de ser
probabilístico y afirma propagación CIERTA hacia esa celda. Ver
`docs/limitations.md` — el mandato de honestidad de CLAUDE.md exige que
esto se documente explícitamente, no que quede implícito en un
`np.clip`.

## Caso analítico verificado: propagación circular

Terreno plano (elevación uniforme), sin viento, combustible homogéneo,
ignición en un único punto central: con `base_spread_prob=0.99` (spread
casi determinista, para aislar la geometría del ruido probabilístico), a
los 10 días la extensión quemada es EXACTAMENTE 10 celdas en las cuatro
direcciones cardinales (norte, sur, este, oeste) desde el punto de
ignición — isotropía perfecta, como se espera de una regla sin ningún
término direccional activo (pendiente y viento ambos neutralizados por
construcción).
