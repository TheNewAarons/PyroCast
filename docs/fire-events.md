# Definición de "evento de incendio"

`features/fire_state/` reconstruye, a partir de detecciones activas de
FIRMS (`shared.schemas.FireDetection`), qué detecciones pertenecen al
mismo incendio y qué superficie ocupó ese incendio día a día.

## 1. Clustering espaciotemporal (`features/fire_state/clustering.py`)

Un **evento** es un clúster de detecciones conectadas por una relación de
vecindad simple: dos detecciones A y B son "vecinas" si están a lo sumo
`spatial_eps_m` de distancia (haversine, sobre la esfera terrestre) Y a lo
sumo `temporal_eps` de diferencia temporal. El clustering final es la
componente conexa de esta relación (unión de conjuntos/union-find) — **no**
comparación directa por pares: si A-B son vecinas y B-C son vecinas, A y C
quedan en el mismo evento aunque A-C por sí solas no lo sean. Esto modela
un incendio que se mueve o crece de forma continua en el tiempo (un
"gusano" de detecciones encadenadas), no una bola fija alrededor de un
punto de origen.

Es equivalente a ejecutar DBSCAN con `min_samples=1` sobre una métrica
precomputada que combina distancia espacial y temporal — implementado a
mano en vez de agregar `scikit-learn` como dependencia de `features` (ver
`docs/decisions.md`).

**Parámetros (configurables por llamada, con default documentado):**

| Parámetro | Default | Por qué |
|---|---|---|
| `spatial_eps_m` | 750 m | ~2x el tamaño de píxel nominal de VIIRS (375 m) — une detecciones contiguas del mismo incendio sin fusionar focos separados por más de un par de píxeles VIIRS. |
| `temporal_eps` | 2 días | VIIRS revisita el área ~1 vez/día; 2 días tolera un día de nubosidad perdido sin fusionar episodios de incendio distintos y no relacionados. |

Ninguno de los dos valores viene de una calibración contra incendios
reales de Chile — son heurísticas razonables basadas en la resolución
nominal del sensor, documentadas como tales. `models/evaluation/`
(backtesting, sin implementar) es el lugar donde debería, a futuro,
ajustarse contra eventos reales.

## 2. Rasterización (`features/fire_state/rasterize.py`)

Para cada evento, se produce una máscara binaria (`True`/`False` por
celda de la grilla de trabajo — `features/grid/`) por cada día entre la
primera y la última detección del evento (`event.start_date` a
`event.end_date`, inclusive):

1. **Días con detección real**: cada detección se reproyecta a `grid.crs`
   y se le aplica un buffer circular fijo de radio `buffer_m` (default:
   375 m, el tamaño de píxel nominal de VIIRS). La unión de todos los
   círculos de detecciones de ese día, rasterizada sobre la grilla, es la
   máscara de ese día.
2. **Días sin detección dentro del rango del evento** ("días de hueco",
   p. ej. por nubosidad): se interpola LINEALMENTE el indicador binario
   (0/1) de cada celda entre el día ancla anterior y el siguiente con
   detección real. Como ambos extremos son 0 o 1, el valor interpolado
   solo puede ser 0 si AMBAS anclas son 0 en esa celda — en la práctica,
   esto equivale a tomar la **unión** de las máscaras de las dos anclas
   más cercanas para ese día de hueco.
3. **Nunca se extrapola**: un día de hueco sin ancla en alguno de los dos
   lados (fuera del rango `[start_date, end_date]` del evento) queda
   fuera de la salida — no existe fuera de ese rango — y dentro del
   rango pero sin ancla en un lado (imposible dado que start/end_date
   son por definición días con detección) no ocurre por construcción.

## 3. Simplificación explícita frente a la literatura

WildfireCube (el paper de referencia de este proyecto, ver CLAUDE.md)
reconstruye la superficie quemada mediante **kriging espaciotemporal**
sobre las detecciones — un método geoestadístico que estima la
incertidumbre espacial de la interpolación y puede producir un frente de
fuego suavizado y físicamente más plausible que una unión de círculos.

PyroCast usa en cambio:
- Un **buffer espacial fijo** (no kriging) alrededor de cada detección
  puntual — ignora completamente la incertidumbre de geolocalización real
  del sensor (que varía con el ángulo de barrido) y no captura la forma
  real del frente de fuego entre detecciones cercanas.
- **Interpolación temporal lineal** del indicador binario (equivalente a
  unión de máscaras ancla) en vez de una interpolación espaciotemporal
  conjunta — un incendio que se apaga y luego se reactiva en un lugar
  distinto dentro de la misma ventana `temporal_eps` se rellena como si
  hubiera seguido ardiendo continuamente en ambos lugares durante el
  hueco, lo cual sobreestima la superficie quemada en ese caso.

Esta es una simplificación deliberada de una sola persona, documentada
también en `docs/limitations.md`, no un método validado contra
reconstrucciones reales de perímetros de incendio en Chile.
