# Definición de "evento de incendio"

`features/fire_state/` reconstruye, a partir de detecciones activas de
FIRMS (`shared.schemas.FireDetection`), qué detecciones pertenecen al
mismo incendio y qué EXTENSIÓN ACTIVA de fuego tuvo ese incendio cada día
— **no** la superficie quemada acumulada. Cada máscara diaria es
independiente de las anteriores: una celda que ardió ayer y no hoy vuelve
a `False`, no queda marcada como "ya quemada". Reconstruir la superficie
quemada acumulada (unión de todas las máscaras diarias del evento hasta
la fecha) es responsabilidad de quien consuma esta salida, no de
`features/fire_state/` — un consumidor que necesite saber "¿esta celda ya
ardió alguna vez en este evento?" (p. ej. un autómata celular que no debe
reencender una celda ya consumida) debe calcular esa unión acumulada él
mismo.

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

**El "encadenamiento" (chaining) no tiene, en la práctica, un límite
temporal ni espacial fijo**, aunque `temporal_eps`/`spatial_eps_m` sí lo
son para un PAR de detecciones — es la propiedad por diseño de la
componente conexa (sección anterior): una secuencia de detecciones cada
una a ≤2 días y ≤750 m de la anterior puede encadenar un evento arbitrariamente
largo en el tiempo y en el espacio (verificado en la revisión final:
16 detecciones cada 1.9 días encadenan un evento de 30 días completo;
120 detecciones cada 700 m encadenan un evento de 83 km). Esto es
exactamente lo que el diseño busca (un incendio real se mueve y crece de
forma continua), pero significa que la afirmación "no fusiona episodios
distintos dentro de la misma ventana de `temporal_eps`" solo es cierta
detección-a-detección, no para el evento completo — un incendio disperso
por nubosidad/humo (común bajo el viento Puelche que impulsa los
megaincendios chilenos) puede fragmentarse en más eventos de los reales,
y a la inversa, una cadena de detecciones separadas puede unir episodios
que un observador humano consideraría distintos. Ninguna de las dos
direcciones de error está acotada por los parámetros por sí solos.

## 2. Rasterización (`features/fire_state/rasterize.py`)

Para cada evento, se produce una máscara binaria (`True`/`False` por
celda de la grilla de trabajo — `features/grid/`) por cada día entre la
primera y la última detección del evento (`event.start_date` a
`event.end_date`, inclusive):

1. **Días con detección real**: cada detección se reproyecta a `grid.crs`
   y se le aplica un buffer circular fijo de **radio** `buffer_m`
   (default: 375 m). La unión de todos los círculos de detecciones de ese
   día, rasterizada sobre la grilla, es la máscara de ese día.

   **`buffer_m` es un radio, no el tamaño de píxel**: un círculo de
   radio 375 m cubre ~441 786 m², ~3.1 veces el área de un píxel VIIRS de
   375×375 m (140 625 m²) — no es un intento (fallido) de igualar el área
   del píxel, es una sobre-cobertura deliberada para compensar la
   incertidumbre real de geolocalización del sensor y el crecimiento del
   píxel fuera de nadir (ninguno de los dos modelado explícitamente
   aquí). Sin calibrar contra incendios reales de Chile — ver
   `docs/limitations.md`.

   Un día con detecciones que termina sin ningún píxel marcado (p. ej.
   `buffer_m` menor a medio píxel de la grilla, o una detección fuera de
   sus bounds) levanta `ValueError` en vez de devolver una máscara
   toda-`False` indistinguible de "no hubo fuego real" ese día.
2. **Días sin detección dentro del rango del evento** ("días de hueco",
   p. ej. por nubosidad): se interpola LINEALMENTE el indicador binario
   (0/1) de cada celda entre el día ancla anterior y el siguiente con
   detección real. Como ambos extremos son 0 o 1, el valor interpolado
   solo puede ser 0 si AMBAS anclas son 0 en esa celda — en la práctica,
   esto equivale a tomar la **unión** de las máscaras de las dos anclas
   más cercanas para ese día de hueco.
3. **Nunca se extrapola**: `fill_temporal_gaps` deja vacío (todo `False`)
   cualquier día sin ancla en alguno de los dos lados del rango pedido —
   incluyendo un rango más ancho que `[event.start_date, event.end_date]`
   pasado explícitamente. `build_fire_state` (el camino habitual, usado
   por `features/dataset/`) llama a `fill_temporal_gaps` exactamente con
   `event.start_date`/`event.end_date` como rango, y esos dos son por
   definición días con detección — así que, EN ESE CAMINO ESPECÍFICO, el
   caso de un día sin ancla en un solo lado dentro del rango pedido no se
   presenta nunca. La función en sí (llamada directamente, con un rango
   más ancho) sí implementa y prueba ese caso explícitamente.

## 3. Simplificación explícita frente a la literatura

WildfireCube (el paper de referencia de este proyecto, ver CLAUDE.md)
reconstruye la superficie quemada mediante **kriging espaciotemporal**
sobre las detecciones — un método geoestadístico que estima la
incertidumbre espacial de la interpolación y puede producir un frente de
fuego suavizado y físicamente más plausible que una unión de círculos.

PyroCast usa en cambio, y produce una extensión ACTIVA de fuego por día
(no una superficie quemada acumulada, ver arriba):
- Un **buffer espacial fijo** (no kriging) alrededor de cada detección
  puntual — ignora completamente la incertidumbre de geolocalización real
  del sensor (que varía con el ángulo de barrido) y no captura la forma
  real del frente de fuego entre detecciones cercanas. El radio elegido
  (375 m) ya sobre-cubre ~3.1x el área nominal de un píxel VIIRS por
  diseño (ver arriba) — no es una fuente adicional de error no
  disclosed, pero sí un sesgo sistemático hacia extensiones más grandes
  que las reales.
- **Interpolación temporal lineal** del indicador binario (equivalente a
  unión de máscaras ancla) en vez de una interpolación espaciotemporal
  conjunta — un incendio que se apaga y luego se reactiva en un lugar
  distinto dentro de la misma ventana `temporal_eps` se rellena como si
  hubiera seguido ardiendo continuamente en ambos lugares durante el
  hueco, lo cual sobreestima la extensión activa en ese caso.

Esta es una simplificación deliberada de una sola persona, documentada
también en `docs/limitations.md`, no un método validado contra
reconstrucciones reales de perímetros de incendio en Chile.
