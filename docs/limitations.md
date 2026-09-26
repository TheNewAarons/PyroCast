# Limitaciones conocidas

Este documento se actualiza con cada hallazgo real de la evaluación
contra incendios de Chile. Nunca se suaviza ni se elimina una métrica
negativa para que el proyecto "se vea mejor" (ver CLAUDE.md).

## Limitaciones de diseño (conocidas desde el bootstrap, no hallazgos de evaluación)

- **Resolución de ERA5-Land vs. grilla de trabajo**: ERA5-Land tiene
  resolución nativa de ~9 km. Se interpola a la grilla de 250 m del
  proyecto, lo que introduce un artefacto de downscaling — los campos de
  viento/temperatura/humedad/precipitación tendrán variabilidad
  espacial artificialmente suave dentro de cada celda de 9 km original.
  Esto debe mencionarse explícitamente en cualquier resultado que use
  clima como insumo.
- **Resolución espacio-temporal reducida frente a la literatura**: el
  paper de referencia (WildfireCube) trabaja a 30 m / 3 h. PyroCast usa
  250 m / diario por ser un proyecto de una sola persona; esto es una
  simplificación deliberada, no una réplica del estado del arte, y los
  resultados no son directamente comparables a los de ese paper.
- **`bbox` de eventos como texto libre en `fire_event`**: la columna
  `bbox` es un `String` (no un tipo estructurado) en este bootstrap
  inicial; si se necesita indexar o filtrar espacialmente por bbox más
  adelante, migrar a un tipo estructurado o derivarlo de `geom`.
- **`docker compose up` no verificado end-to-end**: el bootstrap inicial
  se hizo en un entorno sin Docker instalado (ver `docs/decisions.md`).
  El Dockerfile y compose se revisaron estáticamente y se simuló la
  resolución de dependencias fuera de Docker, pero nadie ha confirmado
  todavía que el contenedor `api` realmente arranca y pasa su healthcheck
  dentro de Docker real.

## Herramienta de investigación

Herramienta de investigación. No usar para decisiones operativas de
combate de incendios sin validación de CONAF/SENAPRED.
