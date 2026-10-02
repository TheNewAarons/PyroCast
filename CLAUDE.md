# PyroCast

Sistema de pronóstico de propagación de incendios forestales mediante fusión de datos satelitales y meteorológicos abiertos. Dado un punto de ignición (o un incendio activo detectado), predice la probabilidad de propagación del fuego en los días siguientes, usando un modelo físico simple (autómata celular) como referencia y un modelo de aprendizaje profundo (U-Net) como mejora, ambos calibrados y evaluados contra incendios reales.

## Problema que resuelve

Los incendios forestales en el centro-sur de Chile (Biobío, Ñuble, La Araucanía) son una amenaza estructural: la temporada 2025-2026 dejó más de 34.000 hectáreas quemadas, 21 fallecidos y miles de viviendas destruidas. Un análisis del sector señala que el problema no es la falta de tecnología disponible, sino que opera en silos. PyroCast integra en un solo pipeline reproducible las fuentes que hoy están dispersas: terreno, clima, vegetación y detecciones activas de fuego.

## Alcance y honestidad (obligatorio)

- Este es un **proyecto de investigación/portafolio académico**, no una herramienta operativa de combate de incendios. Todo README, toda pantalla web y todo informe deben incluir un aviso visible: *"Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED."*
- Nunca se afirma que el modelo "predice con certeza" ni se ocultan métricas negativas. `docs/limitations.md` es honesto y se actualiza con cada hallazgo real.
- Todas las fuentes de datos son abiertas y gratuitas, pero requieren registro (API keys). Nunca se commitean credenciales; van en `.env`, fuera de git.
- Ningún test de CI hace llamadas reales a APIs externas (FIRMS, CDS, Copernicus Data Space). Se usan fixtures grabadas (`responses` o `vcrpy`).

## Alcance geográfico y de resolución (decisión de diseño, no accidente)

El paper de referencia (WildfireCube) trabaja a 30 m / 3 h con recursos de un equipo de investigación grande. Para un proyecto de una persona, la resolución equivalente es inviable por volumen de datos y cómputo. Decisiones:

- **Área de estudio por defecto:** bounding box configurable que cubre Biobío, Ñuble y La Araucanía (las regiones más golpeadas en 2026). Configurable en `shared/config.py`, no hardcodeado en cada módulo.
- **CRS:** EPSG:32719 (UTM 19S) para cálculos planares (distancias, pendiente, viento).
- **Resolución espacial:** 250 m. **Resolución temporal:** diaria (no horaria).
- Esto es una simplificación deliberada frente a la literatura y debe quedar documentada como tal, no presentada como equivalente.

## Arquitectura

```
Fuentes abiertas (FIRMS, Copernicus DEM, ERA5-Land, Sentinel-2, ESA WorldCover)
        │
        ▼
  ingestion/   →  descarga cruda, cacheada, versionada por fecha/tile
        │
        ▼
  features/    →  grilla común, terreno, clima, vegetación, estado del fuego
        │
        ▼
  models/
    ├── cellular_automata/  →  baseline físico simple, sin entrenamiento
    └── deep/                →  U-Net, preentrenado en dataset público + fine-tuning en Chile
        │
        ▼
  models/evaluation/  →  métricas, backtesting contra incendios reales, calibración
        │
        ▼
  serving/     →  API FastAPI + mapa web (Leaflet)
        │
        ▼
  deploy/vercel/ →  despliegue gratuito (un proyecto Vercel, datos compactos vía GitHub Release; ver DEPLOY.md)
```

## Fuentes de datos (todas abiertas, requieren API key gratuita)

| Fuente | Qué entrega | Acceso |
|---|---|---|
| NASA FIRMS | Detecciones activas de fuego (VIIRS, 375 m) | API REST, requiere `MAP_KEY` gratuito |
| Copernicus DEM GLO-30 | Elevación, pendiente, orientación | OpenTopography API o bucket AWS de Copernicus |
| ERA5-Land (Copernicus CDS) | Viento, temperatura, humedad, precipitación | `cdsapi`, requiere cuenta gratuita |
| Sentinel-2 L2A | NDVI / proxy de combustible vegetal | Copernicus Data Space Ecosystem (openEO o sentinelhub-py) |
| ESA WorldCover | Tipo de cobertura de suelo (proxy de tipo de combustible) | Descarga pública AWS |

Limitación documentada desde ya: ERA5-Land tiene ~9 km de resolución nativa; se interpola a la grilla de 250 m, lo que introduce un artefacto de downscaling que debe mencionarse en resultados.

## Stack y convenciones

- Python 3.12, `uv` (workspace con miembros: `ingestion`, `features`, `models`, `serving`, `shared`), `ruff`, `mypy --strict` en `shared/` y `features/`, `pytest`.
- Geoespacial: `rasterio`, `xarray`, `rioxarray`, `geopandas`, `shapely`, `zarr` para los cubos espaciotemporales (NO se guardan rásteres grandes en Postgres).
- Metadatos de eventos y resultados: PostgreSQL + PostGIS.
- Modelos: `numpy` (autómata celular, vectorizado), `torch` (U-Net), `scikit-learn` (calibración isotónica).
- Servido: FastAPI + Jinja2 + Leaflet.js (vía CDN). Sin framework de frontend pesado.
- CLI de ingesta y entrenamiento con `typer`.
- Identificadores de código en inglés; docstrings, comentarios y documentación en español.
- Cada módulo nuevo lleva tests. Los tests de red usan fixtures grabadas, nunca la red real.
- Reproducibilidad: semillas fijas, splits por evento (no por píxel) para evitar fuga de datos entre train/val/test, configuración de grilla versionada.

## Comandos

```
make up               # levanta postgis y la api
make ingest-firms      # descarga detecciones activas (rango de fechas configurable)
make ingest-terrain     # DEM + derivados
make ingest-weather      # ERA5-Land
make ingest-vegetation    # Sentinel-2 + WorldCover
make build-dataset         # ensambla el dataset espaciotemporal por evento
make run-ca                 # simula el autómata celular baseline
make train                   # entrena/fine-tunea el U-Net
make calibrate                 # calibración isotónica
make backtest                    # evalúa contra incendios reales de Chile 2026
make report                        # genera docs/results.md y .html (solo lee bench/results/ y docs/)
make report-artifacts               # calcula curvas/mapas/descriptores desde checkpoint + datos (los deja en bench/results/)
make serve                          # levanta la API + mapa web
make test / lint / typecheck
make audit                           # pip-audit sobre uv.lock (también corre en CI)
make demo                             # API + mapa con datos SINTÉTICOS, sin credenciales
uv run --package serving python scripts/build_deploy_data.py --tag data-AAAA-MM-DD   # paquete de datos para Vercel (DEPLOY.md)
```

## Forma de trabajar

- Antes de escribir código en una tarea grande, propone un plan corto; si hay ambigüedad real, pregunta, si no, decide y sigue.
- Después de cada tarea: corre tests, lint y typecheck, y resume qué hiciste y qué queda pendiente.
- Si una decisión de diseño cambia algo de este archivo, actualízalo junto con `docs/decisions.md`.
- No agregues dependencias sin justificarlas en `docs/decisions.md`.
- Nunca elimines o suavices una métrica negativa del reporte para que el proyecto "se vea mejor".
