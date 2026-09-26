# PyroCast

Sistema de pronóstico de propagación de incendios forestales mediante
fusión de datos satelitales y meteorológicos abiertos (Biobío, Ñuble,
La Araucanía).

> **Herramienta de investigación. No usar para decisiones operativas de
> combate de incendios sin validación de CONAF/SENAPRED.**

Ver `CLAUDE.md` para arquitectura, fuentes de datos y convenciones
completas; `docs/decisions.md` para decisiones de diseño y
`docs/limitations.md` para limitaciones conocidas.

## Quickstart

```bash
cp .env.example .env   # completar credenciales, ver comentarios en el archivo
uv sync --all-packages --group dev
make up                # levanta postgis + api (solo /healthz por ahora)
make test               # corre los tests de los 5 paquetes del workspace
make lint
make typecheck
```

Ningún comando anterior requiere credenciales reales: `make test`,
`make lint` y `make typecheck` pasan en verde sin ningún valor en `.env`
más allá de placeholders de prueba (ver `shared/tests/test_config.py`).
