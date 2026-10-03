# Despliegue gratuito de PyroCast

> **Herramienta de investigación. No usar para decisiones operativas de combate de incendios sin validación de CONAF/SENAPRED.**

Guía para publicar el mapa y la API de PyroCast en `*.vercel.app` usando solo planes gratuitos, sin tarjeta.

**Desplegado (2026-10-02):** <https://pyrocast-hazel.vercel.app>. Proyecto de Vercel `pyrocast`, creado y desplegado con la CLI (ver "Despliegue con la CLI"); todavía sin enlace a GitHub, así que un push **no** redespliega solo.

## Qué se despliega (y por qué no es el plan "Django + React + Neon + JWT")

El repositorio no tiene `backend/` Django ni `frontend/` React. La parte web es **una sola app FastAPI** (`serving/`) que sirve la API (`/predict`, `/active-fires`, `/api/health/`) **y** el mapa (Jinja2 + Leaflet por CDN, sin build de frontend). Esa app:

- **no usa base de datos**: PostGIS solo lo usa el pipeline de datos, que corre offline;
- **no tiene usuarios ni login**: la API es pública y de solo lectura;
- **no recibe archivos**.

Por eso el despliegue es **un único proyecto de Vercel**, sin Neon, sin JWT, sin Cloudinary, sin CORS (el mapa llama a la API del mismo origen) y sin workflow de migraciones (no hay migraciones). Si algún día se publica el pipeline, la sección "Agregar Neon más adelante" explica cómo.

```
GitHub (main) ──push──► Vercel (proyecto "pyrocast", Root Directory = deploy/vercel)
                          │  build: uv instala deps mínimas fijadas (deploy/vercel/uv.lock)
                          │         python build.py → copia el código del monorepo a _vendor/
                          │                         → descarga datos (GitHub Release) y verifica sha256
                          ▼
                        Función Python (FastAPI) en https://<proyecto>.vercel.app
                          ├─ /            mapa
                          ├─ /predict     predicción (autómata celular)
                          ├─ /active-fires detecciones de NASA FIRMS (con FIRMS_MAP_KEY)
                          └─ /api/health/ y /healthz
GitHub Actions: tests, lint, mypy, pip-audit, build de Docker y "deploy-smoke" (reproduce el build de Vercel). No despliega.
```

### El problema de los datos, resuelto

`/predict` lee rásteres procesados (`data/processed/`, no versionados). Pesan 2.1 GB, sobre todo el clima a 250 m, y el límite de una función Python de Vercel es de 500 MB. `scripts/build_deploy_data.py` arma un **paquete de 32 MB** con solo lo que `/predict` lee:
- **Capas estáticas:** se recomprimen sin cambiar su resolución.
- **Clima:** solo los 5 campos usados, promediados a 2 km. ERA5-Land tiene ~9 km nativos, así que no se pierde información real: en 5 casos reales probados, las predicciones fueron **idénticas** a las locales con 4 decimales.

El paquete se publica como asset de un **GitHub Release** (el repo es público, gratis) y el build de Vercel lo descarga y **verifica su sha256** (`deploy/vercel/data-bundle.json`). Incluye `ATTRIBUTION.txt` con los avisos que exigen las licencias de los datos.

El bundle total de la función queda en **~215 MB** (dependencias sin `torch` + código + datos), medido en un entorno limpio y verificado en CI.

## Paso a paso

### 0. Antes de empezar

**Estado al 2026-10-02:** el código ya está en `main` en GitHub y el Release `data-2026-10-01` ya está publicado. El proyecto de Vercel ya existe y está desplegado (paso 2b). Los pasos 1 y 2 solo hacen falta para recrearlo desde el panel o enlazarlo a GitHub; lo de abajo, solo si cambias los datos.

1. **Subir los commits.** Vercel despliega lo que está en GitHub, no lo de tu máquina:
   ```bash
   git push origin main
   ```
2. **Publicar el paquete de datos** como GitHub Release. El nombre y el sha256 están en `deploy/vercel/data-bundle.json`:
   ```bash
   uv run --package serving python scripts/build_deploy_data.py --tag data-2026-10-01
   tar -xOzf dist/pyrocast-data-2026-10-01.tar.gz processed/ATTRIBUTION.txt > /tmp/notes.txt
   gh release create data-2026-10-01 dist/pyrocast-data-2026-10-01.tar.gz \
     --title "Datos de despliegue data-2026-10-01" --notes-file /tmp/notes.txt
   ```
   El script es determinista: con los mismos datos produce el mismo sha256. Si cambiaste los datos, el script reescribe `data-bundle.json`; commitea y sube ese archivo **antes** de desplegar. Si no, el build se niega a usar un paquete distinto.
3. Comprueba que la URL de `data-bundle.json` descarga, por ejemplo con `curl -sIL <url> | head -1` (debe responder 200 tras las redirecciones).

### 1. Cuenta de Vercel (gratis, sin tarjeta)

1. Entra a <https://vercel.com/signup> y elige **Continue with GitHub**.
2. Plan: **Hobby** (gratis). Es para uso personal y no comercial (ver "Limitaciones").
3. Autoriza a la app de Vercel en GitHub para que acceda al repositorio `PyroCast`; basta con ese repositorio.

### 2. Importar el repositorio (un solo proyecto)

1. En el panel: **Add New… → Project → Import** junto a `PyroCast`.
2. **Root Directory**: haz clic en **Edit** y elige **`deploy/vercel`**. Es el paso clave.
3. **Framework Preset**: Vercel debería detectar **FastAPI** (lee `deploy/vercel/pyproject.toml`). Si no, elígelo a mano.
4. **Build and Output Settings**: no cambies nada. El build sale de `[tool.vercel.scripts]` en `pyproject.toml`, y un Build Command en el panel **reemplazaría** a `python build.py`.
5. **Environment Variables**: agrega `FIRMS_MAP_KEY` (ver la tabla) para *Production* y *Preview*.
6. **Deploy**. El primer build tarda unos minutos (dependencias geoespaciales y descarga de datos).

En **Settings → Build and Deployment → Root Directory**, deja **activado** "Include files outside the root directory in the Build Step" (viene activado). `build.py` copia el código de `../../shared`, `../../serving`, etc.

### 2b. Despliegue con la CLI (lo que se usó)

Si la app de GitHub de Vercel no está instalada, Vercel no puede enlazar el repo. Así se desplegó:

```bash
npx vercel login                                   # una vez
# proyecto con Root Directory = deploy/vercel (la CLI no fija Root Directory al crear)
echo '{"name":"pyrocast","framework":"fastapi","rootDirectory":"deploy/vercel"}' > /tmp/p.json
npx vercel api /v11/projects -X POST --input /tmp/p.json
npx vercel link --yes --project pyrocast           # desde la raíz del repo (crea .vercel/)
npx vercel deploy --prod --yes                     # sube solo lo que permite .vercelignore
```

`.vercelignore` (en la raíz) sube únicamente los paquetes y `deploy/`: nunca `data/`, `runs/`, `.venv` ni secretos. Para que cada push despliegue solo: instala la app de Vercel en GitHub (<https://github.com/apps/vercel>, elige el repo `PyroCast`) y luego ejecuta `npx vercel git connect`.

**Nota técnica:** los wheels de `rasterio` necesitan `libexpat.so.1`, que el runtime de funciones de Vercel no trae (primer despliegue: `ImportError: libexpat.so.1`). `build.py` la copia desde la imagen de build a `_vendor/lib/` y `app.py` la precarga antes de importar `rasterio`.

### 3. Comprobar

Abre `https://<tu-proyecto>.vercel.app` y recorre la lista de verificación de más abajo.

### 4. Despliegues siguientes

- **Push a `main`**: Vercel despliega a producción solo.
- **Pull request**: Vercel crea un *preview* con su propia URL. GitHub Actions corre los tests pero no despliega; no se duplica.
- **Revertir**: en **Deployments**, sobre un despliegue anterior, **⋯ → Promote to Production** (instantáneo, sin rebuild).

### 5. Actualizar los datos

Ingesta y procesamiento locales (`make ingest-*`) → `scripts/build_deploy_data.py --tag data-AAAA-MM-DD` → `gh release create …` → commit de `deploy/vercel/data-bundle.json` → push. No hay migraciones ni base que actualizar.

### Base de datos, superusuario y migraciones

No aplican: la app servida no tiene modelos de base de datos ni usuarios. El workflow de migraciones con `workflow_dispatch` y `vercel env pull` no se creó porque no habría nada que migrar.

**Agregar Neon más adelante** (solo si se publica el pipeline):
1. En Vercel: **Storage → Create → Neon** (plan gratis). Vercel crea `DATABASE_URL` en el proyecto.
2. En Neon, activar PostGIS con `CREATE EXTENSION postgis;`.
3. Mapear las variables `POSTGRES_*` de `shared/config.py` desde la URL *pooled* con `sslmode=require`.

Hoy esas variables se rellenan con marcadores en `deploy/vercel/app.py` porque la web no las usa.

## Variables de entorno

| Variable | Proyecto | Ejemplo | De dónde sale | ¿Requerida? |
|---|---|---|---|---|
| `FIRMS_MAP_KEY` | Vercel `pyrocast` (Production y Preview) | `a1b2c3…` (32 caracteres) | <https://firms.modaps.eosdis.nasa.gov/api/map_key/> (llega por correo) | Recomendada. Sin ella, `/active-fires` responde un error claro y el mapa muestra "Incendios activos no disponibles". El resto funciona |
| `ENVIRONMENT` | Vercel `pyrocast` | `production` | Valor fijo | No; por defecto `production` (sin `/docs`). Pon `development` solo en *Preview* si quieres `/docs` ahí |
| `CORS_ALLOW_ORIGINS` | Vercel `pyrocast` | `["https://otra-app.vercel.app"]` | Tú, si otra web llama a la API | No; por defecto `[]`, sin CORS (el mapa es del mismo origen). Nunca `*`: la app lo rechaza |
| `PYROCAST_DATA_URL` | Vercel `pyrocast` | `https://github.com/…/releases/download/…/x.tar.gz` | Solo para probar otro paquete sin commitear | No; por defecto, la de `data-bundle.json` |
| `PYROCAST_DATA_SHA256` | Vercel `pyrocast` | `6cb23a35…` | Salida de `build_deploy_data.py` | Obligatoria si usas `PYROCAST_DATA_URL` |
| `DATA_PROCESSED_DIR` | (ninguno) | — | La fija `deploy/vercel/app.py` | No la pongas |
| `CDS_*`, `COPERNICUS_DATASPACE_*`, `POSTGRES_*` | (ninguno) | — | Solo las usa el pipeline local | No; `app.py` pone marcadores. **No subas** tus claves reales de CDS/Copernicus a Vercel |

GitHub Actions no necesita ningún secreto: el CI usa datos sintéticos y no despliega.

**Si todos los jobs del CI fallan en ~3 segundos sin logs** (como pasó en las corridas de `main` hasta el 2026-10-02), el problema no es el código: GitHub no llegó a asignar un runner. Revisa en GitHub:
1. **Settings → Actions → General**: "Allow all actions and reusable workflows" activado.
2. **Settings (de tu cuenta) → Billing and plans**: que no haya un aviso de cuenta bloqueada o pago pendiente. En repos **públicos** los minutos de Actions son gratis e ilimitados; en repos privados el plan Free trae 2000 min/mes.
3. Después, en la pestaña **Actions**, abre la corrida y usa **Re-run all jobs**.

Secretos: `.env`, `.env.local`, `.vercel/` y `dist/` están en `.gitignore`; `deploy/vercel/.env.example` documenta las variables. Un test (`shared/tests/test_no_secrets.py`) falla si se versiona algo con forma de credencial.

## Lista de verificación posdespliegue

El pedido original incluía login JWT, CORS con un frontend aparte, estilos del admin de Django y rutas de React Router. Aquí sus equivalentes reales:

| Verificación | Cómo | Esperado |
|---|---|---|
| Health check | `curl -s https://<app>.vercel.app/api/health/` | `200`, `"status": "ok"`, `static_layers: "ok"`, rango de clima `2025-11-10` a `2026-03-31`, `model: "cellular_automata"` |
| Liveness | `curl -s https://<app>.vercel.app/healthz` | `{"status":"ok"}` |
| Mapa con estilos | Abrir `https://<app>.vercel.app/` | Tema oscuro, tipografías cargadas, aviso de investigación arriba, leyenda con porcentajes |
| Predicción | En el mapa: clic cerca de lat -36.61, lon -72.59, fecha 2025-11-10, **Predecir propagación**. O con `curl -s -X POST https://<app>.vercel.app/predict -H 'Content-Type: application/json' -d '{"lat":-36.61,"lon":-72.59,"date":"2026-01-15","horizon_days":3}'` | Celdas coloreadas y deslizador de días; la respuesta JSON trae `"research_tool": true` |
| Error claro | Fecha fuera de rango, por ejemplo `"date":"2026-09-30"` | `422` con `code: "weather_unavailable"` y el rango disponible, sin número inventado |
| Incendios activos | `curl -s https://<app>.vercel.app/active-fires?days=1` | `200` con puntos (o `502 firms_unavailable` si no configuraste `FIRMS_MAP_KEY`) |
| CORS (equivale a "solo el dominio del frontend") | `curl -sI -H 'Origin: https://otro.example' https://<app>.vercel.app/healthz \| grep -i access-control` | Sin cabecera `access-control-allow-origin` (ningún origen ajeno autorizado) |
| Seguridad | `curl -sI https://<app>.vercel.app/` | `x-content-type-options: nosniff`, `x-frame-options: DENY`, `referrer-policy`, `permissions-policy`; HTTPS forzado por Vercel |
| Sin modo debug ni docs | `curl -s -o /dev/null -w '%{http_code}' https://<app>.vercel.app/docs` | `404` en producción |
| Recarga de rutas | Recargar `/` y `/static/app.css` | `200`. No hay rutas de cliente (no es una SPA): cualquier otra ruta responde `404` de la API, como corresponde |
| Login JWT | — | No aplica: no hay usuarios |

## Limitaciones del plan gratuito y cómo mitigarlas

| Limitación | Impacto | Mitigación |
|---|---|---|
| **Vercel Hobby es solo para uso personal y no comercial** | Un uso comercial o de una organización exige el plan Pro (pago) | Mantener el proyecto como portafolio o investigación. Si pasa a uso institucional, pasar a Pro o mover la imagen Docker existente a otro servicio |
| **Arranque en frío** | La primera petición tras un rato inactivo importa GDAL, rasterio, xarray y pandas (~0.5 s en caliente, varios segundos en frío) | Aceptable para investigación. Un *cron* que llame a `/healthz` mantiene la instancia caliente, pero gasta invocaciones |
| **`maxDuration` de 60 s por petición** | Una predicción con horizonte grande en un punto de mucho combustible podría acercarse | Hoy `/predict` tarda bastante menos que eso (horizonte máx. 7 días, grilla acotada). Si se acerca, bajar el horizonte máximo |
| **Bundle de 500 MB** | Agregar `torch` (el U-Net) lo rompería | El CI (`deploy-smoke`) falla si el bundle pasa de 450 MB. El U-Net no se sirve (`docs/backtest-2026.md` §8) |
| **Sistema de archivos de solo lectura y sin estado** | Los datos son una foto fija; la caché de predicciones es por instancia y se pierde | Actualizar datos = nuevo paquete + Release + push. La caché solo acelera repeticiones |
| **Datos acotados** | Solo hay clima de 57 días (nov. 2025 a mar. 2026) y NDVI de pocas zonas | `/predict` responde `weather_unavailable` con el rango disponible. Ampliar con `make ingest-weather` + nuevo paquete |
| **Cuota de FIRMS** (5000 transacciones / 10 min por clave) | Muchas visitas podrían agotarla | `/active-fires` tiene caché de 10 min por instancia y devuelve un error claro (no una lista vacía) si se agota |
| **Ancho de banda e invocaciones** del plan Hobby (cupos mensuales) | Un pico de tráfico puede pausar el proyecto hasta el mes siguiente | Uso de investigación: holgado. Revisar **Usage** en el panel |
| **API pública sin autenticación ni límite de tasa** | Cualquiera puede llamar `/predict` | Es de solo lectura y barata. Si hace falta, activar reglas de *rate limiting* en **Firewall** de Vercel, si el plan lo permite, o un token simple |
| **Teselas del mapa de Esri sin clave** | Sus términos para tráfico alto no se verificaron (`docs/data-sources.md`) | Uso de investigación de bajo tráfico; si crece, pasar a un proveedor con clave gratuita |
| **El Release de datos debe seguir existiendo** | Si se borra, los builds nuevos fallan (los despliegues ya hechos siguen funcionando) | No borrar Releases `data-*` en uso; el sha256 impide usar un paquete equivocado |

## Desarrollo local (sin cambios)

`make serve` (datos reales locales), `make demo` (datos sintéticos) y `make test lint typecheck` siguen igual. Para probar el build de Vercel en local:

```bash
cd deploy/vercel
uv sync --frozen --group dev
.venv/bin/python build.py --data-file ../../dist/pyrocast-data-2026-10-01.tar.gz
.venv/bin/uvicorn app:app --port 8001     # http://127.0.0.1:8001
```

Con la CLI de Vercel también sirve `vercel dev` (tras `vercel link`, que crea `.vercel/`, ya ignorado por git).
