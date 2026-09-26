# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY shared/pyproject.toml shared/pyproject.toml
COPY serving/pyproject.toml serving/pyproject.toml
COPY shared/src shared/src
COPY serving/src serving/src

RUN uv sync --frozen --package shared --package serving --no-dev

FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 pyrocast && \
    useradd --uid 1000 --gid pyrocast --shell /bin/bash --create-home pyrocast

WORKDIR /app
COPY --from=builder /build/.venv /app/.venv
COPY serving/src /app/serving/src
COPY shared/src /app/shared/src

# /data es el punto de montaje esperado por docker-compose.yml
# (./data:/data + DATA_RAW_DIR=/data/raw, etc.). Se crea y se cede a
# pyrocast aquí para el caso de volumen nombrado o de correr la imagen
# sin compose; con un bind mount de host, el propietario real lo decide
# el directorio del host (ver docs/limitations.md).
RUN mkdir -p /data && chown -R pyrocast:pyrocast /app /data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/shared/src:/app/serving/src"

USER pyrocast

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
