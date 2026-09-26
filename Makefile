.PHONY: up down ingest-firms ingest-terrain ingest-weather ingest-vegetation \
        build-dataset run-ca train calibrate backtest report serve \
        test lint typecheck

up:
	docker compose up -d

down:
	docker compose down

ingest-firms:
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then \
		echo "uso: make ingest-firms START=YYYY-MM-DD END=YYYY-MM-DD [BBOX=west,south,east,north] [SENSOR=VIIRS_SNPP_NRT]"; \
		exit 1; \
	fi
	uv run --package ingestion pyrocast-ingest firms --start $(START) --end $(END) \
		$(if $(BBOX),--bbox $(BBOX)) $(if $(SENSOR),--sensor $(SENSOR))

ingest-terrain:
	@echo "pendiente: módulo ingestion/dem aún no implementado"

ingest-weather:
	@echo "pendiente: módulo ingestion/era5 aún no implementado"

ingest-vegetation:
	@echo "pendiente: módulos ingestion/sentinel2 e ingestion/worldcover aún no implementados"

build-dataset:
	@echo "pendiente: features/dataset aún no implementado"

run-ca:
	@echo "pendiente: models/cellular_automata aún no implementado"

train:
	@echo "pendiente: models/deep aún no implementado"

calibrate:
	@echo "pendiente: models/evaluation (calibración isotónica) aún no implementado"

backtest:
	@echo "pendiente: models/evaluation (backtesting) aún no implementado"

report:
	@echo "pendiente: generación de docs/results.md aún no implementada"

serve:
	uv run --package serving uvicorn serving.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run --package shared pytest shared/tests -v
	uv run --package ingestion pytest ingestion/tests -v
	uv run --package features pytest features/tests -v
	uv run --package models pytest models/tests -v
	uv run --package serving pytest serving/tests -v

lint:
	uv run ruff check .

typecheck:
	uv run mypy --strict shared/src features/src
