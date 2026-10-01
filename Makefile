.PHONY: up down ingest-firms ingest-terrain ingest-weather ingest-vegetation \
        build-dataset run-ca train calibrate backtest report report-artifacts serve \
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
	uv run --package ingestion pyrocast-ingest dem

ingest-weather:
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then \
		echo "uso: make ingest-weather START=YYYY-MM-DD END=YYYY-MM-DD"; \
		exit 1; \
	fi
	uv run --package ingestion pyrocast-ingest era5 --start $(START) --end $(END)

ingest-vegetation:
	@if [ -z "$(YEAR)" ] || [ -z "$(MONTH)" ]; then \
		echo "uso: make ingest-vegetation YEAR=YYYY MONTH=M"; \
		exit 1; \
	fi
	uv run --package ingestion pyrocast-ingest sentinel2 --year $(YEAR) --month $(MONTH)
	uv run --package ingestion pyrocast-ingest worldcover

build-dataset:
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then \
		echo "uso: make build-dataset START=YYYY-MM-DD END=YYYY-MM-DD"; \
		exit 1; \
	fi
	uv run --package features pyrocast-features build-dataset --start $(START) --end $(END)

run-ca:
	uv run --package models pyrocast-models run-ca

train:
	uv run --package models pyrocast-train smoke-test

calibrate:
	uv run --package models pyrocast-calibrate run --fixture

backtest:
	uv run --package models pyrocast-models backtest

# solo lee bench/results/ y docs/ -- no necesita credenciales ni datos locales
report:
	uv run --package models pyrocast-models report

# necesita runs/finetune_2026_v2/best.pt (+ .calibrator.pt) y data/processed/dataset
report-artifacts:
	uv run --package models pyrocast-models report-artifacts --checkpoint runs/finetune_2026_v2/best.pt

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
	uv run mypy --strict shared/src features/src serving/src
