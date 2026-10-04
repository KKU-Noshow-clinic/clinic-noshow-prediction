.PHONY: setup data train lint format test up down logs api pipeline loadtest

setup:      ## install Python 3.11 + locked dependencies
	uv sync

data:       ## download dataset from Kaggle + verify SHA-256
	uv run python scripts/download_data.py

train:      ## run all three experiments and log them to MLflow
	uv run python -m noshow.models.train

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test: lint
	uv run pytest -q

up:         ## start API, MLflow, Prefect, Prometheus, Grafana
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

api:        ## run API locally without Docker
	uv run uvicorn noshow.serving.app:app --reload --port 8000

pipeline: export PREFECT_API_URL ?= http://localhost:4200/api
pipeline:   ## run the Prefect modeling and registry pipeline
	uv run python -m noshow.pipeline.flow

loadtest:   ## TODO(serving workstream): Locust p50/p95/throughput
	@echo "not implemented yet"