.PHONY: setup data train lint format test up down logs api pipeline loadtest parity

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

LT_USERS ?= 50
LT_RATE ?= 10
LT_TIME ?= 2m
LT_HOST ?= http://localhost:8000
LT_NAME ?= run
LT_LABEL ?= current

loadtest:   ## Locust p50/p95/throughput vs configs/slo.yaml -> docs/loadtest_report.md
	mkdir -p loadtest/results
	uv run locust -f loadtest/locustfile.py --host $(LT_HOST) --headless \
		--users $(LT_USERS) --spawn-rate $(LT_RATE) --run-time $(LT_TIME) --csv loadtest/results/$(LT_NAME)
	uv run python scripts/check_slo.py loadtest/results/$(LT_NAME) --label "$(LT_LABEL)" \
		--users $(LT_USERS) --duration $(LT_TIME)

parity:     ## running API vs champion model scored locally (must be identical)
	uv run python scripts/check_serving_parity.py
