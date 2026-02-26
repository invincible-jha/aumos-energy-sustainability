# ============================================================================
# Makefile — AumOS Energy Sustainability
# ============================================================================

.PHONY: install dev lint typecheck test test-cov build docker-build docker-up docker-down clean

# ---------------------------------------------------------------------------
# Development setup
# ---------------------------------------------------------------------------

install:
	pip install -e ".[dev]"

dev:
	uvicorn aumos_energy_sustainability.main:app --reload --host 0.0.0.0 --port 8000

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=aumos_energy_sustainability --cov-report=html --cov-report=term-missing

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build:
	pip install build
	python -m build

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:
	docker build -t aumos-energy-sustainability:latest .

docker-up:
	docker compose -f docker-compose.dev.yml up -d

docker-down:
	docker compose -f docker-compose.dev.yml down -v

docker-logs:
	docker compose -f docker-compose.dev.yml logs -f energy-sustainability

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

migrate:
	psql "$$AUMOS_DATABASE_URL" -f src/aumos_energy_sustainability/migrations/001_initial_schema.sql

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean:
	rm -rf dist/ build/ .eggs/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .mypy_cache/ .ruff_cache/ .pytest_cache/
