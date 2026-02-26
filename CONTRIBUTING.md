# Contributing to aumos-energy-sustainability

Thank you for contributing to AumOS Energy Sustainability.

## Development Setup

```bash
git clone <repo-url>
cd aumos-energy-sustainability
pip install -e ".[dev]"
cp .env.example .env
make docker-up
```

## Code Standards

- Python 3.11+ with strict type hints on all function signatures
- Ruff for linting and formatting (line length 120)
- mypy strict mode — no `Any` without justification
- All new endpoints require request/response schemas in `api/schemas.py`
- Business logic belongs in `core/services.py`, never in routers or repositories
- Every new service method must have a docstring with Args and Returns

## Commit Convention

```
feat: add nuclear_percentage to EnergyProfile
fix: correct carbon_gco2 calculation for zero-energy inferences
refactor: extract ESG score computation to standalone function
test: add unit tests for OptimizationAdvisorService.generate_recommendations
docs: document routing score formula in README
```

## Pull Request Process

1. Branch from `main`: `feature/`, `fix/`, `docs/`
2. Run `make lint typecheck test` — all must pass
3. Update `CHANGELOG.md` under `[Unreleased]`
4. Request review from a maintainer

## Adding a New Region

1. Add the region → zone mapping in `adapters/carbon_api_client.py:REGION_TO_ZONE`
2. Add mock data in `adapters/carbon_api_client.py:_MOCK_DATA`
3. Seed an `EnergyProfile` via the `/api/v1/energy/regions` refresh endpoint or migration

## Security

See [SECURITY.md](SECURITY.md) for our vulnerability disclosure policy.
