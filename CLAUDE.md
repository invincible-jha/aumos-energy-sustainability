# CLAUDE.md — aumos-energy-sustainability

## Purpose

Carbon footprint per AI inference, renewable energy workload routing, ESG sustainability reporting, and optimization recommendations.

## Package

- **Python package**: `aumos_energy_sustainability`
- **Table prefix**: `esg_`
- **Env prefix**: `AUMOS_ENERGY_`
- **Default port**: 8000

## Architecture

Hexagonal — `api/` → `core/` → `adapters/`:

```
src/aumos_energy_sustainability/
├── main.py           # FastAPI lifespan + app factory
├── settings.py       # AUMOS_ENERGY_ config (extends AumOSSettings)
├── api/
│   ├── router.py     # Routes — delegate to services, no logic here
│   └── schemas.py    # Pydantic DTOs for all endpoints
├── core/
│   ├── models.py     # SQLAlchemy ORM (esg_ table prefix)
│   ├── interfaces.py # Protocol ports (ICarbonRecordRepository, etc.)
│   └── services.py   # All domain logic
└── adapters/
    ├── repositories.py      # SQLAlchemy async implementations
    ├── kafka.py             # EnergyEventPublisher
    └── carbon_api_client.py # Electricity Maps HTTP client + mock mode
```

## Domain Services

| Service | Responsibility |
|---------|---------------|
| `CarbonTrackerService` | Track per-inference carbon; compute carbon_gco2 = energy_kwh × intensity |
| `EnergyRouterService` | Score regions by renewable % + latency; persist routing decisions |
| `SustainabilityReportService` | Aggregate carbon records → ESG report with score |
| `OptimizationAdvisorService` | Analyse patterns; generate routing/batching recommendations |

## Database Tables

| Table | Description |
|-------|-------------|
| `esg_carbon_records` | Append-only per-inference footprint (immutable) |
| `esg_energy_profiles` | Regional grid profiles (upserted on refresh) |
| `esg_routing_decisions` | Audit log of all routing choices |
| `esg_sustainability_reports` | Generated ESG reports (status: generating → ready/failed) |
| `esg_optimizations` | Ranked recommendations (status: active → implemented/dismissed) |

## Key Invariants

- Carbon records are append-only — never update or delete
- Routing decisions are always persisted for auditability
- Reports go through `generating → ready | failed` (never skip)
- Optimization recommendations require minimum `min_savings_threshold_kg_co2` savings to be created
- Carbon API falls back to mock data when `AUMOS_ENERGY_CARBON_API_KEY` is empty

## ESG Score Formula

```
ESG Score = (avg_renewable_pct / 100) × 60 + routing_opt_rate × 40
```

Score range: 0–100. Higher is greener.

## Kafka Events

- `aumos.energy.carbon.tracked` — on every carbon record creation
- `aumos.energy.route.decided` — on every routing decision
- `aumos.energy.report.generated` — on ESG report completion
- `aumos.energy.optimizations.generated` — on recommendation batch creation

## Development Commands

```bash
make install    # Install with dev deps
make dev        # uvicorn with hot reload
make lint       # ruff check + format
make typecheck  # mypy strict
make test       # pytest with coverage
make docker-up  # Postgres + Kafka + Redis
```

## Carbon API Mock Mode

Set `AUMOS_ENERGY_CARBON_API_KEY=""` (empty) → `CarbonAPIClient` returns built-in mock data for 13 regions. No external dependency needed for development or CI.

## Adding a New Region

1. Add to `REGION_TO_ZONE` in `adapters/carbon_api_client.py`
2. Add mock data to `_MOCK_DATA` in the same file
3. Call `EnergyRouterService.refresh_profile()` to seed the `esg_energy_profiles` table
