# aumos-energy-sustainability

Carbon footprint per inference, renewable energy routing, ESG sustainability reporting, and optimization recommendations for AumOS AI workloads.

## Overview

`aumos-energy-sustainability` is a standalone AumOS microservice that tracks the environmental impact of AI workloads in real time and surfaces actionable recommendations to reduce carbon emissions. It provides:

- **Carbon tracking**: Record energy consumed and CO2 emitted for every inference call
- **Renewable routing**: Score candidate regions by renewable percentage and latency, route workloads to the greenest available region
- **ESG reporting**: Generate quarterly/annual sustainability reports with per-model and per-region breakdowns, ESG score, and routing optimisation rate
- **Optimization advisor**: Analyse usage patterns and generate ranked recommendations (routing changes, batching improvements, scheduling shifts)

## Architecture

Hexagonal architecture — `api/` → `core/` → `adapters/`:

```
src/aumos_energy_sustainability/
├── __init__.py
├── main.py           # FastAPI lifespan + app factory
├── settings.py       # AUMOS_ENERGY_ env-prefixed config
├── api/
│   ├── router.py     # FastAPI routes — delegate only, no logic
│   └── schemas.py    # Pydantic request/response DTOs
├── core/
│   ├── models.py     # SQLAlchemy ORM (esg_ prefix)
│   ├── interfaces.py # Protocol ports for repositories + adapters
│   └── services.py   # Domain services (all business logic lives here)
└── adapters/
    ├── repositories.py      # SQLAlchemy async implementations
    ├── kafka.py             # Kafka event publisher
    └── carbon_api_client.py # Electricity Maps HTTP client (mock mode if no key)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/energy/carbon/track` | Track inference carbon footprint |
| `GET`  | `/api/v1/energy/carbon/report` | Paginated carbon records |
| `POST` | `/api/v1/energy/route/optimize` | Route workload to renewable region |
| `GET`  | `/api/v1/energy/regions` | List regional energy profiles |
| `POST` | `/api/v1/energy/sustainability/report` | Generate ESG report |
| `GET`  | `/api/v1/energy/sustainability/reports` | List ESG reports |
| `GET`  | `/api/v1/energy/sustainability/reports/{id}` | Get ESG report detail |
| `GET`  | `/api/v1/energy/optimization/recommendations` | List optimization recommendations |
| `POST` | `/api/v1/energy/optimization/recommendations/generate` | Generate fresh recommendations |

Interactive docs available at `http://localhost:8000/docs` when running.

## Database Tables

| Table | Purpose |
|-------|---------|
| `esg_carbon_records` | Per-inference carbon footprint records |
| `esg_energy_profiles` | Regional energy source profiles (renewable %) |
| `esg_routing_decisions` | Workload routing decisions based on energy |
| `esg_sustainability_reports` | Generated ESG sustainability reports |
| `esg_optimizations` | Energy optimization recommendations |

All tables use RLS for tenant isolation (`app.current_tenant` setting).

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Start dependencies
make docker-up

# Run the service (mock carbon API — no key needed for dev)
make dev
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
AUMOS_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/aumos_energy
AUMOS_KAFKA_BOOTSTRAP_SERVERS=localhost:9092
AUMOS_ENERGY_CARBON_API_KEY=        # Leave empty for mock mode
AUMOS_ENERGY_REDIS_URL=redis://localhost:6379/0
```

## Carbon API Integration

The service integrates with [Electricity Maps](https://www.electricitymaps.com/) for real-time carbon intensity data. Set `AUMOS_ENERGY_CARBON_API_KEY` to enable live data. Without a key, the client returns built-in mock data for 13 common cloud regions, enabling full development without external dependencies.

## ESG Score Calculation

```
ESG Score = (avg_renewable_pct / 100) × 60 + routing_opt_rate × 40
```

- `avg_renewable_pct`: Average renewable energy percentage across all inferences in the period
- `routing_opt_rate`: Fraction of workloads routed to regions with renewable_score ≥ 0.5
- Score range: 0–100 (higher is better)

## Events Published

| Topic | Description |
|-------|-------------|
| `aumos.energy.carbon.tracked` | Emitted when a carbon record is created |
| `aumos.energy.route.decided` | Emitted when a routing decision is recorded |
| `aumos.energy.report.generated` | Emitted when an ESG report is ready |
| `aumos.energy.optimizations.generated` | Emitted when new recommendations are created |

## Development

```bash
make lint        # Ruff linting + format check
make typecheck   # mypy strict mode
make test        # pytest with coverage
make test-cov    # pytest with HTML coverage report
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
