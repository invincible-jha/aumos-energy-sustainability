# Changelog

All notable changes to `aumos-energy-sustainability` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-02-26

### Added
- `CarbonTrackerService`: per-inference carbon footprint tracking with Kafka events
- `EnergyRouterService`: renewable-weighted routing to lowest-carbon region
- `SustainabilityReportService`: ESG report generation with per-model/region breakdowns and ESG score
- `OptimizationAdvisorService`: pattern-based optimization recommendations ranked by CO2 savings
- `CarbonAPIClient`: Electricity Maps v3 integration with built-in mock mode for development
- `EnergyEventPublisher`: Kafka adapter for all domain events
- Five PostgreSQL tables with `esg_` prefix and RLS tenant isolation
- FastAPI REST API with 9 endpoints across carbon, routing, reporting, and optimization domains
- Docker multi-stage build with non-root runtime user
- `docker-compose.dev.yml` with Postgres, Kafka (KRaft), and Redis
- Mock carbon intensity data for 13 cloud regions enabling offline development
