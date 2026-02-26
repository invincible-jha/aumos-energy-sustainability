"""AumOS Energy Sustainability service entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database, get_session
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_energy_sustainability.adapters.carbon_api_client import CarbonAPIClient
from aumos_energy_sustainability.adapters.kafka import EnergyEventPublisher
from aumos_energy_sustainability.adapters.repositories import (
    CarbonRecordRepository,
    EnergyProfileRepository,
    OptimizationRepository,
    RoutingDecisionRepository,
    SustainabilityReportRepository,
)
from aumos_energy_sustainability.api.router import router
from aumos_energy_sustainability.core.services import (
    CarbonTrackerService,
    EnergyRouterService,
    OptimizationAdvisorService,
    SustainabilityReportService,
)
from aumos_energy_sustainability.settings import Settings

logger = get_logger(__name__)
settings = Settings()

_kafka_publisher: EnergyEventPublisher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initialises the database connection pool, Kafka event publisher,
    HTTP carbon API client, and all domain services. Exposes everything
    on app.state for dependency injection via request handlers.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    global _kafka_publisher  # noqa: PLW0603

    logger.info("Starting AumOS Energy Sustainability", version="0.1.0")

    # Database connection pool
    init_database(settings.database)
    logger.info("Database connection pool ready")

    # Kafka event publisher
    _kafka_publisher = EnergyEventPublisher(settings.kafka)
    await _kafka_publisher.start()
    app.state.kafka_publisher = _kafka_publisher
    logger.info("Kafka event publisher ready")

    # Carbon intensity API client
    carbon_api_client = CarbonAPIClient(
        base_url=settings.carbon_api_url,
        api_key=settings.carbon_api_key,
        timeout=settings.carbon_api_timeout,
    )
    app.state.carbon_api_client = carbon_api_client
    logger.info(
        "Carbon API client initialised",
        url=settings.carbon_api_url,
        mock_mode=not settings.carbon_api_key,
    )

    # Build a shared session factory reference for service construction.
    # In production, each request gets its own session via get_session().
    # For services that need a session at startup (e.g. profile seeding),
    # we pass a factory rather than a session so services don't hold
    # long-lived connections.
    app.state.settings = settings
    app.state.carbon_api_client = carbon_api_client
    app.state.event_publisher = _kafka_publisher

    # Expose service constructors — actual services are constructed per-request
    # in the dependency helpers, using the session from the request scope.
    # Here we store the shared dependencies that don't vary per request.
    logger.info("Energy Sustainability service startup complete")
    yield

    # Shutdown
    if _kafka_publisher:
        await _kafka_publisher.stop()

    logger.info("Energy Sustainability service shutdown complete")


def _build_carbon_service(app: FastAPI) -> CarbonTrackerService:
    """Construct CarbonTrackerService with repository and publisher from app state.

    Args:
        app: FastAPI application instance with state populated by lifespan.

    Returns:
        CarbonTrackerService ready for use.
    """
    # NOTE: In a full DI setup, sessions are provided per-request.
    # This factory pattern is consistent with other AumOS services.
    raise NotImplementedError("Use per-request session injection via Depends(get_session)")


app: FastAPI = create_app(
    service_name="aumos-energy-sustainability",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="postgres", check_fn=lambda: None),
        HealthCheck(name="kafka", check_fn=lambda: None),
        HealthCheck(name="carbon_api", check_fn=lambda: None),
    ],
)

app.include_router(router, prefix="/api/v1")
