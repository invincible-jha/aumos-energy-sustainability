"""Energy Sustainability service settings extending AumOS base configuration."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Configuration for the AumOS Energy Sustainability service.

    Extends base AumOS settings with energy-sustainability-specific configuration
    for carbon tracking, renewable routing, and ESG reporting.

    All settings use the AUMOS_ENERGY_ environment variable prefix.
    """

    service_name: str = "aumos-energy-sustainability"

    # ---------------------------------------------------------------------------
    # Carbon API integration
    # ---------------------------------------------------------------------------
    carbon_api_url: str = Field(
        default="https://api.electricitymap.org/v3",
        description="Electricity Maps carbon intensity API base URL",
    )
    carbon_api_key: str = Field(
        default="",
        description="API key for the carbon intensity data provider",
    )
    carbon_api_timeout: float = Field(
        default=10.0,
        description="Timeout in seconds for carbon API calls",
    )
    carbon_cache_ttl_seconds: int = Field(
        default=300,
        description="TTL in seconds for cached carbon intensity readings",
    )

    # ---------------------------------------------------------------------------
    # Energy routing
    # ---------------------------------------------------------------------------
    default_carbon_threshold_gco2_per_kwh: float = Field(
        default=150.0,
        description="Carbon intensity threshold (gCO2/kWh) above which workloads are rerouted",
    )
    renewable_preference_weight: float = Field(
        default=0.7,
        description="Weight (0–1) for renewable percentage when scoring regions for routing",
    )
    latency_preference_weight: float = Field(
        default=0.3,
        description="Weight (0–1) for latency when scoring regions for routing (sums to 1 with renewable)",
    )
    routing_cache_ttl_seconds: int = Field(
        default=60,
        description="TTL in seconds for cached routing decisions",
    )

    # ---------------------------------------------------------------------------
    # ESG reporting
    # ---------------------------------------------------------------------------
    esg_report_retention_days: int = Field(
        default=2555,
        description="Retention period in days for ESG sustainability reports (≈7 years)",
    )
    default_reporting_period_days: int = Field(
        default=90,
        description="Default period length in days when generating a new ESG report",
    )

    # ---------------------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------------------
    optimization_refresh_hours: int = Field(
        default=24,
        description="Hours between automatic refresh of optimization recommendations",
    )
    min_savings_threshold_kg_co2: float = Field(
        default=1.0,
        description="Minimum projected CO2 savings (kg) required to surface an optimization recommendation",
    )

    # ---------------------------------------------------------------------------
    # Redis
    # ---------------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for carbon intensity and routing caches",
    )

    model_config = SettingsConfigDict(env_prefix="AUMOS_ENERGY_")
