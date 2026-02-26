"""SQLAlchemy ORM models for the AumOS Energy Sustainability service.

All tables use the `esg_` prefix. Tenant-scoped tables extend AumOSModel
which supplies id (UUID), tenant_id, created_at, and updated_at columns.

Domain model:
  CarbonRecord          — per-inference carbon footprint record
  EnergyProfile         — regional energy source profile (renewable %)
  RoutingDecision       — workload routing decision based on energy profile
  SustainabilityReport  — generated ESG sustainability report
  OptimizationRecord    — energy optimization recommendation
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel


class CarbonRecord(AumOSModel):
    """Per-inference carbon footprint record.

    Captures the energy consumed and CO2 emitted for a single AI inference
    call, along with regional context for downstream ESG reporting.

    Table: esg_carbon_records
    """

    __tablename__ = "esg_carbon_records"

    inference_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="External inference request UUID (cross-service reference, no FK)",
    )
    model_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Identifier of the model that served this inference",
    )
    region: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Cloud/datacenter region where the inference executed (e.g. us-east-1)",
    )
    energy_kwh: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Energy consumed by the inference in kilowatt-hours",
    )
    carbon_intensity_gco2_per_kwh: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Carbon intensity of the grid at inference time in gCO2/kWh",
    )
    carbon_gco2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Total CO2 emitted = energy_kwh * carbon_intensity_gco2_per_kwh",
    )
    renewable_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Fraction of energy from renewable sources at inference time (0–100)",
    )
    tokens_input: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of input tokens processed (LLM inference context)",
    )
    tokens_output: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of output tokens generated (LLM inference context)",
    )
    inference_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Wall-clock inference duration in milliseconds",
    )
    metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Arbitrary metadata: model version, request type, workload tags",
    )


class EnergyProfile(AumOSModel):
    """Regional energy source profile describing the carbon footprint of a region.

    Profiles are refreshed periodically from the carbon API and cached.
    Each region has at most one active profile per tenant.

    Table: esg_energy_profiles
    """

    __tablename__ = "esg_energy_profiles"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "region",
            name="uq_esg_energy_profiles_tenant_region",
        ),
    )

    region: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Cloud/datacenter region identifier (e.g. eu-west-1, us-central1)",
    )
    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="",
        comment="Human-readable region name for reports",
    )
    carbon_intensity_gco2_per_kwh: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Current average grid carbon intensity in gCO2/kWh",
    )
    renewable_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Percentage of electricity from renewable sources (0–100)",
    )
    solar_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Percentage from solar (subset of renewable_percentage)",
    )
    wind_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Percentage from wind (subset of renewable_percentage)",
    )
    hydro_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Percentage from hydro (subset of renewable_percentage)",
    )
    nuclear_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Percentage from nuclear (low-carbon but not renewable)",
    )
    estimated_latency_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=50,
        comment="Estimated round-trip latency to this region in milliseconds",
    )
    last_refreshed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the last successful carbon API refresh",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Soft-delete flag — inactive profiles are excluded from routing",
    )
    source_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Raw metadata returned by the carbon intensity API for this region",
    )

    routing_decisions: Mapped[list["RoutingDecision"]] = relationship(
        "RoutingDecision",
        back_populates="selected_profile",
        cascade="all, delete-orphan",
    )


class RoutingDecision(AumOSModel):
    """Workload routing decision based on energy profile scoring.

    Records which region was selected for a workload, which regions were
    considered, and why the winner was chosen (score breakdown).

    Table: esg_routing_decisions
    """

    __tablename__ = "esg_routing_decisions"

    workload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="External workload UUID to be routed (cross-service reference)",
    )
    workload_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Type of workload: inference | training | batch_processing | fine_tuning",
    )
    selected_region: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Region selected for execution",
    )
    selected_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("esg_energy_profiles.id", ondelete="SET NULL"),
        nullable=True,
        comment="EnergyProfile that was selected (nullable for soft-deleted profiles)",
    )
    candidate_regions: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment=(
            "All regions considered with scores: "
            "[{region, renewable_score, latency_score, composite_score}]"
        ),
    )
    renewable_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Renewable percentage component of the routing score (0–1)",
    )
    latency_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Latency component of the routing score (0–1, inverted — lower is better)",
    )
    composite_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Weighted composite: renewable_weight * renewable_score + latency_weight * latency_score",
    )
    carbon_saved_gco2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Estimated CO2 saved vs. routing to the highest-intensity candidate",
    )
    override_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Manual override explanation if the top-scoring region was not selected",
    )

    selected_profile: Mapped["EnergyProfile | None"] = relationship(
        "EnergyProfile",
        back_populates="routing_decisions",
    )


class SustainabilityReport(AumOSModel):
    """Generated ESG sustainability report covering a time period.

    Reports aggregate carbon footprint, routing efficiency, and per-model
    breakdowns for regulatory and investor-facing ESG disclosure.

    Table: esg_sustainability_reports
    """

    __tablename__ = "esg_sustainability_reports"

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable report title",
    )
    report_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="quarterly",
        index=True,
        comment="monthly | quarterly | annual | custom",
    )
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="UTC start of the reporting period (inclusive)",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="UTC end of the reporting period (exclusive)",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="generating",
        index=True,
        comment="generating | ready | failed",
    )

    # Aggregate metrics
    total_inferences: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of inference calls in the reporting period",
    )
    total_energy_kwh: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total energy consumed in the period in kWh",
    )
    total_carbon_kg_co2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total CO2 emitted in the period in kilograms",
    )
    average_renewable_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Average renewable energy percentage across all inferences",
    )
    carbon_saved_kg_co2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total CO2 saved via renewable routing vs. unoptimised baseline",
    )
    routing_optimisation_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Fraction of workloads routed to renewable-preferred regions (0–1)",
    )

    # Breakdown data
    per_model_breakdown: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Carbon footprint by model: {model_id: {total_carbon_kg, energy_kwh, inferences}}",
    )
    per_region_breakdown: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Carbon footprint by region: {region: {total_carbon_kg, energy_kwh, renewable_pct}}",
    )
    esg_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Computed ESG score (0–100) based on renewable percentage and optimisation rate",
    )
    generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when report generation completed",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error detail if status=failed",
    )
    requested_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User UUID who requested this report",
    )


class OptimizationRecord(AumOSModel):
    """Energy optimization recommendation for a tenant's AI workloads.

    Recommendations are refreshed periodically or on-demand and ranked
    by projected CO2 savings. Implemented recommendations are soft-archived.

    Table: esg_optimizations
    """

    __tablename__ = "esg_optimizations"

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="routing | scheduling | model_efficiency | hardware | batch_consolidation",
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Short recommendation headline",
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Detailed recommendation explanation and rationale",
    )
    target_resource: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Model ID, region, or workload type this recommendation targets",
    )
    projected_savings_kg_co2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Estimated CO2 reduction in kg if this recommendation is implemented",
    )
    projected_savings_kwh: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Estimated energy reduction in kWh if this recommendation is implemented",
    )
    priority: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="medium",
        index=True,
        comment="high | medium | low — based on projected savings magnitude",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        index=True,
        comment="active | implemented | dismissed | superseded",
    )
    implementation_effort: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="medium",
        comment="low | medium | high — estimate of implementation complexity",
    )
    evidence: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Supporting data for the recommendation (usage patterns, cost data, etc.)",
    )
    implemented_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the recommendation was marked as implemented",
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp after which this recommendation should be refreshed or discarded",
    )
