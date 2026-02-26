"""Pydantic request and response schemas for the Energy Sustainability API.

All schemas use strict validation. Request schemas validate inputs at the
system boundary; response schemas are read-only DTO projections.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Carbon tracking schemas
# ---------------------------------------------------------------------------


class TrackCarbonRequest(BaseModel):
    """Request body for POST /api/v1/energy/carbon/track."""

    inference_id: uuid.UUID = Field(description="External inference request UUID")
    model_id: str = Field(min_length=1, max_length=255, description="Model identifier")
    region: str = Field(min_length=1, max_length=100, description="Execution region")
    energy_kwh: float = Field(ge=0.0, description="Energy consumed in kWh")
    carbon_intensity_gco2_per_kwh: float = Field(
        ge=0.0, description="Grid carbon intensity in gCO2/kWh at inference time"
    )
    renewable_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Renewable fraction (0–100)"
    )
    tokens_input: int | None = Field(default=None, ge=0, description="Input token count")
    tokens_output: int | None = Field(default=None, ge=0, description="Output token count")
    inference_duration_ms: int | None = Field(
        default=None, ge=0, description="Wall-clock inference duration in ms"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class CarbonRecordResponse(BaseModel):
    """Response schema for a single carbon footprint record."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    inference_id: uuid.UUID
    model_id: str
    region: str
    energy_kwh: float
    carbon_intensity_gco2_per_kwh: float
    carbon_gco2: float
    renewable_percentage: float
    tokens_input: int | None
    tokens_output: int | None
    inference_duration_ms: int | None
    metadata: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class CarbonReportResponse(BaseModel):
    """Paginated response for GET /api/v1/energy/carbon/report."""

    items: list[CarbonRecordResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Energy routing schemas
# ---------------------------------------------------------------------------


class RouteWorkloadRequest(BaseModel):
    """Request body for POST /api/v1/energy/route/optimize."""

    workload_id: uuid.UUID = Field(description="External workload UUID to route")
    workload_type: str = Field(
        description="inference | training | batch_processing | fine_tuning"
    )
    candidate_regions: list[str] = Field(
        min_length=1,
        description="List of candidate region identifiers",
    )
    override_region: str | None = Field(
        default=None, description="Force routing to this region (bypass scoring)"
    )
    override_reason: str | None = Field(
        default=None, description="Explanation for override (required if override_region is set)"
    )

    @field_validator("candidate_regions")
    @classmethod
    def validate_candidate_regions(cls, regions: list[str]) -> list[str]:
        """Ensure candidate_regions has at least one non-empty entry.

        Args:
            regions: List of region identifiers from request body.

        Returns:
            Validated list of region identifiers.

        Raises:
            ValueError: If list is empty or contains blank strings.
        """
        if not regions:
            raise ValueError("candidate_regions must not be empty")
        if any(not r.strip() for r in regions):
            raise ValueError("All candidate_regions must be non-empty strings")
        return regions


class RegionScoreEntry(BaseModel):
    """Per-region scoring detail within a RoutingDecisionResponse."""

    region: str
    renewable_score: float
    latency_score: float
    composite_score: float
    carbon_intensity: float
    renewable_percentage: float


class RoutingDecisionResponse(BaseModel):
    """Response schema for a routing decision."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    workload_id: uuid.UUID
    workload_type: str
    selected_region: str
    candidate_regions: list[dict[str, Any]]
    renewable_score: float
    latency_score: float
    composite_score: float
    carbon_saved_gco2: float
    override_reason: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class EnergyProfileResponse(BaseModel):
    """Response schema for a regional energy profile."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    region: str
    display_name: str
    carbon_intensity_gco2_per_kwh: float
    renewable_percentage: float
    solar_percentage: float
    wind_percentage: float
    hydro_percentage: float
    nuclear_percentage: float
    estimated_latency_ms: int
    last_refreshed_at: datetime | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class EnergyProfileListResponse(BaseModel):
    """Response schema for GET /api/v1/energy/regions."""

    items: list[EnergyProfileResponse]
    total: int


# ---------------------------------------------------------------------------
# Sustainability report schemas
# ---------------------------------------------------------------------------


class GenerateReportRequest(BaseModel):
    """Request body for POST /api/v1/energy/sustainability/report."""

    title: str = Field(min_length=1, max_length=255, description="Report title")
    report_type: str = Field(
        default="quarterly",
        description="monthly | quarterly | annual | custom",
    )
    period_start: datetime = Field(description="Inclusive start of reporting period (UTC)")
    period_end: datetime = Field(description="Exclusive end of reporting period (UTC)")

    @field_validator("period_end")
    @classmethod
    def validate_period_end_after_start(cls, period_end: datetime, info: Any) -> datetime:
        """Ensure period_end is after period_start.

        Args:
            period_end: End datetime from request.
            info: Pydantic validation info containing other field values.

        Returns:
            Validated period_end.

        Raises:
            ValueError: If period_end <= period_start.
        """
        period_start = info.data.get("period_start")
        if period_start and period_end <= period_start:
            raise ValueError("period_end must be after period_start")
        return period_end


class SustainabilityReportResponse(BaseModel):
    """Response schema for a sustainability report."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    title: str
    report_type: str
    period_start: datetime
    period_end: datetime
    status: str
    total_inferences: int
    total_energy_kwh: float
    total_carbon_kg_co2: float
    average_renewable_percentage: float
    carbon_saved_kg_co2: float
    routing_optimisation_rate: float
    per_model_breakdown: dict[str, Any]
    per_region_breakdown: dict[str, Any]
    esg_score: float | None
    generated_at: datetime | None
    error_message: str | None
    requested_by: uuid.UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SustainabilityReportListResponse(BaseModel):
    """Paginated response for sustainability reports."""

    items: list[SustainabilityReportResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Optimization recommendation schemas
# ---------------------------------------------------------------------------


class OptimizationRecordResponse(BaseModel):
    """Response schema for a single optimization recommendation."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    category: str
    title: str
    description: str
    target_resource: str | None
    projected_savings_kg_co2: float
    projected_savings_kwh: float
    priority: str
    status: str
    implementation_effort: str
    evidence: dict[str, Any]
    implemented_at: datetime | None
    expires_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class OptimizationListResponse(BaseModel):
    """Response schema for GET /api/v1/energy/optimization/recommendations."""

    items: list[OptimizationRecordResponse]
    total: int
