"""FastAPI router for the AumOS Energy Sustainability REST API.

All endpoints are prefixed with /api/v1. Authentication and tenant extraction
are handled by aumos-auth-gateway upstream; tenant_id is available via JWT.

Business logic is never implemented here — routes delegate entirely to services.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status

from aumos_common.errors import ConflictError, NotFoundError
from aumos_common.observability import get_logger

from aumos_energy_sustainability.api.schemas import (
    CarbonRecordResponse,
    CarbonReportResponse,
    EnergyProfileListResponse,
    EnergyProfileResponse,
    GenerateReportRequest,
    OptimizationListResponse,
    OptimizationRecordResponse,
    RouteWorkloadRequest,
    RoutingDecisionResponse,
    SustainabilityReportListResponse,
    SustainabilityReportResponse,
    TrackCarbonRequest,
)
from aumos_energy_sustainability.core.services import (
    CarbonTrackerService,
    EnergyRouterService,
    OptimizationAdvisorService,
    SustainabilityReportService,
)

logger = get_logger(__name__)

router = APIRouter(tags=["energy-sustainability"])


# ---------------------------------------------------------------------------
# Dependency helpers — populated in lifespan via app.state
# ---------------------------------------------------------------------------


def _get_carbon_service(request: Request) -> CarbonTrackerService:
    """Retrieve CarbonTrackerService from app state.

    Args:
        request: FastAPI request with app state populated in lifespan.

    Returns:
        CarbonTrackerService instance.
    """
    return request.app.state.carbon_service  # type: ignore[no-any-return]


def _get_router_service(request: Request) -> EnergyRouterService:
    """Retrieve EnergyRouterService from app state.

    Args:
        request: FastAPI request with app state populated in lifespan.

    Returns:
        EnergyRouterService instance.
    """
    return request.app.state.router_service  # type: ignore[no-any-return]


def _get_report_service(request: Request) -> SustainabilityReportService:
    """Retrieve SustainabilityReportService from app state.

    Args:
        request: FastAPI request with app state populated in lifespan.

    Returns:
        SustainabilityReportService instance.
    """
    return request.app.state.report_service  # type: ignore[no-any-return]


def _get_optimization_service(request: Request) -> OptimizationAdvisorService:
    """Retrieve OptimizationAdvisorService from app state.

    Args:
        request: FastAPI request with app state populated in lifespan.

    Returns:
        OptimizationAdvisorService instance.
    """
    return request.app.state.optimization_service  # type: ignore[no-any-return]


def _tenant_id_from_request(request: Request) -> uuid.UUID:
    """Extract tenant UUID from request headers (set by auth middleware).

    Falls back to a random UUID in development mode.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Tenant UUID.
    """
    tenant_header = request.headers.get("X-Tenant-ID")
    if tenant_header:
        return uuid.UUID(tenant_header)
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# Carbon tracking endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/energy/carbon/track",
    response_model=CarbonRecordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Track inference carbon footprint",
    description=(
        "Record the energy consumed and CO2 emitted for a single AI inference call. "
        "Carbon is computed as energy_kwh * carbon_intensity_gco2_per_kwh."
    ),
)
async def track_carbon(
    request_body: TrackCarbonRequest,
    request: Request,
    service: CarbonTrackerService = Depends(_get_carbon_service),
) -> CarbonRecordResponse:
    """Create a carbon footprint record for an inference call.

    Args:
        request_body: Inference carbon tracking parameters.
        request: FastAPI request for tenant extraction.
        service: CarbonTrackerService dependency.

    Returns:
        CarbonRecordResponse with the persisted record.

    Raises:
        HTTPException 400: If energy or intensity values are negative.
    """
    tenant_id = _tenant_id_from_request(request)

    try:
        record = await service.track_inference(
            tenant_id=tenant_id,
            inference_id=request_body.inference_id,
            model_id=request_body.model_id,
            region=request_body.region,
            energy_kwh=request_body.energy_kwh,
            carbon_intensity_gco2_per_kwh=request_body.carbon_intensity_gco2_per_kwh,
            renewable_percentage=request_body.renewable_percentage,
            tokens_input=request_body.tokens_input,
            tokens_output=request_body.tokens_output,
            inference_duration_ms=request_body.inference_duration_ms,
            metadata=request_body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info(
        "Carbon track API call",
        tenant_id=str(tenant_id),
        record_id=str(record.id),
    )
    return CarbonRecordResponse.model_validate(record)


@router.get(
    "/energy/carbon/report",
    response_model=CarbonReportResponse,
    summary="Carbon footprint report",
    description=(
        "Retrieve a paginated list of carbon footprint records for the current tenant, "
        "with optional filters by time range, region, and model."
    ),
)
async def get_carbon_report(
    since: datetime | None = None,
    until: datetime | None = None,
    region: str | None = None,
    model_id: str | None = None,
    page: int = 1,
    page_size: int = 50,
    request: Request = ...,  # type: ignore[assignment]
    service: CarbonTrackerService = Depends(_get_carbon_service),
) -> CarbonReportResponse:
    """Retrieve paginated carbon records for the current tenant.

    Args:
        since: Filter records created at or after this timestamp.
        until: Filter records created before this timestamp.
        region: Optional region filter.
        model_id: Optional model identifier filter.
        page: Page number (1-based).
        page_size: Records per page (max 100).
        request: FastAPI request for tenant extraction.
        service: CarbonTrackerService dependency.

    Returns:
        CarbonReportResponse with paginated records.
    """
    tenant_id = _tenant_id_from_request(request)
    records, total = await service.get_carbon_report(
        tenant_id,
        since=since,
        until=until,
        region=region,
        model_id=model_id,
        page=page,
        page_size=page_size,
    )

    return CarbonReportResponse(
        items=[CarbonRecordResponse.model_validate(r) for r in records],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Energy routing endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/energy/route/optimize",
    response_model=RoutingDecisionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Route workload to renewable region",
    description=(
        "Score candidate regions by renewable percentage and latency, "
        "select the optimal region, and record the routing decision."
    ),
)
async def route_workload(
    request_body: RouteWorkloadRequest,
    request: Request,
    service: EnergyRouterService = Depends(_get_router_service),
) -> RoutingDecisionResponse:
    """Route a workload to the optimal region based on energy profile.

    Args:
        request_body: Routing request with workload and candidates.
        request: FastAPI request for tenant extraction.
        service: EnergyRouterService dependency.

    Returns:
        RoutingDecisionResponse with selected region and score breakdown.

    Raises:
        HTTPException 400: If workload_type is invalid or candidates are missing.
        HTTPException 404: If no energy profiles exist for any candidate.
    """
    tenant_id = _tenant_id_from_request(request)

    try:
        decision = await service.route_workload(
            tenant_id=tenant_id,
            workload_id=request_body.workload_id,
            workload_type=request_body.workload_type,
            candidate_regions=request_body.candidate_regions,
            override_region=request_body.override_region,
            override_reason=request_body.override_reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return RoutingDecisionResponse.model_validate(decision)


@router.get(
    "/energy/regions",
    response_model=EnergyProfileListResponse,
    summary="List regional energy profiles",
    description=(
        "Retrieve all active regional energy profiles for the current tenant, "
        "including carbon intensity, renewable percentage, and latency."
    ),
)
async def list_regions(
    request: Request,
    service: EnergyRouterService = Depends(_get_router_service),
) -> EnergyProfileListResponse:
    """List all active regional energy profiles.

    Args:
        request: FastAPI request for tenant extraction.
        service: EnergyRouterService dependency.

    Returns:
        EnergyProfileListResponse with all active profiles.
    """
    tenant_id = _tenant_id_from_request(request)
    profiles = await service.get_region_profiles(tenant_id)

    return EnergyProfileListResponse(
        items=[EnergyProfileResponse.model_validate(p) for p in profiles],
        total=len(profiles),
    )


# ---------------------------------------------------------------------------
# Sustainability report endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/energy/sustainability/report",
    response_model=SustainabilityReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate ESG sustainability report",
    description=(
        "Generate a new ESG sustainability report covering a specified time period. "
        "Aggregates carbon footprint, routing optimisation rate, and per-model/region breakdowns."
    ),
)
async def generate_sustainability_report(
    request_body: GenerateReportRequest,
    request: Request,
    service: SustainabilityReportService = Depends(_get_report_service),
) -> SustainabilityReportResponse:
    """Generate an ESG sustainability report.

    Args:
        request_body: Report generation parameters.
        request: FastAPI request for tenant extraction and requestor identity.
        service: SustainabilityReportService dependency.

    Returns:
        SustainabilityReportResponse with aggregated metrics and ESG score.

    Raises:
        HTTPException 400: If report_type is invalid or period_end <= period_start.
    """
    tenant_id = _tenant_id_from_request(request)

    try:
        report = await service.generate_report(
            tenant_id=tenant_id,
            title=request_body.title,
            report_type=request_body.report_type,
            period_start=request_body.period_start,
            period_end=request_body.period_end,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info(
        "ESG report generation API call",
        tenant_id=str(tenant_id),
        report_id=str(report.id),
        status=report.status,
    )
    return SustainabilityReportResponse.model_validate(report)


@router.get(
    "/energy/sustainability/reports",
    response_model=SustainabilityReportListResponse,
    summary="List ESG sustainability reports",
    description="List sustainability reports for the current tenant with optional type filter.",
)
async def list_sustainability_reports(
    report_type: str | None = None,
    page: int = 1,
    page_size: int = 20,
    request: Request = ...,  # type: ignore[assignment]
    service: SustainabilityReportService = Depends(_get_report_service),
) -> SustainabilityReportListResponse:
    """List sustainability reports for the current tenant.

    Args:
        report_type: Optional type filter (monthly | quarterly | annual | custom).
        page: Page number (1-based).
        page_size: Reports per page (max 100).
        request: FastAPI request for tenant extraction.
        service: SustainabilityReportService dependency.

    Returns:
        SustainabilityReportListResponse with paginated results.
    """
    tenant_id = _tenant_id_from_request(request)
    reports, total = await service.list_reports(
        tenant_id,
        report_type=report_type,
        page=page,
        page_size=page_size,
    )

    return SustainabilityReportListResponse(
        items=[SustainabilityReportResponse.model_validate(r) for r in reports],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/energy/sustainability/reports/{report_id}",
    response_model=SustainabilityReportResponse,
    summary="Get ESG sustainability report",
    description="Retrieve a single sustainability report by ID.",
)
async def get_sustainability_report(
    report_id: uuid.UUID,
    request: Request,
    service: SustainabilityReportService = Depends(_get_report_service),
) -> SustainabilityReportResponse:
    """Retrieve a single sustainability report.

    Args:
        report_id: Report UUID.
        request: FastAPI request for tenant extraction.
        service: SustainabilityReportService dependency.

    Returns:
        SustainabilityReportResponse.

    Raises:
        HTTPException 404: If report not found.
    """
    tenant_id = _tenant_id_from_request(request)

    try:
        report = await service.get_report(report_id, tenant_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return SustainabilityReportResponse.model_validate(report)


# ---------------------------------------------------------------------------
# Optimization recommendation endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/energy/optimization/recommendations",
    response_model=OptimizationListResponse,
    summary="List energy optimization recommendations",
    description=(
        "Retrieve active energy optimization recommendations for the current tenant, "
        "sorted by projected CO2 savings descending."
    ),
)
async def list_optimization_recommendations(
    category: str | None = None,
    priority: str | None = None,
    request: Request = ...,  # type: ignore[assignment]
    service: OptimizationAdvisorService = Depends(_get_optimization_service),
) -> OptimizationListResponse:
    """List active optimization recommendations.

    Args:
        category: Optional category filter (routing | scheduling | model_efficiency | ...).
        priority: Optional priority filter (high | medium | low).
        request: FastAPI request for tenant extraction.
        service: OptimizationAdvisorService dependency.

    Returns:
        OptimizationListResponse with recommendations sorted by projected savings.
    """
    tenant_id = _tenant_id_from_request(request)
    recommendations = await service.get_recommendations(
        tenant_id, category=category, priority=priority
    )

    return OptimizationListResponse(
        items=[OptimizationRecordResponse.model_validate(r) for r in recommendations],
        total=len(recommendations),
    )


@router.post(
    "/energy/optimization/recommendations/generate",
    response_model=OptimizationListResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate optimization recommendations",
    description=(
        "Analyse recent carbon footprint data and generate actionable energy "
        "optimization recommendations, ranked by projected CO2 savings."
    ),
)
async def generate_optimization_recommendations(
    analysis_window_days: int = 30,
    request: Request = ...,  # type: ignore[assignment]
    service: OptimizationAdvisorService = Depends(_get_optimization_service),
) -> OptimizationListResponse:
    """Trigger generation of fresh optimization recommendations.

    Args:
        analysis_window_days: Days of carbon history to analyse (default 30).
        request: FastAPI request for tenant extraction.
        service: OptimizationAdvisorService dependency.

    Returns:
        OptimizationListResponse with newly created recommendations.

    Raises:
        HTTPException 400: If analysis_window_days < 1.
    """
    if analysis_window_days < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="analysis_window_days must be at least 1",
        )

    tenant_id = _tenant_id_from_request(request)
    recommendations = await service.generate_recommendations(
        tenant_id, analysis_window_days=analysis_window_days
    )

    return OptimizationListResponse(
        items=[OptimizationRecordResponse.model_validate(r) for r in recommendations],
        total=len(recommendations),
    )
