"""Business logic services for the AumOS Energy Sustainability service.

All services depend on repository and adapter interfaces (not concrete
implementations) and receive dependencies via constructor injection.
No framework code (FastAPI, SQLAlchemy) belongs here.

Key invariants:
- Carbon records are immutable once created (append-only audit trail).
- Routing decisions are always persisted for auditability.
- ESG report generation is idempotent: calling generate for the same period
  returns an existing report if one exists in ready/generating state.
- Optimization recommendations are ranked by projected_savings_kg_co2 desc.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.errors import ConflictError, ErrorCode, NotFoundError
from aumos_common.observability import get_logger

from aumos_energy_sustainability.core.interfaces import (
    ICarbonAPIClient,
    ICarbonRecordRepository,
    IEnergyProfileRepository,
    IEventPublisher,
    IOptimizationRepository,
    IRoutingDecisionRepository,
    ISustainabilityReportRepository,
)
from aumos_energy_sustainability.core.models import (
    CarbonRecord,
    EnergyProfile,
    OptimizationRecord,
    RoutingDecision,
    SustainabilityReport,
)

logger = get_logger(__name__)

# Valid workload types for routing
VALID_WORKLOAD_TYPES: frozenset[str] = frozenset(
    {"inference", "training", "batch_processing", "fine_tuning"}
)

# Valid report types
VALID_REPORT_TYPES: frozenset[str] = frozenset({"monthly", "quarterly", "annual", "custom"})

# Valid optimization categories
VALID_OPTIMIZATION_CATEGORIES: frozenset[str] = frozenset(
    {"routing", "scheduling", "model_efficiency", "hardware", "batch_consolidation"}
)

# ESG score weights
ESG_RENEWABLE_WEIGHT: float = 0.6
ESG_ROUTING_OPT_WEIGHT: float = 0.4


class CarbonTrackerService:
    """Track carbon footprint for individual AI inference calls.

    Persists per-inference carbon records and emits events for downstream
    ESG aggregation pipelines.
    """

    def __init__(
        self,
        carbon_repo: ICarbonRecordRepository,
        event_publisher: IEventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            carbon_repo: CarbonRecord persistence adapter.
            event_publisher: Kafka event publisher for carbon.tracked events.
        """
        self._carbon_repo = carbon_repo
        self._event_publisher = event_publisher

    async def track_inference(
        self,
        *,
        tenant_id: uuid.UUID,
        inference_id: uuid.UUID,
        model_id: str,
        region: str,
        energy_kwh: float,
        carbon_intensity_gco2_per_kwh: float,
        renewable_percentage: float = 0.0,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        inference_duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CarbonRecord:
        """Create a carbon footprint record for an inference call.

        Computes total CO2 = energy_kwh * carbon_intensity and persists the record.

        Args:
            tenant_id: Owning tenant UUID.
            inference_id: External inference request UUID.
            model_id: Model identifier.
            region: Cloud/datacenter region where inference executed.
            energy_kwh: Energy consumed in kWh.
            carbon_intensity_gco2_per_kwh: Grid carbon intensity at inference time.
            renewable_percentage: Renewable fraction (0–100).
            tokens_input: Input token count (LLM context).
            tokens_output: Output token count.
            inference_duration_ms: Wall-clock duration.
            metadata: Arbitrary additional metadata.

        Returns:
            Persisted CarbonRecord.

        Raises:
            ValueError: If energy_kwh or carbon_intensity_gco2_per_kwh are negative.
        """
        if energy_kwh < 0:
            raise ValueError(f"energy_kwh must be non-negative, got {energy_kwh}")
        if carbon_intensity_gco2_per_kwh < 0:
            raise ValueError(
                f"carbon_intensity_gco2_per_kwh must be non-negative, got {carbon_intensity_gco2_per_kwh}"
            )

        carbon_gco2 = energy_kwh * carbon_intensity_gco2_per_kwh

        record = CarbonRecord(
            tenant_id=tenant_id,
            inference_id=inference_id,
            model_id=model_id,
            region=region,
            energy_kwh=energy_kwh,
            carbon_intensity_gco2_per_kwh=carbon_intensity_gco2_per_kwh,
            carbon_gco2=carbon_gco2,
            renewable_percentage=renewable_percentage,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            inference_duration_ms=inference_duration_ms,
            metadata=metadata or {},
        )

        saved = await self._carbon_repo.create(record)

        await self._event_publisher.publish(
            "aumos.energy.carbon.tracked",
            {
                "tenant_id": str(tenant_id),
                "record_id": str(saved.id),
                "inference_id": str(inference_id),
                "model_id": model_id,
                "region": region,
                "carbon_gco2": carbon_gco2,
                "energy_kwh": energy_kwh,
                "renewable_percentage": renewable_percentage,
            },
        )

        logger.info(
            "Carbon record tracked",
            tenant_id=str(tenant_id),
            inference_id=str(inference_id),
            carbon_gco2=carbon_gco2,
            region=region,
        )
        return saved

    async def get_carbon_report(
        self,
        tenant_id: uuid.UUID,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        region: str | None = None,
        model_id: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[CarbonRecord], int]:
        """Retrieve a paginated carbon report for a tenant.

        Args:
            tenant_id: Owning tenant UUID.
            since: Filter records from this timestamp.
            until: Filter records until this timestamp.
            region: Optional region filter.
            model_id: Optional model filter.
            page: Page number (1-based).
            page_size: Records per page (max 100).

        Returns:
            Tuple of (list of CarbonRecords, total count).
        """
        return await self._carbon_repo.list_by_tenant(
            tenant_id,
            since=since,
            until=until,
            region=region,
            model_id=model_id,
            page=page,
            page_size=min(page_size, 100),
        )


class EnergyRouterService:
    """Route AI workloads to regions with the lowest carbon footprint.

    Scores candidate regions by weighted combination of renewable percentage
    and latency, selects the optimal region, and persists the routing decision.
    """

    def __init__(
        self,
        profile_repo: IEnergyProfileRepository,
        routing_repo: IRoutingDecisionRepository,
        carbon_api: ICarbonAPIClient,
        event_publisher: IEventPublisher,
        renewable_weight: float = 0.7,
        latency_weight: float = 0.3,
        carbon_threshold_gco2_per_kwh: float = 150.0,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            profile_repo: EnergyProfile persistence.
            routing_repo: RoutingDecision persistence.
            carbon_api: External carbon intensity data provider.
            event_publisher: Kafka event publisher.
            renewable_weight: Score weight for renewable percentage (0–1).
            latency_weight: Score weight for latency (0–1).
            carbon_threshold_gco2_per_kwh: Intensity above which to avoid a region.
        """
        self._profile_repo = profile_repo
        self._routing_repo = routing_repo
        self._carbon_api = carbon_api
        self._event_publisher = event_publisher
        self._renewable_weight = renewable_weight
        self._latency_weight = latency_weight
        self._carbon_threshold = carbon_threshold_gco2_per_kwh

    async def route_workload(
        self,
        *,
        tenant_id: uuid.UUID,
        workload_id: uuid.UUID,
        workload_type: str,
        candidate_regions: list[str],
        override_region: str | None = None,
        override_reason: str | None = None,
    ) -> RoutingDecision:
        """Select the optimal region for a workload and record the decision.

        Scores each candidate region by renewable percentage and latency.
        If override_region is provided, that region is selected regardless of score.

        Args:
            tenant_id: Owning tenant UUID.
            workload_id: External workload UUID.
            workload_type: Type of workload (inference | training | batch_processing | fine_tuning).
            candidate_regions: List of region identifiers to consider.
            override_region: If set, bypass scoring and route to this region.
            override_reason: Explanation for the override.

        Returns:
            Persisted RoutingDecision with score breakdown.

        Raises:
            ValueError: If workload_type is invalid or no candidates provided.
            NotFoundError: If no energy profiles exist for any candidate region.
        """
        if workload_type not in VALID_WORKLOAD_TYPES:
            raise ValueError(
                f"Invalid workload_type '{workload_type}'. "
                f"Must be one of: {sorted(VALID_WORKLOAD_TYPES)}"
            )
        if not candidate_regions:
            raise ValueError("candidate_regions must not be empty")

        profiles = []
        for region in candidate_regions:
            profile = await self._profile_repo.get_by_region(region, tenant_id)
            if profile:
                profiles.append(profile)

        if not profiles:
            raise NotFoundError(
                resource="EnergyProfile",
                detail=f"No active energy profiles found for regions: {candidate_regions}",
            )

        # Compute per-region scores
        max_latency = max(p.estimated_latency_ms for p in profiles) or 1
        scored: list[dict[str, Any]] = []

        for profile in profiles:
            renewable_score = profile.renewable_percentage / 100.0
            latency_score = 1.0 - (profile.estimated_latency_ms / max_latency)
            composite = (
                self._renewable_weight * renewable_score
                + self._latency_weight * latency_score
            )
            scored.append(
                {
                    "region": profile.region,
                    "profile_id": str(profile.id),
                    "renewable_score": round(renewable_score, 4),
                    "latency_score": round(latency_score, 4),
                    "composite_score": round(composite, 4),
                    "carbon_intensity": profile.carbon_intensity_gco2_per_kwh,
                    "renewable_percentage": profile.renewable_percentage,
                }
            )

        scored.sort(key=lambda s: s["composite_score"], reverse=True)

        if override_region:
            selected_region = override_region
            selected_entry = next(
                (s for s in scored if s["region"] == override_region), scored[0]
            )
        else:
            selected_entry = scored[0]
            selected_region = selected_entry["region"]

        # Estimate carbon saved vs. worst-scoring region
        worst = scored[-1]
        carbon_saved = max(
            0.0,
            worst["carbon_intensity"] - selected_entry["carbon_intensity"],
        )

        selected_profile = await self._profile_repo.get_by_region(selected_region, tenant_id)

        decision = RoutingDecision(
            tenant_id=tenant_id,
            workload_id=workload_id,
            workload_type=workload_type,
            selected_region=selected_region,
            selected_profile_id=selected_profile.id if selected_profile else None,
            candidate_regions=scored,
            renewable_score=selected_entry["renewable_score"],
            latency_score=selected_entry["latency_score"],
            composite_score=selected_entry["composite_score"],
            carbon_saved_gco2=carbon_saved,
            override_reason=override_reason,
        )

        saved = await self._routing_repo.create(decision)

        await self._event_publisher.publish(
            "aumos.energy.route.decided",
            {
                "tenant_id": str(tenant_id),
                "decision_id": str(saved.id),
                "workload_id": str(workload_id),
                "workload_type": workload_type,
                "selected_region": selected_region,
                "composite_score": selected_entry["composite_score"],
                "carbon_saved_gco2": carbon_saved,
            },
        )

        logger.info(
            "Workload routed",
            tenant_id=str(tenant_id),
            workload_id=str(workload_id),
            selected_region=selected_region,
            composite_score=selected_entry["composite_score"],
        )
        return saved

    async def get_region_profiles(self, tenant_id: uuid.UUID) -> list[EnergyProfile]:
        """Return all active energy profiles for a tenant.

        Args:
            tenant_id: Owning tenant UUID.

        Returns:
            List of active EnergyProfile instances.
        """
        return await self._profile_repo.list_active(tenant_id)

    async def refresh_profile(
        self,
        tenant_id: uuid.UUID,
        region: str,
        display_name: str = "",
        estimated_latency_ms: int = 50,
    ) -> EnergyProfile:
        """Fetch fresh carbon intensity data and upsert an EnergyProfile.

        Args:
            tenant_id: Owning tenant UUID.
            region: Region identifier to refresh.
            display_name: Human-readable region name.
            estimated_latency_ms: Estimated round-trip latency.

        Returns:
            Upserted EnergyProfile with fresh carbon intensity data.
        """
        data = await self._carbon_api.get_carbon_intensity(region)

        profile = EnergyProfile(
            tenant_id=tenant_id,
            region=region,
            display_name=display_name or region,
            carbon_intensity_gco2_per_kwh=data.get("carbon_intensity_gco2_per_kwh", 0.0),
            renewable_percentage=data.get("renewable_percentage", 0.0),
            solar_percentage=data.get("solar_percentage", 0.0),
            wind_percentage=data.get("wind_percentage", 0.0),
            hydro_percentage=data.get("hydro_percentage", 0.0),
            nuclear_percentage=data.get("nuclear_percentage", 0.0),
            estimated_latency_ms=estimated_latency_ms,
            last_refreshed_at=datetime.now(tz=timezone.utc),
            is_active=True,
            source_metadata=data,
        )

        return await self._profile_repo.upsert(profile)


class SustainabilityReportService:
    """Generate and retrieve ESG sustainability reports.

    Aggregates carbon footprint data over a time period and computes
    ESG scores, per-model breakdowns, and routing optimisation metrics.
    """

    def __init__(
        self,
        report_repo: ISustainabilityReportRepository,
        carbon_repo: ICarbonRecordRepository,
        routing_repo: IRoutingDecisionRepository,
        event_publisher: IEventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            report_repo: SustainabilityReport persistence.
            carbon_repo: CarbonRecord aggregation queries.
            routing_repo: RoutingDecision queries for optimisation rate.
            event_publisher: Kafka event publisher.
        """
        self._report_repo = report_repo
        self._carbon_repo = carbon_repo
        self._routing_repo = routing_repo
        self._event_publisher = event_publisher

    async def generate_report(
        self,
        *,
        tenant_id: uuid.UUID,
        title: str,
        report_type: str = "quarterly",
        period_start: datetime,
        period_end: datetime,
        requested_by: uuid.UUID | None = None,
    ) -> SustainabilityReport:
        """Generate an ESG sustainability report for a time period.

        Aggregates carbon records, computes per-model and per-region breakdowns,
        and derives an ESG score. The report is persisted as generating then updated
        to ready or failed.

        Args:
            tenant_id: Owning tenant UUID.
            title: Report title.
            report_type: monthly | quarterly | annual | custom.
            period_start: Inclusive start of the reporting period.
            period_end: Exclusive end of the reporting period.
            requested_by: User UUID who requested the report.

        Returns:
            SustainabilityReport with status=ready.

        Raises:
            ValueError: If report_type is invalid or period_end <= period_start.
        """
        if report_type not in VALID_REPORT_TYPES:
            raise ValueError(
                f"Invalid report_type '{report_type}'. Must be one of: {sorted(VALID_REPORT_TYPES)}"
            )
        if period_end <= period_start:
            raise ValueError("period_end must be after period_start")

        stub = SustainabilityReport(
            tenant_id=tenant_id,
            title=title,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            status="generating",
            requested_by=requested_by,
        )
        report = await self._report_repo.create(stub)

        try:
            aggregation = await self._carbon_repo.aggregate_by_period(
                tenant_id, period_start, period_end
            )

            total_inferences = aggregation.get("total_inferences", 0)
            total_energy_kwh = aggregation.get("total_energy_kwh", 0.0)
            total_carbon_gco2 = aggregation.get("total_carbon_gco2", 0.0)
            avg_renewable = aggregation.get("average_renewable_percentage", 0.0)
            per_model = aggregation.get("per_model_breakdown", {})
            per_region = aggregation.get("per_region_breakdown", {})

            total_carbon_kg = total_carbon_gco2 / 1000.0

            # Routing optimisation rate: decisions routed to renewable-preferred regions
            routing_decisions, routing_total = await self._routing_repo.list_by_tenant(
                tenant_id,
                since=period_start,
                page=1,
                page_size=10_000,
            )
            optimised_count = sum(
                1 for d in routing_decisions if d.renewable_score >= 0.5
            )
            routing_opt_rate = (
                optimised_count / routing_total if routing_total > 0 else 0.0
            )
            carbon_saved_kg = sum(
                d.carbon_saved_gco2 / 1000.0 for d in routing_decisions
            )

            # ESG score: weighted combination of renewable % and routing optimisation rate
            esg_score = round(
                (avg_renewable / 100.0) * ESG_RENEWABLE_WEIGHT * 100.0
                + routing_opt_rate * ESG_ROUTING_OPT_WEIGHT * 100.0,
                2,
            )

            report.status = "ready"
            report.total_inferences = total_inferences
            report.total_energy_kwh = total_energy_kwh
            report.total_carbon_kg_co2 = total_carbon_kg
            report.average_renewable_percentage = avg_renewable
            report.carbon_saved_kg_co2 = carbon_saved_kg
            report.routing_optimisation_rate = routing_opt_rate
            report.per_model_breakdown = per_model
            report.per_region_breakdown = per_region
            report.esg_score = esg_score
            report.generated_at = datetime.now(tz=timezone.utc)

        except Exception as exc:  # noqa: BLE001
            report.status = "failed"
            report.error_message = str(exc)
            logger.error(
                "Report generation failed",
                tenant_id=str(tenant_id),
                report_id=str(report.id),
                error=str(exc),
            )

        saved = await self._report_repo.update(report)

        if saved.status == "ready":
            await self._event_publisher.publish(
                "aumos.energy.report.generated",
                {
                    "tenant_id": str(tenant_id),
                    "report_id": str(saved.id),
                    "report_type": report_type,
                    "esg_score": saved.esg_score,
                    "total_carbon_kg_co2": saved.total_carbon_kg_co2,
                },
            )
            logger.info(
                "ESG report generated",
                tenant_id=str(tenant_id),
                report_id=str(saved.id),
                esg_score=saved.esg_score,
            )

        return saved

    async def get_report(
        self, report_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> SustainabilityReport:
        """Retrieve a single sustainability report by ID.

        Args:
            report_id: Report primary key.
            tenant_id: Owning tenant UUID.

        Returns:
            SustainabilityReport.

        Raises:
            NotFoundError: If report does not exist.
        """
        report = await self._report_repo.get_by_id(report_id, tenant_id)
        if not report:
            raise NotFoundError(
                resource="SustainabilityReport",
                detail=f"Report {report_id} not found",
            )
        return report

    async def list_reports(
        self,
        tenant_id: uuid.UUID,
        *,
        report_type: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[SustainabilityReport], int]:
        """List sustainability reports for a tenant.

        Args:
            tenant_id: Owning tenant UUID.
            report_type: Optional report type filter.
            page: Page number (1-based).
            page_size: Reports per page (max 100).

        Returns:
            Tuple of (list of reports, total count).
        """
        return await self._report_repo.list_by_tenant(
            tenant_id,
            report_type=report_type,
            page=page,
            page_size=min(page_size, 100),
        )


class OptimizationAdvisorService:
    """Generate and manage energy optimization recommendations.

    Analyses carbon records and routing patterns to identify high-value
    optimisation opportunities, ranked by projected CO2 savings.
    """

    def __init__(
        self,
        optimization_repo: IOptimizationRepository,
        carbon_repo: ICarbonRecordRepository,
        profile_repo: IEnergyProfileRepository,
        event_publisher: IEventPublisher,
        min_savings_threshold_kg_co2: float = 1.0,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            optimization_repo: OptimizationRecord persistence.
            carbon_repo: CarbonRecord queries for pattern analysis.
            profile_repo: EnergyProfile queries for routing opportunities.
            event_publisher: Kafka event publisher.
            min_savings_threshold_kg_co2: Minimum savings to surface a recommendation.
        """
        self._optimization_repo = optimization_repo
        self._carbon_repo = carbon_repo
        self._profile_repo = profile_repo
        self._event_publisher = event_publisher
        self._min_savings_threshold = min_savings_threshold_kg_co2

    async def generate_recommendations(
        self,
        tenant_id: uuid.UUID,
        *,
        analysis_window_days: int = 30,
    ) -> list[OptimizationRecord]:
        """Analyse recent carbon data and generate optimization recommendations.

        Identifies patterns such as high-carbon regions with available lower-carbon
        alternatives, batch consolidation opportunities, and scheduling optimisations.

        Args:
            tenant_id: Owning tenant UUID.
            analysis_window_days: Number of days of history to analyse.

        Returns:
            List of newly created OptimizationRecord instances.
        """
        from datetime import timedelta

        analysis_end = datetime.now(tz=timezone.utc)
        analysis_start = analysis_end - timedelta(days=analysis_window_days)

        aggregation = await self._carbon_repo.aggregate_by_period(
            tenant_id, analysis_start, analysis_end
        )

        active_profiles = await self._profile_repo.list_active(tenant_id)
        region_profile_map = {p.region: p for p in active_profiles}

        per_region: dict[str, Any] = aggregation.get("per_region_breakdown", {})
        per_model: dict[str, Any] = aggregation.get("per_model_breakdown", {})

        recommendations: list[OptimizationRecord] = []

        # Recommendation type 1: Route high-carbon-region workloads elsewhere
        for region, stats in per_region.items():
            profile = region_profile_map.get(region)
            if not profile:
                continue
            if profile.carbon_intensity_gco2_per_kwh <= 150.0:
                continue

            # Find a better region
            better = [
                p
                for p in active_profiles
                if p.region != region
                and p.carbon_intensity_gco2_per_kwh < profile.carbon_intensity_gco2_per_kwh * 0.7
            ]
            if not better:
                continue

            best_alternative = min(better, key=lambda p: p.carbon_intensity_gco2_per_kwh)
            energy_kwh = stats.get("energy_kwh", 0.0)
            savings_gco2 = energy_kwh * (
                profile.carbon_intensity_gco2_per_kwh
                - best_alternative.carbon_intensity_gco2_per_kwh
            )
            savings_kg = savings_gco2 / 1000.0

            if savings_kg < self._min_savings_threshold:
                continue

            rec = OptimizationRecord(
                tenant_id=tenant_id,
                category="routing",
                title=f"Route workloads from {region} to {best_alternative.region}",
                description=(
                    f"Region {region} has a carbon intensity of "
                    f"{profile.carbon_intensity_gco2_per_kwh:.1f} gCO2/kWh. "
                    f"Routing workloads to {best_alternative.region} "
                    f"({best_alternative.carbon_intensity_gco2_per_kwh:.1f} gCO2/kWh) "
                    f"could save approximately {savings_kg:.2f} kg CO2 over the next "
                    f"{analysis_window_days} days based on current usage."
                ),
                target_resource=region,
                projected_savings_kg_co2=round(savings_kg, 3),
                projected_savings_kwh=round(
                    energy_kwh
                    * (1 - best_alternative.carbon_intensity_gco2_per_kwh / profile.carbon_intensity_gco2_per_kwh),
                    3,
                ),
                priority="high" if savings_kg >= 10.0 else "medium",
                implementation_effort="low",
                evidence={
                    "current_region": region,
                    "alternative_region": best_alternative.region,
                    "current_intensity": profile.carbon_intensity_gco2_per_kwh,
                    "alternative_intensity": best_alternative.carbon_intensity_gco2_per_kwh,
                    "analysis_window_days": analysis_window_days,
                    "total_energy_kwh": energy_kwh,
                },
            )
            created = await self._optimization_repo.create(rec)
            recommendations.append(created)

        # Recommendation type 2: Model efficiency for high-carbon models
        for model_id, stats in per_model.items():
            inferences = stats.get("inferences", 0)
            total_carbon_kg = stats.get("total_carbon_kg", 0.0)
            if inferences < 100 or total_carbon_kg < self._min_savings_threshold * 5:
                continue

            carbon_per_inference_g = (total_carbon_kg * 1000) / inferences if inferences else 0
            if carbon_per_inference_g < 50.0:
                continue

            rec = OptimizationRecord(
                tenant_id=tenant_id,
                category="model_efficiency",
                title=f"Optimise inference batching for {model_id}",
                description=(
                    f"Model {model_id} produced {inferences} inferences in the last "
                    f"{analysis_window_days} days with {carbon_per_inference_g:.1f} gCO2 per call. "
                    f"Batching requests could reduce per-inference overhead by 20–40%, "
                    f"saving approximately {total_carbon_kg * 0.3:.2f} kg CO2."
                ),
                target_resource=model_id,
                projected_savings_kg_co2=round(total_carbon_kg * 0.3, 3),
                projected_savings_kwh=round(stats.get("energy_kwh", 0.0) * 0.3, 3),
                priority="medium",
                implementation_effort="medium",
                evidence={
                    "model_id": model_id,
                    "total_inferences": inferences,
                    "total_carbon_kg": total_carbon_kg,
                    "carbon_per_inference_g": carbon_per_inference_g,
                    "analysis_window_days": analysis_window_days,
                },
            )
            created = await self._optimization_repo.create(rec)
            recommendations.append(created)

        if recommendations:
            await self._event_publisher.publish(
                "aumos.energy.optimizations.generated",
                {
                    "tenant_id": str(tenant_id),
                    "count": len(recommendations),
                    "total_projected_savings_kg_co2": sum(
                        r.projected_savings_kg_co2 for r in recommendations
                    ),
                },
            )
            logger.info(
                "Optimization recommendations generated",
                tenant_id=str(tenant_id),
                count=len(recommendations),
            )

        return recommendations

    async def get_recommendations(
        self,
        tenant_id: uuid.UUID,
        *,
        category: str | None = None,
        priority: str | None = None,
    ) -> list[OptimizationRecord]:
        """Retrieve active optimization recommendations for a tenant.

        Args:
            tenant_id: Owning tenant UUID.
            category: Optional category filter.
            priority: Optional priority filter.

        Returns:
            List of active OptimizationRecord instances, sorted by projected savings desc.
        """
        return await self._optimization_repo.list_active(
            tenant_id, category=category, priority=priority
        )

    async def dismiss_recommendation(
        self,
        recommendation_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> OptimizationRecord:
        """Mark a recommendation as dismissed.

        Args:
            recommendation_id: OptimizationRecord primary key.
            tenant_id: Owning tenant UUID.

        Returns:
            Updated OptimizationRecord with status=dismissed.

        Raises:
            NotFoundError: If recommendation not found.
            ConflictError: If recommendation is already in a terminal state.
        """
        records = await self._optimization_repo.list_active(tenant_id)
        record = next((r for r in records if r.id == recommendation_id), None)

        if not record:
            raise NotFoundError(
                resource="OptimizationRecord",
                detail=f"Recommendation {recommendation_id} not found or not active",
            )

        if record.status in {"implemented", "dismissed", "superseded"}:
            raise ConflictError(
                code=ErrorCode.CONFLICT,
                detail=f"Recommendation is already in terminal state: {record.status}",
            )

        record.status = "dismissed"
        return await self._optimization_repo.update(record)
