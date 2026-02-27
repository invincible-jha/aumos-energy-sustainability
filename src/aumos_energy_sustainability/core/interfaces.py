"""Hexagonal architecture port interfaces for the Energy Sustainability service.

All adapter implementations must satisfy these protocols. Services depend
only on these interfaces — never on concrete SQLAlchemy or HTTP client types.
"""

import uuid
from datetime import datetime
from typing import Any, Protocol

from aumos_energy_sustainability.core.models import (
    CarbonRecord,
    EnergyProfile,
    OptimizationRecord,
    RoutingDecision,
    SustainabilityReport,
)


# ---------------------------------------------------------------------------
# Repository protocols (driven ports — database side)
# ---------------------------------------------------------------------------


class ICarbonRecordRepository(Protocol):
    """Persistence contract for CarbonRecord entities."""

    async def create(self, record: CarbonRecord) -> CarbonRecord:
        """Persist a new CarbonRecord and return the saved instance.

        Args:
            record: The CarbonRecord to persist.

        Returns:
            Persisted CarbonRecord with populated id and timestamps.
        """
        ...

    async def get_by_id(
        self, record_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> CarbonRecord | None:
        """Retrieve a single CarbonRecord by primary key and tenant.

        Args:
            record_id: Record primary key.
            tenant_id: Owning tenant UUID.

        Returns:
            CarbonRecord if found, None otherwise.
        """
        ...

    async def list_by_tenant(
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
        """List CarbonRecords for a tenant with optional filters.

        Args:
            tenant_id: Owning tenant UUID.
            since: Filter records created at or after this timestamp.
            until: Filter records created before this timestamp.
            region: Filter by region identifier.
            model_id: Filter by model identifier.
            page: 1-based page number.
            page_size: Records per page.

        Returns:
            Tuple of (page of records, total count).
        """
        ...

    async def aggregate_by_period(
        self,
        tenant_id: uuid.UUID,
        period_start: datetime,
        period_end: datetime,
    ) -> dict[str, Any]:
        """Aggregate carbon metrics for a tenant over a reporting period.

        Args:
            tenant_id: Owning tenant UUID.
            period_start: Inclusive start of aggregation period.
            period_end: Exclusive end of aggregation period.

        Returns:
            Dict containing total_inferences, total_energy_kwh, total_carbon_gco2,
            average_renewable_percentage, per_model_breakdown, per_region_breakdown.
        """
        ...


class IEnergyProfileRepository(Protocol):
    """Persistence contract for EnergyProfile entities."""

    async def upsert(self, profile: EnergyProfile) -> EnergyProfile:
        """Create or update an EnergyProfile for a region.

        Args:
            profile: Profile with region identifier.

        Returns:
            Persisted EnergyProfile.
        """
        ...

    async def get_by_region(
        self, region: str, tenant_id: uuid.UUID
    ) -> EnergyProfile | None:
        """Retrieve an active EnergyProfile by region for a tenant.

        Args:
            region: Region identifier.
            tenant_id: Owning tenant UUID.

        Returns:
            Active EnergyProfile if found, None otherwise.
        """
        ...

    async def list_active(self, tenant_id: uuid.UUID) -> list[EnergyProfile]:
        """List all active EnergyProfiles for a tenant.

        Args:
            tenant_id: Owning tenant UUID.

        Returns:
            List of active EnergyProfile instances.
        """
        ...

    async def delete(self, region: str, tenant_id: uuid.UUID) -> bool:
        """Soft-delete an EnergyProfile.

        Args:
            region: Region identifier.
            tenant_id: Owning tenant UUID.

        Returns:
            True if profile was found and deleted, False otherwise.
        """
        ...


class IRoutingDecisionRepository(Protocol):
    """Persistence contract for RoutingDecision entities."""

    async def create(self, decision: RoutingDecision) -> RoutingDecision:
        """Persist a new RoutingDecision.

        Args:
            decision: The RoutingDecision to persist.

        Returns:
            Persisted RoutingDecision with populated id.
        """
        ...

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        since: datetime | None = None,
        workload_type: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[RoutingDecision], int]:
        """List RoutingDecisions for a tenant.

        Args:
            tenant_id: Owning tenant UUID.
            since: Filter decisions created at or after this timestamp.
            workload_type: Filter by workload type.
            page: 1-based page number.
            page_size: Records per page.

        Returns:
            Tuple of (page of decisions, total count).
        """
        ...


class ISustainabilityReportRepository(Protocol):
    """Persistence contract for SustainabilityReport entities."""

    async def create(self, report: SustainabilityReport) -> SustainabilityReport:
        """Persist a new SustainabilityReport stub (status=generating).

        Args:
            report: The report to persist.

        Returns:
            Persisted SustainabilityReport.
        """
        ...

    async def update(self, report: SustainabilityReport) -> SustainabilityReport:
        """Save updated report fields (status, metrics).

        Args:
            report: Report with updated fields.

        Returns:
            Updated SustainabilityReport.
        """
        ...

    async def get_by_id(
        self, report_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> SustainabilityReport | None:
        """Retrieve a SustainabilityReport by primary key.

        Args:
            report_id: Report primary key.
            tenant_id: Owning tenant UUID.

        Returns:
            SustainabilityReport if found, None otherwise.
        """
        ...

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        report_type: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[SustainabilityReport], int]:
        """List SustainabilityReports for a tenant.

        Args:
            tenant_id: Owning tenant UUID.
            report_type: Optional filter by report_type.
            page: 1-based page number.
            page_size: Records per page.

        Returns:
            Tuple of (page of reports, total count).
        """
        ...


class IOptimizationRepository(Protocol):
    """Persistence contract for OptimizationRecord entities."""

    async def create(self, record: OptimizationRecord) -> OptimizationRecord:
        """Persist a new OptimizationRecord.

        Args:
            record: The record to persist.

        Returns:
            Persisted OptimizationRecord.
        """
        ...

    async def update(self, record: OptimizationRecord) -> OptimizationRecord:
        """Save updated recommendation fields (status, priority).

        Args:
            record: Record with updated fields.

        Returns:
            Updated OptimizationRecord.
        """
        ...

    async def list_active(
        self,
        tenant_id: uuid.UUID,
        *,
        category: str | None = None,
        priority: str | None = None,
    ) -> list[OptimizationRecord]:
        """List active optimization recommendations for a tenant.

        Args:
            tenant_id: Owning tenant UUID.
            category: Optional filter by recommendation category.
            priority: Optional filter by priority level.

        Returns:
            List of active OptimizationRecord instances ordered by projected_savings_kg_co2 desc.
        """
        ...


# ---------------------------------------------------------------------------
# External adapter protocols (driven ports — outbound side)
# ---------------------------------------------------------------------------


class ICarbonAPIClient(Protocol):
    """Contract for fetching real-time carbon intensity data from an external provider."""

    async def get_carbon_intensity(self, region: str) -> dict[str, Any]:
        """Fetch the current carbon intensity for a region.

        Args:
            region: Region identifier (provider-specific zone code).

        Returns:
            Dict containing carbon_intensity_gco2_per_kwh, renewable_percentage,
            and optional breakdown by source.
        """
        ...

    async def list_zones(self) -> list[dict[str, Any]]:
        """List all supported zones from the carbon intensity provider.

        Returns:
            List of zone dicts with zone_key, country_name, and display_name.
        """
        ...


class IEventPublisher(Protocol):
    """Contract for publishing domain events to the AumOS event bus."""

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish a domain event to a Kafka topic.

        Args:
            topic: Kafka topic name.
            event: Event payload to serialize and publish.
        """
        ...


# ---------------------------------------------------------------------------
# Domain adapter protocols (driven ports — specialist adapter side)
# ---------------------------------------------------------------------------


class IEnergyRouter(Protocol):
    """Contract for carbon-score-based workload routing across regions."""

    async def get_region_carbon_intensity(self, region: str) -> dict[str, Any]:
        """Fetch current carbon intensity and renewable percentage for a region.

        Args:
            region: Cloud region identifier.

        Returns:
            Dict with carbon_intensity_gco2_per_kwh, renewable_percentage,
            is_renewable_peak, and timestamp.
        """
        ...

    async def score_regions(
        self,
        regions: list[str],
        *,
        renewable_weight: float = 0.7,
        latency_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Score and rank candidate regions by composite renewable and latency score.

        Args:
            regions: List of region identifiers to evaluate.
            renewable_weight: Weight for renewable percentage (0–1).
            latency_weight: Weight for latency score (0–1).

        Returns:
            Sorted list of region score dicts (highest score first).
        """
        ...

    async def route_by_carbon_score(
        self,
        workload_id: str,
        candidate_regions: list[str],
        workload_type: str,
    ) -> dict[str, Any]:
        """Select the optimal region for a workload based on carbon score.

        Args:
            workload_id: Unique workload identifier for audit logging.
            candidate_regions: Regions to consider for routing.
            workload_type: Type of workload (inference | training | batch).

        Returns:
            Routing decision dict with selected_region, score, and rationale.
        """
        ...


class IEfficiencyOptimizer(Protocol):
    """Contract for GPU utilization monitoring and batch size optimization."""

    async def monitor_gpu_utilization(
        self, cluster_id: str, namespace: str
    ) -> dict[str, Any]:
        """Collect real-time GPU utilization metrics for a serving cluster.

        Args:
            cluster_id: Kubernetes cluster identifier.
            namespace: Kubernetes namespace to query.

        Returns:
            Dict with per-gpu utilization, memory usage, and aggregate stats.
        """
        ...

    async def optimize_batch_size(
        self,
        model_id: str,
        current_batch_size: int,
        gpu_utilization_pct: float,
        gpu_memory_used_gb: float,
        gpu_memory_total_gb: float,
    ) -> dict[str, Any]:
        """Recommend an optimal batch size based on current GPU utilization.

        Args:
            model_id: Model being served.
            current_batch_size: Current configured batch size.
            gpu_utilization_pct: Observed GPU compute utilization (0–100).
            gpu_memory_used_gb: Currently allocated GPU memory in GB.
            gpu_memory_total_gb: Total available GPU memory in GB.

        Returns:
            Dict with recommended_batch_size and projected efficiency gain.
        """
        ...

    async def compute_efficiency_score(
        self, cluster_id: str, namespace: str
    ) -> dict[str, Any]:
        """Compute an aggregate efficiency score for the cluster.

        Args:
            cluster_id: Kubernetes cluster identifier.
            namespace: Kubernetes namespace.

        Returns:
            Dict with efficiency_score (0–100) and contributing metrics.
        """
        ...


class IModelCompressor(Protocol):
    """Contract for model quantization, pruning, and distillation pipelines."""

    async def quantize_model(
        self,
        model_id: str,
        source_format: str,
        target_precision: str,
        *,
        calibration_dataset_path: str | None = None,
    ) -> dict[str, Any]:
        """Apply quantization to reduce model precision and memory footprint.

        Args:
            model_id: Model to quantize.
            source_format: Source precision (fp32 | fp16 | bf16).
            target_precision: Target precision (int8 | int4).
            calibration_dataset_path: Optional path to calibration data.

        Returns:
            Dict with compression_ratio, quality_impact, and energy_reduction_pct.
        """
        ...

    async def prune_model(
        self,
        model_id: str,
        pruning_method: str,
        sparsity_target: float,
    ) -> dict[str, Any]:
        """Prune model weights to reduce compute requirements.

        Args:
            model_id: Model to prune.
            pruning_method: Pruning strategy (magnitude | structured | random).
            sparsity_target: Target sparsity ratio (0.0–1.0).

        Returns:
            Dict with achieved_sparsity, parameter_reduction_pct, and quality_impact.
        """
        ...

    async def recommend_compression(
        self,
        model_id: str,
        model_size_billion_params: float,
        target_memory_gb: float,
    ) -> dict[str, Any]:
        """Recommend the best compression strategy to meet a memory target.

        Args:
            model_id: Model to compress.
            model_size_billion_params: Model parameter count in billions.
            target_memory_gb: Maximum acceptable memory footprint.

        Returns:
            Dict with recommended_method, expected_quality_impact, and steps.
        """
        ...


class ISustainabilityReporter(Protocol):
    """Contract for ESG emissions tracking and standards-compliant reporting."""

    async def track_scope_1_emissions(
        self,
        tenant_id: str,
        *,
        fuel_type: str,
        fuel_consumed_liters: float,
        period_start: str,
        period_end: str,
    ) -> dict[str, Any]:
        """Record Scope 1 (direct) emissions from on-premise fuel combustion.

        Args:
            tenant_id: Tenant identifier for isolation.
            fuel_type: Type of fuel consumed (diesel | natural_gas | propane).
            fuel_consumed_liters: Volume of fuel consumed in liters.
            period_start: ISO 8601 start of reporting period.
            period_end: ISO 8601 end of reporting period.

        Returns:
            Dict with co2_kg, ch4_kg, n2o_kg, and co2e_kg.
        """
        ...

    async def compile_esg_metrics(
        self, tenant_id: str, period_start: str, period_end: str
    ) -> dict[str, Any]:
        """Compile all ESG emissions metrics for a reporting period.

        Args:
            tenant_id: Tenant identifier.
            period_start: ISO 8601 period start.
            period_end: ISO 8601 period end.

        Returns:
            Dict with scope_1, scope_2, scope_3 totals, water_usage_m3,
            renewable_energy_kwh, and esg_score.
        """
        ...

    async def generate_gri_report(
        self, tenant_id: str, period_start: str, period_end: str
    ) -> dict[str, Any]:
        """Generate a GRI Standards (302, 303, 305) compliant sustainability report.

        Args:
            tenant_id: Tenant identifier.
            period_start: ISO 8601 period start.
            period_end: ISO 8601 period end.

        Returns:
            Dict structured according to GRI disclosure requirements.
        """
        ...


class IGreenScorer(Protocol):
    """Contract for environmental impact scoring and certification tracking."""

    async def compute_workload_carbon_footprint(
        self,
        model_id: str,
        region: str,
        *,
        energy_kwh: float,
        carbon_intensity_gco2_per_kwh: float,
        inference_count: int = 1,
    ) -> dict[str, Any]:
        """Compute the carbon footprint for a workload execution.

        Args:
            model_id: Model identifier.
            region: Execution region.
            energy_kwh: Energy consumed in kWh.
            carbon_intensity_gco2_per_kwh: Grid carbon intensity.
            inference_count: Number of inference calls in this workload.

        Returns:
            Dict with total_carbon_gco2, per_inference_carbon_gco2, and category.
        """
        ...

    async def compute_energy_efficiency_score(
        self,
        region: str,
        pue: float,
        renewable_percentage: float,
        carbon_intensity_gco2_per_kwh: float,
    ) -> dict[str, Any]:
        """Compute an energy efficiency score for a datacenter region.

        Args:
            region: Region identifier.
            pue: Power Usage Effectiveness (1.0 = perfect).
            renewable_percentage: Fraction of energy from renewables (0–100).
            carbon_intensity_gco2_per_kwh: Grid carbon intensity.

        Returns:
            Dict with efficiency_score (0–100), PUE_score, renewable_score,
            carbon_intensity_score, and certification_tier.
        """
        ...

    async def get_certification_status(self, region: str) -> dict[str, Any]:
        """Retrieve the current green certification tier for a region.

        Args:
            region: Region identifier.

        Returns:
            Dict with region, certification_tier, score, and criteria.
        """
        ...


class IOffsetIntegrator(Protocol):
    """Contract for carbon offset marketplace integration and portfolio management."""

    async def list_providers(self) -> list[dict[str, Any]]:
        """List all available carbon offset providers.

        Returns:
            List of provider dicts with provider_id, name, registry, and pricing.
        """
        ...

    async def purchase_offsets(
        self,
        tenant_id: str,
        provider_id: str,
        tonnes_co2: float,
        *,
        project_type: str | None = None,
    ) -> dict[str, Any]:
        """Purchase carbon offsets from a provider.

        Args:
            tenant_id: Tenant making the purchase.
            provider_id: Offset provider identifier.
            tonnes_co2: Quantity of offsets to purchase in tonnes CO2e.
            project_type: Optional filter for project type (forestry | renewable | etc).

        Returns:
            Dict with purchase_id, total_cost_usd, serial_numbers, and status.
        """
        ...

    async def get_portfolio_coverage(self, tenant_id: str) -> dict[str, Any]:
        """Calculate the tenant's offset portfolio vs total emissions.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Dict with total_purchased_tonnes, total_retired_tonnes,
            emissions_tonnes, coverage_pct, and neutrality_classification.
        """
        ...


class IInferenceOptimizer(Protocol):
    """Contract for latency-energy tradeoff optimization in inference serving."""

    async def configure_dynamic_batching(
        self,
        model_id: str,
        latency_tier: str,
        *,
        avg_input_tokens: int = 512,
        avg_output_tokens: int = 256,
        requests_per_second: float = 10.0,
        gpu_memory_gb: float = 40.0,
        current_batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Compute optimal dynamic batching parameters for a latency SLA tier.

        Args:
            model_id: Model to configure batching for.
            latency_tier: SLA tier — real_time | near_real_time |
                batch_interactive | background.
            avg_input_tokens: Average number of input tokens per request.
            avg_output_tokens: Average number of output tokens per request.
            requests_per_second: Observed or expected request arrival rate.
            gpu_memory_gb: Available GPU memory in gigabytes.
            current_batch_size: Existing batch size for comparison (optional).

        Returns:
            Dict with recommended_batch_size, max_queue_delay_ms,
            estimated_energy_per_request_mj, and latency_overhead_ms.
        """
        ...

    async def compute_pareto_frontier(
        self,
        model_id: str,
        *,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        avg_input_tokens: int = 512,
        avg_output_tokens: int = 256,
        base_latency_ms: float = 50.0,
    ) -> dict[str, Any]:
        """Compute the Pareto frontier of latency vs energy for a model workload.

        Args:
            model_id: Model to analyze.
            min_batch_size: Smallest batch size to evaluate.
            max_batch_size: Largest batch size to evaluate.
            avg_input_tokens: Average input token count per request.
            avg_output_tokens: Average output token count per request.
            base_latency_ms: Single-request baseline latency.

        Returns:
            Dict with all_points, pareto_frontier, knee_point, and recommendation.
        """
        ...

    async def compare_ab_experiment(
        self,
        model_id: str,
        experiment_name: str,
    ) -> dict[str, Any]:
        """Compare control vs treatment results for an A/B energy experiment.

        Args:
            model_id: Model the experiment belongs to.
            experiment_name: Name of the experiment to compare.

        Returns:
            Dict with control, treatment, energy_delta_mj, verdict, and explanation.
        """
        ...
