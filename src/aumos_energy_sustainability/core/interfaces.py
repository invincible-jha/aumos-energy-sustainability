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
