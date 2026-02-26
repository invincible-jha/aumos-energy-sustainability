"""SQLAlchemy repository implementations for the Energy Sustainability service.

Each repository implements the corresponding core interface using SQLAlchemy
async sessions. Raw SQL aggregations are used where ORM queries are insufficient.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.database import get_session
from aumos_common.observability import get_logger

from aumos_energy_sustainability.core.models import (
    CarbonRecord,
    EnergyProfile,
    OptimizationRecord,
    RoutingDecision,
    SustainabilityReport,
)

logger = get_logger(__name__)


class CarbonRecordRepository:
    """SQLAlchemy implementation of ICarbonRecordRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session.
        """
        self._session = session

    async def create(self, record: CarbonRecord) -> CarbonRecord:
        """Persist a new CarbonRecord.

        Args:
            record: The CarbonRecord to persist.

        Returns:
            Persisted CarbonRecord with populated id and timestamps.
        """
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        logger.debug("CarbonRecord created", record_id=str(record.id))
        return record

    async def get_by_id(
        self, record_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> CarbonRecord | None:
        """Retrieve a CarbonRecord by primary key and tenant.

        Args:
            record_id: Record primary key.
            tenant_id: Owning tenant UUID.

        Returns:
            CarbonRecord if found, None otherwise.
        """
        result = await self._session.execute(
            select(CarbonRecord).where(
                CarbonRecord.id == record_id,
                CarbonRecord.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

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
        query = select(CarbonRecord).where(CarbonRecord.tenant_id == tenant_id)

        if since:
            query = query.where(CarbonRecord.created_at >= since)
        if until:
            query = query.where(CarbonRecord.created_at < until)
        if region:
            query = query.where(CarbonRecord.region == region)
        if model_id:
            query = query.where(CarbonRecord.model_id == model_id)

        count_result = await self._session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total: int = count_result.scalar_one()

        offset = (page - 1) * page_size
        result = await self._session.execute(
            query.order_by(CarbonRecord.created_at.desc()).offset(offset).limit(page_size)
        )
        return list(result.scalars().all()), total

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
            Dict with aggregated metrics and breakdowns.
        """
        result = await self._session.execute(
            select(
                func.count(CarbonRecord.id).label("total_inferences"),
                func.coalesce(func.sum(CarbonRecord.energy_kwh), 0.0).label("total_energy_kwh"),
                func.coalesce(func.sum(CarbonRecord.carbon_gco2), 0.0).label("total_carbon_gco2"),
                func.coalesce(func.avg(CarbonRecord.renewable_percentage), 0.0).label(
                    "average_renewable_percentage"
                ),
            ).where(
                CarbonRecord.tenant_id == tenant_id,
                CarbonRecord.created_at >= period_start,
                CarbonRecord.created_at < period_end,
            )
        )
        row = result.one()

        # Per-model breakdown
        model_result = await self._session.execute(
            select(
                CarbonRecord.model_id,
                func.count(CarbonRecord.id).label("inferences"),
                func.coalesce(func.sum(CarbonRecord.energy_kwh), 0.0).label("energy_kwh"),
                func.coalesce(func.sum(CarbonRecord.carbon_gco2) / 1000.0, 0.0).label(
                    "total_carbon_kg"
                ),
            )
            .where(
                CarbonRecord.tenant_id == tenant_id,
                CarbonRecord.created_at >= period_start,
                CarbonRecord.created_at < period_end,
            )
            .group_by(CarbonRecord.model_id)
        )
        per_model: dict[str, Any] = {}
        for model_row in model_result.all():
            per_model[model_row.model_id] = {
                "inferences": model_row.inferences,
                "energy_kwh": float(model_row.energy_kwh),
                "total_carbon_kg": float(model_row.total_carbon_kg),
            }

        # Per-region breakdown
        region_result = await self._session.execute(
            select(
                CarbonRecord.region,
                func.count(CarbonRecord.id).label("inferences"),
                func.coalesce(func.sum(CarbonRecord.energy_kwh), 0.0).label("energy_kwh"),
                func.coalesce(func.sum(CarbonRecord.carbon_gco2) / 1000.0, 0.0).label(
                    "total_carbon_kg"
                ),
                func.coalesce(func.avg(CarbonRecord.renewable_percentage), 0.0).label(
                    "renewable_pct"
                ),
            )
            .where(
                CarbonRecord.tenant_id == tenant_id,
                CarbonRecord.created_at >= period_start,
                CarbonRecord.created_at < period_end,
            )
            .group_by(CarbonRecord.region)
        )
        per_region: dict[str, Any] = {}
        for region_row in region_result.all():
            per_region[region_row.region] = {
                "inferences": region_row.inferences,
                "energy_kwh": float(region_row.energy_kwh),
                "total_carbon_kg": float(region_row.total_carbon_kg),
                "renewable_pct": float(region_row.renewable_pct),
            }

        return {
            "total_inferences": row.total_inferences,
            "total_energy_kwh": float(row.total_energy_kwh),
            "total_carbon_gco2": float(row.total_carbon_gco2),
            "average_renewable_percentage": float(row.average_renewable_percentage),
            "per_model_breakdown": per_model,
            "per_region_breakdown": per_region,
        }


class EnergyProfileRepository:
    """SQLAlchemy implementation of IEnergyProfileRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session.
        """
        self._session = session

    async def upsert(self, profile: EnergyProfile) -> EnergyProfile:
        """Create or update an EnergyProfile for a region.

        Args:
            profile: Profile with region identifier.

        Returns:
            Persisted EnergyProfile.
        """
        existing = await self.get_by_region(profile.region, profile.tenant_id)
        if existing:
            existing.carbon_intensity_gco2_per_kwh = profile.carbon_intensity_gco2_per_kwh
            existing.renewable_percentage = profile.renewable_percentage
            existing.solar_percentage = profile.solar_percentage
            existing.wind_percentage = profile.wind_percentage
            existing.hydro_percentage = profile.hydro_percentage
            existing.nuclear_percentage = profile.nuclear_percentage
            existing.last_refreshed_at = profile.last_refreshed_at
            existing.source_metadata = profile.source_metadata
            existing.is_active = True
            await self._session.flush()
            await self._session.refresh(existing)
            return existing

        self._session.add(profile)
        await self._session.flush()
        await self._session.refresh(profile)
        return profile

    async def get_by_region(
        self, region: str, tenant_id: uuid.UUID
    ) -> EnergyProfile | None:
        """Retrieve an active EnergyProfile by region.

        Args:
            region: Region identifier.
            tenant_id: Owning tenant UUID.

        Returns:
            Active EnergyProfile if found, None otherwise.
        """
        result = await self._session.execute(
            select(EnergyProfile).where(
                EnergyProfile.region == region,
                EnergyProfile.tenant_id == tenant_id,
                EnergyProfile.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def list_active(self, tenant_id: uuid.UUID) -> list[EnergyProfile]:
        """List all active EnergyProfiles for a tenant.

        Args:
            tenant_id: Owning tenant UUID.

        Returns:
            List of active EnergyProfile instances.
        """
        result = await self._session.execute(
            select(EnergyProfile)
            .where(
                EnergyProfile.tenant_id == tenant_id,
                EnergyProfile.is_active.is_(True),
            )
            .order_by(EnergyProfile.renewable_percentage.desc())
        )
        return list(result.scalars().all())

    async def delete(self, region: str, tenant_id: uuid.UUID) -> bool:
        """Soft-delete an EnergyProfile.

        Args:
            region: Region identifier.
            tenant_id: Owning tenant UUID.

        Returns:
            True if profile found and deleted, False otherwise.
        """
        result = await self._session.execute(
            update(EnergyProfile)
            .where(
                EnergyProfile.region == region,
                EnergyProfile.tenant_id == tenant_id,
            )
            .values(is_active=False)
        )
        return result.rowcount > 0  # type: ignore[return-value]


class RoutingDecisionRepository:
    """SQLAlchemy implementation of IRoutingDecisionRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session.
        """
        self._session = session

    async def create(self, decision: RoutingDecision) -> RoutingDecision:
        """Persist a new RoutingDecision.

        Args:
            decision: The RoutingDecision to persist.

        Returns:
            Persisted RoutingDecision with populated id.
        """
        self._session.add(decision)
        await self._session.flush()
        await self._session.refresh(decision)
        return decision

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
        query = select(RoutingDecision).where(RoutingDecision.tenant_id == tenant_id)

        if since:
            query = query.where(RoutingDecision.created_at >= since)
        if workload_type:
            query = query.where(RoutingDecision.workload_type == workload_type)

        count_result = await self._session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total: int = count_result.scalar_one()

        offset = (page - 1) * page_size
        result = await self._session.execute(
            query.order_by(RoutingDecision.created_at.desc()).offset(offset).limit(page_size)
        )
        return list(result.scalars().all()), total


class SustainabilityReportRepository:
    """SQLAlchemy implementation of ISustainabilityReportRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session.
        """
        self._session = session

    async def create(self, report: SustainabilityReport) -> SustainabilityReport:
        """Persist a new SustainabilityReport stub.

        Args:
            report: The report stub to persist.

        Returns:
            Persisted SustainabilityReport.
        """
        self._session.add(report)
        await self._session.flush()
        await self._session.refresh(report)
        return report

    async def update(self, report: SustainabilityReport) -> SustainabilityReport:
        """Save updated report fields.

        Args:
            report: Report with updated fields.

        Returns:
            Updated SustainabilityReport.
        """
        await self._session.flush()
        await self._session.refresh(report)
        return report

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
        result = await self._session.execute(
            select(SustainabilityReport).where(
                SustainabilityReport.id == report_id,
                SustainabilityReport.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

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
            page_size: Reports per page.

        Returns:
            Tuple of (page of reports, total count).
        """
        query = select(SustainabilityReport).where(
            SustainabilityReport.tenant_id == tenant_id
        )
        if report_type:
            query = query.where(SustainabilityReport.report_type == report_type)

        count_result = await self._session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total: int = count_result.scalar_one()

        offset = (page - 1) * page_size
        result = await self._session.execute(
            query.order_by(SustainabilityReport.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        return list(result.scalars().all()), total


class OptimizationRepository:
    """SQLAlchemy implementation of IOptimizationRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialise with an async SQLAlchemy session.

        Args:
            session: Async database session.
        """
        self._session = session

    async def create(self, record: OptimizationRecord) -> OptimizationRecord:
        """Persist a new OptimizationRecord.

        Args:
            record: The record to persist.

        Returns:
            Persisted OptimizationRecord with populated id.
        """
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return record

    async def update(self, record: OptimizationRecord) -> OptimizationRecord:
        """Save updated recommendation fields.

        Args:
            record: Record with updated fields.

        Returns:
            Updated OptimizationRecord.
        """
        await self._session.flush()
        await self._session.refresh(record)
        return record

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
            category: Optional category filter.
            priority: Optional priority filter.

        Returns:
            List of active OptimizationRecord instances sorted by savings desc.
        """
        query = select(OptimizationRecord).where(
            OptimizationRecord.tenant_id == tenant_id,
            OptimizationRecord.status == "active",
        )
        if category:
            query = query.where(OptimizationRecord.category == category)
        if priority:
            query = query.where(OptimizationRecord.priority == priority)

        result = await self._session.execute(
            query.order_by(OptimizationRecord.projected_savings_kg_co2.desc())
        )
        return list(result.scalars().all())
