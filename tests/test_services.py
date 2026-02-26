"""Unit tests for Energy Sustainability core services.

Tests use mocked repositories and adapters — no database required.
"""

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aumos_energy_sustainability.core.models import (
    CarbonRecord,
    EnergyProfile,
    OptimizationRecord,
    RoutingDecision,
    SustainabilityReport,
)
from aumos_energy_sustainability.core.services import (
    CarbonTrackerService,
    EnergyRouterService,
    OptimizationAdvisorService,
    SustainabilityReportService,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tenant_id() -> uuid.UUID:
    """Shared tenant UUID for test isolation."""
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture()
def mock_event_publisher() -> AsyncMock:
    """Mock IEventPublisher — captures published events."""
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    return publisher


def _make_carbon_record(tenant_id: uuid.UUID, **overrides: Any) -> CarbonRecord:
    """Build a CarbonRecord with sensible defaults.

    Args:
        tenant_id: Tenant UUID.
        **overrides: Field overrides.

    Returns:
        CarbonRecord with populated defaults.
    """
    record = CarbonRecord(
        tenant_id=tenant_id,
        inference_id=uuid.uuid4(),
        model_id=overrides.get("model_id", "gpt-4"),
        region=overrides.get("region", "us-east-1"),
        energy_kwh=overrides.get("energy_kwh", 0.001),
        carbon_intensity_gco2_per_kwh=overrides.get("carbon_intensity_gco2_per_kwh", 380.0),
        carbon_gco2=overrides.get("carbon_gco2", 0.38),
        renewable_percentage=overrides.get("renewable_percentage", 22.0),
    )
    record.id = uuid.uuid4()
    record.created_at = datetime.now(tz=timezone.utc)
    return record


def _make_energy_profile(
    tenant_id: uuid.UUID,
    region: str = "eu-north-1",
    carbon_intensity: float = 15.0,
    renewable_pct: float = 97.0,
    latency_ms: int = 80,
) -> EnergyProfile:
    """Build an EnergyProfile with sensible defaults.

    Args:
        tenant_id: Tenant UUID.
        region: Region identifier.
        carbon_intensity: gCO2/kWh.
        renewable_pct: Renewable percentage.
        latency_ms: Estimated latency.

    Returns:
        EnergyProfile instance.
    """
    profile = EnergyProfile(
        tenant_id=tenant_id,
        region=region,
        display_name=region,
        carbon_intensity_gco2_per_kwh=carbon_intensity,
        renewable_percentage=renewable_pct,
        estimated_latency_ms=latency_ms,
        is_active=True,
        source_metadata={},
    )
    profile.id = uuid.uuid4()
    return profile


# ---------------------------------------------------------------------------
# CarbonTrackerService tests
# ---------------------------------------------------------------------------


class TestCarbonTrackerService:
    """Tests for CarbonTrackerService."""

    @pytest.fixture()
    def carbon_repo(self) -> AsyncMock:
        """Mock ICarbonRecordRepository."""
        repo = AsyncMock()
        return repo

    @pytest.fixture()
    def service(
        self, carbon_repo: AsyncMock, mock_event_publisher: AsyncMock
    ) -> CarbonTrackerService:
        """CarbonTrackerService with mocked dependencies."""
        return CarbonTrackerService(
            carbon_repo=carbon_repo,
            event_publisher=mock_event_publisher,
        )

    async def test_track_inference_creates_record(
        self,
        service: CarbonTrackerService,
        carbon_repo: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """track_inference should persist a record with computed carbon_gco2."""
        inference_id = uuid.uuid4()
        expected_record = _make_carbon_record(tenant_id, inference_id=inference_id)
        carbon_repo.create = AsyncMock(return_value=expected_record)

        result = await service.track_inference(
            tenant_id=tenant_id,
            inference_id=inference_id,
            model_id="gpt-4",
            region="us-east-1",
            energy_kwh=0.002,
            carbon_intensity_gco2_per_kwh=380.0,
            renewable_percentage=22.0,
        )

        carbon_repo.create.assert_called_once()
        created_record = carbon_repo.create.call_args[0][0]
        assert created_record.carbon_gco2 == pytest.approx(0.002 * 380.0)
        assert result.id == expected_record.id

    async def test_track_inference_rejects_negative_energy(
        self,
        service: CarbonTrackerService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Negative energy_kwh should raise ValueError."""
        with pytest.raises(ValueError, match="energy_kwh must be non-negative"):
            await service.track_inference(
                tenant_id=tenant_id,
                inference_id=uuid.uuid4(),
                model_id="gpt-4",
                region="us-east-1",
                energy_kwh=-0.001,
                carbon_intensity_gco2_per_kwh=380.0,
            )

    async def test_track_inference_publishes_event(
        self,
        service: CarbonTrackerService,
        carbon_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """A carbon.tracked Kafka event should be published on each tracked inference."""
        record = _make_carbon_record(tenant_id)
        carbon_repo.create = AsyncMock(return_value=record)

        await service.track_inference(
            tenant_id=tenant_id,
            inference_id=uuid.uuid4(),
            model_id="llama-3",
            region="eu-north-1",
            energy_kwh=0.0005,
            carbon_intensity_gco2_per_kwh=15.0,
        )

        mock_event_publisher.publish.assert_called_once()
        topic, payload = mock_event_publisher.publish.call_args[0]
        assert topic == "aumos.energy.carbon.tracked"
        assert payload["region"] == "eu-north-1"

    async def test_track_inference_zero_energy_is_valid(
        self,
        service: CarbonTrackerService,
        carbon_repo: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Zero energy_kwh is valid (e.g. cached inference)."""
        record = _make_carbon_record(tenant_id, energy_kwh=0.0, carbon_gco2=0.0)
        carbon_repo.create = AsyncMock(return_value=record)

        result = await service.track_inference(
            tenant_id=tenant_id,
            inference_id=uuid.uuid4(),
            model_id="cached-model",
            region="us-west-2",
            energy_kwh=0.0,
            carbon_intensity_gco2_per_kwh=120.0,
        )

        created = carbon_repo.create.call_args[0][0]
        assert created.carbon_gco2 == 0.0


# ---------------------------------------------------------------------------
# EnergyRouterService tests
# ---------------------------------------------------------------------------


class TestEnergyRouterService:
    """Tests for EnergyRouterService."""

    @pytest.fixture()
    def profile_repo(self, tenant_id: uuid.UUID) -> AsyncMock:
        """Mock IEnergyProfileRepository with two profiles."""
        repo = AsyncMock()
        high_carbon = _make_energy_profile(
            tenant_id, region="us-east-1", carbon_intensity=380.0, renewable_pct=22.0, latency_ms=20
        )
        low_carbon = _make_energy_profile(
            tenant_id, region="eu-north-1", carbon_intensity=15.0, renewable_pct=97.0, latency_ms=80
        )
        repo.get_by_region = AsyncMock(
            side_effect=lambda region, tid: {
                "us-east-1": high_carbon,
                "eu-north-1": low_carbon,
            }.get(region)
        )
        repo.list_active = AsyncMock(return_value=[high_carbon, low_carbon])
        return repo

    @pytest.fixture()
    def routing_repo(self) -> AsyncMock:
        """Mock IRoutingDecisionRepository."""
        repo = AsyncMock()

        async def _create(decision: RoutingDecision) -> RoutingDecision:
            decision.id = uuid.uuid4()
            decision.created_at = datetime.now(tz=timezone.utc)
            return decision

        repo.create = AsyncMock(side_effect=_create)
        return repo

    @pytest.fixture()
    def carbon_api(self) -> AsyncMock:
        """Mock ICarbonAPIClient."""
        return AsyncMock()

    @pytest.fixture()
    def service(
        self,
        profile_repo: AsyncMock,
        routing_repo: AsyncMock,
        carbon_api: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> EnergyRouterService:
        """EnergyRouterService with mocked dependencies."""
        return EnergyRouterService(
            profile_repo=profile_repo,
            routing_repo=routing_repo,
            carbon_api=carbon_api,
            event_publisher=mock_event_publisher,
            renewable_weight=0.7,
            latency_weight=0.3,
        )

    async def test_route_selects_lowest_carbon_region(
        self,
        service: EnergyRouterService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Router should prefer eu-north-1 (97% renewable) over us-east-1 (22%)."""
        decision = await service.route_workload(
            tenant_id=tenant_id,
            workload_id=uuid.uuid4(),
            workload_type="inference",
            candidate_regions=["us-east-1", "eu-north-1"],
        )

        assert decision.selected_region == "eu-north-1"
        assert decision.composite_score > 0.5

    async def test_route_raises_for_invalid_workload_type(
        self,
        service: EnergyRouterService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Invalid workload_type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid workload_type"):
            await service.route_workload(
                tenant_id=tenant_id,
                workload_id=uuid.uuid4(),
                workload_type="invalid_type",
                candidate_regions=["us-east-1"],
            )

    async def test_route_raises_for_empty_candidates(
        self,
        service: EnergyRouterService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Empty candidate_regions should raise ValueError."""
        with pytest.raises(ValueError, match="candidate_regions must not be empty"):
            await service.route_workload(
                tenant_id=tenant_id,
                workload_id=uuid.uuid4(),
                workload_type="inference",
                candidate_regions=[],
            )

    async def test_route_respects_override(
        self,
        service: EnergyRouterService,
        tenant_id: uuid.UUID,
    ) -> None:
        """override_region should bypass scoring and select the specified region."""
        decision = await service.route_workload(
            tenant_id=tenant_id,
            workload_id=uuid.uuid4(),
            workload_type="batch_processing",
            candidate_regions=["us-east-1", "eu-north-1"],
            override_region="us-east-1",
            override_reason="data residency requirement",
        )

        assert decision.selected_region == "us-east-1"
        assert decision.override_reason == "data residency requirement"

    async def test_route_publishes_event(
        self,
        service: EnergyRouterService,
        mock_event_publisher: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """A route.decided event should be published for every routing decision."""
        await service.route_workload(
            tenant_id=tenant_id,
            workload_id=uuid.uuid4(),
            workload_type="inference",
            candidate_regions=["us-east-1", "eu-north-1"],
        )

        mock_event_publisher.publish.assert_called_once()
        topic, payload = mock_event_publisher.publish.call_args[0]
        assert topic == "aumos.energy.route.decided"
        assert payload["selected_region"] == "eu-north-1"


# ---------------------------------------------------------------------------
# SustainabilityReportService tests
# ---------------------------------------------------------------------------


class TestSustainabilityReportService:
    """Tests for SustainabilityReportService."""

    @pytest.fixture()
    def report_repo(self) -> AsyncMock:
        """Mock ISustainabilityReportRepository."""
        repo = AsyncMock()

        async def _create(report: SustainabilityReport) -> SustainabilityReport:
            report.id = uuid.uuid4()
            report.created_at = datetime.now(tz=timezone.utc)
            return report

        async def _update(report: SustainabilityReport) -> SustainabilityReport:
            return report

        repo.create = AsyncMock(side_effect=_create)
        repo.update = AsyncMock(side_effect=_update)
        return repo

    @pytest.fixture()
    def carbon_repo(self) -> AsyncMock:
        """Mock ICarbonRecordRepository with aggregate data."""
        repo = AsyncMock()
        repo.aggregate_by_period = AsyncMock(
            return_value={
                "total_inferences": 10000,
                "total_energy_kwh": 10.0,
                "total_carbon_gco2": 1500.0,
                "average_renewable_percentage": 45.0,
                "per_model_breakdown": {
                    "gpt-4": {"inferences": 8000, "energy_kwh": 8.0, "total_carbon_kg": 1.2}
                },
                "per_region_breakdown": {
                    "eu-north-1": {
                        "inferences": 10000,
                        "energy_kwh": 10.0,
                        "total_carbon_kg": 1.5,
                        "renewable_pct": 45.0,
                    }
                },
            }
        )
        return repo

    @pytest.fixture()
    def routing_repo(self) -> AsyncMock:
        """Mock IRoutingDecisionRepository."""
        repo = AsyncMock()
        decisions = []
        for i in range(8):
            d = MagicMock()
            d.renewable_score = 0.7  # all above 0.5 threshold
            d.carbon_saved_gco2 = 50.0
            decisions.append(d)
        for i in range(2):
            d = MagicMock()
            d.renewable_score = 0.3  # below threshold
            d.carbon_saved_gco2 = 0.0
            decisions.append(d)
        repo.list_by_tenant = AsyncMock(return_value=(decisions, 10))
        return repo

    @pytest.fixture()
    def service(
        self,
        report_repo: AsyncMock,
        carbon_repo: AsyncMock,
        routing_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
    ) -> SustainabilityReportService:
        """SustainabilityReportService with mocked dependencies."""
        return SustainabilityReportService(
            report_repo=report_repo,
            carbon_repo=carbon_repo,
            routing_repo=routing_repo,
            event_publisher=mock_event_publisher,
        )

    async def test_generate_report_creates_ready_report(
        self,
        service: SustainabilityReportService,
        tenant_id: uuid.UUID,
    ) -> None:
        """generate_report should produce a report with status=ready and an ESG score."""
        report = await service.generate_report(
            tenant_id=tenant_id,
            title="Q1 2026 Sustainability Report",
            report_type="quarterly",
            period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )

        assert report.status == "ready"
        assert report.total_inferences == 10000
        assert report.total_carbon_kg_co2 == pytest.approx(1.5)
        assert report.esg_score is not None
        assert 0 <= report.esg_score <= 100

    async def test_generate_report_raises_for_invalid_type(
        self,
        service: SustainabilityReportService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Invalid report_type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid report_type"):
            await service.generate_report(
                tenant_id=tenant_id,
                title="Bad Report",
                report_type="weekly",
                period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                period_end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            )

    async def test_generate_report_raises_when_end_before_start(
        self,
        service: SustainabilityReportService,
        tenant_id: uuid.UUID,
    ) -> None:
        """period_end <= period_start should raise ValueError."""
        with pytest.raises(ValueError, match="period_end must be after period_start"):
            await service.generate_report(
                tenant_id=tenant_id,
                title="Bad Period",
                report_type="monthly",
                period_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                period_end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            )

    async def test_generate_report_computes_routing_opt_rate(
        self,
        service: SustainabilityReportService,
        tenant_id: uuid.UUID,
    ) -> None:
        """routing_optimisation_rate should be 0.8 (8 of 10 decisions above threshold)."""
        report = await service.generate_report(
            tenant_id=tenant_id,
            title="Rate Test",
            report_type="monthly",
            period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )

        assert report.routing_optimisation_rate == pytest.approx(0.8)
