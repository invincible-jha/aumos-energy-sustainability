"""Energy router adapter for aumos-energy-sustainability.

Routes AI workloads to the lowest-carbon renewable grid region: carbon intensity
lookup per region, workload-to-region routing based on carbon score, time-of-day
optimization, migration cost estimation, energy source enforcement, and routing
decision logging.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

from aumos_energy_sustainability.core.interfaces import ICarbonAPIClient

logger = get_logger(__name__)

# Time-of-day solar peak windows (UTC hours) when solar-heavy regions are preferred
SOLAR_PEAK_UTC_HOURS: frozenset[int] = frozenset(range(10, 17))  # 10:00–17:00 UTC

# Time-of-day wind peak windows (UTC hours — typically off-peak evenings and night)
WIND_PEAK_UTC_HOURS: frozenset[int] = frozenset(range(0, 6)) | frozenset(range(20, 24))

# Renewable preference tiers — ordered from strictest to most permissive
RENEWABLE_TIERS: dict[str, float] = {
    "zero_carbon": 90.0,   # >= 90% renewable
    "high_renewable": 70.0,  # >= 70% renewable
    "moderate_renewable": 40.0,  # >= 40% renewable
    "any": 0.0,
}

# Estimated migration costs in USD per GB transferred between regions
INTER_REGION_TRANSFER_COST_USD_PER_GB: float = 0.09


class EnergyRouter:
    """Routes AI workloads to the cleanest available grid region.

    Evaluates regions by real-time carbon intensity, renewable percentage,
    and optionally applies time-of-day constraints for solar/wind peak windows.
    All routing decisions are logged for auditability.
    """

    def __init__(
        self,
        carbon_api: ICarbonAPIClient,
        renewable_preference_tier: str = "high_renewable",
        carbon_threshold_gco2_per_kwh: float = 150.0,
    ) -> None:
        """Initialise the energy router.

        Args:
            carbon_api: External carbon intensity data provider.
            renewable_preference_tier: Minimum renewable tier for preferred routing
                (zero_carbon | high_renewable | moderate_renewable | any).
            carbon_threshold_gco2_per_kwh: Max acceptable carbon intensity.
                Regions above this threshold are deprioritised.
        """
        self._carbon_api = carbon_api
        self._renewable_preference_tier = renewable_preference_tier
        self._carbon_threshold = carbon_threshold_gco2_per_kwh
        self._routing_log: list[dict[str, Any]] = []

    async def get_region_carbon_intensity(
        self,
        region: str,
    ) -> dict[str, Any]:
        """Fetch current carbon intensity data for a region.

        Args:
            region: AumOS region identifier (e.g., eu-north-1).

        Returns:
            Carbon data dict with carbon_intensity_gco2_per_kwh, renewable_percentage,
            and energy source breakdown.
        """
        data = await self._carbon_api.get_carbon_intensity(region)
        logger.debug(
            "Carbon intensity fetched",
            region=region,
            carbon_intensity=data.get("carbon_intensity_gco2_per_kwh"),
            renewable_percentage=data.get("renewable_percentage"),
        )
        return data

    async def score_regions(
        self,
        candidate_regions: list[str],
        renewable_weight: float = 0.7,
        latency_weight: float = 0.3,
        region_latencies_ms: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Score candidate regions by renewable percentage and latency.

        Fetches live carbon data for each region and computes a composite
        score. Regions above the carbon threshold receive a penalty.

        Args:
            candidate_regions: Region identifiers to evaluate.
            renewable_weight: Score weight for renewable percentage (0–1).
            latency_weight: Score weight for latency (0–1). renewable_weight + latency_weight should = 1.
            region_latencies_ms: Optional override latencies per region in ms.

        Returns:
            List of scored region dicts sorted by composite_score descending.
        """
        latencies = region_latencies_ms or {}
        scored: list[dict[str, Any]] = []

        for region in candidate_regions:
            carbon_data = await self.get_region_carbon_intensity(region)
            renewable_pct = carbon_data.get("renewable_percentage", 0.0)
            carbon_intensity = carbon_data.get("carbon_intensity_gco2_per_kwh", 999.0)
            latency_ms = latencies.get(region, 100)

            renewable_score = renewable_pct / 100.0
            max_latency = max(latencies.values()) if latencies else 200
            latency_score = 1.0 - (latency_ms / max(max_latency, 1))

            composite = renewable_weight * renewable_score + latency_weight * latency_score

            # Apply penalty for regions above carbon threshold
            if carbon_intensity > self._carbon_threshold:
                composite *= 0.7

            scored.append({
                "region": region,
                "carbon_intensity_gco2_per_kwh": carbon_intensity,
                "renewable_percentage": renewable_pct,
                "solar_percentage": carbon_data.get("solar_percentage", 0.0),
                "wind_percentage": carbon_data.get("wind_percentage", 0.0),
                "latency_ms": latency_ms,
                "renewable_score": round(renewable_score, 4),
                "latency_score": round(latency_score, 4),
                "composite_score": round(composite, 4),
                "above_carbon_threshold": carbon_intensity > self._carbon_threshold,
            })

        scored.sort(key=lambda s: s["composite_score"], reverse=True)
        return scored

    async def route_by_carbon_score(
        self,
        workload_id: str,
        candidate_regions: list[str],
        workload_type: str = "inference",
        renewable_weight: float = 0.7,
        latency_weight: float = 0.3,
        region_latencies_ms: dict[str, int] | None = None,
        enforce_renewable_tier: bool = True,
    ) -> dict[str, Any]:
        """Select the optimal region for a workload based on carbon score.

        Args:
            workload_id: External workload UUID string.
            candidate_regions: Regions to consider.
            workload_type: inference | training | batch_processing | fine_tuning.
            renewable_weight: Weight for renewable percentage in composite score.
            latency_weight: Weight for latency in composite score.
            region_latencies_ms: Optional per-region latency overrides.
            enforce_renewable_tier: When True, skip regions below renewable preference tier.

        Returns:
            Routing result dict with selected_region, score_breakdown, and rationale.

        Raises:
            ValueError: If no candidate regions are provided or none pass tier filter.
        """
        if not candidate_regions:
            raise ValueError("candidate_regions must not be empty")

        scored_regions = await self.score_regions(
            candidate_regions=candidate_regions,
            renewable_weight=renewable_weight,
            latency_weight=latency_weight,
            region_latencies_ms=region_latencies_ms,
        )

        eligible = scored_regions
        if enforce_renewable_tier:
            min_renewable = RENEWABLE_TIERS.get(self._renewable_preference_tier, 0.0)
            eligible = [r for r in scored_regions if r["renewable_percentage"] >= min_renewable]
            if not eligible:
                # Fall back to any region if tier filtering eliminates all
                eligible = scored_regions

        selected = eligible[0]
        worst = scored_regions[-1]
        carbon_saved_estimate = max(
            0.0,
            worst["carbon_intensity_gco2_per_kwh"] - selected["carbon_intensity_gco2_per_kwh"],
        )

        routing_result: dict[str, Any] = {
            "routing_id": str(uuid.uuid4()),
            "workload_id": workload_id,
            "workload_type": workload_type,
            "selected_region": selected["region"],
            "selected_score": selected["composite_score"],
            "renewable_percentage": selected["renewable_percentage"],
            "carbon_intensity": selected["carbon_intensity_gco2_per_kwh"],
            "carbon_saved_estimate_gco2_per_kwh": round(carbon_saved_estimate, 2),
            "score_breakdown": scored_regions,
            "renewable_tier_enforced": self._renewable_preference_tier if enforce_renewable_tier else None,
            "routed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._routing_log.append(routing_result)

        logger.info(
            "Workload routed by carbon score",
            workload_id=workload_id,
            selected_region=selected["region"],
            renewable_pct=selected["renewable_percentage"],
            carbon_intensity=selected["carbon_intensity_gco2_per_kwh"],
            composite_score=selected["composite_score"],
        )
        return routing_result

    async def apply_time_of_day_optimization(
        self,
        candidate_regions: list[str],
        workload_id: str,
        preferred_source: str = "auto",
        evaluation_time_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """Optimize region selection based on current solar/wind generation windows.

        Inspects the current UTC hour and filters regions where solar or wind
        production is at peak to maximise renewable utilisation.

        Args:
            candidate_regions: Regions to optimize across.
            workload_id: External workload identifier.
            preferred_source: auto | solar | wind — renewable source preference.
            evaluation_time_utc: Evaluation timestamp (defaults to now UTC).

        Returns:
            Time-optimized routing result with recommended_region and rationale.
        """
        now_utc = evaluation_time_utc or datetime.now(tz=timezone.utc)
        current_hour = now_utc.hour

        is_solar_peak = current_hour in SOLAR_PEAK_UTC_HOURS
        is_wind_peak = current_hour in WIND_PEAK_UTC_HOURS

        # Score all regions
        scored = await self.score_regions(candidate_regions)

        # Apply time-of-day preference weighting
        if preferred_source == "solar" or (preferred_source == "auto" and is_solar_peak):
            scored.sort(key=lambda r: r.get("solar_percentage", 0.0), reverse=True)
            active_source = "solar"
        elif preferred_source == "wind" or (preferred_source == "auto" and is_wind_peak):
            scored.sort(key=lambda r: r.get("wind_percentage", 0.0), reverse=True)
            active_source = "wind"
        else:
            active_source = "composite"

        recommended = scored[0] if scored else {}

        result: dict[str, Any] = {
            "workload_id": workload_id,
            "recommended_region": recommended.get("region"),
            "active_source_preference": active_source,
            "is_solar_peak": is_solar_peak,
            "is_wind_peak": is_wind_peak,
            "evaluation_hour_utc": current_hour,
            "region_scores": scored,
            "evaluated_at": now_utc.isoformat(),
        }
        logger.info(
            "Time-of-day energy optimization applied",
            workload_id=workload_id,
            recommended_region=recommended.get("region"),
            active_source=active_source,
            is_solar_peak=is_solar_peak,
            is_wind_peak=is_wind_peak,
        )
        return result

    async def estimate_migration_cost(
        self,
        from_region: str,
        to_region: str,
        data_volume_gb: float,
    ) -> dict[str, Any]:
        """Estimate the cost of migrating a workload between regions.

        Computes data transfer fees and the carbon-saving benefit of the
        migration to allow a cost-benefit assessment.

        Args:
            from_region: Current workload region.
            to_region: Target lower-carbon region.
            data_volume_gb: Volume of data to transfer in gigabytes.

        Returns:
            Migration cost estimate dict with transfer_cost_usd, carbon_benefit, and break_even_note.
        """
        transfer_cost_usd = round(data_volume_gb * INTER_REGION_TRANSFER_COST_USD_PER_GB, 4)

        from_data = await self.get_region_carbon_intensity(from_region)
        to_data = await self.get_region_carbon_intensity(to_region)

        from_intensity = from_data.get("carbon_intensity_gco2_per_kwh", 0.0)
        to_intensity = to_data.get("carbon_intensity_gco2_per_kwh", 0.0)
        intensity_delta = max(0.0, from_intensity - to_intensity)

        # Rough energy for data transfer: ~0.06 kWh per GB
        transfer_energy_kwh = data_volume_gb * 0.06
        transfer_carbon_gco2 = transfer_energy_kwh * to_intensity
        net_carbon_saving_per_kwh = intensity_delta

        logger.info(
            "Migration cost estimated",
            from_region=from_region,
            to_region=to_region,
            data_volume_gb=data_volume_gb,
            transfer_cost_usd=transfer_cost_usd,
            intensity_delta=intensity_delta,
        )

        return {
            "from_region": from_region,
            "to_region": to_region,
            "data_volume_gb": data_volume_gb,
            "transfer_cost_usd": transfer_cost_usd,
            "from_carbon_intensity": from_intensity,
            "to_carbon_intensity": to_intensity,
            "carbon_intensity_delta_gco2_per_kwh": round(intensity_delta, 2),
            "transfer_overhead_gco2": round(transfer_carbon_gco2, 4),
            "net_carbon_saving_per_kwh_inference": round(net_carbon_saving_per_kwh, 2),
            "is_migration_carbon_positive": intensity_delta > 0,
        }

    async def enforce_energy_source_preference(
        self,
        candidate_regions: list[str],
        required_sources: list[str],
        minimum_percentage: float = 20.0,
    ) -> list[str]:
        """Filter regions to those meeting energy source requirements.

        Args:
            candidate_regions: Regions to evaluate.
            required_sources: Energy source types required (solar, wind, hydro, nuclear).
            minimum_percentage: Minimum percentage from required sources.

        Returns:
            List of regions satisfying the energy source requirement, ordered by compliance.
        """
        source_field_map: dict[str, str] = {
            "solar": "solar_percentage",
            "wind": "wind_percentage",
            "hydro": "hydro_percentage",
            "nuclear": "nuclear_percentage",
            "renewable": "renewable_percentage",
        }

        qualifying_regions: list[str] = []
        for region in candidate_regions:
            data = await self.get_region_carbon_intensity(region)
            total_from_required = sum(
                data.get(source_field_map.get(src, ""), 0.0)
                for src in required_sources
                if src in source_field_map
            )
            if total_from_required >= minimum_percentage:
                qualifying_regions.append(region)

        logger.info(
            "Energy source preference filter applied",
            required_sources=required_sources,
            minimum_percentage=minimum_percentage,
            qualifying_count=len(qualifying_regions),
            total_candidates=len(candidate_regions),
        )
        return qualifying_regions

    async def get_routing_history(
        self,
        workload_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve recent routing decisions.

        Args:
            workload_type: Optional filter by workload type.
            limit: Maximum records to return (most recent first).

        Returns:
            List of routing log entries.
        """
        records = list(self._routing_log)
        if workload_type:
            records = [r for r in records if r.get("workload_type") == workload_type]
        return sorted(records, key=lambda r: r["routed_at"], reverse=True)[:limit]


__all__ = ["EnergyRouter"]
