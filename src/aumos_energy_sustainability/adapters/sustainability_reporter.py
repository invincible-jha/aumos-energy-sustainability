"""Sustainability reporter adapter for aumos-energy-sustainability.

ESG compliance report generation: GHG Protocol Scope 1/2/3 tracking, ESG
metric compilation, carbon footprint per workload, water usage estimation,
renewable energy percentage, trend analysis, and GRI/SASB format reports.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# GHG Protocol scope definitions for AI infrastructure
GHG_SCOPE_DEFINITIONS: dict[str, str] = {
    "scope_1": "Direct emissions from owned on-premise data centers (diesel generators, cooling).",
    "scope_2": "Indirect emissions from purchased electricity for AI compute.",
    "scope_3": "Value chain emissions: hardware manufacturing, employee travel, cloud usage.",
}

# Water usage intensity by data center cooling type (liters per kWh)
COOLING_WATER_INTENSITY_L_PER_KWH: dict[str, float] = {
    "air_cooled": 0.0,
    "evaporative": 1.8,
    "liquid_immersion": 0.5,
    "hybrid": 0.9,
    "unknown": 1.0,  # conservative default
}

# GRI standards mapping for AI ESG disclosure
GRI_STANDARDS_MAP: dict[str, str] = {
    "energy_consumption": "GRI 302-1",
    "energy_intensity": "GRI 302-3",
    "ghg_scope_1": "GRI 305-1",
    "ghg_scope_2": "GRI 305-2",
    "ghg_scope_3": "GRI 305-3",
    "water_withdrawal": "GRI 303-3",
    "renewable_energy": "GRI 302-1 (b)",
}

# SASB standards mapping (Technology & Communications)
SASB_STANDARDS_MAP: dict[str, str] = {
    "energy_consumption": "TC-SI-130a.1",
    "renewable_energy_pct": "TC-SI-130a.1",
    "ghg_emissions": "TC-SI-110a.1",
    "data_privacy": "TC-SI-220a.1",
}


class SustainabilityReporter:
    """Generates ESG and GHG compliance reports for AI workloads.

    Aggregates Scope 1/2/3 emissions, water usage, renewable energy percentages,
    and sustainability trends, and formats reports to GRI and SASB standards.
    """

    def __init__(self) -> None:
        """Initialise the sustainability reporter with empty report and emission stores."""
        self._scope_1_records: list[dict[str, Any]] = []
        self._scope_2_records: list[dict[str, Any]] = []
        self._scope_3_records: list[dict[str, Any]] = []
        self._generated_reports: list[dict[str, Any]] = []

    async def track_scope_1_emissions(
        self,
        tenant_id: str,
        source: str,
        co2_kg: float,
        measurement_date: datetime,
        notes: str = "",
    ) -> dict[str, Any]:
        """Record a Scope 1 (direct) GHG emission event.

        Scope 1 covers direct emissions from owned infrastructure: on-site diesel
        generators, direct fuel combustion in owned/leased data centers.

        Args:
            tenant_id: Owning tenant UUID string.
            source: Emission source description (e.g., diesel_generator, natural_gas).
            co2_kg: CO2-equivalent emissions in kilograms.
            measurement_date: When the emission was measured.
            notes: Optional contextual notes.

        Returns:
            Scope 1 emission record dict.
        """
        record: dict[str, Any] = {
            "record_id": str(uuid.uuid4()),
            "scope": "scope_1",
            "tenant_id": tenant_id,
            "source": source,
            "co2_kg": co2_kg,
            "measurement_date": measurement_date.isoformat(),
            "notes": notes,
            "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._scope_1_records.append(record)
        logger.info(
            "Scope 1 emission recorded",
            tenant_id=tenant_id,
            source=source,
            co2_kg=co2_kg,
        )
        return record

    async def track_scope_2_emissions(
        self,
        tenant_id: str,
        region: str,
        energy_kwh: float,
        carbon_intensity_gco2_per_kwh: float,
        renewable_percentage: float,
        measurement_date: datetime,
        market_based: bool = True,
    ) -> dict[str, Any]:
        """Record a Scope 2 (purchased electricity) GHG emission event.

        Scope 2 covers indirect emissions from the electricity consumed for AI compute.
        Supports both market-based (using renewable energy certificates) and
        location-based accounting methods.

        Args:
            tenant_id: Owning tenant UUID string.
            region: Cloud/datacenter region.
            energy_kwh: Energy consumed in kWh.
            carbon_intensity_gco2_per_kwh: Grid carbon intensity.
            renewable_percentage: Renewable energy fraction (0–100).
            measurement_date: When the energy was consumed.
            market_based: If True, applies REC adjustments to reported intensity.

        Returns:
            Scope 2 emission record dict with location_based and market_based figures.
        """
        location_co2_kg = energy_kwh * carbon_intensity_gco2_per_kwh / 1000.0
        renewable_fraction = renewable_percentage / 100.0

        # Market-based: subtract renewable portion from intensity
        if market_based:
            adjusted_intensity = carbon_intensity_gco2_per_kwh * (1.0 - renewable_fraction)
            market_co2_kg = energy_kwh * adjusted_intensity / 1000.0
        else:
            market_co2_kg = location_co2_kg

        record: dict[str, Any] = {
            "record_id": str(uuid.uuid4()),
            "scope": "scope_2",
            "tenant_id": tenant_id,
            "region": region,
            "energy_kwh": energy_kwh,
            "carbon_intensity_gco2_per_kwh": carbon_intensity_gco2_per_kwh,
            "renewable_percentage": renewable_percentage,
            "location_based_co2_kg": round(location_co2_kg, 4),
            "market_based_co2_kg": round(market_co2_kg, 4),
            "reported_co2_kg": round(market_co2_kg if market_based else location_co2_kg, 4),
            "accounting_method": "market_based" if market_based else "location_based",
            "measurement_date": measurement_date.isoformat(),
            "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._scope_2_records.append(record)
        logger.info(
            "Scope 2 emission recorded",
            tenant_id=tenant_id,
            region=region,
            energy_kwh=energy_kwh,
            location_co2_kg=location_co2_kg,
            market_co2_kg=market_co2_kg,
        )
        return record

    async def track_scope_3_emissions(
        self,
        tenant_id: str,
        category: str,
        co2_kg: float,
        description: str,
        measurement_date: datetime,
    ) -> dict[str, Any]:
        """Record a Scope 3 (value chain) GHG emission event.

        Scope 3 categories relevant to AI infrastructure include hardware manufacturing
        (Category 1: purchased goods), cloud provider upstream emissions (Category 1),
        business travel (Category 6), and employee commuting (Category 7).

        Args:
            tenant_id: Owning tenant UUID string.
            category: Scope 3 category name (e.g., hardware_manufacturing, cloud_usage).
            co2_kg: CO2-equivalent emissions in kilograms.
            description: Description of the emission source.
            measurement_date: When the emission was measured or estimated.

        Returns:
            Scope 3 emission record dict.
        """
        record: dict[str, Any] = {
            "record_id": str(uuid.uuid4()),
            "scope": "scope_3",
            "tenant_id": tenant_id,
            "category": category,
            "co2_kg": co2_kg,
            "description": description,
            "measurement_date": measurement_date.isoformat(),
            "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._scope_3_records.append(record)
        logger.info(
            "Scope 3 emission recorded",
            tenant_id=tenant_id,
            category=category,
            co2_kg=co2_kg,
        )
        return record

    async def compile_esg_metrics(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> dict[str, Any]:
        """Compile all ESG metrics for a tenant and time period.

        Args:
            tenant_id: Owning tenant UUID string.
            period_start: Start of the reporting period.
            period_end: End of the reporting period.

        Returns:
            Compiled ESG metrics dict including scope 1/2/3 totals, renewable %,
            intensity metrics, and GRI/SASB standard references.
        """
        start_iso = period_start.isoformat()
        end_iso = period_end.isoformat()

        def in_period(record: dict[str, Any]) -> bool:
            date = record.get("measurement_date", "")
            return start_iso <= date < end_iso and record["tenant_id"] == tenant_id

        scope1_records = [r for r in self._scope_1_records if in_period(r)]
        scope2_records = [r for r in self._scope_2_records if in_period(r)]
        scope3_records = [r for r in self._scope_3_records if in_period(r)]

        scope1_total_kg = sum(r["co2_kg"] for r in scope1_records)
        scope2_total_kg = sum(r["reported_co2_kg"] for r in scope2_records)
        scope3_total_kg = sum(r["co2_kg"] for r in scope3_records)
        total_co2_kg = scope1_total_kg + scope2_total_kg + scope3_total_kg

        total_energy_kwh = sum(r["energy_kwh"] for r in scope2_records)
        avg_renewable_pct = (
            sum(r["renewable_percentage"] * r["energy_kwh"] for r in scope2_records) / max(total_energy_kwh, 0.01)
            if scope2_records else 0.0
        )

        scope3_by_category: dict[str, float] = {}
        for r in scope3_records:
            cat = r["category"]
            scope3_by_category[cat] = scope3_by_category.get(cat, 0.0) + r["co2_kg"]

        return {
            "tenant_id": tenant_id,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "scope_1_total_co2_kg": round(scope1_total_kg, 3),
            "scope_2_total_co2_kg": round(scope2_total_kg, 3),
            "scope_3_total_co2_kg": round(scope3_total_kg, 3),
            "total_co2_kg": round(total_co2_kg, 3),
            "total_energy_kwh": round(total_energy_kwh, 4),
            "average_renewable_percentage": round(avg_renewable_pct, 2),
            "scope_3_by_category": scope3_by_category,
            "scope_1_record_count": len(scope1_records),
            "scope_2_record_count": len(scope2_records),
            "scope_3_record_count": len(scope3_records),
            "gri_references": GRI_STANDARDS_MAP,
            "sasb_references": SASB_STANDARDS_MAP,
        }

    async def estimate_water_usage(
        self,
        energy_kwh: float,
        cooling_type: str = "unknown",
        region: str | None = None,
    ) -> dict[str, Any]:
        """Estimate water consumption from AI compute energy usage.

        Uses Water Usage Effectiveness (WUE) estimates by cooling technology type.
        WUE = total water consumed per kWh of IT energy.

        Args:
            energy_kwh: Energy consumed by AI compute in kWh.
            cooling_type: Datacenter cooling method (air_cooled, evaporative, etc.).
            region: Optional region for climate-adjusted estimates.

        Returns:
            Water usage estimate dict with total_liters and WUE ratio.
        """
        wue = COOLING_WATER_INTENSITY_L_PER_KWH.get(cooling_type, COOLING_WATER_INTENSITY_L_PER_KWH["unknown"])

        # Crude regional climate adjustment (warmer climates need more cooling water)
        if region:
            if any(r in region for r in ("us-east", "ap-south", "sa-east", "af-south")):
                wue *= 1.2
            elif any(r in region for r in ("eu-north", "ca-central")):
                wue *= 0.8

        total_water_liters = energy_kwh * wue
        total_water_gallons = total_water_liters * 0.264172

        return {
            "energy_kwh": energy_kwh,
            "cooling_type": cooling_type,
            "region": region,
            "water_usage_effectiveness_l_per_kwh": round(wue, 3),
            "total_water_liters": round(total_water_liters, 2),
            "total_water_gallons": round(total_water_gallons, 2),
            "gri_reference": "GRI 303-3 (Water Withdrawal)",
        }

    async def analyze_sustainability_trends(
        self,
        tenant_id: str,
        periods: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyse sustainability metric trends across multiple periods.

        Args:
            tenant_id: Owning tenant UUID string.
            periods: List of period metric dicts, each containing:
                period_label, total_co2_kg, energy_kwh, renewable_percentage.

        Returns:
            Trend analysis dict with directional indicators and period-over-period deltas.
        """
        if len(periods) < 2:
            return {
                "tenant_id": tenant_id,
                "message": "At least 2 periods are required for trend analysis.",
                "periods": periods,
            }

        deltas: list[dict[str, Any]] = []
        for i in range(1, len(periods)):
            prev = periods[i - 1]
            curr = periods[i]

            co2_delta_pct = (
                (curr["total_co2_kg"] - prev["total_co2_kg"]) / max(prev["total_co2_kg"], 0.01) * 100
            )
            energy_delta_pct = (
                (curr["energy_kwh"] - prev["energy_kwh"]) / max(prev["energy_kwh"], 0.01) * 100
            )
            renewable_delta_ppt = curr["renewable_percentage"] - prev["renewable_percentage"]

            deltas.append({
                "from_period": prev["period_label"],
                "to_period": curr["period_label"],
                "co2_delta_pct": round(co2_delta_pct, 2),
                "energy_delta_pct": round(energy_delta_pct, 2),
                "renewable_delta_ppt": round(renewable_delta_ppt, 2),
                "co2_improving": co2_delta_pct < 0,
                "renewable_improving": renewable_delta_ppt > 0,
            })

        overall_co2_trend = "declining" if all(d["co2_improving"] for d in deltas) else "mixed" if any(d["co2_improving"] for d in deltas) else "increasing"
        overall_renewable_trend = "increasing" if all(d["renewable_improving"] for d in deltas) else "mixed" if any(d["renewable_improving"] for d in deltas) else "declining"

        return {
            "tenant_id": tenant_id,
            "period_count": len(periods),
            "co2_trend": overall_co2_trend,
            "renewable_trend": overall_renewable_trend,
            "period_deltas": deltas,
            "summary": (
                f"CO2 emissions are {overall_co2_trend}. "
                f"Renewable energy usage is {overall_renewable_trend}."
            ),
        }

    async def generate_gri_report(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        organization_name: str,
    ) -> dict[str, Any]:
        """Generate a GRI Standards-aligned ESG disclosure report.

        Args:
            tenant_id: Owning tenant UUID string.
            period_start: Reporting period start.
            period_end: Reporting period end.
            organization_name: Organization name for the report header.

        Returns:
            GRI report dict structured according to GRI 302, 303, and 305 topics.
        """
        metrics = await self.compile_esg_metrics(tenant_id, period_start, period_end)
        water = await self.estimate_water_usage(metrics["total_energy_kwh"])
        report_id = str(uuid.uuid4())

        report: dict[str, Any] = {
            "report_id": report_id,
            "report_format": "GRI",
            "organization_name": organization_name,
            "tenant_id": tenant_id,
            "reporting_period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
            },
            "gri_302_energy": {
                "standard": "GRI 302",
                "topic": "Energy",
                "302_1_energy_consumption_kwh": metrics["total_energy_kwh"],
                "302_1_b_renewable_energy_kwh": round(
                    metrics["total_energy_kwh"] * metrics["average_renewable_percentage"] / 100, 4
                ),
                "302_3_energy_intensity_kwh_per_inference": None,  # requires inference count
            },
            "gri_303_water": {
                "standard": "GRI 303",
                "topic": "Water and Effluents",
                "303_3_water_withdrawal_liters": water["total_water_liters"],
            },
            "gri_305_emissions": {
                "standard": "GRI 305",
                "topic": "Emissions",
                "305_1_scope_1_co2_kg": metrics["scope_1_total_co2_kg"],
                "305_2_scope_2_co2_kg": metrics["scope_2_total_co2_kg"],
                "305_3_scope_3_co2_kg": metrics["scope_3_total_co2_kg"],
                "305_total_co2_kg": metrics["total_co2_kg"],
                "renewable_energy_percentage": metrics["average_renewable_percentage"],
            },
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._generated_reports.append(report)

        logger.info(
            "GRI report generated",
            report_id=report_id,
            tenant_id=tenant_id,
            total_co2_kg=metrics["total_co2_kg"],
        )
        return report

    async def generate_sasb_report(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        organization_name: str,
    ) -> dict[str, Any]:
        """Generate a SASB Technology & Communications sector-aligned ESG report.

        Args:
            tenant_id: Owning tenant UUID string.
            period_start: Reporting period start.
            period_end: Reporting period end.
            organization_name: Organization name for the report header.

        Returns:
            SASB report dict structured according to TC-SI metric categories.
        """
        metrics = await self.compile_esg_metrics(tenant_id, period_start, period_end)
        report_id = str(uuid.uuid4())

        report: dict[str, Any] = {
            "report_id": report_id,
            "report_format": "SASB",
            "sector": "Technology & Communications — Software & IT Services",
            "organization_name": organization_name,
            "tenant_id": tenant_id,
            "reporting_period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
            },
            "tc_si_130a_1_energy": {
                "standard": "TC-SI-130a.1",
                "description": "Total energy consumed, % grid electricity, % renewable",
                "total_energy_mwh": round(metrics["total_energy_kwh"] / 1000, 4),
                "grid_electricity_pct": 100.0,
                "renewable_energy_pct": metrics["average_renewable_percentage"],
            },
            "tc_si_110a_1_ghg_emissions": {
                "standard": "TC-SI-110a.1",
                "description": "Gross global Scope 1 emissions, % covered by emissions-limiting regulations",
                "scope_1_co2_mt": round(metrics["scope_1_total_co2_kg"] / 1000, 6),
                "scope_2_co2_mt": round(metrics["scope_2_total_co2_kg"] / 1000, 6),
                "scope_3_co2_mt": round(metrics["scope_3_total_co2_kg"] / 1000, 6),
            },
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._generated_reports.append(report)

        logger.info(
            "SASB report generated",
            report_id=report_id,
            tenant_id=tenant_id,
            total_co2_kg=metrics["total_co2_kg"],
        )
        return report


__all__ = ["SustainabilityReporter"]
