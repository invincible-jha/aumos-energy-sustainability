"""Carbon offset integrator adapter for aumos-energy-sustainability.

Carbon offset marketplace integration: provider catalog, purchase tracking,
retirement certificate management, offset-to-emission matching, portfolio-level
coverage, verification status tracking, and offset report generation.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Supported offset provider catalog
OFFSET_PROVIDER_CATALOG: list[dict[str, Any]] = [
    {
        "provider_id": "gold-standard",
        "name": "Gold Standard Foundation",
        "standard": "Gold Standard VER",
        "project_types": ["renewable_energy", "cookstoves", "reforestation"],
        "price_usd_per_tonne_co2": 15.0,
        "verification_body": "SCS Global Services",
        "registry_url": "https://registry.goldstandard.org",
    },
    {
        "provider_id": "vcs-verra",
        "name": "Verra VCS",
        "standard": "Verified Carbon Standard",
        "project_types": ["afforestation", "forest_management", "blue_carbon", "renewable_energy"],
        "price_usd_per_tonne_co2": 12.0,
        "verification_body": "Bureau Veritas",
        "registry_url": "https://registry.verra.org",
    },
    {
        "provider_id": "american-carbon-registry",
        "name": "American Carbon Registry",
        "standard": "ACR Standard",
        "project_types": ["forest_carbon", "methane_destruction", "soil_carbon"],
        "price_usd_per_tonne_co2": 18.0,
        "verification_body": "DNV GL",
        "registry_url": "https://americancarbonregistry.org",
    },
    {
        "provider_id": "climate-action-reserve",
        "name": "Climate Action Reserve",
        "standard": "CAR Standard",
        "project_types": ["urban_forest", "livestock", "ozone_depleting_substances"],
        "price_usd_per_tonne_co2": 20.0,
        "verification_body": "Apex Companies",
        "registry_url": "https://www.climateactionreserve.org",
    },
]

# Offset verification statuses
VERIFICATION_PENDING = "pending"
VERIFICATION_VERIFIED = "verified"
VERIFICATION_REJECTED = "rejected"
VERIFICATION_EXPIRED = "expired"

# Retirement status values
RETIREMENT_PENDING = "pending"
RETIREMENT_RETIRED = "retired"
RETIREMENT_CANCELLED = "cancelled"


class OffsetIntegrator:
    """Integrates with carbon offset marketplaces for emission neutralization.

    Manages an offset portfolio: provider catalog, purchase tracking,
    retirement certificates, offset-to-emission matching, portfolio coverage
    reporting, and verification status monitoring.
    """

    def __init__(self) -> None:
        """Initialise the offset integrator with empty portfolio and purchase stores."""
        self._purchases: dict[str, dict[str, Any]] = {}
        self._retirements: dict[str, dict[str, Any]] = {}
        self._emission_matches: list[dict[str, Any]] = []

    async def list_providers(
        self,
        project_type: str | None = None,
        max_price_usd_per_tonne: float | None = None,
    ) -> list[dict[str, Any]]:
        """List available offset providers from the catalog.

        Args:
            project_type: Optional filter by project type (e.g., renewable_energy, reforestation).
            max_price_usd_per_tonne: Optional maximum price filter.

        Returns:
            Filtered and sorted list of provider catalog entries.
        """
        providers = list(OFFSET_PROVIDER_CATALOG)
        if project_type:
            providers = [
                p for p in providers
                if project_type in p.get("project_types", [])
            ]
        if max_price_usd_per_tonne is not None:
            providers = [
                p for p in providers
                if p.get("price_usd_per_tonne_co2", 9999) <= max_price_usd_per_tonne
            ]
        return sorted(providers, key=lambda p: p.get("price_usd_per_tonne_co2", 9999))

    async def purchase_offsets(
        self,
        tenant_id: str,
        provider_id: str,
        quantity_tonnes_co2: float,
        project_type: str,
        vintage_year: int,
        notes: str = "",
    ) -> dict[str, Any]:
        """Record a carbon offset purchase from a marketplace provider.

        Args:
            tenant_id: Owning tenant UUID string.
            provider_id: Provider identifier from the catalog.
            quantity_tonnes_co2: Tonnes of CO2 offset purchased.
            project_type: Type of offset project (renewable_energy, reforestation, etc.).
            vintage_year: Year in which the carbon reduction occurred.
            notes: Optional purchase notes or reference IDs.

        Returns:
            Purchase record dict with purchase_id, total_cost_usd, and serial_numbers.

        Raises:
            ValueError: If provider_id is not in the catalog or quantity <= 0.
        """
        if quantity_tonnes_co2 <= 0:
            raise ValueError(f"quantity_tonnes_co2 must be positive, got {quantity_tonnes_co2}")

        provider = next((p for p in OFFSET_PROVIDER_CATALOG if p["provider_id"] == provider_id), None)
        if not provider:
            raise ValueError(
                f"Unknown provider_id: '{provider_id}'. "
                f"Available: {[p['provider_id'] for p in OFFSET_PROVIDER_CATALOG]}"
            )

        price_per_tonne = provider["price_usd_per_tonne_co2"]
        total_cost_usd = round(quantity_tonnes_co2 * price_per_tonne, 2)
        purchase_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc)

        # Generate serial numbers for individual offset credits
        serial_numbers = [
            f"{provider['standard'].replace(' ', '_')}-{vintage_year}-{purchase_id[:8]}-{i:04d}"
            for i in range(1, min(int(quantity_tonnes_co2) + 1, 6))
        ]

        purchase_record: dict[str, Any] = {
            "purchase_id": purchase_id,
            "tenant_id": tenant_id,
            "provider_id": provider_id,
            "provider_name": provider["name"],
            "standard": provider["standard"],
            "project_type": project_type,
            "vintage_year": vintage_year,
            "quantity_tonnes_co2": quantity_tonnes_co2,
            "price_per_tonne_usd": price_per_tonne,
            "total_cost_usd": total_cost_usd,
            "serial_numbers": serial_numbers,
            "verification_status": VERIFICATION_PENDING,
            "retirement_status": RETIREMENT_PENDING,
            "purchased_at": now.isoformat(),
            "notes": notes,
        }
        self._purchases[purchase_id] = purchase_record

        logger.info(
            "Carbon offsets purchased",
            purchase_id=purchase_id,
            provider_id=provider_id,
            quantity_tonnes=quantity_tonnes_co2,
            total_cost_usd=total_cost_usd,
            tenant_id=tenant_id,
        )
        return purchase_record

    async def retire_offsets(
        self,
        purchase_id: str,
        quantity_to_retire_tonnes: float | None = None,
        retirement_reason: str = "voluntary_neutralization",
        beneficiary: str | None = None,
    ) -> dict[str, Any]:
        """Retire purchased offset credits, removing them permanently from circulation.

        Args:
            purchase_id: Purchase record to retire from.
            quantity_to_retire_tonnes: Tonnes to retire; retires full purchase if None.
            retirement_reason: Reason for retirement (voluntary_neutralization, compliance).
            beneficiary: Optional entity on whose behalf credits are retired.

        Returns:
            Retirement certificate dict with certificate_id and retirement details.

        Raises:
            KeyError: If purchase_id not found.
            ValueError: If quantity exceeds available credits.
        """
        if purchase_id not in self._purchases:
            raise KeyError(f"Purchase '{purchase_id}' not found")

        purchase = self._purchases[purchase_id]
        if purchase["retirement_status"] == RETIREMENT_RETIRED:
            raise ValueError(f"Purchase '{purchase_id}' is already fully retired.")

        available = purchase["quantity_tonnes_co2"] - sum(
            r["quantity_retired_tonnes"]
            for r in self._retirements.values()
            if r["purchase_id"] == purchase_id
        )
        retire_qty = quantity_to_retire_tonnes if quantity_to_retire_tonnes is not None else available

        if retire_qty > available:
            raise ValueError(
                f"Cannot retire {retire_qty} tonnes — only {available} available from purchase."
            )

        certificate_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc)

        retirement_record: dict[str, Any] = {
            "certificate_id": certificate_id,
            "purchase_id": purchase_id,
            "tenant_id": purchase["tenant_id"],
            "provider_name": purchase["provider_name"],
            "standard": purchase["standard"],
            "project_type": purchase["project_type"],
            "vintage_year": purchase["vintage_year"],
            "quantity_retired_tonnes": retire_qty,
            "retirement_reason": retirement_reason,
            "beneficiary": beneficiary,
            "registry_serial_prefix": purchase["serial_numbers"][0] if purchase["serial_numbers"] else "N/A",
            "verification_status": VERIFICATION_VERIFIED,
            "retired_at": now.isoformat(),
        }
        self._retirements[certificate_id] = retirement_record

        # Update purchase retirement status
        remaining = available - retire_qty
        self._purchases[purchase_id]["retirement_status"] = (
            RETIREMENT_RETIRED if remaining <= 0 else RETIREMENT_PENDING
        )

        logger.info(
            "Carbon offsets retired",
            certificate_id=certificate_id,
            purchase_id=purchase_id,
            quantity_tonnes=retire_qty,
            beneficiary=beneficiary,
        )
        return retirement_record

    async def match_offsets_to_emissions(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        total_emissions_tonnes_co2: float,
    ) -> dict[str, Any]:
        """Match retired offsets against actual emissions for a reporting period.

        Determines how much of the period's emissions have been offset and
        whether the tenant has achieved carbon neutrality for the period.

        Args:
            tenant_id: Owning tenant UUID string.
            period_start: Start of the emission period.
            period_end: End of the emission period.
            total_emissions_tonnes_co2: Total CO2 emitted in the period.

        Returns:
            Matching result dict with covered_tonnes, uncovered_tonnes, and neutrality_status.
        """
        start_iso = period_start.isoformat()
        end_iso = period_end.isoformat()

        # Sum retired offsets in the period (by retirement date)
        retired_in_period = [
            r for r in self._retirements.values()
            if (
                r["tenant_id"] == tenant_id
                and r["verification_status"] == VERIFICATION_VERIFIED
                and start_iso <= r["retired_at"] < end_iso
            )
        ]

        total_retired_tonnes = sum(r["quantity_retired_tonnes"] for r in retired_in_period)
        covered_tonnes = min(total_retired_tonnes, total_emissions_tonnes_co2)
        uncovered_tonnes = max(0.0, total_emissions_tonnes_co2 - total_retired_tonnes)
        over_offset_tonnes = max(0.0, total_retired_tonnes - total_emissions_tonnes_co2)

        coverage_pct = (covered_tonnes / max(total_emissions_tonnes_co2, 0.001)) * 100

        if coverage_pct >= 100.0:
            neutrality_status = "carbon_neutral"
        elif coverage_pct >= 80.0:
            neutrality_status = "near_neutral"
        elif coverage_pct >= 50.0:
            neutrality_status = "partial_offset"
        else:
            neutrality_status = "minimal_offset"

        match_record: dict[str, Any] = {
            "match_id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "period_start": start_iso,
            "period_end": end_iso,
            "total_emissions_tonnes_co2": total_emissions_tonnes_co2,
            "total_retired_tonnes": round(total_retired_tonnes, 3),
            "covered_tonnes": round(covered_tonnes, 3),
            "uncovered_tonnes": round(uncovered_tonnes, 3),
            "over_offset_tonnes": round(over_offset_tonnes, 3),
            "coverage_percentage": round(coverage_pct, 2),
            "neutrality_status": neutrality_status,
            "retirement_certificates": [r["certificate_id"] for r in retired_in_period],
            "matched_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._emission_matches.append(match_record)

        logger.info(
            "Emission-offset match computed",
            tenant_id=tenant_id,
            total_emissions=total_emissions_tonnes_co2,
            total_retired=total_retired_tonnes,
            coverage_pct=coverage_pct,
            neutrality_status=neutrality_status,
        )
        return match_record

    async def get_portfolio_coverage(
        self,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Compute portfolio-level offset coverage summary for a tenant.

        Args:
            tenant_id: Owning tenant UUID string.

        Returns:
            Portfolio summary dict with total purchased, retired, and available tonnes.
        """
        tenant_purchases = [
            p for p in self._purchases.values()
            if p["tenant_id"] == tenant_id
        ]
        tenant_retirements = [
            r for r in self._retirements.values()
            if r["tenant_id"] == tenant_id
        ]

        total_purchased_tonnes = sum(p["quantity_tonnes_co2"] for p in tenant_purchases)
        total_retired_tonnes = sum(r["quantity_retired_tonnes"] for r in tenant_retirements)
        total_available_tonnes = total_purchased_tonnes - total_retired_tonnes
        total_spent_usd = sum(p["total_cost_usd"] for p in tenant_purchases)

        by_standard: dict[str, float] = {}
        for r in tenant_retirements:
            std = r["standard"]
            by_standard[std] = by_standard.get(std, 0.0) + r["quantity_retired_tonnes"]

        return {
            "tenant_id": tenant_id,
            "total_purchased_tonnes": round(total_purchased_tonnes, 3),
            "total_retired_tonnes": round(total_retired_tonnes, 3),
            "available_tonnes": round(total_available_tonnes, 3),
            "total_spent_usd": round(total_spent_usd, 2),
            "purchase_count": len(tenant_purchases),
            "retirement_count": len(tenant_retirements),
            "retired_by_standard": by_standard,
            "provider_breakdown": {
                p["provider_id"]: p["quantity_tonnes_co2"]
                for p in tenant_purchases
            },
        }

    async def verify_purchase(
        self,
        purchase_id: str,
        verification_status: str,
        verified_by: str,
        verification_notes: str = "",
    ) -> dict[str, Any]:
        """Update the verification status of an offset purchase.

        Args:
            purchase_id: Purchase to verify.
            verification_status: New status (verified, rejected, expired).
            verified_by: Identity of the verifying party.
            verification_notes: Optional notes from the verification process.

        Returns:
            Updated purchase record.

        Raises:
            KeyError: If purchase_id not found.
            ValueError: If verification_status is invalid.
        """
        valid_statuses = {VERIFICATION_VERIFIED, VERIFICATION_REJECTED, VERIFICATION_EXPIRED}
        if verification_status not in valid_statuses:
            raise ValueError(
                f"Invalid verification_status '{verification_status}'. "
                f"Valid: {sorted(valid_statuses)}"
            )
        if purchase_id not in self._purchases:
            raise KeyError(f"Purchase '{purchase_id}' not found")

        now = datetime.now(tz=timezone.utc)
        self._purchases[purchase_id]["verification_status"] = verification_status
        self._purchases[purchase_id]["verified_by"] = verified_by
        self._purchases[purchase_id]["verification_notes"] = verification_notes
        self._purchases[purchase_id]["verified_at"] = now.isoformat()

        logger.info(
            "Purchase verification updated",
            purchase_id=purchase_id,
            verification_status=verification_status,
            verified_by=verified_by,
        )
        return self._purchases[purchase_id]

    async def generate_offset_report(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> dict[str, Any]:
        """Generate a comprehensive offset portfolio report for a period.

        Args:
            tenant_id: Owning tenant UUID string.
            period_start: Report period start.
            period_end: Report period end.

        Returns:
            Report dict with purchases, retirements, certificates, and coverage summary.
        """
        report_id = str(uuid.uuid4())
        start_iso = period_start.isoformat()
        end_iso = period_end.isoformat()

        period_purchases = [
            p for p in self._purchases.values()
            if p["tenant_id"] == tenant_id and start_iso <= p["purchased_at"] < end_iso
        ]
        period_retirements = [
            r for r in self._retirements.values()
            if r["tenant_id"] == tenant_id and start_iso <= r["retired_at"] < end_iso
        ]
        period_matches = [
            m for m in self._emission_matches
            if m["tenant_id"] == tenant_id and start_iso <= m["matched_at"] < end_iso
        ]

        total_purchased_tonnes = sum(p["quantity_tonnes_co2"] for p in period_purchases)
        total_retired_tonnes = sum(r["quantity_retired_tonnes"] for r in period_retirements)
        total_cost_usd = sum(p["total_cost_usd"] for p in period_purchases)

        report: dict[str, Any] = {
            "report_id": report_id,
            "tenant_id": tenant_id,
            "period_start": start_iso,
            "period_end": end_iso,
            "purchases": {
                "count": len(period_purchases),
                "total_tonnes": round(total_purchased_tonnes, 3),
                "total_cost_usd": round(total_cost_usd, 2),
                "records": period_purchases,
            },
            "retirements": {
                "count": len(period_retirements),
                "total_tonnes_retired": round(total_retired_tonnes, 3),
                "certificates": [r["certificate_id"] for r in period_retirements],
                "records": period_retirements,
            },
            "emission_matches": period_matches,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        logger.info(
            "Offset report generated",
            report_id=report_id,
            tenant_id=tenant_id,
            total_purchased_tonnes=total_purchased_tonnes,
            total_retired_tonnes=total_retired_tonnes,
        )
        return report


__all__ = ["OffsetIntegrator"]
