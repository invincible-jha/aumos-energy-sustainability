"""HTTP client adapter for the carbon intensity data API.

Implements ICarbonAPIClient against the Electricity Maps API (v3).
All responses are normalised to a common schema regardless of provider.

In development / testing environments, the AUMOS_ENERGY_CARBON_API_KEY
can be left empty — the client will return mock data rather than failing.
"""

from typing import Any

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Mapping of AumOS region identifiers to Electricity Maps zone keys.
# Extend as new regions are onboarded.
REGION_TO_ZONE: dict[str, str] = {
    "us-east-1": "US-MIDA-PJM",
    "us-west-2": "US-NW-PACW",
    "us-central1": "US-MIDW-MISO",
    "eu-west-1": "IE",
    "eu-central-1": "DE",
    "eu-north-1": "SE",
    "ap-southeast-1": "SG",
    "ap-northeast-1": "JP-TK",
    "ap-south-1": "IN-SO",
    "ca-central-1": "CA-ON",
    "sa-east-1": "BR-CS",
    "af-south-1": "ZA",
    "me-south-1": "AE",
}

# Fallback mock data per region for dev/test environments
_MOCK_DATA: dict[str, dict[str, Any]] = {
    "us-east-1": {
        "carbon_intensity_gco2_per_kwh": 380.0,
        "renewable_percentage": 22.0,
        "solar_percentage": 4.0,
        "wind_percentage": 8.0,
        "hydro_percentage": 10.0,
        "nuclear_percentage": 34.0,
    },
    "eu-north-1": {
        "carbon_intensity_gco2_per_kwh": 15.0,
        "renewable_percentage": 97.0,
        "solar_percentage": 2.0,
        "wind_percentage": 45.0,
        "hydro_percentage": 50.0,
        "nuclear_percentage": 0.0,
    },
    "eu-west-1": {
        "carbon_intensity_gco2_per_kwh": 290.0,
        "renewable_percentage": 40.0,
        "solar_percentage": 3.0,
        "wind_percentage": 32.0,
        "hydro_percentage": 5.0,
        "nuclear_percentage": 0.0,
    },
    "eu-central-1": {
        "carbon_intensity_gco2_per_kwh": 350.0,
        "renewable_percentage": 55.0,
        "solar_percentage": 10.0,
        "wind_percentage": 35.0,
        "hydro_percentage": 10.0,
        "nuclear_percentage": 12.0,
    },
    "us-west-2": {
        "carbon_intensity_gco2_per_kwh": 120.0,
        "renewable_percentage": 70.0,
        "solar_percentage": 5.0,
        "wind_percentage": 25.0,
        "hydro_percentage": 40.0,
        "nuclear_percentage": 0.0,
    },
}
_DEFAULT_MOCK: dict[str, Any] = {
    "carbon_intensity_gco2_per_kwh": 400.0,
    "renewable_percentage": 20.0,
    "solar_percentage": 5.0,
    "wind_percentage": 10.0,
    "hydro_percentage": 5.0,
    "nuclear_percentage": 0.0,
}


class CarbonAPIClient:
    """HTTP client for the Electricity Maps carbon intensity API.

    Falls back to built-in mock data when no API key is configured,
    enabling development without external dependencies.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 10.0,
    ) -> None:
        """Initialise the HTTP client.

        Args:
            base_url: Carbon API base URL.
            api_key: API authentication key. Empty string activates mock mode.
            timeout: HTTP request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._mock_mode = not api_key

        if self._mock_mode:
            logger.warning(
                "CarbonAPIClient running in mock mode — no AUMOS_ENERGY_CARBON_API_KEY set"
            )

    def _make_headers(self) -> dict[str, str]:
        """Build request headers.

        Returns:
            Headers dict with auth token if in live mode.
        """
        return {"auth-token": self._api_key} if self._api_key else {}

    async def get_carbon_intensity(self, region: str) -> dict[str, Any]:
        """Fetch the current carbon intensity for a region.

        Args:
            region: AumOS region identifier (e.g. us-east-1).

        Returns:
            Normalised carbon data dict with carbon_intensity_gco2_per_kwh,
            renewable_percentage, and source breakdowns.
        """
        if self._mock_mode:
            data = _MOCK_DATA.get(region, _DEFAULT_MOCK).copy()
            data["region"] = region
            data["source"] = "mock"
            logger.debug("CarbonAPIClient returning mock data", region=region)
            return data

        zone = REGION_TO_ZONE.get(region, region)

        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._make_headers(),
            timeout=self._timeout,
        ) as client:
            response = await client.get(
                "/carbon-intensity/latest",
                params={"zone": zone},
            )
            response.raise_for_status()
            raw = response.json()

        carbon_intensity = float(raw.get("carbonIntensity", 0.0))

        # Electricity Maps returns power breakdown as percentages
        breakdown = raw.get("powerProductionBreakdown", {})
        total_production = sum(v for v in breakdown.values() if isinstance(v, (int, float)) and v > 0)

        def _pct(source: str) -> float:
            value = breakdown.get(source, 0.0) or 0.0
            return (value / total_production * 100.0) if total_production > 0 else 0.0

        solar_pct = _pct("solar")
        wind_onshore_pct = _pct("wind")
        wind_offshore_pct = _pct("wind-offshore")
        hydro_pct = _pct("hydro") + _pct("hydro discharge")
        nuclear_pct = _pct("nuclear")
        renewable_pct = solar_pct + wind_onshore_pct + wind_offshore_pct + hydro_pct

        return {
            "carbon_intensity_gco2_per_kwh": carbon_intensity,
            "renewable_percentage": round(renewable_pct, 2),
            "solar_percentage": round(solar_pct, 2),
            "wind_percentage": round(wind_onshore_pct + wind_offshore_pct, 2),
            "hydro_percentage": round(hydro_pct, 2),
            "nuclear_percentage": round(nuclear_pct, 2),
            "region": region,
            "zone": zone,
            "source": "electricity_maps",
            "raw": raw,
        }

    async def list_zones(self) -> list[dict[str, Any]]:
        """List all supported zones from the carbon intensity provider.

        Returns:
            List of zone dicts with zone_key, country_name, and display_name.
        """
        if self._mock_mode:
            return [
                {"zone_key": zone, "region": region, "source": "mock"}
                for region, zone in REGION_TO_ZONE.items()
            ]

        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._make_headers(),
            timeout=self._timeout,
        ) as client:
            response = await client.get("/zones")
            response.raise_for_status()
            raw = response.json()

        return [
            {
                "zone_key": zone_key,
                "country_name": zone_data.get("countryName", ""),
                "display_name": zone_data.get("zoneName", zone_key),
            }
            for zone_key, zone_data in raw.items()
        ]
