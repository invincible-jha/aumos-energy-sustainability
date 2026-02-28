"""Multi-backend carbon intensity client with Redis caching.

GAP-334: Expand Emission Factor Database.
Supports Electricity Maps API, WattTime, and EPA eGRID as fallback chain.
Redis cache with 5-minute TTL reduces API calls and improves latency.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class CarbonIntensityReading:
    """A single carbon intensity measurement from a data source.

    Attributes:
        zone: Grid zone identifier (e.g., "US-MISO", "DE", "GB").
        carbon_intensity_gco2_per_kwh: gCO2/kWh at time of fetch.
        renewable_percentage: Fraction of renewable generation (0-100).
        source: Data source identifier.
        fetched_at: UTC timestamp of the reading.
    """

    def __init__(
        self,
        zone: str,
        carbon_intensity_gco2_per_kwh: Decimal,
        renewable_percentage: Decimal,
        source: str,
        fetched_at: datetime,
    ) -> None:
        self.zone = zone
        self.carbon_intensity_gco2_per_kwh = carbon_intensity_gco2_per_kwh
        self.renewable_percentage = renewable_percentage
        self.source = source
        self.fetched_at = fetched_at

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "zone": self.zone,
            "carbon_intensity_gco2_per_kwh": str(self.carbon_intensity_gco2_per_kwh),
            "renewable_percentage": str(self.renewable_percentage),
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CarbonIntensityReading":
        """Deserialize from dict (e.g., from Redis cache)."""
        return cls(
            zone=data["zone"],
            carbon_intensity_gco2_per_kwh=Decimal(data["carbon_intensity_gco2_per_kwh"]),
            renewable_percentage=Decimal(data["renewable_percentage"]),
            source=data["source"],
            fetched_at=datetime.fromisoformat(data["fetched_at"]),
        )


class CarbonAPIBackend(Protocol):
    """Protocol for carbon intensity data source backends."""

    async def get_carbon_intensity(self, region: str, zone: str) -> CarbonIntensityReading:
        """Fetch current carbon intensity for a grid zone.

        Args:
            region: AumOS region identifier.
            zone: Grid zone code for the data provider.

        Returns:
            CarbonIntensityReading with live data.
        """
        ...


class ElectricityMapsClient:
    """Electricity Maps API backend (commercial, 200+ global zones).

    Args:
        api_key: Electricity Maps API token.
        http_client: Shared async HTTP client.
    """

    BASE_URL = "https://api.electricitymap.org/v3"

    def __init__(self, api_key: str, http_client: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._client = http_client

    async def get_carbon_intensity(self, region: str, zone: str) -> CarbonIntensityReading:
        """Fetch live carbon intensity from Electricity Maps.

        Args:
            region: AumOS region label (unused by this backend).
            zone: Electricity Maps zone code (e.g., "US-MISO", "DE").

        Returns:
            CarbonIntensityReading from Electricity Maps API.
        """
        response = await self._client.get(
            f"{self.BASE_URL}/carbon-intensity/latest",
            params={"zone": zone},
            headers={"auth-token": self._api_key},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        return CarbonIntensityReading(
            zone=zone,
            carbon_intensity_gco2_per_kwh=Decimal(str(data["carbonIntensity"])),
            renewable_percentage=Decimal(str(data.get("renewablePercentage", 0))),
            source="electricity_maps",
            fetched_at=datetime.now(timezone.utc),
        )


class WattTimeClient:
    """WattTime API backend (US balancing authority areas, free tier available).

    Args:
        username: WattTime account username.
        password: WattTime account password.
        http_client: Shared async HTTP client.
    """

    BASE_URL = "https://api2.watttime.org"

    def __init__(self, username: str, password: str, http_client: httpx.AsyncClient) -> None:
        self._username = username
        self._password = password
        self._client = http_client

    async def get_carbon_intensity(self, region: str, zone: str) -> CarbonIntensityReading:
        """Fetch live marginal operating emission rate from WattTime.

        Args:
            region: AumOS region label.
            zone: WattTime balancing authority abbreviation (e.g., "CAISO_NORTH").

        Returns:
            CarbonIntensityReading with MOER data.
        """
        token_response = await self._client.post(
            f"{self.BASE_URL}/login",
            auth=(self._username, self._password),
            timeout=10.0,
        )
        token_response.raise_for_status()
        token = token_response.json()["token"]

        index_response = await self._client.get(
            f"{self.BASE_URL}/index",
            headers={"Authorization": f"Bearer {token}"},
            params={"ba": zone, "style": "all"},
            timeout=10.0,
        )
        index_response.raise_for_status()
        data = index_response.json()
        return CarbonIntensityReading(
            zone=zone,
            carbon_intensity_gco2_per_kwh=Decimal(str(data.get("moer", 500))),
            renewable_percentage=Decimal("0"),
            source="watttime",
            fetched_at=datetime.now(timezone.utc),
        )


class EPAeGRIDClient:
    """EPA eGRID static annual emission factors (US only, free, no API key).

    US annual averages from EPA eGRID 2022 data.
    Used as third fallback when both Electricity Maps and WattTime are unavailable.
    """

    # eGRID 2022 annual average lb CO2/MWh by US subregion, converted to gCO2/kWh
    # Source: EPA eGRID 2022 Summary Tables
    EGRID_FACTORS_GCO2_KWH: dict[str, float] = {
        "WECC": 320.0,     # Western Interconnection
        "MROE": 700.0,     # Midwest Reliability Organization East
        "MROW": 560.0,     # Midwest Reliability Organization West
        "RFCE": 280.0,     # ReliabilityFirst Corporation East
        "SRSO": 470.0,     # SERC Reliability South
        "SPPS": 480.0,     # Southwest Power Pool South
        "NYLI": 200.0,     # New York Long Island
        "NEWE": 220.0,     # New England
        "US_AVG": 386.0,   # US national average
    }

    async def get_carbon_intensity(self, region: str, zone: str) -> CarbonIntensityReading:
        """Return static EPA eGRID annual emission factor for a US subregion.

        Args:
            region: AumOS region label.
            zone: eGRID subregion code. Falls back to US_AVG.

        Returns:
            CarbonIntensityReading with static annual factor.
        """
        intensity = self.EGRID_FACTORS_GCO2_KWH.get(zone, self.EGRID_FACTORS_GCO2_KWH["US_AVG"])
        return CarbonIntensityReading(
            zone=zone,
            carbon_intensity_gco2_per_kwh=Decimal(str(intensity)),
            renewable_percentage=Decimal("20"),  # US avg renewable ~20%
            source="epa_egrid_static",
            fetched_at=datetime.now(timezone.utc),
        )


class MultiBackendCarbonClient:
    """Carbon intensity client with fallback chain and Redis cache.

    Fallback order: Electricity Maps → WattTime → EPA eGRID → Mock data.
    Redis cache uses 5-minute TTL to reduce API calls.

    Args:
        electricity_maps: Optional Electricity Maps client.
        watttime: Optional WattTime client.
        egrid: Optional EPA eGRID client.
        redis_client: Optional Redis client for caching.
        cache_ttl_seconds: Cache TTL (default 300 = 5 minutes).
    """

    def __init__(
        self,
        electricity_maps: ElectricityMapsClient | None = None,
        watttime: WattTimeClient | None = None,
        egrid: EPAeGRIDClient | None = None,
        redis_client: "Any | None" = None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._backends: list[CarbonAPIBackend] = []
        if electricity_maps is not None:
            self._backends.append(electricity_maps)
        if watttime is not None:
            self._backends.append(watttime)
        if egrid is not None:
            self._backends.append(egrid)
        self._redis = redis_client
        self._cache_ttl = cache_ttl_seconds

    async def get_carbon_intensity(self, region: str, zone: str) -> CarbonIntensityReading:
        """Fetch carbon intensity with fallback chain and Redis cache.

        Args:
            region: AumOS region identifier.
            zone: Grid zone code.

        Returns:
            CarbonIntensityReading from the first available backend.
        """
        cache_key = f"aumos:carbon:{region}:{zone}"

        # Try cache first
        if self._redis is not None:
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    logger.debug("carbon_cache_hit", zone=zone, region=region)
                    return CarbonIntensityReading.from_dict(data)
            except Exception as exc:
                logger.warning("redis_cache_read_error", reason=str(exc))

        # Try backends in fallback order
        last_error: Exception | None = None
        for backend in self._backends:
            try:
                reading = await backend.get_carbon_intensity(region, zone)
                if self._redis is not None:
                    try:
                        await self._redis.setex(
                            cache_key, self._cache_ttl, json.dumps(reading.to_dict())
                        )
                    except Exception as exc:
                        logger.warning("redis_cache_write_error", reason=str(exc))
                logger.info("carbon_intensity_fetched", zone=zone, source=reading.source)
                return reading
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "carbon_backend_failed",
                    backend=type(backend).__name__,
                    zone=zone,
                    reason=str(exc),
                )

        # All backends failed — return a conservative mock value
        logger.error("all_carbon_backends_failed", zone=zone, last_error=str(last_error))
        return CarbonIntensityReading(
            zone=zone,
            carbon_intensity_gco2_per_kwh=Decimal("400"),
            renewable_percentage=Decimal("20"),
            source="mock_fallback",
            fetched_at=datetime.now(timezone.utc),
        )


# Import Any for type hint in __init__
from typing import Any  # noqa: E402
