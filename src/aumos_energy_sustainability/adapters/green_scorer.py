"""Green scorer adapter for aumos-energy-sustainability.

Environmental impact scoring: per-workload carbon footprint computation,
PUE-adjusted energy efficiency scoring, FLOPS/watt model efficiency scoring,
comparative scoring against baselines, green certification thresholds,
scoring methodology documentation, and leaderboard data.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Industry benchmark PUE values for data center efficiency
PUE_BENCHMARKS: dict[str, float] = {
    "hyperscale_leader": 1.10,  # Best-in-class hyperscalers
    "hyperscale_average": 1.20,
    "colocation_best": 1.35,
    "colocation_average": 1.50,
    "enterprise_average": 1.80,
    "industry_average": 2.00,
}

# Green certification threshold levels (green score 0–100)
CERTIFICATION_THRESHOLDS: dict[str, float] = {
    "platinum": 90.0,
    "gold": 75.0,
    "silver": 60.0,
    "bronze": 40.0,
    "unrated": 0.0,
}

# Carbon efficiency benchmark per 1B parameter model per inference (gCO2)
CARBON_EFFICIENCY_BENCHMARKS: dict[str, float] = {
    "top_1pct": 0.001,
    "top_10pct": 0.005,
    "average": 0.020,
    "below_average": 0.050,
    "poor": 0.100,
}


class GreenScorer:
    """Computes environmental impact scores for AI workloads and models.

    Provides PUE-adjusted energy efficiency scoring, FLOPS/watt model scoring,
    comparative analysis against industry benchmarks, and leaderboard data
    to incentivize sustainable AI practices.
    """

    def __init__(
        self,
        baseline_pue: float = PUE_BENCHMARKS["industry_average"],
        baseline_carbon_intensity: float = 400.0,
    ) -> None:
        """Initialise the green scorer.

        Args:
            baseline_pue: Reference PUE for efficiency comparison (default: industry average 2.0).
            baseline_carbon_intensity: Reference carbon intensity gCO2/kWh for comparisons.
        """
        self._baseline_pue = baseline_pue
        self._baseline_carbon_intensity = baseline_carbon_intensity
        self._score_history: list[dict[str, Any]] = []

    async def compute_workload_carbon_footprint(
        self,
        workload_id: str,
        energy_kwh: float,
        carbon_intensity_gco2_per_kwh: float,
        pue: float = 1.5,
        renewable_percentage: float = 0.0,
    ) -> dict[str, Any]:
        """Compute the total carbon footprint for a workload with PUE adjustment.

        Applies the Power Usage Effectiveness factor to account for overhead
        from cooling and power distribution losses.

        Args:
            workload_id: External workload identifier.
            energy_kwh: IT equipment energy consumed in kWh.
            carbon_intensity_gco2_per_kwh: Grid carbon intensity.
            pue: Power Usage Effectiveness (1.0 = perfect, 2.0 = industry average).
            renewable_percentage: Fraction of energy from renewable sources (0–100).

        Returns:
            Carbon footprint dict with total_co2_gco2, pue_adjusted_energy_kwh, and breakdown.
        """
        if pue < 1.0:
            raise ValueError(f"PUE must be >= 1.0, got {pue}")

        # PUE-adjusted total facility energy
        pue_adjusted_energy_kwh = energy_kwh * pue

        # Renewable energy displaces fossil grid intensity
        effective_intensity = carbon_intensity_gco2_per_kwh * (1.0 - renewable_percentage / 100.0)
        total_co2_gco2 = pue_adjusted_energy_kwh * effective_intensity
        total_co2_kg = total_co2_gco2 / 1000.0

        # Overhead from PUE (cooling, power conditioning, lighting)
        overhead_energy_kwh = pue_adjusted_energy_kwh - energy_kwh
        overhead_co2_gco2 = overhead_energy_kwh * effective_intensity

        result: dict[str, Any] = {
            "workload_id": workload_id,
            "it_energy_kwh": energy_kwh,
            "pue": pue,
            "pue_adjusted_energy_kwh": round(pue_adjusted_energy_kwh, 6),
            "overhead_energy_kwh": round(overhead_energy_kwh, 6),
            "carbon_intensity_gco2_per_kwh": carbon_intensity_gco2_per_kwh,
            "effective_intensity_after_renewables": round(effective_intensity, 3),
            "renewable_percentage": renewable_percentage,
            "total_co2_gco2": round(total_co2_gco2, 4),
            "total_co2_kg": round(total_co2_kg, 6),
            "overhead_co2_gco2": round(overhead_co2_gco2, 4),
            "it_co2_gco2": round(total_co2_gco2 - overhead_co2_gco2, 4),
            "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        logger.info(
            "Workload carbon footprint computed",
            workload_id=workload_id,
            total_co2_kg=total_co2_kg,
            pue=pue,
            renewable_percentage=renewable_percentage,
        )
        return result

    async def compute_energy_efficiency_score(
        self,
        actual_pue: float,
        renewable_percentage: float,
        carbon_intensity_gco2_per_kwh: float,
    ) -> dict[str, Any]:
        """Compute an energy efficiency score normalized to industry benchmarks.

        Score components:
          - PUE efficiency: how close to perfect (1.0) the facility PUE is.
          - Renewable fraction: percentage of renewable energy used.
          - Carbon intensity: how clean the grid is relative to baseline.

        Args:
            actual_pue: Measured PUE of the facility.
            renewable_percentage: Renewable energy fraction (0–100).
            carbon_intensity_gco2_per_kwh: Current grid carbon intensity.

        Returns:
            Efficiency score dict with composite_score (0–100), component scores, and tier.
        """
        # PUE score: 100 at PUE=1.0, 0 at PUE=3.0 (clamped)
        pue_score = max(0.0, min(100.0, (3.0 - actual_pue) / (3.0 - 1.0) * 100.0))

        # Renewable score: linear 0–100
        renewable_score = renewable_percentage

        # Carbon intensity score: how far below the baseline is the actual intensity
        carbon_score = max(
            0.0,
            min(100.0, (self._baseline_carbon_intensity - carbon_intensity_gco2_per_kwh)
                / self._baseline_carbon_intensity * 100.0)
        )

        # Weights: PUE 30%, renewable 50%, carbon 20%
        composite = 0.30 * pue_score + 0.50 * renewable_score + 0.20 * carbon_score
        composite = round(composite, 2)

        tier = self._get_certification_tier(composite)

        result: dict[str, Any] = {
            "composite_score": composite,
            "component_scores": {
                "pue_efficiency_score": round(pue_score, 2),
                "renewable_score": round(renewable_score, 2),
                "carbon_intensity_score": round(carbon_score, 2),
            },
            "component_weights": {
                "pue": 0.30,
                "renewable": 0.50,
                "carbon_intensity": 0.20,
            },
            "certification_tier": tier,
            "actual_pue": actual_pue,
            "baseline_pue": self._baseline_pue,
            "pue_vs_baseline": round(self._baseline_pue - actual_pue, 3),
            "scored_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        return result

    async def compute_model_efficiency_score(
        self,
        model_id: str,
        model_version: str,
        flops_per_inference: float,
        energy_per_inference_wh: float,
        quality_score: float,
        parameter_count_billions: float,
    ) -> dict[str, Any]:
        """Score a model's energy efficiency using FLOPS/watt and quality-adjusted metrics.

        Args:
            model_id: Model identifier.
            model_version: Model version.
            flops_per_inference: Floating-point operations per inference call.
            energy_per_inference_wh: Energy consumed per inference in watt-hours.
            quality_score: Normalized quality score (0–100, e.g., benchmark accuracy).
            parameter_count_billions: Model size in billions of parameters.

        Returns:
            Model efficiency score dict with flops_per_watt, quality_adjusted_score, and tier.
        """
        energy_per_inference_j = energy_per_inference_wh * 3600  # Wh to J
        watts_average = energy_per_inference_j / max(0.001, 1.0)  # assume 1s inference

        flops_per_watt = flops_per_inference / max(watts_average, 0.001)

        # Efficiency per parameter: fewer params to achieve same quality = more efficient
        efficiency_per_param = quality_score / max(parameter_count_billions, 0.001)

        # Normalize against a reference model (e.g., GPT-2 at ~1.5B params)
        reference_flops_per_watt = 1e12
        normalized_flops_efficiency = min(100.0, (flops_per_watt / reference_flops_per_watt) * 50.0)

        normalized_param_efficiency = min(100.0, efficiency_per_param * 5.0)

        composite = 0.6 * normalized_flops_efficiency + 0.4 * normalized_param_efficiency
        composite = round(composite, 2)
        tier = self._get_certification_tier(composite)

        result: dict[str, Any] = {
            "model_id": model_id,
            "model_version": model_version,
            "parameter_count_billions": parameter_count_billions,
            "flops_per_inference": flops_per_inference,
            "energy_per_inference_wh": energy_per_inference_wh,
            "flops_per_watt": round(flops_per_watt, 2),
            "quality_score": quality_score,
            "efficiency_per_billion_params": round(efficiency_per_param, 4),
            "normalized_flops_efficiency": round(normalized_flops_efficiency, 2),
            "normalized_param_efficiency": round(normalized_param_efficiency, 2),
            "composite_model_efficiency_score": composite,
            "certification_tier": tier,
            "scored_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._score_history.append(result)

        logger.info(
            "Model efficiency score computed",
            model_id=model_id,
            composite_score=composite,
            flops_per_watt=flops_per_watt,
            tier=tier,
        )
        return result

    async def compare_against_baseline(
        self,
        workload_id: str,
        actual_carbon_gco2: float,
        actual_energy_kwh: float,
        baseline_carbon_gco2: float | None = None,
        baseline_energy_kwh: float | None = None,
    ) -> dict[str, Any]:
        """Compare workload metrics against a baseline to compute relative improvement.

        Args:
            workload_id: Workload identifier.
            actual_carbon_gco2: Measured carbon output in gCO2.
            actual_energy_kwh: Measured energy consumption in kWh.
            baseline_carbon_gco2: Reference carbon (default: industry average baseline).
            baseline_energy_kwh: Reference energy (default: estimated from carbon).

        Returns:
            Comparison dict with carbon_saving_pct, energy_saving_pct, and green_delta_score.
        """
        effective_baseline_carbon = (
            baseline_carbon_gco2
            or (actual_energy_kwh * self._baseline_carbon_intensity)
        )
        effective_baseline_energy = baseline_energy_kwh or (actual_energy_kwh * self._baseline_pue / 1.5)

        carbon_delta = effective_baseline_carbon - actual_carbon_gco2
        energy_delta = effective_baseline_energy - actual_energy_kwh
        carbon_saving_pct = (carbon_delta / max(effective_baseline_carbon, 0.001)) * 100
        energy_saving_pct = (energy_delta / max(effective_baseline_energy, 0.001)) * 100

        # Green delta score: how many points above neutral (50)
        green_delta_score = round(50.0 + (carbon_saving_pct * 0.4 + energy_saving_pct * 0.1), 2)
        green_delta_score = max(0.0, min(100.0, green_delta_score))

        return {
            "workload_id": workload_id,
            "actual_carbon_gco2": actual_carbon_gco2,
            "baseline_carbon_gco2": round(effective_baseline_carbon, 4),
            "carbon_delta_gco2": round(carbon_delta, 4),
            "carbon_saving_pct": round(carbon_saving_pct, 2),
            "actual_energy_kwh": actual_energy_kwh,
            "baseline_energy_kwh": round(effective_baseline_energy, 6),
            "energy_delta_kwh": round(energy_delta, 6),
            "energy_saving_pct": round(energy_saving_pct, 2),
            "green_delta_score": green_delta_score,
            "is_better_than_baseline": carbon_delta >= 0,
        }

    def _get_certification_tier(self, score: float) -> str:
        """Determine certification tier from a score (0–100).

        Args:
            score: Composite green score.

        Returns:
            Certification tier label.
        """
        for tier, threshold in sorted(CERTIFICATION_THRESHOLDS.items(), key=lambda t: -t[1]):
            if score >= threshold:
                return tier
        return "unrated"

    async def get_certification_status(
        self,
        composite_score: float,
    ) -> dict[str, Any]:
        """Determine the green certification status for a composite score.

        Args:
            composite_score: Green score between 0.0 and 100.0.

        Returns:
            Certification status dict with tier, next_tier, and points_required.
        """
        current_tier = self._get_certification_tier(composite_score)
        sorted_tiers = sorted(CERTIFICATION_THRESHOLDS.items(), key=lambda t: t[1])

        next_tier: str | None = None
        points_to_next: float | None = None

        for tier_name, threshold in sorted_tiers:
            if threshold > composite_score:
                next_tier = tier_name
                points_to_next = round(threshold - composite_score, 2)
                break

        return {
            "composite_score": composite_score,
            "current_tier": current_tier,
            "next_tier": next_tier,
            "points_to_next_tier": points_to_next,
            "certification_thresholds": CERTIFICATION_THRESHOLDS,
            "is_certified": current_tier not in ("unrated",),
        }

    async def get_leaderboard(
        self,
        model_ids: list[str] | None = None,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Return a leaderboard of model efficiency scores.

        Args:
            model_ids: Optional list of model IDs to include; returns all if None.
            top_n: Maximum number of entries to return.

        Returns:
            Ranked list of model efficiency score records, highest score first.
        """
        records = [r for r in self._score_history if "model_id" in r]
        if model_ids:
            records = [r for r in records if r["model_id"] in model_ids]

        # Take the latest score per model_id:version
        latest: dict[str, dict[str, Any]] = {}
        for record in records:
            key = f"{record['model_id']}:{record['model_version']}"
            if key not in latest or record["scored_at"] > latest[key]["scored_at"]:
                latest[key] = record

        ranked = sorted(
            latest.values(),
            key=lambda r: r["composite_model_efficiency_score"],
            reverse=True,
        )

        return [
            {
                "rank": idx + 1,
                "model_id": r["model_id"],
                "model_version": r["model_version"],
                "composite_score": r["composite_model_efficiency_score"],
                "certification_tier": r["certification_tier"],
                "flops_per_watt": r["flops_per_watt"],
                "parameter_count_billions": r["parameter_count_billions"],
            }
            for idx, r in enumerate(ranked[:top_n])
        ]

    async def document_scoring_methodology(self) -> dict[str, Any]:
        """Return a structured description of the scoring methodology.

        Returns:
            Methodology documentation dict with formulas, weights, and benchmark references.
        """
        return {
            "version": "1.0",
            "description": "AumOS Green Scorer methodology for AI workload environmental impact.",
            "energy_efficiency_score": {
                "components": {
                    "pue_efficiency": {"weight": 0.30, "formula": "(3.0 - actual_pue) / 2.0 * 100"},
                    "renewable_percentage": {"weight": 0.50, "formula": "renewable_pct (0–100)"},
                    "carbon_intensity": {"weight": 0.20, "formula": "(baseline_intensity - actual_intensity) / baseline_intensity * 100"},
                },
                "composite": "0.30 * pue_score + 0.50 * renewable_score + 0.20 * carbon_score",
            },
            "model_efficiency_score": {
                "components": {
                    "flops_per_watt": {"weight": 0.60, "formula": "FLOPS / (energy_wh * 3600)"},
                    "quality_per_param": {"weight": 0.40, "formula": "quality_score / parameter_count_billions"},
                },
                "composite": "0.60 * normalized_flops_score + 0.40 * normalized_param_score",
            },
            "certification_tiers": CERTIFICATION_THRESHOLDS,
            "pue_benchmarks": PUE_BENCHMARKS,
            "carbon_efficiency_benchmarks": CARBON_EFFICIENCY_BENCHMARKS,
            "references": [
                "GHG Protocol Corporate Standard",
                "The Green Grid PUE metric",
                "MLPerf efficiency benchmark",
                "EU AI Act environmental impact provisions",
            ],
        }


__all__ = ["GreenScorer"]
