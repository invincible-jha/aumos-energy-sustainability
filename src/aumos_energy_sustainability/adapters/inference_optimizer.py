"""InferenceOptimizer — latency-energy tradeoff optimization adapter.

Provides dynamic batching tuning, model selection by energy profile, latency
budget allocation, Pareto frontier analysis (latency vs energy), optimization
recommendations, and A/B testing energy comparisons for AI inference workloads.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Latency tiers for SLA classification (milliseconds)
LATENCY_TIERS: dict[str, dict[str, float]] = {
    "real_time": {
        "p50_ms": 50.0,
        "p95_ms": 100.0,
        "p99_ms": 200.0,
        "description": "Interactive use-cases: chatbots, copilots, live translations",
    },
    "near_real_time": {
        "p50_ms": 200.0,
        "p95_ms": 500.0,
        "p99_ms": 1000.0,
        "description": "Semi-interactive: code generation, document analysis",
    },
    "batch_interactive": {
        "p50_ms": 1000.0,
        "p95_ms": 3000.0,
        "p99_ms": 5000.0,
        "description": "Deferred but user-visible: report generation, search",
    },
    "background": {
        "p50_ms": 10000.0,
        "p95_ms": 30000.0,
        "p99_ms": 60000.0,
        "description": "Offline / overnight: data pipelines, bulk enrichment",
    },
}

# Energy consumption model: mJ per token by model family and precision
ENERGY_PER_TOKEN_MJ: dict[str, dict[str, float]] = {
    "7b_fp16": {"input": 0.02, "output": 0.04},
    "7b_int8": {"input": 0.012, "output": 0.024},
    "7b_int4": {"input": 0.008, "output": 0.016},
    "13b_fp16": {"input": 0.04, "output": 0.08},
    "13b_int8": {"input": 0.024, "output": 0.048},
    "13b_int4": {"input": 0.016, "output": 0.032},
    "30b_fp16": {"input": 0.10, "output": 0.20},
    "30b_int8": {"input": 0.06, "output": 0.12},
    "70b_fp16": {"input": 0.22, "output": 0.44},
    "70b_int8": {"input": 0.13, "output": 0.26},
    "70b_int4": {"input": 0.085, "output": 0.17},
}

# Dynamic batching configuration bounds
BATCH_CONFIG_BOUNDS: dict[str, dict[str, int]] = {
    "real_time": {
        "min_batch_size": 1,
        "max_batch_size": 8,
        "max_queue_delay_ms": 10,
    },
    "near_real_time": {
        "min_batch_size": 1,
        "max_batch_size": 32,
        "max_queue_delay_ms": 50,
    },
    "batch_interactive": {
        "min_batch_size": 4,
        "max_batch_size": 64,
        "max_queue_delay_ms": 200,
    },
    "background": {
        "min_batch_size": 16,
        "max_batch_size": 256,
        "max_queue_delay_ms": 2000,
    },
}

# Pareto frontier resolution: number of points sampled
PARETO_FRONTIER_RESOLUTION: int = 20

# Maximum experiments stored in A/B history per model
MAX_AB_HISTORY_PER_MODEL: int = 50


# ---------------------------------------------------------------------------
# InferenceOptimizer adapter
# ---------------------------------------------------------------------------


class InferenceOptimizer:
    """Optimize inference serving for the best latency-energy tradeoff.

    Provides tools to:
    - Tune dynamic batching parameters per SLA tier
    - Measure the energy impact of different batch sizes
    - Select the most energy-efficient model variant within a latency budget
    - Allocate latency budgets across multi-step inference pipelines
    - Compute the Pareto frontier of latency vs energy for a workload
    - Generate actionable optimization recommendations
    - Run and compare A/B experiments on energy consumption
    """

    def __init__(
        self,
        carbon_intensity_gco2_per_kwh: float = 120.0,
        gpu_thermal_design_power_watts: float = 400.0,
    ) -> None:
        """Initialize InferenceOptimizer.

        Args:
            carbon_intensity_gco2_per_kwh: Current grid carbon intensity used
                for converting energy to carbon emissions in recommendations.
            gpu_thermal_design_power_watts: GPU TDP used for energy estimates
                when per-token energy models are not available.
        """
        self._carbon_intensity = carbon_intensity_gco2_per_kwh
        self._gpu_tdp_watts = gpu_thermal_design_power_watts

        # In-memory experiment registry: model_id -> list of experiment records
        self._ab_experiments: dict[str, list[dict[str, Any]]] = {}

        # Batching configuration history per model
        self._batching_configs: dict[str, dict[str, Any]] = {}

        # Optimization recommendation history
        self._recommendations: list[dict[str, Any]] = []

        logger.info(
            "InferenceOptimizer initialized",
            carbon_intensity_gco2_per_kwh=carbon_intensity_gco2_per_kwh,
            gpu_tdp_watts=gpu_thermal_design_power_watts,
        )

    # ------------------------------------------------------------------
    # Dynamic batching configuration
    # ------------------------------------------------------------------

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

        Balances batch size (energy efficiency) against queue delay (latency).
        Larger batches improve GPU utilization but add queue wait time; this
        method finds the maximum batch size that still satisfies the tier SLA.

        Args:
            model_id: Model identifier to configure batching for.
            latency_tier: SLA tier — real_time | near_real_time |
                batch_interactive | background.
            avg_input_tokens: Average number of input tokens per request.
            avg_output_tokens: Average number of output tokens per request.
            requests_per_second: Observed or expected request arrival rate.
            gpu_memory_gb: Available GPU memory in gigabytes.
            current_batch_size: Existing batch size for comparison (optional).

        Returns:
            Dict containing:
            - recommended_batch_size: Suggested batch size.
            - max_queue_delay_ms: Maximum allowed queue accumulation time.
            - estimated_gpu_utilization_pct: Predicted GPU utilization at this batch.
            - estimated_energy_per_request_mj: Per-request energy estimate.
            - latency_overhead_ms: Expected additional latency from batching.
            - vs_current: Comparison dict if current_batch_size was provided.

        Raises:
            ValueError: If latency_tier is not one of the supported tiers.
        """
        if latency_tier not in BATCH_CONFIG_BOUNDS:
            raise ValueError(
                f"Unknown latency_tier '{latency_tier}'. "
                f"Must be one of: {sorted(BATCH_CONFIG_BOUNDS)}"
            )

        bounds = BATCH_CONFIG_BOUNDS[latency_tier]
        tier_spec = LATENCY_TIERS[latency_tier]

        # Compute token throughput needed
        tokens_per_request = avg_input_tokens + avg_output_tokens
        throughput_tokens_per_sec = requests_per_second * tokens_per_request

        # Memory constraint: each request requires roughly tokens * 2 bytes (fp16 KV)
        kv_cache_bytes_per_req = avg_input_tokens * 2 + avg_output_tokens * 2
        kv_cache_bytes_per_req *= 2  # key + value projections
        max_concurrent_by_memory = max(
            1, int((gpu_memory_gb * 1e9 * 0.3) / max(kv_cache_bytes_per_req, 1))
        )

        # Queue delay constraint: at the target RPS, how many requests arrive
        # in one max_queue_delay_ms window?
        max_queue_window_requests = max(
            1,
            int(requests_per_second * bounds["max_queue_delay_ms"] / 1000.0),
        )

        recommended_batch_size = min(
            bounds["max_batch_size"],
            max(bounds["min_batch_size"], max_queue_window_requests),
            max_concurrent_by_memory,
        )

        # Estimate GPU utilization — normalized to throughput / capacity
        gpu_flops_capacity = self._gpu_tdp_watts * 0.5 * 1e12  # rough TFLOPs estimate
        flops_per_token = 2 * avg_input_tokens  # simplified attention estimate
        utilization = min(
            99.0,
            100.0 * throughput_tokens_per_sec * flops_per_token / max(gpu_flops_capacity, 1),
        )

        # Energy per request at this batch size
        # Batching amortizes fixed GPU overhead; energy scales sub-linearly
        energy_per_token_mj = self._estimate_token_energy_mj(model_id)
        energy_per_request_mj = (
            avg_input_tokens * energy_per_token_mj["input"]
            + avg_output_tokens * energy_per_token_mj["output"]
        ) / max(1, recommended_batch_size ** 0.3)  # sub-linear batching efficiency

        # Latency overhead = time spent waiting in queue
        latency_overhead_ms = (
            (1.0 / max(requests_per_second, 0.001)) * 1000.0 * recommended_batch_size * 0.5
        )
        latency_overhead_ms = min(latency_overhead_ms, bounds["max_queue_delay_ms"])

        result: dict[str, Any] = {
            "model_id": model_id,
            "latency_tier": latency_tier,
            "tier_p95_budget_ms": tier_spec["p95_ms"],
            "recommended_batch_size": recommended_batch_size,
            "max_queue_delay_ms": bounds["max_queue_delay_ms"],
            "estimated_gpu_utilization_pct": round(utilization, 1),
            "estimated_energy_per_request_mj": round(energy_per_request_mj, 4),
            "latency_overhead_ms": round(latency_overhead_ms, 1),
            "memory_constrained_max": max_concurrent_by_memory,
            "queue_constrained_max": max_queue_window_requests,
            "configured_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        if current_batch_size is not None:
            current_energy = (
                avg_input_tokens * energy_per_token_mj["input"]
                + avg_output_tokens * energy_per_token_mj["output"]
            ) / max(1, current_batch_size ** 0.3)
            result["vs_current"] = {
                "current_batch_size": current_batch_size,
                "current_energy_per_request_mj": round(current_energy, 4),
                "energy_delta_mj": round(energy_per_request_mj - current_energy, 4),
                "energy_change_pct": round(
                    (energy_per_request_mj - current_energy) / max(current_energy, 0.0001) * 100,
                    1,
                ),
                "batch_size_change": recommended_batch_size - current_batch_size,
            }

        self._batching_configs[model_id] = result

        logger.info(
            "Dynamic batching configured",
            model_id=model_id,
            latency_tier=latency_tier,
            recommended_batch_size=recommended_batch_size,
            estimated_energy_per_request_mj=round(energy_per_request_mj, 4),
        )
        return result

    async def measure_dynamic_batching_energy_impact(
        self,
        model_id: str,
        batch_sizes: list[int],
        *,
        avg_input_tokens: int = 512,
        avg_output_tokens: int = 256,
        samples_per_size: int = 5,
    ) -> dict[str, Any]:
        """Measure and compare energy consumption across different batch sizes.

        Computes estimated energy consumption for each batch size, showing
        the tradeoff curve between batch throughput efficiency and absolute
        energy use, including diminishing returns at large batches.

        Args:
            model_id: Model to evaluate.
            batch_sizes: List of batch sizes to compare.
            avg_input_tokens: Average input token count per request.
            avg_output_tokens: Average output token count per request.
            samples_per_size: Number of energy samples to simulate per batch size.

        Returns:
            Dict containing:
            - measurements: List of per-batch-size measurement dicts.
            - optimal_batch_size: Batch size with best energy efficiency.
            - optimal_energy_per_request_mj: Energy at optimal batch.
            - efficiency_curve: Energy normalized to single-request baseline.
        """
        if not batch_sizes:
            raise ValueError("batch_sizes must not be empty")

        energy_per_token_mj = self._estimate_token_energy_mj(model_id)
        baseline_energy = (
            avg_input_tokens * energy_per_token_mj["input"]
            + avg_output_tokens * energy_per_token_mj["output"]
        )

        measurements: list[dict[str, Any]] = []
        for batch_size in sorted(batch_sizes):
            # Sub-linear energy reduction from batching (amortized fixed costs)
            batch_efficiency = batch_size ** 0.3
            energy_per_req = baseline_energy / batch_efficiency

            # Simulate variance across samples
            simulated_samples = [
                energy_per_req * (1.0 + 0.05 * (i % 3 - 1)) for i in range(samples_per_size)
            ]
            mean_energy = statistics.mean(simulated_samples)
            stdev_energy = statistics.stdev(simulated_samples) if len(simulated_samples) > 1 else 0.0

            carbon_g = mean_energy / 1000.0 / 3_600_000.0 * self._carbon_intensity * 1e6

            measurements.append({
                "batch_size": batch_size,
                "energy_per_request_mj": round(mean_energy, 4),
                "energy_stdev_mj": round(stdev_energy, 4),
                "total_batch_energy_mj": round(mean_energy * batch_size, 4),
                "efficiency_vs_single": round(baseline_energy / max(mean_energy, 0.0001), 3),
                "carbon_per_request_mg_co2": round(carbon_g * 1000, 6),
            })

        optimal = min(measurements, key=lambda m: m["energy_per_request_mj"])
        efficiency_curve = [
            {"batch_size": m["batch_size"], "normalized_energy": m["efficiency_vs_single"]}
            for m in measurements
        ]

        logger.info(
            "Dynamic batching energy impact measured",
            model_id=model_id,
            batch_sizes_count=len(batch_sizes),
            optimal_batch_size=optimal["batch_size"],
        )
        return {
            "model_id": model_id,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "baseline_energy_single_request_mj": round(baseline_energy, 4),
            "measurements": measurements,
            "optimal_batch_size": optimal["batch_size"],
            "optimal_energy_per_request_mj": optimal["energy_per_request_mj"],
            "efficiency_curve": efficiency_curve,
        }

    # ------------------------------------------------------------------
    # Model selection by energy profile
    # ------------------------------------------------------------------

    async def select_model_by_energy_profile(
        self,
        candidate_models: list[dict[str, Any]],
        *,
        latency_budget_ms: float,
        quality_threshold: float = 0.85,
        prioritize_energy: bool = True,
    ) -> dict[str, Any]:
        """Select the most energy-efficient model variant within a latency budget.

        Filters candidates by latency budget and quality threshold, then ranks
        the remaining models by energy consumption. If prioritize_energy is True,
        the lowest-energy option is selected; otherwise the best quality within
        budget is selected.

        Args:
            candidate_models: List of model profile dicts, each containing:
                - model_id: Unique model identifier.
                - model_family: Key into ENERGY_PER_TOKEN_MJ (e.g. "7b_int8").
                - avg_latency_ms: Measured p50 latency for this model.
                - quality_score: Normalized quality metric (0–1, e.g. benchmark).
                - avg_input_tokens: Expected input tokens.
                - avg_output_tokens: Expected output tokens.
            latency_budget_ms: Maximum acceptable p50 latency in milliseconds.
            quality_threshold: Minimum quality_score to consider a model eligible.
            prioritize_energy: If True, rank by energy; if False, rank by quality.

        Returns:
            Dict containing:
            - selected_model_id: Chosen model identifier.
            - selection_reason: Explanation for selection.
            - energy_per_request_mj: Estimated energy for selected model.
            - carbon_per_request_mg_co2: Carbon footprint estimate.
            - eligible_models: Ranked list of all models that passed filters.
            - rejected_models: Models excluded and why.
        """
        eligible: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for model in candidate_models:
            model_id = model.get("model_id", "unknown")
            avg_latency = model.get("avg_latency_ms", float("inf"))
            quality = model.get("quality_score", 0.0)
            model_family = model.get("model_family", "7b_fp16")
            avg_input = model.get("avg_input_tokens", 512)
            avg_output = model.get("avg_output_tokens", 256)

            if avg_latency > latency_budget_ms:
                rejected.append({
                    "model_id": model_id,
                    "reason": f"avg_latency_ms {avg_latency:.0f} exceeds budget {latency_budget_ms:.0f}",
                })
                continue

            if quality < quality_threshold:
                rejected.append({
                    "model_id": model_id,
                    "reason": f"quality_score {quality:.2f} below threshold {quality_threshold:.2f}",
                })
                continue

            energy_per_token = self._estimate_token_energy_mj(model_id, model_family)
            energy_per_req = (
                avg_input * energy_per_token["input"] + avg_output * energy_per_token["output"]
            )
            carbon_mg = energy_per_req / 1000.0 / 3_600_000.0 * self._carbon_intensity * 1e9

            eligible.append({
                "model_id": model_id,
                "model_family": model_family,
                "avg_latency_ms": avg_latency,
                "quality_score": quality,
                "energy_per_request_mj": round(energy_per_req, 4),
                "carbon_per_request_mg_co2": round(carbon_mg, 6),
                "latency_budget_headroom_ms": round(latency_budget_ms - avg_latency, 1),
            })

        if not eligible:
            raise ValueError(
                f"No candidate models passed filters (budget={latency_budget_ms}ms, "
                f"quality_threshold={quality_threshold}). {len(rejected)} models rejected."
            )

        # Rank eligible models
        if prioritize_energy:
            eligible.sort(key=lambda m: m["energy_per_request_mj"])
            selection_reason = "Lowest energy consumption within latency budget"
        else:
            eligible.sort(key=lambda m: -m["quality_score"])
            selection_reason = "Highest quality score within latency budget"

        selected = eligible[0]
        logger.info(
            "Model selected by energy profile",
            selected_model_id=selected["model_id"],
            energy_per_request_mj=selected["energy_per_request_mj"],
            latency_ms=selected["avg_latency_ms"],
            prioritize_energy=prioritize_energy,
        )

        return {
            "selected_model_id": selected["model_id"],
            "selection_reason": selection_reason,
            "energy_per_request_mj": selected["energy_per_request_mj"],
            "carbon_per_request_mg_co2": selected["carbon_per_request_mg_co2"],
            "avg_latency_ms": selected["avg_latency_ms"],
            "quality_score": selected["quality_score"],
            "eligible_models": eligible,
            "rejected_models": rejected,
            "total_candidates": len(candidate_models),
        }

    # ------------------------------------------------------------------
    # Latency budget allocation
    # ------------------------------------------------------------------

    async def allocate_latency_budget(
        self,
        pipeline_stages: list[dict[str, Any]],
        total_budget_ms: float,
        *,
        allocation_strategy: str = "proportional",
    ) -> dict[str, Any]:
        """Allocate a latency budget across stages of a multi-step inference pipeline.

        Distributes the total latency budget so that energy-intensive stages
        receive tighter constraints, nudging them toward faster (often lower-energy)
        execution paths.

        Strategies:
        - proportional: Allocate based on observed baseline latency share.
        - energy_weighted: Give more budget to low-energy stages (penalize costly ones).
        - equal: Divide evenly among all stages.

        Args:
            pipeline_stages: List of stage dicts, each containing:
                - stage_name: Human-readable identifier.
                - baseline_latency_ms: Observed p50 latency for this stage.
                - energy_per_call_mj: Energy consumed per execution.
                - is_optional: Whether this stage can be skipped under budget pressure.
            total_budget_ms: Total end-to-end latency budget in milliseconds.
            allocation_strategy: One of proportional | energy_weighted | equal.

        Returns:
            Dict containing:
            - stage_allocations: Per-stage budget assignments.
            - total_allocated_ms: Sum of all allocated budgets.
            - budget_utilization_pct: Fraction of total_budget used.
            - slack_ms: Remaining unallocated budget.
            - at_risk_stages: Stages where baseline already exceeds allocation.

        Raises:
            ValueError: If allocation_strategy is invalid or no stages provided.
        """
        valid_strategies = {"proportional", "energy_weighted", "equal"}
        if allocation_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown allocation_strategy '{allocation_strategy}'. "
                f"Must be one of: {sorted(valid_strategies)}"
            )
        if not pipeline_stages:
            raise ValueError("pipeline_stages must not be empty")

        total_baseline = sum(s.get("baseline_latency_ms", 0.0) for s in pipeline_stages)
        total_energy = sum(s.get("energy_per_call_mj", 0.0) for s in pipeline_stages)

        allocations: list[dict[str, Any]] = []

        for stage in pipeline_stages:
            stage_name = stage.get("stage_name", "unknown")
            baseline_ms = stage.get("baseline_latency_ms", 0.0)
            energy_mj = stage.get("energy_per_call_mj", 0.0)
            is_optional = stage.get("is_optional", False)

            if allocation_strategy == "proportional":
                weight = baseline_ms / max(total_baseline, 0.001)
            elif allocation_strategy == "energy_weighted":
                # High-energy stages get less budget (pressure to execute faster)
                energy_fraction = energy_mj / max(total_energy, 0.001)
                weight = max(0.05, (1.0 - energy_fraction) / len(pipeline_stages))
                # Normalize below
            else:  # equal
                weight = 1.0 / len(pipeline_stages)

            allocated_ms = total_budget_ms * weight
            headroom_ms = allocated_ms - baseline_ms

            allocations.append({
                "stage_name": stage_name,
                "baseline_latency_ms": baseline_ms,
                "allocated_budget_ms": round(allocated_ms, 1),
                "headroom_ms": round(headroom_ms, 1),
                "is_optional": is_optional,
                "at_risk": headroom_ms < 0,
                "energy_per_call_mj": energy_mj,
            })

        # For energy_weighted, re-normalize so weights sum to 1
        if allocation_strategy == "energy_weighted":
            raw_sum = sum(a["allocated_budget_ms"] for a in allocations)
            scale = total_budget_ms / max(raw_sum, 0.001)
            for alloc in allocations:
                alloc["allocated_budget_ms"] = round(alloc["allocated_budget_ms"] * scale, 1)
                alloc["headroom_ms"] = round(
                    alloc["allocated_budget_ms"] - alloc["baseline_latency_ms"], 1
                )
                alloc["at_risk"] = alloc["headroom_ms"] < 0

        total_allocated = sum(a["allocated_budget_ms"] for a in allocations)
        at_risk = [a["stage_name"] for a in allocations if a["at_risk"]]

        logger.info(
            "Latency budget allocated",
            strategy=allocation_strategy,
            total_budget_ms=total_budget_ms,
            total_allocated_ms=round(total_allocated, 1),
            at_risk_stages=at_risk,
        )

        return {
            "allocation_strategy": allocation_strategy,
            "total_budget_ms": total_budget_ms,
            "stage_allocations": allocations,
            "total_allocated_ms": round(total_allocated, 1),
            "slack_ms": round(total_budget_ms - total_allocated, 1),
            "budget_utilization_pct": round(total_allocated / max(total_budget_ms, 0.001) * 100, 1),
            "at_risk_stages": at_risk,
        }

    # ------------------------------------------------------------------
    # Pareto frontier: latency vs energy
    # ------------------------------------------------------------------

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

        Samples the batch size space and computes the (latency, energy) tradeoff
        at each point. Returns the non-dominated set — configurations where no
        other configuration is simultaneously faster AND more energy-efficient.

        Args:
            model_id: Model to analyze.
            min_batch_size: Smallest batch size to evaluate.
            max_batch_size: Largest batch size to evaluate.
            avg_input_tokens: Average input token count per request.
            avg_output_tokens: Average output token count per request.
            base_latency_ms: Single-request baseline latency (no batching overhead).

        Returns:
            Dict containing:
            - all_points: All sampled (latency, energy) configurations.
            - pareto_frontier: Non-dominated subset — the efficient frontier.
            - knee_point: Point on the frontier with best latency-energy balance.
            - recommendation: Suggested operating point with explanation.
        """
        energy_per_token = self._estimate_token_energy_mj(model_id)
        baseline_energy = (
            avg_input_tokens * energy_per_token["input"]
            + avg_output_tokens * energy_per_token["output"]
        )

        # Generate candidate batch sizes geometrically spaced
        import math

        if min_batch_size >= max_batch_size:
            batch_sizes = [min_batch_size]
        else:
            log_range = math.log(max_batch_size) - math.log(min_batch_size)
            batch_sizes = [
                max(min_batch_size, min(max_batch_size, int(min_batch_size * math.exp(
                    i * log_range / (PARETO_FRONTIER_RESOLUTION - 1)
                ))))
                for i in range(PARETO_FRONTIER_RESOLUTION)
            ]
            batch_sizes = sorted(set(batch_sizes))

        all_points: list[dict[str, Any]] = []

        for batch_size in batch_sizes:
            # Energy: sub-linear improvement from batching
            energy_per_req = baseline_energy / (batch_size ** 0.3)

            # Latency: increases with batch size due to queue accumulation
            # Modelled as: base_latency + queue_delay where queue grows with batch
            queue_delay_ms = base_latency_ms * 0.1 * math.log1p(batch_size)
            effective_latency_ms = base_latency_ms + queue_delay_ms

            all_points.append({
                "batch_size": batch_size,
                "latency_ms": round(effective_latency_ms, 2),
                "energy_per_request_mj": round(energy_per_req, 4),
                "queue_delay_ms": round(queue_delay_ms, 2),
            })

        # Compute Pareto frontier (non-dominated set)
        pareto: list[dict[str, Any]] = []
        for candidate in all_points:
            dominated = False
            for other in all_points:
                if (
                    other["latency_ms"] <= candidate["latency_ms"]
                    and other["energy_per_request_mj"] <= candidate["energy_per_request_mj"]
                    and (
                        other["latency_ms"] < candidate["latency_ms"]
                        or other["energy_per_request_mj"] < candidate["energy_per_request_mj"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)

        pareto.sort(key=lambda p: p["latency_ms"])

        # Knee point: minimize Euclidean distance from ideal (min latency, min energy)
        if pareto:
            min_lat = min(p["latency_ms"] for p in pareto)
            max_lat = max(p["latency_ms"] for p in pareto)
            min_nrg = min(p["energy_per_request_mj"] for p in pareto)
            max_nrg = max(p["energy_per_request_mj"] for p in pareto)

            def normalized_distance(point: dict[str, Any]) -> float:
                lat_norm = (point["latency_ms"] - min_lat) / max(max_lat - min_lat, 0.001)
                nrg_norm = (
                    point["energy_per_request_mj"] - min_nrg
                ) / max(max_nrg - min_nrg, 0.001)
                return (lat_norm ** 2 + nrg_norm ** 2) ** 0.5

            knee_point = min(pareto, key=normalized_distance)
        else:
            knee_point = all_points[0]

        recommendation = (
            f"Use batch size {knee_point['batch_size']} for the best latency-energy balance. "
            f"This yields {knee_point['latency_ms']:.1f}ms latency and "
            f"{knee_point['energy_per_request_mj']:.4f}mJ per request."
        )

        logger.info(
            "Pareto frontier computed",
            model_id=model_id,
            frontier_size=len(pareto),
            knee_batch_size=knee_point["batch_size"],
        )

        return {
            "model_id": model_id,
            "total_configurations_evaluated": len(all_points),
            "all_points": all_points,
            "pareto_frontier": pareto,
            "pareto_frontier_size": len(pareto),
            "knee_point": knee_point,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Optimization recommendations
    # ------------------------------------------------------------------

    async def generate_optimization_recommendation(
        self,
        model_id: str,
        workload_stats: dict[str, Any],
        *,
        current_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate an actionable inference optimization recommendation.

        Analyses workload statistics and current configuration to identify
        the highest-impact energy optimization action.

        Args:
            model_id: Model identifier.
            workload_stats: Observed workload metrics, expected to contain:
                - avg_latency_ms: Measured average latency.
                - p99_latency_ms: 99th percentile latency.
                - avg_batch_size: Current average batch size.
                - requests_per_second: Observed throughput.
                - gpu_utilization_pct: Average GPU utilization.
                - avg_input_tokens: Average input length.
                - avg_output_tokens: Average output length.
            current_config: Optional current serving configuration dict.

        Returns:
            Dict containing:
            - priority: high | medium | low.
            - action: Recommended action type.
            - description: Human-readable recommendation.
            - estimated_energy_savings_pct: Projected energy reduction.
            - estimated_latency_impact_ms: Latency change if recommendation applied.
            - implementation_steps: Ordered list of steps to apply the recommendation.
        """
        avg_latency = workload_stats.get("avg_latency_ms", 0.0)
        p99_latency = workload_stats.get("p99_latency_ms", 0.0)
        avg_batch = workload_stats.get("avg_batch_size", 1)
        rps = workload_stats.get("requests_per_second", 1.0)
        gpu_util = workload_stats.get("gpu_utilization_pct", 0.0)
        avg_input = workload_stats.get("avg_input_tokens", 512)
        avg_output = workload_stats.get("avg_output_tokens", 256)

        recommendation: dict[str, Any] = {
            "model_id": model_id,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        # Rule 1: Low GPU utilization → increase batch size
        if gpu_util < 30.0 and avg_batch < 16:
            recommended_batch = min(64, avg_batch * 4)
            energy_savings_pct = 100.0 * (1 - (recommended_batch ** 0.3 / avg_batch ** 0.3) ** -1)
            recommendation.update({
                "priority": "high",
                "action": "increase_batch_size",
                "description": (
                    f"GPU utilization is only {gpu_util:.1f}%. "
                    f"Increasing batch size from {avg_batch} to {recommended_batch} "
                    f"will improve GPU efficiency and reduce per-request energy by ~"
                    f"{abs(energy_savings_pct):.0f}%."
                ),
                "estimated_energy_savings_pct": round(abs(energy_savings_pct), 1),
                "estimated_latency_impact_ms": round(avg_latency * 0.1, 1),
                "implementation_steps": [
                    f"Set max_batch_size={recommended_batch} in serving config",
                    "Set max_queue_delay_ms to 2x average inter-request interval",
                    "Monitor p99 latency — it should stay within 20% of current p99",
                    "Gradually increase if p99 remains within SLA",
                ],
            })

        # Rule 2: High p99 tail latency → reduce batch size or enable priority queuing
        elif p99_latency > avg_latency * 5 and avg_batch > 4:
            recommended_batch = max(1, avg_batch // 2)
            recommendation.update({
                "priority": "high",
                "action": "reduce_batch_size_for_tail_latency",
                "description": (
                    f"p99 latency ({p99_latency:.0f}ms) is {p99_latency/max(avg_latency,1):.1f}x "
                    f"the average ({avg_latency:.0f}ms), indicating queuing spikes. "
                    f"Reduce batch size from {avg_batch} to {recommended_batch} "
                    f"and enable priority queuing for real-time requests."
                ),
                "estimated_energy_savings_pct": -5.0,  # slight energy penalty for smaller batches
                "estimated_latency_impact_ms": round(-p99_latency * 0.4, 1),
                "implementation_steps": [
                    f"Set max_batch_size={recommended_batch}",
                    "Enable priority lanes: route interactive requests to batch_size=1 queue",
                    "Route batch/background workloads to higher batch queue",
                    "Monitor p99 improvement over 1-hour window",
                ],
            })

        # Rule 3: Low throughput with high latency → model compression candidate
        elif rps < 2.0 and avg_latency > 500.0:
            recommendation.update({
                "priority": "medium",
                "action": "apply_model_quantization",
                "description": (
                    f"Low throughput ({rps:.1f} RPS) with high latency ({avg_latency:.0f}ms) "
                    f"suggests model is under-serving the GPU. Apply INT8 quantization to "
                    f"reduce memory footprint, increase batch capacity, and cut energy by ~40%."
                ),
                "estimated_energy_savings_pct": 40.0,
                "estimated_latency_impact_ms": round(-avg_latency * 0.2, 1),
                "implementation_steps": [
                    "Run GPTQ or AWQ quantization on model to INT8",
                    "Validate quality on held-out benchmark (expect <1% perplexity increase)",
                    "Deploy quantized model to same serving endpoint",
                    "Run A/B test with 10% traffic split for 24h",
                    "Promote quantized model if latency SLA is met",
                ],
            })

        # Rule 4: Good GPU utilization, near SLA — fine-tune batching
        elif 60.0 <= gpu_util <= 85.0:
            recommendation.update({
                "priority": "low",
                "action": "fine_tune_batching",
                "description": (
                    f"GPU utilization is healthy at {gpu_util:.1f}%. "
                    f"Fine-tune max_queue_delay_ms to extract 5–10% additional energy "
                    f"efficiency without impacting latency SLA."
                ),
                "estimated_energy_savings_pct": 7.0,
                "estimated_latency_impact_ms": round(avg_latency * 0.05, 1),
                "implementation_steps": [
                    "Increase max_queue_delay_ms by 10ms increments",
                    "At each increment, verify p95 latency remains within SLA",
                    "Stop when p95 budget is within 5% of SLA limit",
                ],
            })

        else:
            recommendation.update({
                "priority": "low",
                "action": "monitor",
                "description": (
                    "Current configuration appears well-balanced. "
                    "Continue monitoring for workload pattern changes."
                ),
                "estimated_energy_savings_pct": 0.0,
                "estimated_latency_impact_ms": 0.0,
                "implementation_steps": [
                    "Set up alerts for GPU utilization dropping below 40%",
                    "Set up alerts for p99 latency exceeding SLA * 2",
                    "Review configuration when workload profile changes >20%",
                ],
            })

        recommendation["workload_snapshot"] = {
            "avg_latency_ms": avg_latency,
            "p99_latency_ms": p99_latency,
            "avg_batch_size": avg_batch,
            "requests_per_second": rps,
            "gpu_utilization_pct": gpu_util,
        }

        self._recommendations.append(recommendation)

        logger.info(
            "Optimization recommendation generated",
            model_id=model_id,
            action=recommendation["action"],
            priority=recommendation["priority"],
            energy_savings_pct=recommendation.get("estimated_energy_savings_pct", 0),
        )
        return recommendation

    # ------------------------------------------------------------------
    # A/B testing energy comparison
    # ------------------------------------------------------------------

    async def record_ab_experiment(
        self,
        model_id: str,
        experiment_name: str,
        variant: str,
        *,
        avg_latency_ms: float,
        p99_latency_ms: float,
        energy_per_request_mj: float,
        requests_sampled: int,
        config_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record measurements for one variant in an A/B energy experiment.

        Args:
            model_id: Model being tested.
            experiment_name: Name of the A/B experiment (e.g. "batch_size_16_vs_32").
            variant: Variant identifier — typically "control" or "treatment".
            avg_latency_ms: Mean request latency for this variant.
            p99_latency_ms: 99th percentile latency.
            energy_per_request_mj: Measured average energy per request in millijoules.
            requests_sampled: Number of requests used to compute these metrics.
            config_snapshot: Optional serving configuration associated with this variant.

        Returns:
            Dict with recorded measurement details.
        """
        measurement: dict[str, Any] = {
            "experiment_name": experiment_name,
            "variant": variant,
            "avg_latency_ms": avg_latency_ms,
            "p99_latency_ms": p99_latency_ms,
            "energy_per_request_mj": energy_per_request_mj,
            "requests_sampled": requests_sampled,
            "config_snapshot": config_snapshot or {},
            "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        if model_id not in self._ab_experiments:
            self._ab_experiments[model_id] = []

        experiments = self._ab_experiments[model_id]
        experiments.append(measurement)

        # Trim to max history size
        if len(experiments) > MAX_AB_HISTORY_PER_MODEL:
            self._ab_experiments[model_id] = experiments[-MAX_AB_HISTORY_PER_MODEL:]

        logger.info(
            "A/B experiment measurement recorded",
            model_id=model_id,
            experiment_name=experiment_name,
            variant=variant,
            energy_per_request_mj=energy_per_request_mj,
        )
        return measurement

    async def compare_ab_experiment(
        self,
        model_id: str,
        experiment_name: str,
    ) -> dict[str, Any]:
        """Compare control vs treatment results for an A/B energy experiment.

        Computes statistical comparison of latency and energy between variants,
        and declares a winner based on energy savings while maintaining latency SLA.

        Args:
            model_id: Model the experiment belongs to.
            experiment_name: Name of the experiment to compare.

        Returns:
            Dict containing:
            - experiment_name: The compared experiment.
            - control: Control variant aggregated metrics.
            - treatment: Treatment variant aggregated metrics (if present).
            - energy_delta_mj: Treatment energy minus control energy.
            - energy_change_pct: Percentage change in energy (negative = improvement).
            - latency_regression_ms: Treatment p99 minus control p99.
            - verdict: "treatment_wins" | "control_wins" | "inconclusive".
            - explanation: Human-readable verdict justification.

        Raises:
            ValueError: If experiment data is not found.
        """
        history = self._ab_experiments.get(model_id, [])
        experiment_records = [r for r in history if r["experiment_name"] == experiment_name]

        if not experiment_records:
            raise ValueError(
                f"No A/B experiment data found for model '{model_id}' "
                f"experiment '{experiment_name}'"
            )

        control_records = [r for r in experiment_records if r["variant"] == "control"]
        treatment_records = [r for r in experiment_records if r["variant"] == "treatment"]

        def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
            if not records:
                return {}
            total_requests = sum(r["requests_sampled"] for r in records)
            weighted_lat = sum(r["avg_latency_ms"] * r["requests_sampled"] for r in records)
            weighted_nrg = sum(r["energy_per_request_mj"] * r["requests_sampled"] for r in records)
            max_p99 = max(r["p99_latency_ms"] for r in records)
            return {
                "variant": records[0]["variant"],
                "total_requests_sampled": total_requests,
                "avg_latency_ms": round(weighted_lat / max(total_requests, 1), 2),
                "p99_latency_ms": round(max_p99, 2),
                "avg_energy_per_request_mj": round(weighted_nrg / max(total_requests, 1), 4),
                "measurement_count": len(records),
            }

        control_agg = aggregate(control_records)
        treatment_agg = aggregate(treatment_records)

        result: dict[str, Any] = {
            "model_id": model_id,
            "experiment_name": experiment_name,
            "control": control_agg,
            "treatment": treatment_agg,
        }

        if control_agg and treatment_agg:
            ctrl_nrg = control_agg["avg_energy_per_request_mj"]
            trt_nrg = treatment_agg["avg_energy_per_request_mj"]
            ctrl_p99 = control_agg["p99_latency_ms"]
            trt_p99 = treatment_agg["p99_latency_ms"]

            energy_delta = trt_nrg - ctrl_nrg
            energy_change_pct = energy_delta / max(ctrl_nrg, 0.0001) * 100
            latency_regression_ms = trt_p99 - ctrl_p99

            result["energy_delta_mj"] = round(energy_delta, 4)
            result["energy_change_pct"] = round(energy_change_pct, 1)
            result["latency_regression_ms"] = round(latency_regression_ms, 1)

            # Verdict: treatment wins if energy decreases AND latency does not regress >10%
            if energy_change_pct < -5.0 and latency_regression_ms < ctrl_p99 * 0.10:
                verdict = "treatment_wins"
                explanation = (
                    f"Treatment reduces energy by {abs(energy_change_pct):.1f}% "
                    f"({abs(energy_delta):.4f}mJ/request) with only "
                    f"{latency_regression_ms:.1f}ms p99 latency change — within 10% SLA tolerance."
                )
            elif energy_change_pct > 5.0:
                verdict = "control_wins"
                explanation = (
                    f"Treatment increases energy by {energy_change_pct:.1f}%. "
                    f"Revert to control configuration."
                )
            else:
                verdict = "inconclusive"
                explanation = (
                    f"Energy difference ({energy_change_pct:.1f}%) is within noise threshold. "
                    f"Collect more traffic samples to reach statistical significance."
                )

            result["verdict"] = verdict
            result["explanation"] = explanation

            logger.info(
                "A/B experiment comparison complete",
                model_id=model_id,
                experiment_name=experiment_name,
                verdict=verdict,
                energy_change_pct=round(energy_change_pct, 1),
            )
        else:
            result["verdict"] = "inconclusive"
            result["explanation"] = (
                "Missing control or treatment data — cannot compare variants."
            )

        return result

    async def get_ab_history(self, model_id: str) -> list[dict[str, Any]]:
        """Return all recorded A/B experiment measurements for a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of experiment measurement dicts in recorded order.
        """
        return list(self._ab_experiments.get(model_id, []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_token_energy_mj(
        self,
        model_id: str,
        model_family: str | None = None,
    ) -> dict[str, float]:
        """Estimate per-token energy in millijoules for a model.

        Looks up ENERGY_PER_TOKEN_MJ by model_family. Falls back to heuristics
        derived from model_id name patterns if family is not explicitly provided.

        Args:
            model_id: Model identifier (used for heuristic matching if needed).
            model_family: Optional explicit key into ENERGY_PER_TOKEN_MJ table.

        Returns:
            Dict with 'input' and 'output' energy in millijoules per token.
        """
        if model_family and model_family in ENERGY_PER_TOKEN_MJ:
            return ENERGY_PER_TOKEN_MJ[model_family]

        # Heuristic: match model_id patterns
        model_id_lower = model_id.lower()
        for family_key in ENERGY_PER_TOKEN_MJ:
            if family_key.replace("_", "-") in model_id_lower or family_key in model_id_lower:
                return ENERGY_PER_TOKEN_MJ[family_key]

        # Default: assume mid-range 13b fp16 model
        return ENERGY_PER_TOKEN_MJ["13b_fp16"]


__all__ = ["InferenceOptimizer", "LATENCY_TIERS", "BATCH_CONFIG_BOUNDS", "ENERGY_PER_TOKEN_MJ"]
