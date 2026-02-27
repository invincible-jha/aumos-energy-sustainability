"""GPU efficiency optimizer adapter for aumos-energy-sustainability.

Maximizes GPU utilization: monitoring, batch size optimization, model
parallelism configuration, idle GPU detection, workload consolidation
recommendations, efficiency scoring, and optimization reports.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# GPU utilization thresholds
GPU_IDLE_THRESHOLD_PCT = 10.0       # Below this: GPU is considered idle
GPU_LOW_UTIL_THRESHOLD_PCT = 40.0   # Below this: batch size should be increased
GPU_TARGET_UTIL_PCT = 80.0          # Optimal target utilization

# Model parallelism strategies and their minimum GPU requirements
PARALLELISM_STRATEGIES: dict[str, dict[str, Any]] = {
    "tensor_parallel": {"min_gpus": 2, "scales_at_billions": 13},
    "pipeline_parallel": {"min_gpus": 2, "scales_at_billions": 30},
    "data_parallel": {"min_gpus": 1, "scales_at_billions": 0},
    "mixed_parallel": {"min_gpus": 4, "scales_at_billions": 70},
}


class EfficiencyOptimizer:
    """Maximizes GPU efficiency and energy utilization for AI workloads.

    Analyses GPU utilization metrics, recommends optimal batch sizes and
    parallelism configurations, detects idle hardware, and produces actionable
    consolidation recommendations and efficiency scores.
    """

    def __init__(
        self,
        gpu_idle_threshold_pct: float = GPU_IDLE_THRESHOLD_PCT,
        target_utilization_pct: float = GPU_TARGET_UTIL_PCT,
    ) -> None:
        """Initialise the efficiency optimizer.

        Args:
            gpu_idle_threshold_pct: Utilization percentage below which a GPU is idle.
            target_utilization_pct: Optimal GPU utilization target percentage.
        """
        self._gpu_idle_threshold = gpu_idle_threshold_pct
        self._target_utilization = target_utilization_pct
        self._utilization_history: list[dict[str, Any]] = []
        self._optimization_reports: list[dict[str, Any]] = []

    async def monitor_gpu_utilization(
        self,
        node_id: str,
        gpu_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Record GPU utilization metrics for a node.

        Args:
            node_id: Host node identifier.
            gpu_metrics: List of per-GPU metric dicts with fields:
                gpu_index, utilization_pct, memory_used_mb, memory_total_mb,
                temperature_c, power_draw_watts.

        Returns:
            Summary dict with average utilization, idle GPU count, and efficiency score.
        """
        if not gpu_metrics:
            return {"node_id": node_id, "gpu_count": 0, "average_utilization_pct": 0.0}

        avg_util = sum(g.get("utilization_pct", 0.0) for g in gpu_metrics) / len(gpu_metrics)
        idle_gpus = [
            g for g in gpu_metrics
            if g.get("utilization_pct", 0.0) < self._gpu_idle_threshold
        ]
        low_util_gpus = [
            g for g in gpu_metrics
            if self._gpu_idle_threshold <= g.get("utilization_pct", 0.0) < GPU_LOW_UTIL_THRESHOLD_PCT
        ]

        efficiency_score = min(100.0, (avg_util / self._target_utilization) * 100.0)

        snapshot: dict[str, Any] = {
            "snapshot_id": str(uuid.uuid4()),
            "node_id": node_id,
            "gpu_count": len(gpu_metrics),
            "average_utilization_pct": round(avg_util, 2),
            "idle_gpu_count": len(idle_gpus),
            "low_util_gpu_count": len(low_util_gpus),
            "peak_utilization_pct": max(g.get("utilization_pct", 0.0) for g in gpu_metrics),
            "efficiency_score": round(efficiency_score, 2),
            "per_gpu_metrics": gpu_metrics,
            "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._utilization_history.append(snapshot)

        logger.info(
            "GPU utilization monitored",
            node_id=node_id,
            avg_utilization=avg_util,
            idle_gpus=len(idle_gpus),
            efficiency_score=efficiency_score,
        )
        return snapshot

    async def optimize_batch_size(
        self,
        model_id: str,
        current_batch_size: int,
        avg_gpu_utilization_pct: float,
        memory_used_pct: float,
        sequence_length: int = 512,
        target_utilization_pct: float | None = None,
    ) -> dict[str, Any]:
        """Recommend an optimal batch size to maximize GPU utilization.

        Uses empirical scaling heuristics: if utilization is low and memory
        headroom exists, batch size can be increased proportionally.

        Args:
            model_id: Model identifier for logging context.
            current_batch_size: Current inference batch size.
            avg_gpu_utilization_pct: Current average GPU utilization percentage.
            memory_used_pct: Current GPU memory utilization percentage.
            sequence_length: Token sequence length (affects memory per sample).
            target_utilization_pct: Target utilization; uses manager default if None.

        Returns:
            Optimization recommendation dict with recommended_batch_size and justification.
        """
        target = target_utilization_pct or self._target_utilization
        memory_headroom_pct = 100.0 - memory_used_pct

        if avg_gpu_utilization_pct >= target:
            recommended_batch_size = current_batch_size
            action = "maintain"
            justification = f"GPU utilization {avg_gpu_utilization_pct:.1f}% is already at or above target {target:.1f}%."
        elif memory_headroom_pct < 10.0:
            # Memory-constrained: cannot increase batch size safely
            recommended_batch_size = current_batch_size
            action = "maintain"
            justification = (
                f"GPU memory is {memory_used_pct:.1f}% utilized — insufficient headroom to increase batch size. "
                "Consider model quantization to reduce memory footprint."
            )
        else:
            # Scale batch size by the ratio of target to current utilization
            scale_factor = min(target / max(avg_gpu_utilization_pct, 1.0), memory_headroom_pct / 20.0)
            scale_factor = min(scale_factor, 4.0)  # cap at 4x increase
            recommended_batch_size = max(1, round(current_batch_size * scale_factor))
            action = "increase"
            justification = (
                f"GPU utilization is low ({avg_gpu_utilization_pct:.1f}%). "
                f"Increasing batch size from {current_batch_size} to {recommended_batch_size} "
                f"should bring utilization closer to target {target:.1f}%."
            )

        result: dict[str, Any] = {
            "model_id": model_id,
            "current_batch_size": current_batch_size,
            "recommended_batch_size": recommended_batch_size,
            "action": action,
            "justification": justification,
            "avg_gpu_utilization_pct": avg_gpu_utilization_pct,
            "memory_used_pct": memory_used_pct,
            "memory_headroom_pct": memory_headroom_pct,
            "estimated_throughput_gain_pct": round(
                max(0.0, (recommended_batch_size / max(current_batch_size, 1) - 1.0) * 100), 1
            ),
        }
        logger.info(
            "Batch size optimization computed",
            model_id=model_id,
            current=current_batch_size,
            recommended=recommended_batch_size,
            action=action,
        )
        return result

    async def configure_model_parallelism(
        self,
        model_id: str,
        model_size_billions: float,
        available_gpus: int,
        latency_sensitive: bool = False,
    ) -> dict[str, Any]:
        """Recommend a model parallelism strategy for a given model and GPU pool.

        Args:
            model_id: Model identifier.
            model_size_billions: Model parameter count in billions.
            available_gpus: Number of GPUs available on the node/cluster.
            latency_sensitive: When True, prefer data parallelism to minimise latency.

        Returns:
            Parallelism configuration dict with recommended_strategy and configuration.
        """
        if latency_sensitive or available_gpus == 1:
            strategy = "data_parallel"
        elif model_size_billions >= 70 and available_gpus >= 4:
            strategy = "mixed_parallel"
        elif model_size_billions >= 30 and available_gpus >= 2:
            strategy = "pipeline_parallel"
        elif model_size_billions >= 13 and available_gpus >= 2:
            strategy = "tensor_parallel"
        else:
            strategy = "data_parallel"

        strategy_config = PARALLELISM_STRATEGIES.get(strategy, {})
        tensor_parallel_size = min(available_gpus, 8) if strategy in ("tensor_parallel", "mixed_parallel") else 1
        pipeline_stages = min(available_gpus // 2, 4) if strategy in ("pipeline_parallel", "mixed_parallel") else 1

        config: dict[str, Any] = {
            "model_id": model_id,
            "model_size_billions": model_size_billions,
            "available_gpus": available_gpus,
            "recommended_strategy": strategy,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_stages": pipeline_stages,
            "data_parallel_replicas": max(1, available_gpus // max(tensor_parallel_size, 1)),
            "latency_sensitive": latency_sensitive,
            "estimated_memory_per_gpu_gb": round(
                (model_size_billions * 2.0) / max(tensor_parallel_size, 1), 1
            ),
            "strategy_rationale": (
                f"Model size {model_size_billions}B with {available_gpus} GPUs → "
                f"{strategy} recommended."
            ),
        }
        logger.info(
            "Model parallelism configured",
            model_id=model_id,
            strategy=strategy,
            available_gpus=available_gpus,
            model_size_billions=model_size_billions,
        )
        return config

    async def detect_idle_gpus(
        self,
        cluster_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Identify idle GPUs across a cluster for consolidation.

        Args:
            cluster_metrics: List of node metric dicts, each containing
                node_id, gpu_metrics (list of per-GPU metrics).

        Returns:
            Idle GPU report dict with idle_nodes, idle_gpu_count, and savings_estimate.
        """
        idle_nodes: list[dict[str, Any]] = []
        total_idle_gpus = 0

        for node in cluster_metrics:
            node_id = node.get("node_id", "unknown")
            gpus = node.get("gpu_metrics", [])

            idle_on_node = [
                g for g in gpus
                if g.get("utilization_pct", 0.0) < self._gpu_idle_threshold
            ]
            if idle_on_node:
                idle_nodes.append({
                    "node_id": node_id,
                    "idle_gpu_count": len(idle_on_node),
                    "total_gpu_count": len(gpus),
                    "idle_gpus": [g.get("gpu_index") for g in idle_on_node],
                    "avg_power_watts": sum(g.get("power_draw_watts", 300) for g in idle_on_node) / len(idle_on_node),
                })
                total_idle_gpus += len(idle_on_node)

        # Rough energy savings: idle GPU still draws ~50W at minimum
        idle_power_kw = total_idle_gpus * 0.05  # 50W average idle draw
        annual_kwh_waste = idle_power_kw * 8760
        co2_savings_kg_annual = annual_kwh_waste * 0.35 / 1000  # avg 350 gCO2/kWh

        result: dict[str, Any] = {
            "total_idle_gpus": total_idle_gpus,
            "idle_nodes": idle_nodes,
            "idle_power_kw": round(idle_power_kw, 3),
            "annual_energy_waste_kwh": round(annual_kwh_waste, 2),
            "annual_co2_savings_if_powered_down_kg": round(co2_savings_kg_annual, 2),
            "recommendation": (
                f"Power down or suspend {total_idle_gpus} idle GPU(s) to save "
                f"~{annual_kwh_waste:.0f} kWh/year."
            ) if total_idle_gpus > 0 else "No idle GPUs detected.",
            "detected_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        logger.info(
            "Idle GPU detection complete",
            total_idle_gpus=total_idle_gpus,
            idle_node_count=len(idle_nodes),
        )
        return result

    async def recommend_workload_consolidation(
        self,
        workload_profiles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Recommend workload consolidation to maximise GPU packing efficiency.

        Analyses workload GPU requirements and suggests co-location opportunities
        that would increase utilization without exceeding memory limits.

        Args:
            workload_profiles: List of workload dicts, each containing:
                workload_id, model_id, gpu_memory_required_gb, current_utilization_pct,
                avg_batch_size, workload_type.

        Returns:
            List of consolidation recommendation dicts.
        """
        recommendations: list[dict[str, Any]] = []
        gpu_vram_gb = 80.0  # A100 80GB default

        # Group workloads by utilization — low-util workloads are consolidation candidates
        low_util = [
            w for w in workload_profiles
            if w.get("current_utilization_pct", 100.0) < GPU_LOW_UTIL_THRESHOLD_PCT
        ]

        # Try to pair complementary workloads
        paired: set[str] = set()
        for i, w1 in enumerate(low_util):
            if w1["workload_id"] in paired:
                continue
            for w2 in low_util[i + 1:]:
                if w2["workload_id"] in paired:
                    continue
                combined_memory = (
                    w1.get("gpu_memory_required_gb", 0) + w2.get("gpu_memory_required_gb", 0)
                )
                combined_util = (
                    w1.get("current_utilization_pct", 0) + w2.get("current_utilization_pct", 0)
                )
                if combined_memory <= gpu_vram_gb * 0.9 and combined_util <= 95.0:
                    recommendations.append({
                        "recommendation_id": str(uuid.uuid4()),
                        "type": "co_locate",
                        "workload_a": w1["workload_id"],
                        "workload_b": w2["workload_id"],
                        "combined_memory_gb": round(combined_memory, 1),
                        "combined_utilization_pct": round(combined_util, 1),
                        "gpus_freed": 1,
                        "rationale": (
                            f"Workloads {w1['workload_id']} ({w1.get('current_utilization_pct')}% util) "
                            f"and {w2['workload_id']} ({w2.get('current_utilization_pct')}% util) "
                            f"can share one A100 (combined {combined_memory:.1f}GB / {combined_util:.0f}%)."
                        ),
                    })
                    paired.add(w1["workload_id"])
                    paired.add(w2["workload_id"])
                    break

        logger.info(
            "Workload consolidation recommendations generated",
            total_workloads=len(workload_profiles),
            consolidation_opportunities=len(recommendations),
            gpus_freed=sum(r["gpus_freed"] for r in recommendations),
        )
        return recommendations

    async def compute_efficiency_score(
        self,
        gpu_utilization_pct: float,
        memory_utilization_pct: float,
        batch_efficiency_pct: float,
        power_efficiency_pct: float,
    ) -> dict[str, Any]:
        """Compute a composite GPU efficiency score across multiple dimensions.

        Weights: GPU utilization 40%, memory utilization 25%, batch efficiency 25%, power 10%.

        Args:
            gpu_utilization_pct: Average GPU compute utilization (0–100).
            memory_utilization_pct: GPU VRAM utilization (0–100).
            batch_efficiency_pct: Actual batch size vs. optimal batch size ratio (0–100).
            power_efficiency_pct: FLOPS per watt ratio vs. TDP baseline (0–100).

        Returns:
            Efficiency score dict with composite_score, component_scores, and tier.
        """
        component_scores = {
            "gpu_utilization": round(gpu_utilization_pct, 2),
            "memory_utilization": round(memory_utilization_pct, 2),
            "batch_efficiency": round(batch_efficiency_pct, 2),
            "power_efficiency": round(power_efficiency_pct, 2),
        }
        composite = (
            0.40 * gpu_utilization_pct
            + 0.25 * memory_utilization_pct
            + 0.25 * batch_efficiency_pct
            + 0.10 * power_efficiency_pct
        )
        composite = round(composite, 2)

        if composite >= 80:
            tier = "excellent"
        elif composite >= 60:
            tier = "good"
        elif composite >= 40:
            tier = "moderate"
        else:
            tier = "poor"

        return {
            "composite_score": composite,
            "tier": tier,
            "component_scores": component_scores,
            "recommendation": (
                f"Efficiency tier: {tier} ({composite:.1f}/100). "
                + ("No immediate action required." if tier == "excellent" else
                   "Focus on improving the lowest-scoring component.")
            ),
        }

    async def generate_efficiency_report(
        self,
        tenant_id: str,
        node_ids: list[str],
    ) -> dict[str, Any]:
        """Generate a GPU efficiency report for specified nodes.

        Args:
            tenant_id: Tenant UUID string.
            node_ids: List of node IDs to include in the report.

        Returns:
            Efficiency report dict with per-node summaries and overall score.
        """
        report_id = str(uuid.uuid4())
        relevant_snapshots = [
            s for s in self._utilization_history
            if s.get("node_id") in node_ids
        ]

        if not relevant_snapshots:
            return {
                "report_id": report_id,
                "tenant_id": tenant_id,
                "node_ids": node_ids,
                "message": "No utilization data available for specified nodes.",
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            }

        avg_util = sum(s["average_utilization_pct"] for s in relevant_snapshots) / len(relevant_snapshots)
        total_idle_gpus = sum(s.get("idle_gpu_count", 0) for s in relevant_snapshots)

        report: dict[str, Any] = {
            "report_id": report_id,
            "tenant_id": tenant_id,
            "node_ids": node_ids,
            "snapshot_count": len(relevant_snapshots),
            "average_gpu_utilization_pct": round(avg_util, 2),
            "total_idle_gpu_snapshots": total_idle_gpus,
            "per_node_summaries": [
                {
                    "node_id": s["node_id"],
                    "average_utilization_pct": s["average_utilization_pct"],
                    "efficiency_score": s.get("efficiency_score", 0.0),
                    "idle_gpus": s.get("idle_gpu_count", 0),
                }
                for s in relevant_snapshots
            ],
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._optimization_reports.append(report)
        return report


__all__ = ["EfficiencyOptimizer"]
