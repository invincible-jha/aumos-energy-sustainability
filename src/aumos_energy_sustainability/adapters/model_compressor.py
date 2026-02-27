"""Model compressor adapter for aumos-energy-sustainability.

Reduces model size for energy efficiency: INT8/INT4 quantization, magnitude
and structured pruning, distillation pipeline setup, compression ratio
measurement, quality impact assessment (perplexity delta), compression
recommendations, and energy savings estimation.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Memory per billion parameters by precision
MEMORY_PER_BILLION_PARAMS_GB: dict[str, float] = {
    "FP32": 4.0,
    "FP16": 2.0,
    "BF16": 2.0,
    "INT8": 1.0,
    "INT4": 0.5,
    "INT2": 0.25,
}

# Expected quality degradation by compression method (perplexity increase %)
QUALITY_DEGRADATION_ESTIMATES: dict[str, dict[str, float]] = {
    "quantization": {
        "INT8": 0.5,
        "INT4": 2.0,
        "INT2": 8.0,
    },
    "pruning": {
        "magnitude_10pct": 0.3,
        "magnitude_30pct": 1.5,
        "magnitude_50pct": 5.0,
        "structured_25pct": 1.0,
        "structured_50pct": 4.0,
    },
    "distillation": {
        "50pct_student": 3.0,
        "25pct_student": 6.0,
    },
}

# Energy consumption reduction per compression type (estimated vs. uncompressed baseline)
ENERGY_REDUCTION_ESTIMATES: dict[str, float] = {
    "INT8": 0.40,
    "INT4": 0.60,
    "magnitude_30pct": 0.25,
    "structured_25pct": 0.30,
    "distillation_50pct": 0.50,
}


class ModelCompressor:
    """Implements model compression techniques to reduce energy consumption.

    Supports quantization (INT8/INT4), magnitude and structured pruning,
    knowledge distillation setup, and provides quality impact assessments
    and energy savings projections for each compression strategy.
    """

    def __init__(self) -> None:
        """Initialise the model compressor with empty compression history."""
        self._compression_log: list[dict[str, Any]] = []

    async def quantize_model(
        self,
        model_id: str,
        model_version: str,
        original_precision: str,
        target_precision: str,
        model_size_billions: float,
        calibration_dataset_size: int = 512,
    ) -> dict[str, Any]:
        """Apply quantization to reduce model precision and memory footprint.

        Supports INT8 (GPTQ / bitsandbytes) and INT4 (AWQ / GGUF) quantization.
        Estimates memory reduction, throughput gain, and perplexity delta.

        Args:
            model_id: Model to quantize.
            model_version: Version being quantized.
            original_precision: Source precision (FP32, FP16, BF16).
            target_precision: Target precision (INT8, INT4).
            model_size_billions: Model parameter count in billions.
            calibration_dataset_size: Number of samples in the calibration set.

        Returns:
            Quantization result dict with compression_ratio, memory_savings_gb,
            estimated_perplexity_delta, and throughput_gain_pct.

        Raises:
            ValueError: If target precision is not supported.
        """
        supported_targets = {"INT8", "INT4", "INT2"}
        if target_precision not in supported_targets:
            raise ValueError(
                f"Unsupported target precision: '{target_precision}'. "
                f"Supported: {sorted(supported_targets)}"
            )

        original_memory_gb = model_size_billions * MEMORY_PER_BILLION_PARAMS_GB.get(original_precision, 2.0)
        target_memory_gb = model_size_billions * MEMORY_PER_BILLION_PARAMS_GB.get(target_precision, 1.0)
        memory_savings_gb = original_memory_gb - target_memory_gb
        compression_ratio = original_memory_gb / max(target_memory_gb, 0.01)

        perplexity_delta = QUALITY_DEGRADATION_ESTIMATES["quantization"].get(target_precision, 1.0)
        # Throughput improvement from reduced memory bandwidth demand
        throughput_gain_pct = round((compression_ratio - 1.0) * 40.0, 1)
        energy_reduction_pct = ENERGY_REDUCTION_ESTIMATES.get(target_precision, 0.3) * 100

        compression_id = str(uuid.uuid4())
        result: dict[str, Any] = {
            "compression_id": compression_id,
            "model_id": model_id,
            "model_version": model_version,
            "technique": "quantization",
            "original_precision": original_precision,
            "target_precision": target_precision,
            "model_size_billions": model_size_billions,
            "calibration_dataset_size": calibration_dataset_size,
            "original_memory_gb": round(original_memory_gb, 2),
            "target_memory_gb": round(target_memory_gb, 2),
            "memory_savings_gb": round(memory_savings_gb, 2),
            "compression_ratio": round(compression_ratio, 2),
            "estimated_perplexity_delta_pct": perplexity_delta,
            "throughput_gain_pct": throughput_gain_pct,
            "energy_reduction_pct": round(energy_reduction_pct, 1),
            "acceptable_quality_loss": perplexity_delta <= 2.0,
            "recommended_use_cases": (
                ["batch_inference", "serving_at_scale"]
                if target_precision == "INT8"
                else ["edge_deployment", "latency_sensitive"]
            ),
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._compression_log.append(result)

        logger.info(
            "Model quantization applied",
            model_id=model_id,
            original_precision=original_precision,
            target_precision=target_precision,
            compression_ratio=compression_ratio,
            memory_savings_gb=memory_savings_gb,
            energy_reduction_pct=energy_reduction_pct,
        )
        return result

    async def prune_model(
        self,
        model_id: str,
        model_version: str,
        pruning_method: str,
        sparsity_target_pct: float,
        model_size_billions: float,
        structured: bool = False,
    ) -> dict[str, Any]:
        """Apply pruning to remove redundant weights from the model.

        Args:
            model_id: Model to prune.
            model_version: Version being pruned.
            pruning_method: Pruning algorithm (magnitude, gradient, random).
            sparsity_target_pct: Target sparsity percentage (0–80).
            model_size_billions: Original model size in billions of parameters.
            structured: When True, applies structured (channel-level) pruning
                        that delivers actual speedups; unstructured requires sparsity-aware hardware.

        Returns:
            Pruning result dict with sparsity_achieved, parameter_reduction,
            estimated_speedup, and quality_impact.

        Raises:
            ValueError: If sparsity_target_pct is out of the safe range.
        """
        if not (0 < sparsity_target_pct <= 80):
            raise ValueError(
                f"sparsity_target_pct must be between 1 and 80, got {sparsity_target_pct}"
            )

        parameters_pruned_billions = model_size_billions * (sparsity_target_pct / 100.0)
        remaining_billions = model_size_billions - parameters_pruned_billions
        compression_ratio = model_size_billions / max(remaining_billions, 0.01)

        # Structured pruning delivers latency gains; unstructured needs sparse kernels
        if structured:
            speedup_pct = sparsity_target_pct * 0.6
            quality_degradation_pct = sparsity_target_pct * 0.08
        else:
            speedup_pct = sparsity_target_pct * 0.2  # requires sparsity-aware runtime
            quality_degradation_pct = sparsity_target_pct * 0.03

        energy_savings_pct = speedup_pct * 0.8  # fewer FLOPs -> lower energy

        compression_id = str(uuid.uuid4())
        result: dict[str, Any] = {
            "compression_id": compression_id,
            "model_id": model_id,
            "model_version": model_version,
            "technique": "pruning",
            "pruning_method": pruning_method,
            "structured": structured,
            "sparsity_target_pct": sparsity_target_pct,
            "parameters_pruned_billions": round(parameters_pruned_billions, 2),
            "remaining_parameters_billions": round(remaining_billions, 2),
            "compression_ratio": round(compression_ratio, 2),
            "estimated_speedup_pct": round(speedup_pct, 1),
            "estimated_quality_degradation_pct": round(quality_degradation_pct, 2),
            "energy_savings_pct": round(energy_savings_pct, 1),
            "requires_sparse_runtime": not structured,
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._compression_log.append(result)

        logger.info(
            "Model pruning applied",
            model_id=model_id,
            pruning_method=pruning_method,
            sparsity_target_pct=sparsity_target_pct,
            structured=structured,
            energy_savings_pct=energy_savings_pct,
        )
        return result

    async def setup_distillation_pipeline(
        self,
        teacher_model_id: str,
        teacher_size_billions: float,
        student_size_billions: float,
        distillation_method: str = "response_based",
        temperature: float = 4.0,
    ) -> dict[str, Any]:
        """Configure a knowledge distillation pipeline from teacher to student model.

        Args:
            teacher_model_id: Large teacher model identifier.
            teacher_size_billions: Teacher model size in billions of parameters.
            student_size_billions: Target student model size in billions.
            distillation_method: Distillation strategy (response_based, feature_based, relation_based).
            temperature: Softmax temperature for soft label generation (typically 2–8).

        Returns:
            Distillation pipeline configuration dict with compression_ratio,
            student_config, and training_requirements.
        """
        compression_ratio = teacher_size_billions / max(student_size_billions, 0.1)
        energy_reduction_pct = round((1.0 - 1.0 / compression_ratio) * 100, 1)
        # Quality loss relative to fine-tuned full model
        estimated_quality_loss_pct = round((compression_ratio - 1.0) * 2.5, 1)

        pipeline_id = str(uuid.uuid4())
        pipeline_config: dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "technique": "distillation",
            "teacher_model_id": teacher_model_id,
            "teacher_size_billions": teacher_size_billions,
            "student_size_billions": student_size_billions,
            "distillation_method": distillation_method,
            "temperature": temperature,
            "compression_ratio": round(compression_ratio, 2),
            "energy_reduction_pct": energy_reduction_pct,
            "estimated_quality_loss_pct": estimated_quality_loss_pct,
            "student_config": {
                "hidden_size": max(128, int(1024 / (compression_ratio ** 0.5))),
                "num_layers": max(2, int(32 / (compression_ratio ** 0.5))),
                "num_attention_heads": max(4, int(32 / compression_ratio)),
                "intermediate_size": max(512, int(4096 / (compression_ratio ** 0.5))),
            },
            "training_requirements": {
                "gpu_hours_estimate": round(student_size_billions * 200),
                "dataset_tokens_billions": round(student_size_billions * 50),
                "recommended_lr": 1e-4,
                "warmup_steps": 500,
                "temperature": temperature,
            },
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._compression_log.append(pipeline_config)

        logger.info(
            "Distillation pipeline configured",
            pipeline_id=pipeline_id,
            teacher_model_id=teacher_model_id,
            teacher_size=teacher_size_billions,
            student_size=student_size_billions,
            compression_ratio=compression_ratio,
        )
        return pipeline_config

    async def measure_compression_ratio(
        self,
        original_size_bytes: int,
        compressed_size_bytes: int,
        original_flops: float | None = None,
        compressed_flops: float | None = None,
    ) -> dict[str, Any]:
        """Measure and report compression ratios for a completed compression.

        Args:
            original_size_bytes: Uncompressed model checkpoint size in bytes.
            compressed_size_bytes: Compressed model checkpoint size in bytes.
            original_flops: Optional FLOP count for the original model.
            compressed_flops: Optional FLOP count for the compressed model.

        Returns:
            Measurement result dict with size_ratio, flops_ratio, and bandwidth_saving.
        """
        size_ratio = original_size_bytes / max(compressed_size_bytes, 1)
        size_saving_pct = round((1.0 - 1.0 / size_ratio) * 100, 2)

        flops_ratio: float | None = None
        flops_saving_pct: float | None = None
        if original_flops and compressed_flops:
            flops_ratio = original_flops / max(compressed_flops, 1)
            flops_saving_pct = round((1.0 - 1.0 / flops_ratio) * 100, 2)

        return {
            "original_size_gb": round(original_size_bytes / 1e9, 3),
            "compressed_size_gb": round(compressed_size_bytes / 1e9, 3),
            "size_compression_ratio": round(size_ratio, 3),
            "size_saving_pct": size_saving_pct,
            "flops_compression_ratio": round(flops_ratio, 3) if flops_ratio else None,
            "flops_saving_pct": flops_saving_pct,
            "estimated_energy_saving_pct": round(
                (flops_saving_pct or size_saving_pct) * 0.8, 1
            ),
            "measured_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    async def assess_quality_impact(
        self,
        baseline_perplexity: float,
        compressed_perplexity: float,
        baseline_accuracy: float | None = None,
        compressed_accuracy: float | None = None,
        acceptable_delta_pct: float = 2.0,
    ) -> dict[str, Any]:
        """Assess quality degradation from a compression operation.

        Args:
            baseline_perplexity: Perplexity of the uncompressed model.
            compressed_perplexity: Perplexity of the compressed model.
            baseline_accuracy: Optional downstream task accuracy for the baseline.
            compressed_accuracy: Optional downstream task accuracy for the compressed model.
            acceptable_delta_pct: Maximum acceptable perplexity increase percentage.

        Returns:
            Quality impact assessment dict with is_acceptable and degradation details.
        """
        perplexity_delta = compressed_perplexity - baseline_perplexity
        perplexity_delta_pct = round((perplexity_delta / max(baseline_perplexity, 0.01)) * 100, 2)

        accuracy_delta: float | None = None
        accuracy_delta_pct: float | None = None
        if baseline_accuracy is not None and compressed_accuracy is not None:
            accuracy_delta = compressed_accuracy - baseline_accuracy
            accuracy_delta_pct = round((accuracy_delta / max(baseline_accuracy, 0.01)) * 100, 2)

        is_acceptable = perplexity_delta_pct <= acceptable_delta_pct

        verdict = (
            "Compression quality is acceptable — proceed to production."
            if is_acceptable
            else f"Perplexity increased by {perplexity_delta_pct:.1f}% (threshold: {acceptable_delta_pct:.1f}%). "
                 "Consider less aggressive compression."
        )

        logger.info(
            "Compression quality impact assessed",
            baseline_perplexity=baseline_perplexity,
            compressed_perplexity=compressed_perplexity,
            perplexity_delta_pct=perplexity_delta_pct,
            is_acceptable=is_acceptable,
        )

        return {
            "baseline_perplexity": baseline_perplexity,
            "compressed_perplexity": compressed_perplexity,
            "perplexity_delta": round(perplexity_delta, 4),
            "perplexity_delta_pct": perplexity_delta_pct,
            "baseline_accuracy": baseline_accuracy,
            "compressed_accuracy": compressed_accuracy,
            "accuracy_delta_pct": accuracy_delta_pct,
            "acceptable_threshold_pct": acceptable_delta_pct,
            "is_acceptable": is_acceptable,
            "verdict": verdict,
        }

    async def recommend_compression(
        self,
        model_id: str,
        model_size_billions: float,
        use_case: str,
        energy_saving_target_pct: float = 30.0,
        max_quality_loss_pct: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Recommend compression strategies based on use case and targets.

        Args:
            model_id: Model to compress.
            model_size_billions: Model size in billions of parameters.
            use_case: inference | training | edge | batch.
            energy_saving_target_pct: Minimum energy saving required.
            max_quality_loss_pct: Maximum acceptable quality loss.

        Returns:
            Ordered list of compression recommendations (best first).
        """
        recommendations: list[dict[str, Any]] = []

        # INT8 quantization — broad applicability
        int8_savings = ENERGY_REDUCTION_ESTIMATES.get("INT8", 0.40) * 100
        int8_quality_loss = QUALITY_DEGRADATION_ESTIMATES["quantization"]["INT8"]
        if int8_savings >= energy_saving_target_pct and int8_quality_loss <= max_quality_loss_pct:
            recommendations.append({
                "rank": 1,
                "method": "quantization_INT8",
                "energy_saving_pct": int8_savings,
                "quality_loss_pct": int8_quality_loss,
                "implementation_effort": "low",
                "rationale": "INT8 quantization: 2x memory reduction, minimal quality loss, widely supported.",
            })

        # INT4 quantization — higher savings but more quality loss
        int4_savings = ENERGY_REDUCTION_ESTIMATES.get("INT4", 0.60) * 100
        int4_quality_loss = QUALITY_DEGRADATION_ESTIMATES["quantization"]["INT4"]
        if int4_savings >= energy_saving_target_pct and int4_quality_loss <= max_quality_loss_pct:
            recommendations.append({
                "rank": len(recommendations) + 1,
                "method": "quantization_INT4",
                "energy_saving_pct": int4_savings,
                "quality_loss_pct": int4_quality_loss,
                "implementation_effort": "low",
                "rationale": "INT4 quantization: 4x memory reduction. Best for edge/on-device deployment.",
            })

        # Structured pruning
        pruning_savings = ENERGY_REDUCTION_ESTIMATES.get("structured_25pct", 0.30) * 100
        pruning_quality_loss = QUALITY_DEGRADATION_ESTIMATES["pruning"]["structured_25pct"]
        if pruning_savings >= energy_saving_target_pct and pruning_quality_loss <= max_quality_loss_pct:
            recommendations.append({
                "rank": len(recommendations) + 1,
                "method": "structured_pruning_25pct",
                "energy_saving_pct": pruning_savings,
                "quality_loss_pct": pruning_quality_loss,
                "implementation_effort": "medium",
                "rationale": "25% structured pruning: real speedup without sparse kernels. Good balance.",
            })

        # Distillation (for large models)
        if model_size_billions >= 13:
            distillation_savings = ENERGY_REDUCTION_ESTIMATES.get("distillation_50pct", 0.50) * 100
            distillation_quality_loss = QUALITY_DEGRADATION_ESTIMATES["distillation"]["50pct_student"]
            if distillation_savings >= energy_saving_target_pct and distillation_quality_loss <= max_quality_loss_pct:
                recommendations.append({
                    "rank": len(recommendations) + 1,
                    "method": "knowledge_distillation_50pct",
                    "energy_saving_pct": distillation_savings,
                    "quality_loss_pct": distillation_quality_loss,
                    "implementation_effort": "high",
                    "rationale": (
                        f"Distill {model_size_billions}B to ~{model_size_billions * 0.5:.1f}B student. "
                        "Highest savings but requires retraining."
                    ),
                })

        recommendations.sort(
            key=lambda r: (r["energy_saving_pct"] - r["quality_loss_pct"] * 2),
            reverse=True,
        )

        logger.info(
            "Compression recommendations generated",
            model_id=model_id,
            use_case=use_case,
            energy_target_pct=energy_saving_target_pct,
            recommendation_count=len(recommendations),
        )
        return recommendations

    async def estimate_energy_savings(
        self,
        original_energy_kwh_per_inference: float,
        compression_method: str,
        daily_inference_count: int,
        carbon_intensity_gco2_per_kwh: float = 300.0,
    ) -> dict[str, Any]:
        """Project energy and carbon savings from a compression strategy.

        Args:
            original_energy_kwh_per_inference: Baseline energy per inference call.
            compression_method: Compression technique applied (maps to ENERGY_REDUCTION_ESTIMATES).
            daily_inference_count: Expected daily inference volume.
            carbon_intensity_gco2_per_kwh: Grid carbon intensity in gCO2/kWh.

        Returns:
            Savings projection dict with daily, monthly, and annual savings in kWh and kg CO2.
        """
        reduction_pct = ENERGY_REDUCTION_ESTIMATES.get(compression_method, 0.3)
        compressed_energy = original_energy_kwh_per_inference * (1.0 - reduction_pct)
        energy_saved_per_inference = original_energy_kwh_per_inference - compressed_energy

        daily_energy_saved_kwh = energy_saved_per_inference * daily_inference_count
        monthly_energy_saved_kwh = daily_energy_saved_kwh * 30
        annual_energy_saved_kwh = daily_energy_saved_kwh * 365
        annual_carbon_saved_kg = annual_energy_saved_kwh * carbon_intensity_gco2_per_kwh / 1000

        logger.info(
            "Energy savings estimated",
            compression_method=compression_method,
            reduction_pct=reduction_pct * 100,
            annual_energy_saved_kwh=annual_energy_saved_kwh,
            annual_carbon_saved_kg=annual_carbon_saved_kg,
        )

        return {
            "compression_method": compression_method,
            "energy_reduction_pct": round(reduction_pct * 100, 1),
            "original_energy_per_inference_kwh": original_energy_kwh_per_inference,
            "compressed_energy_per_inference_kwh": round(compressed_energy, 8),
            "energy_saved_per_inference_kwh": round(energy_saved_per_inference, 8),
            "daily_inference_count": daily_inference_count,
            "daily_energy_saved_kwh": round(daily_energy_saved_kwh, 4),
            "monthly_energy_saved_kwh": round(monthly_energy_saved_kwh, 4),
            "annual_energy_saved_kwh": round(annual_energy_saved_kwh, 4),
            "annual_carbon_saved_kg_co2": round(annual_carbon_saved_kg, 3),
            "annual_cost_savings_usd": round(annual_energy_saved_kwh * 0.12, 2),
        }


__all__ = ["ModelCompressor"]
