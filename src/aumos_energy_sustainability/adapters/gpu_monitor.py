"""NVIDIA GPU energy monitor using NVML/pynvml.

GAP-335: GPU-Specific Energy Measurement.
Measures actual GPU power draw during inference via NVIDIA NVML.
Falls back to per-model estimation when NVML is unavailable.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from decimal import Decimal
from typing import Iterator

from aumos_common.observability import get_logger

logger = get_logger(__name__)

try:
    import pynvml  # type: ignore[import]

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class NVIDIAGPUMonitor:
    """Real-time GPU power measurement using NVIDIA NVML.

    Falls back to estimation if NVML is unavailable (CPU-only or non-NVIDIA).
    Power is in watts; energy is computed as power × time.

    For CSRD/SEC disclosures, actual measurements are required when available.
    Without NVML, a conservative 200W estimate is used (middle of T4-A100 range).
    """

    # Conservative per-model power estimates in watts (used when NVML unavailable)
    GPU_POWER_ESTIMATES_W: dict[str, float] = {
        "A100": 400.0,
        "H100": 700.0,
        "V100": 300.0,
        "T4": 70.0,
        "RTX 4090": 450.0,
        "RTX 3090": 350.0,
        "RTX 3080": 320.0,
        "L40": 300.0,
        "A10": 150.0,
    }

    # Fallback when GPU model is unknown
    DEFAULT_ESTIMATE_W: float = 200.0

    def __init__(self) -> None:
        self._initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._initialized = True
                gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info("nvml_initialized", gpu_count=gpu_count)
            except pynvml.NVMLError as exc:
                logger.warning("nvml_init_failed", reason=str(exc))

    @contextmanager
    def measure_inference(self, gpu_index: int = 0) -> Iterator[dict]:
        """Context manager measuring GPU energy during inference.

        Usage::

            with monitor.measure_inference() as result:
                run_inference()
            energy_kwh = result["energy_kwh"]

        Args:
            gpu_index: Index of the GPU to measure (default 0).

        Yields:
            Dict populated with energy_kwh, avg_power_w, peak_power_w,
            duration_seconds, measurement_method after context exits.
        """
        result: dict = {}
        start_time = time.monotonic()
        power_readings: list[float] = []
        handle = None

        if self._initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                # Take an initial reading
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_readings.append(power_mw / 1000.0)
            except pynvml.NVMLError as exc:
                logger.warning("nvml_handle_error", gpu_index=gpu_index, reason=str(exc))

        yield result  # Inference code runs here

        duration = time.monotonic() - start_time

        # Take a final reading after inference
        if self._initialized and handle is not None:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_readings.append(power_mw / 1000.0)
            except pynvml.NVMLError:
                pass

        if power_readings:
            avg_power_w = sum(power_readings) / len(power_readings)
            peak_power_w = max(power_readings)
            method = "nvml"
        else:
            avg_power_w = self.DEFAULT_ESTIMATE_W
            peak_power_w = avg_power_w
            method = "estimate"
            logger.warning("nvml_power_fallback", using_watts=avg_power_w)

        # energy_kwh = avg_watts × hours / 1000
        energy_kwh = (avg_power_w * (duration / 3600)) / 1000

        result.update(
            {
                "energy_kwh": Decimal(str(round(energy_kwh, 8))),
                "avg_power_w": round(avg_power_w, 2),
                "peak_power_w": round(peak_power_w, 2),
                "duration_seconds": round(duration, 3),
                "measurement_method": method,
            }
        )

    def get_gpu_info(self, gpu_index: int = 0) -> dict:
        """Return GPU model name and current power limit.

        Args:
            gpu_index: GPU index (default 0).

        Returns:
            Dict with name, power_limit_w, driver_version. Empty dict if unavailable.
        """
        if not self._initialized:
            return {}
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()
            return {
                "name": name,
                "power_limit_w": power_limit_mw / 1000.0,
                "driver_version": driver,
                "gpu_index": gpu_index,
            }
        except pynvml.NVMLError as exc:
            logger.warning("nvml_get_info_failed", gpu_index=gpu_index, reason=str(exc))
            return {}

    def estimate_from_model_name(self, gpu_model: str) -> float:
        """Return estimated average power consumption for a known GPU model.

        Args:
            gpu_model: GPU model string (e.g., "A100", "T4").

        Returns:
            Estimated average watts. Falls back to DEFAULT_ESTIMATE_W.
        """
        for known_model, watts in self.GPU_POWER_ESTIMATES_W.items():
            if known_model.lower() in gpu_model.lower():
                return watts
        return self.DEFAULT_ESTIMATE_W

    def __del__(self) -> None:
        """Shutdown NVML on garbage collection."""
        if self._initialized and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
