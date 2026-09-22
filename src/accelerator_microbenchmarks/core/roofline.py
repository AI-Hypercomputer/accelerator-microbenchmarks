"""Roofline analysis for benchmark metrics."""

import logging
from typing import Any, TYPE_CHECKING

from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import system

if TYPE_CHECKING:
  from accelerator_microbenchmarks.core import base  # pylint: disable=g-import-not-at-top


def apply_roofline_analysis(
    benchmark_instance: "base.BaseBenchmark[Any]", metrics: dict[str, Any]
) -> dict[str, Any]:
  """Apply roofline estimation to finalized metrics.

  Note on Per-Device vs. Per-Chip Efficiency Invariance:
    Both `actual` throughput and hardware `peak`/`roofline` limits scale
    linearly with `hardware_spec.devices_per_chip`. Consequently, roofline
    efficiency percentages (`(actual / limit) * 100.0`) are mathematically
    identical at the per-device and per-chip levels:
      `(actual_per_device / limit_per_device) == (actual_per_chip /
      limit_per_chip)`
    Therefore, `<domain>_compute_roofline_efficiency_pct` and
    `<domain>_memory_roofline_efficiency_pct` apply invariantly to both scopes.

  Args:
    benchmark_instance: The benchmark instance being analyzed.
    metrics: Dictionary of finalized benchmark metrics to enrich in place.

  Returns:
    The enriched metrics dictionary with roofline limits and efficiency keys.

  Raises:
    KeyError: If the hardware spec is missing peak TFLOPS for the canonical
      fallback dtype.
  """
  mode = getattr(
      benchmark_instance, "roofline_mode", constants.RooflineMode.NONE
  )
  if mode == constants.RooflineMode.NONE:
    return metrics

  hw_spec = getattr(benchmark_instance, "hardware_spec", None)

  if mode == constants.RooflineMode.MEMORY_HBM:
    if hw_spec and hw_spec.hbm:
      bw = hw_spec.peak_hbm_bandwidth_per_device
      metrics["peak_hbm_bw_per_device_gb_s"] = bw
      for prefix in constants.TimingDomain:
        actual_bw = metrics.get(f"{prefix}_bandwidth_per_device_gb_s")
        if actual_bw is not None and actual_bw >= 0 and bw > 0:
          metrics[f"{prefix}_memory_roofline_efficiency_pct"] = (
              actual_bw / bw
          ) * 100.0
    return metrics

  if mode == constants.RooflineMode.COMPUTE:
    if hw_spec and hw_spec.tflops and hw_spec.hbm:
      dtype = benchmark_instance.get_compute_dtype()
      if dtype in hw_spec.tflops.peak_tflops_per_device:
        peak_tflops = hw_spec.tflops.peak_tflops_per_device[dtype]
      else:
        fallback_dtype = system.DEFAULT_FALLBACK_DTYPE
        if fallback_dtype not in hw_spec.tflops.peak_tflops_per_device:
          raise KeyError(
              f"HardwareSpec for '{hw_spec.name}' is invalid: missing peak"
              f" TFLOPS for canonical fallback dtype '{fallback_dtype}'."
          )
        peak_tflops = hw_spec.tflops.peak_tflops_per_device[fallback_dtype]
        logging.warning(
            "Peak TFLOPS setup is missing for dtype '%s' on %s. "
            "Falling back to canonical '%s' peak TFLOPS (%.1f TFLOPS).",
            dtype,
            hw_spec.name,
            fallback_dtype,
            peak_tflops,
        )

      intensity = benchmark_instance.get_arithmetic_intensity()
      bw = hw_spec.peak_hbm_bandwidth_per_device
      roofline_tflops = min(peak_tflops, (intensity * bw) / 1000.0)
      metrics["roofline_tflops_limit_per_device"] = roofline_tflops
      metrics["peak_hbm_bw_per_device_gb_s"] = bw
      for prefix in constants.TimingDomain:
        actual_tflops = metrics.get(f"{prefix}_tflops_per_device")
        if (
            actual_tflops is not None
            and actual_tflops >= 0
            and roofline_tflops > 0
        ):
          metrics[f"{prefix}_compute_roofline_efficiency_pct"] = (
              actual_tflops / roofline_tflops
          ) * 100.0
    return metrics

  return metrics
