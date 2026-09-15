"""Shared constants for JAX benchmarks."""

import enum

MARKER = "MARKER!!!"


class RooflineMode(str, enum.Enum):
  """Roofline calculation mode for microbenchmarks.

  Attributes:
    NONE: No roofline analysis performed; metrics are returned in original form.
    COMPUTE: Compute-bound roofline analysis modeling arithmetic intensity and
      theoretical peak compute (TFLOPS) ceiling. Emits `roofline_tflops_limit`,
      `peak_hbm_bw_gb_s`, and `compute_roofline_efficiency_pct`.
    MEMORY_HBM: Memory-bound roofline analysis evaluating HBM bandwidth against
      theoretical peak hardware bandwidth. Emits `peak_hbm_bw_gb_s` and
      `memory_roofline_efficiency_pct`. Note that smaller transfer sizes will
      report lower efficiency by construction due to fixed launch latency and
      bandwidth ramp; this is expected behavior rather than a performance
      regression.
  """

  NONE = "none"
  COMPUTE = "compute"
  MEMORY_HBM = "memory_hbm"
