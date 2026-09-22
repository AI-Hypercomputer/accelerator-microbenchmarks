"""Shared constants for JAX benchmarks."""

import enum

MARKER = "MARKER!!!"


class ParamEnum(enum.StrEnum):
  """Base class for string-based enums with formatting utilities."""

  @classmethod
  def supported_options_str(cls) -> str:
    """Returns a comma-separated string of all enum values."""
    return ", ".join(cls)


class RooflineMode(str, enum.Enum):
  """Roofline calculation mode for microbenchmarks.

  Attributes:
    NONE: No roofline analysis performed; metrics are returned in original form.
    COMPUTE: Compute-bound roofline analysis modeling arithmetic intensity and
      theoretical peak compute (TFLOPS) ceiling. Emits
      `roofline_tflops_limit_per_device`, `peak_hbm_bw_per_device_gb_s`, and
      `<domain>_compute_roofline_efficiency_pct`.
    MEMORY_HBM: Memory-bound roofline analysis evaluating HBM bandwidth against
      theoretical peak hardware bandwidth. Emits `peak_hbm_bw_per_device_gb_s`
      and `<domain>_memory_roofline_efficiency_pct`. Note that smaller transfer
      sizes will report lower efficiency by construction due to fixed launch
      latency and bandwidth ramp; this is expected behavior rather than a
      performance regression.
  """

  NONE = "none"
  COMPUTE = "compute"
  MEMORY_HBM = "memory_hbm"


class TimingDomain(enum.StrEnum):
  """Canonical timing measurement domains for benchmark metrics."""

  WALL_CLOCK = "wall_clock"
  XPROF = "xprof"
