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


class XprofDeviceMode(ParamEnum):
  """Device aggregation mode when extracting XProf device timings."""

  FIRST_DEVICE = "first_device"
  MAX_DEVICE = "max_device"


HBM_PEAK_TEMP_C = "hbm_peak_temp_c"
PEAK_TEMP_C = "peak_temp_c"
PEAK_TEMP_SOURCE = "peak_temp_source"
HBM_THROTTLE_TIME_PCT = "hbm_throttle_time_pct"
HBM_THROTTLE_MEAN_PCT = "hbm_throttle_mean_pct"
VDD_CORE_THROTTLE_TIME_PCT = "vdd_core_throttle_time_pct"
VDD_CORE_THROTTLE_MEAN_PCT = "vdd_core_throttle_mean_pct"
VDD_CORE_POWER_MEAN_W = "vdd_core_power_mean_w"
HBM_POWER_MEAN_W = "hbm_power_mean_w"
TOTAL_POWER_MEAN_W = "total_power_mean_w"
PSTATE_MIN = "pstate_min"
PSTATE_MAX = "pstate_max"
PSTATE_CHANGES = "pstate_changes"

THERMAL_METRIC_KEYS: tuple[str, ...] = (
    HBM_PEAK_TEMP_C,
    PEAK_TEMP_C,
    PEAK_TEMP_SOURCE,
    HBM_THROTTLE_TIME_PCT,
    HBM_THROTTLE_MEAN_PCT,
    VDD_CORE_THROTTLE_TIME_PCT,
    VDD_CORE_THROTTLE_MEAN_PCT,
    VDD_CORE_POWER_MEAN_W,
    HBM_POWER_MEAN_W,
    TOTAL_POWER_MEAN_W,
    PSTATE_MIN,
    PSTATE_MAX,
    PSTATE_CHANGES,
)

PEAK_TEMP_SOURCE_COMPUTE_DIE = "compute_die"
PEAK_TEMP_SOURCE_HBM = "hbm"

HBM_PEAK_TEMP_C_PRE_SOAKING = f"{HBM_PEAK_TEMP_C}_pre_soaking"
PEAK_TEMP_C_PRE_SOAKING = f"{PEAK_TEMP_C}_pre_soaking"

PSTATE_REQUESTED = "pstate_requested"
LIBTPU_DVFS_P_STATE_FLAG = "--xla_tpu_dvfs_p_state"
TPU_POWER_TRACE_LEVEL = 0

THERMAL_PER_DEVICE = "thermal_per_device"
THERMAL_PER_DEVICE_PRE_SOAKING = "thermal_per_device_pre_soaking"
THERMAL_PRE_SOAKING = "thermal_pre_soaking"

PRE_SOAKING = "pre_soaking"
POST_SOAKING = "post_soaking"
PRE_SOAKING_P50_MS = f"{PRE_SOAKING}_p50_ms"
POST_SOAKING_P50_MS = f"{POST_SOAKING}_p50_ms"
SLOWDOWN_RATIO = "slowdown_ratio"

SOAKING_LATENCY_SUFFIXES: tuple[str, ...] = (
    PRE_SOAKING_P50_MS,
    POST_SOAKING_P50_MS,
    SLOWDOWN_RATIO,
)

WALL_CLOCK_PRE_SOAKING_P50_MS = (
    f"{TimingDomain.WALL_CLOCK}_{PRE_SOAKING_P50_MS}"
)
WALL_CLOCK_POST_SOAKING_P50_MS = (
    f"{TimingDomain.WALL_CLOCK}_{POST_SOAKING_P50_MS}"
)
WALL_CLOCK_SLOWDOWN_RATIO = f"{TimingDomain.WALL_CLOCK}_{SLOWDOWN_RATIO}"

XPROF_PRE_SOAKING_P50_MS = f"{TimingDomain.XPROF}_{PRE_SOAKING_P50_MS}"
XPROF_POST_SOAKING_P50_MS = f"{TimingDomain.XPROF}_{POST_SOAKING_P50_MS}"
XPROF_SLOWDOWN_RATIO = f"{TimingDomain.XPROF}_{SLOWDOWN_RATIO}"

SOAKING_METRIC_KEYS: tuple[str, ...] = (
    WALL_CLOCK_PRE_SOAKING_P50_MS,
    WALL_CLOCK_POST_SOAKING_P50_MS,
    WALL_CLOCK_SLOWDOWN_RATIO,
    XPROF_PRE_SOAKING_P50_MS,
    XPROF_POST_SOAKING_P50_MS,
    XPROF_SLOWDOWN_RATIO,
)

WALL_CLOCK_PRE_SOAKING_TFLOPS_PER_DEVICE = (
    f"{TimingDomain.WALL_CLOCK}_{PRE_SOAKING}_tflops_per_device"
)
WALL_CLOCK_POST_SOAKING_TFLOPS_PER_DEVICE = (
    f"{TimingDomain.WALL_CLOCK}_{POST_SOAKING}_tflops_per_device"
)
XPROF_PRE_SOAKING_TFLOPS_PER_DEVICE = (
    f"{TimingDomain.XPROF}_{PRE_SOAKING}_tflops_per_device"
)
XPROF_POST_SOAKING_TFLOPS_PER_DEVICE = (
    f"{TimingDomain.XPROF}_{POST_SOAKING}_tflops_per_device"
)

XPROF_HBM_PEAK_TEMP_STAT = "hbm fw max temperature(c)"
XPROF_COMPUTE_DIE_PEAK_TEMP_STAT = "compute die fw max temperature(c)"
XPROF_HBM_THROTTLE_PCT_STAT = "hbm fw throttle(%)"
XPROF_VDD_CORE_THROTTLE_PCT_STAT = "vdd core fw throttle(%)"
XPROF_VDD_CORE_POWER_PL1_STAT = "vdd core fw power meter pl1(w)"
XPROF_HBM_POWER_PL1_STAT = "hbm fw power meter pl1(w)"
XPROF_PSTATE_EVENT = "p state"
XPROF_XLA_OPS_LINE = "xla ops"
THERMAL_PRE_SOAKING_METRIC_KEYS: tuple[str, ...] = (
    HBM_PEAK_TEMP_C_PRE_SOAKING,
    PEAK_TEMP_C_PRE_SOAKING,
)
