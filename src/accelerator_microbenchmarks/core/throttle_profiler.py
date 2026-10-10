"""Thermal telemetry and throttling XPlane profiler for JAX benchmarks."""

import collections
from collections.abc import Callable, Iterable, Mapping
import os
import re
from typing import Any

from accelerator_microbenchmarks.core import constants

import os
from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import

_DEVICE_PLANE_RE = re.compile(r"^/device:TPU:\d+$")
_CORE_ID_SUFFIX_RE = re.compile(r"\s+\d+$")
_PS_PER_NS = 1000

_HBM_THROTTLE_SERIES = "hbm_throttle"
_VDD_CORE_THROTTLE_SERIES = "vdd_core_throttle"
_VDD_CORE_POWER_SERIES = "vdd_core_power"
_HBM_POWER_SERIES = "hbm_power"
_PSTATE_SERIES = "pstate"

_EVENT_PEAK_KEYS: Mapping[str, str] = {
    constants.XPROF_HBM_PEAK_TEMP_STAT: constants.HBM_PEAK_TEMP_C,
    constants.XPROF_COMPUTE_DIE_PEAK_TEMP_STAT: constants.PEAK_TEMP_C,
}

_EVENT_SERIES: Mapping[str, str] = {
    constants.XPROF_HBM_THROTTLE_PCT_STAT: _HBM_THROTTLE_SERIES,
    constants.XPROF_VDD_CORE_THROTTLE_PCT_STAT: _VDD_CORE_THROTTLE_SERIES,
    constants.XPROF_VDD_CORE_POWER_PL1_STAT: _VDD_CORE_POWER_SERIES,
    constants.XPROF_HBM_POWER_PL1_STAT: _HBM_POWER_SERIES,
    constants.XPROF_PSTATE_EVENT: _PSTATE_SERIES,
}

_PLANE_STAT_KEYS: Mapping[str, str] = {
    constants.XPROF_HBM_PEAK_TEMP_STAT: constants.HBM_PEAK_TEMP_C,
    constants.XPROF_COMPUTE_DIE_PEAK_TEMP_STAT: constants.PEAK_TEMP_C,
}
_MIN_AGGREGATED_KEYS = frozenset({constants.PSTATE_MIN})


def _normalize_name(name: str) -> str:
  """Normalizes a name by lowercasing, stripping, and removing core ID suffixes."""
  return _CORE_ID_SUFFIX_RE.sub("", name.strip().lower())


def _window(
    intervals: list[tuple[int, int, float]],
    kernel_span: tuple[int | None, int | None],
) -> tuple[int, int] | None:
  """Returns the time window of the given intervals."""
  if kernel_span[0] is not None and kernel_span[1] is not None:
    return kernel_span[0], kernel_span[1]
  if intervals:
    lo = min(iv[0] for iv in intervals)
    hi = max(iv[1] for iv in intervals)
    return (lo, hi) if hi > lo else None
  return None


def _time_pct(
    intervals: list[tuple[int, int, float]],
    kernel_span: tuple[int | None, int | None],
    predicate: Callable[[float], bool],
) -> float | None:
  """Calculates the time percentage of a series of intervals."""
  bounds = _window(intervals, kernel_span)
  if bounds is None:
    return None
  lo, hi = bounds
  covered = active = 0
  for start_ps, end_ps, val in intervals:
    overlap = max(0, min(end_ps, hi) - max(start_ps, lo))
    covered += overlap
    if predicate(val):
      active += overlap
  return 100.0 * active / covered if covered > 0 else None


def _time_weighted_mean(
    intervals: list[tuple[int, int, float]],
    kernel_span: tuple[int | None, int | None],
) -> float | None:
  """Calculates the time-weighted mean of a series of intervals."""
  bounds = _window(intervals, kernel_span)
  if bounds is None:
    return None
  lo, hi = bounds
  covered = 0
  weighted = 0.0
  for start_ps, end_ps, val in intervals:
    overlap = max(0, min(end_ps, hi) - max(start_ps, lo))
    covered += overlap
    weighted += val * overlap
  return weighted / covered if covered > 0 else None


def _pstate_stats(
    intervals: list[tuple[int, int, float]],
    kernel_span: tuple[int | None, int | None],
) -> tuple[float | None, float | None, int | None]:
  """Calculates the min, max, and number of changes for pstate."""
  intervals = sorted(intervals, key=lambda iv: iv[0])
  bounds = _window(intervals, kernel_span)
  if bounds is not None:
    lo, hi = bounds
    earlier = [iv for iv in intervals if iv[0] < lo]
    inside = [iv for iv in intervals if lo <= iv[0] <= hi]
    intervals = earlier[-1:] + inside
  if not intervals:
    return None, None, None
  ordered = [iv[2] for iv in intervals]
  changes = sum(1 for prev, cur in zip(ordered, ordered[1:]) if prev != cur)
  return float(min(ordered)), float(max(ordered)), changes


def _is_positive(value: float) -> bool:
  return value > 0.0


def _peak_temp_source(
    peak_temp_c: float | None, hbm_peak_temp_c: float | None
) -> str | None:
  """Determines the source of the peak temperature."""
  if peak_temp_c is not None:
    return constants.PEAK_TEMP_SOURCE_COMPUTE_DIE
  if hbm_peak_temp_c is not None:
    return constants.PEAK_TEMP_SOURCE_HBM
  return None


def aggregate_thermal_metrics(
    per_device: Iterable[dict[str, Any]],
) -> dict[str, Any]:
  """Aggregates thermal metrics across multiple devices."""
  values: dict[str, list[Any]] = collections.defaultdict(list)
  for metrics in per_device:
    for key, value in metrics.items():
      if value is not None and key != constants.PEAK_TEMP_SOURCE:
        values[key].append(value)
  aggregated: dict[str, Any] = {
      key: min(vals) if key in _MIN_AGGREGATED_KEYS else max(vals)
      for key, vals in values.items()
  }
  aggregated[constants.PEAK_TEMP_SOURCE] = _peak_temp_source(
      aggregated.get(constants.PEAK_TEMP_C),
      aggregated.get(constants.HBM_PEAK_TEMP_C),
  )
  for key in constants.THERMAL_METRIC_KEYS:
    aggregated.setdefault(key, None)
  return aggregated


class XprofThermalProfiler:
  """Parses XPlane traces for TPU thermal telemetry and throttling counters."""

  def __init__(
      self,
      xprof_dir: str = "",
      xspace: xplane_pb2.XSpace | None = None,
  ):
    """Initializes XprofThermalProfiler.

    Args:
      xprof_dir: Directory containing XProf trace files (`*.xplane.pb`).
      xspace: Optional pre-parsed XSpace protobuf object.
    """
    self.xprof_dir = xprof_dir
    self.xspace = xspace

  @staticmethod
  def _requested_pstate() -> int | None:
    """Extracts the requested pstate from the LIBTPU_INIT_ARGS."""
    matches = re.findall(
        rf"(?:^|\s){re.escape(constants.LIBTPU_DVFS_P_STATE_FLAG)}[=\s]+(-?\d+)",
        os.environ.get("LIBTPU_INIT_ARGS", ""),
    )
    if not matches:
      return None
    pstate = int(matches[-1])
    return pstate if pstate >= 0 else None

  @staticmethod
  def _extract_numeric_stat(stat: xplane_pb2.XStat) -> float | None:
    """Extracts a float from a numeric or packed-double XStat protobuf field.

    Args:
      stat: XStat protobuf message.

    Returns:
      Extracted float value, or None if non-numeric.
    """
    value_field = stat.WhichOneof("value")
    if value_field in ("double_value", "int64_value", "uint64_value"):
      return float(getattr(stat, value_field))
    return None

  @classmethod
  def parse_soaking_windows(
      cls, baseline_dir: str, worst_dir: str
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parses thermal metrics for baseline and worst-case soaking windows."""

    post_soaking_per_device = cls(worst_dir).parse_per_device_metrics()
    pre_soaking_per_device = cls(baseline_dir).parse_per_device_metrics()
    post_soaking = aggregate_thermal_metrics(post_soaking_per_device.values())
    pre_soaking = aggregate_thermal_metrics(pre_soaking_per_device.values())
    metrics = dict(post_soaking)
    metrics[constants.HBM_PEAK_TEMP_C_PRE_SOAKING] = pre_soaking.get(
        constants.HBM_PEAK_TEMP_C
    )
    metrics[constants.PEAK_TEMP_C_PRE_SOAKING] = pre_soaking.get(
        constants.PEAK_TEMP_C
    )
    metrics[constants.PSTATE_REQUESTED] = cls._requested_pstate()
    details = {
        constants.THERMAL_PER_DEVICE: post_soaking_per_device,
        constants.THERMAL_PER_DEVICE_PRE_SOAKING: pre_soaking_per_device,
        constants.THERMAL_PRE_SOAKING: pre_soaking,
    }
    return metrics, details

  @classmethod
  def _extract_event_value(
      cls, event: xplane_pb2.XEvent, smeta: dict[int, str]
  ) -> float | None:
    """Extracts the numeric measurement from an XEvent, skipping counter IDs."""
    for stat in event.stats:
      if (
          smeta.get(stat.metadata_id, "").strip().lower()
          == "performance_counter_id"
      ):
        continue
      val = cls._extract_numeric_stat(stat)
      if val is not None:
        return val
    return None

  def _collect_plane(
      self, plane: xplane_pb2.XPlane, samples: dict[str, Any]
  ) -> None:
    """Collects metrics from a single XPlane plane."""
    smeta = {
        meta_id: meta.name for meta_id, meta in plane.stat_metadata.items()
    }
    emeta = {
        meta_id: meta.name for meta_id, meta in plane.event_metadata.items()
    }

    for stat in plane.stats:
      metric_key = _PLANE_STAT_KEYS.get(
          _normalize_name(smeta.get(stat.metadata_id, ""))
      )
      if metric_key is not None:
        val = self._extract_numeric_stat(stat)
        if val is not None:
          samples["peaks"][metric_key].append(val)

    for line in plane.lines:
      line_name = line.name.strip().lower()
      line_start_ps = line.timestamp_ns * _PS_PER_NS
      for event in line.events:
        start_ps = line_start_ps + event.offset_ps
        end_ps = start_ps + event.duration_ps
        event_name = emeta.get(event.metadata_id, "")
        if line_name == constants.XPROF_XLA_OPS_LINE:
          if end_ps > start_ps:
            kstart, kend = samples["kernel_span"]
            samples["kernel_span"] = (
                start_ps if kstart is None else min(kstart, start_ps),
                end_ps if kend is None else max(kend, end_ps),
            )
          continue
        canonical = _normalize_name(event_name)
        peak_key = _EVENT_PEAK_KEYS.get(canonical)
        series = _EVENT_SERIES.get(canonical)
        if peak_key is None and series is None:
          continue
        val = self._extract_event_value(event, smeta)
        if val is None:
          continue
        if peak_key is not None:
          samples["peaks"][peak_key].append(val)
        if series is not None:
          samples["series"][series].append((start_ps, end_ps, val))

  def _load_spaces(self) -> list[xplane_pb2.XSpace]:
    """Loads XSpace protos from the XProf directory or a pre-parsed XSpace."""
    if self.xspace is not None:
      return [self.xspace]
    spaces: list[xplane_pb2.XSpace] = []
    if not self.xprof_dir or not os.path.exists(self.xprof_dir):
      return spaces
    xplane_paths = []
    for root, _, files in os.walk(self.xprof_dir):
      for file in files:
        if file.endswith(".xplane.pb"):
          xplane_paths.append(os.path.join(root, file))
    for xplane_path in sorted(xplane_paths):
      try:
        with open(xplane_path, "rb") as f:
          space = xplane_pb2.XSpace()
          space.ParseFromString(f.read())
        spaces.append(space)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return spaces

  def parse_per_device_metrics(self) -> dict[str, dict[str, Any]]:
    """Parses thermal metrics for each device in the XProf traces."""
    samples: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {
            "peaks": collections.defaultdict(list),
            "series": collections.defaultdict(list),
            "kernel_span": (None, None),
        }
    )
    spaces = self._load_spaces()
    multiple_hosts = len(spaces) > 1
    for i, space in enumerate(spaces):
      for plane in space.planes:
        if _DEVICE_PLANE_RE.match(plane.name) is None:
          continue
        key = f"host{i}:{plane.name}" if multiple_hosts else plane.name
        self._collect_plane(plane, samples[key])

    def to_metrics(s: dict[str, Any]) -> dict[str, Any]:
      vdd_core = _time_weighted_mean(
          s["series"][_VDD_CORE_POWER_SERIES], s["kernel_span"]
      )
      hbm_core = _time_weighted_mean(
          s["series"][_HBM_POWER_SERIES], s["kernel_span"]
      )
      p_min, p_max, p_changes = _pstate_stats(
          s["series"][_PSTATE_SERIES], s["kernel_span"]
      )
      hbm_peak = max(
          s["peaks"].get(constants.HBM_PEAK_TEMP_C) or [None],
          key=lambda x: x if x is not None else float("-inf"),
      )
      peak_c = max(
          s["peaks"].get(constants.PEAK_TEMP_C) or [None],
          key=lambda x: x if x is not None else float("-inf"),
      )
      return {
          constants.HBM_PEAK_TEMP_C: (
              float(hbm_peak) if hbm_peak is not None else None
          ),
          constants.PEAK_TEMP_C: float(peak_c) if peak_c is not None else None,
          constants.PEAK_TEMP_SOURCE: _peak_temp_source(peak_c, hbm_peak),
          constants.HBM_THROTTLE_TIME_PCT: _time_pct(
              s["series"][_HBM_THROTTLE_SERIES], s["kernel_span"], _is_positive
          ),
          constants.HBM_THROTTLE_MEAN_PCT: _time_weighted_mean(
              s["series"][_HBM_THROTTLE_SERIES], s["kernel_span"]
          ),
          constants.VDD_CORE_THROTTLE_TIME_PCT: _time_pct(
              s["series"][_VDD_CORE_THROTTLE_SERIES],
              s["kernel_span"],
              _is_positive,
          ),
          constants.VDD_CORE_THROTTLE_MEAN_PCT: _time_weighted_mean(
              s["series"][_VDD_CORE_THROTTLE_SERIES], s["kernel_span"]
          ),
          constants.VDD_CORE_POWER_MEAN_W: vdd_core,
          constants.HBM_POWER_MEAN_W: hbm_core,
          constants.TOTAL_POWER_MEAN_W: (
              vdd_core + hbm_core
              if vdd_core is not None and hbm_core is not None
              else None
          ),
          constants.PSTATE_MIN: p_min,
          constants.PSTATE_MAX: p_max,
          constants.PSTATE_CHANGES: p_changes,
      }

    return {name: to_metrics(s) for name, s in sorted(samples.items())}

  def parse(self) -> dict[str, Any]:
    """Parses XPlane traces and returns a dictionary keyed by THERMAL_METRIC_KEYS."""
    return aggregate_thermal_metrics(self.parse_per_device_metrics().values())
