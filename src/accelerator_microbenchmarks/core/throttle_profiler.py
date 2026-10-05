"""Thermal telemetry and throttling XPlane profiler for JAX benchmarks."""

import dataclasses
import math
import os
import re
import struct

from accelerator_microbenchmarks.core import constants

import os
from tensorflow.tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import

_DEVICE_PLANE_RE = re.compile(r"^/device:TPU:\d+")
_POWER_THROTTLE_EVENT_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(?:\(\s*([a-z0-9_]+)\s*\))?$",
    re.IGNORECASE,
)


def _classify_thermal_metric(name: str) -> str | None:
  """Maps an XPlane event or plane-stat name to a canonical THERMAL_METRIC_KEY."""
  norm = name.strip().lower()
  if norm == constants.XPROF_HBM_PEAK_TEMP_STAT:
    return constants.HBM_PEAK_TEMP_C
  if norm == constants.XPROF_COMPUTE_DIE_PEAK_TEMP_STAT:
    return constants.PEAK_TEMP_C
  if norm == constants.XPROF_HBM_THROTTLE_PCT_STAT:
    return constants.HBM_THROTTLE_PCT
  if norm == constants.THERMAL_THROTTLE_LEVEL:
    return constants.THERMAL_THROTTLE_LEVEL
  return None


def _parse_thermal_throttle_level(event_name: str) -> float | None:
  """Parses a 'Power Throttle' XLine event name into a thermal throttle level.

  XProf PowerThrottleSubscriber emits events named `"0"` for unthrottled
  intervals and `"<level> (<source>)"` (e.g. `"12 (THERMAL_THROTTLE)"` or
  `"8 (LDIDT_DROOP_THROTTLE)"`) for active cycle-skip arbitration intervals.

  Args:
    event_name: Event metadata name on the `"Power Throttle"` XLine.

  Returns:
    Float throttle level if the event is unthrottled (`0.0`) or attributable to
    `THERMAL_THROTTLE`, or `None` for non-thermal throttle sources.
  """
  match = _POWER_THROTTLE_EVENT_RE.match(event_name.strip())
  if match is None:
    return None
  level = float(match.group(1))
  source = match.group(2)
  if source is None:
    return 0.0 if level == 0.0 else None
  if source.strip().lower() == constants.XPROF_THERMAL_THROTTLE_SOURCE:
    return level
  return None


@dataclasses.dataclass(frozen=True)
class ThermalMetrics:
  """Structured TPU thermal telemetry and throttling metrics."""

  hbm_peak_temp_c: float | None = None
  peak_temp_c: float | None = None
  hbm_throttle_pct: float | None = None
  thermal_throttle_level: float | None = None

  def to_dict(self) -> dict[str, float | None]:
    """Returns thermal metrics as a dictionary keyed by THERMAL_METRIC_KEYS."""
    return {
        constants.HBM_PEAK_TEMP_C: self.hbm_peak_temp_c,
        constants.PEAK_TEMP_C: self.peak_temp_c,
        constants.HBM_THROTTLE_PCT: self.hbm_throttle_pct,
        constants.THERMAL_THROTTLE_LEVEL: self.thermal_throttle_level,
    }


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
  def _decode_packed_double(raw: bytes) -> float | None:
    """Decodes a 64-bit IEEE-754 double from XStat bytes_value."""
    if len(raw) < 8 or (len(raw) > 8 and (raw[-9] & 0x07) != 1):
      return None
    val = float(struct.unpack("<d", raw[-8:])[0])
    return val if math.isfinite(val) else None

  @classmethod
  def _extract_numeric_stat(cls, stat: xplane_pb2.XStat) -> float | None:
    """Extracts a float from a numeric or packed-double XStat protobuf field.

    Args:
      stat: XStat protobuf message.

    Returns:
      Extracted float value, or None if non-numeric.
    """
    value_field = stat.WhichOneof("value")
    if value_field in ("double_value", "int64_value", "uint64_value"):
      return float(getattr(stat, value_field))
    if value_field == "bytes_value":
      return cls._decode_packed_double(stat.bytes_value)
    return None

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

  def _extract_from_xspace(self, space: xplane_pb2.XSpace) -> dict[str, float]:
    """Extracts thermal metrics from a single parsed XSpace proto object.

    Args:
      space: Parsed XSpace protobuf message.

    Returns:
      Dictionary of extracted thermal metric names to float values.
    """
    samples: dict[str, list[float]] = {
        k: [] for k in constants.THERMAL_METRIC_KEYS
    }

    for plane in space.planes:
      if _DEVICE_PLANE_RE.match(plane.name) is None:
        continue
      smeta = {
          meta_id: meta.name for meta_id, meta in plane.stat_metadata.items()
      }
      emeta = {
          meta_id: meta.name for meta_id, meta in plane.event_metadata.items()
      }

      # 1. Plane-level summary stats (keyed by stat_metadata.name)
      for stat in plane.stats:
        metric_key = _classify_thermal_metric(smeta.get(stat.metadata_id, ""))
        if metric_key is not None:
          val = self._extract_numeric_stat(stat)
          if val is not None:
            samples[metric_key].append(val)

      # 2. Timeline counter/telemetry events (keyed by line and event_metadata)
      for line in plane.lines:
        is_power_throttle_line = (
            line.name.strip().lower() == constants.XPROF_POWER_THROTTLE_LINE
        )
        for event in line.events:
          event_name = emeta.get(event.metadata_id, "")
          if is_power_throttle_line:
            level = _parse_thermal_throttle_level(event_name)
            if level is not None:
              samples[constants.THERMAL_THROTTLE_LEVEL].append(level)
            continue
          metric_key = _classify_thermal_metric(event_name)
          if metric_key is not None:
            val = self._extract_event_value(event, smeta)
            if val is not None:
              samples[metric_key].append(val)

    result = {k: float(max(vals)) for k, vals in samples.items() if vals}
    if (
        constants.PEAK_TEMP_C not in result
        and constants.HBM_PEAK_TEMP_C in result
    ):
      result[constants.PEAK_TEMP_C] = result[constants.HBM_PEAK_TEMP_C]
    return result

  def parse_metrics(self) -> ThermalMetrics:
    """Parses XPlane traces into a structured ThermalMetrics instance."""
    spaces: list[xplane_pb2.XSpace] = []
    if self.xspace is not None:
      spaces.append(self.xspace)
    elif self.xprof_dir and os.path.exists(self.xprof_dir):
      for root, _, files in os.walk(self.xprof_dir):
        for file in files:
          if file.endswith(".xplane.pb"):
            xplane_path = os.path.join(root, file)
            try:
              with open(xplane_path, "rb") as f:
                space = xplane_pb2.XSpace()
                space.ParseFromString(f.read())
              spaces.append(space)
            except Exception as e:  # pylint: disable=broad-exception-caught
              print(f"Warning: Failed to read XPlane {xplane_path}: {e}")

    aggregated: dict[str, list[float]] = {
        k: [] for k in constants.THERMAL_METRIC_KEYS
    }
    for space in spaces:
      extracted = self._extract_from_xspace(space)
      for k, val in extracted.items():
        if k in aggregated:
          aggregated[k].append(val)

    return ThermalMetrics(
        hbm_peak_temp_c=(
            float(max(aggregated[constants.HBM_PEAK_TEMP_C]))
            if aggregated[constants.HBM_PEAK_TEMP_C]
            else None
        ),
        peak_temp_c=(
            float(max(aggregated[constants.PEAK_TEMP_C]))
            if aggregated[constants.PEAK_TEMP_C]
            else None
        ),
        hbm_throttle_pct=(
            float(max(aggregated[constants.HBM_THROTTLE_PCT]))
            if aggregated[constants.HBM_THROTTLE_PCT]
            else None
        ),
        thermal_throttle_level=(
            float(max(aggregated[constants.THERMAL_THROTTLE_LEVEL]))
            if aggregated[constants.THERMAL_THROTTLE_LEVEL]
            else None
        ),
    )

  def parse(self) -> dict[str, float | None]:
    """Parses XPlane traces and returns a dictionary keyed by THERMAL_METRIC_KEYS."""
    return self.parse_metrics().to_dict()
