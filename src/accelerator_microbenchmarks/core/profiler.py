"""Profiling operations for JAX benchmarks."""

import dataclasses
import gzip
import json
import os
from typing import Any, Callable, Optional, TypedDict

from accelerator_microbenchmarks.core import constants

import os
MARKER = constants.MARKER


class TraceEvent(TypedDict, total=False):
  """One event of an XProf `.trace.json.gz` file (Chrome trace event format).

  Attributes:
    pid: Trace viewer row (process) the event belongs to, e.g. the row of
      `/device:TPU:0` or `/host:CPU`.
    name: Event name, e.g. an HLO op, a `jit_*` step, or `process_name`
      metadata.
    ts: Start timestamp in microseconds.
    dur: Duration in microseconds.
    args: Extra event arguments, e.g. `name` for `process_name` metadata,
      `tf_op`, or `device_duration_ps`.
  """

  pid: int
  name: str
  ts: float
  dur: float
  args: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class XprofTrace:
  """An XProf trace indexed by device once, for per-device step durations.

  Each "pid" is one row of the trace viewer, i.e. one device such as
  `/device:TPU:0` or `/host:CPU`. `load_xprof_trace` builds this once per
  trace (stage 1); `XprofTrace.device_step_durations_ms` then only looks up
  one pid's entries (stage 2) instead of rescanning every trace event per
  device.

  Attributes:
    timing_events_by_pid: Timing events chosen by `_extract_candidate_events`
      across the whole trace, grouped by pid in their original trace order.
      Events without a pid are grouped under None.
    step_spans_by_pid: `(start, end)` timestamps (us) of each `jit_*` step event
      that has both `ts` and `dur`, grouped by pid and stably sorted by start
      time, so steps with equal start times keep their trace order.
    process_pids: Pid of each `process_name` metadata event keyed by process
      name (e.g. `/device:TPU:0`, `/host:CPU`); the first occurrence wins.
    fallback_pid: Smallest pid among the timing events that have a pid, used
      when the requested device has no `process_name` metadata; None if no
      timing event has a pid.
  """

  timing_events_by_pid: dict[int | None, list[TraceEvent]]
  step_spans_by_pid: dict[int, list[tuple[float, float]]]
  process_pids: dict[str, int | None]
  fallback_pid: int | None

  def device_step_durations_ms(
      self, local_device_id: int | None = 0
  ) -> list[float]:
    """Stage 2 (per device): computes step durations (ms) for one device.

    1. Look up the device's pid from `process_name` metadata, falling back to
       the smallest pid among the timing events.
    2. Compute step durations from that pid's timing events, using its `jit_*`
       step spans (PRIMARY) or each standalone timing event (FALLBACK).

    Args:
      local_device_id: Local hardware ID of the target TPU device
        (`/device:TPU:{id}`), or None to measure `/host:CPU`.

    Returns:
      Per-step durations in milliseconds.

    Raises:
      ValueError: If the device has no metadata pid and no timing event has a
        pid to fall back to.
    """
    # 1. Resolve target PID (metadata -> min timing event PID)
    target_pid = _get_target_pid(self.process_pids, local_device_id)
    if target_pid is None:
      if self.fallback_pid is None:
        raise ValueError(
            "Target PID not found in metadata and no timing event has a PID."
        )
      target_pid = self.fallback_pid
      print(
          "Target PID not found in metadata; falling back to min PID:"
          f" {target_pid}"
      )

    # 2. Group events by step and compute durations (enclosure -> standalone)
    return _calculate_step_durations_ms(
        self.timing_events_by_pid.get(target_pid, []),
        self.step_spans_by_pid.get(target_pid, []),
        target_pid,
    )


def _get_process_pids(events: list[TraceEvent]) -> dict[str, int | None]:
  """Maps each `process_name` metadata name to its PID (first occurrence)."""
  process_pids: dict[str, int | None] = {}
  for event in events:
    if event.get("name") == "process_name":
      process_pids.setdefault(
          event.get("args", {}).get("name", ""), event.get("pid")
      )
  return process_pids


def _get_target_pid(
    process_pids: dict[str, int | None], local_device_id: int | None = 0
) -> int | None:
  """Resolves the measured process PID from XProf metadata in order:

  1. TPU device process (/device:TPU:{local_device_id}) if local_device_id is
  set.
  2. Host CPU process (/host:CPU).
  """
  target_names = []
  if local_device_id is not None:
    target_names.append(f"/device:TPU:{local_device_id}")
  target_names.append("/host:CPU")

  for name in target_names:
    if name in process_pids:
      return process_pids[name]
  return None


def _load_xprof_trace(xprof_dir: str) -> dict[str, Any] | None:
  """Locates, decompresses, and parses the .json.gz XProf trace file."""
  trace_path = None
  for root, _, files in os.walk(xprof_dir):
    for file in files:
      if file.endswith(".json.gz"):
        trace_path = os.path.join(root, file)
        print(f"Found trace file: {trace_path}")
        break
    if trace_path:
      break

  if not trace_path or not os.path.exists(trace_path):
    print(f"No .json.gz trace file found in {xprof_dir}")
    return None

  with open(trace_path, "rb") as f_gz:
    with gzip.GzipFile(fileobj=f_gz) as f:
      return json.loads(f.read())


def _extract_candidate_events(
    events: list[TraceEvent],
    is_xprof_op_fn: Optional[Callable[[TraceEvent], bool]] = None,
) -> list[TraceEvent]:
  """Extracts candidate timing events across all trace processes in order:

  1. MARKER events filtered to SparseCore completion markers ('call-done').
  2. MARKER events filtered to exclude async initiation markers ('-start').
  3. Custom fallback XProf ops (if no MARKER events are found).

  The choice is made across all pids together, so 'call-done' markers on one
  pid suppress plain markers on every other pid.

  Args:
    events: All trace events.
    is_xprof_op_fn: Custom XProf op matcher used when no MARKER events exist.

  Returns:
    The selected timing events, in trace order.
  """
  # 1. Search all trace events for MARKER.
  all_markers = [
      e
      for e in events
      if MARKER in e.get("name", "")
      or MARKER in e.get("args", {}).get("tf_op", "")
  ]
  # 2. Filter MARKER events to SparseCore completion markers ('call-done').
  call_done = [
      e for e in all_markers if e.get("name", "").endswith("call-done")
  ]
  if call_done:
    return call_done

  # 3. Exclude async initiation markers ('-start').
  candidates = [e for e in all_markers if "start" not in e.get("name", "")]
  # 4. Fallback to custom XProf op matching if no markers exist.
  if not candidates and is_xprof_op_fn:
    print(
        f"No '{MARKER}' events found; falling back to custom XProf op matching."
    )
    candidates = [e for e in events if is_xprof_op_fn(e)]
  return candidates


def _calculate_step_durations_ms(
    target_events: list[TraceEvent],
    step_spans: list[tuple[float, float]],
    target_pid: int,
) -> list[float]:
  """Computes step durations by grouping target events by benchmark step:

  - PRIMARY (bounding enclosure): Each step-level enclosure (e.g. 'jit_'
    execution) defines a step. Step duration is the span from earliest target
    start to latest target end within the enclosure.
  - FALLBACK (standalone marker): If no bounding enclosures exist (e.g.,
  host-device
    DMA), each standalone target marker event defines a step.

  Args:
    target_events: Timing events of `target_pid`, in trace order.
    step_spans: `(start, end)` of the `jit_*` step events of `target_pid`,
      sorted by start time.
    target_pid: Pid of the measured device, used only in log messages.

  Returns:
    Per-step durations in milliseconds.
  """
  # 1. FALLBACK: Without bounding enclosures, each standalone marker is a step.
  if not step_spans:
    durations_ms = []
    for e in target_events:
      if e.get("args", {}).get("device_duration_ps"):
        durations_ms.append(float(e["args"]["device_duration_ps"]) / 1e9)
      elif "dur" in e:
        durations_ms.append(float(e["dur"]) / 1e3)
    print(
        "No bounding events ('jit_') found; fallback collected"
        f" {len(durations_ms)} timing events for PID {target_pid}."
    )
    return durations_ms

  # 2. PRIMARY: Span from earliest start to latest end within each enclosure.
  durations_ms = []
  for c_start, c_end in step_spans:
    in_step = [e for e in target_events if c_start <= float(e["ts"]) <= c_end]
    if in_step:
      s_start = min(float(e["ts"]) for e in in_step)
      s_end = max(float(e["ts"]) + float(e["dur"]) for e in in_step)
      durations_ms.append((s_end - s_start) / 1000.0)

  print(
      f"Collected {len(durations_ms)} step events from trace for PID"
      f" {target_pid}."
  )
  return durations_ms


def _group_timing_events_by_pid(
    candidate_events: list[TraceEvent],
) -> dict[int | None, list[TraceEvent]]:
  """Groups timing events by pid, keeping trace order within each pid."""
  by_pid: dict[int | None, list[TraceEvent]] = {}
  for e in candidate_events:
    by_pid.setdefault(e.get("pid"), []).append(e)
  return by_pid


def _get_step_spans_by_pid(
    events: list[TraceEvent],
) -> dict[int, list[tuple[float, float]]]:
  """Returns `(start, end)` of each `jit_*` step event per pid, sorted by start.

  The sort is stable and keyed on the start only, so step events with equal
  start times keep their trace order.

  Args:
    events: All trace events.

  Returns:
    Step spans (us) keyed by pid; events without a pid are skipped.
  """
  spans_by_pid: dict[int, list[tuple[float, float]]] = {}
  for e in events:
    pid = e.get("pid")
    if (
        pid is not None
        and e.get("name", "").startswith("jit_")
        and "ts" in e
        and "dur" in e
    ):
      start = float(e["ts"])
      spans_by_pid.setdefault(pid, []).append((start, start + float(e["dur"])))
  for spans in spans_by_pid.values():
    spans.sort(key=lambda span: span[0])
  return spans_by_pid


def load_xprof_trace(
    xprof_dir: str,
    is_xprof_op_fn: Optional[Callable[[TraceEvent], bool]] = None,
) -> XprofTrace | None:
  """Stage 1 (once per trace): loads an XProf trace and indexes it per device.

  1. Load and decompress the .json.gz trace file.
  2. Pick the timing events across the whole trace, in order: MARKER
     'call-done' events, else MARKER events excluding '-start', else custom
     XProf ops. The choice is made for all pids together.
  3. Index the timing events, `jit_*` step spans, and `process_name` metadata
     per pid, so stage 2 never rescans the trace.

  Args:
    xprof_dir: Directory containing the XProf trace.
    is_xprof_op_fn: Custom XProf op matcher used when no MARKER events exist.

  Returns:
    The indexed trace, or None if no trace file or no timing events are found.
  """
  # 1. Load trace (.json.gz)
  trace = _load_xprof_trace(xprof_dir)
  if not trace:
    return None
  events: list[TraceEvent] = trace.get("traceEvents", [])

  # 2. Extract candidate timing events (markers -> custom ops)
  candidate_events = _extract_candidate_events(events, is_xprof_op_fn)
  if not candidate_events:
    print(f"Warning: No valid timing events found in {xprof_dir}")
    return None

  # 3. Index timing events, step spans, and process metadata per pid.
  return XprofTrace(
      timing_events_by_pid=_group_timing_events_by_pid(candidate_events),
      step_spans_by_pid=_get_step_spans_by_pid(events),
      process_pids=_get_process_pids(events),
      fallback_pid=min(
          (e["pid"] for e in candidate_events if "pid" in e), default=None
      ),
  )


def upload_xprof_trace(xprof_dir: str, cns_dir: str) -> str | None:
  """Uploads xplane to xprof and stores xprof url in CNS. Returns xprof url."""
  return None
