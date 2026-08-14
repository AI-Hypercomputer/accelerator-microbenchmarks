"""Profiling operations for JAX benchmarks."""

import gzip
import json
import os
from typing import Any, Callable, Optional

from accelerator_microbenchmarks.core import constants

import os
MARKER = constants.MARKER


def _get_target_pid(
    trace: dict[str, Any], local_device_id: int | None = 0
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
    for event in trace.get("traceEvents", []):
      if (
          event.get("name") == "process_name"
          and event.get("args", {}).get("name", "") == name
      ):
        return event.get("pid")
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
    events: list[dict[str, Any]],
    is_xprof_op_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
) -> list[dict[str, Any]]:
  """Extracts candidate timing events across all trace processes in order:

  1. MARKER events filtered to SparseCore completion markers ('call-done').
  2. MARKER events filtered to exclude async initiation markers ('-start').
  3. Custom fallback XProf ops (if no MARKER events are found).
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
    all_events: list[dict[str, Any]],
    candidate_events: list[dict[str, Any]],
    target_pid: int,
) -> list[float]:
  """Computes step durations by grouping target events by benchmark step:

  - PRIMARY (bounding enclosure): Each step-level enclosure (e.g. 'jit_'
    execution) defines a step. Step duration is the span from earliest target
    start to latest target end within the enclosure.
  - FALLBACK (standalone marker): If no bounding enclosures exist (e.g.,
  host-device
    DMA), each standalone target marker event defines a step.
  """
  # 1. Filter candidate timing events to the target hardware PID.
  target_events = [e for e in candidate_events if e.get("pid") == target_pid]
  # 2. Extract outermost computation enclosures (e.g. 'jit_') on target PID.
  bounding_events = [
      e
      for e in all_events
      if e.get("pid") == target_pid
      and e.get("name", "").startswith("jit_")
      and "ts" in e
      and "dur" in e
  ]

  # 3. FALLBACK: Treat each standalone marker as a step if no bounding enclosures exist.
  if not bounding_events:
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

  # 4. PRIMARY: Bound target events within each enclosure (earliest start to latest end).
  bounding_events.sort(key=lambda x: float(x["ts"]))
  durations_ms = []
  for c in bounding_events:
    c_start = float(c["ts"])
    c_end = c_start + float(c["dur"])
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


def parse_xprof_durations(
    xprof_dir: str,
    is_xprof_op_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
    local_device_id: int | None = 0,
) -> list[float]:
  """Parses XProf traces across 4 sequential stages to compute step durations:

  1. Load trace: Decompress and parse the .json.gz trace file.
  2. Candidate extraction: Extract timing events in order (MARKER
     'call-done' -> MARKER excluding '-start' -> custom XProf ops).
  3. PID resolution: Resolve target hardware PID from metadata, falling back to
     the minimum PID among candidate events.
  4. Step grouping & duration math: Group target events by benchmark step using
     step-level bounding enclosures (PRIMARY) or standalone markers (FALLBACK).
  """
  # 1. Load trace (.json.gz)
  trace = _load_xprof_trace(xprof_dir)
  if not trace:
    return []
  events = trace.get("traceEvents", [])

  # 2. Extract candidate timing events (markers -> custom ops)
  candidate_events = _extract_candidate_events(events, is_xprof_op_fn)
  if not candidate_events:
    print(f"Warning: No valid timing events found in {xprof_dir}")
    return []

  # 3. Resolve target PID (metadata -> min candidate PID)
  target_pid = _get_target_pid(trace, local_device_id)
  if target_pid is None:
    target_pid = min(e["pid"] for e in candidate_events if "pid" in e)
    print(
        "Target PID not found in metadata; falling back to min PID:"
        f" {target_pid}"
    )

  # 4. Group events by benchmark step and compute durations (enclosure -> standalone)
  return _calculate_step_durations_ms(events, candidate_events, target_pid)


def upload_xprof_trace(xprof_dir: str, cns_dir: str) -> str | None:
  """Uploads xplane to xprof and stores xprof url in CNS. Returns xprof url."""
  return None
