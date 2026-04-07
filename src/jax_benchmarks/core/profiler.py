"""Profiling operations for JAX benchmarks."""

import glob
import gzip
import json
import os
import re
import time
from typing import Any, Dict, List

import numpy as np




def parse_xprof_results(xprof_dir: str, cns_dir: str, metrics: Dict[str, Any]):
  """Parses trace files to extract timing markers and stores xprof url in CNS."""
  from .constants import MARKER

  trace_path = None
  xplane_path = None
  for root, _, files in os.walk(xprof_dir):
    for file in files:
      if file.endswith(".json.gz"):
        trace_path = os.path.join(root, file)
        print(f"Found trace file: {trace_path}")
        xplane_path = os.path.join(
            root, file.replace("trace.json.gz", "xplane.pb")
        )
        print(f"Found xplane file: {xplane_path}")
        break
    if trace_path:
      break

  if xplane_path and os.path.exists(xplane_path):
  elif xplane_path:
    print(f"xplane_path not found: {xplane_path}")

  if not trace_path or not os.path.exists(trace_path):
    print(f"No .json.gz trace file found in {xprof_dir}")
    return metrics

  # Read trace metrics
  with open(trace_path, "rb") as f_gz:
    with gzip.GzipFile(fileobj=f_gz) as f:
      trace = json.loads(f.read())

  marker_done_events = []
  for event in trace.get("traceEvents", []):
    args = event.get("args", {})
    tf_op = args.get("tf_op", "")
    if MARKER in tf_op:
      marker_done_events.append(event)

  # when offloaded to sparse core look for call-done events
  marker_call_done_events = [
      e for e in marker_done_events if e.get("name", "").endswith("call-done")
  ]
  if marker_call_done_events:
    marker_done_events = marker_call_done_events

  if not marker_done_events:
    print(f"Warning: No '{MARKER}' events found in {trace_path}")
    return metrics

  unique_pids = set([e["pid"] for e in marker_done_events])
  print(f"Unique PIDs: {unique_pids}")

  min_pid = min([e["pid"] for e in marker_done_events])
  events_from_min_pid = [e for e in marker_done_events if e["pid"] == min_pid]
  durations_ms = []
  for e in events_from_min_pid:
    if e.get("args", {}).get("device_duration_ps"):
      durations_ms.append(float(e["args"]["device_duration_ps"]) / 1e9)
    elif "dur" in e:
      durations_ms.append(float(e["dur"]) / 1e3)

  print(f"Collected {len(durations_ms)} events from trace for pid {min_pid}.")

  if durations_ms:
    metrics["xprof_avg_ms"] = float(np.mean(durations_ms))
    metrics["xprof_p50_ms"] = float(np.percentile(durations_ms, 50))
    metrics["xprof_p90_ms"] = float(np.percentile(durations_ms, 90))

  return metrics
