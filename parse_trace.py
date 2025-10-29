import pathlib
import os
import gzip
import json
import re
import numpy as np

TIME_PROXIMITY_THRESHOLD_US = 0.1

def get_trace(log_dir: str):
    """Extract the trace object from the log directory.

    Returns:
      A trace object in JSON format.
    """
    # Navigate to the folder with the latest trace dump to find `trace.json.jz`
    trace_folders = (pathlib.Path(log_dir).absolute() / "plugins" / "profile").iterdir()
    latest_trace_folder = max(trace_folders, key=os.path.getmtime)
    trace_jsons = latest_trace_folder.glob("*.trace.json.gz")
    try:
        (trace_json,) = trace_jsons
    except ValueError as value_error:
        raise ValueError(
            f"Invalid trace folder: {latest_trace_folder}"
        ) from value_error

    with gzip.open(trace_json, "rb") as f:
        trace = json.load(f)

    return trace


def get_metrics_from_trace_tpu(trace, task: str) -> list[float]:
    event_matcher = re.compile(task)

    if "traceEvents" not in trace:
        raise KeyError("Key 'traceEvents' not found in trace.")
    
    events = []
    for e in trace["traceEvents"]:
        if "name" in e and event_matcher.match(e["name"]):
            events.append(e)
    
    if not events:
        raise Exception("No events found")
    
    # For each trace, find the TPU with smallest `pid` value and consider it to be TPU-0
    min_pid = min([e["pid"] for e in events])
    events_min_pid = [e for e in events if e["pid"] == min_pid]
    
    # Consolidate the events that are close enough in time. In some cases for
    # AllToAll, it could be split into a few back-to-back all-to-all ops.
    sorted_events = sorted(events_min_pid, key=lambda x: x["ts"])
    merged_durations_us = []
    current_group_start_time = sorted_events[0]["ts"]
    current_group_duration = sorted_events[0]["dur"]
    current_group_timestamp_end = current_group_start_time + current_group_duration

    for e in sorted_events[1:]:
        current_timestamp = e["ts"]
        current_duration = e["dur"]
        
        time_diff = current_timestamp - current_group_timestamp_end
        if time_diff < TIME_PROXIMITY_THRESHOLD_US:
            current_group_duration += current_duration
            current_group_timestamp_end = current_timestamp + current_duration
        else:
            merged_durations_us.append(current_group_duration)
            current_group_start_time = current_timestamp
            current_group_duration = current_duration
            current_group_timestamp_end = current_group_start_time + current_group_duration

    merged_durations_us.append(current_group_duration)
    return merged_durations_us

log_dir = "../outputs_16x16/all_to_all_ici_op_dim_1024"
task = "all_to_all"
trace = get_trace(log_dir)
durations_us = get_metrics_from_trace_tpu(trace, task)
print(len(durations_us))
print(np.median(durations_us))