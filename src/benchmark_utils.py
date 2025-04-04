"""Utility functions for microbenchmarking."""

import datetime
import os
from typing import Dict, Any

import jax
import jsonlines
import numpy as np
import random
import string
import pathlib
import gzip
import json
import re
from collections import defaultdict
import subprocess


def simple_timeit(f, *args, tries=1000, task=None, trace_dir=None) -> float:
    """Simple utility to time a function for multiple runs."""
    assert task is not None

    if trace_dir:
        return timeit_from_trace(f, *args, tries=tries, task=task, trace_dir=trace_dir)

    outcomes_ms = []
    jax.block_until_ready(f(*args))  # warm it up!
    for _ in range(tries):
        jax.devices()  # Force synchronization across devices
        s = datetime.datetime.now()
        jax.block_until_ready(f(*args))
        e = datetime.datetime.now()
        outcomes_ms.append(1000 * (e - s).total_seconds())
    return outcomes_ms


def get_trace(log_dir: str) -> dict[str, Any]:
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


def get_metrics_from_trace(trace: dict[str, Any], task: str) -> float:
    event_matcher = re.compile(task)
    if "traceEvents" not in trace:
        raise KeyError("Key 'traceEvents' not found in trace.")

    events = []
    for e in trace["traceEvents"]:
        if "name" in e and event_matcher.match(e["name"]):
            events.append(e)

    events_by_run_id = defaultdict(list)
    for e in events:
        run_id = e["args"]["run_id"] if "args" in e and "run_id" in e["args"] else "0"
        events_by_run_id[run_id].append(e)
    durations_ms = []
    try:
        # Duration is in us.
        durations_ms = [
            max([e["dur"] for e in es]) / 1e3 for run_id, es in events_by_run_id.items()
        ]
    except KeyError:
        print("KeyError: Key 'dur' not found in the event object")
        raise
    return durations_ms


def is_local_directory_path(dir: str) -> bool:
    """
    Returns true if the path is a local path.
    """
    if not dir:  # Handle None or empty string
        return False

    # Heuristics for local paths
    return dir.startswith("/") or dir.startswith("./") or dir.startswith("../")


def timeit_from_trace(f, *args, tries=10, task=None, trace_dir=None) -> float:
    """
    Time a function with jax.profiler and get the run time from the trace.
    """
    LOCAL_TRACE_DIR = "/tmp/microbenchmarks_tmptrace"

    jax.block_until_ready(f(*args))  # warm it up!

    trace_name = f"t_{task}_" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=10)
    )
    trace_full_dir = f"{trace_dir}/{trace_name}"
    tmp_trace_dir = trace_full_dir
    # If the trace_dir isn't a local path, create one for dumping the trace for parsing and getting metrics.
    if trace_dir and not is_local_directory_path(trace_dir):
        tmp_trace_dir = f"{LOCAL_TRACE_DIR}/{trace_name}"
    with jax.profiler.trace(tmp_trace_dir):
        for _ in range(tries):
            jax.devices()  # Force synchronization across devices
            with jax.profiler.TraceAnnotation(task):
                jax.block_until_ready(f(*args))

    trace = get_trace(tmp_trace_dir)

    if trace_full_dir != tmp_trace_dir:
        # Upload the traces to desired location
        upload_to_storage(trace_dir=trace_full_dir, local_file=tmp_trace_dir)
    return get_metrics_from_trace(trace, task)


def maybe_write_metrics_file(
    metrics_dir, metrics, metadata, test_name, test_start_time, test_end_time
):
    """Writes metrics to a JSONL file to be consumed by the XLML metrics pipeline."""

    # Only write metrics from one host.
    if jax.process_index() != 0:
        return

    jsonl_name = "metrics_report.jsonl"
    jsonl_path = metrics_dir + "/" + jsonl_name
    metadata.update(
        {
            "testsuite": "microbenchmark",
            "test_name": f"{test_name}",
            "test_start_timestamp": f"{test_start_time}",
            "test_end_timestamp": f"{test_end_time}",
        }
    )
    metrics_data = {
        "metrics": metrics,
        "dimensions": metadata,
    }
    # Make sure the metadata value is a string.
    for key, value in metadata.items():
        metadata[key] = str(value)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    print(f"Writing metrics to JSONL file: {jsonl_path}")
    with jsonlines.open(jsonl_path, mode="a") as writer:
        writer.write(metrics_data)


def upload_to_storage(trace_dir: str, local_file: str):
    """
    Uploads a local file to a specified storage location.
    """

    if trace_dir.startswith("gs://"):  # Google Cloud Storage (GCS)
        try:
            subprocess.run(
                ["gsutil", "cp", "-r", local_file, trace_dir],
                check=True,
                capture_output=True,
            )

        except subprocess.CalledProcessError as e:
            print(
                f"Failed to upload '{local_file}' to GCS: '{trace_dir}'. Error: {e.stderr.decode()}"
            )
    else:
        raise KeyError(f"{trace_dir} is not a valid GCS path.")


class MetricsStatistics:
    """
    Represents statistics for a list of metrics.
    """

    def __init__(self, metrics_list, metrics_name: str):
        self.metrics_list = metrics_list
        self.metrics_name = metrics_name
        self.statistics = self._calculate_statistics()

    def _calculate_statistics(self) -> Dict[str, float]:
        """Calculates the statistics of the metrics list."""
        if not self.metrics_list:
            return {}  # Return an empty dict if metrics_list is empty
        return {
            "p50": np.percentile(self.metrics_list, 50),
            "p90": np.percentile(self.metrics_list, 90),
            "p95": np.percentile(self.metrics_list, 95),
            "p99": np.percentile(self.metrics_list, 99),
            "avg": np.mean(self.metrics_list),
        }

    def __repr__(self):
        return (
            f"MetricsStatistics(metrics_name='{self.metrics_name}', "
            f"statistics={self.statistics})"
        )

    def serialize_statistics(self):
        serialized = {}
        for stat_name, stat_value in self.statistics.items():
            serialized[f"{self.metrics_name}_{stat_name}"] = stat_value
        return serialized
