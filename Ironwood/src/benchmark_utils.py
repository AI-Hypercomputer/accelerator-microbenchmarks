"""Utility functions for microbenchmarking."""

import datetime
import os
from typing import Any, Dict, Tuple, Callable
import glob
import yaml

import jax
import jax.numpy as jnp
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
import shutil
from common import MARKER
from enum import Enum, auto
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import gc
import jax.extend
from tensorflow.tsl.profiler.protobuf import xplane_pb2


def get_real_dtype_bytes(dtype) -> float:
  """Returns the real byte size of a dtype, handling sub-byte types."""
  try:
    return jnp.finfo(dtype).bits / 8
  except Exception:
    try:
      return jnp.iinfo(dtype).bits / 8
    except Exception:
      return dtype.itemsize


# The dictionary to map a JAX (collective) function to its main HLO.
TARGET_TASK_NAME_COLLECTIVES_MAP = {
    "all_to_all_ici_op": r"all-to-all.[0-9]+",
    "all_gather_ici_op": r"all-gather.[0-9]+",
    "psum_ici_op": r"all-reduce.[0-9]+",
    "ppermute_ici_op": r"collective-permute.[0-9]+",
    "single_device_hbm_copy": r"copy.[0-9]+",
}


class ShardingStrategy(Enum):
  """Defines different sharding strategies for tensors."""

  NO_SHARDING = auto()
  SHARDING_ON_ALL_DEVICES_WITH_M = auto()
  SHARDING_ON_SINGLE_CHIP_WITH_M = (
      auto()
  )  # Only sharding on the two core of one single chip
  SHARDING_ON_ALL_DEVICES_WITH_N = auto()
  SHARDING_ON_SINGLE_CHIP_WITH_N = auto()



def multiple_iteration_timeit_from_trace_throttling(
    compute_func: Callable,
    data_generator: Callable,
    matrix_dim: str = None,
    tries: int = 17,
    task: str = None,
    trace_dir: str = None,
    gap_strategy: str = None,
) -> list[float]:
  """Time a function with jax.profiler and get the run time from the trace."""
  LOCAL_TRACE_DIR = "/tmp/microbenchmarks_tmptrace"

  if matrix_dim is not None:
    trace_name = f"{task}_dim_{matrix_dim}"
  else:
    trace_name = f"t_{task}_" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=10)
    )

  trace_full_dir = f"{trace_dir}/{trace_name}"
  tmp_trace_dir = trace_full_dir
  # If the trace_dir isn't a local path, create one for dumping the trace for parsing and getting metrics.
  if trace_dir and not is_local_directory_path(trace_dir):
    tmp_trace_dir = f"{LOCAL_TRACE_DIR}/{trace_name}"

  if gap_strategy == "data_gen_once_block_every_iter":
    data_args = data_generator()
    with jax.profiler.trace(tmp_trace_dir):
      for i in range(tries):
        if i % 10 == 0:
          print(
              f"[{task}] Running iteration {i} of {tries} with {matrix_dim}..."
          )
        jax.devices()
        with jax.profiler.StepTraceAnnotation(task, step_num=i):
          with jax.named_scope(f"{MARKER}_{i}"):
            result = compute_func(*data_args)
            jax.block_until_ready(result)
  elif gap_strategy=='data_gen_once_noblock':
    data_args = data_generator()
    with jax.profiler.trace(tmp_trace_dir):
        results = []
        for i in range(tries):
            if i % 10 == 0:
                print(f"[{task}] Running iteration {i} of {tries} with {matrix_dim}...")
            jax.devices()
            with jax.profiler.StepTraceAnnotation(task, step_num=i):
                with jax.named_scope(f"{MARKER}_{i}"):
                    compute_func(*data_args)
                    results.append(True)

        if results:
          jax.block_until_ready(results)
  elif gap_strategy == "data_gen_every_iter_block_every_iter":
    with jax.profiler.trace(tmp_trace_dir):
        for i in range(tries):
            if i % 10 == 0:
                print(f"[{task}] Running iteration {i} of {tries} with {matrix_dim}...")
            data_args = data_generator()
            jax.devices()
            with jax.profiler.StepTraceAnnotation(task, step_num=i):
                with jax.named_scope(f"{MARKER}_{i}"):
                    result = compute_func(*data_args)
                    jax.block_until_ready(result)
  else:
    raise ValueError(f"Unknown gap strategy: {gap_strategy}")
  trace = get_trace(tmp_trace_dir)

  if trace_full_dir != tmp_trace_dir:
    # Upload the traces to desired location
    upload_to_storage(trace_dir=trace_full_dir, local_file=tmp_trace_dir)
  return multiple_iteration_get_metrics_from_trace(trace)


def clear_jax_memory():
    backend = jax.extend.backend.get_backend()
    for buf in backend.live_buffers():
        buf.delete()
    gc.collect()


def multiple_iteration_timeit_from_trace(
    compute_func: Callable,
    data_generator: Callable,
    matrix_dim: str = None,
    tries: int = 17,
    task: str = None,
    trace_dir: str = None,
    multi_op: bool = False,
) -> list[float]:
    """
    Time a function with jax.profiler and get the run time from the trace.
    """
    LOCAL_TRACE_DIR = "/tmp/microbenchmarks_tmptrace"

    if matrix_dim is not None:
        trace_name = f"{task}_dim_{matrix_dim}"
    else:
        trace_name = f"t_{task}_" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )

    trace_full_dir = f"{trace_dir}/{trace_name}"
    tmp_trace_dir = trace_full_dir
    # If the trace_dir isn't a local path, create one for dumping the trace for parsing and getting metrics.
    if trace_dir and not is_local_directory_path(trace_dir):
        tmp_trace_dir = f"{LOCAL_TRACE_DIR}/{trace_name}"
    # data_args = data_generator()
    with jax.profiler.trace(tmp_trace_dir):
        for i in range(tries):
            if i % 10 == 0:
                print(f"[{task}] Running iteration {i} of {tries} with {matrix_dim}...")
            data_args = data_generator()
            jax.devices()

            with jax.profiler.StepTraceAnnotation(task, step_num=i):
                with jax.named_scope(f"{MARKER}_{i}"):

                    result = compute_func(*data_args)
                    jax.block_until_ready(result)

            # Commenting it out as it's causing issues with GEMM
            # clear_jax_memory()
    trace = get_trace(tmp_trace_dir)

    if trace_full_dir != tmp_trace_dir:
        # Upload the traces to desired location
        upload_to_storage(trace_dir=trace_full_dir, local_file=tmp_trace_dir)
    return multiple_iteration_get_metrics_from_trace(trace, task, tries, multi_op)


def multiple_iteration_get_metrics_from_trace(
    trace: dict[str, Any], task: str = None, tries = 17, multi_op: bool = False) -> list[float]:
    marker_done_events = []
    for event in trace["traceEvents"]:
        args = event.get("args", {})
        tf_op = args.get("tf_op", "")
        if MARKER in tf_op:
            marker_done_events.append(event)
    # when offloaded to sparse core look for call-done events
    marker_call_done_events = [
        e for e in marker_done_events if e.get("name", "").endswith("call-done")
    ]
    if not multi_op and marker_call_done_events:
        marker_done_events = marker_call_done_events
    unique_pids = set([e["pid"] for e in marker_done_events])
    print(f"Unique PIDs: {unique_pids}")
    if not marker_done_events:
        event_matcher = re.compile(task)

        if "traceEvents" not in trace:
          raise KeyError("Key 'traceEvents' not found in trace.")
        events = []
        for e in trace["traceEvents"]:
            if "name" in e and event_matcher.match(e["name"]):
                events.append(e)
        # For each trace, find the TPU with smallest `pid` value and consider it to be TPU-0
        min_pid = min([e["pid"] for e in events])
        events_from_min_pid = [e for e in events if e["pid"] == min_pid]
        print(events_from_min_pid)
        durations_ms = []
        for e in events_from_min_pid:
            if e.get("args", {}).get("device_duration_ps"):
                durations_ms.append(float(e["args"]["device_duration_ps"]) / 1e9)
            elif "dur" in e:
                durations_ms.append(float(e["dur"]) / 1e3)
        if not durations_ms and events_from_min_pid:
            print("Warning: No event duration found in legacy_get_metrics_from_trace_tpu.")
        return durations_ms

    min_pid = min([e["pid"] for e in marker_done_events])
    events_from_min_pid = [e for e in marker_done_events if e["pid"] == min_pid]

    if multi_op and len(events_from_min_pid) > tries:
        if len(events_from_min_pid) % tries != 0:
            raise ValueError(f"Number of events {len(events_from_min_pid)} is not a multiple of tries {tries}.")
        events_from_min_pid.sort(key=lambda t: t["ts"])
        durations_ms = []
        num_ops = len(events_from_min_pid) // tries
        for i in range(0, len(events_from_min_pid), num_ops):
            durations_ms.append(sum([float(e["args"]["device_duration_ps"]) / 1e9 for e in events_from_min_pid[i:i+num_ops]]))
    else:
        durations_ms = [
            float(e["args"]["device_duration_ps"]) / 1e9 for e in events_from_min_pid
        ]
    print(f"Collected {len(durations_ms)} events from trace for pid {min_pid}.")
    print(durations_ms)

    return durations_ms

def iteration_timeit_from_trace(
    compute_func: Callable,
    data_generator: Callable,
    matrix_dim: str=None,
    tries: int=10,
    task: str = None,
    trace_dir: str = None,
    event_name_str_list: list[str] = None) -> list[float]:
    """
    Time a function with jax.profiler and get the run time from the trace.
    """
    LOCAL_TRACE_DIR = "/tmp/microbenchmarks_tmptrace"

    if matrix_dim is not None:
        trace_name = f"{task}_dim_{matrix_dim}"
    else:
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
            data_args = data_generator()
            jax.devices()  # Force synchronization across devices
            with jax.profiler.TraceAnnotation(task):
                result = compute_func(*data_args)
                jax.block_until_ready(result)

    trace = get_trace(tmp_trace_dir)

    if trace_full_dir != tmp_trace_dir:
        # Upload the traces to desired location
        upload_to_storage(trace_dir=trace_full_dir, local_file=tmp_trace_dir)
    return iteration_get_metrics_from_trace(
            trace=trace,
            event_name_str_list=event_name_str_list)


def iteration_get_metrics_from_trace(
    trace: dict[str, Any],
    tf_op_str_list: list[str] = None,
    event_name_str_list: list[str] = None,
) -> list[float]:
    # 1. Handle default inputs
    # If not provided, filter for MARKER in tf_op and no specific event names.
    if tf_op_str_list is None:
        tf_op_str_list = [MARKER]
    if event_name_str_list is None:
        event_name_str_list = []

    # Rename the storage variable to reflect its contents
    selected_events = []

    # 2. Filtering logic
    for event in trace["traceEvents"]:
        # Events without 'args' or 'name' cannot be filtered, skip them.
        args = event.get("args", {})
        tf_op = args.get("tf_op", "")
        event_name = event.get("name", "")

        # Check if the event matches any of the provided filters
        tf_op_matches = any(s in tf_op for s in tf_op_str_list)
        event_name_matches = any(s in event_name for s in event_name_str_list)

        if tf_op_matches or event_name_matches:
            selected_events.append(event)

    if not selected_events:
        print("Collected 0 events with specified filters in the trace.")
        return []

    # 3. Group events by PID (device/core) and sum durations per PID

    # Dictionary structure: pid -> list of events for that pid
    events_by_pid = defaultdict(list)
    for event in selected_events:
        events_by_pid[event["pid"]].append(event)

    # Calculate total duration for each unique device
    durations_ms_list = []

    for pid in sorted(events_by_pid.keys()):
        events = events_by_pid[pid]

        # Sum the device_duration_ps (picoseconds) for all events belonging to this PID
        # CAVEAT: If multiple iterations of the op runs for benchmarking, then the next
        # instruction will sum it for all the iterations which will not be the expected
        # behavior. Find the metadata key which is different for different iteration on
        # same PID. Eg: `group_id`.
        total_duration_ps = sum(
            float(e["args"].get("device_duration_ps", 0)) for e in events
        )

        # Convert picoseconds (ps) to milliseconds (ms)
        total_duration_ms = total_duration_ps / 1e9
        durations_ms_list.append(total_duration_ms)

    # 4. Print summary and return
    print(f"Collected event data for {len(events_by_pid)} unique devices/PIDs.")
    for i, pid in enumerate(sorted(events_by_pid.keys())):
        print(f"Device {i} (PID {pid}): {durations_ms_list[i]:.6f} ms")

    # Return the list of summed durations, one for each device
    return durations_ms_list


def iteration_get_event_metrics_from_trace(
    trace: dict[str, Any],
    event_name_str_list: list[str],
) -> list[float]:
    # Rename the storage variable to reflect its contents
    selected_events = []

    # 1. Filtering logic
    for event in trace["traceEvents"]:
        # Events without 'args' or 'name' cannot be filtered, skip them.
        args = event.get("args", {})
        event_name = event.get("name", "")

        # Check if the event matches any of the provided filters
        event_name_matches = any(s in event_name for s in event_name_str_list)

        if event_name_matches:
            selected_events.append(event)

    if not selected_events:
        print("Collected 0 events with specified filters in the trace.")
        return []

    # 2. Group events by PID (device/core)

    # Dictionary structure: pid -> list of events for that pid
    events_by_pid = defaultdict(list)
    for event in selected_events:
        events_by_pid[event["pid"]].append(event)

    # Calculate total duration for each unique device
    durations_ms_lists = []

    for pid in sorted(events_by_pid.keys()):
        events = events_by_pid[pid]

        # Collect the durarion_ms for each run
        durations_ms_lists.append([
            float(e["args"].get("device_duration_ps", 0)) / 1e9 for e in events
        ])

    # 3. Print summary from the first device and return
    print(f"Average Execution time: {np.mean(durations_ms_lists[0]):.6f} ms")

    # Return the list of durations from the first device
    return durations_ms_lists[0]


def iteration_timeit(
    compute_func: Callable,
    data_generator: Callable,
    matrix_dim: str = None,
    warmup_tries: int = 10,
    tries: int = 10,
    task: str = None,
    trace_dir: str = None
) -> list[float]:
    """
    Simple utility to time a function, ensuring no cache hits
    by generating new data for each iteration.

    Args:
        compute_func: The jitted function to benchmark.
        data_generator: A function that returns a tuple of device-placed args
                        for the compute_func.
        warmup_tries: Number of warmup iterations.
        tries: Number of timed measurement iterations.
        task: Name of the task for logging.
    """
    assert task is not None
    print(f"[{task}] Running warmup loop with {warmup_tries} tries...")
    result = None # To hold the last result for block_until_ready
    for _ in range(warmup_tries):
        # 1. Generate new data for each iteration
        data_args = data_generator()
        # 2. Run compute
        result = compute_func(*data_args)
        # 3. Block on the run
        jax.block_until_ready(result)
    print(f"[{task}] Warmup complete.")

    arg_shapes = [arg.shape for arg in data_args]
    arg_dtypes = [arg.dtype for arg in data_args]
    if isinstance(result, list) or isinstance(result, tuple):
        result_shapes = [r.shape for r in result]
        result_dtypes = [r.dtype for r in result]
    else:
        result_shapes = result.shape
        result_dtypes = result.dtype
    print(f"[{task}] Verified global shapes: {arg_shapes} -> {result_shapes}")
    print(f"[{task}] Verified global dtypes: {arg_dtypes} -> {result_dtypes}")

    if trace_dir is not None:
        if task == "rmsnorm":
		    # If the task is RMSNorm, we specifically target "copy-done" events.
		    # This is often done to capture the time of the asynchronous memory transfer
		    # needed for the normalization layer's input data.
            event_name_str_list = ["copy-done"]
        else:
		    # For all other tasks, use an empty list.
            event_name_str_list = []

        return iteration_timeit_from_trace(
                compute_func,
                data_generator,
                matrix_dim=matrix_dim,
                tries=tries,
                task=task,
                trace_dir=trace_dir,
                event_name_str_list=event_name_str_list)

    outcomes_ms = []
    print(f"[{task}] Running measurement loop with {tries} tries...")

    for i in range(tries):
        # 1. Generate NEW random data (meets "no cache hit" rule)
        data_args = data_generator()
        jax.devices()  # Force synchronization across devices

        # Start timer just before the compute call
        s_time = datetime.datetime.now()

        # 2. Run the operation
        result = compute_func(*data_args)

        # 3. Block until operation is complete
        jax.block_until_ready(result)

        e_time = datetime.datetime.now()
        outcomes_ms.append(1000 * (e_time - s_time).total_seconds())
    return outcomes_ms

def simple_timeit(
    f, *args, matrix_dim=None, tries=10, task=None, trace_dir=None
) -> float:
    """Simple utility to time a function for multiple runs."""
    assert task is not None

    if trace_dir:
        return timeit_from_trace(
            f, *args, matrix_dim=matrix_dim, tries=tries, task=task, trace_dir=trace_dir
        )

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


def find_sparsecore_usage_from_xplane(log_dir: str) -> xplane_pb2.XSpace:
    """Extract the XSpace object from the log directory.

    Returns:
      An XSpace protobuf object.
    """
    print("find_sparsecore_usage_from_xplane: ", log_dir)

    # Handle partial log_dir
    if not (pathlib.Path(log_dir) / "plugins" / "profile").exists():
        potential_dirs = glob.glob(f"{log_dir}*")
        potential_dirs = [d for d in potential_dirs if os.path.isdir(d)]
        potential_dirs.sort(key=os.path.getmtime, reverse=True)

        for d in potential_dirs:
            d_path = pathlib.Path(d)
            if (d_path / "plugins" / "profile").exists():
                log_dir = d
                print(f"Updated log_dir to match partial path: {log_dir}")
                break

            # Check subdirectories recursively
            candidates = list(d_path.glob("**/plugins/profile"))
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                log_dir = str(latest.parent.parent)
                print(f"Updated log_dir via recursive search: {log_dir}")
                break

    trace_folders = (
        pathlib.Path(log_dir).absolute() / "plugins" / "profile"
    ).iterdir()
    latest_trace_folder = max(trace_folders, key=os.path.getmtime)

    # XPlane files usually end with .xplane.pb
    xplane_files = list(latest_trace_folder.glob("*.xplane.pb"))
    try:
        (xplane_file,) = xplane_files
    except ValueError as value_error:
        raise ValueError(
            f"Invalid trace folder: {latest_trace_folder}. Expected 1"
            f" '*.xplane.pb' file, but found {len(xplane_files)}."
        ) from value_error

    with open(xplane_file, "rb") as f:
        serialized_space = f.read()

    space = xplane_pb2.XSpace()
    space.ParseFromString(serialized_space)
    # print("space: ", space)
    sparsecore_found = False
    for _, plane in enumerate(space.planes):
        print("plane: ", plane.name)
        if "SparseCore" in plane.name:
            sparsecore_found = True
            break
    return sparsecore_found


def get_metrics_from_trace(trace: dict[str, Any], task: str) -> list[float]:
    # Check if the given task name is a collective with corresponding TPU opertion.
    # This is a workaround and should be reverted or refactored in future.
    if task in TARGET_TASK_NAME_COLLECTIVES_MAP:
        try:
            task = TARGET_TASK_NAME_COLLECTIVES_MAP[task]
            return get_metrics_from_trace_tpu(trace, task)
        except:
            return [-1.0]
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


def get_metrics_from_trace_tpu(trace: dict[str, Any], task: str) -> list[float]:
    event_matcher = re.compile(task)

    if "traceEvents" not in trace:
        raise KeyError("Key 'traceEvents' not found in trace.")

    events = []
    for e in trace["traceEvents"]:
        if "name" in e and event_matcher.match(e["name"]):
            events.append(e)

    # For each trace, find the TPU with smallest `pid` value and consider it to be TPU-0
    min_pid = min([e["pid"] for e in events])
    events_from_min_pid = [e for e in events if e["pid"] == min_pid]
    try:
        durations_ms = [
            float(e["args"]["device_duration_ps"]) / 1e9 for e in events_from_min_pid
        ]
    except KeyError:
        print("KeyError: Key 'device_duration_ps' not found in the event object")
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


def timeit_from_trace(
    f, *args, matrix_dim=None, tries=10, task=None, trace_dir=None, event_name_str_list: list[str] = None
) -> float:
    """
    Time a function with jax.profiler and get the run time from the trace.
    """
    LOCAL_TRACE_DIR = "/tmp/microbenchmarks_tmptrace"

    jax.block_until_ready(f(*args))  # warm it up!

    if matrix_dim is not None:
        trace_name = f"{task}_dim_{matrix_dim}"
    else:
        trace_name = f"t_{task}_" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )

    trace_full_dir = f"{trace_dir}/{trace_name}"
    tmp_trace_dir = trace_full_dir
    # If the trace_dir isn't a local path, create one for dumping the trace for parsing and getting metrics.
    if trace_dir and not is_local_directory_path(trace_dir):
        tmp_trace_dir = f"{LOCAL_TRACE_DIR}/{trace_name}"
    print(trace_dir)
    with jax.profiler.trace(tmp_trace_dir):
        for _ in range(tries):
            jax.devices()  # Force synchronization across devices
            with jax.profiler.TraceAnnotation(task):
                jax.block_until_ready(f(*args))

    trace = get_trace(tmp_trace_dir)

    if trace_full_dir != tmp_trace_dir:
        # Upload the traces to desired location
        upload_to_storage(trace_dir=trace_full_dir, local_file=tmp_trace_dir)

    if event_name_str_list is not None:
        return iteration_get_event_metrics_from_trace(trace, event_name_str_list=event_name_str_list)

    return iteration_get_metrics_from_trace(trace)


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


def load_yaml_config(config_path: str) -> Dict[str, Any] | None:
    """Loads a YAML config file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return None


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
            "max": np.max(self.metrics_list),
            "num_runs": len(self.metrics_list),
            "min": np.min(self.metrics_list),
            # "all_values": json.dumps(self.metrics_list),
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


def rename_xla_dump(
    tmp_xla_dump_dir: str,
    dest_xla_dump_dir: str,
    benchmark_name: str,
    benchmark_param: Dict[str, Any],
):
    """
    Finds the latest XLA dump file matching '*jit_f*before_optimizations*.txt',
    then identifies all other files that share the same 'jit_f.[unique_id]' identifier
    and renames them to 'benchmark_name_serialized_params.original_suffix_with_extension'.
    """

    serialized_benchmark_param = "_".join(
        f"{key}_{value}" for key, value in benchmark_param.items()
    )
    anchor_pattern = os.path.join(tmp_xla_dump_dir, "*jit_f*before_optimizations*.txt")
    matching_anchor_files = glob.glob(anchor_pattern)

    if not matching_anchor_files:
        print(
            f"No files found for anchor pattern: '{anchor_pattern}'. No files will be renamed."
        )
        return

    # Sort anchor files by modification time (latest first)
    matching_anchor_files.sort(key=os.path.getmtime, reverse=True)
    latest_anchor_file = matching_anchor_files[0]

    # Example: 'module_0080.jit_f.cl_747713181.before_optimizations.txt'
    # This will extract 'module_0080.jit_f.cl_747713181'
    filename_base = os.path.basename(latest_anchor_file)
    jit_id_match = re.search(r"(module.*jit_f\.[^.]+)", filename_base)

    if not jit_id_match:
        print(
            f"Could not extract 'jit_f.[unique_id]' from '{filename_base}'. Cannot proceed with renaming."
        )
        return

    common_jit_id_prefix = jit_id_match.group(1)

    # Find all files in the directory that contain this specific common_jit_id_prefix
    all_related_files_pattern = os.path.join(
        tmp_xla_dump_dir, f"*{common_jit_id_prefix}*"
    )
    all_related_files = glob.glob(all_related_files_pattern)

    if not all_related_files:
        print(
            f"No files found containing '{common_jit_id_prefix}'. This is unexpected if an anchor was found."
        )
        return

    new_base_name = f"{benchmark_name}_{serialized_benchmark_param}"
    after_optimizations_path = input_shape = output_shape = replica_groups = first_replica_group = None

    for original_filepath in all_related_files:
        original_filename = os.path.basename(original_filepath)
        original_suffix_with_extension = ""

        # Find the specific suffix part *after* the common_jit_id_prefix.
        # This regex looks for the common_jit_id_prefix, then captures everything after it,
        # ensuring it starts with a dot if there's more.
        # Example: if original_filename is 'module_0080.jit_f.cl_747713181.after_codegen.txt'
        # and common_jit_id_prefix is 'jit_f.cl_747713181'
        # we want to capture '.after_codegen.txt'
        suffix_match = re.search(
            re.escape(common_jit_id_prefix) + r"(\..*)", original_filename
        )

        if suffix_match:
            original_suffix_with_extension = suffix_match.group(
                1
            )  # e.g., '.after_codegen.txt'

        new_filename = f"{new_base_name}{original_suffix_with_extension}"
        new_filepath = os.path.join(dest_xla_dump_dir, new_filename)
        if "after_optimizations.txt" in original_suffix_with_extension:
            after_optimizations_path = new_filepath

        if original_filepath == new_filepath:
            print(
                f"Skipping: '{original_filename}' already has the desired name or path."
            )
            continue

        # Copy the renamed files to desired location
        if is_local_directory_path(dest_xla_dump_dir):
            try:
                os.makedirs(dest_xla_dump_dir, exist_ok=True)
                shutil.copy(original_filepath, new_filepath)
            except Exception as e:
                print(
                    f"An unexpected error occurred while copy '{original_filepath}': {e}"
                )
        else:
            upload_to_storage(trace_dir=new_filepath, local_file=original_filepath)
    print(f"The XLA dump is stored in {dest_xla_dump_dir}")
    if after_optimizations_path:
        input_shape, output_shape, replica_groups, first_replica_group = (
            extract_hlo_features_from_file(after_optimizations_path)
        )
    else:
        print(
            "No files found with 'after_optimizations.txt' suffix. "
            "Please check the XLA dump directory."
        )
    return json.dumps({
        "after_optimizations_path": after_optimizations_path,
        "hlo_input_shape": input_shape,
        "hlo_output_shape": output_shape,
        "hlo_replica_groups": replica_groups,
        "hlo_first_replica_group": first_replica_group,
    })

def extract_hlo_features_from_file(hlo_file_path: str) -> Tuple[str | None, str | None, str | None, list[int] | None]:
    """
    Extracts input shape, output shape, and replica groups from an HLO file.

    Args:
      hlo_file_path: Path to the HLO dump file (e.g., after_optimizations.txt).

    Returns:
      A tuple containing (input_shape, output_shape, replica_groups_str, first_replica_group),
      or (None, None, None, None) if extraction fails.
    """
    input_shape = None
    output_shape = None
    replica_groups_str = None
    first_replica_group = None

    try:
        with open(hlo_file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: HLO file not found at {hlo_file_path}")
        return None, None, None, None

    # Extract input/output shapes from HloModule line
    # Example: HloModule jit_f, ..., entry_computation_layout={(f32[32,128]{...})->f32[128,128]{...}}
    layout_match = re.search(r"entry_computation_layout={\((.*?)\)->(.*?)}", content)
    if layout_match:
        input_shape = layout_match.group(1)
        output_shape = layout_match.group(2)
        # Further clean shape if layout info is present, e.g., f32[1,2]{1,0} -> f32[1,2]
        input_shape = re.sub(r"{.*}", "", input_shape)
        output_shape = re.sub(r"{.*}", "", output_shape)
    else:
        print(f"Could not find entry_computation_layout in {hlo_file_path} to extract shapes.")

    # Extract replica groups
    # Example: replica_groups={{0,1},{2,3}}, dimensions...
    rg_match = re.search(r"replica_groups=({{[0-9,]+(?:},{[0-9,]+)*}})", content, re.DOTALL)
    if rg_match:
        replica_groups_str = rg_match.group(1)
        try:
            content_rg = replica_groups_str[2:-2]
            first_group_str = content_rg.split('},{')[0]
            first_replica_group = [int(x) for x in first_group_str.split(',')]
        except Exception as e:
            print(f'Could not parse replica_groups in hlo_text: {e}')
            first_replica_group = None
    else:
        print(f"Could not find replica_groups in {hlo_file_path}.")

    return input_shape, output_shape, replica_groups_str, first_replica_group
def get_lhs_named_shading(mesh, strategy: ShardingStrategy):
    match strategy:
        case ShardingStrategy.NO_SHARDING:
            return NamedSharding(mesh, P(None, None))
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_M:
            return NamedSharding(mesh, P("device", None))
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M:
            return NamedSharding(mesh, P("device", None))
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_N:
            return NamedSharding(mesh, P(None, None))
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
            return NamedSharding(mesh, P(None, None))
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_K:
            return NamedSharding(mesh, P(None, "device"))


def get_rhs_named_shading(mesh, strategy: ShardingStrategy):
    match strategy:
        case ShardingStrategy.NO_SHARDING:
            return NamedSharding(mesh, P(None, None))
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_M:
            return NamedSharding(mesh, P(None, None))
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M:
            return NamedSharding(mesh, P(None, None))
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_N:
            return NamedSharding(mesh, P(None, "device"))
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
            return NamedSharding(mesh, P(None, "device"))



def get_out_sharding(strategy: ShardingStrategy):
    match strategy:
        case ShardingStrategy.NO_SHARDING:
            return P(None, None)
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_M:
            return P("device", None)
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M:
            return P("device", None)
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_N:
            return P(None, "device")
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
            return P(None, "device")



def get_rowwise_named_shading(mesh, strategy: ShardingStrategy):
    match strategy:
        case ShardingStrategy.NO_SHARDING:
            return NamedSharding(mesh, P(None, None))
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_M:
            return NamedSharding(mesh, P("device", None))
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M:
            return NamedSharding(mesh, P("device", None))
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_N:
            assert False, f"ShardingStrategy is wrong for this ops: {strategy}"
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
            return False, f"ShardingStrategy is wrong for this ops: {strategy}"


def get_output_named_shading(mesh, strategy: ShardingStrategy):
    match strategy:
        case ShardingStrategy.NO_SHARDING:
            return NamedSharding(mesh, P(None, None))
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_M:
            return NamedSharding(mesh, P("device", None))
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M:
            return NamedSharding(mesh, P("device", None))
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_N:
            return NamedSharding(mesh, P(None, "device"))
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
            return NamedSharding(mesh, P(None, "device"))


def handle_per_device_based_on_sharding(value, strategy: ShardingStrategy):
    match strategy:
        case ShardingStrategy.NO_SHARDING:
            return value
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_M:
            return value // jax.device_count()
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M:
            return value // 2
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_N:
            return value // jax.device_count()
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
            return value // 2



def handle_all_devices_based_on_sharding(value: int, strategy: ShardingStrategy):
    match strategy:
        case ShardingStrategy.NO_SHARDING:
            return value * jax.device_count()
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_M:
            return value
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M:
            return value * jax.device_count() // 2
        case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_N:
            return value
        case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
            return value * jax.device_count() // 2



def handle_based_on_sharding(value: int, strategy: ShardingStrategy):
    total_value = value
    value = handle_per_device_based_on_sharding(value, strategy)
    total_value = handle_all_devices_based_on_sharding(total_value, strategy)
    return value, total_value


def create_mesh(strategy: ShardingStrategy) -> Mesh:
    """Creates a mesh."""
    if (
        strategy == ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M
        or strategy == ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N
    ):
        num_devices = jax.device_count()
        assert num_devices % 2 == 0, "Total devices must be divisible by 2 (chip size)"
        num_chips = num_devices // 2
        mesh_shape = (num_chips, 2)
        mesh_axes = ("chip", "device")
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(mesh_shape), mesh_axes)
    else:
        mesh = Mesh(np.array(jax.devices()), axis_names="device")
    return mesh


def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_param_keys = {
        "time_ms_list",
        "total_flops",
        "total_flops_all_devices",
        "peak_TFLOPS_per_device",
        "total_bytes",
        "total_bytes_all_devices",
    }
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_param_keys
    }
    return metadata


def unified_flops_metrics(
    m: int,
    n: int,
    k: int,
    time_ms_list: list[float],
    total_flops: int,
    total_flops_all_devices: int,
    peak_TFLOPS_per_device: float,
    dtype: str = None,
    b: int = None,
) -> Dict[str, Any]:
    """Calculates the metrics for the naive matmul benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    tflops_per_sec_list = [
        total_flops / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    tflops_per_sec_all_devices = [
        total_flops_all_devices / average_time_s / 10**12
        for average_time_s in average_time_s_list
    ]
    mfu = [
        tflops_per_sec / peak_TFLOPS_per_device
        for tflops_per_sec in tflops_per_sec_list
    ]
    average_time_ms_statistics = MetricsStatistics(
        metrics_list=time_ms_list, metrics_name="step_time_ms"
    )
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec_per_device"
    )
    tflops_per_sec_all_devices_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_all_devices, metrics_name="tflops_per_sec"
    )
    mfu_statistics = MetricsStatistics(metrics_list=mfu, metrics_name="MFU")
    dtype_prefix = f"[{dtype}] " if dtype is not None else ""
    print(
        f"{dtype_prefix}"
        f"Total floating-point ops: {total_flops}, Step Time (median): {average_time_ms_statistics.statistics['p50']:.2f}, "
        f"Throughput (median): {tflops_per_sec_statistics.statistics['p50']:.2f} TFLOP / second / device, "
        f"TotalThroughput (median): {tflops_per_sec_all_devices_statistics.statistics['p50']:.2f} TFLOP / second, "
        f"MFU: {mfu_statistics.statistics['p50']:.2%}"
    )
    # print()
    # time_ms_list =

    # Gather the metrics to report.
    metadata.update(
        {
            "StepTime(median,ms)": average_time_ms_statistics.statistics["p50"],
            "Throughput(median,TFLOP/s/device)": tflops_per_sec_statistics.statistics[
                "p50"
            ],
            "TotalThroughput(median,TFLOP/s)": tflops_per_sec_all_devices_statistics.statistics[
                "p50"
            ],
            "MFU": mfu_statistics.statistics["p50"],
            "total_flops": total_flops,
            # "all_time_ms_list":  f"{json.dumps(time_ms_list)}",
        }
    )
    metrics.update(average_time_ms_statistics.serialize_statistics())
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics.update(tflops_per_sec_all_devices_statistics.serialize_statistics())
    metrics.update(mfu_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics


def unified_bytes_metrics(
    m: int,
    n: int,
    time_ms_list: list[float],
    total_bytes: int,
    total_bytes_all_devices: int = 1e9,
    quant_dtype: str = None,
    dtype: str = None,
) -> Dict[str, Any]:
    """Calculates the metrics for the naive matmul benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    gigabytes_per_sec_list = [
        total_bytes / average_time_s / 10**9 for average_time_s in average_time_s_list
    ]
    digabytes_per_sec_all_devices = [
        total_bytes_all_devices / average_time_s / 10**9
        for average_time_s in average_time_s_list
    ]
    average_time_ms_statistics = MetricsStatistics(
        metrics_list=time_ms_list, metrics_name="step_time_ms"
    )
    gigabytes_per_sec_statistics = MetricsStatistics(
        metrics_list=gigabytes_per_sec_list, metrics_name="Gbytes_per_sec_per_device"
    )
    gigabytes_per_sec_all_devices_statistics = MetricsStatistics(
        metrics_list=digabytes_per_sec_all_devices, metrics_name="Gbytes_per_sec"
    )
    type_prefix = ""
    # Gather the metrics to report.
    if quant_dtype is not None:
        metadata.update({"quant_dtype": quant_dtype})
        metrics.update({"quant_dtype": quant_dtype})
        type_prefix = f"[q={quant_dtype}] "
    if dtype is not None:
        metadata.update({"dtype": dtype})
        metrics.update({"dtype": dtype})
        type_prefix = f"[d={dtype}] "
    print(
        f"{type_prefix}"
        f"Total bytes: {total_bytes}, Step Time (median): {average_time_ms_statistics.statistics['p50']:.2f}, Throughput (median):"
        f" {gigabytes_per_sec_statistics.statistics['p50']:.2f} GBytes / second / device,"
        f" TotalThroughput (median): {gigabytes_per_sec_all_devices_statistics.statistics['p50']:.2f} GBytes / second"
    )
    print()
    metadata.update(
        {
            "StepTime(median,ms)": average_time_ms_statistics.statistics["p50"],
            "Throughput(median,GBytes/s/device)": gigabytes_per_sec_statistics.statistics[
                "p50"
            ],
            "TotalThroughput(median,GBytes/s)": gigabytes_per_sec_all_devices_statistics.statistics[
                "p50"
            ],
            "total_bytes": total_bytes,
        }
    )
    metrics.update(average_time_ms_statistics.serialize_statistics())
    metrics.update(gigabytes_per_sec_statistics.serialize_statistics())
    metrics.update(gigabytes_per_sec_all_devices_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics

def str_to_dtype(dtype_str: str) -> jnp.dtype:
    """Converts a string identifier to a JAX numpy dtype."""
    if dtype_str.lower() == "fp8":
        return jnp.float8_e4m3fn
    elif dtype_str.lower() == "bf16":
        return jnp.bfloat16
    elif dtype_str.lower() == "fp16":
        return jnp.float16
    elif dtype_str.lower() == "fp32":
        return jnp.float32
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")

def get_peak_flops_multiplier(in_dtype_str: str) -> float:
    """
    Returns the peak FLOPS multiplier relative to the baseline
    (PEAK_FLOPS_PER_DEVICE) based on the input data type.
    """
    in_dtype_lower = in_dtype_str.lower()
    if in_dtype_lower in ("fp8", "float8_e4m3fn"):
        # FP8 is 2x faster than BF16
        # The baseline PEAK_FLOPS_PER_DEVICE is 1153.5 * 2 = 2307, which is FP8 peak.
        # So the multiplier should be 1.0
        return 1.0
    elif in_dtype_lower in ("bf16", "bfloat16", "fp16", "float16"):
        # BF16/FP16 is 2x slower than FP8 peak
        return 0.5
    elif in_dtype_lower in ("fp32", "float32"):
        # FP32 is 4x slower than FP8 peak
        return 0.25
    elif in_dtype_lower in ("fp4", "float4_e2m1fn"):
        # FP4/INT4 is treated the same as FP8
        return 1.0
    else:
        raise RuntimeError(f"{in_dtype_lower} is not supported for setting peak_flops_multiplier.")