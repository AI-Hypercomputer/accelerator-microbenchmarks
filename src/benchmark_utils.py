"""Utility functions for microbenchmarking."""

import datetime
import os
from typing import Any, Dict, Tuple, Callable
import glob

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
import shutil
from common import MARKER
from enum import Enum, auto
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# The dictionary to map a JAX (collective) function to its main HLO.
TARGET_TASK_NAME_COLLECTIVES_MAP = {
    "all_to_all_ici_op": r"all-to-all.[0-9]+",
    "all_gather_ici_op": r"all-gather.[0-9]+",
    "psum_ici_op": r"all-reduce.[0-9]+",
    "ppermute_ici_op": r"collective-permute.[0-9]+",
}

class ShardingStrategy(Enum):
    """Defines different sharding strategies for tensors."""
    NO_SHARDING = auto()
    SHARDING_ON_ALL_DEVICES_WITH_M = auto()
    SHARDING_ON_SINGLE_CHIP_WITH_M = auto() # Only sharding on the two core of one single chip
    SHARDING_ON_ALL_DEVICES_WITH_N = auto()
    SHARDING_ON_SINGLE_CHIP_WITH_N = auto()

def iteration_timeit_from_trace(
    compute_func: Callable,
    data_generator: Callable,
    matrix_dim: str=None,
    tries: int=10, 
    task: str = None,
    trace_dir: str = None) -> list[float]:
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
    return iteration_get_metrics_from_trace(trace)

def iteration_get_metrics_from_trace(trace: dict[str, Any]) -> list[float]:
    marker_done_events = []
    for event in trace["traceEvents"]:
        args = event.get("args", {})
        tf_op = args.get("tf_op", "")
        if MARKER in tf_op:
            marker_done_events.append(event)

    # print(marker_done_events)
    min_pid = min([e["pid"] for e in marker_done_events])
    events_from_min_pid = [e for e in marker_done_events if e["pid"] == min_pid]
    # print(events_from_min_pid)
    durations_ms = [
        sum(float(e["args"]["device_duration_ps"]) / 1e9 for e in events_from_min_pid)
    ]
    print("durations_ms: ", durations_ms)
    return durations_ms

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
        return iteration_timeit_from_trace(compute_func, data_generator, matrix_dim=matrix_dim, tries=tries, task=task, trace_dir=trace_dir)

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



def simple_timeit(f, *args, matrix_dim=None, tries=10, task=None, trace_dir=None) -> float:
    """Simple utility to time a function for multiple runs."""
    assert task is not None

    if trace_dir:
        return timeit_from_trace(f, *args, matrix_dim=matrix_dim, tries=tries, task=task, trace_dir=trace_dir)

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


def get_metrics_from_trace(trace: dict[str, Any], task: str) -> list[float]:

    # Check if the given task name is a collective with corresponding TPU opertion.
    # This is a workaround and should be reverted or refactored in future.
    if task in TARGET_TASK_NAME_COLLECTIVES_MAP:
        try:
            task = TARGET_TASK_NAME_COLLECTIVES_MAP[task]
            return get_metrics_from_trace_tpu(trace, task)
        except:
            return [-1.]
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
        durations_ms = [float(e["args"]["device_duration_ps"]) / 1e9 for e in events_from_min_pid]
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


def timeit_from_trace(f, *args, matrix_dim=None, tries=10, task=None, trace_dir=None) -> float:
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

    for original_filepath in all_related_files:
        original_filename = os.path.basename(original_filepath)

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
    if strategy == ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M or strategy == ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
        num_devices = jax.device_count()
        assert num_devices % 2 == 0, "Total devices must be divisible by 2 (chip size)"
        num_chips = num_devices // 2
        mesh_shape = (num_chips, 2)
        mesh_axes = ('chip', 'device')
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(mesh_shape), mesh_axes)
    else:
        mesh = Mesh(np.array(jax.devices()), axis_names="device")
    return mesh

def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_param_keys = {"time_ms_list", "total_flops", "total_flops_all_devices", "peak_TFLOPS_per_device", "total_bytes", "total_bytes_all_devices"}
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_param_keys
    }
    return metadata

def unified_flops_metrics(
    m: int, n: int, k: int, time_ms_list: list[float], total_flops: int, total_flops_all_devices: int, peak_TFLOPS_per_device: float
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
        total_flops_all_devices / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    mfu = [
        tflops_per_sec/peak_TFLOPS_per_device for tflops_per_sec in tflops_per_sec_list
    ]
    average_time_ms_statistics = MetricsStatistics(
        metrics_list=time_ms_list, metrics_name="step_time_ms"
    )
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec_pre_device"
    )
    tflops_per_sec_all_devices_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_all_devices, metrics_name="tflops_per_sec"
    )
    mfu_statistics=MetricsStatistics(
        metrics_list=mfu, metrics_name="MFU"
    )
    print(
        f"Total floating-point ops: {total_flops}, Step Time (median): {average_time_ms_statistics.statistics['p50']:.2f}, "
        f"Throughput (median): {tflops_per_sec_statistics.statistics['p50']:.2f} TFLOP / second / device, "
        f"TotalThroughput (median): {tflops_per_sec_all_devices_statistics.statistics['p50']:.2f} TFLOP / second, "
        f"MFU: {mfu_statistics.statistics['p50']:.2%}"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "StepTime(median,ms)": average_time_ms_statistics.statistics['p50'],
            "Throughput(median,TFLOP/s/device)": tflops_per_sec_statistics.statistics['p50'],
            "TotalThroughput(median,TFLOP/s)": tflops_per_sec_all_devices_statistics.statistics['p50'],
            "MFU": mfu_statistics.statistics['p50'],
            "total_flops": total_flops,
        }
    )
    metrics.update(average_time_ms_statistics.serialize_statistics())
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics.update(tflops_per_sec_all_devices_statistics.serialize_statistics())
    metrics.update(mfu_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics

def unified_bytes_metrics( 
    m: int, n: int, time_ms_list: list[float], total_bytes: int, total_bytes_all_devices: int=1e9
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
        total_bytes_all_devices / average_time_s / 10**9 for average_time_s in average_time_s_list
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
    print(
        f"Total bytes: {total_bytes}, Step Time (median): {average_time_ms_statistics.statistics['p50']:.2f}, Throughput (median):"
        f" {gigabytes_per_sec_statistics.statistics['p50']:.2f} GBytes / second / device,"
        f" TotalThroughput (median): {gigabytes_per_sec_all_devices_statistics.statistics['p50']:.2f} GBytes / second"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "StepTime(median,ms)": average_time_ms_statistics.statistics['p50'],
            "Throughput(median,GBytes/s/device)": gigabytes_per_sec_statistics.statistics['p50'],
            "TotalThroughput(median,GBytes/s)": gigabytes_per_sec_all_devices_statistics.statistics['p50'],
            "total_bytes": total_bytes,
        }
    )
    metrics.update(average_time_ms_statistics.serialize_statistics())
    metrics.update(gigabytes_per_sec_statistics.serialize_statistics())
    metrics.update(gigabytes_per_sec_all_devices_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics