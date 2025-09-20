"""This script runs microbenchmarks and collects metrics.

Sample usage (on TPU vm):
  $ python run_benchmark.py --config=configs/benchmark_collectives.yaml
"""

import argparse
import csv
import datetime
import importlib
import inspect
import itertools
import random
import string
from typing import Any, Callable, Dict, List, Tuple
from benchmark_utils import maybe_write_metrics_file, rename_xla_dump, MetricsStatistics
import jax
import yaml
import ray
from concurrent.futures import ThreadPoolExecutor
import os
import copy
import pandas as pd
import ast
import json

COLLECTIVE_BENCHMARK_MAP = {
    "all_gather": "benchmark_collectives.all_gather_benchmark",
    "psum": "benchmark_collectives.psum_benchmark",
    "psum_scatter": "benchmark_collectives.psum_scatter_benchmark",
    "all_to_all": "benchmark_collectives.all_to_all_benchmark",
    "ppermute": "benchmark_collectives.ppermute_benchmark",
}

MATMUL_BENCHMARK_MAP = {
    "naive_matmul": "benchmark_matmul.naive_matmul",
    "single_host_naive_matmul": "benchmark_matmul.single_host_naive_matmul",
    "multilayer_collective_matmul": ("benchmark_matmul.multilayer_collective_matmul"),
    "collective_matmul_one_direction": (
        "benchmark_matmul.collective_matmul_one_direction"
    ),
    "collective_matmul_two_directions": (
        "benchmark_matmul.collective_matmul_two_directions"
    ),
}
CONVOLUTION_BENCHMARK_MAP = {
    "numpy_convolve": "benchmark_convolution.numpy_convolve",
    "scipy_signal_convolve": "benchmark_convolution.scipy_signal_convolve",
    "scipy_signal_convolve2d": "benchmark_convolution.scipy_signal_convolve2d",
    "lax_conv_general_dilated": ("benchmark_convolution.lax_conv_general_dilated"),
}
ATTENTION_BENCHMARK_MAP = {
    "naive_attention": "benchmark_attention.naive_attention_benchmark",
    "pallas_flash_attention": ("benchmark_attention.pallas_flash_attention_benchmark"),
    "splash_attention": "benchmark_attention.splash_attention_benchmark",
    "flax_nnx_attention": "benchmark_attention.flax_nnx_attention_benchmark",
    "flax_linen_attention": ("benchmark_attention.flax_linen_attention_benchmark"),
    "keras_attention": "benchmark_attention.keras_attention_benchmark",
}
HBM_BENCHMARK_MAP = {
    "single_chip_hbm_copy": "benchmark_hbm.single_chip_hbm_copy",
}
BENCHMARK_MAP = {}
BENCHMARK_MAP.update(COLLECTIVE_BENCHMARK_MAP)
BENCHMARK_MAP.update(MATMUL_BENCHMARK_MAP)
BENCHMARK_MAP.update(CONVOLUTION_BENCHMARK_MAP)
BENCHMARK_MAP.update(ATTENTION_BENCHMARK_MAP)
BENCHMARK_MAP.update(HBM_BENCHMARK_MAP)


# Mapping from dtype string to actual dtype object
dtype_mapping = {
    "bfloat16": jax.numpy.bfloat16,
    "float32": jax.numpy.float32,
    "int32": jax.numpy.int32,
    # Add other dtypes as needed
}

# Always dump HLOs
TMP_XLA_DUMP_DIR = "/tmp/microbenchmarks/hlo_graphs"
os.environ["XLA_FLAGS"] = f"--xla_dump_to={TMP_XLA_DUMP_DIR}"


def get_benchmark_config(config_path: str) -> Dict[str, Any]:
    """Load benchmark configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Dynamically load the benchmark functions.
def get_benchmark_functions(
    benchmark_name: str,
) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Dynamically load the benchmark function and its calculate_metrics function from the predefined map."""
    if benchmark_name not in BENCHMARK_MAP:
        raise ValueError(f"Benchmark {benchmark_name} is not defined in the map.")

    module_path, func_name = BENCHMARK_MAP[benchmark_name].rsplit(".", 1)

    # Get the benchmark function
    try:
        module = importlib.import_module(f"{module_path}")
        benchmark_func = getattr(module, func_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Unable to import {module_path}.{func_name}. ModuleNotFoundError {e}."
        ) from e
    except AttributeError as e:
        raise ValueError(
            f"Unable to import {module_path}.{func_name}. AttributeError {e}."
        ) from e

    # Get the calculate_metrics function
    try:
        calculate_metrics_func = getattr(module, f"{func_name}_calculate_metrics")
    except AttributeError:
        raise ValueError(
            f"Calculate metrics function for {benchmark_name} not found."
        ) from None

    return benchmark_func, calculate_metrics_func


def preprocess_benchmark_param(
    benchmark_param: Dict[str, Any], trace_dir: string = None
) -> Dict[str, Any]:
    """Preprocess the benchmark parameter before running the benchmark."""
    if "dtype" in benchmark_param:
        dtype_str = benchmark_param["dtype"]
        if dtype_str in dtype_mapping:
            benchmark_param["dtype"] = dtype_mapping[dtype_str]
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

    # Handle "SAME_AS_" parameters.
    # For example, if "n" is "SAME_AS_m", then "n" will
    # be set to the same value as "m".
    for key, value in benchmark_param.items():
        if isinstance(value, str) and value.startswith("SAME_AS_"):
            same_as_key = value.split("SAME_AS_")[1]
            if same_as_key not in benchmark_param:
                raise ValueError(
                    f"Parameter {same_as_key} not found in the benchmark_param."
                )
            benchmark_param[key] = benchmark_param[same_as_key]

    benchmark_param["trace_dir"] = trace_dir
    return benchmark_param


def generate_benchmark_params_sweeping(
    benchmark_sweep_params: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate benchmark parameters by sweeping through the specified ranges."""
    generated_params = []
    for sweep_params in benchmark_sweep_params:
        param_sets = {}
        for key, value in sweep_params.items():
            if key.endswith("_range"):
                key = key[:-6]  # Remove the last 6 characters (i.e., '_range')

            if isinstance(value, dict):
                # Extract the range and multiplier
                start = value.get("start")
                end = value.get("end")
                multiplier = value.get("multiplier", None)
                increase_by = value.get("increase_by", None)
                # Generate values in the range
                param_values = []
                current_value = start
                while current_value <= end:
                    param_values.append(current_value)
                    if multiplier:
                        current_value *= multiplier
                    elif increase_by:
                        current_value += increase_by
                    else:
                        raise ValueError(
                            "In sweep mode, user must provide either multiplier or"
                            " increase_by value."
                        )
                # Add the generated values to the param set
                param_sets[key] = param_values
            else:
                # If it's not a range, just add it as a list with one element
                param_sets[key] = [value]

        # Get parameter names in a fixed order
        param_names = list(param_sets.keys())

        # Generate all combinations using itertools.product
        combinations = [
            dict(zip(param_names, values))
            for values in itertools.product(*(param_sets[name] for name in param_names))
        ]
        generated_params += combinations

    return generated_params


def write_to_csv(csv_path: str, calculate_metrics_results: List[Dict[str, Any]]):
    """Writes benchmark metrics to a CSV file.

    This function takes a list of dictionaries, where each dictionary contains
    the 'metadata' and 'metrics' from a benchmark run. It processes each
    dictionary by flattening it, calculating additional statistics for specific
    fields (like 'ici_average_time_ms_list'), and then converting it into a
    pandas DataFrame. All resulting DataFrames are concatenated and written to
    the specified CSV file.

    Args:
        csv_path: The path to the output CSV file.
        calculate_metrics_results: A list of dictionaries with benchmark results.
    """
    if not calculate_metrics_results:
        raise ValueError("0 metrics results are collected.")
    if not isinstance(calculate_metrics_results[0], dict):
        raise ValueError("metrics result is not a dict.")

    def flatten_dict(current_dict: Dict) -> Dict:
        """Recursively flattens a nested dictionary."""
        output_dict = {}
        for key, val in current_dict.items():
            if isinstance(val, Dict):
                output_dict.update(flatten_dict(val))
            else:
                # Try to evaluate string-formatted literals (e.g., "[1, 2, 3]")
                try:
                    output_dict[key] = ast.literal_eval(val)
                except (ValueError, SyntaxError, TypeError):
                    # If it's not a valid literal, keep it as a string.
                    output_dict[key] = val
        return output_dict

    def convert_dict_to_df(target_dict: Dict) -> pd.DataFrame:
        """Converts a single benchmark result dictionary to a pandas DataFrame."""
        flattened_dict = flatten_dict(target_dict)
        
        # This section is specific to collective benchmarks that produce
        # 'ici_average_time_ms_list'.
        if "ici_average_time_ms_list" in flattened_dict:
            # Calculate statistics for the timing list.
            ici_average_time_ms_statistics = MetricsStatistics(
                metrics_list=flattened_dict["ici_average_time_ms_list"],
                metrics_name="ici_average_time_ms",
            ).statistics
            for key, val in ici_average_time_ms_statistics.items():
                flattened_dict["ici_average_time_ms_" + key] = val

            # Convert list to JSON string for CSV storage.
            flattened_dict["ici_average_time_ms_list"] = json.dumps(
                flattened_dict["ici_average_time_ms_list"]
            )

        df = pd.DataFrame(flattened_dict, index=[0])
        return df

    # TODO(hylin2002@)
    # This is a temporary workaround to generate a properly formatted CSV file for the output metrics.
    # We should revert this PR and refactor the code such that metrics object is a flatten dict that can be easily exported as a CSV.
    # For other information that requires nested structures, we should serialize it into a json file."
    try:
        df_list = [convert_dict_to_df(each) for each in calculate_metrics_results]
        df = pd.concat(df_list, ignore_index=True)

        df.to_csv(csv_path, index=False, sep="\t")

        print(f"Metrics written to CSV at {csv_path}.")
    except Exception as e:
        # Temporary workaround to catch all exceptions and print a warning as 
        # `lax_conv_general_dilated` benchmark fails during nightly test runs at `convert_dict_to_df`.
        print(f"Failed to write metrics to CSV: {e}")


def run_single_benchmark(benchmark_config: Dict[str, Any]):
    """Run a single benchmark with one or more configurations."""
    # Extract benchmark details
    benchmark_name = benchmark_config.get("benchmark_name")
    benchmark_params = benchmark_config.get("benchmark_params", [])
    benchmark_sweep_params = benchmark_config.get("benchmark_sweep_params", {})
    if benchmark_sweep_params:
        benchmark_params += generate_benchmark_params_sweeping(benchmark_sweep_params)
    csv_path = benchmark_config.get("csv_path")
    trace_dir = benchmark_config.get("trace_dir")
    xlml_metrics_dir = benchmark_config.get("xlml_metrics_dir")
    xla_dump_dir = benchmark_config.get("xla_dump_dir")
    warmup_tries = benchmark_config.get("warmup_tries")
    warmup_tries = warmup_tries if warmup_tries is not None else 10

    if not benchmark_name:
        raise ValueError("Each benchmark must have a 'benchmark_name'.")

    # Get the benchmark function
    benchmark_func, calculate_metrics_func = get_benchmark_functions(benchmark_name)

    print(f"\n{'=' * 30}Starting benchmark '{benchmark_name}'{'=' * 30}\n")

    # Run the benchmark
    calculate_metrics_results = []
    for benchmark_param in benchmark_params:
        original_benchmark_param = copy.deepcopy(benchmark_param)
        benchmark_param = preprocess_benchmark_param(
            benchmark_param, trace_dir=trace_dir
        )
        print(f"Running benchmark: {benchmark_name} with params: {benchmark_param}")
        test_start_time = (
            datetime.datetime.now(tz=datetime.timezone.utc).isoformat() + "Z"
        )  # "Z" indicates UTC
        benchmark_results = benchmark_func(**benchmark_param, warmup_tries=warmup_tries)
        test_end_time = (
            datetime.datetime.now(tz=datetime.timezone.utc).isoformat() + "Z"
        )

        # Filter benchmark_results to include only keys present in
        # calculate_metrics_func
        calculate_metrics_params = inspect.signature(calculate_metrics_func).parameters
        filtered_benchmark_results = {
            key: value
            for key, value in benchmark_results.items()
            if key in calculate_metrics_params
        }
        # Filter out certain parameters from benchmark_param, eg. "num_runs".
        benchmark_params_to_filter = ["num_runs", "trace_dir"]
        filtered_benchmark_param = {
            key: value
            for key, value in benchmark_param.items()
            if key not in benchmark_params_to_filter
        }
        metadata, metrics = calculate_metrics_func(
            **filtered_benchmark_param, **filtered_benchmark_results
        )
        calculate_metrics_results.append({"metadata": metadata, "metrics": metrics})
        if xlml_metrics_dir:
            maybe_write_metrics_file(
                xlml_metrics_dir,
                metrics,
                metadata,
                benchmark_name,
                test_start_time,
                test_end_time,
            )
        # Post process the xla dump
        if xla_dump_dir:
            rename_xla_dump(
                tmp_xla_dump_dir=TMP_XLA_DUMP_DIR,
                dest_xla_dump_dir=xla_dump_dir,
                benchmark_name=benchmark_name,
                benchmark_param=original_benchmark_param,
            )

    # Dump metrics to file.
    if csv_path:
        test_name = f"t_{benchmark_name}_" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )
        write_to_csv(f"{csv_path}/{test_name}.csv", calculate_metrics_results)


def main(config_path: str, multithreaded: bool):
    """Main function."""
    # Load configuration
    config = get_benchmark_config(config_path)
    benchmarks = config.get("benchmarks")
    if not benchmarks or not isinstance(benchmarks, list):
        raise ValueError("Configuration must contain a 'benchmarks' list.")

    # Clear the tmp dirs.
    if os.path.exists(TMP_XLA_DUMP_DIR):
        for filename in os.listdir(TMP_XLA_DUMP_DIR):
            file_path = os.path.join(TMP_XLA_DUMP_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    if multithreaded:
        ray.init(
            runtime_env=ray.runtime_env.RuntimeEnv(
                address="ray://tpu-ray-cluster-head-svc:10001",
                env_vars={
                    "XLA_IR_DEBUG": "1",
                    "XLA_HLO_DEBUG": "1",
                    "PJRT_DEVICE": "TPU",
                    # "LIBTPU_INIT_ARGS": "--xla_tpu_scoped_vmem_limit_kib=25602",
                },
            )
        )

        # Calculate the number of TPU hosts within our Ray cluster...
        # num_hosts = int(ray.available_resources()["TPU"]) // 4
        print(ray.available_resources())
        # print("Num hosts detected: %d", num_hosts)

        for benchmark_config in benchmarks:
            run_benchmark_multithreaded(benchmark_config)

    else:
        for benchmark_config in benchmarks:
            run_single_benchmark(benchmark_config)


def run_benchmark_multithreaded(benchmark_config):
    # Extract benchmark details
    benchmark_name = benchmark_config.get("benchmark_name")
    benchmark_params = benchmark_config.get("benchmark_params", [])
    benchmark_sweep_params = benchmark_config.get("benchmark_sweep_params", {})
    if benchmark_sweep_params:
        benchmark_params += generate_benchmark_params_sweeping(benchmark_sweep_params)
    csv_path = benchmark_config.get("csv_path")
    if not benchmark_name:
        raise ValueError("Each benchmark must have a 'benchmark_name'.")
    warmup_tries = benchmark_config.get("warmup_tries")
    warmup_tries = warmup_tries if warmup_tries is not None else 10

    # Get the benchmark function
    benchmark_func, calculate_metrics_func = get_benchmark_functions(benchmark_name)

    print(f"\n{'=' * 30}Starting benchmark '{benchmark_name}'{'=' * 30}\n")

    # Start a trace if requested
    test_name = f"t_{benchmark_name}_" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=10)
    )

    # Preprocess benchmark parameters
    preprocessed_benchmark_params = [
        preprocess_benchmark_param(benchmark_param, trace_dir=None)
        for benchmark_param in benchmark_params
    ]
    calculate_metrics_results = []

    # Calculate the number of TPU hosts within our Ray cluster...
    num_hosts = int(ray.available_resources()["TPU"]) // 4
    # print(ray.available_resources())
    print(f"Num hosts detected: {num_hosts}")

    # Run benchmark_func in multiple threads
    with ThreadPoolExecutor(max_workers=num_hosts) as executor:
        # Create a mapping of futures to their corresponding parameters
        future_to_param = {
            executor.submit(benchmark_func, **benchmark_param, warmup_tries=warmup_tries): benchmark_param
            for benchmark_param in preprocessed_benchmark_params
        }

        # Process each future as it completes
        for future in future_to_param:
            benchmark_param = future_to_param[
                future
            ]  # Retrieve the corresponding benchmark_param
            benchmark_results = future.result()  # Get the result from the future

            # Filter benchmark_results to include only keys present in calculate_metrics_func
            calculate_metrics_params = inspect.signature(
                calculate_metrics_func
            ).parameters
            filtered_benchmark_results = {
                key: value
                for key, value in benchmark_results.items()
                if key in calculate_metrics_params
            }

            # Call calculate_metrics_func with the filtered results and benchmark_param
            metadata, metrics = calculate_metrics_func(
                **benchmark_param, **filtered_benchmark_results
            )
            calculate_metrics_results.append({"metadata": metadata, "metrics": metrics})

    if csv_path:
        write_to_csv(f"{csv_path}/{test_name}.csv", calculate_metrics_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run microbenchmarks and collect metrics."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--multithreaded",
        type=bool,
        default=False,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config, args.multithreaded)
