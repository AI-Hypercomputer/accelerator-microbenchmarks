"""Benchmarks HBM(High Bandwidth Memory) bandwidth."""

from typing import Any, Dict, Tuple

from benchmark_utils import simple_timeit, MetricsStatistics
import jax
import jax.numpy as jnp


def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_keys = {"time_ms_list"}
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_keys
    }
    metadata["dtype"] = metadata["dtype"].dtype.itemsize
    return metadata


def single_chip_hbm_copy(
    num_elements: int,
    dtype: jnp.dtype,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks HBM with copy(read and write) on a single device."""

    def f(a):
        return a.copy()

    a = jax.random.normal(jax.random.key(0), (num_elements,)).astype(dtype)

    jitted_f = jax.jit(f)
    # Run once
    output = jitted_f(a)
    jax.block_until_ready(output)

    # Run the benchmark
    time_ms_list = simple_timeit(
        jitted_f,
        a,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="single_chip_hbm_copy",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def single_chip_hbm_copy_calculate_metrics(
    num_elements: int, dtype: jnp.dtype, time_ms_list: list
) -> Dict[str, Any]:
    """Calculates the metrics for the single chip hbm copy benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    # Calculate throughput.
    tensor_size_bytes = num_elements * dtype.dtype.itemsize

    tensor_size_gbytes = (tensor_size_bytes * 2) / 10**9
    time_statistics = MetricsStatistics(
        metrics_list=time_ms_list, metrics_name="time_ms"
    )
    time_s_list = [time_ms / 10**3 for time_ms in time_ms_list]
    bw_gbyte_sec_list = [tensor_size_gbytes / time_s for time_s in time_s_list]
    statistics = MetricsStatistics(
        metrics_list=bw_gbyte_sec_list, metrics_name="bw_gbyte_sec"
    )
    print(
        f"Tensor size: {tensor_size_bytes / 1024**2} MB, time taken (median):"
        f" {time_statistics.statistics['p50']:.4f} ms, bandwidth (median): {statistics.statistics['p50']:.3f} GB/s"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "tensor_size_gbytes": tensor_size_gbytes,
        }
    )
    metrics.update(time_statistics.serialize_statistics())
    metrics.update(statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics
