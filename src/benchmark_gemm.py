"""Benchmarks matmul in various flavors.

1. 
"""

import os
from typing import Any, Dict, Tuple


# pylint: disable=g-importing-member
from benchmark_utils import simple_timeit, MetricsStatistics
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

# pylint: disable=g-importing-member
# Set the environment variable for TPU initialization arguments to optimize
# collective matmul. Setting the flags to false will disable the optimization.
os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true "
    "--xla_enable_async_collective_permute=true "
    "--xla_tpu_enable_all_experimental_scheduler_features=true "
    "--xla_tpu_accumulate_into_mrb=true "
    "--xla_tpu_scoped_vmem_limit_kib=65536 "
    "--xla_tpu_dvfs_p_state=7 "
    # "--xla_tpu_should_accumulate_into_mrb=true" # Unknown XLA Flag
)
TRACE_BASE_DIR = None
METRICS_JSONL_DIR = None
# Matmul shapes: A(M,K) x B(K,N) = C(M,N)
M_STEP_SIZE = 1024
M_START_SIZE = 1024
M_MAX_SIZE = 50000
# The number of layers in the multilayer collective matmul.
# Matmul shapes: A(M,K) x H1(K,K)... x B(K,N) = C(M,N)
LAYERS = 2

def create_mesh() -> Mesh:
    """Creates a mesh."""
    mesh = Mesh(np.array(jax.devices()), axis_names="i")
    return mesh

def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_param_keys = {"time_ms_list"}
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_param_keys
    }
    return metadata

def gemm_simple(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None, warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the OUT<M, N>:BF16 = IN0<M, K>:FP8 x IN1<N, K>:FP8."""

    def f(x, y):
        acc = jax.numpy.einsum("ij,jk->ik", x, y, preferred_element_type=jnp.float32)
        return acc.astype(jnp.bfloat16)

    mesh = create_mesh()
    lhs = jnp.arange(np.prod((m, k))).reshape((m, k)).astype(jnp.float8_e4m3fn)
    rhs = jnp.arange(np.prod((k, n))).reshape((k, n)).astype(jnp.float8_e4m3fn)
    # lhs(m,k): sharded across devices. rhs(k,n): replicated on devices.
    # output(m,n): replicated on devices.
    lhs = jax.device_put(lhs, NamedSharding(mesh, P("i", None)))
    rhs = jax.device_put(rhs, NamedSharding(mesh, P(None, None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(P(), P()),
            out_specs=P(),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(lhs, rhs)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"{lhs.shape=} x {rhs.shape=} = {output.shape=}, {lhs.dtype=}, {rhs.dtype=}, {output.dtype=}")
    # Run the benchmark
    time_ms_list = simple_timeit(
        jit_sharded_f,
        lhs,
        rhs,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="gemm_simple",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def gemm_metrics_all(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    """Calculates the metrics for the naive matmul benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    tflops_per_sec_list = [
        total_flops / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    average_time_ms_statistics = MetricsStatistics(
        metrics_list=time_ms_list, metrics_name="step_time_ms"
    )
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec"
    )
    total_gigabytes_transferred = 2 * (m * k + k * n + m * n) / 10**9
    data_transfer_gbyte_sec_list = [
        total_gigabytes_transferred / average_time_s
        for average_time_s in average_time_s_list
    ]
    data_transfer_gbyte_sec_statistics = MetricsStatistics(
        metrics_list=data_transfer_gbyte_sec_list,
        metrics_name="data_transfer_gbyte_sec",
    )
    print(
        f"Total floating-point ops: {total_flops}, Step Time (median): {average_time_ms_statistics.statistics['p50']:.2f}, Performance (median):"
        f" {tflops_per_sec_statistics.statistics['p50']:.2f} TFLOPs / second, Total GBs transferred (median):"
        f" {total_gigabytes_transferred:.2f} GB, GBs per second:"
        f" {data_transfer_gbyte_sec_statistics.statistics['p50']:.2f} GB/s"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "Step Time (median, ms)": average_time_ms_statistics.statistics['p50'],
            "Throughput (median, TFLOPS)": tflops_per_sec_statistics.statistics['p50'],
            "total_flops": total_flops,
            "total_gigabytes_transferred": total_gigabytes_transferred,
        }
    )
    metrics.update(average_time_ms_statistics.serialize_statistics())
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics.update(data_transfer_gbyte_sec_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics

def gemm_simple_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    return gemm_metrics_all(m, k, n, time_ms_list)

def gemm(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None, warmup_tries: int = 10,
) -> Dict[str, Any]:
    """OUT<M, N>:BF16 = matmul(IN0<M, K>:FP8, IN1<N, K>:FP8) * outer_product(SF0<M, 1>:FP32 * SF1<1, N>:FP32)"""
    def f(x, y, scale_m, scale_n):
        acc = jax.numpy.einsum("ij,jk->ik", x, y, preferred_element_type=jnp.float32)
        scales = scale_m * scale_n
        result_fp32 = acc * scales
        return result_fp32.astype(jnp.bfloat16)

    mesh = create_mesh()
    lhs = jnp.arange(np.prod((m, k))).reshape((m, k)).astype(jnp.float8_e4m3fn)
    rhs = jnp.arange(np.prod((k, n))).reshape((k, n)).astype(jnp.float8_e4m3fn)
    sf0 = jnp.arange(m).reshape((m, 1)).astype(jnp.float32)
    sf1 = jnp.arange(n).reshape((1, n)).astype(jnp.float32)
    # lhs(m,k): sharded across devices. rhs(k,n): replicated on devices.
    # output(m,n): replicated on devices.
    lhs = jax.device_put(lhs, NamedSharding(mesh, P("i", None)))
    rhs = jax.device_put(rhs, NamedSharding(mesh, P(None, None)))
    sf0 = jax.device_put(sf0, NamedSharding(mesh, P("i", None)))
    sf1 = jax.device_put(sf1, NamedSharding(mesh, P(None, None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(P(), P(), P(), P()),
            out_specs=P(),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(lhs, rhs, sf0, sf1)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"{lhs.shape=} x {rhs.shape=} = {output.shape=}, {lhs.dtype=}, {rhs.dtype=}, {sf0.dtype=}, {sf1.dtype=}, {output.dtype=}")
    # Run the benchmark
    time_ms_list = simple_timeit(
        jit_sharded_f,
        lhs,
        rhs,
        sf0,
        sf1,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="gemm",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def gemm_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    return gemm_metrics_all(m, k, n, time_ms_list)


def gemm_accum(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None, warmup_tries: int = 10,
) -> Dict[str, Any]:
    """OUT<M, N>:FP32 += matmul(IN0<M, K>:FP8, IN1<N, K>:FP8) * outer_product(SF0<M, 1>:FP32 * SF1<1, N>:FP32)"""
    def f(out_buffer, x, y, scale_m, scale_n):
        acc = jax.numpy.einsum("ij,jk->ik", x, y, preferred_element_type=jnp.float32)
        scales = scale_m * scale_n
        result_fp32 = acc * scales
        return out_buffer + result_fp32

    mesh = create_mesh()
    lhs = jnp.arange(np.prod((m, k))).reshape((m, k)).astype(jnp.float8_e4m3fn)
    rhs = jnp.arange(np.prod((k, n))).reshape((k, n)).astype(jnp.float8_e4m3fn)
    sf0 = jnp.arange(m).reshape((m, 1)).astype(jnp.float32)
    sf1 = jnp.arange(n).reshape((1, n)).astype(jnp.float32)
    out_buffer = jnp.arange(np.prod((m, n))).reshape((m, n)).astype(jnp.float32)
    # lhs(m,k): sharded across devices. rhs(k,n): replicated on devices.
    # output(m,n): replicated on devices.
    lhs = jax.device_put(lhs, NamedSharding(mesh, P("i", None)))
    rhs = jax.device_put(rhs, NamedSharding(mesh, P(None, None)))
    sf0 = jax.device_put(sf0, NamedSharding(mesh, P("i", None)))
    sf1 = jax.device_put(sf1, NamedSharding(mesh, P(None, None)))
    out_buffer = jax.device_put(out_buffer, NamedSharding(mesh, P("i", None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(P(), P(), P(), P(), P()),
            out_specs=P(),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(out_buffer, lhs, rhs, sf0, sf1)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"Inputs: {lhs.shape=}, {rhs.shape=}, {sf0.shape=}, {sf1.shape=}")
    print(f"Buffers: {out_buffer.shape=}, {output.shape=}")
    print(f"Dtypes: {lhs.dtype=}, {out_buffer.dtype=}, {output.dtype=}")
    # Run the benchmark
    time_ms_list = simple_timeit(
        jit_sharded_f,
        out_buffer,
        lhs,
        rhs,
        sf0,
        sf1,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="gemm_accum",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def gemm_accum_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    """Calculates the metrics for the naive matmul benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    # Calculate FLOPs
    total_flops = 2 * m * k * n + m * n  # Total floating-point operations
    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    tflops_per_sec_list = [
        total_flops / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    average_time_ms_statistics = MetricsStatistics(
        metrics_list=time_ms_list, metrics_name="step_time_ms"
    )
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec"
    )
    total_gigabytes_transferred = 2 * (m * k + k * n + m * n) / 10**9
    data_transfer_gbyte_sec_list = [
        total_gigabytes_transferred / average_time_s
        for average_time_s in average_time_s_list
    ]
    data_transfer_gbyte_sec_statistics = MetricsStatistics(
        metrics_list=data_transfer_gbyte_sec_list,
        metrics_name="data_transfer_gbyte_sec",
    )
    print(
        f"Total floating-point ops: {total_flops}, Step Time (median): {average_time_ms_statistics.statistics['p50']:.2f}, Performance (median):"
        f" {tflops_per_sec_statistics.statistics['p50']:.2f} TFLOPs / second, Total GBs transferred (median):"
        f" {total_gigabytes_transferred:.2f} GB, GBs per second:"
        f" {data_transfer_gbyte_sec_statistics.statistics['p50']:.2f} GB/s"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "Step Time (median, ms)": average_time_ms_statistics.statistics['p50'],
            "Throughput (median, TFLOPS)": tflops_per_sec_statistics.statistics['p50'],
            "total_flops": total_flops,
            "total_gigabytes_transferred": total_gigabytes_transferred,
        }
    )
    metrics.update(average_time_ms_statistics.serialize_statistics())
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics.update(data_transfer_gbyte_sec_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics
