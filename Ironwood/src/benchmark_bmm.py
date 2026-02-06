"""
Benchmarks bmm in various flavors.
Considered ops:
1. bmm
"""

import os
from typing import Any, Dict

# pylint: disable=g-importing-member
from benchmark_utils import (
    iteration_timeit,
    multiple_iteration_timeit_from_trace,
    ShardingStrategy,
    get_lhs_named_shading,
    get_rhs_named_shading,
    get_output_named_shading,
    get_out_sharding,
    create_mesh,
    handle_based_on_sharding,
    unified_flops_metrics,
    str_to_dtype,
    get_peak_flops_multiplier
)
from common import MARKER
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


# pylint: disable=g-importing-member

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
    "--xla_tpu_vmem_scavenging_mode=NONE "
    "--xla_tpu_dvfs_p_state=7"
)

TRACE_BASE_DIR = None
METRICS_JSONL_DIR = None
SHARDING_STRATEGY = ShardingStrategy.NO_SHARDING
SEED = 0
PEAK_FLOPS_PER_DEVICE = 2307  # TFLOP/s for single core(device) of FP8

def single_device_bmm(
    b: int,
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jax.numpy.float8_e4m3fn,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the OUT<B, M, N>:BF16 = IN0<B, M, K>:FP8 x IN1<B, K, N>:FP8. Accumulation is FP32."""

    def f(x, y):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum(
                "bij,bjk->bik", x, y, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    jit_sharded_f = jax.jit(f)

    lhs_shape = (b, m, k)
    rhs_shape = (b, k, n)

    lhs_dtype = dtype
    rhs_dtype = dtype

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, key_lhs, key_rhs = jax.random.split(key, 3)

        # Create random data on host
        lhs_host = jax.random.normal(key_lhs, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(key_rhs, rhs_shape).astype(rhs_dtype)

        # Put on device (HBM)

        return (lhs_host, rhs_host)
    
    # Run the benchmark

    # num_runs = 1

    dtype_str = dtype.dtype.name
    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{b}x{m}x{n}x{k}",
        tries=num_runs,
        task="single_device_bmm",
        trace_dir=trace_dir,
    )

    return {"time_ms_list": time_ms_list}


def single_device_bmm_calculate_metrics(
    b: int,
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = 2 * b * m * k * n  # Total floating-point operations
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
        dtype=dtype.dtype.name,
        b=b,
    )


def multi_host_bmm(
    b: int,
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jax.numpy.float8_e4m3fn,
    num_runs: int = 1,
    trace_dir: str = None,
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARDING,
) -> Dict[str, Any]:
    """Benchmarks multi-host bmm."""
    mesh = create_mesh(sharding_strategy)

    lhs_sharding = get_lhs_named_shading(mesh, sharding_strategy)
    rhs_sharding = get_rhs_named_shading(mesh, sharding_strategy)
    output_sharding = get_output_named_shading(mesh, sharding_strategy)

    def f(x, y):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum(
                "bij,bjk->bik", x, y, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    jit_sharded_f = jax.jit(
        f,
        in_shardings=(lhs_sharding, rhs_sharding),
        out_shardings=output_sharding,
    )

    lhs_shape = (b, m, k)
    rhs_shape = (b, k, n)

    lhs_dtype = dtype
    rhs_dtype = dtype

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, key_lhs, key_rhs = jax.random.split(key, 3)

        # Create random data on host
        lhs_host = jax.random.normal(key_lhs, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(key_rhs, rhs_shape).astype(rhs_dtype)

        # Put on device (HBM) with sharding
        lhs = jax.device_put(lhs_host, lhs_sharding)
        rhs = jax.device_put(rhs_host, rhs_sharding)

        return (lhs, rhs)

    dtype_str = dtype.dtype.name
    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{b}x{m}x{n}x{k}",
        tries=num_runs,
        task="multi_host_bmm",
        trace_dir=trace_dir,
    )

    return {"time_ms_list": time_ms_list}


def multi_host_bmm_calculate_metrics(
    b: int,
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARDING
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = 2 * b * m * k * n  # Total floating-point operations
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, sharding_strategy
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
        dtype=dtype.dtype.name,
        b=b,
    )
