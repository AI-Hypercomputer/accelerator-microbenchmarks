"""Benchmarks gemm + all_reduce for DP gradient sync simulation."""

import os
from typing import Any, Dict, Optional, Callable

# pylint: disable=g-importing-member
from benchmark_utils import (
    iteration_timeit,
    multiple_iteration_timeit_from_trace,
    ShardingStrategy,
    get_lhs_named_shading,
    get_rhs_named_shading,
    get_out_sharding,
    create_mesh,
    handle_based_on_sharding,
    unified_flops_metrics,
    MetricsStatistics,
    get_metrics_helper,
    str_to_dtype,
    get_peak_flops_multiplier,
    unified_bytes_metrics,
)
from common import MARKER
import jax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp


# pylint: disable=g-importing-member


# Matmul shapes: A(M,K) x B(K,N) = C(M,N)
# Then AllReduce(C)
SHARDING_STRATEGY = ShardingStrategy.NO_SHARDING
SEED = 0
PEAK_FLOPS_PER_DEVICE = 2307  # TFLOP/s for single core(device) of FP8

_INITIALIZED = False

def setup_tpu_env():
    global _INITIALIZED
    if _INITIALIZED:
        return
    
    print("Setting LIBTPU_INIT_ARGS...", flush=True)
    os.environ["LIBTPU_INIT_ARGS"] = (
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_enable_async_all_reduce=true "
        "--xla_enable_async_collective_permute=true "
        "--xla_tpu_enable_all_experimental_scheduler_features=true "
        "--xla_tpu_should_accumulate_into_mrb=true "
        "--xla_tpu_scoped_vmem_limit_kib=65536 "
        "--xla_tpu_vmem_scavenging_mode=NONE "
        "--xla_tpu_dvfs_p_state=7 "

        "--xla_tpu_impure_enable_packed_bf16_math_ops=true "
        "--xla_tpu_enable_pincer_short_fusion_emitter=true "
        "--xla_tpu_enable_sparse_core_hierarchical_all_reduce=true "
        "--xla_tpu_use_single_sparse_core_for_all_reduce_offload=true " # Test effect on SC

        "--xla_jf_debug_level=1 "
        "--xla_sc_disable_megacore_partitioning=true "
        "--xla_tpu_disable_sparse_core_collective_offload_remover=true "
        "--xla_tpu_enable_all_reduce_scatter_fusion=false "
        "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true "
        "--xla_tpu_pad_operations_input_tiles=true "
        "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=0 "
        "--xla_tpu_use_tc_device_shape_on_sc=true "
    )

    print("Step 1: Calling jax.distributed.initialize(initialization_timeout=300)...", flush=True)
    jax.distributed.initialize(initialization_timeout=300)
    print("Step 1: jax.distributed.initialize() completed.", flush=True)
    _INITIALIZED = True


def _run_gemm_base(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    num_runs: int,
    trace_dir: str,
    sharding_strategy: ShardingStrategy,
    task_name_suffix: str,
) -> Dict[str, Any]:
    """Shared base function for running GEMM benchmarks."""
    setup_tpu_env()
    dtype_str = dtype.dtype.name
    task_name = f"{task_name_suffix}_{dtype_str}"
    print(f"Running {task_name} benchmark with m={m}, k={k}, n={n}, dtype={dtype_str}, runs={num_runs}", flush=True)

    def f(x, y):
        with jax.named_scope(MARKER):
            # Matmul
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            c = acc.astype(dtype)
            
            # AllReduce (psum)
            out = jax.lax.psum(c, axis_name="device")
            return out

    print("Step 2: Creating Mesh and Shardings...", flush=True)
    mesh = create_mesh(sharding_strategy)
    lhs_sharding = get_lhs_named_shading(mesh, sharding_strategy)
    rhs_sharding = get_rhs_named_shading(mesh, sharding_strategy)
    out_sharding = get_out_sharding(sharding_strategy)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(
                lhs_sharding.spec,
                rhs_sharding.spec,
            ),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    lhs_shape = (m, k)
    rhs_shape = (k, n)
    lhs_dtype = dtype
    rhs_dtype = dtype
    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key
        key, key_lhs, key_rhs = jax.random.split(key, 3)

        # Create random data on host
        lhs_host = jax.random.normal(key_lhs, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(key_rhs, rhs_shape).astype(rhs_dtype)

        # Put on device (HBM)
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)

        return (lhs_device, rhs_device)

    print("Step 3: Starting Execution Loop (includes JIT)...", flush=True)
    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{m}x{n}x{k}",
        tries=num_runs,
        task=task_name,
        trace_dir=trace_dir,
        multi_op=True,
    )
    print("Step 4: Execution Loop Completed.", flush=True)
    
    return {
        "time_ms_list": time_ms_list,
    }


def gemm_all_reduce(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jnp.bfloat16,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the Matmul(A, B) + AllReduce(C)."""
    return _run_gemm_base(
        m, k, n, dtype, num_runs, trace_dir,
        sharding_strategy=ShardingStrategy.NO_SHARDING,
        task_name_suffix="gemm_all_reduce"
    )





def _calculate_metrics_base(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
    sharding_strategy: ShardingStrategy,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Shared metrics calculation for GEMM benchmarks."""
    total_flops = 2 * m * k * n
    total_flops_per_device, total_flops_all_devices = handle_based_on_sharding(
        total_flops, sharding_strategy
    )

    dtype_str = dtype.dtype.name
    peak_flops_multiplier = get_peak_flops_multiplier(dtype_str)
    peak_flops = PEAK_FLOPS_PER_DEVICE * peak_flops_multiplier

    return unified_flops_metrics(
        m, n, k, time_ms_list, total_flops_per_device, total_flops_all_devices, peak_flops, dtype=dtype_str,
    )


def gemm_all_reduce_calculate_metrics(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
) -> Dict[str, Any]:
    # Calculate Bandwidth for Collective (AllReduce)
    # Effective bandwidth for AllReduce is 2 * (N-1)/N * Size.
    # We use Size * 2 as a proxy for total bytes moved (assuming large N).

    metadata, metrics = _calculate_metrics_base(
        m, k, n, dtype, time_ms_list, ShardingStrategy.NO_SHARDING
    )
    
    metadata["type"] = "gemm_all_reduce"
    return metadata, metrics



