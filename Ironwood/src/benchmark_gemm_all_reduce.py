"""Benchmarks gemm + all_reduce for DP gradient sync simulation."""

import os
from typing import Any, Dict

# pylint: disable=g-importing-member
from benchmark_utils import (
    iteration_timeit,
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
import jax.numpy as jnp


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
    "--xla_tpu_dvfs_p_state=7 "
  
    "--xla_jf_debug_level=3 "
    "--xla_sc_disable_megacore_partitioning=true "
    "--xla_tpu_disable_sparse_core_collective_offload_remover=true "
    "--xla_tpu_enable_all_reduce_offload_tracing=true "
    "--xla_tpu_enable_all_reduce_scatter_fusion=false "
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true "
    "--xla_tpu_pad_operations_input_tiles=true "
    "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=0 "
    "--xla_tpu_use_tc_device_shape_on_sc=true "
)

# Matmul shapes: A(M,K) x B(K,N) = C(M,N)
# Then AllReduce(C)
SHARDING_STRATEGY = ShardingStrategy.NO_SHARDING
SEED = 0
PEAK_FLOPS_PER_DEVICE = 2307  # TFLOP/s for single core(device) of FP8


def gemm_all_reduce(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jnp.bfloat16,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the AllReduce(matmul(A, B)).
    
    A: [M, K]
    B: [K, N]
    C = A @ B: [M, N]
    Output = AllReduce(C)
    """

    dtype_str = dtype.dtype.name
    print(f"Running gemm_all_reduce benchmark with m={m}, k={k}, n={n}, dtype={dtype_str}, runs={num_runs}")

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

    # This benchmark simulates the Data Parallel (DP) Backward Pass:
    # 1. Local Gradient Computation: Each device computes `Grads = Activations.T @ GradOutput`.
    #    - Here: `acc = x @ y` corresponds to `(M, K) @ (K, N) -> (M, N)`.
    #    - `K` represents the LOCAL Batch Size (contracting dimension).
    #    - `M` and `N` represent the Weight dimensions (e.g. Hidden Size).
    #    - The input `x` and `y` are effectively local to the device (replicated or split, the compute is local).
    # 2. Gradient Synchronization: `AllReduce(Grads)`.
    #    - `out = psum(c, axis_name="device")` sums the partial gradients across all devices.
    
    # We use `ShardingStrategy.NO_SHARDING` for the mesh.
    # In `benchmark_utils`, this creates a mesh with a single "device" axis containing all devices.
    # Inside `shard_map` (with `check_rep=False` and fully replicated in_specs P(None, None)),
    # each device receives the input arrays and executes the function `f`.
    # `psum("device")` then performs the AllReduce across all devices in the mesh.
    
    mesh = create_mesh(SHARDING_STRATEGY)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    # Note: `out_sharding` for NO_SHARDING is P(None, None).
    # The output of `f` (post-psum) is mathematically consistent across devices (replicated).

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

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{m}x{n}x{k}",
        tries=num_runs,
        task=f"gemm_all_reduce_{dtype_str}",
        trace_dir=trace_dir,
    )
    return {
        "time_ms_list": time_ms_list,
    }


def gemm_all_reduce_calculate_metrics(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
) -> Dict[str, Any]:
    # Calculate FLOPs (Matmul)
    total_flops = 2 * m * k * n
    
    total_flops_per_device, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    
    dtype_str = dtype.dtype.name
    peak_flops_multiplier = get_peak_flops_multiplier(dtype_str)
    peak_flops = PEAK_FLOPS_PER_DEVICE * peak_flops_multiplier
    
    # Calculate Bandwidth for AllReduce
    # AllReduce moves Matrix C: M x N
    matrix_c_size_bytes = m * n * dtype.dtype.itemsize

    metadata, metrics = unified_flops_metrics(
        m, n, k, time_ms_list, total_flops_per_device, total_flops_all_devices, peak_flops, dtype=dtype_str,
        total_bytes=matrix_c_size_bytes,
        bandwidth_metric_name="all_reduce_algo_bw_gbs"
    )
    
    return metadata, metrics


def gemm_only(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jnp.bfloat16,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks only the Matmul part of gemm_all_reduce.
    
    A: [M, K]
    B: [K, N]
    C = A @ B: [M, N]
    """

    dtype_str = dtype.dtype.name
    print(f"Running gemm_only benchmark with m={m}, k={k}, n={n}, dtype={dtype_str}, runs={num_runs}")

    def f(x, y):
        with jax.named_scope(MARKER):
            # Matmul
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            c = acc.astype(dtype)
            return c

    mesh = create_mesh(SHARDING_STRATEGY)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

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

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{m}x{n}x{k}",
        tries=num_runs,
        task=f"gemm_only_{dtype_str}",
        trace_dir=trace_dir,
    )
    return {
        "time_ms_list": time_ms_list,
    }


def gemm_only_calculate_metrics(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
) -> Dict[str, Any]:
    # Calculate FLOPs (Matmul)
    total_flops = 2 * m * k * n
    
    total_flops_per_device, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    
    dtype_str = dtype.dtype.name
    peak_flops_multiplier = get_peak_flops_multiplier(dtype_str)
    peak_flops = PEAK_FLOPS_PER_DEVICE * peak_flops_multiplier
    
    metadata, metrics = unified_flops_metrics(
        m, n, k, time_ms_list, total_flops_per_device, total_flops_all_devices, peak_flops, dtype=dtype_str,
    )
    
    return metadata, metrics


def all_reduce_only(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jnp.bfloat16,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks only the AllReduce part of gemm_all_reduce independently.
    
    Input: C [M, N]
    Output = AllReduce(C)
    """

    dtype_str = dtype.dtype.name
    print(f"Running all_reduce_only benchmark with m={m}, k={k}, n={n}, dtype={dtype_str}, runs={num_runs}")

    def f(c):
        with jax.named_scope(MARKER):
            # AllReduce (psum)
            out = jax.lax.psum(c, axis_name="device")
            return out

    mesh = create_mesh(SHARDING_STRATEGY)
    # Input to AllReduce is the output of Matmul, which is C [M, N]
    # In gemm_all_reduce, C is effectively replicated or sharded depending on strategy, 
    # but here SHARDING_STRATEGY is NO_SHARDING, so it's replicated?
    # Actually, let's double check gemm_all_reduce out_sharding.
    # out_sharding = get_out_sharding(SHARDING_STRATEGY) -> P(None, None) for NO_SHARDING
    
    # So the input to THIS function should match the output of the GEMM part in gemm_all_reduce
    # In gemm_all_reduce:
    # f(x,y): ... return out
    # out_sharding is P(None, None).
    
    # But wait, inside gemm_all_reduce's `f`, `c = acc.astype(dtype)`.
    # `c` is local to the device in shard_map terms if check_rep=False and in_specs are P(None, None).
    # Yes, `gemm_all_reduce` uses `in_specs=(lhs_sharding.spec, rhs_sharding.spec)`.
    # For NO_SHARDING, lhs_sharding is P(None, None), rhs is P(None, None).
    # So `c` is [M, N] per device.
    
    # So here, we want input `c` to be P(None, None) per device.
    
    input_sharding = get_out_sharding(SHARDING_STRATEGY) # Reusing this as it matched C's distribution
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(input_sharding,),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    # Shape of C
    c_shape = (m, n)
    c_dtype = dtype

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key
        key, key_c = jax.random.split(key, 2)

        # Create random data on host
        c_host = jax.random.normal(key_c, c_shape).astype(c_dtype)

        # Put on device (HBM)
        # We need to wrap input_sharding (which is a PartitionSpec) in NamedSharding
        # because device_put needs to know the mesh.
        named_input_sharding = jax.sharding.NamedSharding(mesh, input_sharding)
        c_device = jax.device_put(c_host, named_input_sharding)

        return (c_device,)

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{m}x{n}x{k}",
        tries=num_runs,
        task=f"all_reduce_only_{dtype_str}",
        trace_dir=trace_dir,
    )
    return {
        "time_ms_list": time_ms_list,
    }


def all_reduce_only_calculate_metrics(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
) -> Dict[str, Any]:
    
    # Calculate Bandwidth for AllReduce
    # AllReduce moves Matrix C: M x N
    matrix_c_size_bytes = m * n * dtype.dtype.itemsize
    
    # Use unified_bytes_metrics for bandwidth-bound operations
    # We estimate total_bytes_all_devices assuming full replication or reduction over all devices
    num_devices = jax.device_count()
    total_bytes_all_devices = matrix_c_size_bytes * num_devices

    metadata, metrics = unified_bytes_metrics(
        m, n, time_ms_list,
        total_bytes=matrix_c_size_bytes,
        total_bytes_all_devices=total_bytes_all_devices,
        dtype=dtype.dtype.name
    )
    
    return metadata, metrics
