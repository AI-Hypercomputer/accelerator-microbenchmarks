"""Benchmarks gemm + all_reduce for DP gradient sync simulation."""

import os
from typing import Any, Dict

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
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
        "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_enable_async_all_gather=true "
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
        # "--xla_tpu_enable_all_reduce_offload_tracing=true "
        "--xla_tpu_enable_all_reduce_scatter_fusion=false "
        "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true "
        "--xla_tpu_pad_operations_input_tiles=true "
        "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=0 "
        "--xla_tpu_use_tc_device_shape_on_sc=true "
    )

    print("Calling jax.distributed.initialize(initialization_timeout=300)...", flush=True)
    jax.distributed.initialize(initialization_timeout=300)
    print("jax.distributed.initialize() completed.", flush=True)
    _INITIALIZED = True


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
    setup_tpu_env()
    dtype_str = dtype.dtype.name
    print(f"Running gemm_all_reduce benchmark with m={m}, k={k}, n={n}, dtype={dtype_str}, runs={num_runs}", flush=True)

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
    
    print("Creating mesh...", flush=True)
    mesh = create_mesh(SHARDING_STRATEGY)
    print("Mesh created.", flush=True)
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

    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{m}x{n}x{k}",
        tries=num_runs,
        task=f"gemm_all_reduce_{dtype_str}",
        trace_dir=trace_dir,
        multiple_ops=True,
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
    
    metadata["type"] = "gemm_all_reduce"
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
    A: [M, K]
    B: [K, N]
    C = A @ B: [M, N]
    """
    setup_tpu_env()
    dtype_str = dtype.dtype.name
    print(f"Running gemm_only benchmark with m={m}, k={k}, n={n}, dtype={dtype_str}, runs={num_runs}", flush=True)

    def f(x, y):
        with jax.named_scope(MARKER):
            # Matmul
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            c = acc.astype(dtype)
            return c

    print("Creating mesh...", flush=True)
    mesh = create_mesh(SHARDING_STRATEGY)
    print("Mesh created.", flush=True)
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

    time_ms_list = multiple_iteration_timeit_from_trace(
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
    
    metadata["type"] = "gemm_only"
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
    setup_tpu_env()
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

    time_ms_list = multiple_iteration_timeit_from_trace(
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
    metadata["type"] = "all_reduce_only"
    
    return metadata, metrics


def gemm_reducescatter_allgather(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jnp.bfloat16,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the ReduceScatter(Matmul(A, B)) + AllGather.
    
    A: [M, K]
    B: [K, N]
    C = A @ B: [M, N]
    Scattered = ReduceScatter(C)
    Output = AllGather(Scattered) -> [M, N]
    """
    setup_tpu_env()
    dtype_str = dtype.dtype.name
    print(f"Running gemm_reducescatter_allgather benchmark with m={m}, k={k}, n={n}, dtype={dtype_str}, runs={num_runs}")

    def f(x, y):
        with jax.named_scope(MARKER):
            # Matmul
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            c = acc.astype(dtype)
            
            # ReduceScatter (psum_scatter)
            # The dimension size being scattered must equal the number of participants (devices).
            # So we reshape [M, N] -> [num_devices, M // num_devices, N]
            num_devices = jax.lax.psum(1, axis_name="device")
            m_size = c.shape[0]
            c_reshaped = c.reshape(num_devices, m_size // num_devices, c.shape[1])
            
            # Scatter along dimension 0 (which is now num_devices)
            # Output per device: [M // num_devices, N]
            scattered_c = jax.lax.psum_scatter(
                c_reshaped, 
                axis_name="device", 
                scatter_dimension=0, 
                tiled=False
            )
            
            # AllGather
            # Gather back to [num_devices, M // num_devices, N]
            gathered_c = jax.lax.all_gather(
                scattered_c,
                axis_name="device",
                axis=0,
                tiled=False
            )
            
            # Flatten back to [M, N]
            out = gathered_c.reshape(m_size, c.shape[1])
            return out

    mesh = create_mesh(SHARDING_STRATEGY)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    # Note: `out_sharding` for NO_SHARDING is P(None, None).
    # The output of `f` (post-all_gather) is [M, N] replicated.

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

    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{m}x{n}x{k}",
        tries=num_runs,
        task=f"gemm_reducescatter_allgather_{dtype_str}",
        trace_dir=trace_dir,
        multiple_ops=True,
    )
    return {
        "time_ms_list": time_ms_list,
    }


def gemm_reducescatter_allgather_calculate_metrics(
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
    
    # Calculate Bandwidth for Collective (ReduceScatter + AllGather)
    # Effectively moves Matrix C: M x N twice (once for scattered, once for gathered, theoretically equivalent to AllReduce data volume wise, but split)
    # Actually, AllReduce volume is 2 * (N-1)/N * Size.
    # ReduceScatter is (N-1)/N * Size.
    # AllGather is (N-1)/N * Size.
    # So total is same as AllReduce.
    
    matrix_c_size_bytes = m * n * dtype.dtype.itemsize

    metadata, metrics = unified_flops_metrics(
        m, n, k, time_ms_list, total_flops_per_device, total_flops_all_devices, peak_flops, dtype=dtype_str,
        total_bytes=matrix_c_size_bytes,
        bandwidth_metric_name="collective_algo_bw_gbs"
    )
    
    metadata["type"] = "gemm_reducescatter_allgather"
    return metadata, metrics


def gemm_sharded_all_gather(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jnp.bfloat16,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the AllGather(Matmul(A_sharded, B)).
    
    A: [M, K] -> Sharded along M axis over devices
    B: [K, N] -> Replicated
    C_partial = A_shard @ B: [M/P, N]
    Output = AllGather(C_partial) -> [M, N] (Replicated)
    """
    setup_tpu_env()
    dtype_str = dtype.dtype.name
    print(f"Running gemm_sharded_all_gather benchmark with m={m}, k={k}, n={n}, dtype={dtype_str}, runs={num_runs}")

    def f(x, y):
        with jax.named_scope(MARKER):
            # Matmul
            # x is [M/P, K] (local chunk)
            # y is [K, N] (replicated)
            # acc is [M/P, N]
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            c_partial = acc.astype(dtype)
            
            # AllGather
            # Gather back to [M, N]
            # Since we sharded M, gathering along device axis reconstructs M.
            out = jax.lax.all_gather(
                c_partial,
                axis_name="device",
                axis=0,
                tiled=False
            )
            return out

    mesh = create_mesh(SHARDING_STRATEGY)
    
    # Define sharding specs for inputs
    # A is sharded on M dimension (axis 0) across "device"
    # B is replicated
    # We ignore standard utils here because we want forced sharding logic
    lhs_spec = P("device", None)
    rhs_spec = P(None, None)
    out_spec = P(None, None) # Output is replicated

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(
                lhs_spec,
                rhs_spec,
            ),
            out_specs=out_spec,
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
        # LHS must be sharded
        lhs_sharding = jax.sharding.NamedSharding(mesh, lhs_spec)
        rhs_sharding = jax.sharding.NamedSharding(mesh, rhs_spec)
        
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)

        return (lhs_device, rhs_device)

    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{m}x{n}x{k}",
        tries=num_runs,
        task=f"gemm_sharded_all_gather_{dtype_str}",
        trace_dir=trace_dir,
        multiple_ops=True,
    )
    return {
        "time_ms_list": time_ms_list,
    }


def gemm_sharded_all_gather_calculate_metrics(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
) -> Dict[str, Any]:
    # Calculate FLOPs (Matmul) per device
    total_flops = 2 * m * k * n / jax.device_count()
    
    # Even though compute is shared, we report total "effective" FLOPs 
    # to see system throughput scaling. 
    # handle_based_on_sharding with NO_SHARDING usually returns total = per_device * num_devices for non-sharded.
    # But here we explicitly sharded. 
    # Let's trust handle_based_on_sharding(NO_SHARDING) logic:
    # it treats total_flops as the global mathematical flops.
    # total_flops_per_device will be total / 1 (if no sharding implies replication? Need to check impl).
    # Wait, `handle_based_on_sharding` usually assumes replication if NO_SHARDING is passed.
    # But physically, per device we did 1/P of the work.
    # The metric utility calculates TFLOP/s based on `total_flops`.
    # if we want to report TFLOP/s aligned with hardware capability, we should use total_flops.
    
    total_flops_per_device, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    
    dtype_str = dtype.dtype.name
    peak_flops_multiplier = get_peak_flops_multiplier(dtype_str)
    peak_flops = PEAK_FLOPS_PER_DEVICE * peak_flops_multiplier
    
    # Calculate Bandwidth for AllGather
    # Moves M x N matrix.
    matrix_c_size_bytes = m * n * dtype.dtype.itemsize

    metadata, metrics = unified_flops_metrics(
        m, n, k, time_ms_list, total_flops_per_device, total_flops_all_devices, peak_flops, dtype=dtype_str,
        total_bytes=matrix_c_size_bytes,
        bandwidth_metric_name="all_gather_algo_bw_gbs"
    )
    
    metadata["type"] = "gemm_sharded_all_gather"
    return metadata, metrics


def gemm_k_sharded_all_reduce(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jnp.bfloat16,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the AllReduce(Matmul(A_sharded_k, B_sharded_k)).
    
    A: [M, K] -> Sharded on K axis across devices
    B: [K, N] -> Sharded on K axis across devices
    C_partial = A_partial @ B_partial: [M, N] (Partial Sum)
    Output = AllReduce(C_partial) -> [M, N] (Full Result)
    """
    setup_tpu_env()
    dtype_str = dtype.dtype.name
    print(f"Running gemm_k_sharded_all_reduce benchmark with m={m}, k={k}, n={n}, dtype={dtype_str}, runs={num_runs}")

    def f(x, y):
        with jax.named_scope(MARKER):
            # Matmul
            # x is [M, K/P] (local chunk)
            # y is [K/P, N] (local chunk)
            # acc is [M, N] (Partial Sum)
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            c_partial = acc.astype(dtype)
            
            # AllReduce
            # Sum partial results from all devices to get full result
            out = jax.lax.psum(c_partial, axis_name="device")
            return out

    mesh = create_mesh(SHARDING_STRATEGY)
    
    # Define sharding specs for inputs
    # A is sharded on K dimension (axis 1) across "device"
    # B is sharded on K dimension (axis 0) across "device"
    lhs_spec = P(None, "device")
    rhs_spec = P("device", None)
    out_spec = P(None, None) # Output is replicated

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(
                lhs_spec,
                rhs_spec,
            ),
            out_specs=out_spec,
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
        # Inputs must be sharded
        lhs_sharding = jax.sharding.NamedSharding(mesh, lhs_spec)
        rhs_sharding = jax.sharding.NamedSharding(mesh, rhs_spec)
        
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)

        return (lhs_device, rhs_device)

    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{m}x{n}x{k}",
        tries=num_runs,
        task=f"gemm_k_sharded_all_reduce_{dtype_str}",
        trace_dir=trace_dir,
        multiple_ops=True,
    )
    return {
        "time_ms_list": time_ms_list,
    }


def gemm_k_sharded_all_reduce_calculate_metrics(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
) -> Dict[str, Any]:
    # Calculate FLOPs (Matmul) per device
    total_flops = 2 * m * k * n / jax.device_count()
    
    # K-sharded means each device does 1/P of the work for the same M, N.
    # Total FLOPs across system is still 2*M*K*N.
    
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
    
    metadata["type"] = "gemm_k_sharded_all_reduce"
    return metadata, metrics
