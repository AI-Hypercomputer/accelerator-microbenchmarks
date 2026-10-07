"""
Benchmarks gemm in various flavors.
Considered ops:
1. gemm_simple
2. gemm
3. gemm_accum
4. gemm_multiple_run
"""

import os
from typing import Any, Dict

from benchmark_utils import (
    configure_local_node_tpu,
    create_mesh,
    get_lhs_named_shading,
    get_out_sharding,
    get_output_named_shading,
    get_peak_flops_multiplier,
    get_rhs_named_shading,
    handle_based_on_sharding,
    iteration_timeit,
    multiple_iteration_timeit_from_trace,
    ShardingStrategy,
    str_to_dtype,
    unified_flops_metrics,
)
from common import MARKER

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp


_GEMM_LIBTPU_INIT_ARGS = (
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
_existing_libtpu_args = os.environ.get("LIBTPU_INIT_ARGS", "")
os.environ["LIBTPU_INIT_ARGS"] = (
    f"{_existing_libtpu_args} {_GEMM_LIBTPU_INIT_ARGS}".strip()
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
WITH_SHARDING = True

SHARDING_STRATEGY = ShardingStrategy.NO_SHARDING
SEED = 0
PEAK_FLOPS_PER_DEVICE = 2307  # TFLOP/s for single core(device) of FP8


def gemm_multiple_run(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype = jax.numpy.float8_e4m3fn,
    num_runs: int = 1,
    trace_dir: str = None,
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    """Benchmarks the OUT<M, N>:BF16 = IN0<M, K> dtype x IN1<N, K>:dtype."""

    # Accumulation is FP32. Current supported dtype: float8_e4m3fn,
    # bfloat16.

    def f(x, y):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    if run_on_local_node:
        configure_local_node_tpu()

    mesh = create_mesh(SHARDING_STRATEGY, local_mesh=run_on_local_node)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(lhs_sharding.spec, rhs_sharding.spec),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    lhs_shape = (m, k)
    rhs_shape = (k, n)

    lhs_dtype = dtype
    rhs_dtype = dtype

    if run_on_local_node:
        import numpy as np

        step_idx = 0
        mesh_devices = list(mesh.devices.flat)
        seed_spec = jax.sharding.PartitionSpec(*mesh.axis_names)
        seed_sharding = jax.sharding.NamedSharding(mesh, seed_spec)

        def _gen_local(seed_arr):
            k = jax.random.key(seed_arr.reshape(-1)[0])
            k_lhs, k_rhs = jax.random.split(k)
            lhs = jax.random.normal(k_lhs, lhs_shape).astype(lhs_dtype)
            rhs = jax.random.normal(k_rhs, rhs_shape).astype(rhs_dtype)
            return lhs, rhs

        jit_gen_data = jax.jit(
            shard_map(
                _gen_local,
                mesh,
                in_specs=seed_spec,
                out_specs=(lhs_sharding.spec, rhs_sharding.spec),
                check_rep=False,
            )
        )

        def data_generator():
            """Creates new random data on each device locally without cross-chip ICI DMA."""
            nonlocal step_idx
            val = np.array([SEED + step_idx], dtype=np.int32).reshape(
                (1,) * len(mesh.axis_names)
            )
            step_idx += 1
            shards = [jax.device_put(val, d) for d in mesh_devices]
            seed_dev = jax.make_array_from_single_device_arrays(
                mesh.devices.shape, seed_sharding, shards
            )
            return jit_gen_data(seed_dev)
    else:
        key = jax.random.key(SEED)

        def data_generator():
            """Creates new random data on host and puts it on device."""
            nonlocal key  # Use and update the outer 'key'
            key, key_lhs, key_rhs = jax.random.split(key, 3)

            # Create random data on host
            lhs_host = jax.random.normal(key_lhs, lhs_shape).astype(lhs_dtype)
            rhs_host = jax.random.normal(key_rhs, rhs_shape).astype(rhs_dtype)

            # Put on device (HBM)
            lhs_device = jax.device_put(lhs_host, lhs_sharding)
            rhs_device = jax.device_put(rhs_host, rhs_sharding)

            return (lhs_device, rhs_device)

    # Run the benchmark

    print("Running gemm_multiple_run benchmark", num_runs)
    dtype_str = dtype.dtype.name
    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{dtype_str}_{m}x{n}x{k}",
        tries=num_runs,
        task="gemm_multiple_run",
        trace_dir=trace_dir,
    )
    return {
        "time_ms_list": time_ms_list,
    }


def gemm_multiple_run_calculate_metrics(
    m: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    time_ms_list: list[float],
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    device_count = (
        jax.local_device_count() if run_on_local_node else jax.device_count()
    )
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY, device_count=device_count
    )
    peak_flops = (
        PEAK_FLOPS_PER_DEVICE
        if dtype == jax.numpy.float8_e4m3fn
        else PEAK_FLOPS_PER_DEVICE / 2
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        peak_flops,
        dtype=dtype.dtype.name,
    )


def gemm_simple(
    m: int,
    k: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    """Benchmarks the OUT<M, N>:BF16 = IN0<M, K>:FP8 x IN1<N, K>:FP8."""
    # Accumulation is FP32.

    def f(x, y):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    mesh = create_mesh(SHARDING_STRATEGY, local_mesh=run_on_local_node)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(lhs_sharding.spec, rhs_sharding.spec),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    lhs_shape = (m, k)
    rhs_shape = (k, n)

    lhs_dtype = jax.numpy.float8_e4m3fn
    rhs_dtype = jax.numpy.float8_e4m3fn

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, key_lhs, key_rhs = jax.random.split(key, 3)

        # Create random data on host
        lhs_host = jax.random.normal(key_lhs, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(key_rhs, rhs_shape).astype(rhs_dtype)

        # Put on device (HBM)
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)

        return (lhs_device, rhs_device)

    # Run the benchmark
    num_runs = 1
    # Need to fix gemm timing logic to handle num_runs > 1

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="gemm_simple",
        trace_dir=trace_dir,
    )

    return {"time_ms_list": time_ms_list}


def gemm_simple_calculate_metrics(
    m: int,
    k: int,
    n: int,
    time_ms_list: list[float],
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    device_count = (
        jax.local_device_count() if run_on_local_node else jax.device_count()
    )
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY, device_count=device_count
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )


def gemm_simple_with_dtype(
    m: int,
    k: int,
    n: int,
    in_dtype_str: str,
    out_dtype_str: str,
    num_runs: int = 1,
    trace_dir: str = None,
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    """Benchmarks the OUT<M, N>:BF16 = IN0<M, K>:FP8 x IN1<N, K>:FP8."""
    # Accumulation is FP32.

    # Convert string dtypes to jnp dtypes
    lhs_dtype = str_to_dtype(in_dtype_str)
    rhs_dtype = str_to_dtype(in_dtype_str)
    out_dtype = str_to_dtype(out_dtype_str)

    def f(x, y):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            return acc.astype(out_dtype)

    mesh = create_mesh(SHARDING_STRATEGY, local_mesh=run_on_local_node)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(lhs_sharding.spec, rhs_sharding.spec),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    lhs_shape = (m, k)
    rhs_shape = (k, n)

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, key_lhs, key_rhs = jax.random.split(key, 3)

        # Create random data on host
        lhs_host = jax.random.normal(key_lhs, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(key_rhs, rhs_shape).astype(rhs_dtype)

        # Put on device (HBM)
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)

        return (lhs_device, rhs_device)

    num_runs = 1
    # Need to fix gemm timing logic to handle num_runs > 1

    # Run the benchmark
    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task=f"gemm_simple_{in_dtype_str}_{out_dtype_str}",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def gemm_simple_with_dtype_calculate_metrics(
    m: int,
    k: int,
    n: int,
    in_dtype_str: str,
    out_dtype_str: str,
    time_ms_list: list[float],
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = (2 * k - 1) * m * n  # Total floating-point operations
    device_count = (
        jax.local_device_count() if run_on_local_node else jax.device_count()
    )
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY, device_count=device_count
    )

    # Get the multiplier by calling the utility function
    peak_flops_multiplier = get_peak_flops_multiplier(in_dtype_str)

    metadata, metrics = unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE * peak_flops_multiplier,
    )

    # Add dtype info to metadata for logging
    metadata["in_dtype"] = in_dtype_str
    metadata["out_dtype"] = out_dtype_str

    return metadata, metrics


def gemm(
    m: int,
    k: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    """OUT<M, N>:BF16 = matmul(IN0<M, K>:FP8, IN1<N, K>:FP8) *
    outer_product(SF0<M, 1>:FP32 * SF1<1, N>:FP32)."""

    def f(x, y, scale_m, scale_n):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            scales = scale_m * scale_n
            result_fp32 = acc * scales
            return result_fp32.astype(jnp.bfloat16)

    if run_on_local_node:
        configure_local_node_tpu()

    mesh = create_mesh(SHARDING_STRATEGY, local_mesh=run_on_local_node)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    sf0_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    sf1_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(
                lhs_sharding.spec,
                rhs_sharding.spec,
                sf0_sharding.spec,
                sf1_sharding.spec,
            ),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    lhs_shape = (m, k)
    rhs_shape = (k, n)
    sf0_shape = (m, 1)
    sf1_shape = (1, n)

    lhs_dtype = jnp.float8_e4m3fn
    rhs_dtype = jnp.float8_e4m3fn
    sf0_dtype = jnp.float32
    sf1_dtype = jnp.float32

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        # Create random data on host
        lhs_host = jax.random.normal(k1, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(k2, rhs_shape).astype(rhs_dtype)
        sf0_host = jax.random.normal(k3, sf0_shape).astype(sf0_dtype)
        sf1_host = jax.random.normal(k4, sf1_shape).astype(sf1_dtype)

        # Put on device (HBM)
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)
        sf0_device = jax.device_put(sf0_host, sf0_sharding)
        sf1_device = jax.device_put(sf1_host, sf1_sharding)

        return (lhs_device, rhs_device, sf0_device, sf1_device)

    num_runs = 1
    # Need to fix gemm timing logic to handle num_runs > 1

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="gemm",
        trace_dir=trace_dir,
    )

    return {"time_ms_list": time_ms_list}


def gemm_calculate_metrics(
    m: int,
    k: int,
    n: int,
    time_ms_list: list[float],
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    device_count = (
        jax.local_device_count() if run_on_local_node else jax.device_count()
    )
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY, device_count=device_count
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )


def gemm_accum(
    m: int,
    k: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    """OUT<M, N>:FP32 += matmul(IN0<M, K>:FP8, IN1<N, K>:FP8) *
    outer_product(SF0<M, 1>:FP32 * SF1<1, N>:FP32)."""

    def f(out_buffer, x, y, scale_m, scale_n):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum(
                "ij,jk->ik", x, y, preferred_element_type=jnp.float32
            )
            scales = scale_m * scale_n
            result_fp32 = acc * scales
            return out_buffer + result_fp32

    mesh = create_mesh(SHARDING_STRATEGY, local_mesh=run_on_local_node)

    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    sf0_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    sf1_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_buffer_sharding = get_output_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(
                out_buffer_sharding.spec,
                lhs_sharding.spec,
                rhs_sharding.spec,
                sf0_sharding.spec,
                sf1_sharding.spec,
            ),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    lhs_shape = (m, k)
    rhs_shape = (k, n)
    sf0_shape = (m, 1)
    sf1_shape = (1, n)
    out_buffer_shape = (m, n)

    lhs_dtype = jnp.float8_e4m3fn
    rhs_dtype = jnp.float8_e4m3fn
    sf0_dtype = jnp.float32
    sf1_dtype = jnp.float32
    out_buffer_dtype = jnp.float32

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, k_buf, k1, k2, k3, k4 = jax.random.split(key, 6)

        # Create random data on host
        out_buffer_host = jax.random.normal(k_buf, out_buffer_shape).astype(
            out_buffer_dtype
        )
        lhs_host = jax.random.normal(k1, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(k2, rhs_shape).astype(rhs_dtype)
        sf0_host = jax.random.normal(k3, sf0_shape).astype(sf0_dtype)
        sf1_host = jax.random.normal(k4, sf1_shape).astype(sf1_dtype)

        # Put on device (HBM)
        out_buffer_device = jax.device_put(out_buffer_host, out_buffer_sharding)
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)
        sf0_device = jax.device_put(sf0_host, sf0_sharding)
        sf1_device = jax.device_put(sf1_host, sf1_sharding)

        return (
            out_buffer_device,
            lhs_device,
            rhs_device,
            sf0_device,
            sf1_device,
        )

    num_runs = 1
    # Need to fix gemm timing logic to handle num_runs > 1

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="gemm_accum",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def gemm_accum_calculate_metrics(
    m: int,
    k: int,
    n: int,
    time_ms_list: list[float],
    run_on_local_node: bool = False,
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = 2 * m * k * n + m * n  # Total floating-point operations
    device_count = (
        jax.local_device_count() if run_on_local_node else jax.device_count()
    )
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY, device_count=device_count
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )
