"""
Benchmarks different inference compute operations in various flavors.
Considered ops:
"""

import os
from typing import Any, Dict


# pylint: disable=g-importing-member
from benchmark_utils import (
    iteration_timeit,
    ShardingStrategy,
    create_mesh,
    handle_based_on_sharding,
    get_rowwise_named_shading,
    unified_bytes_metrics,
    get_output_named_shading,
    get_out_sharding,
)
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from flax import nnx
from common import MARKER

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
    "--xla_tpu_dvfs_p_state=7"
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
PEAK_FLOPS_PER_DEVICE = 2307  # TFLOP/s for single core(device) of FP8 under p_state=7


def add(
    m: int,
    n: int,
    dtype: jnp.dtype,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    Z = X + Y
    """

    def f(x, y):
        with jax.named_scope(MARKER):
            return x + y

    mesh = create_mesh(SHARDING_STRATEGY)
    x_sharding = get_output_named_shading(mesh, SHARDING_STRATEGY)
    y_sharding = get_output_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(x_sharding.spec, y_sharding.spec),
            out_specs=out_sharding,
            check_rep=False,
        )
    )
    x_shape = (m, n)
    y_shape = (m, n)
    x_dtype = dtype
    y_dtype = dtype

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, k1, k2 = jax.random.split(key, 3)

        x_host = jax.random.normal(k1, x_shape).astype(x_dtype)
        y_host = jax.random.normal(k2, y_shape).astype(y_dtype)

        x_device = jax.device_put(x_host, x_sharding)
        y_device = jax.device_put(y_host, y_sharding)

        return (x_device, y_device)

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}",
        tries=num_runs,
        task="add",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def add_calculate_metrics(
    m: int, n: int, dtype: jnp.dtype, time_ms_list: list[float]
) -> Dict[str, Any]:
    scale = 2 if dtype == jnp.bfloat16 else 1
    total_bytes = scale * 3 * m * n
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def rmsnorm(
    m: int,
    n: int,
    dtype: jnp.dtype,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    For each row i of N:
    Y_i = X_i / rms(x_i)
    """
    rms_norm_module = nnx.RMSNorm(num_features=n, dtype=dtype, rngs=nnx.Rngs(SEED))

    def f(x):
        with jax.named_scope(MARKER):
            return rms_norm_module(x)

    mesh = create_mesh(SHARDING_STRATEGY)
    x_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=x_sharding.spec,
            out_specs=out_sharding.spec,  # Corrected: single spec, not tuple
            check_rep=False,
        )
    )

    x_shape = (m, n)
    x_dtype = dtype
    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, k1 = jax.random.split(key)
        x_host = jax.random.normal(k1, x_shape).astype(x_dtype)
        x_device = jax.device_put(x_host, x_sharding)
        return (x_device,)

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}",  # Using mxn as dims
        tries=num_runs,
        task="rmsnorm",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def rmsnorm_calculate_metrics(
    m: int, n: int, dtype: jnp.dtype, time_ms_list: list[float]
) -> Dict[str, Any]:
    scale = 2 if dtype == jnp.bfloat16 else 1
    total_bytes = scale * 3 * m * n
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def silu_mul(
    m: int,
    n: int,
    dtype: jnp.dtype,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    silu_mul: Z = silu(X) * Y
    ��� silu(x) = x * sigmoid(x)
    """

    def f(x, y):
        with jax.named_scope(MARKER):
            return jax.nn.silu(x) * y

    mesh = create_mesh(SHARDING_STRATEGY)
    x_sharding = get_output_named_shading(mesh, SHARDING_STRATEGY)
    y_sharding = get_output_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(x_sharding.spec, y_sharding.spec),
            out_specs=out_sharding,
            check_rep=False,
        )
    )
    x_shape = (m, n)
    y_shape = (m, n)
    x_dtype = dtype
    y_dtype = dtype

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, k1, k2 = jax.random.split(key, 3)

        x_host = jax.random.normal(k1, x_shape).astype(x_dtype)
        y_host = jax.random.normal(k2, y_shape).astype(y_dtype)

        x_device = jax.device_put(x_host, x_sharding)
        y_device = jax.device_put(y_host, y_sharding)

        return (x_device, y_device)

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}",
        tries=num_runs,
        task="silu_mul",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def silu_mul_calculate_metrics(
    m: int, n: int, dtype: jnp.dtype, time_ms_list: list[float]
) -> Dict[str, Any]:
    scale = 2 if dtype == jnp.bfloat16 else 1
    total_bytes = scale * 3 * m * n
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def sigmoid(
    m: int,
    n: int,
    dtype: jnp.dtype,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    def f(x):
        with jax.named_scope(MARKER):
            return jax.nn.sigmoid(x)

    mesh = create_mesh(SHARDING_STRATEGY)
    x_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=x_sharding.spec,
            out_specs=out_sharding.spec,  # Corrected: single spec, not tuple
            check_rep=False,
        )
    )

    x_shape = (m, n)
    x_dtype = dtype
    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, k1 = jax.random.split(key)
        x_host = jax.random.normal(k1, x_shape).astype(x_dtype)
        x_device = jax.device_put(x_host, x_sharding)
        return (x_device,)

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}",  # Using mxn as dims
        tries=num_runs,
        task="sigmoid",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def sigmoid_calculate_metrics(
    m: int, n: int, dtype: jnp.dtype, time_ms_list: list[float]
) -> Dict[str, Any]:
    scale = 2 if dtype == jnp.bfloat16 else 1
    total_bytes = scale * 2 * m * n
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


# def get_output_named_shading(mesh, strategy: ShardingStrategy):
#     match strategy:
#         case ShardingStrategy.NO_SHARDING:
#             return NamedSharding(mesh, P(None))
#         case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_M:
#             return NamedSharding(mesh, P("device"))
#         case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M:
#             return NamedSharding(mesh, P("device"))
#         case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_N:
#             assert False, f"ShardingStrategy is wrong for this ops: {strategy}"
#         case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
#             assert False, f"ShardingStrategy is wrong for this ops: {strategy}"

# def get_out_sharding(strategy: ShardingStrategy):
#     match strategy:
#         case ShardingStrategy.NO_SHARDING:
#             return P(None)
#         case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_M:
#             return P("device")
#         case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_M:
#             return P("device")
#         case ShardingStrategy.SHARDING_ON_ALL_DEVICES_WITH_N:
#             assert False, f"ShardingStrategy is wrong for this ops: {strategy}"
#         case ShardingStrategy.SHARDING_ON_SINGLE_CHIP_WITH_N:
#             assert False, f"ShardingStrategy is wrong for this ops: {strategy}"

# def add(m: int, dtype: jnp.dtype, num_runs: int = 1, trace_dir: str = None,
# ) -> Dict[str, Any]:
#     """
#     Z = X + Y
#     """
#     def f(x, y):
#         with jax.named_scope(MARKER):
#             return x + y

#     mesh = create_mesh(SHARDING_STRATEGY)
#     x_sharding = get_output_named_shading(mesh, SHARDING_STRATEGY)
#     y_sharding = get_output_named_shading(mesh, SHARDING_STRATEGY)
#     out_sharding = get_out_sharding(SHARDING_STRATEGY)
#     jit_sharded_f = jax.jit(
#         shard_map(
#             f,
#             mesh,
#             in_specs=(x_sharding.spec, y_sharding.spec),
#             out_specs=out_sharding,
#             check_rep=False,
#         )
#     )
#     x_shape = (m)
#     y_shape = (m)
#     x_dtype = dtype
#     y_dtype = dtype

#     key = jax.random.key(SEED)

#     def data_generator():
#         """Creates new random data on host and puts it on device."""
#         nonlocal key # Use and update the outer 'key'
#         key, k1, k2 = jax.random.split(key, 3)

#         x_host = jax.random.normal(k1, x_shape).astype(x_dtype)
#         y_host = jax.random.normal(k2, y_shape).astype(y_dtype)

#         x_device = jax.device_put(x_host, x_sharding)
#         y_device = jax.device_put(y_host, y_sharding)

#         return (x_device, y_device)

#     time_ms_list = iteration_timeit(
#         jit_sharded_f,
#         data_generator,
#         matrix_dim=f"{m}",
#         tries=num_runs,
#         task="add",
#         trace_dir=trace_dir,
#     )
#     return {"time_ms_list": time_ms_list}

# def add_calculate_metrics(
#     m: int, dtype: jnp.dtype, time_ms_list: list[float]
# ) -> Dict[str, Any]:
#     scale = 2 if dtype == jnp.bfloat16 else 1
#     total_bytes = scale * 3 * m
#     total_bytes, total_bytes_all_devices = handle_based_on_sharding(total_bytes, SHARDING_STRATEGY)
#     return unified_bytes_metrics(m, 0,  time_ms_list, total_bytes, total_bytes_all_devices)
