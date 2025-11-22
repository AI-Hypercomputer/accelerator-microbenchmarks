"""
Benchmarks different compute operations in various flavors.
Considered ops:
1. quantization
2. transpose_quantization
3. quantization_static_scaling
4. transpose_quantization_static_scaling
5. swiglu_fwd
6. swiglu_bwd
7. rmsnorm_fwd
8. rmsnorm_bwd
9. add
"""

import os
from typing import Any, Dict, Callable


# pylint: disable=g-importing-member
from benchmark_utils import (
    iteration_timeit,
    ShardingStrategy,
    get_out_sharding,
    get_rowwise_named_shading,
    get_output_named_shading,
    create_mesh,
    handle_based_on_sharding,
    unified_bytes_metrics,
)
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from qwix import pallas as qpl
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
PEAK_FLOPS_PER_DEVICE = 2307  # TFLOP/s for single core(device) of FP8


def fp8_quantization(
    m: int,
    n: int,
    f: Callable,
    num_runs: int = 1,
    trace_dir: str = None,
    task_name: str = "quantization",
) -> Dict[str, Any]:
    mesh = create_mesh(SHARDING_STRATEGY)
    x_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    out_qvalue_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    out_scale_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=x_sharding.spec,
            out_specs=(out_qvalue_sharding.spec, out_scale_sharding.spec),
            check_rep=False,
        )
    )

    x_shape = (m, n)
    x_dtype = jnp.bfloat16

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
        matrix_dim=f"{m}x{n}",
        tries=num_runs,
        task=task_name,
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def quantization(
    m: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    OUT<M, N>:FP8, SF<M>:FP32 = Quantize(N<M, N>:BF16)
    SF[i] = FP8_MAX / amax(IN[i])
    OUT[i] = cast_fp8(IN[i] / SF[i])
    Dymaic scaling with absmax calibration method
    """

    def f(x):
        with jax.named_scope(MARKER):
            qx = qpl.quantize(
                x,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="absmax",
                channelwise_axes=[0],
            )
            return qx.qvalue, qx.scale

    return fp8_quantization(m, n, f, num_runs, trace_dir, task_name="quantization")


def quantization_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 5 * m * n + 4 * m  # Total floating-point operations
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def quantization_static_scaling(
    m: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    OUT<M, N>:FP8, SF<M>:FP32 = Quantize(N<M, N>:BF16)
    SF[i] = FP8_MAX / amax(IN[i])
    OUT[i] = cast_fp8(IN[i] / SF[i])
    Static scaling with fixed scale value
    """

    def f(x):
        with jax.named_scope(MARKER):
            qx = qpl.quantize(
                x,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="fixed, -224, 224",
                channelwise_axes=[0],
            )
            return qx.qvalue, qx.scale

    return fp8_quantization(
        m, n, f, num_runs, trace_dir, task_name="quantization_static_scaling"
    )


def quantization_static_scaling_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 5 * m * n + 4 * m  # Total floating-point operations
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def transpose_quantization(
    m: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    OUT<N, M>:FP8, SF<N>:FP32 = Quantize(Transpose(N<M, N>:BF16)) for 2D
    SF[i] = FP8_MAX / amax(IN[i])
    OUT[i] = cast_fp8(IN[i] / SF[i])
    Dymaic scaling with absmax calibration method
    """

    def f(x):
        with jax.named_scope(MARKER):
            x = x.T
            qx = qpl.quantize(
                x,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="absmax",
                channelwise_axes=[0],
            )
            return qx.qvalue, qx.scale

    return fp8_quantization(
        m, n, f, num_runs, trace_dir, task_name="transpose_quantization"
    )


def transpose_quantization_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 5 * m * n + 4 * m  # Total floating-point operations
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def transpose_quantization_static_scaling(
    m: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    OUT<N, M>:FP8, SF<N>:FP32 = Quantize(Transpose(N<M, N>:BF16)) for 2D
    SF[i] = FP8_MAX / amax(IN[i])
    OUT[i] = cast_fp8(IN[i] / SF[i])
    Static scaling with fixed scale value
    """

    def f(x):
        with jax.named_scope(MARKER):
            x = x.T
            qx = qpl.quantize(
                x,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="fixed, -224, 224",
                channelwise_axes=[0],
            )
            return qx.qvalue, qx.scale

    return fp8_quantization(
        m, n, f, num_runs, trace_dir, task_name="transpose_quantization_static_scaling"
    )


def transpose_quantization_static_scaling_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 5 * m * n + 4 * m  # Total floating-point operations
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def swiglu_fwd(
    m: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    A, B = Split(X, 2)
    Y = Swish(A) ⊗ B
    """

    def f(x):
        with jax.named_scope(MARKER):
            A, B = jnp.split(x, 2, axis=-1)
            A_fp32 = A.astype(jnp.float32)
            B_fp32 = B.astype(jnp.float32)
            Y_fp32 = jax.nn.silu(A_fp32) * B_fp32
            return Y_fp32.astype(jnp.bfloat16)

    mesh = create_mesh(SHARDING_STRATEGY)
    x_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=x_sharding.spec,
            out_specs=out_sharding.spec,
            check_rep=False,
        )
    )

    x_shape = (m, n)
    x_dtype = jnp.bfloat16

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
        matrix_dim=f"{m}x{n}",
        tries=num_runs,
        task="swiglu_fwd",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def swiglu_fwd_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 2 * (m * n + m * n // 2)
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def swiglu_bwd(
    m: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    Inverse of swiglu_fwd
    """

    def f_fwd(x):
        A, B = jnp.split(x, 2, axis=-1)
        A_fp32 = A.astype(jnp.float32)
        B_fp32 = B.astype(jnp.float32)
        Y_fp32 = jax.nn.silu(A_fp32) * B_fp32
        return Y_fp32.astype(jnp.bfloat16)

    def f(x: jax.Array, dy: jax.Array) -> jax.Array:
        """
        x: The original <M, N> BF16 input.
        dy: The upstream <M, N/2> BF16 gradient.
        """
        # Get the VJP "pullback" function
        # We ignore the forward result (_y)
        _y, pullback_fn = jax.vjp(f_fwd, x)
        with jax.named_scope(MARKER):
            # Call the pullback function with the upstream gradient
            # This IS the backward pass.
            dx = pullback_fn(dy)
            # dx is returned as a tuple (one item per arg of f_fwd)
            return dx[0]

    mesh = create_mesh(SHARDING_STRATEGY)
    x_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    dy_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(x_sharding.spec, dy_sharding.spec),
            out_specs=out_sharding.spec,
            check_rep=False,
        )
    )

    x_shape = (m, n)
    dy_shape = (m, n // 2)
    x_dtype = jnp.bfloat16
    dy_dtype = jnp.bfloat16

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, k1, k2 = jax.random.split(key, 3)
        x_host = jax.random.normal(k1, x_shape).astype(x_dtype)
        dy_host = jax.random.normal(k2, dy_shape).astype(dy_dtype)
        x_device = jax.device_put(x_host, x_sharding)
        dy_device = jax.device_put(dy_host, dy_sharding)
        return (x_device, dy_device)

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}",
        tries=num_runs,
        task="swiglu_bwd",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def swiglu_bwd_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 2 * (2 * m * n + m * n // 2)
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def rmsnorm_fwd(
    m: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    For each row i of N:
    Y_i = X_i / rms(x_i)
    """
    rms_norm_module = nnx.RMSNorm(
        num_features=n, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=nnx.Rngs(SEED)
    )

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
    x_dtype = jnp.bfloat16
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
        task="rmsnorm_fwd",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def rmsnorm_fwd_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 2 * (2 * m * n + m * n)
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def rmsnorm_bwd(
    m: int,
    n: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """
    Inverse of rmsnorm_fwd
    """
    rms_norm_module = nnx.RMSNorm(
        num_features=n, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=nnx.Rngs(SEED)
    )

    def f_fwd(x):
        return rms_norm_module(x)

    def f(x: jax.Array, dy: jax.Array) -> jax.Array:
        """
        x: The original <M, N> BF16 input.
        dy: The upstream <M, N/2> BF16 gradient.
        """
        # Get the VJP "pullback" function
        # We ignore the forward result (_y)
        _y, pullback_fn = jax.vjp(f_fwd, x)
        with jax.named_scope(MARKER):
            # Call the pullback function with the upstream gradient
            # This IS the backward pass.
            dx = pullback_fn(dy)
            # dx is returned as a tuple (one item per arg of f_fwd)
            return dx[0]

    mesh = create_mesh(SHARDING_STRATEGY)
    x_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    dy_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_rowwise_named_shading(mesh, SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(x_sharding.spec, dy_sharding.spec),
            out_specs=out_sharding.spec,
            check_rep=False,
        )
    )
    x_shape = (m, n)
    dy_shape = (m, n)
    x_dtype = jnp.bfloat16
    dy_dtype = jnp.bfloat16

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, k1, k2 = jax.random.split(key, 3)
        x_host = jax.random.normal(k1, x_shape).astype(x_dtype)
        dy_host = jax.random.normal(k2, dy_shape).astype(dy_dtype)
        x_device = jax.device_put(x_host, x_sharding)
        dy_device = jax.device_put(dy_host, dy_sharding)
        return (x_device, dy_device)

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}",  # Using mxn as dims
        tries=num_runs,
        task="rmsnorm_bwd",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def rmsnorm_bwd_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 2 * (2 * m * n + m * n)
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )


def add(
    m: int,
    n: int,
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
    x_dtype = jnp.bfloat16
    y_dtype = jnp.bfloat16

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


def add_calculate_metrics(m: int, n: int, time_ms_list: list[float]) -> Dict[str, Any]:
    total_bytes = 6 * m * n
    total_bytes, total_bytes_all_devices = handle_based_on_sharding(
        total_bytes, SHARDING_STRATEGY
    )
    return unified_bytes_metrics(
        m, n, time_ms_list, total_bytes, total_bytes_all_devices
    )
